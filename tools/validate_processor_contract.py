#!/usr/bin/env python3
"""Validate the LIBERO observation/action and normalization contract end to end."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

EXPECTED_INPUTS = {
    "observation.images.image": {"type": "VISUAL", "shape": [3, 256, 256]},
    "observation.images.image2": {"type": "VISUAL", "shape": [3, 256, 256]},
    "observation.state": {"type": "STATE", "shape": [8]},
}
EXPECTED_OUTPUTS = {"action": {"type": "ACTION", "shape": [7]}}
EXPECTED_PREPROCESSOR_STEPS = [
    "rename_observations_processor",
    "to_batch_processor",
    "smolvla_new_line_processor",
    "tokenizer_processor",
    "device_processor",
    "normalizer_processor",
]
ALLOWED_POSTPROCESSOR_STEPS = [
    ["device_processor", "unnormalizer_processor"],
    ["unnormalizer_processor", "device_processor"],
]
ALLOWED_VLM_MODELS = {
    "HuggingFaceTB/SmolVLM2-500M-Instruct",
    "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
}
STATS_RTOL = 1e-6
STATS_ATOL = 1e-7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_tensors(path: Path) -> dict[str, np.ndarray]:
    with safe_open(path, framework="numpy") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


def _float32_stats(dataset_stats: dict, feature: str, statistic: str) -> np.ndarray:
    return np.asarray(dataset_stats[feature][statistic], dtype=np.float32)


def validate_processor_contract(
    root: Path,
    model_path: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    model = (model_path or root / "artifacts/models/smolvla_libero_paper").resolve()
    dataset = root / "artifacts/datasets/lerobot-libero/meta"
    preprocessor_state_candidates = sorted(
        model.glob("policy_preprocessor_step_*_normalizer_processor.safetensors")
    )
    postprocessor_state_candidates = sorted(
        model.glob("policy_postprocessor_step_*_unnormalizer_processor.safetensors")
    )
    paths = {
        "model_config": model / "config.json",
        "preprocessor_config": model / "policy_preprocessor.json",
        "postprocessor_config": model / "policy_postprocessor.json",
        "preprocessor_state": (
            preprocessor_state_candidates[0]
            if preprocessor_state_candidates
            else model / "MISSING_preprocessor_normalizer_state.safetensors"
        ),
        "postprocessor_state": (
            postprocessor_state_candidates[0]
            if postprocessor_state_candidates
            else model / "MISSING_postprocessor_unnormalizer_state.safetensors"
        ),
        "dataset_info": dataset / "info.json",
        "dataset_stats": dataset / "stats.json",
        "libero_processor_source": root / "vendor/lerobot/src/lerobot/processor/env_processor.py",
        "libero_env_config_source": root / "vendor/lerobot/src/lerobot/envs/configs.py",
    }
    errors: list[str] = []
    assertions: list[dict[str, Any]] = []

    def record(name: str, ok: bool, observed: Any, expected: Any) -> None:
        assertions.append({"name": name, "ok": ok, "observed": observed, "expected": expected})
        if not ok:
            errors.append(f"{name}: observed={observed!r}, expected={expected!r}")

    missing = [str(path) for path in paths.values() if not path.is_file() or path.stat().st_size <= 0]
    record("all_contract_files_exist", not missing, missing, [])
    if missing:
        return {
            "valid": False,
            "errors": errors,
            "project_root": str(root),
            "model_root": str(model),
            "paths": {key: str(path) for key, path in paths.items()},
            "assertions": assertions,
        }

    config = json.loads(paths["model_config"].read_text())
    pre = json.loads(paths["preprocessor_config"].read_text())
    post = json.loads(paths["postprocessor_config"].read_text())
    info = json.loads(paths["dataset_info"].read_text())
    stats = json.loads(paths["dataset_stats"].read_text())
    pre_state = _load_tensors(paths["preprocessor_state"])
    post_state = _load_tensors(paths["postprocessor_state"])

    record(
        "processor_state_file_cardinality",
        len(preprocessor_state_candidates) == 1
        and len(postprocessor_state_candidates) == 1,
        {
            "preprocessor_normalizer": [path.name for path in preprocessor_state_candidates],
            "postprocessor_unnormalizer": [path.name for path in postprocessor_state_candidates],
        },
        {"preprocessor_normalizer_count": 1, "postprocessor_unnormalizer_count": 1},
    )

    record("model.input_features", config.get("input_features") == EXPECTED_INPUTS, config.get("input_features"), EXPECTED_INPUTS)
    record("model.output_features", config.get("output_features") == EXPECTED_OUTPUTS, config.get("output_features"), EXPECTED_OUTPUTS)
    expected_model_controls = {
        "chunk_size": 50,
        "n_action_steps": 1,
        "num_steps": 10,
        "normalization_mapping": {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
        "adapt_to_pi_aloha": False,
        "use_delta_joint_actions_aloha": False,
    }
    for key, expected in expected_model_controls.items():
        record(f"model.{key}", config.get(key) == expected, config.get(key), expected)
    record(
        "model.vlm_model_name",
        config.get("vlm_model_name") in ALLOWED_VLM_MODELS,
        config.get("vlm_model_name"),
        sorted(ALLOWED_VLM_MODELS),
    )
    record(
        "model.pad_language_to",
        config.get("pad_language_to") in {"longest", "max_length"},
        config.get("pad_language_to"),
        ["longest", "max_length"],
    )

    dataset_shapes = {
        key: info.get("features", {}).get(key, {}).get("shape")
        for key in (*EXPECTED_INPUTS, "action")
    }
    expected_dataset_shapes = {
        "observation.images.image": [256, 256, 3],
        "observation.images.image2": [256, 256, 3],
        "observation.state": [8],
        "action": [7],
    }
    record("dataset.feature_shapes", dataset_shapes == expected_dataset_shapes, dataset_shapes, expected_dataset_shapes)

    pre_steps = [step.get("registry_name") for step in pre.get("steps", [])]
    post_steps = [step.get("registry_name") for step in post.get("steps", [])]
    record("preprocessor.step_order", pre_steps == EXPECTED_PREPROCESSOR_STEPS, pre_steps, EXPECTED_PREPROCESSOR_STEPS)
    record(
        "postprocessor.step_order",
        post_steps in ALLOWED_POSTPROCESSOR_STEPS,
        post_steps,
        ALLOWED_POSTPROCESSOR_STEPS,
    )
    pre_by_name = {step.get("registry_name"): step for step in pre.get("steps", [])}
    post_by_name = {step.get("registry_name"): step for step in post.get("steps", [])}
    processor_links = {
        "preprocessor_device": pre_by_name.get("device_processor", {}).get("config", {}).get("device"),
        "postprocessor_device": post_by_name.get("device_processor", {}).get("config", {}).get("device"),
        "preprocessor_state_file": pre_by_name.get("normalizer_processor", {}).get("state_file"),
        "postprocessor_state_file": post_by_name.get("unnormalizer_processor", {}).get("state_file"),
    }
    expected_processor_links = {
        "preprocessor_device": "cuda",
        "postprocessor_device": "cpu",
        "preprocessor_state_file": paths["preprocessor_state"].name,
        "postprocessor_state_file": paths["postprocessor_state"].name,
    }
    record(
        "processor.device_and_state_file_links",
        processor_links == expected_processor_links,
        processor_links,
        expected_processor_links,
    )
    rename_map = pre["steps"][0].get("config", {}).get("rename_map") if pre.get("steps") else None
    record("preprocessor.rename_map", rename_map == {}, rename_map, {})
    tokenizer = pre["steps"][3].get("config", {}) if len(pre.get("steps", [])) > 3 else {}
    tokenizer_name = tokenizer.get("tokenizer_name")
    # LeRobot serializes a tokenizer object into the checkpoint's local
    # ``tokenizer/`` artifact directory.  In that case the config intentionally
    # contains the relative artifact name rather than the original Hub ID.
    # Accept this representation only when all files needed for an offline
    # reload are present; never accept an arbitrary alias.
    tokenizer_artifact = model / "tokenizer"
    tokenizer_artifact_files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    )
    tokenizer_artifact_status = {
        "name": tokenizer_name,
        "directory": str(tokenizer_artifact),
        "required_files": {
            name: (tokenizer_artifact / name).is_file()
            and (tokenizer_artifact / name).stat().st_size > 0
            for name in tokenizer_artifact_files
        },
    }
    local_artifact_ok = tokenizer_name == "tokenizer" and all(
        tokenizer_artifact_status["required_files"].values()
    )
    tokenizer_name_ok = tokenizer_name == config.get("vlm_model_name") or local_artifact_ok
    expected_tokenizer = {
        "max_length": 48,
        "task_key": "task",
        "padding_side": "right",
        "padding": config.get("pad_language_to"),
        "truncation": True,
        "tokenizer_name": tokenizer_name,
    }
    record(
        "preprocessor.tokenizer",
        tokenizer == expected_tokenizer and tokenizer_name_ok,
        tokenizer,
        {
            **expected_tokenizer,
            "tokenizer_name": [config.get("vlm_model_name"), "tokenizer"],
        },
    )
    if tokenizer_name == "tokenizer":
        record(
            "preprocessor.tokenizer_artifact",
            local_artifact_ok,
            tokenizer_artifact_status,
            {"required_files": {name: True for name in tokenizer_artifact_files}},
        )
    expected_norm_map = {
        "VISUAL": "IDENTITY",
        "STATE": "MEAN_STD",
        "ACTION": "MEAN_STD",
    }
    expected_pre_normalizer = {
        "eps": 1e-8,
        "features": {**EXPECTED_INPUTS, **EXPECTED_OUTPUTS},
        "norm_map": expected_norm_map,
    }
    observed_pre_normalizer = pre_by_name.get("normalizer_processor", {}).get(
        "config", {}
    )
    record(
        "preprocessor.normalizer_config",
        observed_pre_normalizer == expected_pre_normalizer,
        observed_pre_normalizer,
        expected_pre_normalizer,
    )
    expected_post_unnormalizer = {
        "eps": 1e-8,
        "features": EXPECTED_OUTPUTS,
        "norm_map": expected_norm_map,
    }
    observed_post_unnormalizer = post_by_name.get("unnormalizer_processor", {}).get(
        "config", {}
    )
    record(
        "postprocessor.unnormalizer_config",
        observed_post_unnormalizer == expected_post_unnormalizer,
        observed_post_unnormalizer,
        expected_post_unnormalizer,
    )

    required_tensor_keys = {
        "action.mean": [7],
        "action.std": [7],
        "observation.state.mean": [8],
        "observation.state.std": [8],
    }
    observed_shapes = {key: list(value.shape) for key, value in pre_state.items()}
    for key, shape in required_tensor_keys.items():
        record(f"preprocessor_state.{key}.shape", observed_shapes.get(key) == shape, observed_shapes.get(key), shape)

    stat_comparisons = []
    for feature in ("action", "observation.state"):
        for statistic in ("mean", "std"):
            key = f"{feature}.{statistic}"
            observed = pre_state.get(key)
            expected = _float32_stats(stats, feature, statistic)
            exact = observed is not None and np.array_equal(observed, expected)
            matches = observed is not None and np.allclose(
                observed, expected, rtol=STATS_RTOL, atol=STATS_ATOL
            )
            max_abs = float(np.max(np.abs(observed - expected))) if observed is not None else None
            max_rel = (
                float(
                    np.max(
                        np.abs(observed - expected)
                        / np.maximum(np.abs(expected), np.finfo(np.float32).tiny)
                    )
                )
                if observed is not None
                else None
            )
            stat_comparisons.append(
                {
                    "tensor": key,
                    "matches_dataset_float32_exactly": exact,
                    "matches_dataset_within_tolerance": matches,
                    "rtol": STATS_RTOL,
                    "atol": STATS_ATOL,
                    "max_abs_difference": max_abs,
                    "max_relative_difference": max_rel,
                    "shape": list(observed.shape) if observed is not None else None,
                }
            )
            record(
                f"normalization.{key}",
                matches,
                {"max_abs_difference": max_abs, "max_relative_difference": max_rel},
                {"rtol": STATS_RTOL, "atol": STATS_ATOL},
            )

    for key in ("action.mean", "action.std"):
        matches = key in post_state and key in pre_state and np.array_equal(post_state[key], pre_state[key])
        record(f"action_roundtrip_stats.{key}", matches, matches, True)

    try:
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import (
            policy_action_to_transition,
            transition_to_policy_action,
        )

        postprocessor_pipeline = PolicyProcessorPipeline.from_pretrained(
            model,
            config_filename="policy_postprocessor.json",
            local_files_only=True,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
        normalized_probe = torch.tensor(
            [[-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0]], dtype=torch.float32
        )
        observed_action = postprocessor_pipeline(normalized_probe)
        action_mean = torch.from_numpy(pre_state["action.mean"]).reshape(1, -1)
        action_std = torch.from_numpy(pre_state["action.std"]).reshape(1, -1)
        expected_action = normalized_probe * action_std + action_mean
        behavioral_action = {
            "input_shape": list(normalized_probe.shape),
            "output_shape": list(observed_action.shape),
            "output_device": str(observed_action.device),
            "matches_saved_mean_std_inverse": torch.allclose(
                observed_action, expected_action, rtol=1e-6, atol=1e-7
            ),
            "maximum_abs_difference": float(
                torch.max(torch.abs(observed_action - expected_action)).item()
            ),
        }
    except Exception as exc:
        behavioral_action = {"error": f"{type(exc).__name__}: {exc}"}
    expected_behavioral_action = {
        "input_shape": [1, 7],
        "output_shape": [1, 7],
        "output_device": "cpu",
        "matches_saved_mean_std_inverse": True,
        "maximum_abs_difference": 0.0,
    }
    record(
        "behavioral_action_unnormalization_contract",
        behavioral_action == expected_behavioral_action,
        behavioral_action,
        expected_behavioral_action,
    )

    processor_source = paths["libero_processor_source"].read_text()
    config_source = paths["libero_env_config_source"].read_text()
    source_contract = {
        "both_cameras_flip_180": "torch.flip(img, dims=[2, 3])" in processor_source,
        "state_is_eef_axisangle_gripper": "torch.cat((eef_pos, eef_axisangle, gripper_qpos), dim=-1)" in processor_source,
        "quaternion_xyzw_uses_w_index_3": "w = quat[:, 3].clamp(-1.0, 1.0)" in processor_source,
        "env_postprocessor_is_empty": "PolicyProcessorPipeline(steps=[])" in config_source,
        "default_control_mode_relative": 'control_mode: str = "relative"' in config_source,
    }
    for name, observed in source_contract.items():
        record(f"source_contract.{name}", observed is True, observed, True)

    # Behavioral probe of the actual installed processor. The binding validator
    # separately proves that installed source and the fixed vendor have the same Git tree.
    try:
        from lerobot.configs import FeatureType
        from lerobot.envs.configs import LiberoEnv
        from lerobot.processor.env_processor import LiberoProcessorStep

        image = torch.arange(18, dtype=torch.float32).reshape(1, 3, 2, 3)
        theta = math.pi / 2
        observation = {
            "observation.images.image": image.clone(),
            "observation.images.image2": (image + 100).clone(),
            "observation.robot_state": {
                "eef": {
                    "pos": torch.tensor([[1.0, 2.0, 3.0]]),
                    "quat": torch.tensor([[0.0, 0.0, math.sin(theta / 2), math.cos(theta / 2)]]),
                },
                "gripper": {"qpos": torch.tensor([[0.1, -0.1]])},
            },
        }
        transformed = LiberoProcessorStep().observation(observation)
        expected_state = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, theta, 0.1, -0.1]])
        behavior = {
            "main_camera_flipped": torch.equal(
                transformed["observation.images.image"], torch.flip(image, dims=[2, 3])
            ),
            "wrist_camera_flipped": torch.equal(
                transformed["observation.images.image2"], torch.flip(image + 100, dims=[2, 3])
            ),
            "state_shape": list(transformed["observation.state"].shape),
            "state_matches": torch.allclose(transformed["observation.state"], expected_state, atol=1e-6),
            "raw_robot_state_removed": "observation.robot_state" not in transformed,
        }
        env_config = LiberoEnv(
            task="libero_spatial",
            task_ids=[0],
            episode_length=400,
            observation_height=256,
            observation_width=256,
        )
        env_preprocessor, env_postprocessor = env_config.get_env_processors()
        env_behavior = {
            "control_mode": env_config.control_mode,
            "action_feature_type": env_config.features["action"].type.value,
            "action_feature_shape": list(env_config.features["action"].shape),
            "camera_feature_shapes": {
                "agentview": list(env_config.features["pixels/agentview_image"].shape),
                "eye_in_hand": list(
                    env_config.features["pixels/robot0_eye_in_hand_image"].shape
                ),
            },
            "hard_reset": env_config.hard_reset,
            "init_states": env_config.init_states,
            "env_preprocessor_steps": [
                type(step).__name__ for step in env_preprocessor.steps
            ],
            "env_postprocessor_steps": [
                type(step).__name__ for step in env_postprocessor.steps
            ],
            "action_feature_enum_matches": (
                env_config.features["action"].type == FeatureType.ACTION
            ),
        }
    except Exception as exc:
        behavior = {"error": f"{type(exc).__name__}: {exc}"}
        env_behavior = {"error": f"{type(exc).__name__}: {exc}"}
    expected_behavior = {
        "main_camera_flipped": True,
        "wrist_camera_flipped": True,
        "state_shape": [1, 8],
        "state_matches": True,
        "raw_robot_state_removed": True,
    }
    record("behavioral_observation_contract", behavior == expected_behavior, behavior, expected_behavior)
    expected_env_behavior = {
        "control_mode": "relative",
        "action_feature_type": "ACTION",
        "action_feature_shape": [7],
        "camera_feature_shapes": {
            "agentview": [256, 256, 3],
            "eye_in_hand": [256, 256, 3],
        },
        "hard_reset": True,
        "init_states": True,
        "env_preprocessor_steps": ["LiberoProcessorStep"],
        "env_postprocessor_steps": [],
        "action_feature_enum_matches": True,
    }
    record(
        "behavioral_environment_config_contract",
        env_behavior == expected_env_behavior,
        env_behavior,
        expected_env_behavior,
    )

    return {
        "valid": not errors,
        "errors": errors,
        "project_root": str(root),
        "model_root": str(model),
        "paths": {key: str(path) for key, path in paths.items()},
        "sha256": {key: sha256(path) for key, path in paths.items()},
        "normalization_stat_comparisons": stat_comparisons,
        "source_contract": source_contract,
        "behavioral_observation_contract": behavior,
        "behavioral_environment_config_contract": env_behavior,
        "behavioral_action_contract": behavioral_action,
        "action_contract": {
            "policy_output": "7D action normalized with dataset MEAN_STD",
            "policy_postprocessor": "exact inverse of the stored policy action MEAN_STD",
            "environment_postprocessor": "identity",
            "controller": "relative",
            "environment_action_feature": "7D ACTION",
        },
        "policy_processing_profile": {
            "vlm_model_name": config.get("vlm_model_name"),
            "language_padding": config.get("pad_language_to"),
            "tokenizer_max_length": config.get("tokenizer_max_length"),
            "postprocessor_step_order": post_steps,
            "preprocessor_state_file": paths["preprocessor_state"].name,
            "postprocessor_state_file": paths["postprocessor_state"].name,
        },
        "assertions": assertions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--model",
        type=Path,
        help="Checkpoint pretrained_model directory; defaults to the paper checkpoint.",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    result = validate_processor_contract(args.root, args.model)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    print(rendered, end="")
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
