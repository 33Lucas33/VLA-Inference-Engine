#!/usr/bin/env python3
"""Parse and validate a formal LIBERO command without creating an env or policy.

This deliberately uses LeRobot's own draccus configuration parser.  It catches
CLI/schema drift before a long rollout while avoiding CUDA initialization and
MuJoCo environment creation.
"""

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import draccus
from lerobot.configs import parser as lerobot_parser
from lerobot.configs.eval import EvalPipelineConfig

from vla_inference_engine.benchmark.protocol import (
    ACTION_SHAPE,
    EPISODES_PER_TASK,
    INPUT_SHAPES,
    SUITE_HORIZONS,
    SUITE_ORDER,
)

SUITES = SUITE_ORDER
EXPECTED_SUITE_HORIZONS = SUITE_HORIZONS
EXPECTED_INPUTS = INPUT_SHAPES
EXPECTED_OUTPUTS = {"action": ACTION_SHAPE}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape(feature: Any) -> tuple[int, ...] | None:
    value = getattr(feature, "shape", None)
    return tuple(value) if value is not None else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return _jsonable(value.value)
    return value


def _find_arg(arguments: list[str], name: str) -> str | None:
    prefix = f"--{name}="
    for index, item in enumerate(arguments):
        if item.startswith(prefix):
            return item[len(prefix) :]
        if item == f"--{name}" and index + 1 < len(arguments):
            return arguments[index + 1]
    return None


def parse_lerobot_command(command: list[str]) -> EvalPipelineConfig:
    if not command or Path(command[0]).name != "lerobot-eval":
        raise ValueError("command must begin with lerobot-eval")
    cli_args = command[1:]
    old_argv = sys.argv
    try:
        # EvalPipelineConfig.__post_init__ intentionally reads policy.path and
        # policy overrides from sys.argv, so mirror the actual executable here.
        sys.argv = [command[0], *cli_args]
        filtered = lerobot_parser.filter_path_args(
            EvalPipelineConfig.__get_path_fields__(), cli_args
        )
        return draccus.parse(config_class=EvalPipelineConfig, args=filtered)
    finally:
        sys.argv = old_argv


def validate_command(command: list[str], expected_seed: int) -> dict[str, Any]:
    errors: list[str] = []
    try:
        cfg = parse_lerobot_command(command)
    except (Exception, SystemExit) as exc:
        return {
            "valid": False,
            "errors": [f"LeRobot rejected the command: {type(exc).__name__}: {exc}"],
            "command": command,
        }

    def check(name: str, actual: Any, expected: Any) -> None:
        if actual != expected:
            errors.append(f"{name}={actual!r}, expected {expected!r}")

    check("env.type", cfg.env.type, "libero")
    observed_suites = tuple(item.strip() for item in str(cfg.env.task).split(",") if item.strip())
    check("env.task", observed_suites, SUITES)
    check("env.task_ids", cfg.env.task_ids, None)
    check("env.control_mode", cfg.env.control_mode, "relative")
    check("env.observation_height", cfg.env.observation_height, 256)
    check("env.observation_width", cfg.env.observation_width, 256)
    check("env.init_states", cfg.env.init_states, True)
    check("env.hard_reset", cfg.env.hard_reset, True)
    check("env.max_parallel_tasks", cfg.env.max_parallel_tasks, 1)
    check("env.episode_length", cfg.env.episode_length, None)
    check("eval.batch_size", cfg.eval.batch_size, 1)
    check("eval.use_async_envs", cfg.eval.use_async_envs, False)
    check("eval.n_episodes", cfg.eval.n_episodes, EPISODES_PER_TASK)
    check("eval.recording", cfg.eval.recording, False)
    check("seed", cfg.seed, expected_seed)

    try:
        from lerobot.envs.libero import TASK_SUITE_MAX_STEPS

        observed_suite_horizons = {
            suite: TASK_SUITE_MAX_STEPS.get(suite) for suite in SUITES
        }
    except Exception as exc:
        observed_suite_horizons = None
        errors.append(
            f"cannot resolve fixed LIBERO suite horizons: {type(exc).__name__}: {exc}"
        )
    else:
        check(
            "env.resolved_suite_horizons",
            observed_suite_horizons,
            EXPECTED_SUITE_HORIZONS,
        )

    if cfg.policy is None:
        errors.append("policy config was not loaded")
        policy_inputs: dict[str, tuple[int, ...] | None] = {}
        policy_outputs: dict[str, tuple[int, ...] | None] = {}
        model_path = None
        model_artifacts = None
    else:
        check("policy.type", cfg.policy.type, "smolvla")
        check("policy.device", str(cfg.policy.device), "cuda")
        check("policy.num_steps", cfg.policy.num_steps, 10)
        check("policy.n_action_steps", cfg.policy.n_action_steps, 1)
        check("policy.chunk_size", cfg.policy.chunk_size, 50)
        check("policy.compile_model", cfg.policy.compile_model, False)
        policy_inputs = {name: _shape(feature) for name, feature in cfg.policy.input_features.items()}
        policy_outputs = {name: _shape(feature) for name, feature in cfg.policy.output_features.items()}
        check("policy.input_features", policy_inputs, EXPECTED_INPUTS)
        check("policy.output_features", policy_outputs, EXPECTED_OUTPUTS)
        model_path = Path(cfg.policy.pretrained_path).resolve()
        required = (
            model_path / "config.json",
            model_path / "model.safetensors",
            model_path / "policy_preprocessor.json",
            model_path / "policy_postprocessor.json",
        )
        for path in required:
            if not path.is_file() or path.stat().st_size <= 0:
                errors.append(f"missing or empty model artifact: {path}")
        if not list(model_path.glob("policy_preprocessor_step_*.safetensors")):
            errors.append(f"missing preprocessor state: {model_path}")
        if not list(model_path.glob("policy_postprocessor_step_*.safetensors")):
            errors.append(f"missing postprocessor state: {model_path}")
        inference_paths = sorted(
            {
                *required,
                *model_path.glob("policy_preprocessor_step_*.safetensors"),
                *model_path.glob("policy_postprocessor_step_*.safetensors"),
            }
        )
        model_artifacts = {
            "model_root": str(model_path),
            "files": {
                str(path.relative_to(model_path)): {
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in inference_paths
                if path.is_file()
            },
        }
        model_artifacts["manifest_sha256"] = hashlib.sha256(
            json.dumps(
                model_artifacts["files"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    output_dir = Path(cfg.output_dir).resolve()
    raw_output = _find_arg(command[1:], "output_dir")
    if raw_output is None:
        errors.append("output_dir must be explicit")
    if output_dir.name != f"seed_{expected_seed}":
        errors.append(
            f"output_dir name is {output_dir.name!r}, expected 'seed_{expected_seed}'"
        )

    return {
        "valid": not errors,
        "errors": errors,
        "command": command,
        "model_artifacts": model_artifacts,
        "parsed": {
            "seed": cfg.seed,
            "output_dir": str(output_dir),
            "model_path": str(model_path) if model_path is not None else None,
            "suites": list(observed_suites),
            "episodes_per_task": cfg.eval.n_episodes,
            "batch_size": cfg.eval.batch_size,
            "use_async_envs": cfg.eval.use_async_envs,
            "control_mode": cfg.env.control_mode,
            "observation_height": cfg.env.observation_height,
            "observation_width": cfg.env.observation_width,
            "init_states": cfg.env.init_states,
            "hard_reset": cfg.env.hard_reset,
            "max_parallel_tasks": cfg.env.max_parallel_tasks,
            "episode_length_override": cfg.env.episode_length,
            "resolved_suite_horizons": observed_suite_horizons,
            "policy_input_features": _jsonable(policy_inputs),
            "policy_output_features": _jsonable(policy_outputs),
            "policy_num_steps": getattr(cfg.policy, "num_steps", None),
            "policy_n_action_steps": getattr(cfg.policy, "n_action_steps", None),
            "policy_compile_model": getattr(cfg.policy, "compile_model", None),
            "expected_tasks": len(SUITES) * 10,
            "expected_episodes": len(SUITES) * 10 * cfg.eval.n_episodes,
            "video_evidence_expected": True,
        },
        "parser_snapshot": _jsonable(asdict(cfg)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-seed", type=int, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    result = validate_command(command, args.expected_seed)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    print(rendered, end="")
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
