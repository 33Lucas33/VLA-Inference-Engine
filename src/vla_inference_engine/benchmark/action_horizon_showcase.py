"""Record paired LIBERO episodes while varying actions executed per prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

TASKS = (("libero_spatial", 0), ("libero_object", 0), ("libero_goal", 0))
ACTION_STEPS = (1, 5, 10, 20)


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def frame_from_observation(observation: dict[str, Any]) -> np.ndarray:
    """Show both cameras upright, as LIBERO's display render does."""
    pixels = observation["pixels"]
    return np.concatenate(
        [pixels[name][0, ::-1, ::-1].copy() for name in ("image", "image2")], axis=1
    )


def record_episode(
    *, env: Any, policy: Any, env_pre: Any, env_post: Any, pre: Any, post: Any,
    action_steps: int, seed: int, output: Path,
) -> dict[str, Any]:
    import imageio.v2 as imageio
    from lerobot.scripts.lerobot_eval import rollout
    from lerobot.utils.random_utils import set_seed

    output.mkdir(parents=True, exist_ok=True)
    video_path = output / "episode.mp4"
    writer = imageio.get_writer(video_path, fps=20, codec="libx264", quality=7)
    original_reset, original_step = env.reset, env.step
    original_chunk = policy._get_action_chunk
    counts = {"environment_steps": 0, "model_calls": 0}

    def reset(*args: Any, **kwargs: Any) -> Any:
        observation, info = original_reset(*args, **kwargs)
        writer.append_data(frame_from_observation(observation))
        return observation, info

    def step(action: Any) -> Any:
        result = original_step(action)
        counts["environment_steps"] += 1
        writer.append_data(frame_from_observation(result[0]))
        return result

    def get_chunk(*args: Any, **kwargs: Any) -> Any:
        counts["model_calls"] += 1
        return original_chunk(*args, **kwargs)

    env.reset, env.step = reset, step
    policy._get_action_chunk = get_chunk
    policy.config.n_action_steps = action_steps
    set_seed(seed)
    start = time.perf_counter()
    try:
        result = rollout(env=env, policy=policy, env_preprocessor=env_pre,
                         env_postprocessor=env_post, preprocessor=pre,
                         postprocessor=post, seeds=[seed])
        elapsed = time.perf_counter() - start
    finally:
        writer.close()
        env.reset, env.step = original_reset, original_step
        policy._get_action_chunk = original_chunk

    return {
        "action_steps_per_prediction": action_steps,
        "environment_steps": counts["environment_steps"],
        "model_calls": counts["model_calls"],
        "success": bool(result["success"].any().item()),
        "rollout_wall_s_with_video": elapsed,
        "video": str(video_path.relative_to(output.parent.parent)),
        "video_bytes": video_path.stat().st_size,
    }


def run(model: Path, output: Path, seed: int) -> None:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv as LiberoConfig
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.random_utils import set_seed

    model, output = model.expanduser().resolve(), output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"showcase directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with (model / "model.safetensors").open("rb") as stream:
        model_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    policy_cfg = PreTrainedConfig.from_pretrained(model)
    policy_cfg.pretrained_path = model
    policy_cfg.device = "cuda"
    policy_cfg.compile_model = False
    policy_cfg.num_steps = 10
    policy_cfg.n_action_steps = 1
    if policy_cfg.chunk_size != 50:
        raise ValueError("this showcase expects the fixed 50-action checkpoint")

    first_suite, first_task = TASKS[0]
    first_env_cfg = LiberoConfig(task=first_suite, task_ids=[first_task],
                                 observation_height=256, observation_width=256,
                                 control_mode="relative", init_states=True,
                                 hard_reset=True, max_parallel_tasks=1)
    set_seed(seed)
    policy = make_policy(cfg=policy_cfg, env_cfg=first_env_cfg)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=model,
        preprocessor_overrides={"device_processor": {"device": "cuda"},
                                "rename_observations_processor": {"rename_map": {}}},
    )
    manifest: dict[str, Any] = {
        "mode": "action_horizon_showcase", "formal_result": False,
        "exploratory_single_episode_per_cell": True,
        "model_path": str(model), "model_sha256": model_sha256,
        "model_chunk_size": 50, "flow_steps": 10,
        "varying_parameter": "n_action_steps", "values": list(ACTION_STEPS),
        "seed": seed, "init_state": 0, "control_mode": "relative",
        "video_fps": 20, "video_cameras": ["image", "image2"],
        "tasks": [{"suite": suite, "task_id": task} for suite, task in TASKS],
        "results": [],
    }
    save_json(output / "manifest.json", manifest)

    for suite, task_id in TASKS:
        env_cfg = LiberoConfig(task=suite, task_ids=[task_id],
                               observation_height=256, observation_width=256,
                               control_mode="relative", init_states=True,
                               hard_reset=True, max_parallel_tasks=1)
        env = make_env(env_cfg, n_envs=1, use_async_envs=False)[suite][task_id]
        try:
            env.envs[0].init_state_id = 0
            env.envs[0]._reset_stride = 0
            env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg,
                                                              policy_cfg=policy_cfg)
            description = env.envs[0].task_description
            for action_steps in ACTION_STEPS:
                cell = output / f"{suite}_{task_id}" / f"k{action_steps}"
                print(f"Recording {suite} task {task_id}, K={action_steps}", flush=True)
                row = record_episode(env=env, policy=policy, env_pre=env_pre,
                                     env_post=env_post, pre=pre, post=post,
                                     action_steps=action_steps, seed=seed, output=cell)
                row.update({"suite": suite, "task_id": task_id, "task": description})
                manifest["results"].append(row)
                save_json(output / "manifest.json", manifest)
                print(json.dumps(row, ensure_ascii=False), flush=True)
        finally:
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    run(args.model, args.out, args.seed)


if __name__ == "__main__":
    main()
