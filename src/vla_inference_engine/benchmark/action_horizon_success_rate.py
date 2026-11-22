"""Measure paired 100-episode LIBERO success rates for action execution horizon K."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from vla_inference_engine.benchmark.action_horizon_showcase import ACTION_STEPS, TASKS


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_rows(path: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {(row["suite"], row["trial"], row["k"]): row for row in rows}


def summarize(rows: dict[tuple[str, int, int], dict[str, Any]], n_episodes: int) -> dict[str, Any]:
    cells = []
    for suite, task_id in TASKS:
        for k in ACTION_STEPS:
            selected = [
                rows[(suite, trial, k)]
                for trial in range(n_episodes)
                if (suite, trial, k) in rows
            ]
            successes = sum(row["success"] for row in selected)
            cells.append({
                "suite": suite, "task_id": task_id, "k": k,
                "completed": len(selected), "target": n_episodes,
                "successes": successes,
                "success_rate": successes / len(selected) if selected else None,
                "mean_environment_steps": (
                    sum(row["environment_steps"] for row in selected) / len(selected)
                    if selected else None
                ),
                "mean_model_calls": (
                    sum(row["model_calls"] for row in selected) / len(selected)
                    if selected else None
                ),
                "mean_inference_total_s": (
                    sum(row["inference_total_s"] for row in selected) / len(selected)
                    if selected else None
                ),
                "mean_rollout_wall_s": (
                    sum(row["rollout_wall_s"] for row in selected) / len(selected)
                    if selected else None
                ),
            })
    return {"complete": all(cell["completed"] == n_episodes for cell in cells), "cells": cells}


def run(model: Path, output: Path, n_episodes: int, first_seed: int) -> None:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv as LiberoConfig
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_eval import rollout
    from lerobot.utils.random_utils import set_seed
    import torch

    model, output = model.expanduser().resolve(), output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    episodes_path = output / "episodes.jsonl"
    done = load_rows(episodes_path)

    policy_cfg = PreTrainedConfig.from_pretrained(model)
    policy_cfg.pretrained_path = model
    policy_cfg.device = "cuda"
    policy_cfg.compile_model = False
    policy_cfg.num_steps = 10
    policy_cfg.n_action_steps = 1
    if policy_cfg.chunk_size != 50:
        raise ValueError("This experiment expects the fixed 50-action checkpoint")

    with (model / "model.safetensors").open("rb") as stream:
        model_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    protocol = {
        "experiment": "action_horizon_success_rate",
        "model_path": str(model), "model_sha256": model_sha256,
        "model_chunk_size": 50, "flow_steps": 10,
        "varying_parameter": "n_action_steps", "k_values": list(ACTION_STEPS),
        "tasks": [{"suite": suite, "task_id": task_id} for suite, task_id in TASKS],
        "episodes_per_cell": n_episodes, "first_seed": first_seed,
        "trial_protocol": "trial i uses seed first_seed+i and init state i modulo task init-state count; the same trial is paired across K",
        "control_mode": "relative", "observation_size": [256, 256],
        "videos_recorded": False, "separate_from_formal_baseline": True,
        "timing": "_get_action_chunk wall time with CUDA synchronize before and after each call; rollout wall time includes environment, preprocessing and policy; no video encoding",
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        if previous != protocol:
            raise ValueError("Output directory belongs to a different protocol")
    else:
        write_json(protocol_path, protocol)

    first_suite, first_task = TASKS[0]
    first_env_cfg = LiberoConfig(
        task=first_suite, task_ids=[first_task], observation_height=256,
        observation_width=256, control_mode="relative", init_states=True,
        hard_reset=True, max_parallel_tasks=1,
    )
    set_seed(first_seed)
    policy = make_policy(cfg=policy_cfg, env_cfg=first_env_cfg)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=model,
        preprocessor_overrides={"device_processor": {"device": "cuda"},
                                "rename_observations_processor": {"rename_map": {}}},
    )

    for suite, task_id in TASKS:
        env_cfg = LiberoConfig(
            task=suite, task_ids=[task_id], observation_height=256,
            observation_width=256, control_mode="relative", init_states=True,
            hard_reset=True, max_parallel_tasks=1,
        )
        env = make_env(env_cfg, n_envs=1, use_async_envs=False)[suite][task_id]
        try:
            init_state_count = len(env.envs[0]._init_states)
            env.envs[0]._reset_stride = 0
            env_pre, env_post = make_env_pre_post_processors(
                env_cfg=env_cfg, policy_cfg=policy_cfg
            )
            for trial in range(n_episodes):
                seed = first_seed + trial
                init_state = trial % init_state_count
                for k in ACTION_STEPS:
                    key = (suite, trial, k)
                    if key in done:
                        continue
                    policy.config.n_action_steps = k
                    env.envs[0].init_state_id = init_state
                    set_seed(seed)
                    inference_call_s: list[float] = []
                    original_chunk = policy._get_action_chunk

                    def timed_chunk(*args: Any, **kwargs: Any) -> Any:
                        torch.cuda.synchronize()
                        call_start = time.perf_counter()
                        action_chunk = original_chunk(*args, **kwargs)
                        torch.cuda.synchronize()
                        inference_call_s.append(time.perf_counter() - call_start)
                        return action_chunk

                    policy._get_action_chunk = timed_chunk
                    start = time.perf_counter()
                    try:
                        result = rollout(
                            env=env, policy=policy, env_preprocessor=env_pre,
                            env_postprocessor=env_post, preprocessor=pre,
                            postprocessor=post, seeds=[seed],
                        )
                        rollout_wall_s = time.perf_counter() - start
                    finally:
                        policy._get_action_chunk = original_chunk
                    environment_steps = result["done"].shape[1]
                    row = {
                        "suite": suite, "task_id": task_id, "trial": trial,
                        "seed": seed, "init_state": init_state,
                        "init_state_count": init_state_count, "k": k,
                        "success": bool(result["success"].any().item()),
                        "environment_steps": environment_steps,
                        "model_calls": len(inference_call_s),
                        "inference_call_s": inference_call_s,
                        "inference_total_s": sum(inference_call_s),
                        "inference_call_mean_s": float(np.mean(inference_call_s)),
                        "inference_call_p50_s": float(np.percentile(inference_call_s, 50)),
                        "inference_call_p95_s": float(np.percentile(inference_call_s, 95)),
                        "inference_first_call_s": inference_call_s[0],
                        "rollout_wall_s": rollout_wall_s,
                    }
                    with episodes_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    done[key] = row
                    write_json(output / "summary.json", summarize(done, n_episodes))
                    print(json.dumps(row, ensure_ascii=False), flush=True)
        finally:
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--first-seed", type=int, default=1000)
    args = parser.parse_args()
    run(args.model, args.out, args.episodes, args.first_seed)


if __name__ == "__main__":
    main()
