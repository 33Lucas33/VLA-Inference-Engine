#!/usr/bin/env python3
"""Prove that the formal paper/local episode pairing is outcome-independent."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.ensure_libero_config import ensure_libero_config
    from tools.summarize_results import EXPECTED_SEED_IDS, SUITE_ORDER
    from tools.validate_eval_command import EXPECTED_SUITE_HORIZONS
except ModuleNotFoundError:
    from ensure_libero_config import ensure_libero_config
    from summarize_results import EXPECTED_SEED_IDS, SUITE_ORDER
    from validate_eval_command import EXPECTED_SUITE_HORIZONS


EXPECTED_HEAD = "8ae2354592ec4480c38134fb75350b8e7992e0e9"
PAIRING_FIX_COMMITS = {
    "remove_libero_internal_reset": "7a3298ea263528b371e2630bb0421cd519b45ef4",
    "freeze_finished_vector_slots": "8135a8a8d1d2b1bdcd95449a3071a722b29a835c",
}


def git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def validate_pairing_protocol(root: Path, preflight_path: Path) -> dict[str, Any]:
    root = root.resolve()
    preflight_path = preflight_path.resolve()
    repo = root / "vendor/lerobot"
    errors: list[str] = []
    assertions: list[dict[str, Any]] = []

    def record(name: str, ok: bool, observed: Any, expected: Any) -> None:
        assertions.append({"name": name, "ok": ok, "observed": observed, "expected": expected})
        if not ok:
            errors.append(f"{name}: observed={observed!r}, expected={expected!r}")

    head_result = git(repo, "rev-parse", "HEAD")
    head = head_result.stdout.strip() if head_result.returncode == 0 else None
    record("lerobot_head", head == EXPECTED_HEAD, head, EXPECTED_HEAD)
    fix_ancestry = {}
    for name, commit in PAIRING_FIX_COMMITS.items():
        result = git(repo, "merge-base", "--is-ancestor", commit, "HEAD")
        fix_ancestry[name] = result.returncode == 0
        record(f"fix_ancestor.{name}", result.returncode == 0, result.returncode, 0)

    try:
        from lerobot.envs.libero import LiberoEnv
        from lerobot.scripts.lerobot_eval import rollout
    except Exception as exc:
        errors.append(f"cannot import fixed LeRobot eval stack: {type(exc).__name__}: {exc}")
        libero_step_source = ""
        rollout_source = ""
        no_internal_reset = False
        explicit_new_rollout = False
        terminal_step_reset_calls = None
    else:
        libero_step_source = inspect.getsource(LiberoEnv.step)
        rollout_source = inspect.getsource(rollout)
        no_internal_reset = "self.reset(" not in libero_step_source
        explicit_new_rollout = "options={NEW_ROLLOUT_OPTION: True}" in rollout_source
        record("libero_step_has_no_internal_reset", no_internal_reset, no_internal_reset, True)
        record("rollout_marks_explicit_new_episode", explicit_new_rollout, explicit_new_rollout, True)

        class FakeInner:
            def __init__(self):
                self.reset_calls = 0

            def step(self, action):
                return {"terminal": True}, 1.0, False, {}

            def check_success(self):
                return True

            def reset(self):
                self.reset_calls += 1
                return {}

        fake = FakeInner()
        env = object.__new__(LiberoEnv)
        env._env = fake
        env._ensure_env = lambda: None
        env._format_raw_obs = lambda raw: raw
        env.task = "pairing_probe"
        env.task_id = 0
        _obs, _reward, terminated, _truncated, _info = LiberoEnv.step(
            env, np.zeros(7, dtype=np.float32)
        )
        terminal_step_reset_calls = fake.reset_calls
        record("terminal_step_is_terminated", terminated is True, terminated, True)
        record(
            "terminal_step_does_not_advance_reset_counter",
            terminal_step_reset_calls == 0,
            terminal_step_reset_calls,
            0,
        )

        class FakeController:
            use_delta = None

        class FakeRobot:
            controller = FakeController()

        class FakeResetInner:
            def __init__(self):
                self.selected_init_state_ids: list[int] = []
                self.received_seeds: list[int] = []
                self.robots = [FakeRobot()]

            def seed(self, seed):
                self.received_seeds.append(int(seed))

            def reset(self):
                return {"init_state_id": -1}

            def set_init_state(self, state):
                state_id = int(state)
                self.selected_init_state_ids.append(state_id)
                return {"init_state_id": state_id}

            def step(self, _action):
                return {"init_state_id": self.selected_init_state_ids[-1]}, 0.0, False, {}

        def trace_formal_resets(start_seed: int) -> dict[str, Any]:
            inner = FakeResetInner()
            probe = object.__new__(LiberoEnv)
            probe._env = inner
            probe._ensure_env = lambda: None
            probe._format_raw_obs = lambda raw: raw
            probe.init_states = True
            probe._init_states = np.arange(50)
            probe.init_state_id = 0
            probe._reset_stride = 1
            probe.num_steps_wait = 0
            probe.control_mode = "relative"
            for episode_index in range(10):
                LiberoEnv.reset(probe, seed=start_seed + episode_index)
            return {
                "formal_seed": start_seed,
                "received_episode_seeds": inner.received_seeds,
                "selected_init_state_ids": inner.selected_init_state_ids,
                "next_init_state_id": probe.init_state_id,
            }

        formal_reset_traces = [trace_formal_resets(seed) for seed in EXPECTED_SEED_IDS]
        expected_state_ids = list(range(10))
        for trace in formal_reset_traces:
            seed = trace["formal_seed"]
            record(
                f"formal_seed_{seed}.episode_seeds",
                trace["received_episode_seeds"] == list(range(seed, seed + 10)),
                trace["received_episode_seeds"],
                list(range(seed, seed + 10)),
            )
            record(
                f"formal_seed_{seed}.init_state_ids",
                trace["selected_init_state_ids"] == expected_state_ids,
                trace["selected_init_state_ids"],
                expected_state_ids,
            )
        unique_formal_init_state_ids = sorted(
            {
                state_id
                for trace in formal_reset_traces
                for state_id in trace["selected_init_state_ids"]
            }
        )
        record(
            "formal_seed_design.unique_init_state_ids_per_task",
            unique_formal_init_state_ids == expected_state_ids,
            unique_formal_init_state_ids,
            expected_state_ids,
        )

    try:
        preflight = json.loads(preflight_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"formal preflight missing or invalid: {preflight_path} ({exc})")
        preflight = {}
    parsed = preflight.get("parsed", {}) if isinstance(preflight, dict) else {}
    expected_preflight = {
        "batch_size": 1,
        "use_async_envs": False,
        "max_parallel_tasks": 1,
        "init_states": True,
        "hard_reset": True,
        "episodes_per_task": 10,
        "suites": list(SUITE_ORDER),
        "episode_length_override": None,
        "resolved_suite_horizons": EXPECTED_SUITE_HORIZONS,
    }
    record("formal_preflight_valid", preflight.get("valid") is True, preflight.get("valid"), True)
    for key, expected in expected_preflight.items():
        record(f"formal_preflight.{key}", parsed.get(key) == expected, parsed.get(key), expected)

    init_state_counts: dict[str, list[int]] = {}
    try:
        ensure_libero_config()
        from libero.libero import benchmark

        benchmark_dict = benchmark.get_benchmark_dict()
        for suite_name in SUITE_ORDER:
            suite = benchmark_dict[suite_name]()
            counts = [len(suite.get_task_init_states(task_id)) for task_id in range(10)]
            init_state_counts[suite_name] = counts
            record(
                f"init_states.{suite_name}",
                len(counts) == 10 and all(count >= 10 for count in counts),
                counts,
                "10 tasks, each with >=10 fixed initial states",
            )
    except Exception as exc:
        errors.append(f"cannot inspect LIBERO fixed initial states: {type(exc).__name__}: {exc}")

    return {
        "valid": not errors,
        "errors": errors,
        "project_root": str(root),
        "lerobot_head": head,
        "pairing_fix_commits": PAIRING_FIX_COMMITS,
        "pairing_fix_ancestry": fix_ancestry,
        "formal_preflight": str(preflight_path),
        "formal_preflight_sha256": hashlib.sha256(preflight_path.read_bytes()).hexdigest()
        if preflight_path.is_file()
        else None,
        "pairing_key": "seed, suite, task_id, episode_index",
        "pairing_scope": "batch_size=1, synchronous env, one task at a time",
        "formal_reset_traces": formal_reset_traces if "formal_reset_traces" in locals() else [],
        "seed_design": {
            "formal_process_seeds": list(EXPECTED_SEED_IDS),
            "episode_seeds_per_task": [
                list(range(seed, seed + 10)) for seed in EXPECTED_SEED_IDS
            ],
            "fixed_init_state_ids_per_process": list(range(10)),
            "unique_fixed_init_state_ids_per_task_across_all_seeds": 10,
            "stochastic_replicates_per_task_init_state": len(EXPECTED_SEED_IDS),
            "interpretation": (
                "Three stochastic replications of fixed init-state IDs 0..9; "
                "not 30 distinct initial states per task."
            ),
        },
        "init_state_counts": init_state_counts,
        "terminal_step_reset_calls": terminal_step_reset_calls,
        "assertions": assertions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    preflight = (
        args.preflight or root / "outputs/preflight/formal_eval_smolvla_libero_paper_seed1000.json"
    ).resolve()
    result = validate_pairing_protocol(root, preflight)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    print(rendered, end="")
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
