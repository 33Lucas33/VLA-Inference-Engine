#!/usr/bin/env python3
"""Index low-performing tasks and failed videos after a >40% baseline gap."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    from tools.summarize_results import SUITE_LABEL, SUITE_ORDER
except ModuleNotFoundError:  # Direct execution adds tools/ rather than the project root to sys.path.
    from summarize_results import SUITE_LABEL, SUITE_ORDER


def _validated_video_path(eval_info_path: Path, raw_path: object) -> str:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"Invalid failed-video path in {eval_info_path}: {raw_path!r}")
    path = Path(raw_path)
    if not path.is_absolute():
        path = eval_info_path.parent / path
    path = path.resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"Failed-video evidence is missing or empty: {path}")
    return str(path)


def diagnose(summary_path: Path) -> dict:
    summary = json.loads(summary_path.read_text())
    if len(summary.get("models", {})) != 1:
        raise ValueError("Expected one model in a per-model metrics_summary.json")
    model_name, model = next(iter(summary["models"].items()))
    triggered = [suite for suite in SUITE_ORDER if model["suites"][suite]["requires_retest"]]

    task_totals: dict[tuple[str, int], dict[str, object]] = defaultdict(
        lambda: {"successes": 0, "episodes": 0, "task_name": "", "language_instruction": ""}
    )
    for row in summary.get("per_task", []):
        key = (row["suite"], int(row["task_id"]))
        aggregate = task_totals[key]
        aggregate["successes"] = int(aggregate["successes"]) + int(row["successes"])
        aggregate["episodes"] = int(aggregate["episodes"]) + int(row["episodes"])
        aggregate["task_name"] = row.get("task_name", "")
        aggregate["language_instruction"] = row.get("language_instruction", "")

    failed_videos: dict[tuple[str, int], list[str]] = defaultdict(list)
    for run in summary.get("runs", []):
        eval_info_path = Path(run["path"]).resolve()
        payload = json.loads(eval_info_path.read_text())
        for task in payload.get("per_task", []):
            suite = task.get("task_group")
            task_id = int(task.get("task_id"))
            metrics = task.get("metrics", {})
            successes = metrics.get("successes", [])
            videos = metrics.get("video_paths", [])
            if len(successes) != len(videos):
                raise ValueError(
                    f"success/video length mismatch in {eval_info_path}: "
                    f"{suite}/{task_id} has {len(successes)} successes and {len(videos)} videos"
                )
            for success, video in zip(successes, videos, strict=True):
                if not success:
                    failed_videos[(suite, task_id)].append(
                        _validated_video_path(eval_info_path, video)
                    )

    low_tasks = {}
    for suite in triggered:
        rows = []
        for (row_suite, task_id), values in task_totals.items():
            if row_suite != suite:
                continue
            episodes = int(values["episodes"])
            successes = int(values["successes"])
            rows.append(
                {
                    "task_id": task_id,
                    "task_name": values["task_name"],
                    "language_instruction": values["language_instruction"],
                    "successes": successes,
                    "episodes": episodes,
                    "success_rate": 100 * successes / episodes if episodes else 0.0,
                    "failed_video_paths": failed_videos[(suite, task_id)],
                }
            )
        low_tasks[suite] = sorted(rows, key=lambda row: (row["success_rate"], row["task_id"]))

    indexed_failures = sum(
        len(task["failed_video_paths"])
        for tasks in low_tasks.values()
        for task in tasks
    )
    return {
        "schema_version": 1,
        "model": model_name,
        "requires_retest": bool(triggered),
        "triggered_suites": triggered,
        "diagnostic_seed": 4000,
        "indexed_failed_videos": indexed_failures,
        "lowest_tasks": low_tasks,
        "required_checks": [
            "checkpoint/data revisions and SHA256",
            "processor order, input keys, normalization statistics and 180-degree image convention",
            "two 256x256 RGB cameras, 8D state, relative 7D action and n_action_steps=1",
            "fixed initial states, hard reset, episode length and task language",
            "failed videos: perception/localization, premature gripper closure, action scale, long horizon, simulator",
            "training updates, scheduler, loss, gradient norm and final checkpoint processors",
        ],
    }


def write_outputs(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "diagnostic_plan.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# 40% 差距诊断计划",
        "",
        f"- 模型：`{result['model']}`",
        f"- 触发诊断：{result['requires_retest']}",
        f"- 触发套件：{', '.join(result['triggered_suites']) or '无'}",
        f"- 独立诊断 seed：{result['diagnostic_seed']}",
        "",
        "## 必查项",
        "",
    ]
    lines.extend(f"- {item}" for item in result["required_checks"])
    for suite in result["triggered_suites"]:
        lines.extend(["", f"## {SUITE_LABEL[suite]} 低成功率任务", ""])
        for task in result["lowest_tasks"][suite]:
            videos = task["failed_video_paths"][:3]
            lines.append(
                f"- task {task['task_id']}：{task['success_rate']:.2f}% "
                f"({task['successes']}/{task['episodes']})；{task['language_instruction']}"
            )
            lines.extend(f"  - 失败视频：`{video}`" for video in videos)
    (output_dir / "diagnostic_plan.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.summary)
    write_outputs(result, args.output_dir)
    print(json.dumps({"requires_retest": result["requires_retest"], "suites": result["triggered_suites"]}))


if __name__ == "__main__":
    main()
