#!/usr/bin/env python3
"""Aggregate LeRobot LIBERO eval_info.json files into paper-ready tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

from vla_inference_engine.benchmark.metrics import wilson_interval as _wilson_interval
from vla_inference_engine.benchmark.protocol import FORMAL_SEEDS, SUITE_ORDER, TASK_IDS

try:
    from tools.ensure_libero_config import ensure_libero_config
except ModuleNotFoundError:  # Direct execution adds tools/ rather than the project root to sys.path.
    from ensure_libero_config import ensure_libero_config


BASELINE_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/evaluation/baselines.json"
BASELINE_CONFIG = json.loads(BASELINE_CONFIG_PATH.read_text())
PRIMARY_COMPARATOR = BASELINE_CONFIG["primary_comparator"]
PAPER_BASELINE = PRIMARY_COMPARATOR["success_rate_percent"]
if list(SUITE_ORDER) != PRIMARY_COMPARATOR["suite_order"]:
    raise ValueError("Baseline suite order differs from the evaluation protocol")
SUITE_LABEL = {
    "libero_spatial": "LIBERO-Spatial",
    "libero_object": "LIBERO-Object",
    "libero_goal": "LIBERO-Goal",
    "libero_10": "LIBERO-Long",
}
EXPECTED_SEED_IDS = FORMAL_SEEDS


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    return _wilson_interval(successes, total, z)


def infer_model_and_seed(root: Path, info_path: Path) -> tuple[str, int | None]:
    rel = info_path.relative_to(root)
    parts = rel.parts
    model = root.name if parts[0].startswith("seed_") else parts[0]
    match = re.search(r"seed_(\d+)", str(rel))
    return model, int(match.group(1)) if match else None


def extract_suite_successes(payload: dict) -> dict[str, list[bool]]:
    by_suite: dict[str, list[bool]] = defaultdict(list)
    for task in payload.get("per_task", []):
        suite = task.get("task_group")
        if suite not in PAPER_BASELINE:
            continue
        values = task.get("metrics", {}).get("successes", [])
        by_suite[suite].extend(bool(value) for value in values)
    return by_suite


def validate_eval_payload(
    payload: dict,
    episodes_per_task: int = 10,
    *,
    eval_info_dir: Path | None = None,
    require_video_evidence: bool = False,
    require_command_preflight: bool = False,
) -> dict:
    """Validate one formal LeRobot LIBERO evaluation at task granularity.

    A pooled count alone is insufficient: a duplicated task can hide a missing
    task while preserving 400 total episodes. This validator therefore checks
    the exact 4 suites x 10 tasks x N episodes matrix and cross-checks LeRobot's
    aggregate counters when present.
    """
    errors: list[str] = []
    task_counts: dict[str, dict[int, int]] = {suite: {} for suite in SUITE_ORDER}
    success_counts: dict[str, int] = {suite: 0 for suite in SUITE_ORDER}
    task_video_paths: list[str] = []
    observed_video_paths: set[str] = set()
    per_task = payload.get("per_task")
    if not isinstance(per_task, list):
        return {
            "valid": False,
            "errors": ["per_task must be a list"],
            "expected_tasks": len(SUITE_ORDER) * len(TASK_IDS),
            "observed_tasks": 0,
            "expected_episodes": len(SUITE_ORDER) * len(TASK_IDS) * episodes_per_task,
            "observed_episodes": 0,
        }

    seen: set[tuple[str, int]] = set()
    observed_episodes = 0
    for index, task in enumerate(per_task):
        if not isinstance(task, dict):
            errors.append(f"per_task[{index}] must be an object")
            continue
        suite = task.get("task_group")
        if suite not in SUITE_ORDER:
            errors.append(f"per_task[{index}] has unexpected suite {suite!r}")
            continue
        try:
            task_id = int(task.get("task_id"))
        except (TypeError, ValueError):
            errors.append(f"per_task[{index}] has invalid task_id {task.get('task_id')!r}")
            continue
        key = (suite, task_id)
        if key in seen:
            errors.append(f"duplicate task entry: {suite}/{task_id}")
        seen.add(key)
        if task_id not in TASK_IDS:
            errors.append(f"unexpected task id: {suite}/{task_id}")

        successes = task.get("metrics", {}).get("successes", [])
        if not isinstance(successes, list):
            errors.append(f"successes must be a list: {suite}/{task_id}")
            continue
        if any(not isinstance(value, bool) for value in successes):
            errors.append(f"successes contains non-boolean values: {suite}/{task_id}")
        count = len(successes)
        task_counts[suite][task_id] = count
        observed_episodes += count
        success_counts[suite] += sum(bool(value) for value in successes)
        if count != episodes_per_task:
            errors.append(
                f"wrong episode count: {suite}/{task_id} has {count}, expected {episodes_per_task}"
            )

        metrics = task.get("metrics", {})
        for metric_name in ("sum_rewards", "max_rewards"):
            values = metrics.get(metric_name, [])
            if not isinstance(values, list) or len(values) != episodes_per_task:
                errors.append(
                    f"{metric_name} length is {len(values) if isinstance(values, list) else 'non-list'}, "
                    f"expected {episodes_per_task}: {suite}/{task_id}"
                )
            elif any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
                errors.append(f"{metric_name} contains non-finite/non-numeric values: {suite}/{task_id}")

        video_paths = metrics.get("video_paths", [])
        if require_video_evidence:
            if not isinstance(video_paths, list) or len(video_paths) != episodes_per_task:
                errors.append(
                    f"video_paths length is "
                    f"{len(video_paths) if isinstance(video_paths, list) else 'non-list'}, "
                    f"expected {episodes_per_task}: {suite}/{task_id}"
                )
            else:
                for raw_path in video_paths:
                    path = Path(raw_path)
                    if not path.is_absolute() and eval_info_dir is not None:
                        path = eval_info_dir / path
                    path = path.resolve()
                    canonical = str(path)
                    task_video_paths.append(canonical)
                    if not path.is_file() or path.stat().st_size <= 0:
                        errors.append(f"missing or empty video: {path}")
                    else:
                        observed_video_paths.add(canonical)

    for suite in SUITE_ORDER:
        observed_ids = set(task_counts[suite])
        missing = sorted(set(TASK_IDS) - observed_ids)
        extra = sorted(observed_ids - set(TASK_IDS))
        if missing:
            errors.append(f"missing tasks in {suite}: {missing}")
        if extra:
            errors.append(f"unexpected tasks in {suite}: {extra}")

    expected_suite_episodes = len(TASK_IDS) * episodes_per_task
    per_group = payload.get("per_group")
    if not isinstance(per_group, dict):
        errors.append("per_group must be an object")
    else:
        for suite in SUITE_ORDER:
            group = per_group.get(suite)
            if not isinstance(group, dict):
                errors.append(f"missing per_group aggregate: {suite}")
                continue
            aggregate_episodes = group.get("n_episodes")
            if aggregate_episodes != expected_suite_episodes:
                errors.append(
                    f"per_group {suite} n_episodes={aggregate_episodes!r}, "
                    f"expected {expected_suite_episodes}"
                )
            expected_rate = 100 * success_counts[suite] / expected_suite_episodes
            observed_rate = group.get("pc_success")
            if not isinstance(observed_rate, (int, float)) or not math.isclose(
                float(observed_rate), expected_rate, abs_tol=1e-9
            ):
                errors.append(
                    f"per_group {suite} pc_success={observed_rate!r}, expected {expected_rate}"
                )

    expected_episodes = len(SUITE_ORDER) * expected_suite_episodes
    if require_video_evidence and len(set(task_video_paths)) != len(task_video_paths):
        duplicate_count = len(task_video_paths) - len(set(task_video_paths))
        errors.append(f"task-level video paths contain {duplicate_count} duplicate episode references")
    overall = payload.get("overall")
    if not isinstance(overall, dict):
        errors.append("overall must be an object")
    elif overall.get("n_episodes") != expected_episodes:
        errors.append(
            f"overall n_episodes={overall.get('n_episodes')!r}, expected {expected_episodes}"
        )
    else:
        for timing_key in ("eval_s", "eval_ep_s"):
            value = overall.get(timing_key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                errors.append(f"overall {timing_key} must be a positive finite number, got {value!r}")

    if require_video_evidence and isinstance(overall, dict):
        overall_videos = overall.get("video_paths", [])
        if not isinstance(overall_videos, list) or len(overall_videos) != expected_episodes:
            errors.append(
                f"overall video_paths length is "
                f"{len(overall_videos) if isinstance(overall_videos, list) else 'non-list'}, "
                f"expected {expected_episodes}"
            )
        else:
            canonical_overall = []
            for raw_path in overall_videos:
                if not isinstance(raw_path, str) or not raw_path:
                    errors.append(f"overall video path is invalid: {raw_path!r}")
                    continue
                path = Path(raw_path)
                if not path.is_absolute() and eval_info_dir is not None:
                    path = eval_info_dir / path
                canonical_overall.append(str(path.resolve()))
            if len(set(canonical_overall)) != len(canonical_overall):
                errors.append("overall video_paths contains duplicate episode references")
            if set(canonical_overall) != set(task_video_paths):
                missing_from_overall = sorted(set(task_video_paths) - set(canonical_overall))
                extra_in_overall = sorted(set(canonical_overall) - set(task_video_paths))
                errors.append(
                    "overall/task video path sets differ: "
                    f"missing_from_overall={missing_from_overall[:5]}, "
                    f"extra_in_overall={extra_in_overall[:5]}"
                )

    command_preflight_valid = False
    if require_command_preflight:
        if eval_info_dir is None:
            errors.append("cannot validate command preflight without eval_info_dir")
        else:
            preflight_path = eval_info_dir / "command_preflight.json"
            try:
                preflight = json.loads(preflight_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"missing or invalid command preflight: {preflight_path} ({exc})")
            else:
                if not isinstance(preflight, dict) or preflight.get("valid") is not True:
                    errors.append(f"command preflight is not valid: {preflight_path}")
                else:
                    parsed = preflight.get("parsed", {})
                    expected_episodes = len(SUITE_ORDER) * len(TASK_IDS) * episodes_per_task
                    expected_dir = str(eval_info_dir.resolve())
                    if parsed.get("output_dir") != expected_dir:
                        errors.append(
                            f"command preflight output_dir={parsed.get('output_dir')!r}, "
                            f"expected {expected_dir!r}"
                        )
                    if parsed.get("expected_episodes") != expected_episodes:
                        errors.append(
                            f"command preflight expected_episodes={parsed.get('expected_episodes')!r}, "
                            f"expected {expected_episodes}"
                        )
                    if parsed.get("suites") != list(SUITE_ORDER):
                        errors.append(
                            f"command preflight suites={parsed.get('suites')!r}, "
                            f"expected {list(SUITE_ORDER)!r}"
                        )
                    command_preflight_valid = not any(
                        "command preflight" in error for error in errors
                    )

    return {
        "valid": not errors,
        "errors": errors,
        "expected_tasks": len(SUITE_ORDER) * len(TASK_IDS),
        "observed_tasks": len(seen),
        "expected_episodes": expected_episodes,
        "observed_episodes": observed_episodes,
        "video_evidence_required": require_video_evidence,
        "expected_videos": expected_episodes if require_video_evidence else 0,
        "observed_nonempty_videos": len(observed_video_paths),
        "observed_unique_video_paths": len(set(task_video_paths)),
        "command_preflight_required": require_command_preflight,
        "command_preflight_valid": command_preflight_valid,
        "task_episode_counts": {
            suite: {str(task_id): count for task_id, count in sorted(counts.items())}
            for suite, counts in task_counts.items()
        },
    }


def load_task_metadata() -> dict[tuple[str, int], dict[str, str]]:
    ensure_libero_config()
    from libero.libero import benchmark

    suites = benchmark.get_benchmark_dict()
    metadata = {}
    for suite_name in SUITE_ORDER:
        suite = suites[suite_name]()
        for task_id, task in enumerate(suite.tasks):
            metadata[(suite_name, task_id)] = {
                "task_name": task.name,
                "language_instruction": task.language,
            }
    return metadata


def aggregate(
    root: Path,
    expected_seeds: int = 3,
    episodes_per_task: int = 10,
    expected_seed_ids: tuple[int, ...] = EXPECTED_SEED_IDS,
    require_video_evidence: bool = False,
    require_command_preflight: bool = False,
) -> dict:
    paths = sorted(root.rglob("eval_info.json"))
    if not paths:
        raise FileNotFoundError(f"No eval_info.json found below {root}")

    task_metadata = load_task_metadata()
    runs: list[dict] = []
    pooled: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    seed_rates: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_task_rows: list[dict] = []

    for path in paths:
        payload = json.loads(path.read_text())
        model, seed = infer_model_and_seed(root, path)
        validation = validate_eval_payload(
            payload,
            episodes_per_task,
            eval_info_dir=path.parent,
            require_video_evidence=require_video_evidence,
            require_command_preflight=require_command_preflight,
        )
        suite_values = extract_suite_successes(payload)
        run = {
            "model": model,
            "seed": seed,
            "path": str(path),
            "protocol_validation": validation,
            "timing": {
                "eval_s": payload.get("overall", {}).get("eval_s"),
                "eval_ep_s": payload.get("overall", {}).get("eval_ep_s"),
            },
            "suites": {},
        }
        for suite in SUITE_ORDER:
            values = suite_values.get(suite, [])
            success_count = sum(values)
            total = len(values)
            rate = 100 * success_count / total if total else float("nan")
            run["suites"][suite] = {"successes": success_count, "episodes": total, "rate": rate}
            pooled[model][suite].extend(values)
            if total:
                seed_rates[model][suite].append(rate)

        for task in payload.get("per_task", []):
            suite = task.get("task_group")
            if suite not in PAPER_BASELINE:
                continue
            task_id = int(task.get("task_id"))
            successes = [bool(value) for value in task.get("metrics", {}).get("successes", [])]
            per_task_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "suite": suite,
                    "task_id": task_id,
                    **task_metadata.get((suite, task_id), {"task_name": "", "language_instruction": ""}),
                    "successes": sum(successes),
                    "episodes": len(successes),
                    "success_rate": 100 * sum(successes) / len(successes) if successes else float("nan"),
                }
            )
        runs.append(run)

    models = {}
    expected_suite_episodes = expected_seeds * 10 * episodes_per_task
    for model, suite_map in pooled.items():
        suite_summary = {}
        all_values: list[bool] = []
        requires_retest = False
        for suite in SUITE_ORDER:
            values = suite_map.get(suite, [])
            all_values.extend(values)
            successes = sum(values)
            total = len(values)
            rate = 100 * successes / total if total else float("nan")
            low, high = wilson_interval(successes, total)
            rates = seed_rates[model].get(suite, [])
            seed_mean = statistics.mean(rates) if rates else float("nan")
            seed_sd = statistics.stdev(rates) if len(rates) > 1 else 0.0 if rates else float("nan")
            baseline = PAPER_BASELINE[suite]
            shortfall_pp = baseline - rate if total else float("nan")
            relative_shortfall = 100 * shortfall_pp / baseline if total else float("nan")
            suite_retest = bool(total and (shortfall_pp > 40 or relative_shortfall > 40))
            requires_retest |= suite_retest
            suite_summary[suite] = {
                "successes": successes,
                "episodes": total,
                "success_rate": rate,
                "wilson_95_low": low,
                "wilson_95_high": high,
                "seed_mean": seed_mean,
                "seed_sd": seed_sd,
                "paper_baseline": baseline,
                "shortfall_percentage_points": shortfall_pp,
                "relative_shortfall_percent": relative_shortfall,
                "requires_retest": suite_retest,
            }

        overall_successes = sum(all_values)
        overall_total = len(all_values)
        overall_rate = 100 * overall_successes / overall_total if overall_total else float("nan")
        overall_low, overall_high = wilson_interval(overall_successes, overall_total)
        suite_macro = statistics.mean(
            item["success_rate"] for item in suite_summary.values() if item["episodes"]
        )
        observed_seeds = sorted({run["seed"] for run in runs if run["model"] == model and run["seed"] is not None})
        model_runs = [run for run in runs if run["model"] == model]
        seed_wall_s = {
            str(run["seed"]): float(run["timing"]["eval_s"])
            for run in model_runs
            if isinstance(run.get("timing", {}).get("eval_s"), (int, float))
            and math.isfinite(float(run["timing"]["eval_s"]))
            and float(run["timing"]["eval_s"]) > 0
        }
        evaluation_wall_s = (
            sum(seed_wall_s.values()) if len(seed_wall_s) == len(model_runs) else None
        )
        expected_ids = list(expected_seed_ids)
        protocol_complete = (
            len(model_runs) == expected_seeds
            and observed_seeds == expected_ids
            and all(run["protocol_validation"]["valid"] for run in model_runs)
            and all(item["episodes"] == expected_suite_episodes for item in suite_summary.values())
        )
        models[model] = {
            "suites": suite_summary,
            "overall": {
                "successes": overall_successes,
                "episodes": overall_total,
                "micro_success_rate": overall_rate,
                "macro_suite_success_rate": suite_macro,
                "wilson_95_low": overall_low,
                "wilson_95_high": overall_high,
                "paper_macro_baseline": PRIMARY_COMPARATOR["macro_average_percent"],
                "evaluation_wall_s": evaluation_wall_s,
                "evaluation_wall_hours": evaluation_wall_s / 3600 if evaluation_wall_s else None,
                "evaluation_mean_episode_s": evaluation_wall_s / overall_total
                if evaluation_wall_s and overall_total
                else None,
                "evaluation_seed_wall_s": seed_wall_s,
                "evaluation_time_definition": "sum of eval_info.overall.eval_s over formal seeds",
            },
            "observed_seeds": observed_seeds,
            "expected_seed_ids": expected_ids,
            "expected_seed_count": expected_seeds,
            "expected_episodes_per_suite": expected_suite_episodes,
            "requires_video_evidence": require_video_evidence,
            "requires_command_preflight": require_command_preflight,
            "protocol_complete": protocol_complete,
            "requires_retest": requires_retest,
        }

    return {
        "source_root": str(root),
        "baseline_config": str(BASELINE_CONFIG_PATH),
        "primary_comparator": PRIMARY_COMPARATOR,
        "protocol": {
            "expected_seeds": expected_seeds,
            "expected_seed_ids": list(expected_seed_ids),
            "tasks_per_suite": 10,
            "episodes_per_task": episodes_per_task,
            "expected_episodes_per_suite": expected_suite_episodes,
        },
        "runs": runs,
        "models": models,
        "per_task": per_task_rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")

    with (output_dir / "metrics_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model", "suite", "successes", "episodes", "success_rate", "wilson_95_low",
                "wilson_95_high", "seed_mean", "seed_sd", "paper_baseline",
                "shortfall_percentage_points", "relative_shortfall_percent", "requires_retest",
            ],
        )
        writer.writeheader()
        for model, model_info in summary["models"].items():
            for suite, values in model_info["suites"].items():
                writer.writerow({"model": model, "suite": suite, **values})

    with (output_dir / "per_task.csv").open("w", newline="") as handle:
        fieldnames = [
            "model", "seed", "suite", "task_id", "task_name", "language_instruction",
            "successes", "episodes", "success_rate",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary["per_task"])

    task_rows: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in summary["per_task"]:
        task_rows[(row["model"], row["suite"], int(row["task_id"]))].append(row)
    expected_seed_ids = summary.get("protocol", {}).get("expected_seed_ids", [])
    task_lines = ["# LIBERO 逐任务结果", ""]
    for model in summary["models"]:
        task_lines.extend([f"## {model}", ""])
        for suite in SUITE_ORDER:
            task_lines.extend(
                [
                    f"### {SUITE_LABEL[suite]}",
                    "",
                    "| task ID | task name | 语言指令 | "
                    + " | ".join(f"seed {seed}" for seed in expected_seed_ids)
                    + " | pooled |",
                    "|---:|---|---|"
                    + "|".join("---:" for _ in expected_seed_ids)
                    + "|---:|",
                ]
            )
            for task_id in TASK_IDS:
                rows = sorted(
                    task_rows.get((model, suite, task_id), []),
                    key=lambda row: int(row["seed"]) if row.get("seed") is not None else -1,
                )
                by_seed = {int(row["seed"]): row for row in rows if row.get("seed") is not None}
                successes = sum(int(row["successes"]) for row in rows)
                episodes = sum(int(row["episodes"]) for row in rows)
                metadata = rows[0] if rows else {"task_name": "", "language_instruction": ""}
                task_name = str(metadata.get("task_name", "")).replace("|", "\\|").replace("\n", " ")
                instruction = (
                    str(metadata.get("language_instruction", ""))
                    .replace("|", "\\|")
                    .replace("\n", " ")
                )
                seed_cells = []
                for seed in expected_seed_ids:
                    row = by_seed.get(int(seed))
                    seed_cells.append(
                        f"{row['successes']}/{row['episodes']}" if row is not None else "缺失"
                    )
                pooled_rate = 100 * successes / episodes if episodes else float("nan")
                task_lines.append(
                    f"| {task_id} | {task_name} | {instruction} | "
                    + " | ".join(seed_cells)
                    + f" | {successes}/{episodes} ({pooled_rate:.2f}%) |"
                )
            task_lines.append("")
    (output_dir / "per_task.md").write_text("\n".join(task_lines) + "\n")

    lines = ["# LIBERO 指标汇总", ""]
    for model, model_info in summary["models"].items():
        lines.extend(
            [
                f"## {model}",
                "",
                f"- 协议完整性：{'通过' if model_info['protocol_complete'] else '未通过'}；"
                f"观测 seeds={model_info['observed_seeds']}，预注册 seeds="
                f"{model_info['expected_seed_ids']}；每套件期望 "
                f"{model_info['expected_episodes_per_suite']} episodes。",
                "",
                "| 套件 | 成功率 | Wilson 95% CI | seed 均值 ± SD | 论文基线 | 相对缺口 | 重测 |",
                "|---|---:|---:|---:|---:|---:|:---:|",
            ]
        )
        for suite in SUITE_ORDER:
            row = model_info["suites"][suite]
            lines.append(
                f"| {SUITE_LABEL[suite]} | {row['success_rate']:.2f}% ({row['successes']}/{row['episodes']}) "
                f"| [{row['wilson_95_low']:.2f}, {row['wilson_95_high']:.2f}] "
                f"| {row['seed_mean']:.2f} ± {row['seed_sd']:.2f} | {row['paper_baseline']:.1f}% "
                f"| {row['relative_shortfall_percent']:.2f}% | {'是' if row['requires_retest'] else '否'} |"
            )
        overall = model_info["overall"]
        if isinstance(overall.get("evaluation_wall_hours"), (int, float)):
            timing_line = (
                f"- LeRobot rollout-core wall-clock：{overall['evaluation_wall_hours']:.3f} h；"
                f"平均 {overall['evaluation_mean_episode_s']:.3f} s/episode；"
                f"各 seed 秒数：{overall['evaluation_seed_wall_s']}"
            )
        else:
            timing_line = "- LeRobot rollout-core wall-clock：不可用（协议未完整）"
        lines.extend(
            [
                "",
                f"- 四套件 macro 成功率：{overall['macro_suite_success_rate']:.2f}%",
                f"- 合并 episode 成功率：{overall['micro_success_rate']:.2f}% "
                f"（95% CI [{overall['wilson_95_low']:.2f}, {overall['wilson_95_high']:.2f}]）",
                timing_line,
                f"- 是否触发 40% 差距诊断：{'是' if model_info['requires_retest'] else '否'}",
                "",
            ]
        )
    (output_dir / "metrics_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Directory containing one or more eval_info.json files")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--expected-seeds", type=int, default=3)
    parser.add_argument("--expected-seed-ids", default="1000,2000,3000")
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--require-video-evidence", action="store_true")
    parser.add_argument("--require-command-preflight", action="store_true")
    args = parser.parse_args()
    expected_seed_ids = tuple(int(item.strip()) for item in args.expected_seed_ids.split(",") if item.strip())
    if len(expected_seed_ids) != args.expected_seeds:
        parser.error("--expected-seed-ids count must equal --expected-seeds")
    summary = aggregate(
        args.root.resolve(), args.expected_seeds, args.episodes_per_task, expected_seed_ids,
        args.require_video_evidence,
        args.require_command_preflight,
    )
    write_outputs(summary, args.output_dir.resolve())
    print(json.dumps({name: info["requires_retest"] for name, info in summary["models"].items()}))


if __name__ == "__main__":
    main()
