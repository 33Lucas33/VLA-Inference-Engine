#!/usr/bin/env python3
"""Build a direct paper-checkpoint/local-checkpoint/public-baseline comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

try:
    from tools.summarize_results import (
        BASELINE_CONFIG,
        EXPECTED_SEED_IDS,
        PAPER_BASELINE,
        SUITE_LABEL,
        SUITE_ORDER,
    )
except ModuleNotFoundError:  # Direct execution adds tools/ rather than the project root to sys.path.
    from summarize_results import (
        BASELINE_CONFIG,
        EXPECTED_SEED_IDS,
        PAPER_BASELINE,
        SUITE_LABEL,
        SUITE_ORDER,
    )


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_816


def only_model(path: Path) -> tuple[str, dict]:
    payload = json.loads(path.read_text())
    models = payload.get("models", {})
    if len(models) != 1:
        raise ValueError(f"Expected exactly one model in {path}, found {list(models)}")
    return next(iter(models.items()))


def episode_clusters(summary_path: Path) -> dict[tuple[int, str, int], tuple[bool, ...]]:
    """Load the exact paired episode order from a protocol-complete summary's raw runs."""
    summary = json.loads(summary_path.read_text())
    clusters: dict[tuple[int, str, int], tuple[bool, ...]] = {}
    for run in summary.get("runs", []):
        seed = int(run["seed"])
        eval_path = Path(run["path"]).resolve()
        payload = json.loads(eval_path.read_text())
        for task in payload.get("per_task", []):
            suite = task.get("task_group")
            task_id = int(task.get("task_id"))
            successes = task.get("metrics", {}).get("successes", [])
            key = (seed, suite, task_id)
            if key in clusters:
                raise ValueError(f"Duplicate paired outcome cluster: {key}")
            if suite not in SUITE_ORDER or task_id not in range(10):
                raise ValueError(f"Unexpected paired outcome cluster: {key}")
            if len(successes) != 10 or any(not isinstance(value, bool) for value in successes):
                raise ValueError(f"Paired cluster {key} does not contain 10 boolean episodes")
            clusters[key] = tuple(successes)
    expected = {
        (seed, suite, task_id)
        for seed in EXPECTED_SEED_IDS
        for suite in SUITE_ORDER
        for task_id in range(10)
    }
    if set(clusters) != expected:
        missing = sorted(expected - set(clusters))
        extra = sorted(set(clusters) - expected)
        raise ValueError(f"Paired outcome matrix mismatch: missing={missing}, extra={extra}")
    return clusters


def _quantile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _mcnemar_exact_p(paper_only: int, local_only: int) -> float:
    discordant = paper_only + local_only
    if discordant == 0:
        return 1.0
    tail_limit = min(paper_only, local_only)
    tail = sum(math.comb(discordant, index) for index in range(tail_limit + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def paired_suite_statistics(
    paper_clusters: dict[tuple[int, str, int], tuple[bool, ...]],
    local_clusters: dict[tuple[int, str, int], tuple[bool, ...]],
    suite: str,
    suite_index: int,
) -> dict:
    keys = [(seed, suite, task_id) for seed in EXPECTED_SEED_IDS for task_id in range(10)]
    contingency = {"both_success": 0, "paper_only": 0, "local_only": 0, "both_fail": 0}
    cluster_differences = []
    task_state_differences: dict[tuple[int, int], int] = {
        (task_id, episode_index): 0
        for task_id in range(10)
        for episode_index in range(10)
    }
    for key in keys:
        _seed, _suite, task_id = key
        paper_values = paper_clusters[key]
        local_values = local_clusters[key]
        cluster_differences.append(sum(local_values) - sum(paper_values))
        for episode_index, (paper_success, local_success) in enumerate(
            zip(paper_values, local_values, strict=True)
        ):
            task_state_differences[(task_id, episode_index)] += int(local_success) - int(
                paper_success
            )
            if paper_success and local_success:
                contingency["both_success"] += 1
            elif paper_success:
                contingency["paper_only"] += 1
            elif local_success:
                contingency["local_only"] += 1
            else:
                contingency["both_fail"] += 1

    paired_n = sum(contingency.values())
    difference = 100 * (contingency["local_only"] - contingency["paper_only"]) / paired_n
    rng = random.Random(BOOTSTRAP_SEED + suite_index)
    bootstrap = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sampled_difference = sum(rng.choice(cluster_differences) for _ in cluster_differences)
        bootstrap.append(100 * sampled_difference / paired_n)
    bootstrap.sort()

    # The three process-level seeds reuse fixed LIBERO init-state IDs 0..9. This
    # second, pre-result sensitivity interval resamples tasks and then fixed
    # init-state IDs while retaining all three stochastic seed replications for
    # each selected benchmark unit. It complements the seed×task cluster CI and
    # prevents the repeated states from being described as 30 unique states.
    hierarchy_rng = random.Random(BOOTSTRAP_SEED + 1_000 + suite_index)
    hierarchical_bootstrap = []
    task_ids = list(range(10))
    episode_indices = list(range(10))
    for _ in range(BOOTSTRAP_RESAMPLES):
        sampled_difference = 0
        for _task_draw in task_ids:
            sampled_task_id = hierarchy_rng.choice(task_ids)
            for _state_draw in episode_indices:
                sampled_episode_index = hierarchy_rng.choice(episode_indices)
                sampled_difference += task_state_differences[
                    (sampled_task_id, sampled_episode_index)
                ]
        hierarchical_bootstrap.append(100 * sampled_difference / paired_n)
    hierarchical_bootstrap.sort()
    return {
        "paired_n": paired_n,
        **contingency,
        "paired_difference_pp": difference,
        "paired_cluster_bootstrap_95_low": _quantile(bootstrap, 0.025),
        "paired_cluster_bootstrap_95_high": _quantile(bootstrap, 0.975),
        "paired_hierarchical_bootstrap_95_low": _quantile(hierarchical_bootstrap, 0.025),
        "paired_hierarchical_bootstrap_95_high": _quantile(hierarchical_bootstrap, 0.975),
        "mcnemar_exact_p": _mcnemar_exact_p(
            contingency["paper_only"], contingency["local_only"]
        ),
    }


def _apply_holm(rows: list[dict]) -> None:
    ordered = sorted(enumerate(rows), key=lambda item: item[1]["mcnemar_exact_p"])
    running_max = 0.0
    adjusted = {}
    total = len(ordered)
    for rank, (index, row) in enumerate(ordered):
        candidate = min(1.0, (total - rank) * row["mcnemar_exact_p"])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    for index, row in enumerate(rows):
        row["mcnemar_holm_adjusted_p"] = adjusted[index]
        row["mcnemar_holm_significant_0_05"] = adjusted[index] < 0.05


def compare(paper_path: Path, local_path: Path) -> dict:
    paper_name, paper = only_model(paper_path)
    local_name, local = only_model(local_path)
    paper_clusters = episode_clusters(paper_path)
    local_clusters = episode_clusters(local_path)
    if set(paper_clusters) != set(local_clusters):
        raise ValueError("Paper/local raw outcomes do not share the same paired matrix")
    rows = []
    for suite_index, suite in enumerate(SUITE_ORDER):
        paper_rate = paper["suites"][suite]["success_rate"]
        local_rate = local["suites"][suite]["success_rate"]
        baseline = PAPER_BASELINE[suite]
        paired = paired_suite_statistics(paper_clusters, local_clusters, suite, suite_index)
        if not math.isclose(
            paired["paired_difference_pp"], local_rate - paper_rate, abs_tol=1e-9
        ):
            raise ValueError(f"Raw paired outcomes disagree with saved suite rates: {suite}")
        rows.append(
            {
                "suite": suite,
                "suite_label": SUITE_LABEL[suite],
                "public_baseline": baseline,
                "paper_checkpoint": paper_rate,
                "local_checkpoint": local_rate,
                "paper_minus_public_pp": paper_rate - baseline,
                "local_minus_public_pp": local_rate - baseline,
                "local_minus_paper_checkpoint_pp": local_rate - paper_rate,
                "paper_requires_retest": paper["suites"][suite]["requires_retest"],
                "local_requires_retest": local["suites"][suite]["requires_retest"],
                **paired,
            }
        )
    _apply_holm(rows)
    return {
        "paper_model": paper_name,
        "local_model": local_name,
        "paper_protocol_complete": paper["protocol_complete"],
        "local_protocol_complete": local["protocol_complete"],
        "requires_retest": paper["requires_retest"] or local["requires_retest"],
        "baseline_retrieved_on": BASELINE_CONFIG["retrieved_on"],
        "baseline_source_verification": BASELINE_CONFIG["source_verification"],
        "checkpoint_comparison_scope": BASELINE_CONFIG["checkpoint_comparison_scope"],
        "evaluation_implementation": BASELINE_CONFIG["evaluation_implementation"],
        "reported_reproduction_context": BASELINE_CONFIG.get("reported_reproduction_context", []),
        "paired_analysis": {
            "pairing_key": "seed, suite, task_id, episode_index",
            "paired_episodes_per_suite": 300,
            "fixed_init_state_ids_per_task": list(range(10)),
            "unique_fixed_init_states_per_task_across_seeds": 10,
            "stochastic_replicates_per_task_init_state": len(EXPECTED_SEED_IDS),
            "bootstrap_cluster": "seed x task (10 episodes per cluster)",
            "sensitivity_bootstrap": (
                "hierarchical task -> fixed init-state ID; retain all three seed replications"
            ),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "confidence_interval": "two-sided percentile 95%",
            "hypothesis_test": "two-sided exact McNemar",
            "multiple_comparison_correction": "Holm across four suites",
        },
        "rows": rows,
    }


def write_outputs(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output_dir / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result["rows"][0].keys())
        writer.writeheader()
        writer.writerows(result["rows"])

    lines = [
        "# SmolVLA LIBERO 对照",
        "",
        f"- 论文 checkpoint 协议完整：{result['paper_protocol_complete']}",
        f"- 本地 checkpoint 协议完整：{result['local_protocol_complete']}",
        f"- 是否触发 40% 诊断复测：{result['requires_retest']}",
        "- 基线来源绑定：arXiv v1 LIBERO 表格 SHA256 `"
        f"{result['baseline_source_verification']['arxiv']['table_sha256']}`；固定 LeRobot LIBERO 文档 "
        f"`{result['baseline_source_verification']['lerobot_libero_doc']['sha256']}`。",
        "- 比较范围：相同 LIBERO episode 条件下的部署级复现对照；论文与本地 checkpoint 的 VLM "
        "初始化不同，不是相同初始化的因果训练消融。",
        "- 本机 episode horizons：Spatial/Object/Goal/Long = 280/280/300/520；论文 Table 2 未逐套件披露该参数。",
        "",
        "| 套件 | 公开基线 | 论文 checkpoint | 本地 checkpoint | 本地−论文权重 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        lines.append(
            f"| {row['suite_label']} | {row['public_baseline']:.2f}% "
            f"| {row['paper_checkpoint']:.2f}% | {row['local_checkpoint']:.2f}% "
            f"| {row['local_minus_paper_checkpoint_pp']:+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "## 论文 checkpoint 与本地 checkpoint 的配对分析",
            "",
            "episode 按 seed / suite / task / episode index 配对；主 CI 按 seed×task 聚类 bootstrap；"
            "另报告 task→固定初态 ID 分层 bootstrap 敏感性区间并保留每个初态的 3 次随机复现。"
            "p 值为常规 episode 级双侧 McNemar 精确检验并对四套件作 Holm 校正，需与聚类区间共同解释。",
            "",
            "| 套件 | 配对差值 | seed×task CI | task→初态敏感性 CI | paper-only / local-only | McNemar p | Holm p |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["rows"]:
        lines.append(
            f"| {row['suite_label']} | {row['paired_difference_pp']:+.2f} pp "
            f"| [{row['paired_cluster_bootstrap_95_low']:+.2f}, "
            f"{row['paired_cluster_bootstrap_95_high']:+.2f}] "
            f"| [{row['paired_hierarchical_bootstrap_95_low']:+.2f}, "
            f"{row['paired_hierarchical_bootstrap_95_high']:+.2f}] "
            f"| {row['paper_only']} / {row['local_only']} "
            f"| {row['mcnemar_exact_p']:.6g} | {row['mcnemar_holm_adjusted_p']:.6g} |"
        )
    contexts = result.get("reported_reproduction_context", [])
    if contexts:
        lines.extend(
            [
                "",
                "## 当前公开复现上下文（不参与 40% 触发）",
                "",
                "以下是官方 LeRobot tracker 上尚未建立完整 parity 的用户自报结果，不能替代论文基线。",
                "",
                "| 来源 | checkpoint/设置 | Spatial | Object | Goal | Long |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for item in contexts:
            rates = item["success_rate_percent"]
            lines.append(
                f"| [tracker]({item['source_url']}) | {item['checkpoint']}；{item['protocol_note']} "
                f"| {rates['libero_spatial']:.1f}% | {rates['libero_object']:.1f}% "
                f"| {rates['libero_goal']:.1f}% | {rates['libero_10']:.1f}% |"
            )
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.paper, args.local)
    write_outputs(result, args.output_dir)
    print(json.dumps({"requires_retest": result["requires_retest"]}))


if __name__ == "__main__":
    main()
