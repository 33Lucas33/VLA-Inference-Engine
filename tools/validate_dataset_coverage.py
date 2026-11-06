#!/usr/bin/env python3
"""Validate exact task, episode, and frame coverage of the fixed LIBERO dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.dataset as arrow_dataset
import pyarrow.parquet as parquet

try:
    from tools.ensure_libero_config import ensure_libero_config
    from tools.summarize_results import SUITE_LABEL, SUITE_ORDER
except ModuleNotFoundError:
    from ensure_libero_config import ensure_libero_config
    from summarize_results import SUITE_LABEL, SUITE_ORDER


EXPECTED_REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
EXPECTED_EPISODES = 1693
EXPECTED_FRAMES = 273465
EXPECTED_TASKS = 40
TASK_FIELDS = (
    "suite",
    "suite_label",
    "task_id",
    "dataset_task_index",
    "language_instruction",
    "episodes",
    "frames",
    "mean_frames_per_episode",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_language_map() -> tuple[dict[str, tuple[str, int]], list[str]]:
    ensure_libero_config()
    from libero.libero import benchmark

    mapping: dict[str, tuple[str, int]] = {}
    errors: list[str] = []
    benchmark_dict = benchmark.get_benchmark_dict()
    for suite in SUITE_ORDER:
        suite_object = benchmark_dict[suite]()
        for task_id in range(10):
            language = suite_object.get_task(task_id).language
            if language in mapping:
                errors.append(f"official task language is duplicated: {language!r}")
            mapping[language] = (suite, task_id)
    return mapping, errors


def validate_dataset_coverage(dataset_root: Path) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    info_path = dataset_root / "meta/info.json"
    tasks_path = dataset_root / "meta/tasks.parquet"
    data_root = dataset_root / "data"
    errors: list[str] = []
    try:
        info = json.loads(info_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"dataset info is missing or invalid: {info_path} ({exc})")
        info = {}
    try:
        task_records = parquet.read_table(tasks_path).to_pylist()
    except Exception as exc:
        errors.append(f"dataset task table is missing or invalid: {tasks_path} ({exc})")
        task_records = []

    try:
        expected_languages, language_errors = official_language_map()
    except Exception as exc:
        errors.append(f"cannot load official LIBERO task languages: {type(exc).__name__}: {exc}")
        expected_languages, language_errors = {}, []
    errors.extend(language_errors)
    index_to_language: dict[int, str] = {}
    for row in task_records:
        try:
            task_index = int(row["task_index"])
            language = str(row["__index_level_0__"])
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"invalid task metadata row: {row!r} ({exc})")
            continue
        if task_index in index_to_language:
            errors.append(f"duplicate dataset task_index: {task_index}")
        index_to_language[task_index] = language

    task_episodes: dict[int, set[int]] = defaultdict(set)
    task_frames: dict[int, int] = defaultdict(int)
    episode_tasks: dict[int, set[int]] = defaultdict(set)
    frame_rows = 0
    data_files = sorted(data_root.rglob("*.parquet")) if data_root.is_dir() else []
    try:
        scanner = arrow_dataset.dataset(data_root, format="parquet").scanner(
            columns=["episode_index", "task_index"]
        )
        for batch in scanner.to_batches():
            episodes = batch.column("episode_index").to_pylist()
            tasks = batch.column("task_index").to_pylist()
            frame_rows += batch.num_rows
            for episode, task in zip(episodes, tasks, strict=True):
                episode_id, task_index = int(episode), int(task)
                task_episodes[task_index].add(episode_id)
                task_frames[task_index] += 1
                episode_tasks[episode_id].add(task_index)
    except Exception as exc:
        errors.append(f"cannot scan dataset parquet coverage: {type(exc).__name__}: {exc}")

    observed_indices = sorted(index_to_language)
    if observed_indices != list(range(EXPECTED_TASKS)):
        errors.append(f"dataset task indices are {observed_indices}, expected 0..39")
    languages = set(index_to_language.values())
    if len(languages) != EXPECTED_TASKS:
        errors.append("dataset language instructions are not 40 unique values")
    unknown = sorted(languages - set(expected_languages))
    missing = sorted(set(expected_languages) - languages)
    if unknown:
        errors.append(f"dataset has unknown task languages: {unknown}")
    if missing:
        errors.append(f"dataset lacks official task languages: {missing}")
    if frame_rows != EXPECTED_FRAMES or info.get("total_frames") != frame_rows:
        errors.append(
            f"frame count metadata={info.get('total_frames')}, scanned={frame_rows}, expected={EXPECTED_FRAMES}"
        )
    episode_ids = sorted(episode_tasks)
    if episode_ids != list(range(EXPECTED_EPISODES)):
        errors.append("episode indices are not the exact contiguous range 0..1692")
    mixed_episodes = sorted(episode for episode, tasks in episode_tasks.items() if len(tasks) != 1)
    if mixed_episodes:
        errors.append(f"episodes map to multiple tasks: {mixed_episodes[:20]}")
    if len(episode_ids) != EXPECTED_EPISODES or info.get("total_episodes") != len(episode_ids):
        errors.append(
            f"episode count metadata={info.get('total_episodes')}, scanned={len(episode_ids)}, "
            f"expected={EXPECTED_EPISODES}"
        )
    if info.get("total_tasks") != EXPECTED_TASKS:
        errors.append(f"metadata total_tasks={info.get('total_tasks')!r}, expected 40")

    rows: list[dict[str, Any]] = []
    for dataset_task_index, language in index_to_language.items():
        suite_task = expected_languages.get(language)
        if suite_task is None:
            continue
        suite, task_id = suite_task
        episode_count = len(task_episodes.get(dataset_task_index, set()))
        frames = task_frames.get(dataset_task_index, 0)
        if episode_count <= 0 or frames <= 0:
            errors.append(f"dataset task {dataset_task_index} has no coverage")
        rows.append(
            {
                "suite": suite,
                "suite_label": SUITE_LABEL[suite],
                "task_id": task_id,
                "dataset_task_index": dataset_task_index,
                "language_instruction": language,
                "episodes": episode_count,
                "frames": frames,
                "mean_frames_per_episode": frames / episode_count if episode_count else None,
            }
        )
    suite_index = {suite: index for index, suite in enumerate(SUITE_ORDER)}
    rows.sort(key=lambda row: (suite_index[row["suite"]], row["task_id"]))
    expected_matrix = {(suite, task_id) for suite in SUITE_ORDER for task_id in range(10)}
    if {(row["suite"], row["task_id"]) for row in rows} != expected_matrix:
        errors.append("dataset-to-official suite/task mapping is not a complete 4x10 matrix")

    suites = {}
    for suite in SUITE_ORDER:
        suite_rows = [row for row in rows if row["suite"] == suite]
        suites[suite] = {
            "suite_label": SUITE_LABEL[suite],
            "tasks": len(suite_rows),
            "episodes": sum(row["episodes"] for row in suite_rows),
            "frames": sum(row["frames"] for row in suite_rows),
            "minimum_episodes_per_task": min((row["episodes"] for row in suite_rows), default=0),
            "maximum_episodes_per_task": max((row["episodes"] for row in suite_rows), default=0),
        }
    return {
        "valid": not errors,
        "errors": errors,
        "dataset_root": str(dataset_root),
        "dataset_revision": EXPECTED_REVISION,
        "metadata": {
            "info_sha256": sha256(info_path) if info_path.is_file() else None,
            "tasks_sha256": sha256(tasks_path) if tasks_path.is_file() else None,
            "data_parquet_files": len(data_files),
        },
        "totals": {
            "episodes": len(episode_ids),
            "frames": frame_rows,
            "tasks": len(rows),
            "mixed_task_episodes": len(mixed_episodes),
        },
        "suites": suites,
        "minimum_episodes_per_task": min((row["episodes"] for row in rows), default=0),
        "maximum_episodes_per_task": max((row["episodes"] for row in rows), default=0),
        "tasks": rows,
    }


def write_tables(result: dict[str, Any], csv_path: Path | None, md_path: Path | None) -> None:
    rows = result["tasks"]
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=TASK_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    if md_path:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# LIBERO 训练数据逐任务覆盖",
            "",
            f"- 验证：{'通过' if result['valid'] else '失败'}",
            f"- 总计：{result['totals']['episodes']} episodes、{result['totals']['frames']} frames、"
            f"{result['totals']['tasks']} tasks；每任务 {result['minimum_episodes_per_task']}–"
            f"{result['maximum_episodes_per_task']} episodes。",
            "",
            "| Suite | Task ID | Dataset index | Episodes | Frames | Mean frames/episode | Language instruction |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for row in rows:
            lines.append(
                f"| {row['suite_label']} | {row['task_id']} | {row['dataset_task_index']} | "
                f"{row['episodes']} | {row['frames']} | {row['mean_frames_per_episode']:.2f} | "
                f"{row['language_instruction']} |"
            )
        md_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--csv-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    args = parser.parse_args()
    result = validate_dataset_coverage(args.dataset_root)
    write_tables(result, args.csv_out, args.md_out)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    print(rendered, end="")
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
