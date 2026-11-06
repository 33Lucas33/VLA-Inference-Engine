#!/usr/bin/env python3
"""Validate disk-space gates for both formal LIBERO model evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

try:
    from tools.record_storage_gate import GIB
    from tools.summarize_results import EXPECTED_SEED_IDS
except ModuleNotFoundError:
    from record_storage_gate import GIB
    from summarize_results import EXPECTED_SEED_IDS


EXPECTED_MINIMUM_FREE_GIB = 100.0


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_storage_history(
    root: Path,
    history_path: Path,
    model_labels: list[str],
    seed_ids: list[int] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    history_path = history_path.resolve()
    seed_ids = list(EXPECTED_SEED_IDS if seed_ids is None else seed_ids)
    errors: list[str] = []
    events: list[dict[str, Any]] = []
    try:
        lines = history_path.read_text().splitlines()
    except OSError as exc:
        errors.append(f"storage history missing or unreadable: {history_path} ({exc})")
        lines = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            errors.append(f"storage history contains an empty line at {line_number}")
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"storage history line {line_number} is invalid JSON: {exc}")
            continue
        if not isinstance(row, dict):
            errors.append(f"storage history line {line_number} is not an object")
            continue
        events.append(row)

    expected_root = str(root)
    expected_labels = list(model_labels)
    relevant: list[tuple[int, dict[str, Any]]] = []
    for index, row in enumerate(events):
        if row.get("model_label") not in expected_labels:
            continue
        relevant.append((index, row))
        mount = row.get("mount") if isinstance(row.get("mount"), dict) else {}
        options = str(mount.get("options") or "").split(",")
        if row.get("project_root") != expected_root:
            errors.append(f"event {index} project root is not the ext4 project root")
        if mount.get("fstype") != "ext4" or "rw" not in options:
            errors.append(f"event {index} mount is not ext4 rw")
        if row.get("minimum_free_gib") != EXPECTED_MINIMUM_FREE_GIB:
            errors.append(f"event {index} minimum-free threshold is not 100 GiB")
        if row.get("minimum_free_bytes") != int(EXPECTED_MINIMUM_FREE_GIB * GIB):
            errors.append(f"event {index} minimum-free byte threshold is invalid")
        free_bytes = row.get("free_bytes")
        free_gib = row.get("free_gib")
        if (
            not isinstance(free_bytes, int)
            or free_bytes <= 0
            or not isinstance(free_gib, (int, float))
            or not math.isclose(float(free_gib), free_bytes / GIB, rel_tol=0, abs_tol=1e-9)
        ):
            errors.append(f"event {index} free-space fields are invalid")

    pairs: list[dict[str, Any]] = []
    summary_events: dict[str, int] = {}
    for model_label in expected_labels:
        model_events = [(index, row) for index, row in relevant if row.get("model_label") == model_label]
        model_post_indices: list[int] = []
        for seed in seed_ids:
            seed_events = [(index, row) for index, row in model_events if row.get("seed") == seed]
            pre = [
                (index, row)
                for index, row in seed_events
                if row.get("phase") == "pre_seed"
                and row.get("enforced") is True
                and row.get("enough_free") is True
                and row.get("valid") is True
            ]
            post = [
                (index, row)
                for index, row in seed_events
                if row.get("phase") == "post_seed"
                and row.get("enforced") is False
                and row.get("valid") is True
            ]
            ordered_pair = next(
                (
                    (pre_index, pre_row, post_index, post_row)
                    for post_index, post_row in reversed(post)
                    for pre_index, pre_row in reversed(pre)
                    if pre_index < post_index
                ),
                None,
            )
            if ordered_pair is None:
                errors.append(f"{model_label} seed {seed} lacks a valid ordered pre/post storage gate")
                continue
            pre_index, pre_row, post_index, post_row = ordered_pair
            model_post_indices.append(post_index)
            pairs.append(
                {
                    "model_label": model_label,
                    "seed": seed,
                    "pre_event_index": pre_index,
                    "post_event_index": post_index,
                    "pre_free_bytes": pre_row["free_bytes"],
                    "post_free_bytes": post_row["free_bytes"],
                }
            )

        summaries = [
            index
            for index, row in model_events
            if row.get("phase") == "post_summary"
            and row.get("seed") is None
            and row.get("enforced") is False
            and row.get("valid") is True
        ]
        qualifying_summary = next(
            (
                index
                for index in reversed(summaries)
                if len(model_post_indices) == len(seed_ids) and index > max(model_post_indices)
            ),
            None,
        )
        if qualifying_summary is None:
            errors.append(f"{model_label} lacks a valid post-summary storage snapshot")
        else:
            summary_events[model_label] = qualifying_summary

    failed_enforced_events = sum(
        row.get("phase") == "pre_seed" and row.get("enforced") is True and row.get("valid") is not True
        for _index, row in relevant
    )
    return {
        "valid": not errors,
        "errors": errors,
        "project_root": expected_root,
        "history": str(history_path),
        "history_sha256": _sha256(history_path),
        "event_count": len(events),
        "relevant_event_count": len(relevant),
        "expected_model_labels": expected_labels,
        "expected_seed_ids": seed_ids,
        "expected_minimum_free_gib": EXPECTED_MINIMUM_FREE_GIB,
        "validated_pairs": pairs,
        "post_summary_event_indices": summary_events,
        "failed_enforced_event_count": failed_enforced_events,
        "minimum_valid_pre_seed_free_gib": min(
            (pair["pre_free_bytes"] / GIB for pair in pairs), default=None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--model-labels", required=True)
    parser.add_argument("--seed-ids", default=",".join(str(seed) for seed in EXPECTED_SEED_IDS))
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    labels = [item.strip() for item in args.model_labels.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seed_ids.split(",") if item.strip()]
    result = validate_storage_history(args.root, args.history, labels, seeds)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    print(rendered, end="")
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
