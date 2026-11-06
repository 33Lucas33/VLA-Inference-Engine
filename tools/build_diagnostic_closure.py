#!/usr/bin/env python3
"""Build a hash-bound diagnostic closure from reviewed evidence and seed 4000 results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from tools.validate_diagnostic_closure import (
        sha256,
        triggered_models,
        validate_diagnostic_closure,
    )
except ModuleNotFoundError:
    from validate_diagnostic_closure import sha256, triggered_models, validate_diagnostic_closure


def load_object(path: Path, label: str) -> dict:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is missing or invalid: {path} ({exc})") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def relative(path: Path, root: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(root.resolve()))
    except ValueError as exc:
        raise ValueError(f"diagnostic artifact escapes project root: {path}") from exc


def build_closure(root: Path, review_path: Path, comparison_path: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    comparison_path = (
        comparison_path or root / "results/comparison/comparison.json"
    ).resolve()
    comparison = load_object(comparison_path, "comparison")
    expected = triggered_models(comparison)
    if comparison.get("requires_retest") is not True or not expected:
        raise ValueError("comparison does not require a diagnostic closure")

    review = load_object(review_path.resolve(), "diagnostic review")
    review_records = review.get("records")
    if not isinstance(review_records, list):
        raise ValueError("diagnostic review records must be a list")
    by_model = {
        record.get("model_label"): record
        for record in review_records
        if isinstance(record, dict) and isinstance(record.get("model_label"), str)
    }
    if len(by_model) != len(review_records) or set(by_model) != set(expected):
        raise ValueError(
            f"diagnostic review models are {sorted(by_model)}, expected {sorted(expected)}"
        )

    records = []
    for model_label, suites in expected.items():
        reviewed = by_model[model_label]
        source = root / f"results/{model_label}/metrics_summary.json"
        plan = root / f"results/diagnostics/{model_label}/diagnostic_plan.json"
        processor_contract = (
            root
            / f"results/diagnostics/{model_label}/processor_contract_validation.json"
        )
        retest = root / f"results/{model_label}_diagnostic_seed4000/metrics_summary.json"
        for label, path in (
            ("source summary", source),
            ("diagnostic plan", plan),
            ("processor contract", processor_contract),
            ("retest summary", retest),
        ):
            if not path.is_file() or path.stat().st_size <= 0:
                raise ValueError(f"{label} is missing or empty: {path}")
        records.append(
            {
                "model_label": model_label,
                "triggered_suites": suites,
                "diagnostic_seed": 4000,
                "source_summary": relative(source, root),
                "source_summary_sha256": sha256(source),
                "diagnostic_plan": relative(plan, root),
                "diagnostic_plan_sha256": sha256(plan),
                "processor_contract": relative(processor_contract, root),
                "processor_contract_sha256": sha256(processor_contract),
                "retest_summary": relative(retest, root),
                "retest_summary_sha256": sha256(retest),
                "root_cause": reviewed.get("root_cause"),
                "fix": reviewed.get("fix"),
                "failure_video_review": reviewed.get("failure_video_review"),
                "retest_conclusion": reviewed.get("retest_conclusion"),
                "resolved": True,
            }
        )

    closure = {
        "schema_version": 2,
        "resolved": True,
        "comparison_sha256": sha256(comparison_path),
        "review_source": relative(review_path, root),
        "review_source_sha256": sha256(review_path),
        "records": records,
    }
    validation = validate_diagnostic_closure(root, comparison_path, closure)
    if validation.get("valid") is not True:
        raise ValueError(
            "refusing to write an invalid diagnostic closure: "
            + "; ".join(validation.get("errors", []))
        )
    return closure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validation-out", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    comparison = (args.comparison or root / "results/comparison/comparison.json").resolve()
    output = (args.output or root / "results/diagnostics/diagnostic_closure.json").resolve()
    result = build_closure(root, args.review, comparison)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    validation = validate_diagnostic_closure(root, comparison, result)
    validation_out = (
        args.validation_out or root / "results/diagnostics/closure_validation.json"
    ).resolve()
    validation_out.parent.mkdir(parents=True, exist_ok=True)
    validation_out.write_text(json.dumps(validation, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"valid": True, "output": str(output), "records": len(result["records"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
