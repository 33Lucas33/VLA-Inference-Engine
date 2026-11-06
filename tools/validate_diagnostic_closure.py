#!/usr/bin/env python3
"""Validate the evidence closure required after a preregistered >40% gap."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from tools.summarize_results import SUITE_ORDER
    from tools.validate_formal_summary import validate_formal_summary
except ModuleNotFoundError:
    from summarize_results import SUITE_ORDER
    from validate_formal_summary import validate_formal_summary


PLACEHOLDERS = {"todo", "tbd", "unknown", "n/a", "待定", "未知", "无"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, errors: list[str], label: str) -> dict | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{label} missing or invalid: {path} ({exc})")
        return None
    if not isinstance(value, dict):
        errors.append(f"{label} root is not an object: {path}")
        return None
    return value


def triggered_models(comparison: dict) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    roles = (
        ("paper_model", "paper_requires_retest"),
        ("local_model", "local_requires_retest"),
    )
    rows = comparison.get("rows", [])
    for model_key, flag_key in roles:
        label = comparison.get(model_key)
        suites = [row.get("suite") for row in rows if isinstance(row, dict) and row.get(flag_key) is True]
        suites = [suite for suite in suites if suite in SUITE_ORDER]
        if label and suites:
            result[str(label)] = suites
    return result


def meaningful_text(record: dict, key: str, errors: list[str], label: str) -> None:
    value = record.get(key)
    if (
        not isinstance(value, str)
        or len(value.strip()) < 20
        or value.strip().lower() in PLACEHOLDERS
    ):
        errors.append(f"{label}.{key} must be a specific explanation of at least 20 characters")


def check_hashed_artifact(
    root: Path,
    record: dict,
    path_key: str,
    hash_key: str,
    expected_path: Path,
    errors: list[str],
    label: str,
) -> tuple[Path, dict | None]:
    raw = record.get(path_key)
    path = (root / raw).resolve() if isinstance(raw, str) and not Path(raw).is_absolute() else Path(raw or "").resolve()
    try:
        path.relative_to(root)
    except ValueError:
        errors.append(f"{label}.{path_key} escapes the project root: {path}")
    if path != expected_path.resolve():
        errors.append(f"{label}.{path_key}={path}, expected {expected_path.resolve()}")
    payload = load_json(path, errors, f"{label}.{path_key}")
    if path.is_file():
        observed_hash = sha256(path)
        if record.get(hash_key) != observed_hash:
            errors.append(
                f"{label}.{hash_key}={record.get(hash_key)!r}, expected current SHA256 {observed_hash}"
            )
    return path, payload


def validate_retest_summary(payload: dict | None, model_label: str, errors: list[str]) -> None:
    if payload is None:
        return
    models = payload.get("models", {})
    if not isinstance(models, dict) or len(models) != 1:
        errors.append(f"{model_label} retest summary must contain exactly one model")
        return
    _name, model = next(iter(models.items()))
    if model.get("protocol_complete") is not True:
        errors.append(f"{model_label} seed4000 retest protocol is incomplete")
    if model.get("observed_seeds") != [4000] or model.get("expected_seed_ids") != [4000]:
        errors.append(f"{model_label} retest must use only independent seed 4000")
    if model.get("expected_seed_count") != 1:
        errors.append(f"{model_label} retest expected seed count is not 1")
    if model.get("expected_episodes_per_suite") != 100:
        errors.append(f"{model_label} retest expected episodes per suite is not 100")
    if model.get("requires_video_evidence") is not True:
        errors.append(f"{model_label} retest lacks required video evidence mode")
    if model.get("requires_command_preflight") is not True:
        errors.append(f"{model_label} retest lacks required command-preflight mode")
    suites = model.get("suites", {})
    for suite in SUITE_ORDER:
        row = suites.get(suite) if isinstance(suites, dict) else None
        if not isinstance(row, dict) or row.get("episodes") != 100:
            errors.append(f"{model_label} retest {suite} does not contain 100 episodes")
    overall = model.get("overall", {})
    if not isinstance(overall, dict) or overall.get("episodes") != 400:
        errors.append(f"{model_label} retest does not contain 400 total episodes")


def validate_diagnostic_closure(
    root: Path,
    comparison_path: Path,
    closure: dict | None,
) -> dict[str, Any]:
    root = root.resolve()
    errors: list[str] = []
    comparison = load_json(comparison_path, errors, "comparison")
    if comparison is None:
        return {"valid": False, "required": True, "errors": errors}
    expected = triggered_models(comparison)
    required = comparison.get("requires_retest") is True
    if not required:
        return {"valid": True, "required": False, "errors": [], "triggered_models": {}}
    if not expected:
        errors.append("comparison requires_retest=true but no model/suite trigger is encoded")
    if closure is None:
        errors.append("diagnostic closure is required but missing")
        return {"valid": False, "required": True, "errors": errors, "triggered_models": expected}

    if closure.get("schema_version") != 2:
        errors.append("diagnostic closure schema_version must be 2")
    if closure.get("resolved") is not True:
        errors.append("diagnostic closure resolved is not true")
    current_comparison_hash = sha256(comparison_path)
    if closure.get("comparison_sha256") != current_comparison_hash:
        errors.append("diagnostic closure comparison SHA256 does not match the current comparison")
    review_raw = closure.get("review_source")
    review_path = (
        (root / review_raw).resolve()
        if isinstance(review_raw, str) and not Path(review_raw).is_absolute()
        else Path(review_raw or "").resolve()
    )
    try:
        review_path.relative_to((root / "results/diagnostics").resolve())
    except ValueError:
        errors.append(
            "diagnostic closure review_source must remain under results/diagnostics: "
            f"{review_path}"
        )
    review_payload = load_json(review_path, errors, "diagnostic review source")
    if review_path.is_file() and closure.get("review_source_sha256") != sha256(review_path):
        errors.append("diagnostic closure review_source SHA256 does not match the current review")
    records = closure.get("records")
    if not isinstance(records, list):
        errors.append("diagnostic closure records must be a list")
        records = []
    by_model = {
        record.get("model_label"): record for record in records if isinstance(record, dict)
    }
    if len(by_model) != len(records):
        errors.append("diagnostic closure has invalid or duplicate model records")
    if set(by_model) != set(expected):
        errors.append(
            f"diagnostic closure models are {sorted(str(item) for item in by_model)}, "
            f"expected {sorted(expected)}"
        )
    review_records = review_payload.get("records") if review_payload is not None else None
    review_models = {
        record.get("model_label")
        for record in review_records
        if isinstance(record, dict)
    } if isinstance(review_records, list) else set()
    if review_models != set(expected) or (
        isinstance(review_records, list) and len(review_models) != len(review_records)
    ):
        errors.append(
            f"diagnostic review models are {sorted(str(item) for item in review_models)}, "
            f"expected {sorted(expected)}"
        )
    review_by_model = {
        record.get("model_label"): record
        for record in review_records
        if isinstance(record, dict)
    } if isinstance(review_records, list) else {}

    validated_records = []
    retest_raw_validations = {}
    for model_label, suites in expected.items():
        record = by_model.get(model_label)
        if not isinstance(record, dict):
            continue
        label = f"record[{model_label}]"
        reviewed_record = review_by_model.get(model_label)
        if isinstance(reviewed_record, dict):
            for key in (
                "root_cause",
                "fix",
                "failure_video_review",
                "retest_conclusion",
            ):
                if record.get(key) != reviewed_record.get(key):
                    errors.append(f"{label}.{key} differs from the hashed diagnostic review")
        if record.get("resolved") is not True:
            errors.append(f"{label}.resolved is not true")
        if record.get("triggered_suites") != suites:
            errors.append(f"{label}.triggered_suites={record.get('triggered_suites')!r}, expected {suites!r}")
        if record.get("diagnostic_seed") != 4000:
            errors.append(f"{label}.diagnostic_seed must be 4000")
        for key in ("root_cause", "fix", "retest_conclusion"):
            meaningful_text(record, key, errors, label)

        review = record.get("failure_video_review")
        if not isinstance(review, dict):
            errors.append(f"{label}.failure_video_review must be an object")
        else:
            reviewed = review.get("reviewed_count")
            categories = review.get("categories")
            if not isinstance(reviewed, int) or reviewed <= 0:
                errors.append(f"{label} must document at least one reviewed failure video")
            if (
                not isinstance(categories, dict)
                or not categories
                or any(not isinstance(value, int) or value < 0 for value in categories.values())
            ):
                errors.append(f"{label} failure-video categories must be nonnegative integer counts")
            elif isinstance(reviewed, int) and sum(categories.values()) != reviewed:
                errors.append(f"{label} failure-video category counts must sum to reviewed_count")
            meaningful_text(review, "notes", errors, f"{label}.failure_video_review")

        source_path = root / f"results/{model_label}/metrics_summary.json"
        plan_path = root / f"results/diagnostics/{model_label}/diagnostic_plan.json"
        processor_contract_path = (
            root
            / f"results/diagnostics/{model_label}/processor_contract_validation.json"
        )
        retest_path = root / f"results/{model_label}_diagnostic_seed4000/metrics_summary.json"
        _source, source = check_hashed_artifact(
            root, record, "source_summary", "source_summary_sha256", source_path, errors, label
        )
        _plan, plan = check_hashed_artifact(
            root, record, "diagnostic_plan", "diagnostic_plan_sha256", plan_path, errors, label
        )
        _processor_contract, processor_contract = check_hashed_artifact(
            root,
            record,
            "processor_contract",
            "processor_contract_sha256",
            processor_contract_path,
            errors,
            label,
        )
        _retest, retest = check_hashed_artifact(
            root, record, "retest_summary", "retest_summary_sha256", retest_path, errors, label
        )
        if source is not None:
            source_models = source.get("models", {})
            if model_label not in source_models or source_models[model_label].get("protocol_complete") is not True:
                errors.append(f"{label} source summary is not the protocol-complete formal model result")
        if plan is not None:
            if (
                plan.get("model") != model_label
                or plan.get("requires_retest") is not True
                or plan.get("triggered_suites") != suites
                or plan.get("diagnostic_seed") != 4000
            ):
                errors.append(f"{label} diagnostic plan does not match the current trigger")
        if processor_contract is not None and (
            processor_contract.get("valid") is not True
            or processor_contract.get("errors") != []
        ):
            errors.append(f"{label} processor contract is not a clean passing validation")
        validate_retest_summary(retest, model_label, errors)
        if retest is not None:
            retest_label = f"{model_label}_diagnostic_seed4000"
            raw_validation = validate_formal_summary(
                retest_path,
                root / f"results/raw/{retest_label}",
                retest_label,
                expected_seed_ids=(4000,),
            )
            retest_raw_validations[model_label] = raw_validation
            errors.extend(
                f"{label} raw seed4000 evidence: {error}"
                for error in raw_validation["errors"]
            )
            saved_raw_validation = load_json(
                root / f"results/{retest_label}/formal_validation.json",
                errors,
                f"{label} seed4000 formal-validation snapshot",
            )
            if saved_raw_validation is not None and saved_raw_validation != raw_validation:
                errors.append(
                    f"{label} seed4000 formal-validation snapshot is stale or mismatched"
                )
        validated_records.append(model_label)

    return {
        "valid": not errors,
        "required": True,
        "errors": errors,
        "comparison_sha256": current_comparison_hash,
        "triggered_models": expected,
        "validated_records": validated_records,
        "retest_raw_validations": retest_raw_validations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--closure", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    comparison = (args.comparison or root / "results/comparison/comparison.json").resolve()
    closure_path = (args.closure or root / "results/diagnostics/diagnostic_closure.json").resolve()
    closure = load_json(closure_path, [], "diagnostic closure") if closure_path.is_file() else None
    result = validate_diagnostic_closure(root, comparison, closure)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    print(rendered, end="")
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
