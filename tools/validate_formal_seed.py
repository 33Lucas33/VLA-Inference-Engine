#!/usr/bin/env python3
"""Validate whether one formal seed is safe to reuse without a rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from tools.summarize_results import validate_eval_payload
    from tools.validate_formal_summary import (
        _load_object,
        _validate_run_artifacts,
        environment_markers_from_provenance,
        sha256,
    )
except ModuleNotFoundError:
    from summarize_results import validate_eval_payload
    from validate_formal_summary import (
        _load_object,
        _validate_run_artifacts,
        environment_markers_from_provenance,
        sha256,
    )


def validate_formal_seed(
    eval_path: Path,
    project_root: Path,
    model_label: str,
    seed: int,
    episodes_per_task: int = 10,
    expected_model: Path | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    eval_path = eval_path.resolve()
    errors: list[str] = []
    expected_path = (
        project_root / f"results/raw/{model_label}/seed_{seed}/eval_info.json"
    ).resolve()
    if eval_path != expected_path:
        errors.append(f"eval_info path is {eval_path}, expected {expected_path}")

    payload = _load_object(eval_path, errors, "seed eval_info")
    provenance_path, markers = environment_markers_from_provenance(project_root, errors)
    fresh_validation = None
    artifact_validation = None
    if payload is not None:
        fresh_validation = validate_eval_payload(
            payload,
            episodes_per_task,
            eval_info_dir=eval_path.parent,
            require_video_evidence=True,
            require_command_preflight=True,
        )
        errors.extend(f"raw protocol: {item}" for item in fresh_validation["errors"])
        artifact_validation = _validate_run_artifacts(
            eval_path,
            fresh_validation,
            project_root,
            model_label,
            seed,
            markers,
            expected_model,
        )
        errors.extend(
            f"run provenance: {item}" for item in artifact_validation["errors"]
        )

    return {
        "valid": not errors,
        "errors": errors,
        "project_root": str(project_root),
        "model_label": model_label,
        "seed": seed,
        "episodes_per_task": episodes_per_task,
        "eval_info": str(eval_path),
        "eval_info_sha256": sha256(eval_path) if eval_path.is_file() else None,
        "provenance": str(provenance_path),
        "provenance_sha256": sha256(provenance_path) if provenance_path.is_file() else None,
        "environment_markers": list(markers),
        "protocol_validation": fresh_validation,
        "artifact_validation": artifact_validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_info", type=Path)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--expected-model", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    result = validate_formal_seed(
        args.eval_info,
        args.project_root,
        args.model_label,
        args.seed,
        args.episodes_per_task,
        args.expected_model,
    )
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    print(rendered, end="")
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
