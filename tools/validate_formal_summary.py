#!/usr/bin/env python3
"""Rebuild and validate a formal LIBERO summary from raw evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from pathlib import Path
from typing import Any

try:
    from tools.summarize_results import EXPECTED_SEED_IDS, aggregate
    from tools.validate_eval_command import validate_command
except ModuleNotFoundError:  # Direct execution adds tools/ rather than the project root to sys.path.
    from summarize_results import EXPECTED_SEED_IDS, aggregate
    from validate_eval_command import validate_command


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path, errors: list[str], label: str) -> dict | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{label} missing or invalid: {path} ({exc})")
        return None
    if not isinstance(payload, dict):
        errors.append(f"{label} root is not a JSON object: {path}")
        return None
    return payload


def environment_markers_from_provenance(
    project_root: Path,
    errors: list[str],
) -> tuple[Path, tuple[str, ...]]:
    """Load the fixed runtime profile shared by seed and summary validators."""
    provenance_path = project_root / "outputs/provenance/assets.json"
    provenance = _load_object(provenance_path, errors, "project provenance")
    markers: tuple[str, ...] = ()
    if provenance is not None:
        if provenance.get("valid") is not True:
            errors.append("project provenance is not valid")
        if Path(provenance.get("project_root", "")).resolve() != project_root:
            errors.append("project provenance root does not match formal-summary project root")
        system = provenance.get("system", {})
        packages = system.get("packages", {}) if isinstance(system, dict) else {}
        gpu_parts = [part.strip() for part in str(system.get("nvidia_smi", "")).split(",")]
        python_version = str(system.get("python", "")).split()[0]
        commit = str(provenance.get("lerobot", {}).get("observed_commit", ""))
        values = [
            *gpu_parts[:2],
            f"Python {python_version}",
            *(f"{name}=={packages.get(name)}" for name in ("torch", "hf_libero", "mujoco")),
            f"{commit}#egg=lerobot",
        ]
        markers = tuple(value for value in values if value and not value.endswith("==None"))
    return provenance_path, markers


def _validate_run_artifacts(
    eval_path: Path,
    fresh_validation: dict,
    project_root: Path,
    model_label: str,
    seed: int,
    environment_markers: tuple[str, ...],
    expected_model: Path | None = None,
) -> dict[str, Any]:
    run_dir = eval_path.parent
    errors: list[str] = []
    paths = {
        "command": run_dir / "command.sh",
        "preflight": run_dir / "command_preflight.json",
        "protocol_validation": run_dir / "protocol_validation.json",
        "environment": run_dir / "environment.txt",
        "eval_log": project_root / f"outputs/logs/eval/{model_label}/seed_{seed}.log",
    }
    for label, path in paths.items():
        if not path.is_file() or path.stat().st_size <= 0:
            errors.append(f"missing or empty run artifact {label}: {path}")

    preflight = _load_object(paths["preflight"], errors, "run command preflight")
    if preflight is not None:
        try:
            command_tokens = shlex.split(paths["command"].read_text())
        except (OSError, ValueError) as exc:
            errors.append(f"cannot parse command.sh: {exc}")
        else:
            if command_tokens != preflight.get("command"):
                errors.append("command.sh does not exactly match command_preflight.json command")
            fresh_preflight = validate_command(command_tokens, seed)
            if expected_model is not None:
                observed_model = fresh_preflight.get("parsed", {}).get("model_path")
                if not observed_model or Path(observed_model).resolve() != expected_model.resolve():
                    errors.append("run checkpoint differs from the requested model")
            if fresh_preflight.get("valid") is not True:
                errors.extend(
                    f"fresh command preflight: {item}"
                    for item in fresh_preflight.get("errors", [])
                )
            if preflight != fresh_preflight:
                errors.append(
                    "command_preflight.json is stale or differs from a fresh parser/model-hash validation"
                )

    stored_validation = _load_object(
        paths["protocol_validation"], errors, "stored protocol validation"
    )
    if stored_validation is not None and stored_validation != fresh_validation:
        errors.append("stored protocol_validation.json is stale or differs from fresh validation")

    try:
        environment = paths["environment"].read_text()
    except OSError:
        environment = ""
    for marker in environment_markers:
        if marker not in environment:
            errors.append(f"environment.txt is missing fixed runtime marker: {marker}")

    hashes = {
        label: sha256(path) if path.is_file() else None
        for label, path in paths.items()
    }
    return {
        "valid": not errors,
        "errors": errors,
        "artifacts": {label: str(path) for label, path in paths.items()},
        "sha256": hashes,
    }


def validate_formal_summary(
    summary_path: Path,
    raw_root: Path,
    expected_model_label: str,
    expected_seed_ids: tuple[int, ...] = EXPECTED_SEED_IDS,
    expected_model: Path | None = None,
) -> dict[str, Any]:
    """Reaggregate all raw runs and require byte-independent semantic equality.

    ``aggregate`` performs the authoritative 4-suite x 10-task x 10-episode
    validation, including all 1,200 nonempty videos and command preflights.
    Requiring its result to equal the saved summary prevents a valid-looking but
    stale or edited metrics file from entering the final report.
    """

    summary_path = summary_path.resolve()
    raw_root = raw_root.resolve()
    project_root = summary_path.parents[2]
    errors: list[str] = []
    summary = _load_object(summary_path, errors, "metrics summary")
    provenance_path, environment_markers = environment_markers_from_provenance(
        project_root, errors
    )
    expected_seed_ids = tuple(expected_seed_ids)
    if not expected_seed_ids or len(set(expected_seed_ids)) != len(expected_seed_ids):
        raise ValueError(f"expected_seed_ids must be nonempty and unique: {expected_seed_ids!r}")
    expected_paths = [raw_root / f"seed_{seed}" / "eval_info.json" for seed in expected_seed_ids]
    observed_paths = sorted(raw_root.rglob("eval_info.json")) if raw_root.is_dir() else []
    if observed_paths != expected_paths:
        errors.append(
            "raw eval_info paths do not exactly match preregistered seeds: "
            f"observed={[str(path) for path in observed_paths]}, "
            f"expected={[str(path) for path in expected_paths]}"
        )

    rebuilt = None
    if summary is not None and observed_paths == expected_paths:
        try:
            rebuilt = aggregate(
                raw_root,
                expected_seeds=len(expected_seed_ids),
                episodes_per_task=10,
                expected_seed_ids=expected_seed_ids,
                require_video_evidence=True,
                require_command_preflight=True,
            )
        except Exception as exc:
            errors.append(f"cannot reaggregate raw formal evidence: {type(exc).__name__}: {exc}")

    if rebuilt is not None:
        models = rebuilt.get("models", {})
        if set(models) != {expected_model_label}:
            errors.append(
                f"rebuilt model labels are {sorted(str(item) for item in models)}, "
                f"expected only {expected_model_label!r}"
            )
        if summary != rebuilt:
            errors.append("saved metrics summary does not exactly match a fresh raw-evidence aggregation")

    run_evidence = []
    if rebuilt is not None:
        for run in rebuilt.get("runs", []):
            path = Path(run["path"]).resolve()
            validation = run.get("protocol_validation", {})
            artifacts = _validate_run_artifacts(
                path,
                validation,
                project_root,
                expected_model_label,
                int(run["seed"]),
                environment_markers,
                expected_model,
            )
            run_evidence.append(
                {
                    "seed": run.get("seed"),
                    "eval_info": str(path),
                    "eval_info_sha256": sha256(path),
                    "valid": validation.get("valid") is True,
                    "observed_episodes": validation.get("observed_episodes"),
                    "observed_nonempty_videos": validation.get("observed_nonempty_videos"),
                    "command_preflight_valid": validation.get("command_preflight_valid"),
                    "run_artifacts_valid": artifacts["valid"],
                    "run_artifacts": artifacts,
                    "errors": validation.get("errors", []),
                }
            )
            if validation.get("valid") is not True:
                errors.append(f"seed {run.get('seed')} raw protocol validation failed")
            errors.extend(
                f"seed {run.get('seed')} run provenance: {error}"
                for error in artifacts["errors"]
            )

    model = rebuilt.get("models", {}).get(expected_model_label, {}) if rebuilt else {}
    if rebuilt is not None and model.get("protocol_complete") is not True:
        errors.append("fresh raw-evidence aggregation is not protocol-complete")
    total_videos = sum(int(row.get("observed_nonempty_videos") or 0) for row in run_evidence)
    expected_videos = 400 * len(expected_seed_ids)
    if rebuilt is not None and total_videos != expected_videos:
        errors.append(f"observed {total_videos} nonempty videos, expected {expected_videos}")

    return {
        "valid": not errors,
        "errors": errors,
        "summary": str(summary_path),
        "summary_sha256": sha256(summary_path) if summary_path.is_file() else None,
        "provenance": str(provenance_path),
        "provenance_sha256": sha256(provenance_path) if provenance_path.is_file() else None,
        "environment_markers": list(environment_markers),
        "raw_root": str(raw_root),
        "expected_model_label": expected_model_label,
        "expected_seeds": list(expected_seed_ids),
        "canonical_summary_match": rebuilt is not None and summary == rebuilt,
        "raw_run_count": len(run_evidence),
        "observed_nonempty_videos": total_videos,
        "runs": run_evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--expected-model", type=Path)
    parser.add_argument("--expected-seed-ids", default=",".join(str(seed) for seed in EXPECTED_SEED_IDS))
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    expected_seed_ids = tuple(
        int(value.strip()) for value in args.expected_seed_ids.split(",") if value.strip()
    )
    result = validate_formal_summary(
        args.summary, args.raw_root, args.model_label, expected_seed_ids, args.expected_model
    )
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    print(rendered, end="")
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
