#!/usr/bin/env python3
"""Prepare immutable, evidence-bound diagnosis inputs after formal comparison."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from tools.diagnose_gap import diagnose, write_outputs
    from tools.summarize_results import SUITE_ORDER
    from tools.validate_diagnostic_closure import sha256, triggered_models
except ModuleNotFoundError:  # Direct execution adds tools/ rather than the project root to sys.path.
    from diagnose_gap import diagnose, write_outputs
    from summarize_results import SUITE_ORDER
    from validate_diagnostic_closure import sha256, triggered_models


def _load_object(path: Path, label: str) -> dict:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is missing or invalid: {path} ({exc})") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be a JSON object: {path}")
    return payload


def _safe_label(label: str) -> None:
    if not label or Path(label).name != label or label in {".", ".."}:
        raise ValueError(f"Unsafe model label in comparison: {label!r}")


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as exc:
        raise ValueError(f"Diagnostic artifact escapes project root: {path}") from exc


def prepare(
    root: Path,
    comparison_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    root = root.resolve()
    comparison_path = (comparison_path or root / "results/comparison/comparison.json").resolve()
    output_dir = (output_dir or root / "results/diagnostics").resolve()
    _relative(comparison_path, root)
    _relative(output_dir, root)

    comparison = _load_object(comparison_path, "comparison")
    expected = triggered_models(comparison)
    required = comparison.get("requires_retest") is True
    if required != bool(expected):
        raise ValueError(
            "comparison requires_retest is inconsistent with its per-model/per-suite trigger flags"
        )

    protocol_flags: dict[str, bool] = {}
    for role in ("paper", "local"):
        label = comparison.get(f"{role}_model")
        if isinstance(label, str):
            protocol_flags[label] = comparison.get(f"{role}_protocol_complete") is True

    plans: list[dict] = []
    for model_label, suites in expected.items():
        _safe_label(model_label)
        if protocol_flags.get(model_label) is not True:
            raise ValueError(f"Triggered model is not marked protocol-complete: {model_label}")

        source_path = (root / "results" / model_label / "metrics_summary.json").resolve()
        source = _load_object(source_path, f"source summary for {model_label}")
        models = source.get("models")
        if not isinstance(models, dict) or set(models) != {model_label}:
            raise ValueError(
                f"Source summary must contain only comparison model {model_label!r}: {source_path}"
            )
        source_model = models[model_label]
        if not isinstance(source_model, dict) or source_model.get("protocol_complete") is not True:
            raise ValueError(f"Source summary is not protocol-complete: {source_path}")
        source_suites = source_model.get("suites", {})
        source_triggers = [
            suite
            for suite in SUITE_ORDER
            if isinstance(source_suites, dict)
            for values in [source_suites.get(suite)]
            if isinstance(values, dict) and values.get("requires_retest") is True
        ]
        if source_triggers != suites:
            raise ValueError(
                f"Comparison/source trigger mismatch for {model_label}: "
                f"comparison={suites}, source={source_triggers}"
            )

        result = diagnose(source_path)
        if (
            result.get("model") != model_label
            or result.get("triggered_suites") != suites
            or result.get("requires_retest") is not True
            or result.get("diagnostic_seed") != 4000
        ):
            raise ValueError(f"Generated diagnostic plan does not match comparison: {model_label}")
        if int(result.get("indexed_failed_videos", 0)) <= 0:
            raise ValueError(f"No failed-video evidence was indexed for triggered model: {model_label}")

        model_dir = output_dir / model_label
        write_outputs(result, model_dir)
        plan_json = model_dir / "diagnostic_plan.json"
        plan_md = model_dir / "diagnostic_plan.md"
        review_template = model_dir / "review_template.json"
        review_template.write_text(
            json.dumps(
                {
                    "model_label": model_label,
                    "root_cause": "",
                    "fix": "",
                    "failure_video_review": {
                        "reviewed_count": 0,
                        "categories": {
                            "perception_or_localization": 0,
                            "gripper_timing": 0,
                            "action_scale": 0,
                            "long_horizon": 0,
                            "simulator": 0,
                        },
                        "notes": "",
                    },
                    "retest_conclusion": "",
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
        plans.append(
            {
                "model_label": model_label,
                "triggered_suites": suites,
                "diagnostic_seed": 4000,
                "source_summary": _relative(source_path, root),
                "source_summary_sha256": sha256(source_path),
                "diagnostic_plan": _relative(plan_json, root),
                "diagnostic_plan_sha256": sha256(plan_json),
                "diagnostic_plan_markdown": _relative(plan_md, root),
                "review_template": _relative(review_template, root),
                "indexed_failed_videos": result["indexed_failed_videos"],
                "indexed_tasks": sum(len(items) for items in result["lowest_tasks"].values()),
            }
        )

    index = {
        "schema_version": 1,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison": _relative(comparison_path, root),
        "comparison_sha256": sha256(comparison_path),
        "requires_retest": required,
        "triggered_models": expected,
        "plans": plans,
        "next_action": (
            "Review failed videos and record root cause/fix before explicitly approving seed 4000 retest."
            if required
            else "No diagnostic retest is required by the preregistered threshold."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n"
    )
    lines = [
        "# 40% 差距诊断索引",
        "",
        f"- comparison SHA256：`{index['comparison_sha256']}`",
        f"- 是否需要诊断复测：{required}",
        f"- 触发模型：{', '.join(expected) or '无'}",
        "",
    ]
    for plan in plans:
        lines.extend(
            [
                f"## {plan['model_label']}",
                "",
                f"- 触发套件：{', '.join(plan['triggered_suites'])}",
                f"- 已索引失败视频：{plan['indexed_failed_videos']}",
                f"- 诊断计划：`{plan['diagnostic_plan_markdown']}`",
                f"- 根因/视频审查模板：`{plan['review_template']}`",
                f"- 独立复测 seed：{plan['diagnostic_seed']}（审查并批准后才运行）",
                "",
            ]
        )
    lines.append(index["next_action"])
    (output_dir / "index.md").write_text("\n".join(lines) + "\n")
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = prepare(args.root, args.comparison, args.output_dir)
    print(
        json.dumps(
            {
                "requires_retest": result["requires_retest"],
                "triggered_models": result["triggered_models"],
                "index": "results/diagnostics/index.json",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
