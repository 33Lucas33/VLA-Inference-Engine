import json

import pytest

from tools import build_diagnostic_closure as builder
from tools.validate_diagnostic_closure import sha256


def make_inputs(tmp_path):
    comparison = tmp_path / "results/comparison/comparison.json"
    comparison.parent.mkdir(parents=True)
    comparison.write_text(
        json.dumps(
            {
                "paper_model": "paper",
                "local_model": "local",
                "requires_retest": True,
                "rows": [
                    {
                        "suite": "libero_goal",
                        "paper_requires_retest": True,
                        "local_requires_retest": False,
                    }
                ],
            }
        )
    )
    source = tmp_path / "results/paper/metrics_summary.json"
    plan = tmp_path / "results/diagnostics/paper/diagnostic_plan.json"
    processor_contract = tmp_path / "results/diagnostics/paper/processor_contract_validation.json"
    retest = tmp_path / "results/paper_diagnostic_seed4000/metrics_summary.json"
    for path, payload in (
        (source, {"source": True}),
        (plan, {"model": "paper"}),
        (processor_contract, {"valid": True, "errors": []}),
        (retest, {"retest": True}),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    review = tmp_path / "results/diagnostics/diagnostic_review.json"
    review.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "model_label": "paper",
                        "root_cause": "The processor convention differed from the fixed evaluation protocol.",
                        "fix": "Aligned the processor ordering and reran every task with the fixed protocol.",
                        "failure_video_review": {
                            "reviewed_count": 2,
                            "categories": {"action_scale": 2},
                            "notes": "Reviewed two representative failed videos from the triggered goal suite.",
                        },
                        "retest_conclusion": "The independent seed 4000 evaluation completed after the correction.",
                    }
                ]
            }
        )
    )
    return comparison, source, plan, processor_contract, retest, review


def test_closure_builder_hashes_current_evidence_and_validates_before_write(tmp_path, monkeypatch):
    comparison, source, plan, processor_contract, retest, review = make_inputs(tmp_path)
    monkeypatch.setattr(
        builder,
        "validate_diagnostic_closure",
        lambda root, comparison_path, closure: {"valid": True, "errors": []},
    )
    result = builder.build_closure(tmp_path, review, comparison)
    assert result["resolved"] is True
    assert result["comparison_sha256"] == sha256(comparison)
    assert result["review_source_sha256"] == sha256(review)
    record = result["records"][0]
    assert record["triggered_suites"] == ["libero_goal"]
    assert record["source_summary_sha256"] == sha256(source)
    assert record["diagnostic_plan_sha256"] == sha256(plan)
    assert record["processor_contract_sha256"] == sha256(processor_contract)
    assert record["retest_summary_sha256"] == sha256(retest)


def test_closure_builder_requires_review_for_every_triggered_model(tmp_path):
    comparison, _source, _plan, _processor, _retest, review = make_inputs(tmp_path)
    review.write_text('{"records": []}')
    with pytest.raises(ValueError, match="review models"):
        builder.build_closure(tmp_path, review, comparison)
