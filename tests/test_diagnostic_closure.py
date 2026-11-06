import copy
import json

from tools import validate_diagnostic_closure as validator
from tools.validate_diagnostic_closure import sha256, validate_diagnostic_closure


def write_comparison(path, *, requires_retest):
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "paper_model": "paper",
                "local_model": "local",
                "requires_retest": requires_retest,
                "rows": [
                    {
                        "suite": "libero_goal",
                        "paper_requires_retest": requires_retest,
                        "local_requires_retest": False,
                    }
                ],
            }
        )
    )


def test_diagnostic_closure_not_required_without_trigger(tmp_path):
    comparison = tmp_path / "results/comparison/comparison.json"
    write_comparison(comparison, requires_retest=False)
    result = validate_diagnostic_closure(tmp_path, comparison, None)
    assert result["valid"] is True
    assert result["required"] is False


def test_resolved_boolean_cannot_bypass_diagnostic_evidence(tmp_path):
    comparison = tmp_path / "results/comparison/comparison.json"
    write_comparison(comparison, requires_retest=True)
    result = validate_diagnostic_closure(tmp_path, comparison, {"resolved": True})
    assert result["valid"] is False
    assert result["required"] is True
    assert result["triggered_models"] == {"paper": ["libero_goal"]}
    assert any("schema_version" in error for error in result["errors"])
    assert any("comparison SHA256" in error for error in result["errors"])
    assert any("records" in error for error in result["errors"])


def test_complete_diagnostic_closure_revalidates_seed4000_raw_evidence(tmp_path, monkeypatch):
    comparison = tmp_path / "results/comparison/comparison.json"
    write_comparison(comparison, requires_retest=True)

    source = tmp_path / "results/paper/metrics_summary.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({"models": {"paper": {"protocol_complete": True}}}))

    plan = tmp_path / "results/diagnostics/paper/diagnostic_plan.json"
    plan.parent.mkdir(parents=True)
    plan.write_text(
        json.dumps(
            {
                "model": "paper",
                "requires_retest": True,
                "triggered_suites": ["libero_goal"],
                "diagnostic_seed": 4000,
            }
        )
    )

    processor_contract = tmp_path / "results/diagnostics/paper/processor_contract_validation.json"
    processor_contract.write_text(json.dumps({"valid": True, "errors": []}))

    retest = tmp_path / "results/paper_diagnostic_seed4000/metrics_summary.json"
    retest.parent.mkdir(parents=True)
    retest.write_text(
        json.dumps(
            {
                "models": {
                    "paper_diagnostic_seed4000": {
                        "protocol_complete": True,
                        "observed_seeds": [4000],
                        "expected_seed_ids": [4000],
                        "expected_seed_count": 1,
                        "expected_episodes_per_suite": 100,
                        "requires_video_evidence": True,
                        "requires_command_preflight": True,
                        "suites": {
                            suite: {"episodes": 100}
                            for suite in (
                                "libero_spatial",
                                "libero_object",
                                "libero_goal",
                                "libero_10",
                            )
                        },
                        "overall": {"episodes": 400},
                    }
                }
            }
        )
    )

    raw_validation = {
        "valid": True,
        "errors": [],
        "expected_seeds": [4000],
        "observed_nonempty_videos": 400,
    }
    formal = retest.parent / "formal_validation.json"
    formal.write_text(json.dumps(raw_validation))
    monkeypatch.setattr(validator, "validate_formal_summary", lambda *args, **kwargs: raw_validation)

    review_record = {
        "model_label": "paper",
        "root_cause": "The evaluation processor convention was inconsistent with the fixed protocol.",
        "fix": "Aligned processor ordering and reran the complete independent-seed evaluation.",
        "failure_video_review": {
            "reviewed_count": 2,
            "categories": {"action_scale": 2},
            "notes": "Reviewed two representative failed videos selected from the triggered suite.",
        },
        "retest_conclusion": "The independent seed 4000 rerun completed with protocol-valid evidence.",
    }
    review = tmp_path / "results/diagnostics/diagnostic_review.json"
    review.write_text(json.dumps({"records": [review_record]}))

    closure = {
        "schema_version": 2,
        "resolved": True,
        "comparison_sha256": sha256(comparison),
        "review_source": "results/diagnostics/diagnostic_review.json",
        "review_source_sha256": sha256(review),
        "records": [
            {
                "model_label": "paper",
                "triggered_suites": ["libero_goal"],
                "diagnostic_seed": 4000,
                "source_summary": "results/paper/metrics_summary.json",
                "source_summary_sha256": sha256(source),
                "diagnostic_plan": "results/diagnostics/paper/diagnostic_plan.json",
                "diagnostic_plan_sha256": sha256(plan),
                "processor_contract": "results/diagnostics/paper/processor_contract_validation.json",
                "processor_contract_sha256": sha256(processor_contract),
                "retest_summary": "results/paper_diagnostic_seed4000/metrics_summary.json",
                "retest_summary_sha256": sha256(retest),
                "root_cause": review_record["root_cause"],
                "fix": review_record["fix"],
                "failure_video_review": review_record["failure_video_review"],
                "retest_conclusion": review_record["retest_conclusion"],
                "resolved": True,
            }
        ],
    }
    result = validate_diagnostic_closure(tmp_path, comparison, closure)
    assert result["valid"] is True, result["errors"]
    assert result["retest_raw_validations"]["paper"]["observed_nonempty_videos"] == 400

    tampered_review_copy = copy.deepcopy(closure)
    tampered_review_copy["records"][0]["fix"] = (
        "A different unreviewed fix was inserted directly into the closure artifact."
    )
    result = validate_diagnostic_closure(tmp_path, comparison, tampered_review_copy)
    assert result["valid"] is False
    assert any("differs from the hashed diagnostic review" in error for error in result["errors"])

    processor_contract.write_text(json.dumps({"valid": False, "errors": ["drift"]}))
    result = validate_diagnostic_closure(tmp_path, comparison, closure)
    assert result["valid"] is False
    assert any("processor contract" in error for error in result["errors"])
