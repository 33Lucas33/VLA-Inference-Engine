import json

import pytest

from tools.prepare_gap_diagnostics import prepare
from tools.validate_diagnostic_closure import sha256

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def write_comparison(root, *, triggered):
    path = root / "results/comparison/comparison.json"
    path.parent.mkdir(parents=True)
    rows = []
    for suite in SUITES:
        rows.append(
            {
                "suite": suite,
                "paper_requires_retest": suite in triggered,
                "local_requires_retest": False,
            }
        )
    path.write_text(
        json.dumps(
            {
                "paper_model": "paper",
                "local_model": "local",
                "paper_protocol_complete": True,
                "local_protocol_complete": True,
                "requires_retest": bool(triggered),
                "rows": rows,
            }
        )
    )
    return path


def write_source_summary(root, *, video_exists=True):
    eval_dir = root / "raw/seed_1000"
    eval_dir.mkdir(parents=True)
    video = eval_dir / "failed.mp4"
    if video_exists:
        video.write_bytes(b"video-evidence")
    eval_info = eval_dir / "eval_info.json"
    eval_info.write_text(
        json.dumps(
            {
                "per_task": [
                    {
                        "task_group": "libero_goal",
                        "task_id": 3,
                        "metrics": {
                            "successes": [False],
                            "video_paths": ["failed.mp4"],
                        },
                    }
                ]
            }
        )
    )
    suites = {
        suite: {"requires_retest": suite == "libero_goal"}
        for suite in SUITES
    }
    summary = {
        "models": {
            "paper": {
                "protocol_complete": True,
                "suites": suites,
            }
        },
        "per_task": [
            {
                "suite": "libero_goal",
                "task_id": 3,
                "task_name": "goal-task",
                "language_instruction": "put the object in the target bowl",
                "successes": 0,
                "episodes": 1,
            }
        ],
        "runs": [{"path": str(eval_info)}],
    }
    path = root / "results/paper/metrics_summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(summary))
    return path


def test_prepare_writes_evidence_bound_trigger_index(tmp_path):
    comparison = write_comparison(tmp_path, triggered={"libero_goal"})
    source = write_source_summary(tmp_path)

    result = prepare(tmp_path)

    assert result["requires_retest"] is True
    assert result["triggered_models"] == {"paper": ["libero_goal"]}
    assert result["comparison_sha256"] == sha256(comparison)
    assert len(result["plans"]) == 1
    record = result["plans"][0]
    assert record["source_summary_sha256"] == sha256(source)
    assert record["indexed_failed_videos"] == 1
    plan_path = tmp_path / record["diagnostic_plan"]
    assert record["diagnostic_plan_sha256"] == sha256(plan_path)
    plan = json.loads(plan_path.read_text())
    assert plan["diagnostic_seed"] == 4000
    assert plan["triggered_suites"] == ["libero_goal"]
    assert plan["lowest_tasks"]["libero_goal"][0]["failed_video_paths"] == [
        str((tmp_path / "raw/seed_1000/failed.mp4").resolve())
    ]


def test_prepare_no_trigger_needs_no_source_summary(tmp_path):
    write_comparison(tmp_path, triggered=set())
    result = prepare(tmp_path)
    assert result["requires_retest"] is False
    assert result["plans"] == []
    assert (tmp_path / "results/diagnostics/index.md").is_file()


def test_prepare_refuses_missing_failure_video(tmp_path):
    write_comparison(tmp_path, triggered={"libero_goal"})
    write_source_summary(tmp_path, video_exists=False)
    with pytest.raises(ValueError, match="missing or empty"):
        prepare(tmp_path)


def test_prepare_refuses_comparison_source_trigger_mismatch(tmp_path):
    write_comparison(tmp_path, triggered={"libero_goal"})
    source = write_source_summary(tmp_path)
    payload = json.loads(source.read_text())
    payload["models"]["paper"]["suites"]["libero_goal"]["requires_retest"] = False
    source.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Comparison/source trigger mismatch"):
        prepare(tmp_path)
