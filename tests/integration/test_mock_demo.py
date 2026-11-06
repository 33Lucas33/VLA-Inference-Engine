import json
from argparse import Namespace

from vla_inference_engine.cli import run_mock


def test_mock_demo_has_explicit_hold_and_replayable_trace(tmp_path):
    output = tmp_path / "mock"
    assert run_mock(Namespace(out=str(output), ticks=100, seed=7, drop_every=1)) == 0
    summary = json.loads((output / "summary.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    events = [json.loads(line) for line in (output / "trace.jsonl").read_text().splitlines()]
    assert summary["ticks"] == 100
    assert summary["holds"] == 100
    assert summary["minimum_queue_coverage_ms"] == 0.0
    assert summary["dispatched"] == 0
    assert manifest["seed"] == 7
    assert manifest["gpu_required"] is False
    assert {"episode_reset", "observation_captured", "inference_requested",
            "inference_timeout", "action_held", "episode_done"} <= {e["event"] for e in events}
    assert not any(e["event"] == "action_dispatched" for e in events)


def test_normal_mock_dispatches_100_ticks(tmp_path):
    output = tmp_path / "normal"
    run_mock(Namespace(out=str(output), ticks=100, seed=0, drop_every=0))
    summary = json.loads((output / "summary.json").read_text())
    assert summary["dispatched"] == 100
    assert summary["holds"] == 0
