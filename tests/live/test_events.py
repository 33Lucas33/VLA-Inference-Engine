"""Control commands must release exactly the requested simulator steps."""

import threading
import time
from types import SimpleNamespace

import pytest

from vla_inference_engine.live.events import RunGate, Selection
from vla_inference_engine.live.worker import LiveWorker, StopRequested


def test_single_step_and_pause_at_step_boundary():
    gate = RunGate()
    assert gate.command("step") == "STEPPING"
    assert gate.checkpoint() == "PAUSED"
    assert gate.command("resume") == "RUNNING"
    assert gate.command("pause") == "PAUSING"
    assert gate.checkpoint() == "PAUSED"
    assert gate.command("step") == "STEPPING"
    assert gate.checkpoint() == "PAUSED"
    assert gate.command("stop") == "STOPPING"
    assert gate.checkpoint() == "stop"


def test_invalid_control_does_not_release_a_step():
    gate = RunGate()
    with pytest.raises(ValueError):
        gate.command("resume")
    assert gate.mode == "READY"
    gate.command("start")
    with pytest.raises(ValueError):
        gate.command("step")
    assert gate.mode == "RUNNING"


def test_selection_rejects_unregistered_model_or_task():
    valid = dict(suite="libero_spatial", task_id=0,
                 model="smolvla_libero_paper", seed=1000, init_state=0)
    assert Selection.parse(valid).task_id == 0
    for update in ({"model": "../../tmp"}, {"task_id": 10}, {"init_state": -1}):
        with pytest.raises(ValueError):
            Selection.parse(valid | update)


def test_pause_arriving_after_checkpoint_is_confirmed_at_same_boundary():
    worker = object.__new__(LiveWorker)
    worker.selection = Selection.parse(dict(suite="libero_spatial", task_id=0,
                                            model="smolvla_libero_paper"))
    worker.step = 1
    worker.env = SimpleNamespace(envs=[SimpleNamespace(
        _env=SimpleNamespace(check_success=lambda: False))])
    worker.condition = threading.Condition()
    worker.gate = RunGate()
    worker.gate.command("start")
    worker.last_action = {}
    worker.last_inference_ms = 0.0
    worker.started_at = time.monotonic()
    events = []
    worker.emit = lambda kind, **fields: events.append((kind, fields))

    def capture_with_late_pause():
        with worker.condition:
            worker.gate.command("pause")
            worker.condition.notify_all()
        return {"image": "", "image2": ""}

    worker.capture = capture_with_late_pause
    def run_render():
        try:
            worker.render(None)
        except StopRequested:
            pass

    thread = threading.Thread(target=run_render)
    thread.start()
    try:
        for _ in range(100):
            if any(kind == "state" and data["state"] == "PAUSED" for kind, data in events):
                break
            time.sleep(0.01)
        assert worker.gate.mode == "PAUSED"
        assert any(kind == "state" and data["state"] == "PAUSED" for kind, data in events)
    finally:
        with worker.condition:
            worker.gate.command("stop")
            worker.condition.notify_all()
        thread.join(timeout=2)
    assert not thread.is_alive()
