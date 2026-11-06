import numpy as np

from vla_inference_engine.contracts import ActionChunk, ActionSchema
from vla_inference_engine.control import ActionQueue


def chunk(*, generation=1, sequence=1, capture=0, received=0,
          actions=None, schema="mock-v1"):
    if actions is None:
        actions = np.full((3, 2), 0.25, dtype=np.float32)
    return ActionChunk(generation, sequence, capture, received, 50_000_000, actions, schema)


def admit(queue, value, *, now=0, tick=0, deadline=200_000_000):
    return queue.admit(value, now_ns=now, current_tick=tick,
                       origin_mono_ns=0, deadline_ns=deadline)


def test_reset_invalidates_prior_generation_and_does_not_turn_hold_into_zero_action():
    queue = ActionQueue(ActionSchema("mock-v1", 2))
    queue.reset(1)
    assert admit(queue, chunk()).accepted
    assert np.allclose(queue.select(0), [0.25, 0.25])
    queue.reset(2)
    assert len(queue) == 0
    assert admit(queue, chunk()).reason == "stale_generation"
    assert queue.select(0) is None


def test_late_invalid_and_duplicate_chunks_are_rejected_atomically():
    queue = ActionQueue(ActionSchema("mock-v1", 2))
    queue.reset(1)
    bad = np.array([[np.nan, 0.0], [0.2, 0.2]], dtype=np.float32)
    assert admit(queue, chunk(actions=bad)).reason == "nonfinite_action"
    assert len(queue) == 0
    assert admit(queue, chunk(capture=0, received=250_000_000), now=250_000_000,
                 tick=5).reason == "deadline_expired"
    assert admit(queue, chunk()).accepted
    assert admit(queue, chunk()).reason == "stale_sequence"
    assert len(queue) == 3


def test_dispatched_slot_cannot_be_overwritten():
    queue = ActionQueue(ActionSchema("mock-v1", 2))
    queue.reset(1)
    assert admit(queue, chunk()).accepted
    assert np.allclose(queue.select(0), [0.25, 0.25])
    newer = chunk(sequence=2, actions=np.full((3, 2), 0.8, dtype=np.float32))
    assert admit(queue, newer, tick=1).accepted
    # Existing future slots remain unchanged with the P0 append rule.
    assert np.allclose(queue.select(1), [0.25, 0.25])
    assert np.allclose(queue.select(2), [0.25, 0.25])
