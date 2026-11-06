from __future__ import annotations

import numpy as np

from vla_inference_engine.contracts import ActionChunk, ObservationFrame
from vla_inference_engine.control.clock import MonotonicClock


class MockEnvironment:
    def __init__(self, clock: MonotonicClock, action_dim: int = 2) -> None:
        self.clock = clock
        self.action_dim = action_dim
        self.generation_id = 0
        self.tick_id = 0
        self.state = np.zeros(action_dim, dtype=np.float32)
        self.hold_count = 0
        self.apply_count = 0

    def reset(self, seed: int) -> ObservationFrame:
        self.generation_id += 1
        self.tick_id = 0
        self.state = np.zeros(self.action_dim, dtype=np.float32)
        self.hold_count = 0
        self.apply_count = 0
        return self.capture()

    def capture(self) -> ObservationFrame:
        return ObservationFrame(
            self.generation_id, self.tick_id, self.clock.now_ns(),
            {"observation.state": self.state.copy()}, "mock_task",
        )

    def apply(self, action: np.ndarray) -> None:
        if action.shape != (self.action_dim,) or action.dtype != np.float32:
            raise ValueError("mock action shape/dtype mismatch")
        self.state += action
        self.apply_count += 1
        self.tick_id += 1

    def hold(self) -> None:
        # A hold is a distinct environment operation, never an implicit zero action.
        self.hold_count += 1
        self.tick_id += 1

    def close(self) -> None:
        pass


class MockPolicy:
    def __init__(self, clock: MonotonicClock, *, action_dim: int = 2, horizon: int = 4,
                 dt_ns: int = 50_000_000, drop_every: int = 0,
                 schema_id: str = "mock-v1") -> None:
        self.clock = clock
        self.action_dim = action_dim
        self.horizon = horizon
        self.dt_ns = dt_ns
        self.drop_every = drop_every
        self.schema_id = schema_id
        self.sequence_id = 0

    def predict_chunk(self, observation: ObservationFrame) -> ActionChunk | None:
        self.sequence_id += 1
        if self.drop_every and self.sequence_id % self.drop_every == 0:
            return None
        actions = np.full((self.horizon, self.action_dim), 0.1, dtype=np.float32)
        return ActionChunk(
            observation.generation_id, self.sequence_id, observation.capture_mono_ns,
            self.clock.now_ns(), self.dt_ns, actions, self.schema_id,
        )
