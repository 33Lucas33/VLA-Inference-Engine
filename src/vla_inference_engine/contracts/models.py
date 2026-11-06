from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class ObservationFrame:
    generation_id: int
    tick_id: int
    capture_mono_ns: int
    features: Mapping[str, np.ndarray]
    task: str | None = None


@dataclass(frozen=True)
class ActionChunk:
    generation_id: int
    sequence_id: int
    source_capture_mono_ns: int
    received_mono_ns: int
    dt_ns: int
    actions: np.ndarray
    action_schema_id: str


@dataclass(frozen=True)
class ActionSchema:
    schema_id: str
    action_dim: int
    lower: float = -1.0
    upper: float = 1.0

    def validate(self, chunk: ActionChunk) -> str | None:
        actions = chunk.actions
        if chunk.action_schema_id != self.schema_id:
            return "schema_mismatch"
        if not isinstance(actions, np.ndarray) or actions.ndim != 2:
            return "invalid_shape"
        if actions.shape[0] < 1 or actions.shape[1] != self.action_dim:
            return "invalid_shape"
        if actions.dtype != np.float32:
            return "invalid_dtype"
        if not np.isfinite(actions).all():
            return "nonfinite_action"
        if np.any(actions < self.lower) or np.any(actions > self.upper):
            return "out_of_bounds"
        if chunk.dt_ns <= 0:
            return "invalid_dt"
        return None


class PolicyAdapter(Protocol):
    def predict_chunk(self, observation: ObservationFrame) -> ActionChunk | None: ...


class EnvironmentAdapter(Protocol):
    def reset(self, seed: int) -> ObservationFrame: ...
    def capture(self) -> ObservationFrame: ...
    def apply(self, action: np.ndarray) -> None: ...
    def hold(self) -> None: ...
    def close(self) -> None: ...
