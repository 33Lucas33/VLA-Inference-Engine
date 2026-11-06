from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vla_inference_engine.contracts import ActionChunk, ObservationFrame, PolicyAdapter

if TYPE_CHECKING:
    from vla_inference_engine.control.clock import MonotonicClock


@dataclass(frozen=True)
class InferenceResult:
    chunk: ActionChunk | None
    request_mono_ns: int
    result_mono_ns: int

    @property
    def inference_ms(self) -> float:
        return (self.result_mono_ns - self.request_mono_ns) / 1_000_000


class SyncInferenceBackend:
    def __init__(self, policy: PolicyAdapter, clock: MonotonicClock) -> None:
        self.policy = policy
        self.clock = clock

    def infer(self, observation: ObservationFrame) -> InferenceResult:
        start = self.clock.now_ns()
        chunk = self.policy.predict_chunk(observation)
        return InferenceResult(chunk, start, self.clock.now_ns())
