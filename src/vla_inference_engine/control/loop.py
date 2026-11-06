from __future__ import annotations

from dataclasses import dataclass

from vla_inference_engine.contracts import EnvironmentAdapter, PolicyAdapter
from vla_inference_engine.control.clock import MonotonicClock
from vla_inference_engine.control.queue import ActionQueue
from vla_inference_engine.inference.sync import SyncInferenceBackend
from vla_inference_engine.telemetry import TraceRecorder


@dataclass(frozen=True)
class LoopSummary:
    ticks: int
    dispatched: int
    holds: int
    rejected_chunks: int
    generation_id: int
    minimum_queue_coverage_ms: float


class ControlLoop:
    def __init__(
        self,
        environment: EnvironmentAdapter,
        policy: PolicyAdapter,
        queue: ActionQueue,
        clock: MonotonicClock,
        trace: TraceRecorder,
        *,
        dt_ns: int = 50_000_000,
        deadline_ns: int = 200_000_000,
    ) -> None:
        self.environment = environment
        self.policy = policy
        self.backend = SyncInferenceBackend(policy, clock)
        self.queue = queue
        self.clock = clock
        self.trace = trace
        self.dt_ns = dt_ns
        self.deadline_ns = deadline_ns

    def run(self, *, ticks: int, seed: int) -> LoopSummary:
        if ticks < 1 or self.dt_ns < 1 or self.deadline_ns < 1:
            raise ValueError("ticks, dt_ns and deadline_ns must be positive")
        first = self.environment.reset(seed)
        self.queue.reset(first.generation_id)
        origin = first.capture_mono_ns
        self.trace.record("episode_reset", self.clock.now_ns(), seed=seed,
                          generation_id=first.generation_id, task=first.task)
        dispatched = holds = rejected = 0
        minimum_coverage_ms = float("inf")
        try:
            for tick in range(ticks):
                obs = self.environment.capture()
                now = self.clock.now_ns()
                self.trace.record("observation_captured", now, generation_id=obs.generation_id,
                                  tick_id=tick, capture_mono_ns=obs.capture_mono_ns,
                                  feature_shapes={k: list(v.shape) for k, v in obs.features.items()})
                if len(self.queue) == 0:
                    self.trace.record("inference_requested", now, generation_id=obs.generation_id,
                                      tick_id=tick, capture_mono_ns=obs.capture_mono_ns)
                    result = self.backend.infer(obs)
                    chunk = result.chunk
                    received = result.result_mono_ns
                    if chunk is not None:
                        self.trace.record("inference_completed", received, generation_id=chunk.generation_id,
                                          sequence_id=chunk.sequence_id, received_mono_ns=chunk.received_mono_ns)
                        result = self.queue.admit(chunk, now_ns=received, current_tick=tick,
                                                  origin_mono_ns=origin, deadline_ns=self.deadline_ns)
                        self.trace.record("chunk_admitted" if result.accepted else "chunk_rejected",
                                          received, generation_id=chunk.generation_id,
                                          sequence_id=chunk.sequence_id, reason=result.reason,
                                          first_tick=result.first_tick, slots=result.slots)
                        if not result.accepted:
                            rejected += 1
                    else:
                        self.trace.record("inference_timeout", received, generation_id=obs.generation_id,
                                          tick_id=tick)
                action = self.queue.select(tick)
                coverage_ms = self.queue.coverage_ms(tick + 1, self.dt_ns)
                minimum_coverage_ms = min(minimum_coverage_ms, coverage_ms)
                if action is None:
                    self.environment.hold()
                    holds += 1
                    self.trace.record("action_held", self.clock.now_ns(), generation_id=obs.generation_id,
                                      tick_id=tick, reason="queue_empty",
                                      queue_coverage_ms=coverage_ms)
                else:
                    self.environment.apply(action)
                    dispatched += 1
                    self.trace.record("action_dispatched", self.clock.now_ns(),
                                      generation_id=obs.generation_id, tick_id=tick,
                                      action=action.tolist(), action_schema_id=self.queue.schema.schema_id,
                                      queue_coverage_ms=coverage_ms)
                if hasattr(self.clock, "advance_ns"):
                    self.clock.advance_ns(self.dt_ns)
            self.trace.record("episode_done", self.clock.now_ns(), generation_id=first.generation_id,
                              ticks=ticks, dispatched=dispatched, holds=holds)
            return LoopSummary(ticks, dispatched, holds, rejected, first.generation_id,
                               minimum_coverage_ms)
        finally:
            self.queue.revoke()
            self.environment.close()
