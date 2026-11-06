from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vla_inference_engine.contracts import ActionChunk, ActionSchema


@dataclass(frozen=True)
class Admission:
    accepted: bool
    reason: str
    first_tick: int | None = None
    slots: int = 0


class ActionQueue:
    """Single-writer, finite, generation-aware action buffer for P0/P3."""

    def __init__(self, schema: ActionSchema, max_slots: int = 64,
                 dt_ns: int = 50_000_000) -> None:
        if max_slots < 1 or dt_ns < 1:
            raise ValueError("max_slots and dt_ns must be positive")
        self.schema = schema
        self.max_slots = max_slots
        self.dt_ns = dt_ns
        self.generation_id = -1
        self.last_sequence_id = -1
        self.last_dispatched_tick = -1
        self._slots: dict[int, np.ndarray] = {}

    def reset(self, generation_id: int) -> None:
        if generation_id <= self.generation_id:
            raise ValueError("generation_id must increase")
        self.generation_id = generation_id
        self.last_sequence_id = -1
        self.last_dispatched_tick = -1
        self._slots.clear()

    def revoke(self) -> None:
        self._slots.clear()

    def admit(
        self,
        chunk: ActionChunk,
        *,
        now_ns: int,
        current_tick: int,
        origin_mono_ns: int,
        deadline_ns: int,
    ) -> Admission:
        if chunk.generation_id != self.generation_id:
            return Admission(False, "stale_generation")
        if chunk.sequence_id <= self.last_sequence_id:
            return Admission(False, "stale_sequence")
        if chunk.source_capture_mono_ns > chunk.received_mono_ns or chunk.received_mono_ns > now_ns:
            return Admission(False, "invalid_clock_order")
        if now_ns - chunk.source_capture_mono_ns > deadline_ns:
            return Admission(False, "deadline_expired")
        error = self.schema.validate(chunk)
        if error:
            return Admission(False, error)
        if chunk.dt_ns != self.dt_ns:
            return Admission(False, "dt_mismatch")
        # The arrival time determines the first still-executable tick. Past slots
        # and already dispatched ticks are never rewritten.
        arrival_tick = max(0, (chunk.received_mono_ns - origin_mono_ns + chunk.dt_ns - 1) // chunk.dt_ns)
        first_tick = max(current_tick, arrival_tick, self.last_dispatched_tick + 1)
        available = self.max_slots - len(self._slots)
        if available <= 0:
            return Admission(False, "buffer_full")
        inserted = 0
        for offset, action in enumerate(chunk.actions):
            target = first_tick + offset
            if target in self._slots:
                continue
            if inserted >= available:
                break
            self._slots[target] = action.copy()
            inserted += 1
        if inserted == 0:
            return Admission(False, "no_free_slots")
        self.last_sequence_id = chunk.sequence_id
        return Admission(True, "admitted", first_tick, inserted)

    def select(self, tick: int) -> np.ndarray | None:
        if tick <= self.last_dispatched_tick:
            raise ValueError("tick must increase")
        stale = [key for key in self._slots if key < tick]
        for key in stale:
            self._slots.pop(key)
        self.last_dispatched_tick = tick
        return self._slots.pop(tick, None)

    def coverage_ms(self, next_tick: int, dt_ns: int) -> float:
        count = 0
        while next_tick + count in self._slots:
            count += 1
        return count * dt_ns / 1_000_000

    def __len__(self) -> int:
        return len(self._slots)
