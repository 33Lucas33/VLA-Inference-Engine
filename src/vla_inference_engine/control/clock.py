from __future__ import annotations

import time
from typing import Protocol


class MonotonicClock(Protocol):
    def now_ns(self) -> int: ...


class SystemClock:
    def now_ns(self) -> int:
        return time.monotonic_ns()


class FakeClock:
    def __init__(self, start_ns: int = 0) -> None:
        self.value_ns = start_ns

    def now_ns(self) -> int:
        return self.value_ns

    def advance_ns(self, amount_ns: int) -> None:
        if amount_ns < 0:
            raise ValueError("time cannot move backwards")
        self.value_ns += amount_ns
