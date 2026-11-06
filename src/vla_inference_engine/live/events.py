"""Small, serializable live-session contracts and control state machine."""

from __future__ import annotations

from dataclasses import dataclass

from vla_inference_engine.benchmark.protocol import SUITE_HORIZONS, TASK_IDS


@dataclass(frozen=True)
class Selection:
    suite: str
    task_id: int
    model: str
    seed: int = 1000
    init_state: int = 0

    @classmethod
    def parse(cls, data: dict) -> Selection:
        suite = str(data.get("suite", ""))
        task_id = int(data.get("task_id", -1))
        model = str(data.get("model", ""))
        seed = int(data.get("seed", 1000))
        init_state = int(data.get("init_state", 0))
        if suite not in SUITE_HORIZONS or task_id not in TASK_IDS:
            raise ValueError("Unknown LIBERO suite or task")
        if model not in {"smolvla_libero_paper", "smolvla_libero_100k"}:
            raise ValueError("Unknown checkpoint label")
        if seed < 0 or init_state < 0 or init_state > 99:
            raise ValueError("Seed and init_state must be nonnegative; init_state <= 99")
        return cls(suite, task_id, model, seed, init_state)


class RunGate:
    """The worker calls checkpoint() only between simulator steps."""

    def __init__(self):
        self.mode = "READY"
        self.stopped = False

    def command(self, name: str) -> str:
        if name == "stop":
            self.stopped = True
            self.mode = "STOPPING"
        elif self.stopped:
            raise ValueError("Session is stopping")
        elif name == "start" and self.mode == "READY":
            self.mode = "RUNNING"
        elif name == "resume" and self.mode == "PAUSED":
            self.mode = "RUNNING"
        elif name == "pause" and self.mode == "RUNNING":
            self.mode = "PAUSING"
        elif name == "step" and self.mode in {"READY", "PAUSED"}:
            self.mode = "STEPPING"
        else:
            raise ValueError(f"Cannot {name} from {self.mode}")
        return self.mode

    def checkpoint(self) -> str:
        if self.stopped:
            return "stop"
        if self.mode == "PAUSING":
            self.mode = "PAUSED"
        elif self.mode == "STEPPING":
            self.mode = "PAUSED"
        return self.mode
