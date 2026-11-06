from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Self


class TraceRecorder:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._handle = path.open("w", encoding="utf-8")

    def record(self, event: str, mono_ns: int, **fields: Any) -> None:
        self._handle.write(json.dumps({"event": event, "mono_ns": mono_ns, **fields}, sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
