"""Single local session supervisor; simulator and CUDA live in a child process."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from vla_inference_engine.benchmark.libero import inspect_smolvla_checkpoint
from vla_inference_engine.live.events import Selection

TERMINAL = {"FINISHED", "CLOSED", "ERROR"}


class LiveSession:
    def __init__(self, root: Path, selection: Selection, session_id: str | None = None,
                 generation: int = 0):
        self.root = root
        self.selection = selection
        self.id = session_id or uuid.uuid4().hex
        self.generation = generation
        self.dir = root / "runs/live" / self.id / f"generation_{generation}"
        self.dir.mkdir(parents=True, exist_ok=False)
        self.state = "LOADING"
        self.event_seq = 0
        self.latest: dict = {"type": "state", "state": "LOADING", "step": 0}
        self.latest_frame: dict | None = None
        self.lock = threading.RLock()
        self.command_ids: set[str] = set()
        model_path = (root / "artifacts/models" / selection.model).resolve()
        spec = inspect_smolvla_checkpoint(model_path)
        if not spec["valid"]:
            raise ValueError(f"Checkpoint invalid: {spec['errors']}")
        config = {"session_id": self.id, "generation": generation,
                  "selection": selection.__dict__, "model_path": str(model_path),
                  "run_dir": str(self.dir)}
        (self.dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        self.stderr = (self.dir / "worker.log").open("w", encoding="utf-8")
        environment = dict(os.environ)
        environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                           HF_DATASETS_OFFLINE="1")
        self.proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "vla_inference_engine.live.worker", str(self.dir / "config.json")],
            cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.stderr, text=True, bufsize=1, env=environment,
        )
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            try:
                event = json.loads(line)
                if event.get("generation") != self.generation:
                    continue
                with self.lock:
                    self.event_seq += 1
                    event["event_seq"] = self.event_seq
                    self.latest = event
                    if event["type"] == "frame":
                        self.latest_frame = event
                    if event["type"] == "state":
                        self.state = event["state"]
                    elif event["type"] == "command" and event["state"] in {
                        "RUNNING", "PAUSING", "STEPPING", "STOPPING"
                    }:
                        self.state = event["state"]
            except (ValueError, KeyError):
                continue
        self.proc.wait()
        with self.lock:
            if self.state not in TERMINAL:
                self.state = "ERROR"
                self.latest = {"type": "state", "state": "ERROR",
                               "error": f"Worker exited with code {self.proc.returncode}; see worker.log",
                               "generation": self.generation}
                self.event_seq += 1
                self.latest["event_seq"] = self.event_seq
        self.stderr.close()

    def snapshot(self) -> dict:
        with self.lock:
            return {"session_id": self.id, "generation": self.generation,
                    "selection": self.selection.__dict__, "state": self.state,
                    "event_seq": self.event_seq, "latest": self.latest,
                    "frame": self.latest_frame, "run_dir": str(self.dir)}

    def send(self, command: str, command_id: str) -> dict:
        if command not in {"start", "pause", "resume", "step", "stop"}:
            raise ValueError("Unknown command")
        with self.lock:
            if command_id in self.command_ids:
                return {"duplicate": True, "state": self.state}
            if self.proc.poll() is not None or self.state in TERMINAL:
                raise ValueError("Session has finished")
            if command == "start" and self.state != "READY":
                raise ValueError("Start requires READY")
            if command == "pause" and self.state != "RUNNING":
                raise ValueError("Pause requires RUNNING")
            if command == "resume" and self.state != "PAUSED":
                raise ValueError("Resume requires PAUSED")
            if command == "step" and self.state not in {"READY", "PAUSED"}:
                raise ValueError("Step requires READY or PAUSED")
            self.command_ids.add(command_id)
            assert self.proc.stdin is not None
            self.proc.stdin.write(json.dumps({"command": command, "command_id": command_id}) + "\n")
            self.proc.stdin.flush()
            return {"accepted": True, "state": self.state}

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.send("stop", uuid.uuid4().hex)
            except (ValueError, BrokenPipeError):
                pass
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
        if self.proc.stdin is not None:
            self.proc.stdin.close()


class SessionManager:
    def __init__(self, root: Path):
        self.root = root
        self.active: LiveSession | None = None
        self.lock = threading.RLock()

    def create(self, selection: Selection) -> LiveSession:
        with self.lock:
            if self.active is not None:
                self.active.close()
            self.active = LiveSession(self.root, selection)
            return self.active

    def get(self, session_id: str) -> LiveSession:
        with self.lock:
            if self.active is None or self.active.id != session_id:
                raise ValueError("Unknown or superseded session")
            return self.active

    def reset(self, session_id: str) -> LiveSession:
        with self.lock:
            old = self.get(session_id)
            old.close()
            self.active = LiveSession(self.root, old.selection, session_id=old.id,
                                      generation=old.generation + 1)
            return self.active

    def close(self) -> None:
        with self.lock:
            if self.active is not None:
                self.active.close()
