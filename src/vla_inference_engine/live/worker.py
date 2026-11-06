"""One model, one LIBERO task and one CUDA context per child process."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
import threading
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from vla_inference_engine.benchmark.protocol import SUITE_HORIZONS
from vla_inference_engine.live.events import RunGate, Selection


class StopRequested(Exception):
    pass


class LiveWorker:
    def __init__(self, config: dict):
        self.selection = Selection.parse(config["selection"])
        self.session_id = config["session_id"]
        self.generation = int(config["generation"])
        self.run_dir = Path(config["run_dir"])
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trace = (self.run_dir / "trace.jsonl").open("a", encoding="utf-8")
        self.emit_lock = threading.Lock()
        self.condition = threading.Condition()
        self.gate = RunGate()
        self.step = 0
        self.last_action: dict[str, list[float]] = {}
        self.last_inference_ms = 0.0
        self.started_at = time.monotonic()
        self.env: Any = None
        self.model_path = Path(config["model_path"])
        self.completed = False
        self.success: bool | None = None
        self.finish_reason: str | None = None

    def emit(self, kind: str, **data: Any) -> None:
        event = {"type": kind, "session_id": self.session_id,
                 "generation": self.generation, "step": self.step,
                 "time_ns": time.monotonic_ns(), **data}
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        with self.emit_lock:
            if kind == "frame":
                frame_dir = self.run_dir / "frames"
                frame_dir.mkdir(exist_ok=True)
                for name, encoded in data["cameras"].items():
                    (frame_dir / f"{self.step:04d}_{name}.jpg").write_bytes(base64.b64decode(encoded))
                trace_event = {**event, "cameras": sorted(data["cameras"])}
            else:
                trace_event = event
            self.trace.write(json.dumps(trace_event, ensure_ascii=False) + "\n")
            self.trace.flush()
            sys.__stdout__.write(line + "\n")
            sys.__stdout__.flush()

    def listen(self) -> None:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                command = request["command"]
                with self.condition:
                    mode = self.gate.command(command)
                    self.condition.notify_all()
                self.emit("command", command=command, command_id=request.get("command_id"), state=mode)
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                self.emit("command_error", error=str(exc))
        with self.condition:
            self.gate.command("stop")
            self.condition.notify_all()

    @staticmethod
    def vector(tensor: Any) -> list[float]:
        import torch

        if isinstance(tensor, torch.Tensor):
            return tensor.detach().to("cpu").reshape(-1).float().tolist()
        return [float(x) for x in tensor.reshape(-1).tolist()]

    @staticmethod
    def image_base64(image: Any) -> str:
        from PIL import Image

        # LIBERO's official render() flips both axes for display; do the same for each camera.
        with io.BytesIO() as stream:
            Image.fromarray(image[::-1, ::-1].copy()).save(stream, format="JPEG", quality=75)
            return base64.b64encode(stream.getvalue()).decode("ascii")

    def capture(self) -> dict[str, Any]:
        raw = self.env.envs[0]._env.env._get_observations()
        pixels = self.env.envs[0]._format_raw_obs(raw)["pixels"]
        return {name: self.image_base64(frame) for name, frame in pixels.items()}

    def save_videos(self) -> list[str]:
        """Replay one observation per step at a declared 10 fps, independent of wall time."""
        import imageio.v2 as imageio
        import numpy as np
        from PIL import Image

        frame_dir = self.run_dir / "frames"
        videos = []
        for camera in ("image", "image2"):
            frames = sorted(frame_dir.glob(f"*_{camera}.jpg"))
            if not frames:
                continue
            output = self.run_dir / f"{camera}.mp4"
            with imageio.get_writer(output, fps=10, codec="libx264", quality=7) as writer:
                for frame in frames:
                    with Image.open(frame) as image:
                        writer.append_data(np.asarray(image.convert("RGB")))
            videos.append(output.name)
        return videos

    def render(self, _env: Any) -> None:
        terminal = self.step >= SUITE_HORIZONS[self.selection.suite]
        if self.step and not terminal:
            terminal = bool(self.env.envs[0]._env.check_success())
        with self.condition:
            if self.step:
                mode = self.gate.checkpoint()
            else:
                mode = self.gate.mode
        if mode == "stop":
            raise StopRequested()
        snapshot = self.capture()
        self.emit("frame", state=mode, cameras=snapshot,
                  actions=self.last_action, inference_ms=self.last_inference_ms,
                  elapsed_s=round(time.monotonic() - self.started_at, 3),
                  horizon=SUITE_HORIZONS[self.selection.suite])
        if mode == "PAUSED":
            self.emit("state", state="PAUSED")
        if self.step == 0:
            self.emit("state", state="READY")
        if terminal:
            return
        with self.condition:
            while self.gate.mode in {"READY", "PAUSED", "PAUSING"} and not self.gate.stopped:
                # A pause may arrive after checkpoint() but before this wait begins.
                if self.gate.mode == "PAUSING":
                    self.gate.checkpoint()
                    self.emit("state", state="PAUSED")
                self.condition.wait(timeout=0.5)
            if self.gate.stopped:
                raise StopRequested()

    def execute(self) -> None:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.configs import LiberoEnv as LiberoConfig
        from lerobot.envs.factory import make_env, make_env_pre_post_processors
        from lerobot.policies.factory import make_policy, make_pre_post_processors
        from lerobot.scripts.lerobot_eval import rollout
        from lerobot.utils.constants import ACTION
        from lerobot.utils.random_utils import set_seed

        set_seed(self.selection.seed)
        cfg = LiberoConfig(task=self.selection.suite, task_ids=[self.selection.task_id],
                           observation_height=256, observation_width=256,
                           control_mode="relative", init_states=True, hard_reset=True,
                           max_parallel_tasks=1)
        policy_cfg = PreTrainedConfig.from_pretrained(self.model_path)
        policy_cfg.pretrained_path = self.model_path
        policy_cfg.device = "cuda"
        policy_cfg.compile_model = False
        policy_cfg.n_action_steps = 1
        policy_cfg.num_steps = 10
        self.emit("state", state="LOADING")
        self.env = make_env(cfg, n_envs=1, use_async_envs=False)[self.selection.suite][
            self.selection.task_id]
        if self.selection.init_state >= len(self.env.envs[0]._init_states):
            raise ValueError(
                f"init_state {self.selection.init_state} out of range "
                f"[0, {len(self.env.envs[0]._init_states) - 1}]"
            )
        self.env.envs[0].init_state_id = self.selection.init_state
        self.env.envs[0]._reset_stride = 0
        policy = make_policy(cfg=policy_cfg, env_cfg=cfg)
        policy.eval()
        pre, post = make_pre_post_processors(
            policy_cfg=policy_cfg, pretrained_path=self.model_path,
            preprocessor_overrides={"device_processor": {"device": "cuda"},
                                    "rename_observations_processor": {"rename_map": {}}},
        )
        env_pre, env_post = make_env_pre_post_processors(env_cfg=cfg, policy_cfg=policy_cfg)
        original_select = policy.select_action

        def select_tap(observation: dict) -> Any:
            start = time.monotonic()
            result = original_select(observation)
            self.last_action = {"normalized": self.vector(result)}
            self.last_inference_ms = round((time.monotonic() - start) * 1000, 3)
            return result

        def post_tap(action: Any) -> Any:
            result = post(action)
            self.last_action["processed"] = self.vector(result)
            return result

        def env_post_tap(transition: dict) -> dict:
            result = env_post(transition)
            self.last_action["environment"] = self.vector(result[ACTION])
            self.step += 1
            return result

        policy.select_action = select_tap
        self.emit("task", language=self.env.envs[0].task_description,
                  suite=self.selection.suite, task_id=self.selection.task_id,
                  seed=self.selection.seed, init_state=self.selection.init_state)
        result = rollout(
            env=self.env, policy=policy,
            env_preprocessor=env_pre, env_postprocessor=env_post_tap,
            preprocessor=pre, postprocessor=post_tap,
            seeds=[self.selection.seed], render_callback=self.render,
        )
        success = bool(result["success"].any().item())
        self.completed = True
        self.success = success
        self.finish_reason = "success" if success else "horizon"
        self.emit("state", state="FINISHED", success=success,
                  reason=self.finish_reason)

    def run(self) -> int:
        threading.Thread(target=self.listen, daemon=True).start()
        try:
            # Vendor import and simulator setup print progress to stdout. Keep stdout JSON-only for IPC.
            with redirect_stdout(sys.stderr):
                self.execute()
            return 0
        except StopRequested:
            self.finish_reason = "user_stop"
            self.emit("state", state="CLOSED", reason="user_stop")
            return 0
        except Exception as exc:
            self.finish_reason = "error"
            traceback.print_exc(file=sys.stderr)
            self.emit("state", state="ERROR", error=f"{type(exc).__name__}: {exc}")
            return 1
        finally:
            if self.env is not None:
                try:
                    self.env.close()
                except Exception as exc:
                    print(f"Environment close failed: {exc}", file=sys.stderr)
            try:
                videos = self.save_videos()
                video_error = None
            except Exception as exc:
                videos = []
                video_error = f"{type(exc).__name__}: {exc}"
                print(f"Video export failed: {video_error}", file=sys.stderr)
            try:
                with (self.model_path / "model.safetensors").open("rb") as model_file:
                    model_sha256 = hashlib.file_digest(model_file, "sha256").hexdigest()
            except OSError as exc:
                model_sha256 = None
                print(f"Checkpoint hash failed: {exc}", file=sys.stderr)
            (self.run_dir / "summary.json").write_text(json.dumps({
                "mode": "interactive", "formal_result": False,
                "session_id": self.session_id, "generation": self.generation,
                "selection": self.selection.__dict__, "steps": self.step,
                "completed": self.completed, "success": self.success,
                "reason": self.finish_reason, "model_sha256": model_sha256,
                "videos": videos, "video_fps": 10, "video_error": video_error,
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.trace.close()


def main() -> int:
    config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    return LiveWorker(config).run()


if __name__ == "__main__":
    raise SystemExit(main())
