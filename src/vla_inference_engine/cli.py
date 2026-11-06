from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from vla_inference_engine.adapters import MockEnvironment, MockPolicy
from vla_inference_engine.benchmark import build_eval_command, inspect_smolvla_checkpoint
from vla_inference_engine.benchmark.runtime import project_root
from vla_inference_engine.benchmark.smoke import run_smoke
from vla_inference_engine.contracts import ActionSchema
from vla_inference_engine.control import ActionQueue, ControlLoop, FakeClock
from vla_inference_engine.telemetry import TraceRecorder


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def _git_commit(root: Path) -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                            capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def run_mock(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[2]
    out = Path(args.out).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"run directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    config = {"mode": "mock", "ticks": args.ticks, "seed": args.seed,
              "control_dt_ns": 50_000_000, "action_dim": 2,
              "chunk_horizon": 4, "drop_every": args.drop_every}
    _write_json(out / "resolved_config.json", config)
    config_hash = hashlib.sha256((out / "resolved_config.json").read_bytes()).hexdigest()
    clock = FakeClock()
    environment = MockEnvironment(clock)
    policy = MockPolicy(clock, drop_every=args.drop_every)
    queue = ActionQueue(ActionSchema("mock-v1", 2))
    with TraceRecorder(out / "trace.jsonl") as trace:
        loop = ControlLoop(environment, policy, queue, clock, trace)
        summary = loop.run(ticks=args.ticks, seed=args.seed)
    summary_dict = {"ticks": summary.ticks, "dispatched": summary.dispatched,
                    "holds": summary.holds, "rejected_chunks": summary.rejected_chunks,
                    "generation_id": summary.generation_id,
                    "minimum_queue_coverage_ms": summary.minimum_queue_coverage_ms,
                    "final_state": environment.state.tolist(),
                    "trace": "trace.jsonl"}
    _write_json(out / "summary.json", summary_dict)
    _write_json(out / "manifest.json", {
        "run_id": out.name, "started_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(root), "python_version": sys.version.split()[0],
        "platform": platform.platform(), "seed": args.seed,
        "resolved_config_sha256": config_hash, "mode": "mock", "gpu_required": False,
    })
    print(json.dumps({"run_dir": str(out), **summary_dict}, ensure_ascii=False))
    return 0


def inspect_smolvla(args: argparse.Namespace) -> int:
    payload = inspect_smolvla_checkpoint(Path(args.model).expanduser())
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["valid"] else 2


def libero_command(args: argparse.Namespace) -> int:
    model = Path(args.model).expanduser()
    spec = inspect_smolvla_checkpoint(model)
    if not spec["valid"]:
        print(json.dumps(spec, indent=2, ensure_ascii=False), file=sys.stderr)
        return 2
    command = build_eval_command(
        lerobot_eval=args.lerobot_eval, model_path=model,
        suite=args.suite, task_id=args.task_id, episodes=args.episodes,
        seed=args.seed, output_dir=Path(args.out).expanduser(),
    )
    print(shlex.join(command))
    return 0


def eval_preflight(args: argparse.Namespace) -> int:
    root = project_root()
    env = dict(os.environ)
    if args.model:
        env["MODEL_PATH"] = str(Path(args.model).expanduser().absolute())
    subprocess.run(["bash", str(root / "scripts/preflight_eval.sh")], cwd=root,
                   env=env, check=True)
    return 0


def eval_smoke(args: argparse.Namespace) -> int:
    result = run_smoke(model=Path(args.model), out=Path(args.out), suite=args.suite,
                       task_id=args.task_id, episodes=args.episodes, seed=args.seed)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def eval_formal(args: argparse.Namespace) -> int:
    root = project_root()
    env = dict(os.environ)
    env["MODEL_PATH"] = str(Path(args.model).expanduser().resolve())
    env["EVAL_LABEL"] = args.label
    subprocess.run(["bash", str(root / "scripts/eval_all_suites.sh")], cwd=root,
                   env=env, check=True)
    return 0


def live(args: argparse.Namespace) -> int:
    root = project_root()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise ValueError("The interactive viewer currently binds to localhost only")
    # Reuse the fixed Conda/EGL environment without loading CUDA in the CLI process.
    try:
        subprocess.run([
            "bash", "-c",
            'source "$1"; exec "$PYTHON_BIN" -m uvicorn '
            'vla_inference_engine.live.server:app --host "$2" --port "$3"',
            "vla-live", str(root / "scripts/common.sh"), args.host, str(args.port),
        ], cwd=root, check=True)
    except KeyboardInterrupt:
        return 130
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VLA inference engine development CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    mock = sub.add_parser("mock", help="Run the CPU-only P0 control-loop demo")
    mock.add_argument("--ticks", type=int, default=100)
    mock.add_argument("--seed", type=int, default=0)
    mock.add_argument("--drop-every", type=int, default=0,
                      help="Drop every Nth mock request; 1 makes the queue hold each tick")
    mock.add_argument("--out", required=True)
    mock.set_defaults(func=run_mock)

    inspect = sub.add_parser("inspect-smolvla", help="Inspect checkpoint metadata without loading weights")
    inspect.add_argument("--model", required=True)
    inspect.set_defaults(func=inspect_smolvla)

    command = sub.add_parser("libero-command", help="Print a fixed LeRobot smoke command; do not execute")
    command.add_argument("--model", required=True)
    command.add_argument("--suite", default="libero_spatial")
    command.add_argument("--task-id", type=int, default=0)
    command.add_argument("--episodes", type=int, default=1)
    command.add_argument("--seed", type=int, default=1000)
    command.add_argument("--out", required=True)
    command.add_argument("--lerobot-eval", default="lerobot-eval")
    command.set_defaults(func=libero_command)

    preflight = sub.add_parser("eval-preflight", help="Validate fixed LIBERO assets and protocol without rollout")
    preflight.add_argument("--model")
    preflight.set_defaults(func=eval_preflight)

    smoke = sub.add_parser("eval-smoke", help="Run and validate one LIBERO task; not a formal score")
    smoke.add_argument("--model", required=True)
    smoke.add_argument("--suite", default="libero_spatial")
    smoke.add_argument("--task-id", type=int, default=0)
    smoke.add_argument("--episodes", type=int, default=1)
    smoke.add_argument("--seed", type=int, default=1000)
    smoke.add_argument("--out", required=True)
    smoke.set_defaults(func=eval_smoke)

    formal = sub.add_parser("eval-formal", help="Run the fixed 4-suite, 3-seed LIBERO protocol")
    formal.add_argument("--model", required=True)
    formal.add_argument("--label", required=True)
    formal.set_defaults(func=eval_formal)

    viewer = sub.add_parser("live", help="Open the local interactive LIBERO viewer")
    viewer.add_argument("--host", default="127.0.0.1")
    viewer.add_argument("--port", type=int, default=8765)
    viewer.set_defaults(func=live)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
