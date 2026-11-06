"""One-task LIBERO smoke run and validation; never a formal score."""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

from .libero import build_eval_command, inspect_smolvla_checkpoint
from .runtime import evaluation_environment, project_root


def validate_smoke_result(payload: dict[str, Any], run_dir: Path, *, suite: str,
                          task_id: int, episodes: int) -> dict[str, Any]:
    errors: list[str] = []
    rows = payload.get("per_task")
    if not isinstance(rows, list) or len(rows) != 1:
        errors.append("per_task must contain exactly one task")
        rows = []
    videos: list[str] = []
    successes: list[bool] = []
    if rows:
        row = rows[0]
        if not isinstance(row, dict):
            errors.append("task record must be an object")
            row = {}
        if row.get("task_group") != suite or row.get("task_id") != task_id:
            errors.append("task identity differs from the requested smoke task")
        metrics = row.get("metrics", {})
        if not isinstance(metrics, dict):
            errors.append("task metrics must be an object")
            metrics = {}
        successes = metrics.get("successes", [])
        if not isinstance(successes, list) or len(successes) != episodes or any(
            type(value) is not bool for value in successes
        ):
            errors.append("successes must contain the requested number of booleans")
            successes = []
        for name in ("sum_rewards", "max_rewards"):
            values = metrics.get(name)
            if not isinstance(values, list) or len(values) != episodes:
                errors.append(f"{name} must contain {episodes} values")
        videos = metrics.get("video_paths", [])
        if not isinstance(videos, list) or len(videos) != episodes:
            errors.append(f"video_paths must contain {episodes} paths")
            videos = []
    canonical = []
    for value in videos:
        if not isinstance(value, str) or not value:
            errors.append("video path is empty or invalid")
            continue
        path = Path(value)
        path = (path if path.is_absolute() else run_dir / path).resolve()
        canonical.append(str(path))
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"video is missing or empty: {path}")
    if len(canonical) != len(set(canonical)):
        errors.append("video paths are duplicated")
    per_group = payload.get("per_group", {})
    group = per_group.get(suite, {}) if isinstance(per_group, dict) else {}
    if not isinstance(group, dict) or group.get("n_episodes") != episodes:
        errors.append("suite episode count differs from the request")
    overall = payload.get("overall", {})
    if not isinstance(overall, dict):
        errors.append("overall must be an object")
        overall = {}
    if overall.get("n_episodes") != episodes:
        errors.append("overall episode count differs from the request")
    overall_videos = overall.get("video_paths", [])
    if isinstance(overall_videos, list) and overall_videos:
        actual = set()
        for value in overall_videos:
            path = Path(value)
            actual.add(str((path if path.is_absolute() else run_dir / path).resolve()))
        if actual != set(canonical):
            errors.append("overall video set differs from task video set")
    return {"valid": not errors, "errors": errors, "suite": suite, "task_id": task_id,
            "episodes": episodes, "successes": successes, "videos": canonical,
            "formal_result": False}


def run_smoke(*, model: Path, out: Path, suite: str, task_id: int, episodes: int,
              seed: int) -> dict[str, Any]:
    root = project_root()
    model_label = model.name
    model = model.expanduser().resolve()
    out = out.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"smoke output is not empty: {out}")
    spec = inspect_smolvla_checkpoint(model)
    if not spec["valid"]:
        raise ValueError(f"checkpoint metadata failed: {spec['errors']}")
    env = evaluation_environment(root, model=model, label=model_label)
    command = build_eval_command(lerobot_eval="lerobot-eval", model_path=model,
                                 suite=suite, task_id=task_id, episodes=episodes,
                                 seed=seed, output_dir=out)
    subprocess.run(["bash", str(root / "scripts/gpu_gate.sh")], cwd=root, env=env, check=True)
    subprocess.run(["bash", str(root / "scripts/preflight_eval.sh")], cwd=root, env=env, check=True)
    subprocess.run([
        env["PYTHON_BIN"], str(root / "tools/record_storage_gate.py"),
        f"--root={root}", f"--history={root / 'outputs/storage/evaluation_checks.jsonl'}",
        "--phase=pre_seed", "--model-label=smoke", f"--seed={seed}",
        "--minimum-free-gib=1", "--enforce",
    ], cwd=root, env=env, check=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "command.sh").write_text(shlex.join(command) + "\n", encoding="utf-8")
    digest = hashlib.sha256()
    with (model / "model.safetensors").open("rb") as weights:
        for chunk in iter(lambda: weights.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    model_hash = digest.hexdigest()
    (out / "manifest.json").write_text(json.dumps({
        "mode": "smoke", "formal_result": False, "model": str(model),
        "model_safetensors_sha256": model_hash, "suite": suite,
        "task_id": task_id, "episodes": episodes, "seed": seed,
        "command": command,
        "provenance": str(root / "outputs/provenance/assets.json"),
    }, indent=2) + "\n", encoding="utf-8")
    with (out / "eval.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=root, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    payload = json.loads((out / "eval_info.json").read_text(encoding="utf-8"))
    result = validate_smoke_result(payload, out, suite=suite, task_id=task_id,
                                   episodes=episodes)
    (out / "smoke_validation.json").write_text(json.dumps(result, indent=2) + "\n",
                                                 encoding="utf-8")
    if not result["valid"]:
        raise ValueError(f"smoke result failed validation: {result['errors']}")
    return {"run_dir": str(out), **result}
