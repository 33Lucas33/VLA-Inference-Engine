"""Read task names without constructing a simulator or loading model weights."""

from __future__ import annotations

from pathlib import Path

from vla_inference_engine.benchmark.protocol import SUITE_HORIZONS, TASK_IDS


def tasks() -> dict[str, list[dict]]:
    from libero.libero import benchmark

    available = benchmark.get_benchmark_dict()
    result = {}
    for suite in SUITE_HORIZONS:
        selected = available[suite]()
        result[suite] = [
            {"id": task_id, "name": selected.get_task(task_id).name,
             "language": selected.get_task(task_id).language}
            for task_id in TASK_IDS
        ]
    return result


def models(root: Path) -> list[dict]:
    entries = []
    for label in ("smolvla_libero_paper", "smolvla_libero_100k"):
        path = root / "artifacts/models" / label
        entries.append({"label": label, "available": (path / "model.safetensors").is_file()})
    return entries
