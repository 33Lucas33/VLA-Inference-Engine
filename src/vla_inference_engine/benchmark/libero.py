from __future__ import annotations

import json
from pathlib import Path

from .protocol import (
    ACTION_SHAPE,
    EPISODES_PER_TASK,
    INFERENCE_FLAGS,
    INPUT_SHAPES,
    SUITE_HORIZONS,
    SUITE_ORDER,
    TASK_IDS,
)

SUITES = SUITE_HORIZONS
EXPECTED_INPUTS = {key: list(shape) for key, shape in INPUT_SHAPES.items()}
EXPECTED_ACTION = list(ACTION_SHAPE)


def inspect_smolvla_checkpoint(model_path: Path) -> dict:
    """Validate metadata only; no model load or GPU allocation."""
    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    inputs = {key: value.get("shape") for key, value in config.get("input_features", {}).items()}
    output = config.get("output_features", {}).get("action", {}).get("shape")
    errors = [f"{key}: expected {shape}, got {inputs.get(key)}"
              for key, shape in EXPECTED_INPUTS.items() if inputs.get(key) != shape]
    if output != EXPECTED_ACTION:
        errors.append(f"action: expected {EXPECTED_ACTION}, got {output}")
    for name in ("model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
        candidate = model_path / name
        if not candidate.is_file() or candidate.stat().st_size == 0:
            errors.append(f"missing or empty: {name}")
    return {
        "valid": not errors, "errors": errors, "model_path": str(model_path.resolve()),
        "inputs": inputs, "action_shape": output, "chunk_size": config.get("chunk_size"),
        "n_action_steps": config.get("n_action_steps"), "num_steps": config.get("num_steps"),
        "tokenizer_max_length": config.get("tokenizer_max_length"),
    }


def build_eval_command(*, lerobot_eval: str, model_path: Path, suite: str, task_id: int,
                       episodes: int, seed: int, output_dir: Path) -> list[str]:
    """Build the fixed LeRobot LIBERO smoke command without launching a process."""
    if suite not in SUITES or task_id not in TASK_IDS or episodes not in range(1, 4):
        raise ValueError("suite, task_id or episode count is outside the smoke protocol")
    return _build_command(lerobot_eval, model_path, [suite], task_id, episodes, seed, output_dir)


def build_formal_command(*, model_path: Path, seed: int, output_dir: Path,
                         lerobot_eval: str = "lerobot-eval") -> list[str]:
    """Build one full-suite run; the formal runner owns the three-seed matrix."""
    return _build_command(lerobot_eval, model_path, SUITE_ORDER, None,
                          EPISODES_PER_TASK, seed, output_dir)


def _build_command(lerobot_eval, model_path, suites, task_id, episodes, seed, output_dir):
    return [
        lerobot_eval, f"--policy.path={model_path.expanduser().resolve()}", *INFERENCE_FLAGS,
        f"--env.task={','.join(suites)}",
        *([f"--env.task_ids=[{task_id}]"] if task_id is not None else []),
        f"--eval.n_episodes={episodes}", f"--seed={seed}",
        f"--output_dir={output_dir.expanduser().resolve()}",
    ]
