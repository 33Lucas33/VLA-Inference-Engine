from pathlib import Path

from tools.validate_eval_command import (
    EXPECTED_SUITE_HORIZONS,
    parse_lerobot_command,
    validate_command,
)
from vla_inference_engine.benchmark import build_eval_command
from vla_inference_engine.benchmark.libero import build_formal_command


def formal_command(root: Path, seed: int = 1000) -> list[str]:
    return build_formal_command(
        model_path=root / "artifacts/models/smolvla_libero_paper",
        seed=seed, output_dir=root / f"results/raw/test/seed_{seed}",
    )


def test_formal_eval_command_parses_with_installed_lerobot():
    root = Path(__file__).resolve().parents[1]
    result = validate_command(formal_command(root), 1000)
    assert result["valid"], result["errors"]
    assert result["parsed"]["expected_episodes"] == 400
    assert result["parsed"]["episode_length_override"] is None
    assert result["parsed"]["resolved_suite_horizons"] == EXPECTED_SUITE_HORIZONS
    assert result["parsed"]["policy_input_features"] == {
        "observation.images.image": [3, 256, 256],
        "observation.images.image2": [3, 256, 256],
        "observation.state": [8],
    }
    assert result["model_artifacts"]["files"]["model.safetensors"]["bytes"] > 0
    assert len(result["model_artifacts"]["files"]["model.safetensors"]["sha256"]) == 64
    assert len(result["model_artifacts"]["manifest_sha256"]) == 64


def test_formal_eval_command_rejects_protocol_drift():
    root = Path(__file__).resolve().parents[1]
    command = formal_command(root)
    command[command.index("--eval.n_episodes=10")] = "--eval.n_episodes=9"
    result = validate_command(command, 1000)
    assert result["valid"] is False
    assert any("eval.n_episodes" in error for error in result["errors"])

    horizon_command = formal_command(root)
    horizon_command.insert(-2, "--env.episode_length=400")
    horizon_result = validate_command(horizon_command, 1000)
    assert horizon_result["valid"] is False
    assert any("env.episode_length" in error for error in horizon_result["errors"])


def test_smoke_command_is_accepted_by_fixed_lerobot_parser():
    root = Path(__file__).resolve().parents[1]
    command = build_eval_command(
        lerobot_eval="lerobot-eval",
        model_path=root / "artifacts/models/smolvla_libero_paper",
        suite="libero_spatial", task_id=0, episodes=1, seed=1000,
        output_dir=root / "runs/smoke-parser-check",
    )
    cfg = parse_lerobot_command(command)
    assert cfg.env.task == "libero_spatial"
    assert cfg.env.task_ids == [0]
    assert cfg.eval.n_episodes == 1
    assert cfg.env.control_mode == "relative"
