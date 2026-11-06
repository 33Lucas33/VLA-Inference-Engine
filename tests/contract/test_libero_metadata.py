import json

from vla_inference_engine.benchmark import build_eval_command, inspect_smolvla_checkpoint


def test_smolvla_metadata_and_fixed_smoke_command(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    config = {
        "input_features": {
            "observation.images.image": {"shape": [3, 256, 256]},
            "observation.images.image2": {"shape": [3, 256, 256]},
            "observation.state": {"shape": [8]},
        },
        "output_features": {"action": {"shape": [7]}},
        "chunk_size": 50, "num_steps": 10, "n_action_steps": 1,
    }
    (model / "config.json").write_text(json.dumps(config))
    for name in ("model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
        (model / name).write_bytes(b"nonempty")
    assert inspect_smolvla_checkpoint(model)["valid"] is True
    command = build_eval_command(lerobot_eval="lerobot-eval", model_path=model,
                                 suite="libero_spatial", task_id=2, episodes=1,
                                 seed=1000, output_dir=tmp_path / "run")
    assert "--env.hard_reset=true" in command
    assert "--env.control_mode=relative" in command
    assert "--policy.compile_model=false" in command
    assert "--env.task_ids=[2]" in command
    assert "--eval.n_episodes=1" in command
