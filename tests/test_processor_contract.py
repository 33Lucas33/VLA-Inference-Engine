import json
import shutil
from pathlib import Path

from tools.validate_processor_contract import validate_processor_contract


def test_paper_checkpoint_processor_contract_matches_dataset_and_env():
    root = Path(__file__).resolve().parents[1]
    result = validate_processor_contract(root)
    assert result["valid"] is True, result["errors"]
    assert all(
        row["matches_dataset_within_tolerance"]
        for row in result["normalization_stat_comparisons"]
    )
    assert all(
        row["matches_dataset_float32_exactly"]
        for row in result["normalization_stat_comparisons"]
        if row["tensor"] != "observation.state.std"
    )
    state_std = next(
        row for row in result["normalization_stat_comparisons"]
        if row["tensor"] == "observation.state.std"
    )
    assert state_std["matches_dataset_float32_exactly"] is False
    assert state_std["max_abs_difference"] < 3e-7
    assert result["behavioral_observation_contract"]["state_shape"] == [1, 8]
    assert result["behavioral_environment_config_contract"] == {
        "control_mode": "relative",
        "action_feature_type": "ACTION",
        "action_feature_shape": [7],
        "camera_feature_shapes": {
            "agentview": [256, 256, 3],
            "eye_in_hand": [256, 256, 3],
        },
        "hard_reset": True,
        "init_states": True,
        "env_preprocessor_steps": ["LiberoProcessorStep"],
        "env_postprocessor_steps": [],
        "action_feature_enum_matches": True,
    }
    assert result["action_contract"]["controller"] == "relative"
    assert result["behavioral_action_contract"] == {
        "input_shape": [1, 7],
        "output_shape": [1, 7],
        "output_device": "cpu",
        "matches_saved_mean_std_inverse": True,
        "maximum_abs_difference": 0.0,
    }


def test_processor_contract_accepts_an_explicit_checkpoint_directory():
    root = Path(__file__).resolve().parents[1]
    model = root / "artifacts/models/smolvla_libero_paper"
    result = validate_processor_contract(root, model)
    assert result["valid"] is True, result["errors"]
    assert result["model_root"] == str(model.resolve())


def test_processor_contract_accepts_the_preregistered_local_processor_profile(tmp_path):
    root = Path(__file__).resolve().parents[1]
    paper = root / "artifacts/models/smolvla_libero_paper"
    model = tmp_path / "local_pretrained_model"
    model.mkdir()
    for filename in (
        "config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
    ):
        shutil.copy2(paper / filename, model / filename)
    shutil.copy2(
        paper / "policy_postprocessor_step_1_unnormalizer_processor.safetensors",
        model / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    )

    config = json.loads((model / "config.json").read_text())
    config["vlm_model_name"] = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    (model / "config.json").write_text(json.dumps(config))

    pre = json.loads((model / "policy_preprocessor.json").read_text())
    pre["steps"][3]["config"]["tokenizer_name"] = config["vlm_model_name"]
    (model / "policy_preprocessor.json").write_text(json.dumps(pre))

    post = json.loads((model / "policy_postprocessor.json").read_text())
    post["steps"] = [post["steps"][1], post["steps"][0]]
    post["steps"][0]["state_file"] = (
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    )
    (model / "policy_postprocessor.json").write_text(json.dumps(post))

    result = validate_processor_contract(root, model)
    assert result["valid"] is True, result["errors"]
    assert result["policy_processing_profile"] == {
        "vlm_model_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "language_padding": "longest",
        "tokenizer_max_length": 48,
        "postprocessor_step_order": ["unnormalizer_processor", "device_processor"],
        "preprocessor_state_file": "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "postprocessor_state_file": "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    }
