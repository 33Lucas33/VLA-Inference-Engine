import json
from pathlib import Path

import pytest

from tools import validate_formal_summary as validator


@pytest.fixture(autouse=True)
def stub_fresh_command_validation(monkeypatch):
    """Keep synthetic fixtures independent of the real LeRobot/model parser."""
    monkeypatch.setattr(
        validator,
        "validate_command",
        lambda command, seed: {"valid": True, "command": command},
    )


def make_rebuilt(raw_root, model_label="paper", seed_ids=validator.EXPECTED_SEED_IDS):
    project_root = raw_root.parents[1]
    provenance = project_root / "outputs/provenance/assets.json"
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_text(
        json.dumps(
            {
                "valid": True,
                "project_root": str(project_root),
                "system": {
                    "python": "3.12.13 | test build",
                    "nvidia_smi": "NVIDIA GeForce RTX 3090, 580.173.02, 24576, 8.6",
                    "packages": {
                        "torch": "2.11.0+cu128",
                        "hf_libero": "0.1.4",
                        "mujoco": "3.8.1",
                        "streamlit": "1.61.1",
                    },
                },
                "lerobot": {
                    "observed_commit": "8ae2354592ec4480c38134fb75350b8e7992e0e9"
                },
            }
        )
    )
    runs = []
    for seed in seed_ids:
        path = raw_root / f"seed_{seed}/eval_info.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"seed": seed}))
        protocol_validation = {
            "valid": True,
            "observed_episodes": 400,
            "observed_nonempty_videos": 400,
            "command_preflight_valid": True,
            "errors": [],
        }
        command = ["lerobot-eval", f"--seed={seed}"]
        (path.parent / "command.sh").write_text(f"lerobot-eval --seed={seed}\n")
        (path.parent / "command_preflight.json").write_text(
            json.dumps({"valid": True, "command": command})
        )
        (path.parent / "protocol_validation.json").write_text(
            json.dumps(protocol_validation)
        )
        (path.parent / "environment.txt").write_text(
            "NVIDIA GeForce RTX 3090, 580.173.02\nPython 3.12.13\n"
            "torch==2.11.0+cu128\nhf_libero==0.1.4\nmujoco==3.8.1\nstreamlit==1.61.1\n"
            "-e git+https://github.com/huggingface/lerobot.git@"
            "8ae2354592ec4480c38134fb75350b8e7992e0e9#egg=lerobot\n"
        )
        eval_log = project_root / f"outputs/logs/eval/{model_label}/seed_{seed}.log"
        eval_log.parent.mkdir(parents=True, exist_ok=True)
        eval_log.write_text("formal evaluation completed\n")
        runs.append(
            {
                "model": model_label,
                "seed": seed,
                "path": str(path.resolve()),
                "protocol_validation": protocol_validation,
            }
        )
    return {
        "runs": runs,
        "models": {model_label: {"protocol_complete": True}},
    }


def test_formal_summary_rebuilds_and_binds_raw_evidence(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw/paper"
    rebuilt = make_rebuilt(raw_root)
    summary = tmp_path / "results/paper/metrics_summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps(rebuilt))
    monkeypatch.setattr(validator, "aggregate", lambda *args, **kwargs: rebuilt)

    result = validator.validate_formal_summary(summary, raw_root, "paper")

    assert result["valid"] is True
    assert result["canonical_summary_match"] is True
    assert result["raw_run_count"] == 3
    assert result["observed_nonempty_videos"] == 1200
    assert [row["seed"] for row in result["runs"]] == [1000, 2000, 3000]


def test_reuse_rejects_a_different_requested_checkpoint(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw/paper"
    rebuilt = make_rebuilt(raw_root)
    run = rebuilt["runs"][0]
    eval_path = Path(run["path"])
    preflight_path = eval_path.parent / "command_preflight.json"
    preflight = json.loads(preflight_path.read_text())
    preflight["parsed"] = {"model_path": str(tmp_path / "old-model")}
    preflight_path.write_text(json.dumps(preflight))
    monkeypatch.setattr(validator, "validate_command", lambda *_: preflight)
    args = (eval_path, run["protocol_validation"], tmp_path, "paper", 1000, ())
    good = validator._validate_run_artifacts(*args, expected_model=tmp_path / "old-model")
    assert good["valid"], good["errors"]
    bad = validator._validate_run_artifacts(*args, expected_model=tmp_path / "new-model")
    assert not bad["valid"]
    assert any("requested model" in error for error in bad["errors"])


def test_formal_summary_refuses_stale_saved_summary(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw/paper"
    rebuilt = make_rebuilt(raw_root)
    summary = tmp_path / "results/paper/metrics_summary.json"
    summary.parent.mkdir(parents=True)
    stale = {**rebuilt, "stale": True}
    summary.write_text(json.dumps(stale))
    monkeypatch.setattr(validator, "aggregate", lambda *args, **kwargs: rebuilt)

    result = validator.validate_formal_summary(summary, raw_root, "paper")

    assert result["valid"] is False
    assert result["canonical_summary_match"] is False
    assert any("fresh raw-evidence aggregation" in error for error in result["errors"])


def test_formal_summary_requires_exact_preregistered_seed_paths(tmp_path):
    raw_root = tmp_path / "raw/paper"
    rebuilt = make_rebuilt(raw_root)
    (raw_root / "seed_3000/eval_info.json").unlink()
    wrong = raw_root / "seed_4000/eval_info.json"
    wrong.parent.mkdir(parents=True)
    wrong.write_text("{}")
    summary = tmp_path / "results/paper/metrics_summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps(rebuilt))

    result = validator.validate_formal_summary(summary, raw_root, "paper")

    assert result["valid"] is False
    assert result["raw_run_count"] == 0
    assert any("preregistered seeds" in error for error in result["errors"])


def test_formal_summary_supports_independent_seed4000_retest(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw/paper_diagnostic_seed4000"
    rebuilt = make_rebuilt(
        raw_root,
        model_label="paper_diagnostic_seed4000",
        seed_ids=(4000,),
    )
    summary = tmp_path / "results/paper_diagnostic_seed4000/metrics_summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps(rebuilt))
    monkeypatch.setattr(validator, "aggregate", lambda *args, **kwargs: rebuilt)

    result = validator.validate_formal_summary(
        summary,
        raw_root,
        "paper_diagnostic_seed4000",
        expected_seed_ids=(4000,),
    )
    assert result["valid"] is True
    assert result["expected_seeds"] == [4000]
    assert result["raw_run_count"] == 1
    assert result["observed_nonempty_videos"] == 400


def test_formal_summary_requires_complete_per_seed_provenance(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw/paper"
    rebuilt = make_rebuilt(raw_root)
    summary = tmp_path / "results/paper/metrics_summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps(rebuilt))
    monkeypatch.setattr(validator, "aggregate", lambda *args, **kwargs: rebuilt)
    (raw_root / "seed_2000/environment.txt").unlink()

    result = validator.validate_formal_summary(summary, raw_root, "paper")
    assert result["valid"] is False
    assert any("environment" in error for error in result["errors"])
    assert result["runs"][1]["run_artifacts_valid"] is False


def test_formal_summary_uses_project_provenance_profile_dynamically(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw/paper"
    rebuilt = make_rebuilt(raw_root)
    summary = tmp_path / "results/paper/metrics_summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps(rebuilt))
    monkeypatch.setattr(validator, "aggregate", lambda *args, **kwargs: rebuilt)
    provenance = tmp_path / "outputs/provenance/assets.json"
    payload = json.loads(provenance.read_text())
    payload["system"]["packages"]["torch"] = "9.9.9+future"
    provenance.write_text(json.dumps(payload))

    result = validator.validate_formal_summary(summary, raw_root, "paper")
    assert result["valid"] is False
    assert result["environment_markers"].count("torch==9.9.9+future") == 1
    assert any("torch==9.9.9+future" in error for error in result["errors"])
