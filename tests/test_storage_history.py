import json

from tools.record_storage_gate import GIB
from tools.validate_storage_history import validate_storage_history


def event(root, model, seed, phase, *, enforced=False, valid=True, free_gib=200.0):
    return {
        "valid": valid,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "phase": phase,
        "project_root": str(root.resolve()),
        "model_label": model,
        "seed": seed,
        "minimum_free_gib": 100.0,
        "minimum_free_bytes": 100 * GIB,
        "enforced": enforced,
        "enough_free": free_gib >= 100,
        "total_bytes": 500 * GIB,
        "used_bytes": int((500 - free_gib) * GIB),
        "free_bytes": int(free_gib * GIB),
        "free_gib": free_gib,
        "mount": {"target": "/", "source": "/dev/sda4", "fstype": "ext4", "options": "rw,relatime"},
        "errors": [],
    }


def complete_history(root, path):
    rows = []
    for model in ("paper", "local"):
        for seed in (1000, 2000, 3000):
            rows.append(event(root, model, seed, "pre_seed", enforced=True))
            rows.append(event(root, model, seed, "post_seed"))
        rows.append(event(root, model, None, "post_summary"))
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_storage_history_requires_both_models_three_ordered_seed_pairs(tmp_path):
    history = tmp_path / "checks.jsonl"
    complete_history(tmp_path, history)
    result = validate_storage_history(tmp_path, history, ["paper", "local"])
    assert result["valid"] is True, result["errors"]
    assert len(result["validated_pairs"]) == 6
    assert result["post_summary_event_indices"] == {"paper": 6, "local": 13}
    assert result["minimum_valid_pre_seed_free_gib"] == 200.0


def test_storage_history_rejects_missing_pre_gate_and_non_rw_mount(tmp_path):
    history = tmp_path / "checks.jsonl"
    complete_history(tmp_path, history)
    rows = [json.loads(line) for line in history.read_text().splitlines()]
    rows.pop(0)
    rows[0]["mount"]["options"] = "ro,relatime"
    history.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = validate_storage_history(tmp_path, history, ["paper", "local"])
    assert result["valid"] is False
    assert any("paper seed 1000" in error for error in result["errors"])
    assert any("not ext4 rw" in error for error in result["errors"])


def test_diagnostic_label_is_not_a_formal_pair_but_changes_bound_history(tmp_path):
    history = tmp_path / "checks.jsonl"
    complete_history(tmp_path, history)
    before = validate_storage_history(tmp_path, history, ["paper", "local"])
    diagnostic = event(
        tmp_path,
        "paper_diagnostic_seed4000",
        4000,
        "pre_seed",
        enforced=True,
    )
    with history.open("a") as handle:
        handle.write(json.dumps(diagnostic) + "\n")
    after = validate_storage_history(tmp_path, history, ["paper", "local"])
    assert after["valid"] is True, after["errors"]
    assert len(after["validated_pairs"]) == 6
    assert after["relevant_event_count"] == before["relevant_event_count"]
    assert after["event_count"] == before["event_count"] + 1
    assert after["history_sha256"] != before["history_sha256"]
