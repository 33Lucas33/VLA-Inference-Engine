import json

from tools.record_storage_gate import GIB, append_snapshot, inspect_storage


def test_storage_snapshot_records_mount_and_appends_jsonl(tmp_path):
    history = tmp_path / "checks.jsonl"
    payload = inspect_storage(
        tmp_path,
        phase="pre_seed",
        model_label="paper",
        seed=1000,
        minimum_free_gib=0,
        enforced=True,
    )
    assert payload["valid"] is True, payload["errors"]
    assert payload["enough_free"] is True
    assert payload["free_bytes"] > 0
    assert payload["mount"]["fstype"]
    append_snapshot(history, payload)
    append_snapshot(history, payload)
    rows = [json.loads(line) for line in history.read_text().splitlines()]
    assert rows == [payload, payload]


def test_storage_snapshot_rejects_enforced_threshold_above_volume_size(tmp_path):
    initial = inspect_storage(
        tmp_path,
        phase="post_seed",
        model_label="local",
        seed=3000,
        minimum_free_gib=0,
        enforced=False,
    )
    payload = inspect_storage(
        tmp_path,
        phase="pre_seed",
        model_label="local",
        seed=3000,
        minimum_free_gib=initial["total_bytes"] / GIB + 1,
        enforced=True,
    )
    assert payload["valid"] is False
    assert payload["enough_free"] is False
    assert any("below the enforced" in error for error in payload["errors"])
