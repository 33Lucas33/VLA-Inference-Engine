from pathlib import Path

from tools.validate_dataset_coverage import validate_dataset_coverage, write_tables


def test_fixed_dataset_covers_all_official_libero_tasks_and_frames():
    root = Path(__file__).resolve().parents[1]
    result = validate_dataset_coverage(root / "artifacts/datasets/lerobot-libero")
    assert result["valid"] is True, result["errors"]
    assert result["totals"] == {
        "episodes": 1693,
        "frames": 273465,
        "tasks": 40,
        "mixed_task_episodes": 0,
    }
    assert len(result["tasks"]) == 40
    assert result["minimum_episodes_per_task"] == 29
    assert result["maximum_episodes_per_task"] == 50


def test_missing_dataset_emits_invalid_but_readable_empty_tables(tmp_path):
    result = validate_dataset_coverage(tmp_path / "missing")
    assert result["valid"] is False
    assert result["errors"]
    csv_path = tmp_path / "coverage.csv"
    md_path = tmp_path / "coverage.md"
    write_tables(result, csv_path, md_path)
    assert csv_path.read_text().startswith("suite,suite_label,task_id")
    assert "验证：失败" in md_path.read_text()
