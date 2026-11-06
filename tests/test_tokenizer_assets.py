from pathlib import Path

from tools.validate_tokenizer_assets import (
    EXPECTED_FILE_MANIFEST_SHA256,
    EXPECTED_FILE_SHA256,
    EXPECTED_REVISION,
    TOKENIZER_REPOS,
    validate_tokenizer_assets,
)


def test_fixed_offline_tokenizers_are_identical_for_all_libero_tasks():
    root = Path(__file__).resolve().parents[1]
    result = validate_tokenizer_assets(root)
    assert result["valid"] is True, result["errors"]
    assert result["expected_revision"] == EXPECTED_REVISION
    assert result["expected_file_sha256"] == EXPECTED_FILE_SHA256
    assert result["tokenizer_file_manifest_sha256"] == EXPECTED_FILE_MANIFEST_SHA256
    for snapshot in result["snapshots"].values():
        assert snapshot["file_sha256"] == EXPECTED_FILE_SHA256
    profiles = result["tokenization_profiles"]
    assert profiles[TOKENIZER_REPOS[0]] == profiles[TOKENIZER_REPOS[1]]
    assert profiles[TOKENIZER_REPOS[0]]["maximum_raw_token_length"] <= 48
