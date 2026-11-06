import json

import pytest
from huggingface_hub.hf_api import RepoFile

from tools.record_hub_tree import record_tree


def test_downloaded_tree_records_remote_lfs_hash_and_refuses_partial_assets(tmp_path):
    remote_hash = "a" * 64

    class Hub:
        def list_repo_tree(self, repo_id, **kwargs):
            assert repo_id == "example/model" and kwargs["revision"] == "b" * 40
            return [RepoFile(path="model.safetensors", size=4, oid="blob-id",
                             lfs={"size": 4, "oid": remote_hash, "pointerSize": 100})]

    kwargs = dict(repo_id="example/model", repo_type="model", revision="b" * 40,
                  root=tmp_path, api=Hub())
    with pytest.raises(ValueError, match="missing or wrong-sized"):
        record_tree(**kwargs)
    (tmp_path / "model.safetensors").write_bytes(b"data")
    path = record_tree(**kwargs)
    row = json.loads(path.read_text())["files"]["model.safetensors"]
    assert row == {"size": 4, "blob_id": "blob-id", "lfs_sha256": remote_hash}
