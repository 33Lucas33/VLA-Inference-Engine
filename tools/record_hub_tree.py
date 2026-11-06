"""Save pinned Hub metadata required by the project's asset preflight.

hf download creates download-cache metadata, not this project's trees/*.json.
Remote sizes and LFS hashes must come from the pinned Hub tree, not local files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile


def record_tree(*, repo_id: str, repo_type: str, revision: str, root: Path,
                api=None) -> Path:
    api = api or HfApi()
    files = {}
    for item in api.list_repo_tree(repo_id, repo_type=repo_type, revision=revision, recursive=True):
        if not isinstance(item, RepoFile):
            continue
        local = root / item.path
        if not local.is_file() or local.stat().st_size != item.size:
            raise ValueError(f"missing or wrong-sized downloaded asset: {local}")
        row = {"size": item.size, "blob_id": item.blob_id}
        if item.lfs:
            row["lfs_sha256"] = item.lfs.sha256
        files[item.path] = row
    if not files:
        raise ValueError("pinned Hub tree contains no files")
    target = root / ".cache/huggingface/trees" / f"{revision}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps({"format_version": 1, "files": files},
                                    indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--repo-type", choices=("model", "dataset"), default="model")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(record_tree(repo_id=args.repo, repo_type=args.repo_type,
                      revision=args.revision, root=args.root))


if __name__ == "__main__":
    main()
