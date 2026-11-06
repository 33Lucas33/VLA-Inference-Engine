#!/usr/bin/env python3
"""Download the exact tokenizer files and pin each local cache main ref."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

try:
    from tools.validate_tokenizer_assets import (
        EXPECTED_REVISION,
        REQUIRED_FILES,
        TOKENIZER_REPOS,
        validate_tokenizer_assets,
    )
except ModuleNotFoundError:
    from validate_tokenizer_assets import (
        EXPECTED_REVISION,
        REQUIRED_FILES,
        TOKENIZER_REPOS,
        validate_tokenizer_assets,
    )


def pin_tokenizer_assets(root: Path) -> dict:
    root = root.resolve()
    cache_root = root / ".cache/huggingface/hub"
    snapshots = {}
    for repo_id in TOKENIZER_REPOS:
        snapshot = Path(
            snapshot_download(
                repo_id=repo_id,
                revision=EXPECTED_REVISION,
                allow_patterns=list(REQUIRED_FILES),
                cache_dir=cache_root,
            )
        ).resolve()
        expected_snapshot = (
            cache_root
            / ("models--" + repo_id.replace("/", "--"))
            / "snapshots"
            / EXPECTED_REVISION
        ).resolve()
        if snapshot != expected_snapshot:
            raise RuntimeError(
                f"snapshot path for {repo_id} is {snapshot}, expected {expected_snapshot}"
            )
        ref = expected_snapshot.parents[1] / "refs/main"
        ref.parent.mkdir(parents=True, exist_ok=True)
        # huggingface_hub reads refs verbatim as directory names; a trailing
        # newline makes offline `revision=main` resolve to a nonexistent path.
        ref.write_text(EXPECTED_REVISION)
        snapshots[repo_id] = str(snapshot)

    validation = validate_tokenizer_assets(root)
    if validation.get("valid") is not True:
        raise RuntimeError(
            "pinned tokenizer cache failed validation: "
            + "; ".join(validation.get("errors", []))
        )
    return {
        "valid": True,
        "revision": EXPECTED_REVISION,
        "snapshots": snapshots,
        "validation_assertions": len(validation["assertions"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(pin_tokenizer_assets(args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
