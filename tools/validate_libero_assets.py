#!/usr/bin/env python3
"""Validate the pinned LIBERO simulator asset snapshot before offline evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REQUIRED_DIRS = (
    "articulated_objects",
    "scenes",
    "stable_hope_objects",
    "stable_scanned_objects",
    "textures",
    "turbosquid_objects",
)
REQUIRED_FILES = (
    "serving_region.xml",
    "scenes/libero_tabletop_base_style.xml",
    "scenes/libero_kitchen_tabletop_base_style.xml",
    "scenes/libero_living_room_tabletop_base_style.xml",
    "scenes/libero_coffee_table_base_style.xml",
    "scenes/libero_study_base_style.xml",
    "scenes/libero_floor_base_style.xml",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_assets(root: Path, revision: str, output: Path | None = None) -> dict:
    root = root.expanduser().resolve()
    errors: list[str] = []
    if not root.is_dir():
        errors.append(f"asset root is not a directory: {root}")
    files = []
    if root.is_dir():
        files = sorted(p for p in root.rglob("*") if p.is_file() and ".cache" not in p.parts)
        for relative in REQUIRED_DIRS:
            if not (root / relative).is_dir():
                errors.append(f"missing required directory: {relative}")
        for relative in REQUIRED_FILES:
            path = root / relative
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"missing/empty required file: {relative}")
        errors.extend(
            f"empty asset file: {p.relative_to(root)}" for p in files if p.stat().st_size == 0
        )

    manifest = [
        {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in files
    ]
    result = {
        "valid": not errors and len(files) == 586,
        "repo_id": "lerobot/libero-assets",
        "repo_type": "dataset",
        "revision": revision,
        "asset_root": str(root),
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in manifest),
        "required_dirs": list(REQUIRED_DIRS),
        "required_files": list(REQUIRED_FILES),
        "errors": errors + ([] if len(files) == 586 else [f"expected 586 files, found {len(files)}"]),
        "manifest": manifest,
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_assets(args.root, args.revision, args.output)
    print(json.dumps({k: result[k] for k in ("valid", "repo_id", "revision", "file_count", "total_bytes", "errors")}, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
