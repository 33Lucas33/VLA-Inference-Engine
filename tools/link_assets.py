"""Link external evaluation assets into the repository's expected layout.

The links are local, ignored by Git, and may point to a separately downloaded
asset collection. Existing files or links to another target are never replaced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ASSETS = {
    "paper_model": "artifacts/models/smolvla_libero_paper",
    "base_model": "artifacts/models/smolvla_base",
    "dataset": "artifacts/datasets/lerobot-libero",
    "libero_assets": "artifacts/libero_assets",
    "lerobot_source": "vendor/lerobot",
    "huggingface_cache": ".cache/huggingface",
    "libero_config": ".cache/libero",
}


def link_assets(root: Path, sources: dict[str, Path],
                targets: dict[str, str] | None = None) -> dict[str, str]:
    root = root.resolve()
    targets = targets or ASSETS
    resolved = {key: path.expanduser().resolve(strict=True) for key, path in sources.items()}
    for key, source in resolved.items():
        if not source.is_dir():
            raise NotADirectoryError(f"{key}: {source}")
        target = root / targets[key]
        if target.is_symlink():
            if target.resolve() != source:
                raise FileExistsError(f"{target} already points to {target.resolve()}")
        elif target.exists():
            raise FileExistsError(f"{target} already exists and is not a symlink")
    for key, source in resolved.items():
        target = root / targets[key]
        if not target.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source, target_is_directory=True)
    return {key: str(root / targets[key]) for key in resolved}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
                        help="Directory containing artifacts/, vendor/ and .cache/ assets")
    parser.add_argument("--project-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--local-model", type=Path,
                        help="Optional local trained checkpoint directory")
    args = parser.parse_args()
    if not args.source_root and not args.local_model:
        parser.error("provide --source-root or --local-model")
    sources = {}
    if args.source_root:
        source_root = args.source_root.expanduser().resolve()
        sources = {key: source_root / relative for key, relative in ASSETS.items()}
    targets = dict(ASSETS)
    if args.local_model:
        targets["local_model"] = "artifacts/models/smolvla_libero_100k"
        sources["local_model"] = args.local_model
    print(json.dumps(link_assets(args.project_root, sources, targets), indent=2))


if __name__ == "__main__":
    main()
