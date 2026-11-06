#!/usr/bin/env python3
"""Create LIBERO's non-interactive path config from the installed package."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import yaml


def ensure_libero_config(config_dir: Path | None = None) -> Path:
    target_dir = Path(config_dir or os.environ.get("LIBERO_CONFIG_PATH", "~/.libero")).expanduser()
    target = target_dir / "config.yaml"
    requested_assets = os.environ.get("LIBERO_ASSETS_PATH")
    if target.is_file():
        # Keep an existing user config intact, but make the project-pinned asset
        # location explicit when the managed runner provides one.  LIBERO's
        # runtime reads this key for provenance/configuration even though older
        # package releases resolve assets through get_assets_path().
        if requested_assets:
            payload = yaml.safe_load(target.read_text()) or {}
            if payload.get("assets") != requested_assets:
                payload["assets"] = requested_assets
                target.write_text(yaml.safe_dump(payload, sort_keys=True))
        return target

    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError("The LIBERO package is not installed")
    benchmark_root = Path(next(iter(spec.submodule_search_locations))) / "libero"
    required = [benchmark_root / "bddl_files", benchmark_root / "init_files"]
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Installed LIBERO package is incomplete: {missing}")

    assets_path = Path(requested_assets).expanduser() if requested_assets else benchmark_root / "assets"
    payload = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(benchmark_root.parent / "datasets"),
        "assets": str(assets_path),
    }
    target_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(payload, sort_keys=True))
    return target


if __name__ == "__main__":
    print(ensure_libero_config())
