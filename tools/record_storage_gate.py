#!/usr/bin/env python3
"""Record an auditable disk-space gate around long LIBERO evaluation runs."""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

GIB = 2**30


def _mount_info(root: Path) -> tuple[dict[str, Any] | None, str | None]:
    result = subprocess.run(
        ["findmnt", "--json", "-T", str(root), "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        return None, f"findmnt failed: {result.stderr.strip()}"
    try:
        payload = json.loads(result.stdout)
        rows = payload["filesystems"]
        row = rows[0]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        return None, f"cannot parse findmnt output: {exc}"
    return {
        "target": row.get("target"),
        "source": row.get("source"),
        "fstype": row.get("fstype"),
        "options": row.get("options"),
    }, None


def inspect_storage(
    root: Path,
    *,
    phase: str,
    model_label: str,
    seed: int | None,
    minimum_free_gib: float,
    enforced: bool,
) -> dict[str, Any]:
    root = root.resolve()
    errors: list[str] = []
    if not root.is_dir():
        errors.append(f"project root is not a directory: {root}")
        total = used = free = 0
        mount = None
    else:
        usage = shutil.disk_usage(root)
        total, used, free = usage.total, usage.used, usage.free
        mount, mount_error = _mount_info(root)
        if mount_error:
            errors.append(mount_error)

    minimum_free_bytes = int(minimum_free_gib * GIB)
    enough_free = free >= minimum_free_bytes
    if enforced and not enough_free:
        errors.append(
            f"free disk is {free / GIB:.3f} GiB, below the enforced "
            f"{minimum_free_gib:.3f} GiB threshold"
        )
    return {
        "valid": not errors,
        "timestamp": datetime.now().astimezone().isoformat(),
        "phase": phase,
        "project_root": str(root),
        "model_label": model_label,
        "seed": seed,
        "minimum_free_gib": minimum_free_gib,
        "minimum_free_bytes": minimum_free_bytes,
        "enforced": enforced,
        "enough_free": enough_free,
        "total_bytes": total,
        "used_bytes": used,
        "free_bytes": free,
        "free_gib": free / GIB,
        "mount": mount,
        "errors": errors,
    }


def append_snapshot(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument(
        "--phase",
        required=True,
        choices=("pre_seed", "post_seed", "skip_complete", "post_summary"),
    )
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--minimum-free-gib", type=float, default=100.0)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    if args.minimum_free_gib < 0:
        parser.error("--minimum-free-gib must be non-negative")

    payload = inspect_storage(
        args.root,
        phase=args.phase,
        model_label=args.model_label,
        seed=args.seed,
        minimum_free_gib=args.minimum_free_gib,
        enforced=args.enforce,
    )
    append_snapshot(args.history, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    if not payload["valid"]:
        if args.enforce and not payload["enough_free"] and len(payload["errors"]) == 1:
            raise SystemExit(4)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
