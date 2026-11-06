#!/usr/bin/env python3
"""Inspect or safely rebind the LeRobot editable install to this project vendor."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

EXPECTED_COMMIT = "8ae2354592ec4480c38134fb75350b8e7992e0e9"


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _git(repo: Path, *args: str) -> str | None:
    result = _run(["git", "-C", str(repo), *args])
    return result.stdout.strip() if result.returncode == 0 else None


def _import_probe() -> dict[str, Any]:
    code = r'''
import importlib.metadata as metadata
import json
from pathlib import Path
import lerobot

distribution = metadata.distribution("lerobot")
direct_url = None
for item in distribution.files or []:
    if str(item).endswith("direct_url.json"):
        path = Path(distribution.locate_file(item))
        direct_url = json.loads(path.read_text())
        break
print(json.dumps({
    "module_file": str(Path(lerobot.__file__).resolve()),
    "version": metadata.version("lerobot"),
    "distribution_path": str(Path(distribution._path).resolve()),
    "direct_url": direct_url,
}))
'''
    result = _run([sys.executable, "-c", code])
    if result.returncode != 0:
        raise RuntimeError(f"cannot probe LeRobot import: {result.stderr.strip()}")
    return json.loads(result.stdout)


def _repo_from_module(module_file: Path) -> Path | None:
    # <repo>/src/lerobot/__init__.py
    try:
        candidate = module_file.parents[2]
    except IndexError:
        return None
    return candidate if (candidate / ".git").exists() else None


def inspect_binding(root: Path) -> dict[str, Any]:
    root = root.resolve()
    target_repo = (root / "vendor/lerobot").resolve()
    errors: list[str] = []
    try:
        probe = _import_probe()
    except Exception as exc:
        probe = {}
        errors.append(f"{type(exc).__name__}: {exc}")

    module_file = Path(probe["module_file"]).resolve() if probe.get("module_file") else None
    source_repo = _repo_from_module(module_file) if module_file else None
    target_commit = _git(target_repo, "rev-parse", "HEAD")
    target_tree = _git(target_repo, "rev-parse", "HEAD^{tree}")
    target_status = _git(target_repo, "status", "--porcelain", "--untracked-files=no")
    source_commit = _git(source_repo, "rev-parse", "HEAD") if source_repo else None
    source_tree = _git(source_repo, "rev-parse", "HEAD^{tree}") if source_repo else None
    source_status = (
        _git(source_repo, "status", "--porcelain", "--untracked-files=no")
        if source_repo
        else None
    )
    bound_to_project_vendor = False
    if module_file is not None:
        try:
            module_file.relative_to(target_repo / "src")
        except ValueError:
            pass
        else:
            bound_to_project_vendor = True

    target_valid = target_commit == EXPECTED_COMMIT and target_status == ""
    source_equivalent = (
        source_commit == EXPECTED_COMMIT
        and source_tree == target_tree
        and source_status == ""
        and target_valid
    )
    if not target_valid:
        errors.append("project vendor is not the fixed clean LeRobot commit")
    if not source_equivalent:
        errors.append("installed LeRobot source is not Git-tree equivalent to the project vendor")

    return {
        "valid": not errors,
        "errors": errors,
        "python": sys.executable,
        "project_root": str(root),
        "expected_commit": EXPECTED_COMMIT,
        "module_file": str(module_file) if module_file else None,
        "distribution_path": probe.get("distribution_path"),
        "direct_url": probe.get("direct_url"),
        "package_version": probe.get("version"),
        "source_repo": str(source_repo) if source_repo else None,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "source_worktree_clean": source_status == "",
        "target_repo": str(target_repo),
        "target_commit": target_commit,
        "target_tree": target_tree,
        "target_worktree_clean": target_status == "",
        "source_tree_equivalent_to_target": source_equivalent,
        "bound_to_project_vendor": bound_to_project_vendor,
        "ready_for_evaluation": source_equivalent and bound_to_project_vendor,
    }


def rebind(root: Path) -> dict[str, Any]:
    root = root.resolve()
    before = inspect_binding(root)
    if before["valid"] is not True:
        return {
            "valid": False,
            "errors": ["refusing to rebind because the training-time source is not equivalent"],
            "before": before,
            "after": None,
            "command": None,
        }
    command = [
        str(Path(sys.executable).resolve().parent / "uv"),
        "pip",
        "install",
        "--python",
        sys.executable,
        "--no-deps",
        "-e",
        str((root / "vendor/lerobot").resolve()),
    ]
    process = _run(command)
    after = inspect_binding(root) if process.returncode == 0 else None
    errors = []
    if process.returncode != 0:
        errors.append(f"editable rebind command failed: {process.stderr.strip()}")
    elif after is None or after["ready_for_evaluation"] is not True:
        errors.append("LeRobot import is not bound to the project vendor after reinstall")
    return {
        "valid": not errors,
        "errors": errors,
        "before": before,
        "after": after,
        "command": command,
        "command_returncode": process.returncode,
        "command_stdout": process.stdout,
        "command_stderr": process.stderr,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--mode", choices=("inspect", "rebind"), default="inspect")
    parser.add_argument("--require-project-binding", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    result = inspect_binding(args.root) if args.mode == "inspect" else rebind(args.root)
    if args.mode == "inspect" and args.require_project_binding and result["ready_for_evaluation"] is not True:
        result = dict(result)
        result["valid"] = False
        result["errors"] = [*result["errors"], "LeRobot is not bound to the project vendor"]
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    print(rendered, end="")
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
