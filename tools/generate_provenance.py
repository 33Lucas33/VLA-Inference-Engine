#!/usr/bin/env python3
"""Generate a machine-readable provenance and integrity report for all fixed assets."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

try:
    from tools.manage_lerobot_binding import inspect_binding
except ModuleNotFoundError:
    from manage_lerobot_binding import inspect_binding


ASSETS = (
    {
        "name": "smolvla_base",
        "repo": "lerobot/smolvla_base",
        "revision": "c83c3163b8ca9b7e67c509fffd9121e66cb96205",
        "relative_root": "artifacts/models/smolvla_base",
    },
    {
        "name": "smolvla_libero_paper",
        "repo": "HuggingFaceVLA/smolvla_libero",
        "revision": "6721902bc4d61e50a3bfdb11dfb4cb626f05d102",
        "relative_root": "artifacts/models/smolvla_libero_paper",
    },
    {
        "name": "lerobot_libero_dataset",
        "repo": "lerobot/libero",
        "revision": "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4",
        "relative_root": "artifacts/datasets/lerobot-libero",
    },
)
LEROBOT_COMMIT = "8ae2354592ec4480c38134fb75350b8e7992e0e9"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str | None]:
    names = (
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "lerobot",
        "hf_libero",
        "mujoco",
        "robosuite",
        "gymnasium",
        "safetensors",
        "streamlit",
    )
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def generate(root: Path) -> tuple[dict, list[dict]]:
    root = root.resolve()
    asset_results = []
    tree_rows = []
    all_valid = True

    for spec in ASSETS:
        asset_root = root / spec["relative_root"]
        tree_path = asset_root / ".cache/huggingface/trees" / f"{spec['revision']}.json"
        if not tree_path.is_file():
            asset_results.append({**spec, "valid": False, "error": f"missing {tree_path}"})
            all_valid = False
            continue
        tree = json.loads(tree_path.read_text()).get("files", {})
        missing = []
        size_mismatches = []
        for relative_path, metadata in sorted(tree.items()):
            local_path = asset_root / relative_path
            if not local_path.is_file():
                missing.append(relative_path)
                actual_size = None
            else:
                actual_size = local_path.stat().st_size
                if actual_size != metadata["size"]:
                    size_mismatches.append(
                        {"path": relative_path, "expected": metadata["size"], "actual": actual_size}
                    )
            tree_rows.append(
                {
                    "asset": spec["name"],
                    "path": relative_path,
                    "expected_size": metadata["size"],
                    "actual_size": actual_size,
                    "blob_id": metadata.get("blob_id", ""),
                    "lfs_sha256": metadata.get("lfs_sha256", ""),
                }
            )

        critical_hashes = {}
        if spec["name"].startswith("smolvla"):
            tree_hash = tree.get("model.safetensors", {}).get("lfs_sha256")
            observed_hash = sha256(asset_root / "model.safetensors") if (asset_root / "model.safetensors").is_file() else None
            critical_hashes["model.safetensors"] = {
                "expected_sha256": tree_hash,
                "observed_sha256": observed_hash,
                "matches": bool(tree_hash and observed_hash == tree_hash),
            }
        else:
            for relative_path in ("meta/info.json", "meta/stats.json", "meta/tasks.parquet"):
                local_path = asset_root / relative_path
                if local_path.is_file():
                    critical_hashes[relative_path] = {"observed_sha256": sha256(local_path)}

        critical_valid = all(item.get("matches", True) for item in critical_hashes.values())
        valid = not missing and not size_mismatches and critical_valid
        all_valid &= valid
        asset_results.append(
            {
                **spec,
                "valid": valid,
                "file_count": len(tree),
                "total_expected_bytes": sum(item["size"] for item in tree.values()),
                "tree_manifest": str(tree_path.relative_to(root)),
                "tree_manifest_sha256": sha256(tree_path),
                "missing_files": missing,
                "size_mismatches": size_mismatches,
                "critical_hashes": critical_hashes,
            }
        )

    vendor = root / "vendor/lerobot"
    commit = command_output(["git", "-C", str(vendor), "rev-parse", "HEAD"])
    dirty = command_output(["git", "-C", str(vendor), "status", "--porcelain"])
    vendor_valid = commit == LEROBOT_COMMIT and dirty == ""
    all_valid &= vendor_valid
    import_binding = inspect_binding(root)
    binding_valid = import_binding.get("ready_for_evaluation") is True
    all_valid &= binding_valid

    dataset_info_path = root / "artifacts/datasets/lerobot-libero/meta/info.json"
    dataset_info = json.loads(dataset_info_path.read_text())
    system = {
        "python": sys.version,
        "python_prefix": sys.prefix,
        "platform": platform.platform(),
        "kernel": platform.release(),
        "packages": package_versions(),
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ]
        ),
    }
    report = {
        "valid": all_valid,
        "project_root": str(root),
        "lerobot": {
            "expected_commit": LEROBOT_COMMIT,
            "observed_commit": commit,
            "working_tree_clean": dirty == "",
            "status_porcelain": dirty.splitlines(),
            "valid": vendor_valid,
            "import_binding": import_binding,
        },
        "assets": asset_results,
        "dataset_summary": {
            key: dataset_info.get(key)
            for key in ("codebase_version", "robot_type", "total_episodes", "total_frames", "total_tasks", "fps", "splits")
        },
        "system": system,
    }
    return report, tree_rows


def write_outputs(report: dict, tree_rows: list[dict], output_dir: Path) -> None:
    import csv

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "assets.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output_dir / "assets_tree_manifest.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tree_rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(tree_rows)
    lines = [
        "# 实验资产与环境溯源",
        "",
        f"- 完整性：{'通过' if report['valid'] else '失败'}",
        f"- LeRobot commit：`{report['lerobot']['observed_commit']}`；"
        f"工作树干净：{report['lerobot']['working_tree_clean']}",
        f"- LeRobot import：`{report['lerobot']['import_binding']['module_file']}`；"
        f"绑定当前 project vendor：{report['lerobot']['import_binding']['bound_to_project_vendor']}",
        f"- GPU / driver：`{report['system']['nvidia_smi']}`",
        f"- Python prefix：`{report['system']['python_prefix']}`",
        "",
        "| 资产 | repo | revision | 文件数 | 字节数 | 大小/关键哈希校验 |",
        "|---|---|---|---:|---:|:---:|",
    ]
    for asset in report["assets"]:
        lines.append(
            f"| {asset['name']} | `{asset['repo']}` | `{asset['revision']}` "
            f"| {asset.get('file_count', 0)} | {asset.get('total_expected_bytes', 0)} "
            f"| {'通过' if asset['valid'] else '失败'} |"
        )
    dataset = report["dataset_summary"]
    lines.extend(
        [
            "",
            f"- 数据集：{dataset['total_episodes']} episodes、{dataset['total_frames']} frames、"
            f"{dataset['total_tasks']} tasks、{dataset['fps']} fps，schema `{dataset['codebase_version']}`。",
            "- `assets_tree_manifest.tsv` 保存逐文件预期大小、Git blob ID 和 Hugging Face LFS SHA256。",
            "- `pip-freeze.txt` 与 `conda-explicit.txt` 分别用于 Python 与 conda 包复现。",
        ]
    )
    (output_dir / "ASSET_PROVENANCE.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = (args.output_dir or root / "outputs/provenance").resolve()
    report, rows = generate(root)
    write_outputs(report, rows, output_dir)
    print(json.dumps({"valid": report["valid"], "output_dir": str(output_dir)}))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
