#!/usr/bin/env python3
"""Prove that both SmolVLA tokenizer names resolve offline to one fixed snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

EXPECTED_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
TOKENIZER_REPOS = (
    "HuggingFaceTB/SmolVLM2-500M-Instruct",
    "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
)
REQUIRED_FILES = (
    "added_tokens.json",
    "config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
EXPECTED_FILE_SHA256 = {
    "added_tokens.json": "74135b8664b56088c0006f1c8e848d79a8eba003411f72ebf1dc2ee96227be3a",
    "config.json": "ea6bc1237e96247f6258de3e202e2e62b93d6f386dc47e7b36b5588bf3a15e17",
    "merges.txt": "0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510",
    "special_tokens_map.json": "2dfea2a426162316ff1567c82bc6d36d9690cd9f90455f075c77daca78b45c60",
    "tokenizer.json": "5ece781dc8d2b2f3e2f289ca0ae50b17cfc27dd27bfe7971bb8241e0b964331a",
    "tokenizer_config.json": "dd9ce2ab89a3dd881bd9378f1a79b943a064b9275a7e1706d5b7b47b68977913",
    "vocab.json": "82b84012e3add4d01d12ba14442026e49b8cbbaead1f79ecf3d919784f82dc79",
}
EXPECTED_FILE_MANIFEST_SHA256 = "1ad4ee98b3dd8b02decee8991bb673656a3f5ae73cb81760db55d96598bf7aa1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _repo_cache(cache_root: Path, repo_id: str) -> Path:
    return cache_root / ("models--" + repo_id.replace("/", "--"))


def validate_tokenizer_assets(root: Path) -> dict[str, Any]:
    root = root.resolve()
    cache_root = root / ".cache/huggingface/hub"
    tasks_path = root / "artifacts/datasets/lerobot-libero/meta/tasks.parquet"
    errors: list[str] = []
    assertions: list[dict[str, Any]] = []

    def record(name: str, ok: bool, observed: Any, expected: Any) -> None:
        assertions.append(
            {"name": name, "ok": bool(ok), "observed": observed, "expected": expected}
        )
        if not ok:
            errors.append(f"{name}: observed={observed!r}, expected={expected!r}")

    if not tasks_path.is_file() or tasks_path.stat().st_size <= 0:
        record("tasks_file", False, str(tasks_path), "nonempty tasks.parquet")
        return {
            "valid": False,
            "errors": errors,
            "project_root": str(root),
            "assertions": assertions,
        }

    table = pq.read_table(tasks_path)
    task_rows = table.to_pylist()
    task_indices = [row.get("task_index") for row in task_rows]
    instructions = [row.get("__index_level_0__") for row in task_rows]
    record("tasks.count", len(instructions) == 40, len(instructions), 40)
    record("tasks.indices", task_indices == list(range(40)), task_indices, list(range(40)))
    record(
        "tasks.language_instructions",
        all(isinstance(item, str) and item.strip() for item in instructions)
        and len(set(instructions)) == 40,
        {"nonempty": sum(isinstance(item, str) and bool(item.strip()) for item in instructions),
         "unique": len(set(item for item in instructions if isinstance(item, str)))},
        {"nonempty": 40, "unique": 40},
    )

    snapshots: dict[str, dict[str, Any]] = {}
    for repo_id in TOKENIZER_REPOS:
        repo_cache = _repo_cache(cache_root, repo_id)
        ref_path = repo_cache / "refs/main"
        # Do not strip: huggingface_hub uses the ref contents verbatim when it
        # builds the snapshot path, so even a trailing newline breaks offline loading.
        ref_value = ref_path.read_text() if ref_path.is_file() else None
        record(f"{repo_id}.refs_main", ref_value == EXPECTED_REVISION, ref_value, EXPECTED_REVISION)
        snapshot = repo_cache / "snapshots" / EXPECTED_REVISION
        file_hashes: dict[str, str | None] = {}
        resolved_files: dict[str, str] = {}
        for filename in REQUIRED_FILES:
            path = snapshot / filename
            ok = path.is_file() and path.stat().st_size > 0
            record(f"{repo_id}.file.{filename}", ok, str(path), "nonempty fixed-snapshot file")
            file_hashes[filename] = sha256(path) if ok else None
            record(
                f"{repo_id}.sha256.{filename}",
                file_hashes[filename] == EXPECTED_FILE_SHA256[filename],
                file_hashes[filename],
                EXPECTED_FILE_SHA256[filename],
            )
            resolved_files[filename] = str(path.resolve()) if ok else str(path)
        snapshots[repo_id] = {
            "repo_cache": str(repo_cache),
            "refs_main": ref_value,
            "snapshot": str(snapshot),
            "file_sha256": file_hashes,
            "resolved_files": resolved_files,
        }

    reference_hashes = snapshots[TOKENIZER_REPOS[0]]["file_sha256"]
    record(
        "tokenizer_aliases.file_hashes_equal",
        snapshots[TOKENIZER_REPOS[1]]["file_sha256"] == reference_hashes,
        {repo: snapshots[repo]["file_sha256"] for repo in TOKENIZER_REPOS},
        "identical hashes for every required tokenizer file",
    )

    tokenization_profiles: dict[str, dict[str, Any]] = {}
    if all(isinstance(item, str) for item in instructions):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            from transformers import AutoTokenizer

            prompts = [instruction + "\n" for instruction in instructions]
            for repo_id in TOKENIZER_REPOS:
                tokenizer = AutoTokenizer.from_pretrained(
                    repo_id,
                    cache_dir=cache_root,
                    local_files_only=True,
                )
                raw_lengths = [
                    len(tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"])
                    for prompt in prompts
                ]
                batch1_ids = [
                    tokenizer(
                        [prompt],
                        max_length=48,
                        truncation=True,
                        padding="longest",
                        padding_side="right",
                        return_attention_mask=True,
                    )["input_ids"][0]
                    for prompt in prompts
                ]
                batch1_masks = [
                    tokenizer(
                        [prompt],
                        max_length=48,
                        truncation=True,
                        padding="longest",
                        padding_side="right",
                        return_attention_mask=True,
                    )["attention_mask"][0]
                    for prompt in prompts
                ]
                profile = {
                    "vocab_size": len(tokenizer),
                    "pad_token_id": tokenizer.pad_token_id,
                    "bos_token_id": tokenizer.bos_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "raw_token_lengths": raw_lengths,
                    "maximum_raw_token_length": max(raw_lengths),
                    "batch1_input_ids_sha256": canonical_sha256(batch1_ids),
                    "batch1_attention_masks_sha256": canonical_sha256(batch1_masks),
                    "all_attention_masks_are_one": all(
                        all(value == 1 for value in mask) for mask in batch1_masks
                    ),
                }
                tokenization_profiles[repo_id] = profile
                record(
                    f"{repo_id}.no_task_truncation",
                    profile["maximum_raw_token_length"] <= 48,
                    profile["maximum_raw_token_length"],
                    "<=48",
                )
        except Exception as exc:
            record(
                "offline_tokenizer_load",
                False,
                f"{type(exc).__name__}: {exc}",
                "both tokenizers load from fixed local cache",
            )

    if len(tokenization_profiles) == 2:
        record(
            "tokenizer_aliases.task_tokenization_equal",
            tokenization_profiles[TOKENIZER_REPOS[0]]
            == tokenization_profiles[TOKENIZER_REPOS[1]],
            tokenization_profiles,
            "identical offline profiles for all 40 newline-terminated tasks",
        )

    return {
        "valid": not errors,
        "errors": errors,
        "project_root": str(root),
        "cache_root": str(cache_root),
        "expected_revision": EXPECTED_REVISION,
        "tokenizer_repos": list(TOKENIZER_REPOS),
        "required_files": list(REQUIRED_FILES),
        "expected_file_sha256": EXPECTED_FILE_SHA256,
        "tokenizer_file_manifest_sha256": canonical_sha256(reference_hashes),
        "expected_file_manifest_sha256": EXPECTED_FILE_MANIFEST_SHA256,
        "tasks_path": str(tasks_path),
        "tasks_sha256": sha256(tasks_path),
        "snapshots": snapshots,
        "tokenization_profiles": tokenization_profiles,
        "assertions": assertions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    result = validate_tokenizer_assets(args.root)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    print(rendered, end="")
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
