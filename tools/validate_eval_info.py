#!/usr/bin/env python3
"""Strictly validate one formal LIBERO eval_info.json file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.summarize_results import validate_eval_payload
except ModuleNotFoundError:
    from summarize_results import validate_eval_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_info", type=Path)
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--require-video-evidence", action="store_true")
    parser.add_argument("--require-command-preflight", action="store_true")
    args = parser.parse_args()

    try:
        payload = json.loads(args.eval_info.read_text())
        result = validate_eval_payload(
            payload,
            args.episodes_per_task,
            eval_info_dir=args.eval_info.parent,
            require_video_evidence=args.require_video_evidence,
            require_command_preflight=args.require_command_preflight,
        )
    except (OSError, json.JSONDecodeError) as exc:
        result = {"valid": False, "errors": [f"cannot load eval_info: {exc}"]}

    rendered = json.dumps(result, indent=2) + "\n"
    if args.json_out:
        args.json_out.write_text(rendered)
    print(rendered, end="")
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
