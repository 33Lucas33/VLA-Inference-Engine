"""Emit the shared formal command as NUL-delimited argv for Bash."""

import argparse
import sys
from pathlib import Path

from vla_inference_engine.benchmark.libero import build_formal_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    command = build_formal_command(model_path=args.model, seed=args.seed, output_dir=args.out)
    for argument in command:
        sys.stdout.buffer.write(argument.encode() + b"\0")


if __name__ == "__main__":
    main()
