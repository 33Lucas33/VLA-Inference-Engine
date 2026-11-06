"""Repository runtime: all evaluation entry points use scripts/common.sh."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def project_root() -> Path:
    root = Path(__file__).resolve().parents[3]
    if not (root / "scripts/common.sh").is_file():
        raise RuntimeError("Evaluation requires a repository checkout and pip install -e .")
    return root


def evaluation_environment(root: Path, *, model: Path, label: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update(MODEL_PATH=str(model.expanduser().resolve()), EVAL_LABEL=label)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    # Positional argument avoids shell interpolation of paths. common.sh selects
    # the same Conda runtime as formal evaluation, even if the CLI uses another Python.
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; exec "$PYTHON_BIN" -c '
         "'import json, os; print(json.dumps(dict(os.environ)))'",
         "vla-evaluation-env", str(root / "scripts/common.sh")],
        cwd=root, env=env, capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)
