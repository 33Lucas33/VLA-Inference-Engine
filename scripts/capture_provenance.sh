#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

OUTPUT_DIR="${PROJECT_ROOT}/outputs/provenance"
mkdir -p "${OUTPUT_DIR}"
"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/generate_provenance.py" \
  --root "${PROJECT_ROOT}" --output-dir "${OUTPUT_DIR}"
"${PYTHON_BIN}" -m pip freeze >"${OUTPUT_DIR}/pip-freeze.txt"

if CONDA_BIN="$(find_conda_executable)"; then
  "${CONDA_BIN}" list --explicit -p "${CONDA_ENV_PATH}" >"${OUTPUT_DIR}/conda-explicit.txt"
else
  echo "conda executable unavailable" >"${OUTPUT_DIR}/conda-explicit.txt"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -q >"${OUTPUT_DIR}/nvidia-smi-q.txt"
else
  echo "nvidia-smi unavailable" >"${OUTPUT_DIR}/nvidia-smi-q.txt"
fi
uname -a >"${OUTPUT_DIR}/uname.txt"
