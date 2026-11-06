#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

PAPER_MODEL_PATH="${PAPER_MODEL_PATH:-${PROJECT_ROOT}/artifacts/models/smolvla_libero_paper}"
LOCAL_MODEL_PATH="${LOCAL_MODEL_PATH:-${PROJECT_ROOT}/artifacts/models/smolvla_libero_100k}"
if [[ ! -f "${LOCAL_MODEL_PATH}/model.safetensors" ]]; then
  echo "Local checkpoint missing: ${LOCAL_MODEL_PATH}; set LOCAL_MODEL_PATH explicitly." >&2
  exit 2
fi

MODEL_PATH="${PAPER_MODEL_PATH}" EVAL_LABEL=smolvla_libero_paper \
  bash "${PROJECT_ROOT}/scripts/eval_all_suites.sh"
MODEL_PATH="${LOCAL_MODEL_PATH}" EVAL_LABEL=smolvla_libero_100k \
  bash "${PROJECT_ROOT}/scripts/eval_all_suites.sh"
"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/compare_models.py" \
  --paper "${PROJECT_ROOT}/results/smolvla_libero_paper/metrics_summary.json" \
  --local "${PROJECT_ROOT}/results/smolvla_libero_100k/metrics_summary.json" \
  --output-dir "${PROJECT_ROOT}/results/comparison"
"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/prepare_gap_diagnostics.py" --root "${PROJECT_ROOT}"
