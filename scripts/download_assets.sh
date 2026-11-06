#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

mkdir -p "${PROJECT_ROOT}/artifacts/models" "${PROJECT_ROOT}/artifacts/datasets" "${HF_HOME}"

download_pinned() {
  "${HF_BIN}" download "$1" --repo-type "$2" --revision "$3" --local-dir "$4"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/record_hub_tree.py" \
    --repo "$1" --repo-type "$2" --revision "$3" --root "$4"
}
download_pinned "${BASE_MODEL_REPO}" model "${BASE_MODEL_REVISION}" \
  "${PROJECT_ROOT}/artifacts/models/smolvla_base"
download_pinned "${PAPER_MODEL_REPO}" model "${PAPER_MODEL_REVISION}" \
  "${PROJECT_ROOT}/artifacts/models/smolvla_libero_paper"
download_pinned "${LIBERO_ASSETS_REPO}" dataset "${LIBERO_ASSETS_REVISION}" "${LIBERO_ASSETS_PATH}"
download_pinned "${DATASET_REPO}" dataset "${DATASET_REVISION}" \
  "${PROJECT_ROOT}/artifacts/datasets/lerobot-libero"
"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/pin_tokenizer_assets.py" --root="${PROJECT_ROOT}"

bash "${PROJECT_ROOT}/scripts/capture_provenance.sh"

echo "Assets downloaded with pinned revisions."
