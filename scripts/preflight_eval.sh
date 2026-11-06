#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/artifacts/models/smolvla_libero_paper}"
EVAL_LABEL="${EVAL_LABEL:-$(basename "${MODEL_PATH}")}"
if [[ ! "${EVAL_LABEL}" =~ ^[A-Za-z0-9._-]+$ || "${EVAL_LABEL}" == "." || "${EVAL_LABEL}" == ".." ]]; then
  echo "EVAL_LABEL must be a safe single path component." >&2
  exit 2
fi
PREFLIGHT_DIR="${PROJECT_ROOT}/outputs/preflight"
mkdir -p "${PREFLIGHT_DIR}"
for required in \
  "${MODEL_PATH}/model.safetensors" \
  "${PROJECT_ROOT}/artifacts/models/smolvla_base/model.safetensors" \
  "${PROJECT_ROOT}/artifacts/datasets/lerobot-libero/meta/info.json" \
  "${PROJECT_ROOT}/vendor/lerobot/src/lerobot/__init__.py" \
  "${LIBERO_ASSETS_PATH}/.cache/huggingface/trees/${LIBERO_ASSETS_REVISION}.json"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing evaluation asset: ${required}; run scripts/download_assets.sh or tools/link_assets.py." >&2
    exit 2
  fi
done

"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/ensure_libero_config.py" >/dev/null
bash "${PROJECT_ROOT}/scripts/capture_provenance.sh"
"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/validate_libero_assets.py" \
  --root "${LIBERO_ASSETS_PATH}" --revision "${LIBERO_ASSETS_REVISION}" \
  --output "${PREFLIGHT_DIR}/libero_assets_validation.json"
"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/validate_dataset_coverage.py" \
  --dataset-root "${PROJECT_ROOT}/artifacts/datasets/lerobot-libero" \
  --json-out "${PREFLIGHT_DIR}/dataset_coverage.json" >/dev/null
"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/validate_tokenizer_assets.py" \
  --root "${PROJECT_ROOT}" --json-out "${PREFLIGHT_DIR}/tokenizer_assets_validation.json" >/dev/null
"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/validate_processor_contract.py" \
  --root "${PROJECT_ROOT}" --model "${MODEL_PATH}" \
  --json-out "${PREFLIGHT_DIR}/processor_contract_${EVAL_LABEL}.json" >/dev/null

build_formal_eval_command "${MODEL_PATH}" 1000 \
  "${PROJECT_ROOT}/results/raw/${EVAL_LABEL}/seed_1000"
formal_command=("${EVAL_COMMAND[@]}")
"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/validate_eval_command.py" \
  --expected-seed=1000 --json-out="${PREFLIGHT_DIR}/formal_eval_${EVAL_LABEL}_seed1000.json" \
  -- "${formal_command[@]}" >/dev/null
"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/validate_pairing_protocol.py" \
  --root "${PROJECT_ROOT}" --preflight "${PREFLIGHT_DIR}/formal_eval_${EVAL_LABEL}_seed1000.json" \
  --json-out "${PREFLIGHT_DIR}/pairing_${EVAL_LABEL}_validation.json" >/dev/null
echo "Evaluation preflight passed for ${MODEL_PATH}; evidence: ${PREFLIGHT_DIR}"
