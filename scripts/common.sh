#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/resolve_env.sh"
if ! CONDA_ENV_PATH="$(find_named_conda_env libero)"; then
  echo "Cannot locate the conda environment named 'libero'. Set CONDA_ENV_PATH explicitly." >&2
  exit 2
fi
export PYTHON_BIN="${CONDA_ENV_PATH}/bin/python"
HF_BIN="${CONDA_ENV_PATH}/bin/hf"

LEROBOT_COMMIT="8ae2354592ec4480c38134fb75350b8e7992e0e9"
DATASET_REPO="lerobot/libero"
DATASET_REVISION="a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
BASE_MODEL_REPO="lerobot/smolvla_base"
BASE_MODEL_REVISION="c83c3163b8ca9b7e67c509fffd9121e66cb96205"
PAPER_MODEL_REPO="HuggingFaceVLA/smolvla_libero"
PAPER_MODEL_REVISION="6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
LIBERO_ASSETS_REPO="lerobot/libero-assets"
LIBERO_ASSETS_REVISION="0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"

export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${PROJECT_ROOT}/artifacts/datasets}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${PROJECT_ROOT}/.cache/libero}"
export LIBERO_ASSETS_PATH="${LIBERO_ASSETS_PATH:-${PROJECT_ROOT}/artifacts/libero_assets}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PATH="${CONDA_ENV_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV_PATH}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing conda environment at ${CONDA_ENV_PATH}. Run scripts/bootstrap_env.sh first." >&2
  exit 2
fi

# Set EVAL_COMMAND as argv. A temporary stream preserves spaces/newlines and
# propagates Python failures, unlike process substitution inside mapfile.
build_formal_eval_command() {
  local stream
  stream="$(mktemp)" || return
  if "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/build_eval_command.py" \
    --model="$1" --seed="$2" --out="$3" >"${stream}"; then
    mapfile -d '' -t EVAL_COMMAND <"${stream}"
    rm -f "${stream}"
  else
    rm -f "${stream}"
    return 1
  fi
}
