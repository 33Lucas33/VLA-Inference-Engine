#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/artifacts/models/smolvla_libero_paper}"
EVAL_LABEL="${EVAL_LABEL:-$(basename "${MODEL_PATH}")}"
export MODEL_PATH EVAL_LABEL
SEEDS="${EVAL_SEEDS:-1000 2000 3000}"
EPISODES="${EVAL_EPISODES:-10}"
if [[ "${SEEDS}" != "1000 2000 3000" || "${EPISODES}" != "10" ]]; then
  echo "Formal protocol requires seeds 1000 2000 3000 and 10 episodes per task." >&2
  exit 2
fi
if [[ ! "${EVAL_LABEL}" =~ ^[A-Za-z0-9._-]+$ || "${EVAL_LABEL}" == "." || "${EVAL_LABEL}" == ".." ]]; then
  echo "EVAL_LABEL must be a safe single path component." >&2
  exit 2
fi
bash "${PROJECT_ROOT}/scripts/preflight_eval.sh"
MIN_FREE_GIB="${EVAL_MIN_FREE_GIB:-20}"
STORAGE_HISTORY="${PROJECT_ROOT}/outputs/storage/evaluation_checks.jsonl"
read -r -a SEED_ARRAY <<<"${SEEDS}"
SEED_IDS_CSV="$(IFS=,; echo "${SEED_ARRAY[*]}")"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

for seed in ${SEEDS}; do
  output_dir="${PROJECT_ROOT}/results/raw/${EVAL_LABEL}/seed_${seed}"
  log_dir="${PROJECT_ROOT}/outputs/logs/eval/${EVAL_LABEL}"
  command_sidecar="${log_dir}/seed_${seed}.command.sh"
  preflight_sidecar="${log_dir}/seed_${seed}.preflight.json"
  log_sidecar="${log_dir}/seed_${seed}.log"
  if [[ -f "${output_dir}/eval_info.json" ]]; then
    if "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/validate_formal_seed.py" \
      "${output_dir}/eval_info.json" \
      --project-root="${PROJECT_ROOT}" --model-label="${EVAL_LABEL}" \
      --seed="${seed}" --episodes-per-task="${EPISODES}" --expected-model="${MODEL_PATH}" \
      --json-out="${output_dir}/formal_seed_validation.json" >/dev/null; then
      echo "Skipping fully provenance-complete evaluation: ${output_dir}"
      "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/record_storage_gate.py" \
        --root="${PROJECT_ROOT}" --history="${STORAGE_HISTORY}" \
        --phase=skip_complete --model-label="${EVAL_LABEL}" --seed="${seed}" \
        --minimum-free-gib="${MIN_FREE_GIB}"
      continue
    fi
    echo "eval_info.json exists but fails full seed/provenance validation: ${output_dir}" >&2
  fi

  failed_dir=""
  if [[ -e "${output_dir}" ]]; then
    failed_dir="${PROJECT_ROOT}/results/failed_runs/${EVAL_LABEL}/seed_${seed}_$(date +%Y%m%d_%H%M%S)_$$"
    mkdir -p "$(dirname "${failed_dir}")"
    mv "${output_dir}" "${failed_dir}"
    echo "Preserved incomplete/invalid run at ${failed_dir}; starting a clean retry." >&2
  fi

  # command/preflight/log live outside output_dir until the rollout exits. If a
  # process dies before they are copied into the raw run, preserve those exact
  # sidecars as well instead of silently overwriting the failure evidence.
  for sidecar in "${command_sidecar}" "${preflight_sidecar}" "${log_sidecar}"; do
    if [[ -e "${sidecar}" ]]; then
      if [[ -z "${failed_dir}" ]]; then
        failed_dir="${PROJECT_ROOT}/results/failed_runs/${EVAL_LABEL}/seed_${seed}_$(date +%Y%m%d_%H%M%S)_$$"
        mkdir -p "${failed_dir}"
        echo "Preserving stale external seed artifacts at ${failed_dir}." >&2
      fi
      mkdir -p "${failed_dir}/external_run_artifacts"
      mv "${sidecar}" "${failed_dir}/external_run_artifacts/"
    fi
  done

  "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/record_storage_gate.py" \
    --root="${PROJECT_ROOT}" --history="${STORAGE_HISTORY}" \
    --phase=pre_seed --model-label="${EVAL_LABEL}" --seed="${seed}" \
    --minimum-free-gib="${MIN_FREE_GIB}" --enforce

  # A user workload may start between long seed runs. Re-check only when this
  # seed actually needs a rollout; protocol-complete runs remain CPU-skippable.
  bash "${PROJECT_ROOT}/scripts/wait_for_gpu.sh"

  mkdir -p "${log_dir}"
  build_formal_eval_command "${MODEL_PATH}" "${seed}" "${output_dir}"
  command=("${EVAL_COMMAND[@]}")

  printf '%q ' "${command[@]}" >"${command_sidecar}"
  printf '\n' >>"${command_sidecar}"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/validate_eval_command.py" \
    --expected-seed="${seed}" \
    --json-out="${preflight_sidecar}" \
    -- "${command[@]}"
  "${command[@]}" 2>&1 | tee "${log_sidecar}"

  cp "${command_sidecar}" "${output_dir}/command.sh"
  cp "${preflight_sidecar}" "${output_dir}/command_preflight.json"

  "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/validate_eval_info.py" \
    "${output_dir}/eval_info.json" --episodes-per-task="${EPISODES}" \
    --require-video-evidence \
    --require-command-preflight \
    --json-out="${output_dir}/protocol_validation.json"
  {
    date --iso-8601=seconds
    uname -a
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    "${PYTHON_BIN}" --version
    "${PYTHON_BIN}" -m pip freeze
  } >"${output_dir}/environment.txt"

  "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/validate_formal_seed.py" \
    "${output_dir}/eval_info.json" \
    --project-root="${PROJECT_ROOT}" --model-label="${EVAL_LABEL}" \
    --seed="${seed}" --episodes-per-task="${EPISODES}" --expected-model="${MODEL_PATH}" \
    --json-out="${output_dir}/formal_seed_validation.json"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/record_storage_gate.py" \
    --root="${PROJECT_ROOT}" --history="${STORAGE_HISTORY}" \
    --phase=post_seed --model-label="${EVAL_LABEL}" --seed="${seed}" \
    --minimum-free-gib="${MIN_FREE_GIB}"
done

"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/summarize_results.py" \
  "${PROJECT_ROOT}/results/raw/${EVAL_LABEL}" --output-dir "${PROJECT_ROOT}/results/${EVAL_LABEL}" \
  --require-video-evidence --expected-seeds="${#SEED_ARRAY[@]}" \
  --require-command-preflight \
  --expected-seed-ids="${SEED_IDS_CSV}"
"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/validate_formal_summary.py" \
  "${PROJECT_ROOT}/results/${EVAL_LABEL}/metrics_summary.json" \
  --raw-root="${PROJECT_ROOT}/results/raw/${EVAL_LABEL}" \
  --model-label="${EVAL_LABEL}" --expected-model="${MODEL_PATH}" \
  --expected-seed-ids="${SEED_IDS_CSV}" \
  --json-out="${PROJECT_ROOT}/results/${EVAL_LABEL}/formal_validation.json"
"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/record_storage_gate.py" \
  --root="${PROJECT_ROOT}" --history="${STORAGE_HISTORY}" \
  --phase=post_summary --model-label="${EVAL_LABEL}" \
  --minimum-free-gib="${MIN_FREE_GIB}"
