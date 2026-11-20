#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
exec "${PYTHON_BIN}" -m vla_inference_engine.benchmark.action_horizon_showcase \
  --model "${PROJECT_ROOT}/artifacts/models/smolvla_libero_paper" \
  --out "${PROJECT_ROOT}/runs/action-horizon-showcase/$(date +%Y%m%d-%H%M%S)-$$" \
  "$@"
