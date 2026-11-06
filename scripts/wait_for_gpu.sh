#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERVAL_SECONDS="${GPU_CHECK_INTERVAL_SECONDS:-3600}"

while true; do
  if bash "${PROJECT_ROOT}/scripts/gpu_gate.sh"; then
    break
  else
    status=$?
  fi
  if [[ "${status}" != "3" ]]; then
    echo "GPU gate failed with status ${status}; fix the gate/runtime before retrying." >&2
    exit "${status}"
  fi
  echo "GPU is busy; next check in ${INTERVAL_SECONDS}s." >&2
  sleep "${INTERVAL_SECONDS}"
done

echo "GPU gate passed."
