#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="${PROJECT_ROOT}/vendor/lerobot"
COMMIT="8ae2354592ec4480c38134fb75350b8e7992e0e9"
if [[ ! -e "${VENDOR}" ]]; then
  mkdir -p "${PROJECT_ROOT}/vendor"
  git clone --filter=blob:none https://github.com/huggingface/lerobot.git "${VENDOR}"
  git -C "${VENDOR}" checkout --detach "${COMMIT}"
fi
observed="$(git -C "${VENDOR}" rev-parse HEAD)"
if [[ "${observed}" != "${COMMIT}" ]]; then
  echo "LeRobot commit mismatch: ${observed}; expected ${COMMIT}" >&2
  exit 2
fi
echo "LeRobot source ready: ${VENDOR} @ ${COMMIT}"
