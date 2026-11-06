#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "${PROJECT_ROOT}/scripts/fetch_lerobot.sh"
source "${PROJECT_ROOT}/scripts/resolve_env.sh"
if ! CONDA_BIN="$(find_conda_executable)"; then
  echo "Cannot find conda/mamba. Set CONDA_BIN to its executable path." >&2
  exit 2
fi

if ENV_PATH="$(find_named_conda_env libero "${CONDA_BIN}")"; then
  echo "Reusing existing conda environment: ${ENV_PATH}"
else
  "${CONDA_BIN}" env create -f "${PROJECT_ROOT}/environment.yml"
  ENV_PATH="$(find_named_conda_env libero "${CONDA_BIN}")"
fi
# egl-probe 1.0.2 cannot be built by CMake 4.x. Keep CMake 3.x on PATH during build.
"${CONDA_BIN}" install -p "${ENV_PATH}" -c conda-forge 'cmake<4' 'ffmpeg<9' ninja -y
"${ENV_PATH}/bin/python" -m pip install 'uv==0.12.3'

INSTALL_MODE="${SMOLVLA_INSTALL_MODE:-locked}"
LOCK_FILE="${PROJECT_ROOT}/requirements-linux-cu128.lock.txt"
case "${INSTALL_MODE}" in
  locked)
    if [[ ! -s "${LOCK_FILE}" ]]; then
      echo "Locked install requested but lock file is missing: ${LOCK_FILE}" >&2
      exit 2
    fi
    # The lock is the exact non-local pip freeze from the validated RTX 3090/cu128
    # run. LeRobot itself is installed from the vendored fixed commit below.
    PATH="${ENV_PATH}/bin:${PATH}" "${ENV_PATH}/bin/uv" pip sync \
      --python "${ENV_PATH}/bin/python" --torch-backend=cu128 --strict "${LOCK_FILE}"
    PATH="${ENV_PATH}/bin:${PATH}" "${ENV_PATH}/bin/uv" pip install \
      --python "${ENV_PATH}/bin/python" --no-deps -e "${PROJECT_ROOT}/vendor/lerobot"
    ;;
  resolve)
    "${ENV_PATH}/bin/uv" pip install --python "${ENV_PATH}/bin/python" \
      --torch-backend=cu128 torch torchvision
    PATH="${ENV_PATH}/bin:${PATH}" "${ENV_PATH}/bin/uv" pip install \
      --python "${ENV_PATH}/bin/python" \
      -e "${PROJECT_ROOT}/vendor/lerobot[smolvla,libero,training,evaluation]"
    "${ENV_PATH}/bin/uv" pip install --python "${ENV_PATH}/bin/python" 'cmake<4'
    ;;
  *)
    echo "SMOLVLA_INSTALL_MODE must be 'locked' or 'resolve', got: ${INSTALL_MODE}" >&2
    exit 2
    ;;
esac

"${ENV_PATH}/bin/python" -m pip check
"${ENV_PATH}/bin/python" -m pip install --no-deps -e "${PROJECT_ROOT}"
LD_LIBRARY_PATH="${ENV_PATH}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  CUDA_VISIBLE_DEVICES="" "${ENV_PATH}/bin/python" -c \
  'import torch, lerobot, libero, mujoco; print(torch.__version__, lerobot.__version__, mujoco.__version__)'
