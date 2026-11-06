#!/usr/bin/env bash
# Shared, side-effect-free conda discovery helpers. This file is sourced by other scripts.

find_conda_executable() {
  local candidate
  if [[ -n "${CONDA_BIN:-}" && -x "${CONDA_BIN}" ]]; then
    printf '%s\n' "${CONDA_BIN}"
    return 0
  fi
  if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    printf '%s\n' "${CONDA_EXE}"
    return 0
  fi
  for candidate in conda mamba micromamba; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      command -v "${candidate}"
      return 0
    fi
  done
  for candidate in \
    "${HOME}/miniforge3/bin/conda" \
    "${HOME}/miniconda3/bin/conda" \
    "${HOME}/anaconda3/bin/conda" \
    "${HOME}/mambaforge/bin/conda"; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

find_named_conda_env() {
  local env_name="$1"
  local conda_executable="${2:-}"
  local base_python env_path

  if [[ -n "${CONDA_ENV_PATH:-}" && -x "${CONDA_ENV_PATH}/bin/python" ]]; then
    printf '%s\n' "${CONDA_ENV_PATH}"
    return 0
  fi
  if [[ -n "${CONDA_PREFIX:-}" && "$(basename "${CONDA_PREFIX}")" == "${env_name}" \
    && -x "${CONDA_PREFIX}/bin/python" ]]; then
    printf '%s\n' "${CONDA_PREFIX}"
    return 0
  fi
  if [[ -z "${conda_executable}" ]]; then
    conda_executable="$(find_conda_executable)" || return 1
  fi

  base_python="$(dirname "${conda_executable}")/python"
  if [[ -x "${base_python}" ]]; then
    env_path="$("${conda_executable}" env list --json | "${base_python}" -c \
      'import json,sys; name=sys.argv[1]; paths=json.load(sys.stdin)["envs"]; print(next((p for p in paths if p.rstrip("/").split("/")[-1] == name), ""))' \
      "${env_name}")"
  else
    env_path="$("${conda_executable}" env list | awk -v name="${env_name}" '$1 == name {print $NF; exit}')"
  fi
  if [[ -n "${env_path}" && -x "${env_path}/bin/python" ]]; then
    printf '%s\n' "${env_path}"
    return 0
  fi
  return 1
}
