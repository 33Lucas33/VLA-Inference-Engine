#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${GPU_GATE_LOG_DIR:-${PROJECT_ROOT}/outputs/gpu_gate}"
MAX_IDLE_MEMORY_MIB="${MAX_IDLE_MEMORY_MIB:-2048}"
MAX_IDLE_UTILIZATION="${MAX_IDLE_UTILIZATION:-15}"
mkdir -p "${LOG_DIR}"

timestamp="$(date --iso-8601=seconds)"
gpu_row="$(nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | head -n 1)"
IFS=',' read -r gpu_index gpu_name memory_total memory_used utilization temperature <<<"${gpu_row}"
gpu_index="${gpu_index// /}"
memory_used="${memory_used// /}"
utilization="${utilization// /}"
temperature="${temperature// /}"
gpu_name="$(echo "${gpu_name}" | sed 's/^ *//;s/ *$//')"

compute_rows="$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null || true)"
compute_count="$(printf '%s\n' "${compute_rows}" | sed '/^[[:space:]]*$/d' | wc -l)"

status="idle"
reason="none"
exit_code=0
if (( compute_count > 0 )); then
  status="busy"
  reason="compute_process"
  exit_code=3
elif (( memory_used > MAX_IDLE_MEMORY_MIB )); then
  status="busy"
  reason="memory_threshold"
  exit_code=3
elif (( utilization > MAX_IDLE_UTILIZATION )); then
  status="busy"
  reason="utilization_threshold"
  exit_code=3
fi

processes_json="$(printf '%s\n' "${compute_rows}" | sed '/^[[:space:]]*$/d' | sed 's/\\/\\\\/g;s/"/\\"/g' | paste -sd ';' -)"
printf '{"timestamp":"%s","status":"%s","reason":"%s","gpu_index":%s,"gpu_name":"%s","memory_used_mib":%s,"memory_total_mib":%s,"utilization_percent":%s,"temperature_c":%s,"compute_processes":"%s"}\n' \
  "${timestamp}" "${status}" "${reason}" "${gpu_index}" "${gpu_name}" "${memory_used}" \
  "$(echo "${memory_total}" | tr -d ' ')" "${utilization}" "${temperature}" "${processes_json}" \
  | tee -a "${LOG_DIR}/checks.jsonl"

exit "${exit_code}"
