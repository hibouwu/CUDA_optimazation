#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BIN="${ROOT}/build/real_transpose_bench"
RESULT_DIR="${ROOT}/results/ncu"

case_selector="${CASE:-all}"
width="${WIDTH:-4096}"
height="${HEIGHT:-4096}"
iters="${ITERS:-1}"
args=("$@")
for ((index = 0; index < ${#args[@]}; ++index)); do
  case "${args[index]}" in
    --case) case_selector="${args[index + 1]}" ;;
    --width) width="${args[index + 1]}" ;;
    --height) height="${args[index + 1]}" ;;
    --iters) iters="${args[index + 1]}" ;;
  esac
done

METRICS="${METRICS:-l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum}"

"${SCRIPT_DIR}/build.sh"
NCU_BIN="$(command -v ncu || true)"
if [[ -z "${NCU_BIN}" ]]; then
  echo "ncu not found" >&2
  exit 1
fi

NCU_CMD=("${NCU_BIN}")
USE_SUDO=0
if [[ "${EUID}" -ne 0 ]] &&
    grep -q '^RmProfilingAdminOnly: 1$' /proc/driver/nvidia/params 2>/dev/null; then
  if ! sudo -n true 2>/dev/null; then
    echo "GPU performance counters are restricted to administrators." >&2
    echo "Run this script as root or ask an administrator to enable profiling counters." >&2
    exit 1
  fi
  NCU_CMD=(sudo -n "$(readlink -f "${NCU_BIN}")")
  USE_SUDO=1
  echo "GPU performance counters are admin-only; profiling with sudo."
fi

mkdir -p "${RESULT_DIR}"
if [[ "${USE_SUDO}" -eq 1 ]]; then
  sudo -n rm -f "${RESULT_DIR}"/*.csv "${RESULT_DIR}"/*.png
else
  rm -f "${RESULT_DIR}"/*.csv "${RESULT_DIR}"/*.png
fi

status=0
mapfile -t cases < <(
  "${BIN}" --case "${case_selector}" --list-cases
)
for name in "${cases[@]}"; do
  output="${RESULT_DIR}/${name}.csv"
  rc=0
  echo "Profiling ${name}"
  "${NCU_CMD[@]}" --force-overwrite --metrics "${METRICS}" --csv \
    --log-file "${output}" \
    "${BIN}" --case "${name}" --width "${width}" --height "${height}" \
    --iters "${iters}" --warmups 0 --repeats 1 ||
    rc=$?
  if [[ "${USE_SUDO}" -eq 1 && -e "${output}" ]]; then
    sudo -n chown "$(id -u):$(id -g)" "${output}"
  fi
  if [[ "${rc}" -ne 0 ]]; then
    status=1
  fi
done

python3 "${SCRIPT_DIR}/parse_ncu_results.py" || status=1
if [[ "${status}" -ne 0 ]]; then
  echo 'Some NCU runs failed. Metric names can vary by GPU/NCU version.' >&2
  echo 'Inspect with: ncu --query-metrics | grep -Ei "global|sector|request|bank_conflict"' >&2
fi
exit "${status}"
