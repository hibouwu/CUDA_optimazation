#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BIN="${ROOT}/build/tma_bench"
RESULT_DIR="${ROOT}/results/ncu"
ITERS="${ITERS:-100}"
WARMUPS="${WARMUPS:-0}"
REPEATS="${REPEATS:-1}"
case_selector="${CASE:-all}"
args=("$@")
for ((index = 0; index < ${#args[@]}; ++index)); do
  case "${args[index]}" in
    --case) case_selector="${args[index + 1]}" ;;
    --iters) ITERS="${args[index + 1]}" ;;
    --warmups) WARMUPS="${args[index + 1]}" ;;
    --repeats) REPEATS="${args[index + 1]}" ;;
  esac
done

METRICS="${METRICS:-smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,l1tex__data_bank_reads.sum,l1tex__data_bank_writes.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__t_requests_pipe_lsu_mem_shared_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_shared_op_st.sum}"

echo "Building benchmark"
"${SCRIPT_DIR}/build.sh"
NCU_BIN="$(command -v ncu || true)"
if [[ -z "${NCU_BIN}" ]]; then
  echo "ncu is not on PATH." >&2
  exit 1
fi

NCU_CMD=("${NCU_BIN}")
USE_SUDO=0
if [[ "${EUID}" -ne 0 ]] &&
    grep -q '^RmProfilingAdminOnly: 1$' /proc/driver/nvidia/params 2>/dev/null; then
  if ! sudo -n true 2>/dev/null; then
    echo "GPU performance counters are restricted to administrators." >&2
    echo "Run this script as root or ask an administrator to enable profiling counters." >&2
    echo "Reference: https://developer.nvidia.com/ERR_NVGPUCTRPERM" >&2
    exit 1
  fi
  NCU_CMD=(sudo -n "$(readlink -f "${NCU_BIN}")")
  USE_SUDO=1
  echo "GPU performance counters are admin-only; profiling with sudo."
fi

mkdir -p "${RESULT_DIR}"
rm -f "${RESULT_DIR}"/*.csv "${RESULT_DIR}"/*.png

mapfile -t cases < <("${BIN}" --case "${case_selector}" --list-cases)
if [[ "${#cases[@]}" -eq 0 ]]; then
  echo "No cases matched selector '${case_selector}'." >&2
  exit 1
fi

profile() {
  local name="$1"
  local output="${RESULT_DIR}/${name}.csv"
  local rc=0

  echo "Profiling ${name}"
  if [[ "${USE_SUDO}" -eq 1 ]]; then
    sudo -n rm -f "${output}"
  else
    rm -f "${output}"
  fi

  "${NCU_CMD[@]}" --check-exit-code off --metrics "${METRICS}" --csv \
      --log-file "${output}" \
      "${BIN}" --quiet --case "${name}" --iters "${ITERS}" \
      --warmups "${WARMUPS}" --repeats "${REPEATS}" || rc=$?

  if [[ "${USE_SUDO}" -eq 1 && -e "${output}" ]]; then
    sudo -n chown "$(id -u):$(id -g)" "${output}"
  fi

  if [[ "${rc}" -ne 0 ]]; then
    echo "Profiling ${name} failed. One or more metrics may not exist, or profiling may be blocked." >&2
    echo 'Inspect candidates with: ncu --query-metrics | grep -Ei "tma|tensor|shared|bank|mio"' >&2
    return 1
  fi
}

status=0
for case_name in "${cases[@]}"; do
  profile "${case_name}" || status=1
done

python3 "${SCRIPT_DIR}/parse_ncu_results.py" || status=1
exit "${status}"
