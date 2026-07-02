#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BIN="${BENCH_DIR}/build/smem_bank_bench"
RESULT_DIR="${BENCH_DIR}/results/ncu"
ITERS="${ITERS:-100000}"
WARMUPS="${WARMUPS:-0}"
REPEATS="${REPEATS:-1}"

# Override with a comma-separated list if these metric names differ on your GPU:
# METRICS="metric_a,metric_b" ./run_ncu.sh
METRICS="${METRICS:-l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,smsp__inst_executed_op_shared_ld.sum}"

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
    exit 1
  fi
  NCU_CMD=(sudo -n "$(readlink -f "${NCU_BIN}")")
  USE_SUDO=1
  echo "GPU performance counters are admin-only; profiling with sudo."
fi

mkdir -p "${RESULT_DIR}"
rm -f "${RESULT_DIR}"/*.csv "${RESULT_DIR}"/*.png
cases=(
  v0
  v1a
  v1b
  v1c
  v1d
  v1e
  v2
  v3
  v4a
  v4b
)

profile() {
  local name="$1"
  local output="${RESULT_DIR}/${name}.csv"
  local rc=0
  shift
  echo "Profiling ${name}"
  if [[ "${USE_SUDO}" -eq 1 ]]; then
    sudo -n rm -f "${output}"
  else
    rm -f "${output}"
  fi
  "${NCU_CMD[@]}" --metrics "${METRICS}" --csv --log-file "${output}" \
      "${BIN}" "$@" --warmups "${WARMUPS}" --repeats "${REPEATS}" || rc=$?
  if [[ "${USE_SUDO}" -eq 1 && -e "${output}" ]]; then
    sudo -n chown "$(id -u):$(id -g)" "${output}"
  fi
  if [[ "${rc}" -ne 0 ]]; then
    echo "Profiling ${name} failed. One or more metrics may not exist." >&2
    echo 'Inspect candidates with: ncu --query-metrics | grep -Ei "bank|shared|l1tex"' >&2
    return 1
  fi
}

status=0
for case_name in "${cases[@]}"; do
  profile "${case_name}" --case "${case_name}" --iters "${ITERS}" || status=1
done
python3 "${SCRIPT_DIR}/parse_ncu_results.py" || status=1
exit "${status}"
