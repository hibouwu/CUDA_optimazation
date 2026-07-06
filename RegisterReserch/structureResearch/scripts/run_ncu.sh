#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RESULT_DIR="${ROOT}/results/ncu"
TARGET="${TARGET:-ptx}"
CASE_SELECTOR="${CASE:-all}"
ITERS="${ITERS:-1000}"
WARMUPS="${WARMUPS:-0}"
REPEATS="${REPEATS:-1}"

# Override with a comma-separated list if these metric names differ on your GPU:
# METRICS="metric_a,metric_b" ./scripts/run_ncu.sh
METRICS="${METRICS:-smsp__inst_executed.sum,smsp__inst_issued.sum,smsp__pipe_fu_core_active.avg,smsp__inst_executed_op_fp32.sum,smsp__inst_executed_op_integer.sum,smsp__inst_executed_op_logic.sum}"

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
if [[ "${USE_SUDO}" -eq 1 ]]; then
  sudo -n find "${RESULT_DIR}" -maxdepth 1 -type f \
      \( -name '*.csv' -o -name '*.ncu-rep' -o -name '*.png' \) -delete
else
  find "${RESULT_DIR}" -maxdepth 1 -type f \
      \( -name '*.csv' -o -name '*.ncu-rep' -o -name '*.png' \) -delete
fi

profile_ptx() {
  local case_name="$1"
  local output="${RESULT_DIR}/${case_name}.csv"
  local rc=0

  echo "Profiling PTX case ${case_name}"
  if [[ "${USE_SUDO}" -eq 1 ]]; then
    sudo -n rm -f "${output}"
  else
    rm -f "${output}"
  fi

  "${NCU_CMD[@]}" --check-exit-code off --target-processes all \
      --metrics "${METRICS}" --csv --log-file "${output}" \
      "${ROOT}/build/register_bench" --case "${case_name}" \
      --iters "${ITERS}" --warmups "${WARMUPS}" --repeats "${REPEATS}" || rc=$?

  if [[ "${USE_SUDO}" -eq 1 && -e "${output}" ]]; then
    sudo -n chown "$(id -u):$(id -g)" "${output}"
  fi

  if [[ "${rc}" -ne 0 ]]; then
    echo "Profiling ${case_name} failed. One or more metrics may not exist, or profiling may be blocked." >&2
    echo 'Inspect candidates with: ncu --query-metrics | grep -Ei "inst|logic|integer|fp32|pipe"' >&2
    return 1
  fi
}

profile_bank_scan() {
  local case_name="$1"
  local family="$2"
  local cubin="${ROOT}/results/${family}/${case_name}.cubin"
  local output="${RESULT_DIR}/${case_name}.csv"
  local rc=0

  if [[ ! -f "${cubin}" ]]; then
    echo "Missing ${cubin}; generating patched cubins first."
    if [[ "${family}" == "bank_scan" ]]; then
      python3 "${SCRIPT_DIR}/patch_main_scan.py" --family lop3
    else
      python3 "${SCRIPT_DIR}/patch_main_scan.py" --family ffma
    fi
  fi

  echo "Profiling ${family} case ${case_name}"
  if [[ "${USE_SUDO}" -eq 1 ]]; then
    sudo -n rm -f "${output}"
  else
    rm -f "${output}"
  fi

  "${NCU_CMD[@]}" --check-exit-code off --target-processes all \
      --metrics "${METRICS}" --csv --log-file "${output}" \
      "${ROOT}/build/sass_register_bench" \
      --iters "${ITERS}" --warmups "${WARMUPS}" --repeats "${REPEATS}" \
      --cubin "${cubin}" || rc=$?

  if [[ "${USE_SUDO}" -eq 1 && -e "${output}" ]]; then
    sudo -n chown "$(id -u):$(id -g)" "${output}"
  fi

  if [[ "${rc}" -ne 0 ]]; then
    echo "Profiling ${case_name} failed. One or more metrics may not exist, or profiling may be blocked." >&2
    echo 'Inspect candidates with: ncu --query-metrics | grep -Ei "inst|logic|integer|fp32|pipe|bank"' >&2
    return 1
  fi
}

status=0
profiled_cases=()
case "${TARGET}" in
  ptx)
    cases=(
      R0_imad_chain
      R1_imad_independent_x4
      R2_reuse_hot_x4
      R3_bank_dense_x4
      R4_bank_sparse_x4
    )
    if [[ "${CASE_SELECTOR}" != "all" ]]; then
      cases=("${CASE_SELECTOR}")
    fi
    for case_name in "${cases[@]}"; do
      profile_ptx "${case_name}" || status=1
      profiled_cases+=("${case_name}")
    done
    ;;
  bank_scan|bank_scan_ffma)
    family="${TARGET}"
    manifest="${ROOT}/results/${family}/manifest.csv"
    if [[ "${family}" == "bank_scan" ]]; then
      python3 "${SCRIPT_DIR}/patch_main_scan.py" --family lop3
    else
      python3 "${SCRIPT_DIR}/patch_main_scan.py" --family ffma
    fi
    mapfile -t all_cases < <(python3 - "${manifest}" <<'PY'
import csv
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
with manifest.open(newline="", encoding="utf-8") as stream:
    for row in csv.DictReader(stream):
        print(row["case"])
PY
)
    if [[ "${CASE_SELECTOR}" != "all" ]]; then
      all_cases=("${CASE_SELECTOR}")
    fi
    for case_name in "${all_cases[@]}"; do
      profile_bank_scan "${case_name}" "${family}" || status=1
      profiled_cases+=("${case_name}")
    done
    ;;
  *)
    echo "Unsupported TARGET='${TARGET}'. Use TARGET=ptx, TARGET=bank_scan or TARGET=bank_scan_ffma." >&2
    exit 1
    ;;
esac

python3 "${SCRIPT_DIR}/parse_ncu_results.py" "${profiled_cases[@]}" || status=1

exit "${status}"
