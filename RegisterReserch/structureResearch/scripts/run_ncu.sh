#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BIN="${ROOT}/build/register_bench"
RESULT_DIR="${ROOT}/results/ncu"
ITERS="${ITERS:-10000}"
case_selector="${CASE:-all}"
args=("$@")

for ((index = 0; index < ${#args[@]}; ++index)); do
  case "${args[index]}" in
    --case) case_selector="${args[index + 1]}" ;;
    --iters) ITERS="${args[index + 1]}" ;;
  esac
done

METRICS="${METRICS:-gpu__time_duration.sum,smsp__cycles_active.avg,smsp__inst_executed.sum,smsp__issue_active.avg,smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct}"

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
    echo "GPU performance counters require administrator access." >&2
    exit 1
  fi
  NCU_CMD=(sudo -n "$(readlink -f "${NCU_BIN}")")
  USE_SUDO=1
fi

mkdir -p "${RESULT_DIR}"
rm -f "${RESULT_DIR}"/*.csv "${RESULT_DIR}"/*.png
mapfile -t cases < <("${BIN}" --case "${case_selector}" --list-cases)

status=0
for case_name in "${cases[@]}"; do
  output="${RESULT_DIR}/${case_name}.csv"
  echo "Profiling ${case_name}"
  profiled=0
  for attempt in 1 2 3; do
    if [[ "${USE_SUDO}" -eq 1 ]]; then
      sudo -n rm -f "${output}"
    else
      rm -f "${output}"
    fi
    rc=0
    "${NCU_CMD[@]}" --check-exit-code off --metrics "${METRICS}" --csv \
      --log-file "${output}" "${BIN}" --quiet --case "${case_name}" \
      --iters "${ITERS}" --warmups 0 --repeats 1 || rc=$?
    if [[ "${rc}" -eq 0 ]] &&
        ! grep -q '^==ERROR==' "${output}" 2>/dev/null; then
      profiled=1
      break
    fi
    echo "Retrying ${case_name} (${attempt}/3)" >&2
    sleep 1
  done
  if [[ "${USE_SUDO}" -eq 1 && -e "${output}" ]]; then
    sudo -n chown "$(id -u):$(id -g)" "${output}"
  fi
  if [[ "${profiled}" -ne 1 ]]; then
    echo "Profiling ${case_name} failed after 3 attempts." >&2
    status=1
  fi
done

python3 "${SCRIPT_DIR}/parse_ncu_results.py" || status=1
exit "${status}"
