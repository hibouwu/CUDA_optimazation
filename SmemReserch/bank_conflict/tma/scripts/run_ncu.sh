#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BIN="${ROOT}/build/tma_bench"
RESULT_DIR="${ROOT}/results/ncu"

case_selector="${CASE:-all}"
iters="${ITERS:-100}"
args=("$@")
for ((index = 0; index < ${#args[@]}; ++index)); do
  case "${args[index]}" in
    --case) case_selector="${args[index + 1]}" ;;
    --iters) iters="${args[index + 1]}" ;;
  esac
done

METRICS="${METRICS:-smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,l1tex__data_bank_reads.sum,l1tex__data_bank_writes.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__t_requests_pipe_lsu_mem_shared_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_shared_op_st.sum}"

"${SCRIPT_DIR}/build.sh"
command -v ncu >/dev/null || { echo "ncu not found" >&2; exit 1; }
mkdir -p "${RESULT_DIR}"
rm -f "${RESULT_DIR}"/*.csv

status=0
mapfile -t cases < <("${BIN}" --case "${case_selector}" --list-cases)
for name in "${cases[@]}"; do
  ncu --force-overwrite --metrics "${METRICS}" --csv \
    --log-file "${RESULT_DIR}/${name}.csv" \
    "${BIN}" --case "${name}" --iters "${iters}" --warmups 0 --repeats 1 ||
    status=1
done

if [[ "${status}" -ne 0 ]]; then
  echo 'Some NCU runs failed; metric names vary by GPU and NCU release.' >&2
  echo 'Query: ncu --query-metrics | grep -Ei "tma|tensor|shared|bank|mio"' >&2
fi
exit "${status}"
