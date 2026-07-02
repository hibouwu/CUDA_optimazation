#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BIN="${ROOT}/build/transpose_2d_bench"
RESULT_DIR="${ROOT}/results/ncu"
ITERS="${ITERS:-100000}"
METRICS="${METRICS:-l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_shared_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum,smsp__sass_inst_executed_op_shared_ld_pred_on.sum}"

case_selector="all"
args=("$@")
for ((i = 0; i < ${#args[@]}; ++i)); do
  if [[ "${args[i]}" == "--case" && $((i + 1)) -lt ${#args[@]} ]]; then
    case_selector="${args[i + 1]}"
  fi
done

"${SCRIPT_DIR}/build.sh"
command -v ncu >/dev/null || { echo "ncu not found" >&2; exit 1; }
mkdir -p "${RESULT_DIR}"

status=0
mapfile -t cases < <("${BIN}" --case "${case_selector}" --list-cases)
for name in "${cases[@]}"; do
  ncu --force-overwrite --metrics "${METRICS}" --csv \
    --log-file "${RESULT_DIR}/${name}.csv" \
    "${BIN}" --case "${name}" --iters "${ITERS}" --warmups 0 --repeats 1 ||
    status=1
done

if [[ "${status}" -ne 0 ]]; then
  echo 'Query: ncu --query-metrics | grep -Ei "bank|shared|l1tex|sass_inst_executed"' >&2
fi
exit "${status}"
