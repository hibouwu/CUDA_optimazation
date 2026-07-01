#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ITERS="${ITERS:-100000}"
METRICS="${METRICS:-l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum}"
"${SCRIPT_DIR}/build.sh"
command -v ncu >/dev/null || { echo "ncu not found" >&2; exit 1; }
mkdir -p "${ROOT}/results/ncu"; status=0
for name in load_pitch32 load_pitch33 store_pitch32 store_pitch33; do
  ncu --force-overwrite --metrics "${METRICS}" --csv --log-file "${ROOT}/results/ncu/${name}.csv" \
    "${ROOT}/build/transpose_2d_bench" --case "${name}" --iters "${ITERS}" --warmups 0 --repeats 1 || status=1
done
[[ "${status}" -eq 0 ]] || echo 'Query: ncu --query-metrics | grep -Ei "bank|shared|l1tex"' >&2
exit "${status}"
