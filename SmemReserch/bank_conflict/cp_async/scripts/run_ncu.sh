#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ITERS="${ITERS:-10000}"
METRICS="${METRICS:-l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ldgsts.sum,l1tex__t_sectors_pipe_lsu_mem_shared_op_ldgsts.sum,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct}"
"${SCRIPT_DIR}/build.sh"
command -v ncu >/dev/null || { echo "ncu not found" >&2; exit 1; }
mkdir -p "${ROOT}/results/ncu"
BIN="${ROOT}/build/cp_async_bench"; status=0
for name in contiguous source_broadcast; do
  ncu --force-overwrite --metrics "${METRICS}" --csv --log-file "${ROOT}/results/ncu/${name}.csv" \
    "${BIN}" --case "${name}" --iters "${ITERS}" --warmups 0 --repeats 1 || status=1
done
for stride in 4 8 16 32; do
  ncu --force-overwrite --metrics "${METRICS}" --csv --log-file "${ROOT}/results/ncu/stride_${stride}.csv" \
    "${BIN}" --case stride --stride-words "${stride}" --iters "${ITERS}" --warmups 0 --repeats 1 || status=1
done
[[ "${status}" -eq 0 ]] || echo 'Query: ncu --query-metrics | grep -Ei "ldgsts|async|shared|mio"' >&2
exit "${status}"
