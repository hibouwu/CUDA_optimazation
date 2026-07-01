#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ITERS="${ITERS:-100000}"
METRICS="${METRICS:-l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_shared_op_st.sum}"
"${SCRIPT_DIR}/build.sh"
command -v ncu >/dev/null || { echo "ncu not found" >&2; exit 1; }
mkdir -p "${ROOT}/results/ncu"
BIN="${ROOT}/build/st_shared_1d_bench"
cases=(baseline same_bank_32way_2d same_address v2_contiguous v4_contiguous v2_multicast_pairs v4_multicast_quads)
status=0
for name in "${cases[@]}"; do
  ncu --force-overwrite --metrics "${METRICS}" --csv --log-file "${ROOT}/results/ncu/${name}.csv" \
    "${BIN}" --case "${name}" --iters "${ITERS}" --warmups 0 --repeats 1 || status=1
done
for stride in 1 2 4 8 16 32; do
  ncu --force-overwrite --metrics "${METRICS}" --csv --log-file "${ROOT}/results/ncu/stride_${stride}.csv" \
    "${BIN}" --case stride --stride "${stride}" --iters "${ITERS}" --warmups 0 --repeats 1 || status=1
done
if [[ "${status}" -ne 0 ]]; then
  echo 'Query valid metrics: ncu --query-metrics | grep -Ei "bank|shared|l1tex"' >&2
fi
exit "${status}"
