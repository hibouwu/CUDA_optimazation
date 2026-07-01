#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ITERS="${ITERS:-10000}"
METRICS="${METRICS:-smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,l1tex__data_bank_reads.sum,l1tex__data_bank_writes.sum}"
"${SCRIPT_DIR}/build.sh"
command -v ncu >/dev/null || { echo "ncu not found" >&2; exit 1; }
mkdir -p "${ROOT}/results/ncu"; status=0
for name in swizzle_none swizzle_32b swizzle_64b swizzle_128b; do
  ncu --force-overwrite --metrics "${METRICS}" --csv --log-file "${ROOT}/results/ncu/${name}.csv" \
    "${ROOT}/build/tma_bench" --case "${name}" --iters "${ITERS}" --warmups 0 --repeats 1 || status=1
done
[[ "${status}" -eq 0 ]] || echo 'Query: ncu --query-metrics | grep -Ei "tma|tensor|shared|mio"' >&2
exit "${status}"
