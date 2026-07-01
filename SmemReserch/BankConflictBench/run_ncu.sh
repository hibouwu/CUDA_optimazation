#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BIN="${SCRIPT_DIR}/../build/smem_bank_bench"
RESULT_DIR="${SCRIPT_DIR}/results/ncu"
ITERS="${ITERS:-100000}"

# Override with a comma-separated list if these metric names differ on your GPU:
# METRICS="metric_a,metric_b" ./run_ncu.sh
METRICS="${METRICS:-l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum,smsp__sass_average_branch_targets_threads_uniform.pct}"

if [[ ! -x "${BIN}" ]]; then
  echo "Missing ${BIN}; run ${SCRIPT_DIR}/build.sh first." >&2
  exit 1
fi
if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu is not on PATH." >&2
  exit 1
fi

mkdir -p "${RESULT_DIR}"
cases=(
  baseline
  same_bank_32way_2d
  broadcast
  multicast_hash
  v4_contiguous
  v2_multicast_pairs
  v4_multicast_quads
)

profile() {
  local name="$1"
  shift
  echo "Profiling ${name}"
  if ! ncu --metrics "${METRICS}" --csv --log-file "${RESULT_DIR}/${name}.csv" \
      "${BIN}" "$@"; then
    echo "Profiling ${name} failed. One or more metrics may not exist." >&2
    echo 'Inspect candidates with: ncu --query-metrics | grep -Ei "bank|shared|l1tex"' >&2
    return 1
  fi
}

status=0
for case_name in "${cases[@]}"; do
  profile "${case_name}" --case "${case_name}" --iters "${ITERS}" || status=1
done
for stride in 1 2 4 8 16 32; do
  profile "stride_${stride}" --case stride --stride "${stride}" --offset 0 \
    --iters "${ITERS}" || status=1
done
exit "${status}"

