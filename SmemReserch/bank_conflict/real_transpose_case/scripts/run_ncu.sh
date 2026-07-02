#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BIN="${ROOT}/build/real_transpose_bench"
RESULT_DIR="${ROOT}/results/ncu"

case_selector="${CASE:-all}"
width="${WIDTH:-4096}"
height="${HEIGHT:-4096}"
iters="${ITERS:-1}"
args=("$@")
for ((index = 0; index < ${#args[@]}; ++index)); do
  case "${args[index]}" in
    --case) case_selector="${args[index + 1]}" ;;
    --width) width="${args[index + 1]}" ;;
    --height) height="${args[index + 1]}" ;;
    --iters) iters="${args[index + 1]}" ;;
  esac
done

METRICS="${METRICS:-l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum}"

"${SCRIPT_DIR}/build.sh"
command -v ncu >/dev/null || { echo "ncu not found" >&2; exit 1; }
mkdir -p "${RESULT_DIR}"
rm -f "${RESULT_DIR}"/*.csv "${RESULT_DIR}"/*.png

status=0
mapfile -t cases < <(
  "${BIN}" --case "${case_selector}" --list-cases
)
for name in "${cases[@]}"; do
  ncu --force-overwrite --metrics "${METRICS}" --csv \
    --log-file "${RESULT_DIR}/${name}.csv" \
    "${BIN}" --case "${name}" --width "${width}" --height "${height}" \
    --iters "${iters}" --warmups 1 --repeats 1 ||
    status=1
done

python3 "${SCRIPT_DIR}/parse_ncu_results.py" || status=1
if [[ "${status}" -ne 0 ]]; then
  echo 'Some NCU runs failed. Metric names can vary by GPU/NCU version.' >&2
  echo 'Inspect with: ncu --query-metrics | grep -Ei "global|sector|request|bank_conflict"' >&2
fi
exit "${status}"
