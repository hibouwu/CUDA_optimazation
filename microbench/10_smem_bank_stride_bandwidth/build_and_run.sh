#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
RESULT_DIR="${SCRIPT_DIR}/results"
NVCC="${NVCC:-nvcc}"

COMMON_FLAGS=(
  -O3
  -std=c++17
  -gencode arch=compute_110a,code=sm_110a
)

BIN="${BUILD_DIR}/smem_bank_stride_bandwidth"
CSV="${RESULT_DIR}/smem_bank_stride_bandwidth.csv"

build() {
  mkdir -p "${BUILD_DIR}"
  "${NVCC}" "${COMMON_FLAGS[@]}" "${SCRIPT_DIR}/smem_bank_stride_bandwidth.cu" -o "${BIN}"
}

collect_sass() {
  mkdir -p "${RESULT_DIR}"
  if command -v cuobjdump >/dev/null 2>&1; then
    cuobjdump --dump-sass "${BIN}" > "${RESULT_DIR}/smem_bank_stride_bandwidth.sass"
    grep -E 'Function : _Z|LDS|STS|CS2R|BAR' \
      "${RESULT_DIR}/smem_bank_stride_bandwidth.sass" | head -n 220 \
      > "${RESULT_DIR}/sass_summary.txt" || true
  fi
}

run_default() {
  mkdir -p "${RESULT_DIR}"
  local common_args=(
    --iters "${ITERS:-8192}"
    --warmup-iters "${WARMUP_ITERS:-128}"
    --threads "${THREADS:-256}"
    --shared-words "${SHARED_WORDS:-8192}"
    --blocks-per-sm "${BLOCKS_PER_SM:-1}"
  )
  "${BIN}" --csv-header | tee "${CSV}"
  local mode stride
  for mode in read write; do
    for stride in 1 2 4 8 16 32; do
      "${BIN}" --mode "${mode}" --stride-words "${stride}" "${common_args[@]}" --csv | tee -a "${CSV}"
    done
  done
  collect_sass
  echo "Wrote ${CSV}"
}

CMD="${1:-run}"
case "${CMD}" in
  clean)
    rm -rf "${BUILD_DIR}" "${RESULT_DIR}"
    ;;
  build-only)
    build
    collect_sass
    echo "Built ${BIN}"
    ;;
  run)
    build
    run_default
    ;;
  run-one)
    build
    shift || true
    "${BIN}" "$@"
    ;;
  validate)
    build
    shift || true
    "${SCRIPT_DIR}/scripts/smem_bank_validation.py" "$@"
    ;;
  ncu)
    build
    shift || true
    "${SCRIPT_DIR}/scripts/ncu_smem_bank_validation.py" "$@"
    ;;
  *)
    echo "Usage: $0 clean|build-only|run|run-one|validate|ncu"
    exit 2
    ;;
esac
