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
  -lcuda
)

# Optional local Fedora/CUDA-header compatibility controls. They are omitted on
# Thor unless explicitly requested, and the unified campaign records them.
if [[ -n "${NVCC_HOST_COMPILER:-}" ]]; then
  COMMON_FLAGS+=( -ccbin "${NVCC_HOST_COMPILER}" )
fi
if [[ "${NVCC_HOST_UNDEF_GNU_SOURCE:-0}" == 1 ]]; then
  COMMON_FLAGS+=(
    -Xcompiler=-U_GNU_SOURCE
    -D_DEFAULT_SOURCE
    -D_POSIX_C_SOURCE=200809L
    -D_XOPEN_SOURCE=700
    -D_XOPEN_SOURCE_EXTENDED=1
    -D_LARGEFILE64_SOURCE=1
    -D_ATFILE_SOURCE=1
  )
fi

BIN="${BUILD_DIR}/tma_gmem_smem_bandwidth"
CSV="${RESULT_DIR}/tma_gmem_smem_bandwidth.csv"

build() {
  mkdir -p "${BUILD_DIR}"
  "${NVCC}" "${COMMON_FLAGS[@]}" "${SCRIPT_DIR}/tma_gmem_smem_bandwidth.cu" -o "${BIN}"
}

collect_sass() {
  mkdir -p "${RESULT_DIR}"
  if command -v cuobjdump >/dev/null 2>&1; then
    cuobjdump --dump-sass "${BIN}" > "${RESULT_DIR}/tma_gmem_smem_bandwidth.sass"
    grep -E 'Function : _Z|CP_ASYNC|CPT|TMA|LDGSTS|MEMBAR|MBARRIER|CS2R' \
      "${RESULT_DIR}/tma_gmem_smem_bandwidth.sass" | head -n 260 \
      > "${RESULT_DIR}/sass_summary.txt" || true
  fi
}

run_default() {
  mkdir -p "${RESULT_DIR}"
  local common_args=(
    --iters "${ITERS:-4096}"
    --warmup-iters "${WARMUP_ITERS:-32}"
    --blocks-per-sm "${BLOCKS_PER_SM:-1}"
    --threads "${THREADS:-128}"
    --tile-bytes "${TILE_BYTES:-32768}"
    --slots "${SLOTS:-4}"
    --inflight "${INFLIGHT:-1}"
  )
  "${BIN}" --csv-header | tee "${CSV}"
  "${BIN}" --mode l2-hit --bytes "${L2_BYTES:-16777216}" "${common_args[@]}" --csv | tee -a "${CSV}"
  "${BIN}" --mode dram-stream --bytes "${DRAM_BYTES:-268435456}" "${common_args[@]}" --csv | tee -a "${CSV}"
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
    "${SCRIPT_DIR}/scripts/tma_validation.py" "$@"
    ;;
  ncu)
    build
    shift || true
    "${SCRIPT_DIR}/scripts/ncu_tma_validation.py" "$@"
    ;;
  *)
    echo "Usage: $0 clean|build-only|run|run-one|validate|ncu"
    exit 2
    ;;
esac
