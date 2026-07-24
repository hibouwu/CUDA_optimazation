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

BIN="${BUILD_DIR}/dsmem_topology_contention"
CSV="${RESULT_DIR}/dsmem_topology_contention.csv"

build() {
  mkdir -p "${BUILD_DIR}"
  "${NVCC}" "${COMMON_FLAGS[@]}" "${SCRIPT_DIR}/dsmem_topology_contention.cu" -o "${BIN}"
}

collect_sass() {
  mkdir -p "${RESULT_DIR}"
  if command -v cuobjdump >/dev/null 2>&1; then
    cuobjdump --dump-sass "${BIN}" > "${RESULT_DIR}/dsmem_topology_contention.sass"
    grep -E 'Function : _Z|LD\.E\.128|ST\.E\.128|LDS|STS|BAR|MEMBAR|CS2R' \
      "${RESULT_DIR}/dsmem_topology_contention.sass" | head -n 260 \
      > "${RESULT_DIR}/sass_summary.txt" || true
  fi
}

run_default() {
  mkdir -p "${RESULT_DIR}"
  local common_args=(
    --iters "${ITERS:-4096}"
    --warmup-iters "${WARMUP_ITERS:-64}"
    --threads "${THREADS:-256}"
    --cluster-size "${CLUSTER_SIZE:-4}"
    --shared-bytes "${SHARED_BYTES:-65536}"
  )
  "${BIN}" --csv-header | tee "${CSV}"
  local mode
  for mode in ring-read-d1 ring-read-d2 fanin-read-root0 ring-write-d1 ring-write-d2 fanin-write-root0; do
    "${BIN}" --mode "${mode}" "${common_args[@]}" --csv | tee -a "${CSV}"
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
    "${SCRIPT_DIR}/scripts/dsmem_topology_validation.py" "$@"
    ;;
  ncu)
    build
    shift || true
    "${SCRIPT_DIR}/scripts/ncu_dsmem_topology_validation.py" "$@"
    ;;
  *)
    echo "Usage: $0 clean|build-only|run|run-one|validate|ncu"
    exit 2
    ;;
esac
