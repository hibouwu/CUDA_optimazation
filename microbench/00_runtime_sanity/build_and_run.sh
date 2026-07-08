#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MICROBENCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
NVCC="${NVCC:-nvcc}"

COMMON_FLAGS=(
  -O3
  -std=c++17
  -DTC3_SM110_HOST_HAS_TCGEN05=1
  -gencode arch=compute_110a,code=sm_110a
  -I"${MICROBENCH_DIR}/common"
)

BIN="${BUILD_DIR}/sm110_runtime_sanity"

usage() {
  cat <<EOF
Usage:
  $0 clean
  $0 build-only
  $0 run
EOF
}

build() {
  mkdir -p "${BUILD_DIR}"
  "${NVCC}" "${COMMON_FLAGS[@]}" "${SCRIPT_DIR}/demo.cu" -o "${BIN}"
}

CMD="${1:-run}"

case "${CMD}" in
  clean)
    rm -rf "${BUILD_DIR}"
    echo "Cleaned ${BUILD_DIR}"
    ;;
  build-only)
    build
    echo "Built ${BIN}"
    ;;
  run)
    build
    "${BIN}"
    ;;
  --help|-h)
    usage
    ;;
  *)
    usage
    exit 1
    ;;
esac
