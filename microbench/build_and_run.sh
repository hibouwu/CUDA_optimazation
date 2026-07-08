#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
NVCC="${NVCC:-nvcc}"

COMMON_FLAGS=(
  -O3
  -std=c++17
  -DTC3_SM110_HOST_HAS_TCGEN05=1
  -gencode arch=compute_110a,code=sm_110a
  -I"${SCRIPT_DIR}/common"
)

SANITY_BIN="${BUILD_DIR}/sm110_runtime_sanity"
TCGEN05_BIN="${BUILD_DIR}/sm110_tcgen05_tmem_probe"
CLC_BIN="${BUILD_DIR}/sm110_clc_persistent_tmem_probe"

build_sanity() {
  "${NVCC}" "${COMMON_FLAGS[@]}" \
    "${SCRIPT_DIR}/00_runtime_sanity/demo.cu" \
    -o "${SANITY_BIN}"
}

build_tcgen05() {
  "${NVCC}" "${COMMON_FLAGS[@]}" \
    "${SCRIPT_DIR}/01_tcgen05_tmem_probe/demo.cu" \
    -o "${TCGEN05_BIN}"
}

build_clc() {
  "${NVCC}" "${COMMON_FLAGS[@]}" \
    "${SCRIPT_DIR}/02_clc_persistent_tmem_probe/demo.cu" \
    -o "${CLC_BIN}"
}

usage() {
  cat <<EOF
Usage:
  $0 clean
  $0 build-only
  $0 sanity
  $0 tcgen05
  $0 clc [tiles] [workers_per_sm]
  $0 all
EOF
}

mkdir -p "${BUILD_DIR}"
CMD="${1:-all}"

case "${CMD}" in
  clean)
    rm -rf "${BUILD_DIR}"
    echo "Cleaned ${BUILD_DIR}"
    ;;
  build-only)
    build_sanity
    build_tcgen05
    build_clc
    echo "Built all SM110 Thor component demos into ${BUILD_DIR}"
    ;;
  sanity)
    build_sanity
    "${SANITY_BIN}"
    ;;
  tcgen05)
    build_tcgen05
    "${TCGEN05_BIN}"
    ;;
  clc)
    build_clc
    "${CLC_BIN}" "${2:-128}" "${3:-1}"
    ;;
  all)
    build_sanity
    build_tcgen05
    build_clc
    "${SANITY_BIN}"
    "${TCGEN05_BIN}"
    "${CLC_BIN}" "${2:-128}" "${3:-1}"
    ;;
  --help|-h)
    usage
    ;;
  *)
    usage
    exit 1
    ;;
esac

