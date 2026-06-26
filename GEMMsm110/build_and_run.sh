#!/usr/bin/env bash
set -euo pipefail

# Build and run GEMMsm110 benchmarks for Thor/SM110.
#
# Usage:
#   ./build_and_run.sh                    # build + run N=1024 all backends
#   ./build_and_run.sh 2048               # run N=2048
#   ./build_and_run.sh 2048 tc3           # run N=2048 tc3 only
#   ./build_and_run.sh build-only         # build only, don't run
#   ./build_and_run.sh clean              # remove build artifacts

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BIN="${BUILD_DIR}/gemm_sm110_bench"

ARG="${1:-1024}"

if [[ "${ARG}" == "clean" ]]; then
  rm -rf "${BUILD_DIR}"
  echo "Cleaned ${BUILD_DIR}"
  exit 0
fi

mkdir -p "${BUILD_DIR}"

echo "=== Building ==="
nvcc -O3 -std=c++17 \
  -DTC3_SM110_HOST_HAS_TCGEN05=1 \
  -gencode arch=compute_110a,code=sm_110a \
  -I"${SCRIPT_DIR}/include" \
  "${SCRIPT_DIR}/src/main.cu" \
  -lcublas -lcuda \
  -o "${BIN}"

echo "=== Build done ==="

if [[ "${ARG}" == "build-only" ]]; then
  exit 0
fi

N="${ARG}"
FILTER="${2:-all}"

echo "=== Running N=${N} filter=${FILTER} ==="
"${BIN}" "${N}" "${FILTER}"
