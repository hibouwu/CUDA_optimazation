#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BINARY="${BUILD_DIR}/requant_epilogue_benchmark"
CSV="${BUILD_DIR}/requant_epilogue_benchmark.csv"

NVCC="${NVCC:-nvcc}"
WARMUP="${WARMUP:-10}"
ITERATIONS="${ITERATIONS:-100}"
SEED="${SEED:-1234}"

mkdir -p "${BUILD_DIR}"

"${NVCC}" \
  -O3 \
  -std=c++17 \
  -Xcompiler=-U_GNU_SOURCE \
  -D_DEFAULT_SOURCE \
  -D_POSIX_C_SOURCE=200809L \
  -D_XOPEN_SOURCE=700 \
  -D_XOPEN_SOURCE_EXTENDED=1 \
  -D_LARGEFILE64_SOURCE=1 \
  -D_ATFILE_SOURCE=1 \
  -gencode arch=compute_110a,code=sm_110a \
  -I"${PROJECT_DIR}/include" \
  "${SCRIPT_DIR}/requant_epilogue_benchmark.cu" \
  -o "${BINARY}"

rm -f "${CSV}"

distributions=(
  uniform
  normal
  laplace
  outlier
  lognormal
  constant
)

shapes=(
  "256 256"
  "1024 1024"
  "4096 1024"
)

for shape in "${shapes[@]}"; do
  read -r rows cols <<<"${shape}"
  for distribution in "${distributions[@]}"; do
    echo "=== ${rows}x${cols} ${distribution} ==="
    "${BINARY}" \
      --rows "${rows}" \
      --cols "${cols}" \
      --distribution "${distribution}" \
      --seed "${SEED}" \
      --warmup "${WARMUP}" \
      --iterations "${ITERATIONS}" \
      --csv "${CSV}"
  done
done

echo "Results: ${CSV}"
