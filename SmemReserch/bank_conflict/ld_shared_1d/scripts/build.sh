#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SOURCE_DIR="${BENCH_DIR}/src"
BUILD_DIR="${BENCH_DIR}/build"

cmake_args=(
  -S "${SOURCE_DIR}"
  -B "${BUILD_DIR}"
  "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH:-110}"
)

cmake "${cmake_args[@]}"
cmake --build "${BUILD_DIR}" --parallel
echo "Built ${BUILD_DIR}/smem_bank_bench"
