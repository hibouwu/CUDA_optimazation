#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${SCRIPT_DIR}/../src"
BUILD_DIR="${SCRIPT_DIR}/../build"

cmake_args=(-S "${SOURCE_DIR}" -B "${BUILD_DIR}")
if [[ -n "${CUDA_ARCH:-}" ]]; then
  cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}")
fi

cmake "${cmake_args[@]}"
cmake --build "${BUILD_DIR}" --parallel
echo "Built ${BUILD_DIR}/smem_bank_bench"

