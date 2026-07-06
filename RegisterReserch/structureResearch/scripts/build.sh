#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cmake -S "${ROOT}/src" -B "${ROOT}/build" \
  "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH:-110}"
cmake --build "${ROOT}/build" --parallel
echo "Built ${ROOT}/build/register_bench"
echo "Built ${ROOT}/build/sass_register_bench"
echo "Built ${ROOT}/build/sass_template.sm_${CUDA_ARCH:-110}.cubin"
echo "Built ${ROOT}/build/sass_lop3_template.sm_${CUDA_ARCH:-110}.cubin"
echo "Built ${ROOT}/build/sass_lop3_wide_template.sm_${CUDA_ARCH:-110}.cubin"
echo "Built ${ROOT}/build/sass_imad_wide_template.sm_${CUDA_ARCH:-110}.cubin"
