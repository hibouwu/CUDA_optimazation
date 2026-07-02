#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
args=(-S "${ROOT}/src" -B "${ROOT}/build")
[[ -n "${CUDA_ARCH:-}" ]] &&
  args+=("-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}")
cmake "${args[@]}"
cmake --build "${ROOT}/build" --parallel
