#!/usr/bin/env bash
set -euo pipefail

# Build and run GEMMsm110 programs for NVIDIA Thor / SM110 / sm_110a.
#
# Usage:
#   ./build_and_run.sh clean
#   ./build_and_run.sh build-only
#   ./build_and_run.sh sanity
#   ./build_and_run.sh tc3-minimal
#   ./build_and_run.sh 1024 cublas_tc
#   ./build_and_run.sh 1024 cutlass
#   ./build_and_run.sh 1024 tc3
#   ./build_and_run.sh 1024 tc5a
#   ./build_and_run.sh 1024 tc5b

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BENCH_BIN="${BUILD_DIR}/gemm_sm110_bench"
SANITY_BIN="${BUILD_DIR}/runtime_sanity"
TC3_MINIMAL_BIN="${BUILD_DIR}/tc3_minimal"
BACKEND_TIMEOUT_SECONDS="${3:-${BACKEND_TIMEOUT_SECONDS:-30}}"
BACKEND_KILL_GRACE_SECONDS="${BACKEND_KILL_GRACE_SECONDS:-5}"
CUTLASS_ROOT="${CUTLASS_ROOT:-${SCRIPT_DIR}/../../third_party/cutlass}"

NVCC="${NVCC:-nvcc}"
COMMON_FLAGS=(
  -O3
  -std=c++17
  --expt-relaxed-constexpr
  -diag-suppress=20012
  -diag-suppress=20013
  -diag-suppress=20015
  -DTC3_SM110_HOST_HAS_TCGEN05=1
  -gencode arch=compute_110a,code=sm_110a
  -I"${SCRIPT_DIR}/include"
  -I"${CUTLASS_ROOT}/include"
  -I"${CUTLASS_ROOT}/tools/util/include"
)

build_sanity() {
  "${NVCC}" "${COMMON_FLAGS[@]}" \
    "${SCRIPT_DIR}/src/runtime_sanity.cu" \
    -o "${SANITY_BIN}"
}

build_tc3_minimal() {
  "${NVCC}" "${COMMON_FLAGS[@]}" \
    "${SCRIPT_DIR}/src/tc3_minimal.cu" \
    -o "${TC3_MINIMAL_BIN}"
}

build_benchmark() {
  if [[ ! -f "${CUTLASS_ROOT}/include/cutlass/cutlass.h" ]]; then
    echo "CUTLASS headers not found under ${CUTLASS_ROOT}" >&2
    echo "Set CUTLASS_ROOT to a CUTLASS 4.5.2 checkout." >&2
    exit 2
  fi
  "${NVCC}" "${COMMON_FLAGS[@]}" \
    "${SCRIPT_DIR}/src/main.cu" \
    -lcublas \
    -o "${BENCH_BIN}"
}

run_backend_with_timeout() {
  local n="$1"
  local backend="$2"
  local status

  echo "=== Running N=${n} backend=${backend} timeout=${BACKEND_TIMEOUT_SECONDS}s ==="
  set +e
  timeout --foreground --signal=TERM \
    --kill-after="${BACKEND_KILL_GRACE_SECONDS}s" \
    "${BACKEND_TIMEOUT_SECONDS}s" \
    "${BENCH_BIN}" "${n}" "${backend}"
  status=$?
  set -e

  if [[ "${status}" -eq 124 || "${status}" -eq 137 ]]; then
    echo "=== Backend ${backend} timed out after ${BACKEND_TIMEOUT_SECONDS}s and was stopped ===" >&2
    return 124
  fi
  return "${status}"
}

usage() {
  cat <<EOF
Usage:
  $0 clean
  $0 build-only
  $0 sanity
  $0 tc3-minimal
  $0 [N] [all|cublas_tc|cutlass|tc3|tc4|tc5|tc5a|tc5b] [timeout_seconds]
EOF
}

ARG="${1:-1024}"

if [[ "${ARG}" == "clean" ]]; then
  rm -rf "${BUILD_DIR}"
  echo "Cleaned ${BUILD_DIR}"
  exit 0
fi

mkdir -p "${BUILD_DIR}"

case "${ARG}" in
  build-only)
    echo "=== Building benchmark ==="
    build_benchmark
    echo "=== Build done ==="
    ;;
  sanity)
    echo "=== Building runtime sanity ==="
    build_sanity
    echo "=== Running runtime sanity ==="
    "${SANITY_BIN}"
    ;;
  tc3-minimal)
    echo "=== Building tc3 minimal probe ==="
    build_tc3_minimal
    echo "=== Running tc3 minimal probe ==="
    "${TC3_MINIMAL_BIN}"
    ;;
  --help|-h)
    usage
    ;;
  *)
    N="${ARG}"
    if [[ ! "${BACKEND_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
      echo "Invalid backend timeout: ${BACKEND_TIMEOUT_SECONDS} (expected positive integer seconds)" >&2
      exit 2
    fi
    if [[ ! "${BACKEND_KILL_GRACE_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
      echo "Invalid backend kill grace: ${BACKEND_KILL_GRACE_SECONDS} (expected positive integer seconds)" >&2
      exit 2
    fi
    FILTER="${2:-all}"
    echo "=== Building benchmark ==="
    build_benchmark
    echo "Backend timeout: ${BACKEND_TIMEOUT_SECONDS}s (SIGKILL grace: ${BACKEND_KILL_GRACE_SECONDS}s)"
    run_backend_with_timeout "${N}" "${FILTER}"
    ;;
esac
