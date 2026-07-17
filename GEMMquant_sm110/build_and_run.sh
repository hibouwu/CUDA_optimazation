#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BENCH_BIN="${BUILD_DIR}/quant_gemm_sm110_bench"
FP4_PROBE_BIN="${BUILD_DIR}/fp4_cublaslt_probe"
NVFP4_CUTLASS_BIN="${BUILD_DIR}/cutlass_72b_nvfp4_nvfp4_sm110"
MXFP4_CUTLASS_BIN="${BUILD_DIR}/cutlass_72a_mxfp4_bf16_sm110"
GENERATED_DIR="${BUILD_DIR}/generated"
CUTLASS_FP4_ROOT="${CUTLASS_FP4_ROOT:-/xplorer/op630/_deps/flash-attention/csrc/cutlass}"
NVCC="${NVCC:-nvcc}"

COMMON_FLAGS=(
  -O3
  -std=c++17
  --expt-relaxed-constexpr
  -gencode arch=compute_110a,code=sm_110a
)

build_benchmark() {
  mkdir -p "${BUILD_DIR}"
  "${NVCC}" "${COMMON_FLAGS[@]}" \
    "${SCRIPT_DIR}/src/quant_gemm_bench.cu" \
    -lcublas \
    -lcublasLt \
    -o "${BENCH_BIN}"
}

build_fp4_probe() {
  mkdir -p "${BUILD_DIR}"
  "${NVCC}" "${COMMON_FLAGS[@]}" \
    "${SCRIPT_DIR}/src/fp4_cublaslt_probe.cu" \
    -lcublasLt \
    -o "${FP4_PROBE_BIN}"
}

build_nvfp4_cutlass_72b() {
  local source="${CUTLASS_FP4_ROOT}/examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu"
  local generated_source="${GENERATED_DIR}/cutlass_72b_nvfp4_nvfp4_sm110.cu"
  local generated_float8="${GENERATED_DIR}/cutlass/float8.h"

  if [[ ! -f "${source}" ]]; then
    echo "CUTLASS FP4 source not found: ${source}" >&2
    echo "Set CUTLASS_FP4_ROOT to a CUTLASS 4.3.4+ checkout with example 72b." >&2
    exit 2
  fi
  if [[ ! -f "${CUTLASS_FP4_ROOT}/include/cutlass/float8.h" ]]; then
    echo "CUTLASS float8.h not found under ${CUTLASS_FP4_ROOT}" >&2
    exit 2
  fi

  mkdir -p "${BUILD_DIR}" "${GENERATED_DIR}/cutlass"

  perl -0pe 's/if \(props\.major != 10 \|\| \(props\.minor != 0 && props\.minor != 1 && props\.minor != 3\)\) \{\n    std::cerr << "This example requires a GPU with compute capability 100a\|f, 101a\|f, or 103a\|f\)\." << std::endl;\n    return 0;\n  \}/if (!((props.major == 10 && (props.minor == 0 || props.minor == 1 || props.minor == 3)) || (props.major == 11 && props.minor == 0))) {\n    std::cerr << "This example requires a Blackwell GPU with compute capability 100, 101, 103, or 110." << std::endl;\n    return 0;\n  \}/' \
    "${source}" > "${generated_source}"

  perl -0pe 's/#if \(defined\(CUTLASS_ARCH_MMA_SM100A_ENABLED\)/#if defined(__CUDA_ARCH__) \&\& (defined(CUTLASS_ARCH_MMA_SM100A_ENABLED)/g; s/#if \(defined\(CUTLASS_ARCH_MMA_SM100F_ENABLED\)/#if defined(__CUDA_ARCH__) \&\& (defined(CUTLASS_ARCH_MMA_SM100F_ENABLED)/g' \
    "${CUTLASS_FP4_ROOT}/include/cutlass/float8.h" > "${generated_float8}"

  "${NVCC}" "${COMMON_FLAGS[@]}" \
    -diag-suppress=20012 \
    -diag-suppress=20013 \
    -diag-suppress=20015 \
    -DCUTLASS_ARCH_MMA_SM110A_ENABLED=1 \
    -DCUTLASS_ENABLE_GDC_FOR_SM100=1 \
    -I"${GENERATED_DIR}" \
    -I"${SCRIPT_DIR}/include" \
    -I"${CUTLASS_FP4_ROOT}/include" \
    -I"${CUTLASS_FP4_ROOT}/tools/util/include" \
    -I"${CUTLASS_FP4_ROOT}/examples/common" \
    "${generated_source}" \
    -lcuda \
    -o "${NVFP4_CUTLASS_BIN}"
}

build_mxfp4_cutlass_72a() {
  local source="${CUTLASS_FP4_ROOT}/examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm.cu"
  local generated_source="${GENERATED_DIR}/cutlass_72a_mxfp4_bf16_sm110.cu"
  local generated_float8="${GENERATED_DIR}/cutlass/float8.h"

  if [[ ! -f "${source}" ]]; then
    echo "CUTLASS FP4 source not found: ${source}" >&2
    echo "Set CUTLASS_FP4_ROOT to a CUTLASS 4.3.4+ checkout with example 72a." >&2
    exit 2
  fi
  if [[ ! -f "${CUTLASS_FP4_ROOT}/include/cutlass/float8.h" ]]; then
    echo "CUTLASS float8.h not found under ${CUTLASS_FP4_ROOT}" >&2
    exit 2
  fi

  mkdir -p "${BUILD_DIR}" "${GENERATED_DIR}/cutlass"

  perl -0pe 's/cutlass::nv_float4_t<cutlass::float_e2m1_t>/cutlass::mx_float4_t<cutlass::float_e2m1_t>/g; s/if \(props\.major != 10 \|\| \(props\.minor != 0 && props\.minor != 1 && props\.minor != 3\)\) \{\n    std::cerr << "This example requires a GPU with compute capability 100a\|f, 101a\|f, or 103a\|f\)\." << std::endl;\n    return 0;\n  \}/if (!((props.major == 10 && (props.minor == 0 || props.minor == 1 || props.minor == 3)) || (props.major == 11 && props.minor == 0))) {\n    std::cerr << "This example requires a Blackwell GPU with compute capability 100, 101, 103, or 110." << std::endl;\n    return 0;\n  \}/' \
    "${source}" > "${generated_source}"

  perl -0pe 's/#if \(defined\(CUTLASS_ARCH_MMA_SM100A_ENABLED\)/#if defined(__CUDA_ARCH__) \&\& (defined(CUTLASS_ARCH_MMA_SM100A_ENABLED)/g; s/#if \(defined\(CUTLASS_ARCH_MMA_SM100F_ENABLED\)/#if defined(__CUDA_ARCH__) \&\& (defined(CUTLASS_ARCH_MMA_SM100F_ENABLED)/g' \
    "${CUTLASS_FP4_ROOT}/include/cutlass/float8.h" > "${generated_float8}"

  "${NVCC}" "${COMMON_FLAGS[@]}" \
    -diag-suppress=20012 \
    -diag-suppress=20013 \
    -diag-suppress=20015 \
    -DCUTLASS_ARCH_MMA_SM110A_ENABLED=1 \
    -DCUTLASS_ENABLE_GDC_FOR_SM100=1 \
    -I"${GENERATED_DIR}" \
    -I"${SCRIPT_DIR}/include" \
    -I"${CUTLASS_FP4_ROOT}/include" \
    -I"${CUTLASS_FP4_ROOT}/tools/util/include" \
    -I"${CUTLASS_FP4_ROOT}/examples/common" \
    "${generated_source}" \
    -lcuda \
    -o "${MXFP4_CUTLASS_BIN}"
}

usage() {
  cat <<EOF
Usage:
  $0 clean
  $0 build-only
  $0 build-fp4-cutlass
  $0 nvfp4-cutlass [N]
  $0 mxfp4-cutlass [N]
  $0 fp4-probe [N]
  $0 [N] [all|fp8_q0_cuda_naive|fp8_q1_cuda_vec4cols|fp8_q2_cuda_vec8cols|fp8_q3_cuda_vec16cols|fp8_q4_mma_m16n8k32_global|fp8_q5_mma_m16n8k32_smem64|fp8_q6_mma_m16n8k32_smem64x128|fp8_q7_mma_m16n8k32_smem128x64|fp8_q8_cublaslt_matmul|int8_q0_cuda_naive|int8_q1_cuda_vec4cols|int8_q2_cuda_vec8cols|int8_q3_cuda_vec16cols|int8_q4_wmma_m16n16k16|int8_q5_wmma_m16n16k16_8warp|int8_q6_wmma_m32n8k16|int8_q7_wmma_m8n32k16|int8_q8_wmma_m32n64k16_smem|int8_q9_wmma_m32n32k16_reuse_a|int8_q10_wmma_m32n64k16_reuse_a|int8_q11_wmma_m32n128k16_reuse_a|int8_q12_wmma_m128n64k16_4warp_reuse_a|int8_q13_wmma_m128n128k16_8warp_reuse_a|int8_q14_wmma_m256n64k16_8warp_reuse_a|int8_q15_wmma_m128n64k16_4warp_reuse_a_bcol|int8_q16_wmma_m128n128k16_8warp_reuse_a_bcol|int8_q17_wmma_m256n64k16_8warp_reuse_a_bcol|int8_q18_mma_m16n8k32_smem64|int8_q19_cublas_gemmex]
EOF
}

ARG="${1:-1024}"

if [[ "${ARG}" == "clean" ]]; then
  rm -rf "${BUILD_DIR}"
  echo "Cleaned ${BUILD_DIR}"
  exit 0
fi

if [[ "${ARG}" == "build-only" ]]; then
  build_benchmark
  echo "Built ${BENCH_BIN}"
  exit 0
fi

if [[ "${ARG}" == "build-fp4-cutlass" ]]; then
  build_nvfp4_cutlass_72b
  build_mxfp4_cutlass_72a
  echo "Built ${NVFP4_CUTLASS_BIN}"
  echo "Built ${MXFP4_CUTLASS_BIN}"
  exit 0
fi

if [[ "${ARG}" == "nvfp4-cutlass" ]]; then
  if [[ ! -x "${NVFP4_CUTLASS_BIN}" ]]; then
    build_nvfp4_cutlass_72b
  fi
  n="${2:-1024}"
  "${NVFP4_CUTLASS_BIN}" \
    --m="${n}" \
    --n="${n}" \
    --k="${n}" \
    --iterations="${NVFP4_CUTLASS_ITERATIONS:-100}" \
    --swizzle="${NVFP4_CUTLASS_SWIZZLE:-2}"
  exit 0
fi

if [[ "${ARG}" == "mxfp4-cutlass" ]]; then
  if [[ ! -x "${MXFP4_CUTLASS_BIN}" ]]; then
    build_mxfp4_cutlass_72a
  fi
  n="${2:-1024}"
  "${MXFP4_CUTLASS_BIN}" \
    --m="${n}" \
    --n="${n}" \
    --k="${n}" \
    --iterations="${MXFP4_CUTLASS_ITERATIONS:-100}" \
    --swizzle="${MXFP4_CUTLASS_SWIZZLE:-1}"
  exit 0
fi

if [[ "${ARG}" == "fp4-probe" ]]; then
  build_fp4_probe
  "${FP4_PROBE_BIN}" "${2:-1024}"
  exit 0
fi

if [[ "${ARG}" == "--help" || "${ARG}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ ! -x "${BENCH_BIN}" ]]; then
  build_benchmark
fi

"${BENCH_BIN}" "$@"
