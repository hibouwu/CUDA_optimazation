#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${BUILD_DIR:-}" ]]; then
  if [[ -f /.dockerenv ]]; then
    BUILD_DIR="${ROOT_DIR}/build_docker"
  else
    BUILD_DIR="${ROOT_DIR}/build"
  fi
fi
GEMM_SUITE="${GEMM_SUITE:-all}"
case "${GEMM_SUITE}" in
  all)
    DEFAULT_OUT_DIR="${ROOT_DIR}/results/gemm/all"
    DEFAULT_AGG_CSV="gemm_sweep.csv"
    ;;
  fp32)
    DEFAULT_OUT_DIR="${ROOT_DIR}/results/gemm/fp32"
    DEFAULT_AGG_CSV="gemm_fp32_sweep.csv"
    ;;
  tensor_core)
    DEFAULT_OUT_DIR="${ROOT_DIR}/results/gemm/tensor_core"
    DEFAULT_AGG_CSV="gemm_tensor_core_sweep.csv"
    ;;
  cublas|v1|v2|v3|v3a|v3b|v4|v5|v6|v7|v8a|v8b|v8c|cublas_tc|tc1|tc2|tc3a|tc3b)
    DEFAULT_OUT_DIR="${ROOT_DIR}/results/gemm/backend_${GEMM_SUITE}"
    DEFAULT_AGG_CSV="gemm_${GEMM_SUITE}_sweep.csv"
    ;;
  *)
    echo "Unknown GEMM_SUITE=${GEMM_SUITE}." >&2
    echo "Use all, fp32, tensor_core, cublas, v1, v2, v3, v3a, v3b, v4, v5, v6, v7, v8a, v8b, v8c, cublas_tc, tc1, tc2, tc3a, or tc3b." >&2
    exit 1
    ;;
esac
OUT_DIR="${OUT_DIR:-${DEFAULT_OUT_DIR}}"
RAW_DIR="${OUT_DIR}/raw"
PRESET="${PRESET:-default}"
case "${PRESET}" in
  quick)
    DEFAULT_GEMM_SIZES="128 256 512 1024"
    ;;
  default)
    DEFAULT_GEMM_SIZES="128 256 512 1024 2048 4096"
    ;;
  full)
    DEFAULT_GEMM_SIZES="128 256 512 1024 2048 4096 8192"
    ;;
  *)
    echo "Unknown PRESET=${PRESET}. Use quick, default, full, or set GEMM_SIZES explicitly." >&2
    exit 1
    ;;
esac
GEMM_SIZES="${GEMM_SIZES:-${DEFAULT_GEMM_SIZES}}"
TRIALS="${TRIALS:-3}"

mkdir -p "${BUILD_DIR}" "${RAW_DIR}"

detect_cuda_arch() {
  if [[ -n "${CUDA_ARCH:-}" ]]; then
    echo "${CUDA_ARCH}"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local compute_cap
    compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '. ')"
    if [[ "${compute_cap}" =~ ^[0-9]+$ ]]; then
      echo "${compute_cap}"
      return
    fi
  fi
  echo "86"
}

CUDA_ARCH="$(detect_cuda_arch)"

ensure_python() {
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
    return
  fi
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
    return
  fi
  if [[ "${EUID}" -eq 0 ]] && command -v apt-get >/dev/null 2>&1; then
    echo "python3 not found, installing python3-minimal for plotting."
    apt-get update
    apt-get install -y python3-minimal
    PYTHON_BIN="python3"
    return
  fi
  echo "python3 is required for plotting. Install python3 or run inside the CUDA container as root." >&2
  exit 1
}

build_gemm() {
  if command -v cmake >/dev/null 2>&1; then
    if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]] &&
       ! grep -qx "CMAKE_HOME_DIRECTORY:INTERNAL=${ROOT_DIR}" \
         "${BUILD_DIR}/CMakeCache.txt"; then
      echo "CMake cache in ${BUILD_DIR} was created for another source path." >&2
      echo "Remove it with: rm -rf ${BUILD_DIR}" >&2
      echo "Or set BUILD_DIR to a clean directory." >&2
      exit 1
    fi
    cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
    cmake --build "${BUILD_DIR}" -j --target gemm_bench
  else
    echo "cmake not found, building gemm_bench with nvcc."
    nvcc -O3 -std=c++17 -arch="sm_${CUDA_ARCH}" \
      -I"${ROOT_DIR}/GEMM/include" \
      "${ROOT_DIR}/GEMM/src/main.cu" \
      -lcublas -lcuda \
      -o "${BUILD_DIR}/gemm_bench"
  fi
}

run_gemm_size() {
  local n="$1"
  local trial="$2"
  local run_dir="${RAW_DIR}/N${n}/trial_${trial}"
  mkdir -p "${run_dir}"
  echo "Running GEMM N=${n}, trial=${trial}/${TRIALS}"
  (
    cd "${run_dir}"
    "${BUILD_DIR}/gemm_bench" "${n}" "${GEMM_SUITE}" | tee "stdout.txt"
  )
  awk -F, -v trial="${trial}" -v suite="${GEMM_SUITE}" '
    NR == 1 { next }
    suite == "all" { print $0 "," trial; next }
    suite == "fp32" && $4 == "fp32" { print $0 "," trial; next }
    suite == "tensor_core" && $5 == "cuBLAS Tensor Core" { print $0 "," trial; next }
    suite == "cublas" && $1 == "cublas" { print $0 "," trial; next }
    suite == "cublas_tc" && $1 == "cublas_tc" { print $0 "," trial; next }
    suite ~ /^v[0-9]+[ab]?$/ && $1 == "cublas" { print $0 "," trial; next }
    suite ~ /^v[0-9]+[ab]?$/ && $1 == suite { print $0 "," trial; next }
    suite == "tc1" && $1 == "cublas_tc" { print $0 "," trial; next }
    suite == "tc1" && $1 == "tc1" { print $0 "," trial; next }
    suite == "tc2" && $1 == "cublas_tc" { print $0 "," trial; next }
    suite == "tc2" && $1 == "tc2" { print $0 "," trial; next }
    suite == "tc3a" && $1 == "cublas_tc" { print $0 "," trial; next }
    suite == "tc3a" && $1 == "tc3a" { print $0 "," trial; next }
    suite == "tc3b" && $1 == "cublas_tc" { print $0 "," trial; next }
    suite == "tc3b" && $1 == "tc3b" { print $0 "," trial; next }
  ' \
    "${run_dir}/sgemm_benchmark.csv" >> "${AGG_CSV}"
}

build_gemm
ensure_python

AGG_CSV="${OUT_DIR}/${DEFAULT_AGG_CSV}"
FIG_DIR="${OUT_DIR}/figures"
printf 'BackendId,Version,N,Precision,Reference,TimeMs,GFLOPS,RatioToReference,Matched,Trial\n' > "${AGG_CSV}"

for n in ${GEMM_SIZES}; do
  for trial in $(seq 1 "${TRIALS}"); do
    run_gemm_size "${n}" "${trial}"
  done
done

rm -rf "${FIG_DIR}"
mkdir -p "${FIG_DIR}"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/plot_benchmarks.py" \
  --gemm "${AGG_CSV}" \
  --out-dir "${FIG_DIR}"

echo "GEMM experiment data: ${OUT_DIR}"
echo "GEMM figures: ${FIG_DIR}"
