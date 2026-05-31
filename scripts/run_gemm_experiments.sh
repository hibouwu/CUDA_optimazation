#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/results/gemm}"
RAW_DIR="${OUT_DIR}/raw"
PRESET="${PRESET:-default}"
case "${PRESET}" in
  quick)
    DEFAULT_GEMM_SIZES="512 1024"
    ;;
  default)
    DEFAULT_GEMM_SIZES="256 384 512 640 768 896 1024 1280 1536 1792 2048"
    ;;
  full)
    DEFAULT_GEMM_SIZES="128 256 384 512 640 768 896 1024 1280 1536 1792 2048 2560 3072 3584 4096"
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
    cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
    cmake --build "${BUILD_DIR}" -j --target gemm_bench
  else
    echo "cmake not found, building gemm_bench with nvcc."
    nvcc -O3 -std=c++17 -arch="sm_${CUDA_ARCH}" \
      -I"${ROOT_DIR}/GEMM/include" \
      "${ROOT_DIR}/GEMM/src/main.cu" \
      -lcublas \
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
    "${BUILD_DIR}/gemm_bench" "${n}" | tee "stdout.txt"
  )
  awk -v trial="${trial}" 'NR > 1 { print $0 "," trial }' \
    "${run_dir}/sgemm_benchmark.csv" >> "${AGG_CSV}"
}

build_gemm
ensure_python

AGG_CSV="${OUT_DIR}/sgemm_sweep.csv"
printf 'Version,N,TimeMs,GFLOPS,RatioToCuBLAS,Matched,Trial\n' > "${AGG_CSV}"

for n in ${GEMM_SIZES}; do
  for trial in $(seq 1 "${TRIALS}"); do
    run_gemm_size "${n}" "${trial}"
  done
done

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/plot_benchmarks.py" \
  --gemm "${AGG_CSV}" \
  --out-dir "${OUT_DIR}/figures"

echo "GEMM experiment data: ${OUT_DIR}"
echo "GEMM figures: ${OUT_DIR}/figures"
