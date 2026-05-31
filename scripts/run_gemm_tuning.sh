#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build_cuda13}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/results/gemm_tuning}"
RAW_DIR="${OUT_DIR}/raw"
TUNE_SIZES="${TUNE_SIZES:-256 512 1024 2048}"
CUDA_ARCH="${CUDA_ARCH:-}"

mkdir -p "${BUILD_DIR}" "${RAW_DIR}"

detect_cuda_arch() {
  if [[ -n "${CUDA_ARCH}" ]]; then
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

if command -v cmake >/dev/null 2>&1; then
  cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
  cmake --build "${BUILD_DIR}" -j --target gemm_tune_blocksize
else
  nvcc -O3 -std=c++17 -arch="sm_${CUDA_ARCH}" \
    -I"${ROOT_DIR}/GEMM/include" \
    "${ROOT_DIR}/GEMM/src/tune_blocksize.cu" \
    -lcublas \
    -o "${BUILD_DIR}/gemm_tune_blocksize"
fi

AGG_CSV="${OUT_DIR}/blocksize_tuning.csv"
FIGURE_DIR="${OUT_DIR}/figures"
printf 'Kernel,Config,N,BM,BN,BK,TM,TN,Threads,SharedMemoryBytes,TimeMs,GFLOPS,RatioToCuBLAS,Matched\n' > "${AGG_CSV}"

for n in ${TUNE_SIZES}; do
  run_dir="${RAW_DIR}/N${n}"
  mkdir -p "${run_dir}"
  echo "Running GEMM blocksize tuning N=${n}"
  (
    cd "${run_dir}"
    "${BUILD_DIR}/gemm_tune_blocksize" "${n}" | tee "stdout.txt"
  )
  tail -n +2 "${run_dir}/gemm_blocksize_tuning.csv" >> "${AGG_CSV}"
done

python3 "${ROOT_DIR}/scripts/plot_benchmarks.py" \
  --tuning "${AGG_CSV}" \
  --out-dir "${FIGURE_DIR}"

echo "GEMM tuning data: ${AGG_CSV}"
echo "GEMM tuning figures: ${FIGURE_DIR}"
echo "Top matched configs:"
awk -F, 'NR > 1 && $NF == 1 { print $12 "," $1 "," $2 ",N=" $3 ",ratio=" $13 }' "${AGG_CSV}" \
  | sort -t, -k1,1nr \
  | head -20
