#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build_sm110}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/results/gemm_sm110}"
RAW_DIR="${OUT_DIR}/raw"
GEMM_SUITE="${GEMM_SUITE:-all}"
GEMM_SIZES="${GEMM_SIZES:-128 256 512 1024 2048}"
TRIALS="${TRIALS:-3}"
CUDA_ARCH="${CUDA_ARCH:-}"

case "${GEMM_SUITE}" in
  all|cublas_tc|tc3|tc4|tc5|tc5a|tc5b) ;;
  *)
    echo "Unknown GEMM_SUITE=${GEMM_SUITE}. Use all, cublas_tc, tc3, tc4, tc5, tc5a, or tc5b." >&2
    exit 1
    ;;
esac

mkdir -p "${BUILD_DIR}" "${RAW_DIR}"

EXTRA_GENCODE=()
if [[ -z "${CUDA_ARCH}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '. ')"
fi
if [[ "${CUDA_ARCH}" =~ ^[0-9]+$ ]]; then
  EXTRA_GENCODE=(-gencode "arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}")
fi

nvcc -O3 -std=c++17 \
  -DTC3_SM110_HOST_HAS_TCGEN05=1 \
  -gencode arch=compute_110a,code=sm_110a \
  "${EXTRA_GENCODE[@]}" \
  -I"${ROOT_DIR}/GEMMsm110/include" \
  "${ROOT_DIR}/GEMMsm110/src/main.cu" \
  -lcublas -lcuda \
  -o "${BUILD_DIR}/gemm_sm110_bench"

AGG_CSV="${OUT_DIR}/gemm_sm110_sweep.csv"
printf 'BackendId,Version,N,Precision,Reference,TimeMs,GFLOPS,RatioToReference,Matched,Trial\n' > "${AGG_CSV}"

for n in ${GEMM_SIZES}; do
  for trial in $(seq 1 "${TRIALS}"); do
    run_dir="${RAW_DIR}/N${n}/trial_${trial}"
    mkdir -p "${run_dir}"
    echo "Running GEMMsm110 N=${n}, trial=${trial}/${TRIALS}"
    (
      cd "${run_dir}"
      "${BUILD_DIR}/gemm_sm110_bench" "${n}" "${GEMM_SUITE}" | tee stdout.txt
    )
    awk -F, -v trial="${trial}" '
      NR == 1 { next }
      { print $0 "," trial }
    ' "${run_dir}/sgemm_sm110_benchmark.csv" >> "${AGG_CSV}"
  done
done

echo "GEMMsm110 experiment data: ${OUT_DIR}"
