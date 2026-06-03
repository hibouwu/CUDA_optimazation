#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/results/transformer}"
RAW_DIR="${OUT_DIR}/raw"
PRESET="${PRESET:-default}"
case "${PRESET}" in
  quick)
    DEFAULT_TRANSFORMER_SHAPES="1:128:768:12:64 1:1024:4096:32:128"
    ;;
  default)
    DEFAULT_TRANSFORMER_SHAPES="1:128:768:12:64 1:512:768:12:64 1:1024:768:12:64 1:512:4096:32:128 1:1024:4096:32:128 1:2048:4096:32:128"
    ;;
  full)
    DEFAULT_TRANSFORMER_SHAPES="1:128:768:12:64 1:256:768:12:64 1:512:768:12:64 1:1024:768:12:64 1:2048:768:12:64 1:512:4096:32:128 1:1024:4096:32:128 1:2048:4096:32:128 1:4096:4096:32:128"
    ;;
  *)
    echo "Unknown PRESET=${PRESET}. Use quick, default, full, or set TRANSFORMER_SHAPES explicitly." >&2
    exit 1
    ;;
esac
TRANSFORMER_SHAPES="${TRANSFORMER_SHAPES:-${DEFAULT_TRANSFORMER_SHAPES}}"
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

build_transformer() {
  if command -v cmake >/dev/null 2>&1; then
    cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
    cmake --build "${BUILD_DIR}" -j --target transformer_bench
  else
    echo "cmake not found, building transformer_bench with nvcc."
    nvcc -O3 -std=c++17 -arch="sm_${CUDA_ARCH}" \
      -I"${ROOT_DIR}/TRANSFORMER/include" \
      "${ROOT_DIR}/TRANSFORMER/src/main.cu" \
      -o "${BUILD_DIR}/transformer_bench"
  fi
}

run_transformer_shape() {
  local shape="$1"
  local trial="$2"
  local batch seq hidden heads head_dim
  IFS=":" read -r batch seq hidden heads head_dim <<< "${shape}"
  if [[ -z "${batch}" || -z "${seq}" || -z "${hidden}" || -z "${heads}" || -z "${head_dim}" ]]; then
    echo "Invalid shape '${shape}'. Expected B:S:H:heads:head_dim, for example 1:1024:4096:32:128." >&2
    exit 1
  fi

  local run_dir="${RAW_DIR}/B${batch}_S${seq}_H${hidden}_heads${heads}_D${head_dim}/trial_${trial}"
  mkdir -p "${run_dir}"
  echo "Running TRANSFORMER B=${batch}, S=${seq}, H=${hidden}, heads=${heads}, head_dim=${head_dim}, trial=${trial}/${TRIALS}"
  (
    cd "${run_dir}"
    "${BUILD_DIR}/transformer_bench" "${batch}" "${seq}" "${hidden}" "${heads}" "${head_dim}" | tee "stdout.txt"
  )
  awk -v trial="${trial}" 'NR > 1 { print $0 "," trial }' \
    "${run_dir}/transformer_benchmark.csv" >> "${AGG_CSV}"
}

build_transformer
ensure_python

AGG_CSV="${OUT_DIR}/transformer_sweep.csv"
printf 'Operator,Version,Batch,SeqLen,Hidden,NumHeads,HeadDim,TimeMs,BandwidthGBps,Matched,Trial\n' > "${AGG_CSV}"

for shape in ${TRANSFORMER_SHAPES}; do
  for trial in $(seq 1 "${TRIALS}"); do
    run_transformer_shape "${shape}" "${trial}"
  done
done

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/plot_benchmarks.py" \
  --transformer "${AGG_CSV}" \
  --out-dir "${OUT_DIR}/figures"

echo "TRANSFORMER experiment data: ${OUT_DIR}"
echo "TRANSFORMER figures: ${OUT_DIR}/figures"
