#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
RESULT_DIR="${SCRIPT_DIR}/results"
NVCC="${NVCC:-nvcc}"

COMMON_FLAGS=(
  -O3
  -std=c++17
  -Xptxas
  -dlcm=cg
  -gencode arch=compute_110a,code=sm_110a
)

BIN="${BUILD_DIR}/l2_throughput"
CSV="${RESULT_DIR}/l2_throughput.csv"

usage() {
  cat <<EOF
Usage:
  $0 clean
  $0 build-only
  $0 run
  $0 run-one [benchmark args...]
  $0 validate [validation args...]

Default run sweeps:
  read-same, read-unique, write-unique

Environment controls for "run":
  ITERS=4096
  WARMUP_ITERS=64
  BLOCKS_PER_SM=4
  THREADS_PER_BLOCK=256
  BYTES=<working-set bytes, optional>
EOF
}

build() {
  mkdir -p "${BUILD_DIR}"
  "${NVCC}" "${COMMON_FLAGS[@]}" "${SCRIPT_DIR}/demo.cu" -o "${BIN}"
}

run_default() {
  mkdir -p "${RESULT_DIR}"
  local common_args=(
    --iters "${ITERS:-4096}"
    --warmup-iters "${WARMUP_ITERS:-64}"
    --blocks-per-sm "${BLOCKS_PER_SM:-4}"
    --threads "${THREADS_PER_BLOCK:-256}"
  )
  if [[ -n "${BYTES:-}" ]]; then
    common_args+=(--bytes "${BYTES}")
  fi

  "${BIN}" --csv-header | tee "${CSV}"
  local mode
  for mode in read-same read-unique write-unique; do
    "${BIN}" --mode "${mode}" "${common_args[@]}" --csv | tee -a "${CSV}"
  done
  if command -v cuobjdump >/dev/null 2>&1; then
    cuobjdump --dump-sass "${BIN}" > "${RESULT_DIR}/l2_throughput.sass"
    grep -E 'LDG\.E\.128|LDG\.E\.STRONG|STG\.E\.128|Function : _Z20l2_throughput_kernel' \
      "${RESULT_DIR}/l2_throughput.sass" | head -n 180 \
      > "${RESULT_DIR}/sass_summary.txt" || true
  fi
  echo "Wrote ${CSV}"
}

CMD="${1:-run}"

case "${CMD}" in
  clean)
    rm -rf "${BUILD_DIR}" "${RESULT_DIR}"
    echo "Cleaned ${BUILD_DIR} and ${RESULT_DIR}"
    ;;
  build-only)
    build
    echo "Built ${BIN}"
    ;;
  run)
    build
    run_default
    ;;
  run-one)
    build
    shift || true
    "${BIN}" "$@"
    ;;
  validate)
    shift || true
    "${SCRIPT_DIR}/scripts/l2_validation.py" "$@"
    ;;
  --help|-h)
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
