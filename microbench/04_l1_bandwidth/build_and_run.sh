#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
RESULT_DIR="${SCRIPT_DIR}/results"
NVCC="${NVCC:-nvcc}"

COMMON_FLAGS=(
  -O3
  -std=c++17
  -gencode arch=compute_110a,code=sm_110a
)

BIN="${BUILD_DIR}/l1_bandwidth"
CSV="${RESULT_DIR}/l1_bandwidth.csv"

usage() {
  cat <<EOF
Usage:
  $0 clean
  $0 build-only
  $0 run
  $0 run-one [benchmark args...]
  $0 validate
  $0 ncu
EOF
}

build() {
  mkdir -p "${BUILD_DIR}"
  "${NVCC}" "${COMMON_FLAGS[@]}" "${SCRIPT_DIR}/l1_bandwidth.cu" -o "${BIN}"
}

collect_sass() {
  mkdir -p "${RESULT_DIR}"
  if command -v cuobjdump >/dev/null 2>&1; then
    cuobjdump --dump-sass "${BIN}" > "${RESULT_DIR}/l1_bandwidth.sass"
    grep -E 'Function : _Z|LDG|LD\.E\.128|STG|ST\.E\.128|CACHE|CCTL|MEMBAR|CS2R' \
      "${RESULT_DIR}/l1_bandwidth.sass" | head -n 240 \
      > "${RESULT_DIR}/sass_summary.txt" || true
  fi
}

run_default() {
  mkdir -p "${RESULT_DIR}"
  local common_args=(
    --iters "${ITERS:-4096}"
    --warmup-rounds "${WARMUP_ROUNDS:-2}"
    --threads "${THREADS:-256}"
    --bytes-per-cta "${BYTES_PER_CTA:-16384}"
  )
  "${BIN}" --csv-header | tee "${CSV}"
  local mode
  for mode in read-ca read-cg write-wb write-cg; do
    "${BIN}" --mode "${mode}" "${common_args[@]}" --csv | tee -a "${CSV}"
  done
  collect_sass
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
    collect_sass
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
    build
    shift || true
    "${SCRIPT_DIR}/scripts/l1_validation.py" "$@"
    ;;
  ncu)
    build
    shift || true
    "${SCRIPT_DIR}/scripts/ncu_l1_validation.py" "$@"
    ;;
  --help|-h)
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
