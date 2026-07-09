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

BIN="${BUILD_DIR}/sm110_tcgen05_tmem_size_probe"
CSV="${RESULT_DIR}/tmem_size_probe.csv"

usage() {
  cat <<EOF
Usage:
  $0 clean
  $0 build-only
  $0 run-one <columns>
  $0 run [columns...]

Default run sweeps representative low values plus every column from 500 to 560,
then 768 and 1024. Each column is tested in a fresh process so illegal
instruction failures do not poison later probes.
EOF
}

build() {
  mkdir -p "${BUILD_DIR}"
  "${NVCC}" "${COMMON_FLAGS[@]}" "${SCRIPT_DIR}/demo.cu" -o "${BIN}"
}

default_columns() {
  printf '%s\n' 32 64 128 256 384 448 480 496
  seq 500 560
  printf '%s\n' 768 1024
}

run_one() {
  local columns="$1"
  "${BIN}" --columns "${columns}" --csv
}

run_sweep() {
  mkdir -p "${RESULT_DIR}"
  {
    echo "columns,requested_bytes,requested_kib,tmem_base,tail_column,front_ok,tail_ok,status,cuda_error"
    local columns
    for columns in "$@"; do
      local line rc
      set +e
      line="$(run_one "${columns}" 2>&1)"
      rc=$?
      set -e
      if [[ -n "${line}" && "${line}" == "${columns},"* ]]; then
        echo "${line}"
      else
        local bytes=$((columns * 512))
        # Keep the CSV rectangular even if the process fails before printing a row.
        printf '%s,%s,%.1f,0,0,0,0,process_failed,rc_%s\n' \
          "${columns}" "${bytes}" "$(awk "BEGIN {print ${bytes}/1024}")" "${rc}"
        if [[ -n "${line}" ]]; then
          printf '# %s\n' "${line}" >&2
        fi
      fi
    done
  } | tee "${CSV}"
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
  run-one)
    if [[ $# -ne 2 ]]; then
      usage
      exit 2
    fi
    build
    run_one "$2"
    ;;
  run)
    build
    shift || true
    if [[ $# -gt 0 ]]; then
      run_sweep "$@"
    else
      mapfile -t cols < <(default_columns)
      run_sweep "${cols[@]}"
    fi
    ;;
  --help|-h)
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
