#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BIN="${ROOT}/build/real_transpose_bench"
RESULT_DIR="${ROOT}/results"
OUTPUT="${RESULT_DIR}/basic_results.csv"

args=("$@")
has_option() {
  local wanted="$1"
  local argument
  for argument in "${args[@]}"; do
    [[ "${argument}" == "${wanted}" ]] && return 0
  done
  return 1
}

has_option --case || args+=(--case "${CASE:-all}")
has_option --width || args+=(--width "${WIDTH:-4096}")
has_option --height || args+=(--height "${HEIGHT:-4096}")
has_option --iters || args+=(--iters "${ITERS:-10}")
has_option --warmups || args+=(--warmups "${WARMUPS:-2}")
has_option --repeats || args+=(--repeats "${REPEATS:-10}")

"${SCRIPT_DIR}/build.sh"
mkdir -p "${RESULT_DIR}"
"${BIN}" "${args[@]}" > "${OUTPUT}"
python3 "${SCRIPT_DIR}/parse_results.py"
