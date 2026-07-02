#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BIN="${ROOT}/build/transpose_2d_bench"
RESULT_DIR="${ROOT}/results"
OUTPUT="${RESULT_DIR}/basic_results.csv"
ITERS="${ITERS:-100000}"
WARMUPS="${WARMUPS:-5}"
REPEATS="${REPEATS:-20}"

case_selector="all"
args=("$@")
for ((i = 0; i < ${#args[@]}; ++i)); do
  if [[ "${args[i]}" == "--case" && $((i + 1)) -lt ${#args[@]} ]]; then
    case_selector="${args[i + 1]}"
  fi
done

"${SCRIPT_DIR}/build.sh"
mkdir -p "${RESULT_DIR}"
"${BIN}" --case "${case_selector}" --iters "${ITERS}" --warmups "${WARMUPS}" \
  --repeats "${REPEATS}" > "${OUTPUT}"
python3 "${SCRIPT_DIR}/parse_results.py"
