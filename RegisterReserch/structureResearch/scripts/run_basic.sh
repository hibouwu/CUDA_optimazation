#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BIN="${ROOT}/build/register_bench"
RESULT_DIR="${ROOT}/results"
OUTPUT="${RESULT_DIR}/basic_results.csv"

"${SCRIPT_DIR}/build.sh"
mkdir -p "${RESULT_DIR}"

"${BIN}" --case "${CASE:-all}" --iters "${ITERS:-100000}" \
  --warmups "${WARMUPS:-5}" --repeats "${REPEATS:-20}" "$@" > "${OUTPUT}"

echo "Wrote ${OUTPUT}"
python3 "${SCRIPT_DIR}/parse_results.py"
