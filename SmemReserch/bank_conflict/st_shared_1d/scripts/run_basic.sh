#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ITERS="${ITERS:-100000}"; WARMUPS="${WARMUPS:-5}"; REPEATS="${REPEATS:-20}"
"${SCRIPT_DIR}/build.sh"
mkdir -p "${ROOT}/results"
BIN="${ROOT}/build/st_shared_1d_bench"
OUT="${ROOT}/results/basic_results.csv"
"${BIN}" --case all --iters "${ITERS}" --warmups "${WARMUPS}" --repeats "${REPEATS}" > "${OUT}"
for stride in 1 2 4 8 16 32; do
  "${BIN}" --case stride --stride "${stride}" --iters "${ITERS}" \
    --warmups "${WARMUPS}" --repeats "${REPEATS}" | tail -n +2 >> "${OUT}"
done
python3 "${SCRIPT_DIR}/parse_results.py"

