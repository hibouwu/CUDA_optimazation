#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BIN="${BENCH_DIR}/build/smem_bank_bench"
RESULT_DIR="${BENCH_DIR}/results"
OUTPUT="${RESULT_DIR}/basic_results.csv"
ITERS="${ITERS:-100000}"
WARMUPS="${WARMUPS:-5}"
REPEATS="${REPEATS:-20}"

echo "Building benchmark"
"${SCRIPT_DIR}/build.sh"

mkdir -p "${RESULT_DIR}"
"${BIN}" --case all --iters "${ITERS}" --warmups "${WARMUPS}" --repeats "${REPEATS}" > "${OUTPUT}"
for stride in 1 2 4 8 16 32; do
  "${BIN}" --case stride --stride "${stride}" --offset 0 --iters "${ITERS}" \
    --warmups "${WARMUPS}" --repeats "${REPEATS}" |
    tail -n +2 >> "${OUTPUT}"
done

echo "Wrote ${OUTPUT}"
python3 "${SCRIPT_DIR}/parse_results.py"
