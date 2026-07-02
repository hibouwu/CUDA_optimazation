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

echo "Wrote ${OUTPUT}"
python3 "${SCRIPT_DIR}/parse_results.py"
