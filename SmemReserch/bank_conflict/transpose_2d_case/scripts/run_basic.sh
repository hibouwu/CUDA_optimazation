#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ITERS="${ITERS:-100000}"; WARMUPS="${WARMUPS:-5}"; REPEATS="${REPEATS:-20}"
"${SCRIPT_DIR}/build.sh"
mkdir -p "${ROOT}/results"
"${ROOT}/build/transpose_2d_bench" --case all --iters "${ITERS}" \
  --warmups "${WARMUPS}" --repeats "${REPEATS}" > "${ROOT}/results/basic_results.csv"
python3 "${SCRIPT_DIR}/parse_results.py"

