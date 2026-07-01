#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BIN="${SCRIPT_DIR}/../build/smem_bank_bench"
RESULT_DIR="${SCRIPT_DIR}/results"
OUTPUT="${RESULT_DIR}/basic_results.csv"
ITERS="${ITERS:-100000}"

if [[ ! -x "${BIN}" ]]; then
  echo "Missing ${BIN}; run ${SCRIPT_DIR}/build.sh first." >&2
  exit 1
fi

mkdir -p "${RESULT_DIR}"
"${BIN}" --case all --iters "${ITERS}" > "${OUTPUT}"
for stride in 1 2 4 8 16 32; do
  "${BIN}" --case stride --stride "${stride}" --offset 0 --iters "${ITERS}" |
    tail -n +2 >> "${OUTPUT}"
done

echo "Wrote ${OUTPUT}"

