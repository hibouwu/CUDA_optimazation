#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BIN="${ROOT}/build/sass_register_bench"
RESULT_DIR="${ROOT}/results/bank_scan"
OUTPUT="${RESULT_DIR}/results.csv"

"${SCRIPT_DIR}/build.sh"
python3 "${SCRIPT_DIR}/patch_bank_scan.py"

cubins=()
while IFS= read -r cubin; do
  cubins+=(--cubin "${cubin}")
done < <(
  find "${RESULT_DIR}" -maxdepth 1 \
    \( -name 'B*.cubin' -o -name 'L*.cubin' -o -name 'M*.cubin' \
       -o -name 'T*.cubin' \) \
    -print | sort
)

"${BIN}" --iters "${ITERS:-20000}" --warmups "${WARMUPS:-3}" \
  --repeats "${REPEATS:-10}" "${cubins[@]}" "$@" > "${OUTPUT}"

echo "Wrote ${OUTPUT}"
python3 "${SCRIPT_DIR}/plot_bank_scan.py"
