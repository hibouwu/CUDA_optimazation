#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BIN="${ROOT}/build/sass_register_bench"
RESULT_DIR="${ROOT}/results/sass_patched"
OUTPUT="${RESULT_DIR}/results.csv"

"${SCRIPT_DIR}/build.sh"
python3 "${SCRIPT_DIR}/patch_sass.py"

cubins=()
while IFS= read -r cubin; do
  cubins+=(--cubin "${cubin}")
done < <(find "${RESULT_DIR}" -maxdepth 1 -name 'S*.cubin' -print | sort)

"${BIN}" --iters "${ITERS:-100000}" --warmups "${WARMUPS:-5}" \
  --repeats "${REPEATS:-20}" "${cubins[@]}" "$@" > "${OUTPUT}"

echo "Wrote ${OUTPUT}"
column -s, -t < "${OUTPUT}"
python3 "${SCRIPT_DIR}/plot_experiments.py"
