#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BIN="${BENCH_DIR}/build/smem_bank_bench"
RESULT_DIR="${BENCH_DIR}/results/sass"
CUOBJDUMP="${CUOBJDUMP:-$(command -v cuobjdump || true)}"
NVDISASM="${NVDISASM:-$(command -v nvdisasm || true)}"

"${SCRIPT_DIR}/build.sh"

if [[ -z "${CUOBJDUMP}" ]]; then
  echo "cuobjdump is required to inspect the executable." >&2
  exit 1
fi

mkdir -p "${RESULT_DIR}"
CUOBJDUMP_SASS="${RESULT_DIR}/smem_bank_bench.cuobjdump.sass"
"${CUOBJDUMP}" --dump-sass "${BIN}" > "${CUOBJDUMP_SASS}"

if ! rg -q '\bLDS(\.|[[:space:]])' "${CUOBJDUMP_SASS}"; then
  echo "SASS validation failed: no shared-load LDS instruction found." >&2
  exit 1
fi
if ! rg -q '\bSTS(\.|[[:space:]])' "${CUOBJDUMP_SASS}"; then
  echo "SASS validation failed: no shared-store STS instruction found." >&2
  exit 1
fi

cuobjdump_lds="$(rg -c '\bLDS(\.|[[:space:]])' "${CUOBJDUMP_SASS}")"
cuobjdump_sts="$(rg -c '\bSTS(\.|[[:space:]])' "${CUOBJDUMP_SASS}")"
if ((cuobjdump_lds < 40)); then
  echo "SASS validation failed: expected at least 40 LDS instructions, found ${cuobjdump_lds}." >&2
  exit 1
fi

echo "cuobjdump: ${cuobjdump_lds} LDS, ${cuobjdump_sts} STS"
echo "Wrote ${CUOBJDUMP_SASS}"

if [[ -z "${NVDISASM}" ]]; then
  echo "nvdisasm is not on PATH; cuobjdump validation is complete."
  exit 0
fi

rm -f "${RESULT_DIR}"/*.cubin
(
  cd "${RESULT_DIR}"
  "${CUOBJDUMP}" --extract-elf all "${BIN}" >/dev/null
)

NVDISASM_SASS="${RESULT_DIR}/smem_bank_bench.nvdisasm.sass"
: > "${NVDISASM_SASS}"
shopt -s nullglob
cubins=("${RESULT_DIR}"/*.cubin)
if [[ "${#cubins[@]}" -eq 0 ]]; then
  echo "nvdisasm validation failed: cuobjdump extracted no cubin." >&2
  exit 1
fi
for cubin in "${cubins[@]}"; do
  "${NVDISASM}" "${cubin}" >> "${NVDISASM_SASS}"
done

if ! rg -q '\bLDS(\.|[[:space:]])' "${NVDISASM_SASS}" ||
    ! rg -q '\bSTS(\.|[[:space:]])' "${NVDISASM_SASS}"; then
  echo "nvdisasm validation failed: LDS or STS is missing." >&2
  exit 1
fi

nvdisasm_lds="$(rg -c '\bLDS(\.|[[:space:]])' "${NVDISASM_SASS}")"
nvdisasm_sts="$(rg -c '\bSTS(\.|[[:space:]])' "${NVDISASM_SASS}")"
if ((nvdisasm_lds < 40)); then
  echo "nvdisasm validation failed: expected at least 40 LDS instructions, found ${nvdisasm_lds}." >&2
  exit 1
fi

echo "nvdisasm: ${nvdisasm_lds} LDS, ${nvdisasm_sts} STS"
echo "Wrote ${NVDISASM_SASS}"
