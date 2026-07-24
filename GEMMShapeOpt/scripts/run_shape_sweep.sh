#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_DIR="${ROOT_DIR}/GEMMShapeOpt"
BENCH_BIN="${BENCH_BIN:-${ROOT_DIR}/GEMMsm110/build/gemm_sm110_bench}"
SHAPES_CSV="${SHAPES_CSV:-${PROJECT_DIR}/shapes/default_shapes.csv}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/results/gemm_shape_opt/$(date +%Y%m%d_%H%M%S)}"
RAW_DIR="${OUT_DIR}/raw"
AGG_CSV="${OUT_DIR}/shape_sweep.csv"
ANALYSIS_MD="${OUT_DIR}/analysis.md"
BACKENDS="${BACKENDS:-cublas_tc shapeopt cutlass tc5a tc5b}"
EPILOGUES="${EPILOGUES:-none}"
BACKEND_TIMEOUT_SECONDS="${BACKEND_TIMEOUT_SECONDS:-60}"
BACKEND_KILL_GRACE_SECONDS="${BACKEND_KILL_GRACE_SECONDS:-5}"
ALLOW_STALE_BENCH="${ALLOW_STALE_BENCH:-0}"
SWEEP_MIN_RATIO="${SWEEP_MIN_RATIO:-0.90}"
REFERENCE_BACKENDS="${REFERENCE_BACKENDS:-cublas_tc}"
ENFORCE_MIN_RATIO="${ENFORCE_MIN_RATIO:-1}"

if [[ ! -x "${BENCH_BIN}" ]]; then
  echo "Benchmark binary not found or not executable: ${BENCH_BIN}" >&2
  echo "Build it first, for example: cd ${ROOT_DIR}/GEMMsm110 && ./build_and_run.sh build-only" >&2
  exit 2
fi
if [[ "${ALLOW_STALE_BENCH}" != "1" &&
      ( "${ROOT_DIR}/GEMMsm110/src/main.cu" -nt "${BENCH_BIN}" ||
        "${ROOT_DIR}/GEMMsm110/include/cublaslt_reference.cuh" -nt "${BENCH_BIN}" ||
        "${ROOT_DIR}/GEMMsm110/include/gemm_benchmark.cuh" -nt "${BENCH_BIN}" ||
        "${ROOT_DIR}/GEMMsm110/include/sm110_backend_registry.cuh" -nt "${BENCH_BIN}" ) ]]; then
  echo "Benchmark binary is older than GEMMsm110 cuBLASLt reference sources." >&2
  echo "Rebuild first: cd ${ROOT_DIR}/GEMMsm110 && ./build_and_run.sh build-only" >&2
  echo "To intentionally use the stale binary, set ALLOW_STALE_BENCH=1." >&2
  exit 2
fi
if [[ ! -f "${SHAPES_CSV}" ]]; then
  echo "Shape CSV not found: ${SHAPES_CSV}" >&2
  exit 2
fi
if [[ ! "${BACKEND_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid BACKEND_TIMEOUT_SECONDS=${BACKEND_TIMEOUT_SECONDS}" >&2
  exit 2
fi
if [[ ! "${BACKEND_KILL_GRACE_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid BACKEND_KILL_GRACE_SECONDS=${BACKEND_KILL_GRACE_SECONDS}" >&2
  exit 2
fi
if [[ ! "${SWEEP_MIN_RATIO}" =~ ^([0-9]+)(\.[0-9]+)?$ ]]; then
  echo "Invalid SWEEP_MIN_RATIO=${SWEEP_MIN_RATIO}" >&2
  exit 2
fi
if [[ "${ENFORCE_MIN_RATIO}" != "0" && "${ENFORCE_MIN_RATIO}" != "1" ]]; then
  echo "Invalid ENFORCE_MIN_RATIO=${ENFORCE_MIN_RATIO} (expected 0 or 1)" >&2
  exit 2
fi

mkdir -p "${RAW_DIR}"
printf 'Category,M,N,K,Note,Epilogue,BackendId,Version,Precision,Reference,TimeMs,GFLOPS,RatioToReference,Matched,Status\n' > "${AGG_CSV}"

append_result() {
  local category="$1"
  local m="$2"
  local n="$3"
  local k="$4"
  local note="$5"
  local epilogue="$6"
  local status="$7"
  local csv_path="$8"

  if [[ -f "${csv_path}" ]]; then
    awk -F, -v category="${category}" -v m="${m}" -v n="${n}" -v k="${k}" \
      -v note="${note}" -v epilogue="${epilogue}" -v status="${status}" '
      NR == 1 { next }
      {
        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
          category, m, n, k, note, epilogue, $1, $2, $4, $5, $6, $7, $8, $9, status
      }
    ' "${csv_path}" >> "${AGG_CSV}"
  else
    printf '%s,%s,%s,%s,%s,%s,missing,missing,fp16->fp32,missing,0,0,0,0,%s\n' \
      "${category}" "${m}" "${n}" "${k}" "${note}" "${epilogue}" "${status}" >> "${AGG_CSV}"
  fi
}

tail -n +2 "${SHAPES_CSV}" | while IFS=, read -r category m n k note; do
  [[ -z "${category}" ]] && continue
  [[ "${category}" =~ ^# ]] && continue
  for epilogue in ${EPILOGUES}; do
    for backend in ${BACKENDS}; do
      run_dir="${RAW_DIR}/${category}_M${m}_N${n}_K${k}/${epilogue}/${backend}"
      mkdir -p "${run_dir}"
      echo "=== category=${category} M=${m} N=${n} K=${k} epilogue=${epilogue} backend=${backend} ==="
      set +e
      (
        cd "${run_dir}"
        timeout --foreground --signal=TERM \
          --kill-after="${BACKEND_KILL_GRACE_SECONDS}s" \
          "${BACKEND_TIMEOUT_SECONDS}s" \
          "${BENCH_BIN}" "${m}" "${n}" "${k}" "${backend}" "${epilogue}" > stdout.txt 2>&1
      )
      status=$?
      set -e
      if [[ "${status}" -eq 0 ]]; then
        append_result "${category}" "${m}" "${n}" "${k}" "${note}" "${epilogue}" "ok" \
          "${run_dir}/sgemm_sm110_benchmark.csv"
      elif [[ "${status}" -eq 124 || "${status}" -eq 137 ]]; then
        append_result "${category}" "${m}" "${n}" "${k}" "${note}" "${epilogue}" "timeout" \
          "${run_dir}/sgemm_sm110_benchmark.csv"
      else
        append_result "${category}" "${m}" "${n}" "${k}" "${note}" "${epilogue}" "failed_${status}" \
          "${run_dir}/sgemm_sm110_benchmark.csv"
      fi
    done
  done
done

analyze_args=(
  --csv "${AGG_CSV}"
  --out "${ANALYSIS_MD}"
  --min-ratio "${SWEEP_MIN_RATIO}"
  --reference-backends "${REFERENCE_BACKENDS}"
)
if [[ "${ENFORCE_MIN_RATIO}" == "1" ]]; then
  analyze_args+=(--fail-under)
fi

set +e
python3 "${PROJECT_DIR}/scripts/analyze_shape_sweep.py" "${analyze_args[@]}"
analysis_status=$?
set -e

echo "Wrote ${AGG_CSV}"
echo "Wrote ${ANALYSIS_MD}"
if [[ "${analysis_status}" -ne 0 ]]; then
  echo "One or more shapes failed the ${SWEEP_MIN_RATIO} non-reference performance gate." >&2
  exit "${analysis_status}"
fi
