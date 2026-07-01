#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEMM_ROOT="${ROOT_DIR}/GEMMsm110"
BUILD_DIR="${BUILD_DIR:-${GEMM_ROOT}/build}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/results/gemm_sm110}"
RAW_DIR="${OUT_DIR}/raw"
FIG_DIR="${OUT_DIR}/figures"
GEMM_SUITE="${GEMM_SUITE:-all}"
PRESET="${PRESET:-default}"
TRIALS="${TRIALS:-10}"
BACKEND_TIMEOUT_SECONDS="${BACKEND_TIMEOUT_SECONDS:-30}"
BACKEND_KILL_GRACE_SECONDS="${BACKEND_KILL_GRACE_SECONDS:-5}"
VERBOSE="${VERBOSE:-0}"
CUTLASS_ROOT="${CUTLASS_ROOT:-${ROOT_DIR}/../third_party/cutlass}"
NVCC="${NVCC:-nvcc}"

case "${GEMM_SUITE}" in
  all|references|stage0|stage1|stage2|stage3|stage4|stage5|\
  cublas_tc|cutlass|tc0|tc1a|tc1b|tc2a|tc2b|tc3|tc4a|tc4b|tc4c|tc5a|tc5b) ;;
  *)
    echo "Unknown GEMM_SUITE=${GEMM_SUITE}. Use all, references, stage0..stage5, or a concrete backend ID." >&2
    exit 1
    ;;
esac

case "${PRESET}" in
  quick)
    DEFAULT_GEMM_SIZES="128 256 512 1024"
    ;;
  default)
    DEFAULT_GEMM_SIZES="128 256 512 1024 2048 4096"
    ;;
  full)
    DEFAULT_GEMM_SIZES="128 256 512 1024 2048 4096"
    ;;
  *)
    echo "Unknown PRESET=${PRESET}. Use quick, default, full, or set GEMM_SIZES explicitly." >&2
    exit 1
    ;;
esac
GEMM_SIZES="${GEMM_SIZES:-${DEFAULT_GEMM_SIZES}}"

if [[ ! "${BACKEND_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid BACKEND_TIMEOUT_SECONDS=${BACKEND_TIMEOUT_SECONDS}. Expected a positive integer." >&2
  exit 1
fi
if [[ ! "${BACKEND_KILL_GRACE_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid BACKEND_KILL_GRACE_SECONDS=${BACKEND_KILL_GRACE_SECONDS}. Expected a positive integer." >&2
  exit 1
fi
if [[ "${VERBOSE}" != "0" && "${VERBOSE}" != "1" ]]; then
  echo "Invalid VERBOSE=${VERBOSE}. Use VERBOSE=0 or VERBOSE=1." >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}" "${RAW_DIR}"

ensure_python() {
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
    return
  fi
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
    return
  fi
  if [[ "${EUID}" -eq 0 ]] && command -v apt-get >/dev/null 2>&1; then
    echo "python3 not found, installing python3-minimal for plotting."
    apt-get update
    apt-get install -y python3-minimal
    PYTHON_BIN="python3"
    return
  fi
  echo "python3 is required for plotting. Install python3 or run inside the CUDA container as root." >&2
  exit 1
}

warn_render_access() {
  if [[ -e /dev/dri/renderD128 ]] && [[ ! -r /dev/dri/renderD128 ]]; then
    echo "Current shell cannot access /dev/dri/renderD128." >&2
    echo "If benchmark launch fails, rerun with: sg render -c 'bash ${BASH_SOURCE[0]}'" >&2
  fi
}

build_gemm_sm110() {
  if [[ ! -f "${CUTLASS_ROOT}/include/cutlass/cutlass.h" ]]; then
    echo "CUTLASS headers not found under ${CUTLASS_ROOT}" >&2
    echo "Set CUTLASS_ROOT to a CUTLASS 4.5.2 checkout." >&2
    exit 2
  fi

  "${NVCC}" \
    -O3 \
    -std=c++17 \
    --expt-relaxed-constexpr \
    -diag-suppress=20012 \
    -diag-suppress=20013 \
    -diag-suppress=20015 \
    -DTC3_SM110_HOST_HAS_TCGEN05=1 \
    -gencode arch=compute_110a,code=sm_110a \
    -I"${GEMM_ROOT}/include" \
    -I"${CUTLASS_ROOT}/include" \
    -I"${CUTLASS_ROOT}/tools/util/include" \
    "${GEMM_ROOT}/src/main.cu" \
    -lcuda \
    -lcublas \
    -o "${BUILD_DIR}/gemm_sm110_bench"
}

expand_suite_backends() {
  case "$1" in
    all)
      printf '%s\n' cublas_tc cutlass tc0 tc1a tc1b tc2a tc2b tc3 tc4a tc4b tc4c tc5a tc5b
      ;;
    references)
      printf '%s\n' cublas_tc cutlass
      ;;
    stage0)
      printf '%s\n' cublas_tc tc0
      ;;
    stage1)
      printf '%s\n' cublas_tc tc1a tc1b
      ;;
    stage2)
      printf '%s\n' cublas_tc tc2a tc2b
      ;;
    stage3)
      printf '%s\n' cublas_tc tc3
      ;;
    stage4)
      printf '%s\n' cublas_tc tc4a tc4b tc4c
      ;;
    stage5)
      printf '%s\n' cublas_tc tc5a tc5b
      ;;
    *)
      printf '%s\n' "$1"
      ;;
  esac
}

append_rows() {
  local csv_path="$1"
  local trial="$2"
  local include_reference="$3"

  if [[ ! -f "${csv_path}" ]]; then
    return
  fi

  awk -F, -v trial="${trial}" -v include_reference="${include_reference}" '
    NR == 1 { next }
    include_reference == "1" { print $0 "," trial; next }
    $1 != "cublas_tc" { print $0 "," trial }
  ' "${csv_path}" >> "${AGG_CSV}"
}

print_backend_summary() {
  local csv_path="$1"
  local backend="$2"

  awk -F, -v backend="${backend}" '
    $1 == backend {
      if ($9 != "1" && $6 == 0 && $7 == 0) {
        printf "%9s    | %11s        | %6s  | unavailable\n",
               "--", "--", "--"
        found = 1
        exit
      }
      printf "%9.4f ms | %11.1f GFLOPS | %6.3fx | matched=%s\n",
             $6, $7, $8, $9
      found = 1
      exit
    }
    END {
      if (!found) {
        print "result row missing"
      }
    }
  ' "${csv_path}"
}

run_backend_once() {
  local n="$1"
  local trial="$2"
  local backend="$3"
  local include_reference="$4"
  local run_dir="${RAW_DIR}/N${n}/trial_${trial}/${backend}"
  local status=0

  mkdir -p "${run_dir}"
  if [[ "${VERBOSE}" == "1" ]]; then
    echo "--- N=${n} backend=${backend} trial=${trial}/${TRIALS} ---"
  else
    printf '  trial %02d/%02d | %-9s | ' "${trial}" "${TRIALS}" "${backend}"
  fi

  set +e
  if [[ "${VERBOSE}" == "1" ]]; then
    (
      cd "${run_dir}"
      timeout --foreground --signal=TERM \
        --kill-after="${BACKEND_KILL_GRACE_SECONDS}s" \
        "${BACKEND_TIMEOUT_SECONDS}s" \
        "${BUILD_DIR}/gemm_sm110_bench" "${n}" "${backend}" 2>&1 |
        tee stdout.txt
    )
    status=$?
  else
    (
      cd "${run_dir}"
      timeout --foreground --signal=TERM \
        --kill-after="${BACKEND_KILL_GRACE_SECONDS}s" \
        "${BACKEND_TIMEOUT_SECONDS}s" \
        "${BUILD_DIR}/gemm_sm110_bench" "${n}" "${backend}" \
        > stdout.txt 2>&1
    )
    status=$?
  fi
  set -e

  append_rows "${run_dir}/sgemm_sm110_benchmark.csv" "${trial}" "${include_reference}"

  if [[ "${status}" -eq 124 || "${status}" -eq 137 ]]; then
    [[ "${VERBOSE}" == "1" ]] || echo "TIMEOUT"
    echo "Warning: backend ${backend} timed out after ${BACKEND_TIMEOUT_SECONDS}s for N=${n}, trial=${trial}." >&2
    [[ "${VERBOSE}" == "1" ]] || sed 's/^/    | /' "${run_dir}/stdout.txt" >&2
    return 124
  fi
  if [[ "${status}" -ne 0 ]]; then
    [[ "${VERBOSE}" == "1" ]] || echo "FAILED (exit ${status})"
    echo "Warning: backend ${backend} exited with status ${status} for N=${n}, trial=${trial}." >&2
    [[ "${VERBOSE}" == "1" ]] || sed 's/^/    | /' "${run_dir}/stdout.txt" >&2
    return "${status}"
  fi
  if [[ ! -f "${run_dir}/sgemm_sm110_benchmark.csv" ]]; then
    [[ "${VERBOSE}" == "1" ]] || echo "FAILED (CSV missing)"
    echo "Warning: backend ${backend} did not produce sgemm_sm110_benchmark.csv for N=${n}, trial=${trial}." >&2
    return 1
  fi

  if [[ "${VERBOSE}" == "1" ]]; then
    printf 'summary | %-9s | ' "${backend}"
  fi
  print_backend_summary "${run_dir}/sgemm_sm110_benchmark.csv" "${backend}"
}

run_trial() {
  local n="$1"
  local trial="$2"
  local backend
  local -a backends=()
  local include_reference

  mapfile -t backends < <(expand_suite_backends "${GEMM_SUITE}")
  for backend in "${backends[@]}"; do
    include_reference="1"
    if [[ "${#backends[@]}" -gt 1 && "${backend}" != "cublas_tc" ]]; then
      include_reference="0"
    fi
    if ! run_backend_once "${n}" "${trial}" "${backend}" "${include_reference}"; then
      HAD_FAILURES=1
    fi
  done
}

echo "=== Building GEMMsm110 benchmark ==="
build_gemm_sm110
ensure_python
warn_render_access

AGG_CSV="${OUT_DIR}/gemm_sm110_sweep.csv"
HAD_FAILURES=0
printf 'BackendId,Version,N,Precision,Reference,TimeMs,GFLOPS,RatioToReference,Matched,Trial\n' > "${AGG_CSV}"

echo "=== Sweep: suite=${GEMM_SUITE}, trials=${TRIALS}, sizes=${GEMM_SIZES} ==="
for n in ${GEMM_SIZES}; do
  echo
  echo "N=${n}"
  echo "  trial       | backend   |   time       | performance   | vs cuBLAS | check"
  for trial in $(seq 1 "${TRIALS}"); do
    run_trial "${n}" "${trial}"
  done
done

rm -rf "${FIG_DIR}"
mkdir -p "${FIG_DIR}"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/plot_benchmarks.py" \
  --gemm "${AGG_CSV}" \
  --out-dir "${FIG_DIR}"

echo "GEMMsm110 experiment data: ${OUT_DIR}"
echo "GEMMsm110 figures: ${FIG_DIR}"

if [[ "${HAD_FAILURES}" -ne 0 ]]; then
  echo "Completed with backend failures or timeouts. See ${RAW_DIR} for partial results." >&2
  exit 1
fi
