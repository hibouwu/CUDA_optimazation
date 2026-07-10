#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build}"
OUT_DIR="${OUT_DIR:-${ROOT}/ncu_reports}"
ITERS="${ITERS:-1000}"
FREQ_HZ="${FREQ_HZ:-}"
NCU_METRICS_FILE="${NCU_METRICS_FILE-}"
NCU_METRICS="${NCU_METRICS-}"
NCU_SET="${NCU_SET-}"
NCU_EXTRA_ARGS="${NCU_EXTRA_ARGS-}"
CASES="${CASES-}"
SHAPES="${SHAPES-}"
PRECISIONS="${PRECISIONS-}"
MAX_REPORTS="${MAX_REPORTS:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
DRY_RUN="${DRY_RUN:-0}"
FAIL_ON_NCU_ERROR="${FAIL_ON_NCU_ERROR:-0}"

default_cases=(
  ss_mma_only
  ts_mma_only
  ss_mma_mainloop_k2
  ss_mma_mainloop_k4
  ss_mma_mainloop_k8
  ss_mma_mainloop_k16
  ts_cp_mma_mainloop_a2_k2
  ts_cp_mma_mainloop_a2_k4
  ts_cp_mma_mainloop_a2_k8
  ts_cp_mma_mainloop_a2_k16
  tcgen05_cp_only
  ts_cp_mma_serial_a1
  ts_cp_mma_overlap_a2
  ts_cp_mma_warp_split_a2
)
default_shapes=(m128n256 m128n128 m128n64)
default_precisions=(fp4 fp8 bf16)

usage() {
  cat <<EOF
Usage:
  ./run_ncu_reports.sh

Environment filters:
  CASES="ss_mma_only ss_mma_mainloop_k16 ts_cp_mma_overlap_a2"
  SHAPES="m128n256"
  PRECISIONS="fp4 bf16"
  MAX_REPORTS=4
  SKIP_EXISTING=1
  DRY_RUN=1

NCU controls:
  ITERS=1000
  FREQ_HZ=1575000000
  OUT_DIR=./ncu_reports_key
  NCU_METRICS_FILE=./ncu_tcgen05_cp_mma_metrics.txt
  NCU_METRICS='regex:.*'
  NCU_SET=full
  NCU_EXTRA_ARGS='--replay-mode kernel'
  FAIL_ON_NCU_ERROR=1
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

split_words() {
  local value="$1"
  value="${value//,/ }"
  # shellcheck disable=SC2206
  local words=(${value})
  printf '%s\n' "${words[@]}"
}

if [[ -z "${NCU_METRICS}" && -z "${NCU_SET}" && -z "${NCU_METRICS_FILE}" ]]; then
  NCU_METRICS_FILE="${ROOT}/ncu_tcgen05_cp_mma_metrics.txt"
fi

if ! command -v ncu >/dev/null 2>&1; then
  echo "ERROR: ncu not found in PATH" >&2
  exit 1
fi

if [[ ! -d "${BUILD_DIR}" ]]; then
  echo "ERROR: build directory not found: ${BUILD_DIR}" >&2
  echo "Run python3 run_thor_tcgen05_cp_mma_report.py first to build benchmarks." >&2
  exit 1
fi

if [[ -z "${FREQ_HZ}" ]]; then
  if [[ -r /sys/class/devfreq/gpu-gpc-0/cur_freq ]]; then
    FREQ_HZ="$(cat /sys/class/devfreq/gpu-gpc-0/cur_freq)"
  else
    FREQ_HZ="1575000000"
  fi
fi

mkdir -p "${OUT_DIR}"
if [[ -n "${NCU_METRICS_FILE}" ]]; then
  if [[ ! -r "${NCU_METRICS_FILE}" ]]; then
    echo "ERROR: NCU_METRICS_FILE not readable: ${NCU_METRICS_FILE}" >&2
    exit 1
  fi
  NCU_METRICS="$(
    sed -e 's/#.*//' -e '/^[[:space:]]*$/d' "${NCU_METRICS_FILE}" \
      | tr '\n' ',' \
      | sed 's/,$//'
  )"
fi

if [[ -n "${CASES}" ]]; then
  mapfile -t cases < <(split_words "${CASES}")
else
  cases=("${default_cases[@]}")
fi
if [[ -n "${SHAPES}" ]]; then
  mapfile -t shapes < <(split_words "${SHAPES}")
else
  shapes=("${default_shapes[@]}")
fi
if [[ -n "${PRECISIONS}" ]]; then
  mapfile -t precisions < <(split_words "${PRECISIONS}")
else
  precisions=("${default_precisions[@]}")
fi

prefix_for_case() {
  local case_id="$1"
  if [[ "${case_id}" == tcgen05_* ]]; then
    printf '%s\n' "${case_id}"
  else
    printf 'tcgen05_%s\n' "${case_id}"
  fi
}

bins=()
for case_id in "${cases[@]}"; do
  prefix="$(prefix_for_case "${case_id}")"
  for shape in "${shapes[@]}"; do
    for precision in "${precisions[@]}"; do
      candidate="${BUILD_DIR}/${prefix}_${shape}_${precision}_benchmark"
      if [[ -x "${candidate}" ]]; then
        bins+=("${candidate}")
      else
        echo "WARN: missing benchmark binary: ${candidate}" >&2
      fi
    done
  done
done

if (( ${#bins[@]} == 0 )); then
  echo "ERROR: no benchmark binaries found in ${BUILD_DIR}" >&2
  echo "Run python3 run_thor_tcgen05_cp_mma_report.py first to build benchmarks." >&2
  exit 1
fi

count=0
failures=0
for bin in "${bins[@]}"; do
  name="$(basename "${bin}")"
  report="${OUT_DIR}/${name}.ncu-rep"
  log="${OUT_DIR}/${name}.log"
  if [[ "${SKIP_EXISTING}" == "1" && -s "${report}" && -s "${log}" ]]; then
    echo "skip existing ${name}"
    continue
  fi
  if (( MAX_REPORTS > 0 && count >= MAX_REPORTS )); then
    break
  fi
  count=$((count + 1))

  ncu_args=()
  if [[ -n "${NCU_METRICS}" ]]; then
    ncu_args+=(--metrics "${NCU_METRICS}")
  elif [[ -n "${NCU_SET}" ]]; then
    ncu_args+=(--set "${NCU_SET}")
  else
    ncu_args+=(--set full)
  fi
  if [[ -n "${NCU_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    extra_args=(${NCU_EXTRA_ARGS})
    ncu_args+=("${extra_args[@]}")
  fi

  echo "+ ncu ${name} -> ${report}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '  ncu'
    printf ' %q' "${ncu_args[@]}" --target-processes all --force-overwrite \
      --export "${report}" --print-details all --print-metric-name name \
      --print-units base --print-fp "${bin}" "${ITERS}" "${FREQ_HZ}"
    printf '\n'
    continue
  fi

  set +e
  ncu \
    "${ncu_args[@]}" \
    --target-processes all \
    --force-overwrite \
    --export "${report}" \
    --print-details all \
    --print-metric-name name \
    --print-units base \
    --print-fp \
    "${bin}" "${ITERS}" "${FREQ_HZ}" \
    >"${log}" 2>&1
  ncu_status=$?
  set -e
  if grep -q "ERR_NVGPUCTRPERM" "${log}"; then
    echo "  WARN: GPU performance counter permission denied; metrics were not collected." >&2
  fi
  if grep -q "No metrics to collect found in sections" "${log}"; then
    echo "  WARN: ncu found no collectable section metrics." >&2
  fi
  if grep -q "Unknown metric" "${log}"; then
    echo "  WARN: at least one requested NCU metric is unknown on this tool/driver." >&2
  fi
  if (( ncu_status != 0 )); then
    failures=$((failures + 1))
    echo "  WARN: ncu exited with status ${ncu_status}; see ${log}" >&2
    if [[ "${FAIL_ON_NCU_ERROR}" == "1" ]]; then
      exit "${ncu_status}"
    fi
  fi
  echo "  log: ${log}"
done

echo "NCU reports written to ${OUT_DIR}"
if (( failures > 0 )); then
  echo "NCU completed with ${failures} failed collection(s); logs were still written." >&2
fi
