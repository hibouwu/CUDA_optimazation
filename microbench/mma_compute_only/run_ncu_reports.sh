#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT}/build"
OUT_DIR="${OUT_DIR:-${ROOT}/ncu_reports}"
ITERS="${ITERS:-1000}"
FREQ_HZ="${FREQ_HZ:-}"
NCU_METRICS_FILE="${NCU_METRICS_FILE-}"
NCU_METRICS="${NCU_METRICS-}"
NCU_SET="${NCU_SET-}"
NCU_EXTRA_ARGS="${NCU_EXTRA_ARGS-}"

if [[ -z "${NCU_METRICS}" && -z "${NCU_SET}" && -z "${NCU_METRICS_FILE}" ]]; then
  NCU_METRICS_FILE="${ROOT}/ncu_mma_anomaly_metrics.txt"
fi

if ! command -v ncu >/dev/null 2>&1; then
  echo "ERROR: ncu not found in PATH" >&2
  exit 1
fi

if [[ ! -d "${BUILD_DIR}" ]]; then
  echo "ERROR: build directory not found: ${BUILD_DIR}" >&2
  echo "Run ./run_thor_tcgen05_report.py first to build benchmarks." >&2
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
shopt -s nullglob
shapes=(m128n256 m128n128 m128n64)
launches=(single_warp_block full_sm_4warp_block)
precisions=(fp4 fp8 bf16)
bins=()
for launch in "${launches[@]}"; do
  for shape in "${shapes[@]}"; do
    for precision in "${precisions[@]}"; do
      candidate="${BUILD_DIR}/tcgen05_${launch}_${shape}_${precision}_benchmark"
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
  echo "Run ./run_thor_tcgen05_report.py first to build benchmarks." >&2
  exit 1
fi

for bin in "${bins[@]}"; do
  name="$(basename "${bin}")"
  report="${OUT_DIR}/${name}.ncu-rep"
  log="${OUT_DIR}/${name}.log"
  echo "+ ncu ${name} -> ${report}"
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
  if grep -q "ERR_NVGPUCTRPERM" "${log}"; then
    echo "  WARN: GPU performance counter permission denied; metrics were not collected." >&2
  fi
  if grep -q "No metrics to collect found in sections" "${log}"; then
    echo "  WARN: ncu found no collectable section metrics." >&2
  fi
  echo "  log: ${log}"
done

echo "NCU reports written to ${OUT_DIR}"
