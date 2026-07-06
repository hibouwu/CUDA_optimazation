#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BIN="${ROOT}/build/sass_register_bench"
STAGE="${STAGE:-all}"
OPCODES="${OPCODES:-all}"
BUILT=0

usage() {
  cat >&2 <<'EOF'
Usage:
  STAGE=all|main|tuple|physical OPCODES=all|lop3,imad,ffma ./scripts/run_opcode_suite.sh

Stages:
  main      LOP3 and FFMA stride bank scans
  tuple     LOP3 and IMAD wide tuple scans
  physical  LOP3 and IMAD source-slot/pressure probes
  all       all stages above
EOF
}

want_stage() {
  [[ "${STAGE}" == "all" || "${STAGE}" == "$1" ]]
}

want_opcode() {
  local opcode="$1"
  local normalized=" ${OPCODES//,/ } "
  [[ "${normalized}" == *" all "* || "${normalized}" == *" ${opcode} "* ]]
}

tuple_target() {
  if want_opcode lop3 && want_opcode imad; then
    echo all
  elif want_opcode lop3; then
    echo lop3
  elif want_opcode imad; then
    echo imad
  fi
}

run_step() {
  echo
  echo "== $* =="
  "$@"
}

ensure_build() {
  if [[ "${BUILT}" -eq 0 ]]; then
    run_step "${SCRIPT_DIR}/build.sh"
    BUILT=1
  fi
}

run_bench() {
  local result_dir="$1"
  local output="$2"
  local cubin_mode="$3"
  local iters="$4"
  local warmups="$5"
  local repeats="$6"
  shift 6

  local cubins=()
  if [[ "${cubin_mode}" == "main" ]]; then
    while IFS= read -r cubin; do
      cubins+=(--cubin "${cubin}")
    done < <(
      find "${result_dir}" -maxdepth 1 \
        \( -name 'B*.cubin' -o -name 'L*.cubin' -o -name 'M*.cubin' \
           -o -name 'T*.cubin' \) \
        -print | sort
    )
  else
    while IFS= read -r cubin; do
      cubins+=(--cubin "${cubin}")
    done < <(find "${result_dir}" -maxdepth 1 -name '*.cubin' -print | sort)
  fi

  "${BIN}" --iters "${iters}" --warmups "${warmups}" \
    --repeats "${repeats}" "${cubins[@]}" "$@" > "${output}"
  echo "Wrote ${output}"
}

run_main_scan() {
  local opcode="$1"
  shift
  local result_dir

  case "${opcode}" in
    lop3)
      result_dir="${ROOT}/results/bank_scan"
      ;;
    ffma)
      result_dir="${ROOT}/results/bank_scan_ffma"
      ;;
    *) return 1 ;;
  esac

  ensure_build
  run_step python3 "${SCRIPT_DIR}/patch_main_scan.py" --family "${opcode}"
  run_bench "${result_dir}" "${result_dir}/results.csv" main \
    "${ITERS:-20000}" "${WARMUPS:-3}" "${REPEATS:-10}" "$@"
  run_step python3 "${SCRIPT_DIR}/plot_main_scan.py" --family "${opcode}"
}

run_tuple_scan() {
  local family="$1"
  shift
  local result_dir="${ROOT}/results/tuple_scan_${family}"

  ensure_build
  run_step python3 "${SCRIPT_DIR}/patch_tuple_scan.py" \
    --family "${family}" --cases "${CASES:-40}"
  run_bench "${result_dir}" "${result_dir}/results.csv" all \
    "${ITERS:-5000}" "${WARMUPS:-2}" "${REPEATS:-5}" "$@"
  run_step python3 "${SCRIPT_DIR}/analyze_tuple_scan.py" "${family}"
}

run_physical_probe() {
  local family="$1"
  shift
  local result_dir="${ROOT}/results/physical_probe/${family}"

  ensure_build
  run_step python3 "${SCRIPT_DIR}/physical_probe.py" patch --family "${family}"
  run_bench "${result_dir}" "${result_dir}/results.csv" all \
    "${ITERS:-5000}" "${WARMUPS:-2}" "${REPEATS:-5}" "$@"
  run_step python3 "${SCRIPT_DIR}/physical_probe.py" analyze --family "${family}"
}

case "${STAGE}" in
  all|main|tuple|physical) ;;
  -h|--help) usage; exit 0 ;;
  *) usage; exit 1 ;;
esac

if want_stage main; then
  if want_opcode lop3; then
    run_main_scan lop3 "$@"
  fi
  if want_opcode ffma; then
    run_main_scan ffma "$@"
  fi
  if want_opcode imad; then
    echo "Skipping IMAD main scan: no full stride bank-scan template is validated."
  fi
fi

if want_stage tuple; then
  target="$(tuple_target)"
  case "${target}" in
    all) run_tuple_scan lop3 "$@"; run_tuple_scan imad "$@" ;;
    lop3|imad) run_tuple_scan "${target}" "$@" ;;
  esac
  if want_opcode ffma; then
    echo "Skipping FFMA tuple scan: current tuple harness covers LOP3/IMAD only."
  fi
fi

if want_stage physical; then
  if want_opcode lop3 || want_opcode imad; then
    ensure_build
    run_step python3 "${SCRIPT_DIR}/physical_probe.py" query-metrics || true
  fi
  target="$(tuple_target)"
  case "${target}" in
    all) run_physical_probe lop3 "$@"; run_physical_probe imad "$@" ;;
    lop3|imad) run_physical_probe "${target}" "$@" ;;
  esac
  if want_opcode ffma; then
    echo "Skipping FFMA physical probe: current physical harness covers LOP3/IMAD only."
  fi
fi
