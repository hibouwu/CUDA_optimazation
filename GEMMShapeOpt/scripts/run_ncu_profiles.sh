#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_DIR="${ROOT_DIR}/GEMMShapeOpt"
BENCH_BIN="${BENCH_BIN:-${ROOT_DIR}/GEMMsm110/build/gemm_sm110_bench}"
CASES_CSV="${CASES_CSV:-${PROJECT_DIR}/profiles/ncu_cases.csv}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/results/gemm_shape_opt/ncu/$(date +%Y%m%d_%H%M%S)}"
NCU_BIN="${NCU_BIN:-ncu}"
NCU_SECTIONS="${NCU_SECTIONS:-SpeedOfLight MemoryWorkloadAnalysis MemoryWorkloadAnalysis_Tables SchedulerStats WarpStateStats Occupancy LaunchStats InstructionStats}"
NCU_REPLAY_MODE="${NCU_REPLAY_MODE:-kernel}"

if [[ ! -x "${BENCH_BIN}" ]]; then
  echo "Benchmark binary not found or not executable: ${BENCH_BIN}" >&2
  exit 2
fi
if [[ ! -f "${CASES_CSV}" ]]; then
  echo "NCU cases CSV not found: ${CASES_CSV}" >&2
  exit 2
fi
if ! command -v "${NCU_BIN}" >/dev/null 2>&1; then
  echo "ncu not found: ${NCU_BIN}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
STATUS_CSV="${OUT_DIR}/ncu_status.csv"
printf 'Suite,Case,M,N,K,Backend,Epilogue,Status,Report,Details,RawCsv,Notes\n' > "${STATUS_CSV}"

section_args=()
for section in ${NCU_SECTIONS}; do
  section_args+=(--section "${section}")
done

tail -n +2 "${CASES_CSV}" | while IFS=, read -r suite case_name m n k backend epilogue kernel_name launch_skip launch_count notes; do
  [[ -z "${suite}" ]] && continue
  [[ "${suite}" =~ ^# ]] && continue

  case_dir="${OUT_DIR}/${suite}/${case_name}"
  mkdir -p "${case_dir}"
  export_base="${case_dir}/profile"
  details_txt="${case_dir}/details.txt"
  raw_csv="${case_dir}/raw.csv"
  stdout_txt="${case_dir}/stdout.txt"

  ncu_args=(
    --force-overwrite
    --target-processes all
    --kernel-name-base demangled
    --launch-skip "${launch_skip:-0}"
    --launch-count "${launch_count:-1}"
    --replay-mode "${NCU_REPLAY_MODE}"
    "${section_args[@]}"
    --export "${export_base}"
    --log-file "${details_txt}"
  )
  if [[ -n "${kernel_name}" ]]; then
    ncu_args+=(--kernel-name "${kernel_name}")
  fi

  echo "=== NCU ${suite}/${case_name}: M=${m} N=${n} K=${k} backend=${backend} epilogue=${epilogue} ==="
  set +e
  "${NCU_BIN}" "${ncu_args[@]}" "${BENCH_BIN}" "${m}" "${n}" "${k}" "${backend}" "${epilogue}" > "${stdout_txt}" 2>&1
  status=$?
  set -e

  report_path="${export_base}.ncu-rep"
  if [[ -f "${export_base}" && ! -f "${report_path}" ]]; then
    report_path="${export_base}"
  fi
  if [[ "${status}" -eq 0 && -f "${report_path}" ]]; then
    "${NCU_BIN}" --import "${report_path}" --csv --page raw --log-file "${raw_csv}" >/dev/null 2>&1 || true
    printf '%s,%s,%s,%s,%s,%s,%s,ok,%s,%s,%s,%s\n' \
      "${suite}" "${case_name}" "${m}" "${n}" "${k}" "${backend}" "${epilogue}" \
      "${report_path}" "${details_txt}" "${raw_csv}" "${notes}" >> "${STATUS_CSV}"
  else
    printf '%s,%s,%s,%s,%s,%s,%s,failed_%s,%s,%s,%s,%s\n' \
      "${suite}" "${case_name}" "${m}" "${n}" "${k}" "${backend}" "${epilogue}" \
      "${status}" "${report_path}" "${details_txt}" "${raw_csv}" "${notes}" >> "${STATUS_CSV}"
  fi
done

echo "Wrote ${STATUS_CSV}"
