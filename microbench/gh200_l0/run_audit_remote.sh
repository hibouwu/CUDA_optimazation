#!/usr/bin/env bash
set -euo pipefail
cd "${1:?New experiment directory required}"
test ! -e raw.jsonl || exit 1
romeo_load_armgpu_env
spack load cuda@12.9.0
hostname > environment.txt
date -u >> environment.txt
date +%z >> environment.txt
printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:-unknown}" >> environment.txt
nvcc --version >> environment.txt
nvidia-smi -q >> environment.txt
printf '%s\n' 'nvcc -std=c++17 -O3 -lineinfo -gencode arch=compute_90a,code=sm_90a -Xptxas=-v audit.cu -o audit' > compile_command.txt
nvcc -std=c++17 -O3 -lineinfo -gencode arch=compute_90a,code=sm_90a -Xptxas=-v audit.cu -o audit > compile.log 2>&1
cuobjdump --dump-sass audit > sass.txt
sha256sum audit audit.cu audit_cases.json sass.txt run_audit_remote.sh > SHA256SUMS
nvidia-smi --query-gpu=timestamp,uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv -lms 50 > telemetry.csv &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true' EXIT
timeout --signal=TERM --kill-after=5 210 ./audit > raw.jsonl 2> run.stderr
kill "$telemetry_pid" 2>/dev/null || true
wait "$telemetry_pid" 2>/dev/null || true
trap - EXIT
nvidia-smi -q > environment_after.txt
if command -v ncu > /dev/null; then
  ncu --version > ncu_version.txt
  set +e
  timeout --signal=TERM --kill-after=5 25 ncu --clock-control none --cache-control none --metrics smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__inst_executed_pipe_tensor.sum --csv --log-file ncu_fma.csv ./audit 0 > ncu_fma.stdout 2> ncu_fma.stderr
  printf '%s\n' "$?" > ncu_fma.exit
  timeout --signal=TERM --kill-after=5 25 ncu --clock-control none --cache-control none --metrics sm__inst_executed_pipe_tensor.sum --csv --log-file ncu_wgmma.csv ./audit 6 > ncu_wgmma.stdout 2> ncu_wgmma.stderr
  printf '%s\n' "$?" > ncu_wgmma.exit
  set -e
fi
printf 'GH200_AUDIT_COMPLETE\n'
