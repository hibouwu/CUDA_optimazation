#!/usr/bin/env bash
set -euo pipefail
experiment_dir=${1:?Specify a new experiment directory}
cd "$experiment_dir"
test ! -e raw.jsonl || { printf 'Refusing to overwrite raw.jsonl\n' >&2; exit 1; }
romeo_load_armgpu_env
spack load cuda@12.9.0
hostname > environment.txt
date -u >> environment.txt
printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:-unknown}" >> environment.txt
nvcc --version >> environment.txt
nvidia-smi -q >> environment.txt
printf '%s\n' 'nvcc -std=c++17 -O3 -lineinfo -gencode arch=compute_90a,code=sm_90a -Xptxas=-v stress.cu -o stress' > compile_command.txt
nvcc -std=c++17 -O3 -lineinfo -gencode arch=compute_90a,code=sm_90a -Xptxas=-v stress.cu -o stress > compile.log 2>&1
cuobjdump --dump-sass stress > sass.txt
sha256sum stress.cu stress stress_cases.json sass.txt > SHA256SUMS
nvidia-smi --query-gpu=timestamp,uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv -lms 100 > telemetry.csv &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true' EXIT
timeout --signal=TERM --kill-after=5 220 ./stress > raw.jsonl 2> run.stderr
nvidia-smi -q > environment_after.txt
printf 'GH200_SUSTAINED_COMPLETE\n'
