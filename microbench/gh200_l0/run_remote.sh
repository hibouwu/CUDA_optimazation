#!/usr/bin/env bash
set -euo pipefail
experiment_dir=${1:?请指定含 probe.cu 的新远程实验目录}
cd "$experiment_dir"
if [[ -e raw.jsonl ]]; then
  printf '已有 raw.jsonl；请使用新的实验目录。\n' >&2
  exit 1
fi
romeo_load_armgpu_env
spack load cuda@12.9.0
hostname > environment.txt
date -u >> environment.txt
nvcc --version >> environment.txt
nvidia-smi -q >> environment.txt
printf '%s\n' 'nvcc -std=c++17 -O3 -lineinfo -gencode arch=compute_90a,code=sm_90a -Xptxas=-v probe.cu -o probe' > compile_command.txt
nvcc -std=c++17 -O3 -lineinfo -gencode arch=compute_90a,code=sm_90a -Xptxas=-v probe.cu -o probe > compile.log 2>&1
cuobjdump --dump-sass probe > sass.txt
sha256sum probe.cu probe sass.txt > SHA256SUMS
# Read-only telemetry; no clock/power changes on the shared node.
nvidia-smi --query-gpu=timestamp,uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv -lms 100 > telemetry.csv &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true' EXIT
timeout 180 ./probe 7 > raw.jsonl 2> run.stderr
nvidia-smi -q > environment_after.txt
printf 'GH200_L0_COMPLETE\n'
