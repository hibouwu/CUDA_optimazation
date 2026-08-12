#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 RUN_ID [runner arguments...]" >&2
  exit 2
fi

run_id=$1
shift
if [[ ! $run_id =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "invalid RUN_ID: $run_id" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
run_dir="$repo_root/results/sm110_gemm_campaign/$run_id"
mkdir -p -- "$run_dir"
pid_file="$run_dir/launcher.pid"
if [[ -f $pid_file ]]; then
  prior_pid=$(<"$pid_file")
  if [[ $prior_pid =~ ^[0-9]+$ ]] && kill -0 "$prior_pid" 2>/dev/null; then
    echo "campaign already running: run_id=$run_id pid=$prior_pid" >&2
    exit 1
  fi
fi

cd -- "$repo_root"
nohup python3 microbench/sm110_gemm_campaign/run_compute_campaign.py \
  --run-id "$run_id" --trials 10 --iters 10000 "$@" \
  >"$run_dir/launcher.log" 2>&1 &
runner_pid=$!
echo "$runner_pid" >"$pid_file"
echo "launched run_id=$run_id pid=$runner_pid"
echo "log=$run_dir/launcher.log"
