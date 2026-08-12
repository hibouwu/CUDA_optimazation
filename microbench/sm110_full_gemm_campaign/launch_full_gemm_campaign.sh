#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "usage: $0 RUN_ID [runner arguments...]" >&2
  exit 2
fi
run_id=$1
shift
if [[ ! $run_id =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "invalid RUN_ID" >&2
  exit 2
fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$script_dir/../.." && pwd)
run_dir="$repo/results/sm110_full_gemm_campaign/$run_id"
mkdir -p -- "$run_dir"
if [[ -f $run_dir/launcher.pid ]]; then
  pid=$(<"$run_dir/launcher.pid")
  if [[ $pid =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    echo "already running pid=$pid" >&2
    exit 1
  fi
fi
cd -- "$repo"
nohup python3 microbench/sm110_full_gemm_campaign/run_full_gemm_campaign.py \
  --run-id "$run_id" "$@" > "$run_dir/launcher.log" 2>&1 &
pid=$!
echo "$pid" > "$run_dir/launcher.pid"
echo "launched run_id=$run_id pid=$pid"
echo "log=$run_dir/launcher.log"
