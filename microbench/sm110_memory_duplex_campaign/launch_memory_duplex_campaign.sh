#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 RUN_ID EXPECTED_COMMIT [runner arguments...]" >&2
  exit 2
fi
run_id=$1
expected_commit=$2
shift 2
if [[ ! $run_id =~ ^[A-Za-z0-9._-]+$ || ! $expected_commit =~ ^[0-9a-f]{40}$ ]]; then
  echo "invalid run ID or expected commit" >&2
  exit 2
fi
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd -- "$repo_root"
run_dir="results/sm110_memory_duplex_campaign/$run_id"
mkdir -p -- "$run_dir"
nohup python3 microbench/sm110_memory_duplex_campaign/run_memory_duplex_campaign.py \
  --run-id "$run_id" --expected-commit "$expected_commit" "$@" \
  >"$run_dir/launcher.log" 2>&1 &
runner_pid=$!
echo "$runner_pid" >"$run_dir/launcher.pid"
echo "launched run_id=$run_id pid=$runner_pid log=$run_dir/launcher.log"
