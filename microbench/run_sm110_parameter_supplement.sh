#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: microbench/run_sm110_parameter_supplement.sh RUN_ID EXPECTED_COMMIT

Runs the NCU-qualified TMA payload surface and the simultaneous hot-L2 plus
cold-DRAM-read/write-path proxy surface sequentially on Thor. Each detached
runner is waited for and audited before the next GPU campaign starts. The cold
proxy does not claim physical external write-byte closure.
EOF
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi
run_id=$1
expected_commit=$2
if [[ ! $run_id =~ ^[A-Za-z0-9._-]+$ || ! $expected_commit =~ ^[0-9a-f]{40}$ ]]; then
  usage
  exit 2
fi
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$repo_root"
if [[ $(git rev-parse HEAD) != "$expected_commit" ]]; then
  echo "wrong checkout: expected $expected_commit" >&2
  exit 1
fi
if [[ -n $(git status --short --untracked-files=no) ]]; then
  echo "tracked worktree changes are not allowed" >&2
  git status --short --untracked-files=no >&2
  exit 1
fi

wait_and_audit() {
  local label=$1
  local run_dir=$2
  local auditor=$3
  local pid_file="$run_dir/launcher.pid"
  if [[ ! -f $pid_file ]]; then
    echo "$label launcher did not create $pid_file" >&2
    exit 1
  fi
  local runner_pid
  runner_pid=$(<"$pid_file")
  if [[ ! $runner_pid =~ ^[0-9]+$ ]]; then
    echo "$label launcher wrote invalid PID: $runner_pid" >&2
    exit 1
  fi
  while kill -0 "$runner_pid" 2>/dev/null; do
    sleep 5
  done
  if [[ ! -f $run_dir/campaign_status.json ]]; then
    echo "$label has no final status" >&2
    tail -n 80 "$run_dir/launcher.log" >&2 || true
    exit 1
  fi
  local status
  status=$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1])).get("status", ""))' \
    "$run_dir/campaign_status.json")
  if [[ $status != complete ]]; then
    echo "$label ended with status=$status" >&2
    tail -n 80 "$run_dir/launcher.log" >&2 || true
    exit 1
  fi
  python3 "$auditor" "$run_dir"
}

tma_id="${run_id}-tma-payload"
tma_dir="results/sm110_tma_payload_campaign/$tma_id"
bash microbench/sm110_tma_payload_campaign/launch_tma_payload_campaign.sh \
  "$tma_id" "$expected_commit" --ncu --trial-timeout-seconds 120 \
  --ncu-timeout-seconds 300 --nvcc-host-undef-gnu-source
wait_and_audit tma-payload "$tma_dir" \
  microbench/sm110_tma_payload_campaign/audit_campaign.py

duplex_id="${run_id}-memory-duplex"
duplex_dir="results/sm110_memory_duplex_campaign/$duplex_id"
bash microbench/sm110_memory_duplex_campaign/launch_memory_duplex_campaign.sh \
  "$duplex_id" "$expected_commit" --trial-timeout-seconds 120 \
  --ncu-timeout-seconds 300 --nvcc-host-undef-gnu-source
wait_and_audit memory-duplex "$duplex_dir" \
  microbench/sm110_memory_duplex_campaign/audit_campaign.py

cat <<EOF
PARAMETER_SUPPLEMENT_COMPLETE
tma_payload_dir=$tma_dir
memory_duplex_dir=$duplex_dir
EOF
