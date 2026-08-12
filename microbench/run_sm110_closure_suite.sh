#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: microbench/run_sm110_closure_suite.sh SUITE_ID EXPECTED_COMMIT [--ncu]

Runs compute, component, and full-GEMM campaigns sequentially on Thor, waits
for each detached runner, and audits each result before starting the next one.
SUITE_ID may contain only letters, digits, dots, underscores, and hyphens.
EXPECTED_COMMIT is the exact 40-hex commit supplied with the campaign.
EOF
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
  exit 2
fi

suite_id=$1
if [[ ! $suite_id =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "invalid SUITE_ID: $suite_id" >&2
  exit 2
fi
expected_commit=$2
if [[ ! $expected_commit =~ ^[0-9a-f]{40}$ ]]; then
  echo "invalid EXPECTED_COMMIT: $expected_commit" >&2
  exit 2
fi

collect_ncu=0
if [[ $# -eq 3 ]]; then
  if [[ $3 != --ncu ]]; then
    usage
    exit 2
  fi
  collect_ncu=1
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
cd -- "$repo_root"

expected_branch=codex/thor-sm110-gemm-bounds
actual_branch=$(git branch --show-current)
actual_commit=$(git rev-parse HEAD)
if [[ $actual_branch != "$expected_branch" || $actual_commit != "$expected_commit" ]]; then
  echo "wrong checkout: expected $expected_branch@$expected_commit" >&2
  echo "actual checkout: $actual_branch@$actual_commit" >&2
  exit 1
fi
if [[ -n $(git status --short --untracked-files=no) ]]; then
  echo "tracked worktree changes are not allowed during evidence collection" >&2
  git status --short --untracked-files=no >&2
  exit 1
fi

compute_id="${suite_id}-compute"
component_id="${suite_id}-components"
full_id="${suite_id}-full"

wait_and_audit() {
  local label=$1
  local run_dir=$2
  local auditor=$3
  shift 3
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
  echo "$label running: pid=$runner_pid log=$run_dir/launcher.log"
  while kill -0 "$runner_pid" 2>/dev/null; do
    sleep 5
  done
  if [[ ! -f $run_dir/campaign_status.json ]]; then
    echo "$label has no campaign_status.json" >&2
    tail -n 80 "$run_dir/launcher.log" >&2 || true
    exit 1
  fi
  local status
  status=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$run_dir/campaign_status.json")
  if [[ $status != complete ]]; then
    echo "$label ended with status=$status" >&2
    tail -n 80 "$run_dir/launcher.log" >&2 || true
    exit 1
  fi
  python3 "$auditor" "$run_dir" "$@"
  echo "$label audit passed"
}

compute_args=()
compute_audit_args=()
full_args=()
if [[ $collect_ncu -eq 1 ]]; then
  compute_args+=(--ncu)
  compute_audit_args+=(--require-ncu)
  full_args+=(--ncu)
fi

bash microbench/sm110_gemm_campaign/launch_compute_campaign.sh \
  "$compute_id" "${compute_args[@]}"
wait_and_audit compute \
  "results/sm110_gemm_campaign/$compute_id" \
  microbench/sm110_gemm_campaign/audit_campaign.py \
  "${compute_audit_args[@]}"

bash microbench/sm110_gemm_component_campaign/launch_component_campaign.sh \
  "$component_id"
wait_and_audit component \
  "results/sm110_gemm_component_campaign/$component_id" \
  microbench/sm110_gemm_component_campaign/audit_campaign.py

bash microbench/sm110_full_gemm_campaign/launch_full_gemm_campaign.sh \
  "$full_id" "${full_args[@]}"
wait_and_audit full-gemm \
  "results/sm110_full_gemm_campaign/$full_id" \
  microbench/sm110_full_gemm_campaign/audit_campaign.py

cat <<EOF
SUITE_COMPLETE
compute_dir=results/sm110_gemm_campaign/$compute_id
component_dir=results/sm110_gemm_component_campaign/$component_id
full_gemm_dir=results/sm110_full_gemm_campaign/$full_id
EOF
