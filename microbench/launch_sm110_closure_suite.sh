#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 SUITE_ID EXPECTED_COMMIT [--ncu]" >&2
  exit 2
fi

suite_id=$1
expected_commit=$2
if [[ ! $suite_id =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "invalid SUITE_ID: $suite_id" >&2
  exit 2
fi
if [[ ! $expected_commit =~ ^[0-9a-f]{40}$ ]]; then
  echo "invalid EXPECTED_COMMIT: $expected_commit" >&2
  exit 2
fi
if [[ $# -eq 3 && $3 != --ncu ]]; then
  echo "only --ncu is accepted as the optional argument" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
cd -- "$repo_root"

run_dir="results/sm110_closure_suite/$suite_id"
mkdir -p -- "$run_dir"
pid_file="$run_dir/suite_launcher.pid"
log_file="$run_dir/suite_launcher.log"

if [[ -f $pid_file ]]; then
  prior_pid=$(<"$pid_file")
  if [[ $prior_pid =~ ^[0-9]+$ ]] && kill -0 "$prior_pid" 2>/dev/null; then
    prior_command=""
    if [[ -r /proc/$prior_pid/cmdline ]]; then
      prior_command=$(tr '\0' ' ' < "/proc/$prior_pid/cmdline")
    fi
    if [[ $prior_command == *"run_sm110_closure_suite.sh"*"$suite_id"* ]]; then
      echo "suite already running pid=$prior_pid log=$log_file"
      exit 0
    fi
    echo "refusing to overwrite live unrelated PID $prior_pid from $pid_file" >&2
    echo "command: $prior_command" >&2
    exit 1
  fi
fi

command=(bash microbench/run_sm110_closure_suite.sh "$suite_id" "$expected_commit")
if [[ $# -eq 3 ]]; then
  command+=(--ncu)
fi

nohup "${command[@]}" >"$log_file" 2>&1 &
pid=$!
echo "$pid" > "$pid_file"

echo "suite launched pid=$pid"
echo "log=$log_file"
echo "Ctrl-C on 'tail -f $log_file' only stops log viewing; the suite stays running."
