#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  microbench/sm110_resource_supplement.sh start SUITE_ID
  microbench/sm110_resource_supplement.sh resume SUITE_ID
  microbench/sm110_resource_supplement.sh status SUITE_ID
  microbench/sm110_resource_supplement.sh finish SUITE_ID

Collects only the exact TMA resource contracts needed by the stricter model.
It does not rerun known L2 physical ceilings or the historical compute/full
GEMM campaigns.
EOF
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi
action=$1
suite_id=$2
if [[ ! $suite_id =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "invalid SUITE_ID" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$script_dir/.." && pwd)
cd -- "$repo"

expected_branch=codex/sm110-all-precision-closure
suite_dir="results/sm110_resource_suite/$suite_id"
contract_path="$suite_dir/run_contract.json"
resource_run_id="$suite_id-resources"
resource_dir="results/sm110_gemm_resource_campaign/$resource_run_id"
log_path="$suite_dir/suite_launcher.log"

capture_oc() {
  local output_path=$1
  local found=0
  : > "$output_path"
  for counter in /sys/class/hwmon/hwmon*/oc*_event_cnt; do
    if [[ -r $counter ]]; then
      printf '%s\t%s\n' "$counter" "$(<"$counter")" >> "$output_path"
      found=1
    fi
  done
  if [[ $found -ne 1 ]]; then
    echo "no readable overcurrent counters" >&2
    return 1
  fi
}

contract_commit() {
  python3 - "$contract_path" <<'PY'
import json
import pathlib
import sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text())["expected_commit"]
if not isinstance(value, str):
    raise SystemExit("expected_commit is not a string")
print(value)
PY
}

verify_platform() {
  nvpmodel_path=$(command -v nvpmodel || true)
  jetson_clocks_path=$(command -v jetson_clocks || true)
  ncu_path=$(command -v ncu || true)
  if [[ -z $nvpmodel_path || -z $jetson_clocks_path || -z $ncu_path ]]; then
    echo "nvpmodel, jetson_clocks, and ncu are required" >&2
    return 1
  fi
  power_mode=$($nvpmodel_path -q 2>&1)
  if [[ $power_mode != *"MAXN"* ]]; then
    echo "MAXN is not active" >&2
    printf '%s\n' "$power_mode" >&2
    return 1
  fi
  gpu_devfreq=/sys/class/devfreq/gpu-gpc-0
  for node in min_freq max_freq cur_freq governor; do
    if [[ ! -r $gpu_devfreq/$node ]]; then
      echo "missing readable GPU devfreq node: $gpu_devfreq/$node" >&2
      return 1
    fi
  done
  min_freq=$(<"$gpu_devfreq/min_freq")
  max_freq=$(<"$gpu_devfreq/max_freq")
  cur_freq=$(<"$gpu_devfreq/cur_freq")
  governor=$(<"$gpu_devfreq/governor")
  if [[ $min_freq != 1575000000 || $max_freq != 1575000000 \
        || $cur_freq != 1575000000 || $governor != performance ]]; then
    echo "GPU clock contract is not 1.575 GHz/performance" >&2
    printf 'min=%s max=%s cur=%s governor=%s\n' \
      "$min_freq" "$max_freq" "$cur_freq" "$governor" >&2
    return 1
  fi
}

capture_preflight() {
  local output_path=$1
  {
    printf 'captured_at_utc='
    date --utc --iso-8601=ns
    printf '\n=== git ===\n'
    git branch --show-current
    git rev-parse HEAD
    git status --short --untracked-files=no
    printf '\n=== nvpmodel ===\n'
    "$nvpmodel_path" -q
    printf '\n=== GPU devfreq ===\n'
    for node in available_frequencies min_freq max_freq cur_freq governor; do
      pathname="$gpu_devfreq/$node"
      if [[ -r $pathname ]]; then
        printf '%s=' "$node"
        tr -d '\n' < "$pathname"
        printf '\n'
      fi
    done
    printf '\n=== jetson_clocks ===\n'
    sudo "$jetson_clocks_path" --show
    printf '\n=== overcurrent counters ===\n'
    grep "" /sys/class/hwmon/hwmon*/oc*_event_cnt || true
  } > "$output_path" 2>&1
}

launch_supervisor() {
  local expected_commit=$1
  local redirection=$2
  if [[ $redirection == append ]]; then
    nohup bash microbench/run_sm110_resource_supplement.sh \
      "$suite_id" "$expected_commit" >> "$log_path" 2>&1 &
  else
    nohup bash microbench/run_sm110_resource_supplement.sh \
      "$suite_id" "$expected_commit" > "$log_path" 2>&1 &
  fi
  pid=$!
  echo "$pid" > "$suite_dir/suite_launcher.pid"
  printf 'resource supplement launched pid=%s\n' "$pid"
  printf 'expected_commit=%s\n' "$expected_commit"
  printf 'log=%s\n' "$log_path"
}

case "$action" in
  start)
    actual_branch=$(git branch --show-current)
    expected_commit=$(git rev-parse HEAD)
    if [[ $actual_branch != "$expected_branch" ]]; then
      echo "wrong branch: expected $expected_branch, got $actual_branch" >&2
      exit 1
    fi
    if [[ -n $(git status --short --untracked-files=no) ]]; then
      echo "tracked worktree changes are not allowed" >&2
      git status --short --untracked-files=no >&2
      exit 1
    fi
    if [[ -e $suite_dir || -e $resource_dir ]]; then
      echo "refusing to reuse evidence ID: $suite_id" >&2
      exit 1
    fi

    verify_platform

    mkdir -p -- "$suite_dir"
    python3 - "$contract_path" "$suite_id" "$resource_run_id" \
      "$expected_branch" "$expected_commit" <<'PY'
import datetime
import hashlib
import json
import pathlib
import sys
dependencies = [
    "microbench/sm110_resource_supplement.sh",
    "microbench/run_sm110_resource_supplement.sh",
    "microbench/sm110_gemm_resource_campaign/audit_resource_suite.py",
]
payload = {
    "schema_version": 1,
    "kind": "exact_resource_supplement",
    "suite_id": sys.argv[2],
    "resource_run_id": sys.argv[3],
    "expected_branch": sys.argv[4],
    "expected_commit": sys.argv[5],
    "ncu_required": True,
    "platform_dependencies": {
        path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
        for path in dependencies
    },
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
pathlib.Path(sys.argv[1]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY
    capture_preflight "$suite_dir/preflight.txt"
    capture_oc "$suite_dir/oc_before.tsv"
    launch_supervisor "$expected_commit" truncate
    ;;

  resume)
    if [[ ! -f $contract_path ]]; then
      echo "missing run contract: $contract_path" >&2
      exit 1
    fi
    expected_commit=$(contract_commit)
    if [[ $(git branch --show-current) != "$expected_branch" \
          || $(git rev-parse HEAD) != "$expected_commit" ]]; then
      echo "resume requires the frozen branch and commit" >&2
      exit 1
    fi
    if [[ -n $(git status --short --untracked-files=no) ]]; then
      echo "tracked worktree changes are not allowed" >&2
      exit 1
    fi
    if [[ -f $suite_dir/suite_launcher.pid ]]; then
      old_pid=$(<"$suite_dir/suite_launcher.pid")
      if [[ $old_pid =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
        echo "resource supervisor is still running pid=$old_pid" >&2
        exit 1
      fi
    fi
    if grep -q '^RESOURCE_SUPPLEMENT_COMPLETE$' "$log_path" 2>/dev/null; then
      echo "resource supplement is already complete; run finish" >&2
      exit 1
    fi
    if [[ -e $suite_dir/oc_after.tsv ]]; then
      echo "cannot resume after the OC interval was closed" >&2
      exit 1
    fi
    verify_platform
    resume_token=$(date --utc +%Y%m%dT%H%M%SZ)
    resume_preflight="$suite_dir/resume_preflight.$resume_token.txt"
    resume_oc="$suite_dir/oc_resume.$resume_token.tsv"
    if [[ -e $resume_preflight || -e $resume_oc ]]; then
      echo "resume snapshot collision; wait one second and retry" >&2
      exit 1
    fi
    capture_preflight "$resume_preflight"
    capture_oc "$resume_oc"
    python3 - "$suite_dir/oc_before.tsv" "$resume_oc" <<'PY'
import pathlib
import sys
def read(path):
    rows = {}
    for line in pathlib.Path(path).read_text().splitlines():
        name, value = line.split("\t")
        if name in rows:
            raise SystemExit("duplicate OC counter")
        rows[name] = int(value)
    return rows
before = read(sys.argv[1])
current = read(sys.argv[2])
if set(before) != set(current) or any(current[k] < before[k] for k in before):
    raise SystemExit("OC counters reset or changed; use a new suite ID")
PY
    printf 'RESOURCE_SUPPLEMENT_RESUME token=%s\n' "$resume_token" >> "$log_path"
    launch_supervisor "$expected_commit" append
    ;;

  status)
    if [[ ! -f $contract_path ]]; then
      echo "missing run contract: $contract_path" >&2
      exit 1
    fi
    printf '%s\n' '=== contract ==='
    cat "$contract_path"
    printf '%s\n' '=== process ==='
    if [[ -f $suite_dir/suite_launcher.pid ]]; then
      pid=$(<"$suite_dir/suite_launcher.pid")
      ps -o pid,ppid,pgid,sid,state,etime,time,wchan:28,args -p "$pid" || true
    else
      echo "suite launcher PID is not present"
    fi
    printf '%s\n' '=== campaign status ==='
    if [[ -f $resource_dir/campaign_status.json ]]; then
      cat "$resource_dir/campaign_status.json"
    else
      echo "not_started"
    fi
    printf '%s\n' '=== log tail ==='
    tail -n 80 "$log_path" || true
    ;;

  finish)
    if [[ ! -f $contract_path ]]; then
      echo "missing run contract: $contract_path" >&2
      exit 1
    fi
    expected_commit=$(contract_commit)
    actual_commit=$(git rev-parse HEAD)
    if [[ $actual_commit != "$expected_commit" ]]; then
      echo "wrong checkout: expected $expected_commit, got $actual_commit" >&2
      exit 1
    fi
    if ! grep -q '^RESOURCE_SUPPLEMENT_COMPLETE$' "$log_path"; then
      echo "resource supplement is not complete; run status" >&2
      exit 1
    fi
    if [[ -e $suite_dir/oc_after.tsv ]]; then
      echo "using existing immutable post-run OC snapshot"
    else
      capture_oc "$suite_dir/oc_after.tsv"
    fi
    python3 microbench/sm110_gemm_resource_campaign/audit_resource_suite.py \
      "$suite_dir" --expected-commit "$expected_commit" \
      | tee "$suite_dir/suite_audit.json"
    sha256sum \
      "$suite_dir/run_contract.json" \
      "$suite_dir/preflight.txt" \
      "$suite_dir/oc_before.tsv" \
      "$suite_dir/oc_after.tsv" \
      "$suite_dir/suite_launcher.log" \
      "$suite_dir/suite_audit.json" \
      "$resource_dir/artifact_sha256.txt" \
      "$resource_dir/summary.json" \
      > "$suite_dir/artifact_sha256.txt"
    printf '%s\n' '=== overcurrent delta ==='
    diff -u "$suite_dir/oc_before.tsv" "$suite_dir/oc_after.tsv" || true
    printf 'RESOURCE_SUITE_FINISHED suite_id=%s\n' "$suite_id"
    ;;

  *)
    usage
    exit 2
    ;;
esac
