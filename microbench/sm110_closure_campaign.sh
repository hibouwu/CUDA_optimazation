#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  microbench/sm110_closure_campaign.sh start SUITE_ID
  microbench/sm110_closure_campaign.sh status SUITE_ID
  microbench/sm110_closure_campaign.sh finish SUITE_ID

start verifies the current checkout and Thor MAXN/clock contract, captures
preflight and overcurrent baselines, then launches the NCU-enabled closure
suite detached. finish captures post-run counters and independently imports
the audited campaign into model_inputs.json.
EOF
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

action=$1
suite_id=$2
if [[ ! $suite_id =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "invalid SUITE_ID: $suite_id" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
cd -- "$repo_root"

expected_branch=codex/thor-sm110-gemm-bounds-v2
suite_dir="results/sm110_closure_suite/$suite_id"
contract_path="$suite_dir/run_contract.json"

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

read_contract_field() {
  local field=$1
  python3 - "$contract_path" "$field" <<'PY'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())
value = data[sys.argv[2]]
if not isinstance(value, (str, int, float, bool)):
    raise SystemExit(f"contract field is not scalar: {sys.argv[2]}")
print(value)
PY
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
    if [[ -e $suite_dir ]]; then
      echo "refusing to reuse evidence directory: $suite_dir" >&2
      echo "choose a new SUITE_ID" >&2
      exit 1
    fi

    nvpmodel_path=$(command -v nvpmodel || true)
    jetson_clocks_path=$(command -v jetson_clocks || true)
    if [[ -z $nvpmodel_path || -z $jetson_clocks_path ]]; then
      echo "nvpmodel and jetson_clocks must both be available" >&2
      exit 1
    fi
    power_mode=$($nvpmodel_path -q 2>&1)
    if [[ $power_mode != *"MAXN"* ]]; then
      echo "MAXN is not active; configure the board before start" >&2
      printf '%s\n' "$power_mode" >&2
      exit 1
    fi
    gpu_devfreq=/sys/class/devfreq/gpu-gpc-0
    for node in min_freq max_freq cur_freq governor; do
      if [[ ! -r $gpu_devfreq/$node ]]; then
        echo "missing readable GPU devfreq node: $gpu_devfreq/$node" >&2
        exit 1
      fi
    done
    min_freq=$(<"$gpu_devfreq/min_freq")
    max_freq=$(<"$gpu_devfreq/max_freq")
    cur_freq=$(<"$gpu_devfreq/cur_freq")
    governor=$(<"$gpu_devfreq/governor")
    if [[ $min_freq != 1575000000 || $max_freq != 1575000000 \
          || $cur_freq != 1575000000 || $governor != performance ]]; then
      echo "GPU clock contract is not locked at 1.575 GHz/performance" >&2
      printf 'min=%s max=%s cur=%s governor=%s\n' \
        "$min_freq" "$max_freq" "$cur_freq" "$governor" >&2
      exit 1
    fi

    mkdir -p -- "$suite_dir"
    python3 - "$contract_path" "$suite_id" "$expected_branch" \
      "$expected_commit" <<'PY'
import datetime
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "suite_id": sys.argv[2],
    "expected_branch": sys.argv[3],
    "expected_commit": sys.argv[4],
    "ncu_required": True,
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

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
      printf '\n=== overcurrent counters before campaign ===\n'
      grep "" /sys/class/hwmon/hwmon*/oc*_event_cnt || true
    } > "$suite_dir/preflight.txt" 2>&1
    capture_oc "$suite_dir/oc_before.tsv"

    bash microbench/launch_sm110_closure_suite.sh \
      "$suite_id" "$expected_commit" --ncu
    printf 'expected_commit=%s\n' "$expected_commit"
    printf 'runbook=Docs/blackwell_tensorcore/THOR_CLOSURE_RUNBOOK.md\n'
    ;;

  status)
    if [[ ! -f $contract_path ]]; then
      echo "missing run contract: $contract_path" >&2
      exit 1
    fi
    printf '%s\n' '=== contract ==='
    cat "$contract_path"
    printf '%s\n' '=== suite process ==='
    if [[ -f $suite_dir/suite_launcher.pid ]]; then
      suite_pid=$(<"$suite_dir/suite_launcher.pid")
      ps -o pid,ppid,pgid,sid,state,etime,time,wchan:28,args \
        -p "$suite_pid" || true
    else
      echo "suite launcher PID is not present"
    fi
    printf '%s\n' '=== campaign states ==='
    python3 - "$suite_id" <<'PY'
import json
import pathlib
import sys

suite_id = sys.argv[1]
campaigns = {
    "epilogue_preflight": pathlib.Path(
        f"results/sm110_epilogue_probe/{suite_id}-epilogue-preflight/summary.json"),
    "compute": pathlib.Path(
        f"results/sm110_gemm_campaign/{suite_id}-compute/campaign_status.json"),
    "component": pathlib.Path(
        f"results/sm110_gemm_component_campaign/{suite_id}-components/campaign_status.json"),
    "full_gemm": pathlib.Path(
        f"results/sm110_full_gemm_campaign/{suite_id}-full/campaign_status.json"),
}
for name, path in campaigns.items():
    if not path.is_file():
        print(f"{name}: not_started")
        continue
    data = json.loads(path.read_text())
    state = data.get("status", data.get("pass", "unknown"))
    completed = data.get("completed_cases")
    total = data.get("total_cases")
    current = data.get("current_case")
    suffix = ""
    if completed is not None:
        suffix += f" completed={completed}/{total}"
    if current:
        suffix += f" current={current}"
    print(f"{name}: {state}{suffix}")
PY
    printf '%s\n' '=== suite log tail ==='
    tail -n 60 "$suite_dir/suite_launcher.log" || true
    ;;

  finish)
    if [[ ! -f $contract_path ]]; then
      echo "missing run contract: $contract_path" >&2
      exit 1
    fi
    expected_commit=$(read_contract_field expected_commit)
    actual_commit=$(git rev-parse HEAD)
    if [[ $actual_commit != "$expected_commit" ]]; then
      echo "wrong checkout for finish: expected $expected_commit, got $actual_commit" >&2
      exit 1
    fi
    if ! grep -q '^SUITE_COMPLETE$' "$suite_dir/suite_launcher.log"; then
      echo "suite is not complete; run the status command" >&2
      exit 1
    fi
    if [[ -e $suite_dir/oc_after.tsv ]]; then
      echo "using existing immutable post-run counter evidence: $suite_dir/oc_after.tsv"
    else
      capture_oc "$suite_dir/oc_after.tsv"
    fi
    printf '%s\n' '=== overcurrent delta ==='
    diff -u "$suite_dir/oc_before.tsv" "$suite_dir/oc_after.tsv" || true

    model_dir="results/sm110_model_closure/$suite_id"
    mkdir -p "$model_dir"
    if [[ -f $model_dir/model_inputs.json ]]; then
      echo "using existing model import: $model_dir/model_inputs.json"
    else
      python3 -m scripts.sm110_gemm_model.cli import-closure \
        --repo-root . \
        --suite-id "$suite_id" \
        --expected-commit "$expected_commit" \
        --output "$model_dir/model_inputs.json" \
        | tee "$model_dir/import_audit.json"
    fi
    python3 -m scripts.sm110_gemm_model.cli audit \
      --repo-root . \
      --capacities scripts/sm110_gemm_model/profiles/capacities.json \
      --closure-import "$model_dir/model_inputs.json" \
      | tee "$model_dir/model_audit.json"
    python3 -m scripts.sm110_gemm_model.cli coverage \
      --repo-root . \
      --capacities scripts/sm110_gemm_model/profiles/capacities.json \
      --closure-import "$model_dir/model_inputs.json" \
      | tee "$model_dir/coverage.json"
    sha256sum \
      "$suite_dir/preflight.txt" \
      "$suite_dir/oc_before.tsv" \
      "$suite_dir/oc_after.tsv" \
      "results/sm110_gemm_campaign/$suite_id-compute/summary.json" \
      "results/sm110_gemm_component_campaign/$suite_id-components/summary.json" \
      "results/sm110_full_gemm_campaign/$suite_id-full/summary.json" \
      "$model_dir/model_inputs.json" \
      > "$model_dir/artifact_sha256.txt"
    printf 'model_inputs=%s\n' "$model_dir/model_inputs.json"
    ;;

  *)
    usage
    exit 2
    ;;
esac
