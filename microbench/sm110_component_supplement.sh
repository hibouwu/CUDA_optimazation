#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  microbench/sm110_component_supplement.sh start SUPPLEMENT_ID BASE_SUITE_ID BASE_EXPECTED_COMMIT
  microbench/sm110_component_supplement.sh status SUPPLEMENT_ID
  microbench/sm110_component_supplement.sh finish SUPPLEMENT_ID

The supplement preserves an already audited base suite's compute/full-GEMM
evidence and collects only the current 14-case component campaign.  The final
composite import independently audits both source commits and platform
intervals; it does not relabel the two runs as one execution.
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

action=$1
supplement_id=$2
if [[ ! $supplement_id =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "invalid SUPPLEMENT_ID: $supplement_id" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
cd -- "$repo_root"

expected_branch=codex/thor-sm110-gemm-bounds-v2
suite_dir="results/sm110_closure_suite/$supplement_id"
contract_path="$suite_dir/run_contract.json"
component_run_id="$supplement_id-components"
component_dir="results/sm110_gemm_component_campaign/$component_run_id"
log_path="$suite_dir/supplement_launcher.log"

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
    if [[ $# -ne 4 ]]; then
      usage
      exit 2
    fi
    base_suite_id=$3
    base_expected_commit=$4
    if [[ ! $base_suite_id =~ ^[A-Za-z0-9._-]+$ ]]; then
      echo "invalid BASE_SUITE_ID: $base_suite_id" >&2
      exit 2
    fi
    if [[ ! $base_expected_commit =~ ^[0-9a-f]{40}$ ]]; then
      echo "invalid BASE_EXPECTED_COMMIT: $base_expected_commit" >&2
      exit 2
    fi
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
    if [[ -e $suite_dir || -e $component_dir ]]; then
      echo "refusing to reuse supplement evidence: $supplement_id" >&2
      exit 1
    fi

    base_suite_dir="results/sm110_closure_suite/$base_suite_id"
    python3 - "$base_suite_dir/run_contract.json" \
      "$base_suite_id" "$base_expected_commit" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text())
if (data.get("schema_version") != 1
        or data.get("suite_id") != sys.argv[2]
        or data.get("expected_commit") != sys.argv[3]
        or data.get("ncu_required") is not True):
    raise SystemExit("base run_contract.json does not match requested base evidence")
PY
    for required in \
      "$base_suite_dir/oc_after.tsv" \
      "$base_suite_dir/suite_launcher.log" \
      "results/sm110_epilogue_probe/$base_suite_id-epilogue-preflight/summary.json" \
      "results/sm110_gemm_campaign/$base_suite_id-compute/COMPLETE" \
      "results/sm110_full_gemm_campaign/$base_suite_id-full/COMPLETE"
    do
      if [[ ! -f $required ]]; then
        echo "base evidence is incomplete: $required" >&2
        exit 1
      fi
    done
    if ! grep -q '^SUITE_COMPLETE$' "$base_suite_dir/suite_launcher.log"; then
      echo "base suite is not complete: $base_suite_id" >&2
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
    python3 - "$contract_path" "$supplement_id" "$expected_branch" \
      "$expected_commit" "$base_suite_id" "$base_expected_commit" <<'PY'
import datetime
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 2,
    "kind": "component_supplement",
    "supplement_id": sys.argv[2],
    "component_run_id": f"{sys.argv[2]}-components",
    "expected_branch": sys.argv[3],
    "expected_commit": sys.argv[4],
    "base_suite_id": sys.argv[5],
    "base_expected_commit": sys.argv[6],
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
      printf '\n=== overcurrent counters before component supplement ===\n'
      grep "" /sys/class/hwmon/hwmon*/oc*_event_cnt || true
    } > "$suite_dir/preflight.txt" 2>&1
    capture_oc "$suite_dir/oc_before.tsv"

    nohup bash microbench/run_sm110_component_supplement.sh \
      "$supplement_id" "$expected_commit" > "$log_path" 2>&1 &
    pid=$!
    echo "$pid" > "$suite_dir/supplement_launcher.pid"
    printf 'supplement launched pid=%s\n' "$pid"
    printf 'expected_commit=%s\n' "$expected_commit"
    printf 'log=%s\n' "$log_path"
    ;;

  status)
    if [[ $# -ne 2 || ! -f $contract_path ]]; then
      usage
      exit 2
    fi
    printf '%s\n' '=== contract ==='
    cat "$contract_path"
    printf '%s\n' '=== supplement process ==='
    if [[ -f $suite_dir/supplement_launcher.pid ]]; then
      supplement_pid=$(<"$suite_dir/supplement_launcher.pid")
      ps -o pid,ppid,pgid,sid,state,etime,time,wchan:28,args \
        -p "$supplement_pid" || true
    else
      echo "supplement launcher PID is not present"
    fi
    printf '%s\n' '=== component state ==='
    if [[ -f $component_dir/campaign_status.json ]]; then
      cat "$component_dir/campaign_status.json"
    else
      echo "not_started"
    fi
    printf '%s\n' '=== supplement log tail ==='
    tail -n 80 "$log_path" || true
    ;;

  finish)
    if [[ $# -ne 2 || ! -f $contract_path ]]; then
      usage
      exit 2
    fi
    component_expected_commit=$(read_contract_field expected_commit)
    base_suite_id=$(read_contract_field base_suite_id)
    base_expected_commit=$(read_contract_field base_expected_commit)
    actual_commit=$(git rev-parse HEAD)
    if [[ $actual_commit != "$component_expected_commit" ]]; then
      echo "wrong checkout for finish: expected $component_expected_commit, got $actual_commit" >&2
      exit 1
    fi
    if ! grep -q '^COMPONENT_SUPPLEMENT_COMPLETE$' "$log_path"; then
      echo "component supplement is not complete; run status" >&2
      exit 1
    fi
    if [[ ! -f $suite_dir/oc_after.tsv ]]; then
      echo "missing immutable post-run counter snapshot" >&2
      exit 1
    fi
    printf '%s\n' '=== component supplement overcurrent delta ==='
    diff -u "$suite_dir/oc_before.tsv" "$suite_dir/oc_after.tsv" || true

    model_dir="results/sm110_model_closure/$supplement_id"
    mkdir -p "$model_dir"
    if [[ -f $model_dir/model_inputs.json ]]; then
      echo "using existing composite import: $model_dir/model_inputs.json"
    else
      python3 -m scripts.sm110_gemm_model.cli import-composite-closure \
        --repo-root . \
        --composite-id "$supplement_id" \
        --base-suite-id "$base_suite_id" \
        --base-expected-commit "$base_expected_commit" \
        --component-expected-commit "$component_expected_commit" \
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
    python3 -m scripts.sm110_gemm_model.cli report-closure \
      --closure-import "$model_dir/model_inputs.json" \
      --repo-root . \
      --capacities scripts/sm110_gemm_model/profiles/capacities.json \
      --hardware scripts/sm110_gemm_model/profiles/thor_sm110.json \
      --schedules scripts/sm110_gemm_model/examples/schedules.json \
      --output-json "$model_dir/closure_analysis.json" \
      --output-markdown "$model_dir/closure_summary.md" \
      | tee "$model_dir/report_audit.json"
    sha256sum \
      "results/sm110_closure_suite/$base_suite_id/preflight.txt" \
      "results/sm110_gemm_campaign/$base_suite_id-compute/summary.json" \
      "results/sm110_full_gemm_campaign/$base_suite_id-full/summary.json" \
      "$suite_dir/preflight.txt" \
      "$suite_dir/oc_before.tsv" \
      "$suite_dir/oc_after.tsv" \
      "$component_dir/summary.json" \
      "$model_dir/model_inputs.json" \
      "$model_dir/closure_analysis.json" \
      "$model_dir/closure_summary.md" \
      > "$model_dir/artifact_sha256.txt"
    printf 'model_inputs=%s\n' "$model_dir/model_inputs.json"
    ;;

  *)
    usage
    exit 2
    ;;
esac
