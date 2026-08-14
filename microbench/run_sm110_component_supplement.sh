#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SUPPLEMENT_ID EXPECTED_COMMIT" >&2
  exit 2
fi

supplement_id=$1
expected_commit=$2
if [[ ! $supplement_id =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "invalid SUPPLEMENT_ID: $supplement_id" >&2
  exit 2
fi
if [[ ! $expected_commit =~ ^[0-9a-f]{40}$ ]]; then
  echo "invalid EXPECTED_COMMIT: $expected_commit" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
cd -- "$repo_root"

suite_dir="results/sm110_closure_suite/$supplement_id"
component_run_id="$supplement_id-components"
component_dir="results/sm110_gemm_component_campaign/$component_run_id"

capture_oc_once() {
  local output_path="$suite_dir/oc_after.tsv"
  local found=0
  if [[ -e $output_path ]]; then
    return 0
  fi
  : > "$output_path"
  for counter in /sys/class/hwmon/hwmon*/oc*_event_cnt; do
    if [[ -r $counter ]]; then
      printf '%s\t%s\n' "$counter" "$(<"$counter")" >> "$output_path"
      found=1
    fi
  done
  if [[ $found -ne 1 ]]; then
    echo "no readable overcurrent counters for post-run snapshot" >&2
    return 1
  fi
}

trap 'capture_oc_once || true' EXIT

actual_commit=$(git rev-parse HEAD)
if [[ $actual_commit != "$expected_commit" ]]; then
  echo "wrong checkout: expected $expected_commit, got $actual_commit" >&2
  exit 1
fi

python3 microbench/sm110_gemm_component_campaign/run_component_campaign.py \
  --run-id "$component_run_id" \
  --trial-timeout-seconds 120 \
  --nvcc-host-undef-gnu-source
campaign_status=$?
capture_oc_once || exit 1
if [[ $campaign_status -ne 0 ]]; then
  echo "COMPONENT_SUPPLEMENT_FAILED returncode=$campaign_status" >&2
  exit "$campaign_status"
fi

python3 microbench/sm110_gemm_component_campaign/audit_campaign.py \
  "$component_dir" \
  | tee "$suite_dir/component_audit.json"
audit_status=${PIPESTATUS[0]}
if [[ $audit_status -ne 0 ]]; then
  echo "COMPONENT_SUPPLEMENT_AUDIT_FAILED returncode=$audit_status" >&2
  exit "$audit_status"
fi

printf '%s\n' COMPONENT_SUPPLEMENT_COMPLETE
