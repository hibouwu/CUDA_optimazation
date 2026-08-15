#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SUITE_ID EXPECTED_COMMIT" >&2
  exit 2
fi
suite_id=$1
expected_commit=$2
if [[ ! $suite_id =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "invalid SUITE_ID" >&2
  exit 2
fi
if [[ ! $expected_commit =~ ^[0-9a-f]{40}$ ]]; then
  echo "invalid EXPECTED_COMMIT" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$script_dir/.." && pwd)
cd -- "$repo"
if [[ $(git rev-parse HEAD) != "$expected_commit" ]]; then
  echo "wrong checkout in causal supervisor" >&2
  exit 1
fi

mkdir -p -- results
exec 9>results/sm110_campaign.lock
if ! flock -n 9; then
  echo "another SM110 evidence campaign holds results/sm110_campaign.lock" >&2
  exit 1
fi

run_id="$suite_id-causal"
suite_dir="results/sm110_causal_suite/$suite_id"
python3 microbench/sm110_gemm_causal_campaign/run_causal_campaign.py \
  --run-id "$run_id" --expected-commit "$expected_commit" --ncu
python3 microbench/sm110_gemm_causal_campaign/audit_campaign.py \
  "results/sm110_gemm_causal_campaign/$run_id" \
  --require-ncu --expected-commit "$expected_commit"
printf '%s\n' CAUSAL_CAMPAIGN_COMPLETE
if [[ -e $suite_dir/oc_after.tsv ]]; then
  echo "post-run OC snapshot already exists before supervisor completion" >&2
  exit 1
fi
found=0
: > "$suite_dir/oc_after.tsv"
for counter in /sys/class/hwmon/hwmon*/oc*_event_cnt; do
  if [[ -r $counter ]]; then
    printf '%s\t%s\n' "$counter" "$(<"$counter")" \
      >> "$suite_dir/oc_after.tsv"
    found=1
  fi
done
if [[ $found -ne 1 ]]; then
  echo "no readable post-run overcurrent counters" >&2
  exit 1
fi
python3 microbench/sm110_gemm_causal_campaign/audit_causal_suite.py \
  "$suite_dir" --expected-commit "$expected_commit"
printf '%s\n' CAUSAL_SUITE_COMPLETE
