#!/usr/bin/env bash
set -euo pipefail
probe_output="$1"
probe_script_dir="$(cd "$(dirname "$BASH_SOURCE")" && pwd)"
test -n "$SLURM_JOB_ID"
romeo_load_armgpu_env
spack load cuda@12.9.0
exec python3 "$probe_script_dir/run_campaign.py" --output "$probe_output"
