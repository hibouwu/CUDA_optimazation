#!/usr/bin/env bash
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=armgpu
set -euo pipefail
: "${SLURM_JOB_ID:?run this launcher inside a valid ROMEO Slurm allocation}"
probe_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
romeo_load_armgpu_env
spack load cuda@12.9.0
export PYTHONDONTWRITEBYTECODE=1
exec python3 -B "$probe_script_dir/run_suite.py" "$@"
