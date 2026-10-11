set -euo pipefail
scontrol show job -dd "$SLURM_JOB_ID" > allocation.txt
nvcc --version > compiler.txt
python3 collect.py
