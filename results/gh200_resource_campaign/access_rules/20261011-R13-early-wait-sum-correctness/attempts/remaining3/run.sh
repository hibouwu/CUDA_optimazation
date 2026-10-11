#!/bin/bash
set -euo pipefail
scontrol show job -dd "$SLURM_JOB_ID" >allocation.txt
python3 validate_allocation.py
python3 collect.py
