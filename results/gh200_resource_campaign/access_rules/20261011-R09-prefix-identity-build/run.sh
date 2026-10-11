#!/usr/bin/env bash
set -euo pipefail
scontrol show job -dd "$SLURM_JOB_ID" > allocation.txt
nvcc --version > compiler.txt
python3 -c 'import os,json;print(json.dumps({k:os.environ.get(k) for k in ("SLURM_JOB_ID","CUDA_VISIBLE_DEVICES","SLURM_JOB_GPUS","SLURM_STEP_GPUS")}))' > allocation-env.json
mkdir -p build
flags=(-std=c++17 -O3 -DNDEBUG -gencode=arch=compute_90a,code=sm_90a -lineinfo --ptxas-options=-v -DV06_CFG=1 -Isource/probes -Isource/cutlass/include -Isource/cutlass/tools/util/include)
nvcc "${flags[@]}" -DR09_PLAIN_PREFIX_HOST source/probes/r18.cu -ldl -o build/plain_host >build/plain.log 2>&1
nvcc -Isource/overlay "${flags[@]}" -DR09_PLAIN_PREFIX_HOST -DV06_TRACE -DR18_DUAL_CLOCK source/probes/r18.cu -ldl -o build/dual_host >build/dual.log 2>&1
for name in plain_prefix.so plain_host dual_host; do cuobjdump --dump-sass build/$name >build/$name.sass; done
python3 - <<'CHECK'
import subprocess,json
for name in ['plain_host','dual_host']:
 data=subprocess.check_output(['strings','build/'+name])
 assert b'r09-plain-module-sw1-16/v1' in data
 assert b'trace-memset-device-sync-per-call' in data
print(json.dumps({'compiled_prefix_identity':True,'gpu_launch':False}))
CHECK
