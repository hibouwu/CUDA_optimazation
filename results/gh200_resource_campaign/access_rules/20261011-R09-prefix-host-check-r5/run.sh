#!/usr/bin/env bash
set -euo pipefail
scontrol show job -dd "$SLURM_JOB_ID" > allocation.txt
nvcc --version > compiler.txt
python3 -c 'import os,json;print(json.dumps({k:os.environ.get(k) for k in ("SLURM_JOB_ID","CUDA_VISIBLE_DEVICES","SLURM_JOB_GPUS","SLURM_STEP_GPUS")}))' > allocation-env.json
mkdir -p build
flags=(-std=c++17 -O3 -DNDEBUG -gencode=arch=compute_90a,code=sm_90a -lineinfo --ptxas-options=-v -DV06_CFG=1 -Isource/probes -Isource/cutlass/include -Isource/cutlass/tools/util/include)
nvcc "${flags[@]}" -Xcompiler=-fPIC,-fvisibility=hidden -shared source/probes/r09_plain_prefix.cu -o build/plain_prefix.so >build/library.log 2>&1
nvcc "${flags[@]}" -DR09_PLAIN_PREFIX_HOST source/probes/r18.cu -ldl -o build/plain_host >build/plain.log 2>&1
nvcc -Isource/overlay "${flags[@]}" -DR09_PLAIN_PREFIX_HOST -DV06_TRACE -DR18_DUAL_CLOCK source/probes/r18.cu -ldl -o build/dual_host >build/dual.log 2>&1
for name in plain_prefix.so plain_host dual_host; do cuobjdump --dump-sass build/$name >build/$name.sass; done
args=(--m 2816 --n 1536 --k 4096 --lda 4096 --ldb 1536 --ldd 1536 --alloc-lda 4104 --alloc-ldb 1536 --swizzle 1 --input-mode dyadic --seed 17 --warmup-policy common-sw1-16 --plain-prefix-module "$PWD/build/plain_prefix.so")
./build/plain_host "${args[@]}" >plain.jsonl
./build/dual_host "${args[@]}" >dual.jsonl
python3 - <<'CHECK'
import json
for name in ['plain','dual']:
 events={e['event']:e for e in map(json.loads,open(name+'.jsonl'))}
 call=events['call'];check=events['check'];setup=events['setup']
 assert call['plain_prefix_calls']==16 and call['warmup_calls']==16 and call['plain_prefix_same_buffers']
 assert len(call['plain_prefix_inter_call_us'])==16
 assert setup['stages']==6 and setup['grid']==[1,132,1]
 print(json.dumps({'variant':name,'gpu_uuid':setup['gpu_uuid'],'check':check,'prefix_calls':16,'formal_timing':False}))
CHECK
