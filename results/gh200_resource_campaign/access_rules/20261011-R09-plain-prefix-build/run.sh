#!/usr/bin/env bash
set -euo pipefail
scontrol show job -dd "$SLURM_JOB_ID" > allocation.txt
nvcc --version > compiler.txt
g++ --version > host-compiler.txt
mkdir -p build
flags=(-std=c++17 -O3 -DNDEBUG -gencode=arch=compute_90a,code=sm_90a -lineinfo --ptxas-options=-v -DV06_CFG=1 -Isource/probes -Isource/cutlass/include -Isource/cutlass/tools/util/include)
nvcc "${flags[@]}" -Xcompiler=-fPIC,-fvisibility=hidden -shared source/probes/r09_plain_prefix.cu -o build/plain_prefix.so >build/library.log 2>&1
nvcc "${flags[@]}" source/probes/r18.cu -o build/cfg_b_plain >build/plain.log 2>&1
cuobjdump --dump-sass build/plain_prefix.so >build/library.sass
cuobjdump --dump-sass build/cfg_b_plain >build/plain.sass
nm -D --defined-only build/plain_prefix.so >build/exports.txt
python3 - <<'CHECK'
import ctypes,json
lib=ctypes.CDLL('./build/plain_prefix.so',mode=ctypes.RTLD_LOCAL)
create=lib.r09_plain_prefix_create
create.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p,ctypes.c_char_p,ctypes.c_size_t]
create.restype=ctypes.c_int
handle=ctypes.c_void_p(123); error=ctypes.create_string_buffer(256)
rc=create(None,ctypes.byref(handle),None,error,256)
assert rc==1 and handle.value is None
assert b'invalid plain-prefix ABI arguments' in error.value
print(json.dumps({'module_load':True,'invalid_create_rejected':True,'gpu_launch':False}))
CHECK
