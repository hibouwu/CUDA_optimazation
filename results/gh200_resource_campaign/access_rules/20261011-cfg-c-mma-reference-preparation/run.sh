#!/bin/bash
set -euo pipefail
scontrol show job -dd "$SLURM_JOB_ID" >allocation.txt
nvcc --version >compiler.txt
nvcc -std=c++17 -O3 -DNDEBUG -gencode=arch=compute_90a,code=sm_90a -lineinfo --ptxas-options=-v -Isource -Isource/cutlass/include -Isource/cutlass/tools/util/include source/cfg_c_mma_reference.cu -o build/cfg_c_mma_reference >build/compile.log 2>&1
cuobjdump --dump-sass build/cfg_c_mma_reference >build/reference.sass
cuobjdump --dump-resource-usage build/cfg_c_mma_reference >build/reference.resources
