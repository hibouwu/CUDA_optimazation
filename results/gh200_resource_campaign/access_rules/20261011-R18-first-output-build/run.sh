#!/bin/bash
set -euo pipefail
scontrol show job -dd "$SLURM_JOB_ID" >allocation.txt
nvcc --version >compiler.txt
nvcc -DR18_FIRST_OUTPUT -std=c++17 -O3 -DNDEBUG -gencode=arch=compute_90a,code=sm_90a -lineinfo --ptxas-options=-v -DV06_CFG=0 -DR18_STAGE_COUNT=6 -DR18_DUAL_CLOCK -DV06_TRACE -Isource/overlay -Isource/probes -Isource/cutlass/include -Isource/cutlass/tools/util/include source/probes/r18.cu -o build/cfg_a_s6_dual >build/cfg_a_s6_dual.log 2>&1
cuobjdump --dump-sass build/cfg_a_s6_dual >build/cfg_a_s6_dual.sass
cuobjdump --dump-elf-symbols build/cfg_a_s6_dual >build/cfg_a_s6_dual.symbols
