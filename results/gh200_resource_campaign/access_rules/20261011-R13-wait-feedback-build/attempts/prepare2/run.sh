#!/bin/bash
set -euo pipefail
scontrol show job -dd "$SLURM_JOB_ID" >allocation.txt
nvcc --version >compiler.txt
nvcc -std=c++17 -O3 -DNDEBUG -gencode=arch=compute_90a,code=sm_90a -lineinfo --ptxas-options=-v -DV06_CFG=1 -DR13_WAIT_SUM -DR13_STAGE_PROGRESS -DV06_TRACE -Isource/overlay -Isource/probes -Isource/cutlass/include -Isource/cutlass/tools/util/include source/probes/r18.cu -o build/cfg_b_stamped >build/cfg_b_stamped.log 2>&1
cuobjdump --dump-sass build/cfg_b_stamped >build/cfg_b_stamped.sass
cuobjdump --dump-resource-usage build/cfg_b_stamped >build/cfg_b_stamped.resources
nvcc -std=c++17 -O3 -DNDEBUG -gencode=arch=compute_90a,code=sm_90a -lineinfo --ptxas-options=-v -DV06_CFG=1 -DR18_STAGE_COUNT=4 -DR13_WAIT_SUM -DR13_STAGE_PROGRESS -DV06_TRACE -Isource/overlay -Isource/probes -Isource/cutlass/include -Isource/cutlass/tools/util/include source/probes/r18.cu -o build/cfg_b_s4_stamped >build/cfg_b_s4_stamped.log 2>&1
cuobjdump --dump-sass build/cfg_b_s4_stamped >build/cfg_b_s4_stamped.sass
cuobjdump --dump-resource-usage build/cfg_b_s4_stamped >build/cfg_b_s4_stamped.resources
