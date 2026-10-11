#!/bin/bash
set -euo pipefail
scontrol show job -dd "$SLURM_JOB_ID" >allocation.txt
nvcc --version >compiler.txt
nvcc -DR18_MATCH_DUAL_LAYOUT -DR09_PLAIN_PREFIX_HOST -DR09_CLOCK_PLAIN -std=c++17 -O3 -DNDEBUG -gencode=arch=compute_90a,code=sm_90a -lineinfo --ptxas-options=-v -DV06_CFG=1 -DR18_STAGE_COUNT=6 -Isource/probes -Isource/cutlass/include -Isource/cutlass/tools/util/include source/probes/r18.cu -o build/cfg_b_plain -ldl >build/cfg_b_plain.log 2>&1
cuobjdump --dump-sass build/cfg_b_plain >build/cfg_b_plain.sass
cuobjdump --dump-resource-usage build/cfg_b_plain >build/cfg_b_plain.resources
nvcc -DR09_PLAIN_PREFIX_HOST -DR09_CLOCK_MATCHED -std=c++17 -O3 -DNDEBUG -gencode=arch=compute_90a,code=sm_90a -lineinfo --ptxas-options=-v -DV06_CFG=1 -DR18_STAGE_COUNT=6 -DR18_DUAL_CLOCK -DV06_TRACE -Isource/overlay -Isource/probes -Isource/cutlass/include -Isource/cutlass/tools/util/include source/probes/r18.cu -o build/cfg_b_matched_dual -ldl >build/cfg_b_matched_dual.log 2>&1
cuobjdump --dump-sass build/cfg_b_matched_dual >build/cfg_b_matched_dual.sass
cuobjdump --dump-resource-usage build/cfg_b_matched_dual >build/cfg_b_matched_dual.resources
nvcc -DR09_PLAIN_PREFIX_HOST -DR09_CLOCK_GUARDED -std=c++17 -O3 -DNDEBUG -gencode=arch=compute_90a,code=sm_90a -lineinfo --ptxas-options=-v -DV06_CFG=1 -DR18_STAGE_COUNT=6 -DR18_DUAL_CLOCK -DV06_TRACE -Isource/overlay -Isource/probes -Isource/cutlass/include -Isource/cutlass/tools/util/include source/probes/r18.cu -o build/cfg_b_guarded_bracket -ldl >build/cfg_b_guarded_bracket.log 2>&1
cuobjdump --dump-sass build/cfg_b_guarded_bracket >build/cfg_b_guarded_bracket.sass
cuobjdump --dump-resource-usage build/cfg_b_guarded_bracket >build/cfg_b_guarded_bracket.resources
nvcc -std=c++17 -O3 -DNDEBUG -shared -Xcompiler -fPIC,-fvisibility=hidden -gencode=arch=compute_90a,code=sm_90a -lineinfo --ptxas-options=-v -DV06_CFG=1 -DR18_STAGE_COUNT=6 -Isource/probes -Isource/cutlass/include -Isource/cutlass/tools/util/include source/probes/r09_plain_prefix.cu -o build/libplain_prefix.so >build/plain_prefix.log 2>&1
cuobjdump --dump-sass build/libplain_prefix.so >build/plain_prefix.sass
cuobjdump --dump-resource-usage build/libplain_prefix.so >build/plain_prefix.resources
