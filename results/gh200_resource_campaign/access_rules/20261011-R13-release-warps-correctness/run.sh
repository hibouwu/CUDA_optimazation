set -euo pipefail
scontrol show job -dd "$SLURM_JOB_ID" > allocation.txt
nvcc --version > compiler.txt
nvidia-smi --query-gpu=uuid,name --format=csv,noheader > gpu.txt
mkdir -p raw full-d
nvcc -std=c++17 -O3 -DNDEBUG -gencode=arch=compute_90a,code=sm_90a -lineinfo --ptxas-options=-v -DV06_CFG=1 -DR13_RELEASE_WARPS -DR18_STAGE_COUNT=4 -Isource/probes -Isource/cutlass/include -Isource/cutlass/tools/util/include source/probes/r18.cu -o build/cfg_b_s4_plain >build/cfg_b_s4_plain.log 2>&1
cuobjdump --dump-sass build/cfg_b_s4_plain >build/cfg_b_s4_plain.sass
cuobjdump --dump-elf-symbols build/cfg_b_s4_plain >build/cfg_b_s4_plain.symbols
nvcc -std=c++17 -O3 -DNDEBUG -gencode=arch=compute_90a,code=sm_90a -lineinfo --ptxas-options=-v -DV06_CFG=1 -DR13_RELEASE_WARPS -DR18_STAGE_COUNT=4 -DR13_STAGE_SPARSE -DV06_TRACE -Isource/overlay -Isource/probes -Isource/cutlass/include -Isource/cutlass/tools/util/include source/probes/r18.cu -o build/cfg_b_s4_stamped >build/cfg_b_s4_stamped.log 2>&1
cuobjdump --dump-sass build/cfg_b_s4_stamped >build/cfg_b_s4_stamped.sass
cuobjdump --dump-elf-symbols build/cfg_b_s4_stamped >build/cfg_b_s4_stamped.symbols
nvcc -std=c++17 -O3 -DNDEBUG -gencode=arch=compute_90a,code=sm_90a -lineinfo --ptxas-options=-v -DV06_CFG=1 -DR13_RELEASE_WARPS -DR18_STAGE_COUNT=6 -Isource/probes -Isource/cutlass/include -Isource/cutlass/tools/util/include source/probes/r18.cu -o build/cfg_b_s6_plain >build/cfg_b_s6_plain.log 2>&1
cuobjdump --dump-sass build/cfg_b_s6_plain >build/cfg_b_s6_plain.sass
cuobjdump --dump-elf-symbols build/cfg_b_s6_plain >build/cfg_b_s6_plain.symbols
nvcc -std=c++17 -O3 -DNDEBUG -gencode=arch=compute_90a,code=sm_90a -lineinfo --ptxas-options=-v -DV06_CFG=1 -DR13_RELEASE_WARPS -DR18_STAGE_COUNT=6 -DR13_STAGE_SPARSE -DV06_TRACE -Isource/overlay -Isource/probes -Isource/cutlass/include -Isource/cutlass/tools/util/include source/probes/r18.cu -o build/cfg_b_s6_stamped >build/cfg_b_s6_stamped.log 2>&1
cuobjdump --dump-sass build/cfg_b_s6_stamped >build/cfg_b_s6_stamped.sass
cuobjdump --dump-elf-symbols build/cfg_b_s6_stamped >build/cfg_b_s6_stamped.symbols
build/cfg_b_s4_plain --mode check-release-warps --check-output full-d/cfg_b_s4_plain.bin --m 2816 --n 1536 --k 4096 --lda 4096 --ldb 1536 --ldd 1536 --storage-m 2816 --storage-n 1536 --zero-m -1 --zero-n -1 --swizzle 1 --input-mode dyadic --seed 17 --alloc-lda 4104 --alloc-ldb 1536 --raster-order heuristic --release-warp-observer 0 >raw/cfg_b_s4_plain_check.jsonl
build/cfg_b_s4_stamped --mode check-release-warps --check-output full-d/cfg_b_s4_stamped.bin --m 2816 --n 1536 --k 4096 --lda 4096 --ldb 1536 --ldd 1536 --storage-m 2816 --storage-n 1536 --zero-m -1 --zero-n -1 --swizzle 1 --input-mode dyadic --seed 17 --alloc-lda 4104 --alloc-ldb 1536 --raster-order heuristic --release-warp-observer 0 >raw/cfg_b_s4_stamped_check.jsonl
cmp full-d/cfg_b_s4_plain.bin full-d/cfg_b_s4_stamped.bin
sha256sum full-d/cfg_b_s4_*.bin >> full-d-sha256.txt
build/cfg_b_s4_stamped --mode warm --warmup-policy common-sw1-16 --m 2816 --n 1536 --k 4096 --lda 4096 --ldb 1536 --ldd 1536 --storage-m 2816 --storage-n 1536 --zero-m -1 --zero-n -1 --swizzle 1 --input-mode dyadic --seed 17 --alloc-lda 4104 --alloc-ldb 1536 --raster-order heuristic --release-warp-observer 0 | gzip >raw/stage4_flag0_smoke.jsonl.gz
build/cfg_b_s4_stamped --mode warm --warmup-policy common-sw1-16 --m 2816 --n 1536 --k 4096 --lda 4096 --ldb 1536 --ldd 1536 --storage-m 2816 --storage-n 1536 --zero-m -1 --zero-n -1 --swizzle 1 --input-mode dyadic --seed 17 --alloc-lda 4104 --alloc-ldb 1536 --raster-order heuristic --release-warp-observer 1 | gzip >raw/stage4_flag1_smoke.jsonl.gz
build/cfg_b_s6_plain --mode check-release-warps --check-output full-d/cfg_b_s6_plain.bin --m 2816 --n 1536 --k 4096 --lda 4096 --ldb 1536 --ldd 1536 --storage-m 2816 --storage-n 1536 --zero-m -1 --zero-n -1 --swizzle 1 --input-mode dyadic --seed 17 --alloc-lda 4104 --alloc-ldb 1536 --raster-order heuristic --release-warp-observer 0 >raw/cfg_b_s6_plain_check.jsonl
build/cfg_b_s6_stamped --mode check-release-warps --check-output full-d/cfg_b_s6_stamped.bin --m 2816 --n 1536 --k 4096 --lda 4096 --ldb 1536 --ldd 1536 --storage-m 2816 --storage-n 1536 --zero-m -1 --zero-n -1 --swizzle 1 --input-mode dyadic --seed 17 --alloc-lda 4104 --alloc-ldb 1536 --raster-order heuristic --release-warp-observer 0 >raw/cfg_b_s6_stamped_check.jsonl
cmp full-d/cfg_b_s6_plain.bin full-d/cfg_b_s6_stamped.bin
sha256sum full-d/cfg_b_s6_*.bin >> full-d-sha256.txt
build/cfg_b_s6_stamped --mode warm --warmup-policy common-sw1-16 --m 2816 --n 1536 --k 4096 --lda 4096 --ldb 1536 --ldd 1536 --storage-m 2816 --storage-n 1536 --zero-m -1 --zero-n -1 --swizzle 1 --input-mode dyadic --seed 17 --alloc-lda 4104 --alloc-ldb 1536 --raster-order heuristic --release-warp-observer 0 | gzip >raw/stage6_flag0_smoke.jsonl.gz
build/cfg_b_s6_stamped --mode warm --warmup-policy common-sw1-16 --m 2816 --n 1536 --k 4096 --lda 4096 --ldb 1536 --ldd 1536 --storage-m 2816 --storage-n 1536 --zero-m -1 --zero-n -1 --swizzle 1 --input-mode dyadic --seed 17 --alloc-lda 4104 --alloc-ldb 1536 --raster-order heuristic --release-warp-observer 1 | gzip >raw/stage6_flag1_smoke.jsonl.gz
printf "all_full_D_comparisons_passed\n" > correctness-result.txt
