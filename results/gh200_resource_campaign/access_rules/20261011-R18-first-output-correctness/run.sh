#!/bin/bash
set -euo pipefail
mkdir -p raw full-d
scontrol show job -dd "$SLURM_JOB_ID" >allocation.txt
env | sort >environment.txt
build/cfg_a_s6_dual --m 960 --n 2816 --k 4096 --lda 4096 --ldb 2816 --ldd 2816 --storage-m 1024 --storage-n 2816 --zero-m 960 --zero-n -1 --swizzle 1 --input-mode dyadic --seed 17 --input-map-m 960 --warmup-policy common-sw1-16 --mode check-first-output --check-output full-d/map960.bin >raw/map960_full_check.jsonl 2>raw/map960_full_check.stderr
build/cfg_a_s6_dual --m 960 --n 2816 --k 4096 --lda 4096 --ldb 2816 --ldd 2816 --storage-m 1024 --storage-n 2816 --zero-m 960 --zero-n -1 --swizzle 1 --input-mode dyadic --seed 17 --input-map-m 960 --warmup-policy common-sw1-16 --mode warm --first-output-observer 0 2>raw/map960_flag0.stderr | gzip >raw/map960_flag0.jsonl.gz
build/cfg_a_s6_dual --m 960 --n 2816 --k 4096 --lda 4096 --ldb 2816 --ldd 2816 --storage-m 1024 --storage-n 2816 --zero-m 960 --zero-n -1 --swizzle 1 --input-mode dyadic --seed 17 --input-map-m 960 --warmup-policy common-sw1-16 --mode warm --first-output-observer 1 2>raw/map960_flag1.stderr | gzip >raw/map960_flag1.jsonl.gz
build/cfg_a_s6_dual --m 960 --n 2816 --k 4096 --lda 4096 --ldb 2816 --ldd 2816 --storage-m 1024 --storage-n 2816 --zero-m 960 --zero-n -1 --swizzle 1 --input-mode dyadic --seed 17 --input-map-m 1024 --warmup-policy common-sw1-16 --mode check-first-output --check-output full-d/map1024.bin >raw/map1024_full_check.jsonl 2>raw/map1024_full_check.stderr
build/cfg_a_s6_dual --m 960 --n 2816 --k 4096 --lda 4096 --ldb 2816 --ldd 2816 --storage-m 1024 --storage-n 2816 --zero-m 960 --zero-n -1 --swizzle 1 --input-mode dyadic --seed 17 --input-map-m 1024 --warmup-policy common-sw1-16 --mode warm --first-output-observer 0 2>raw/map1024_flag0.stderr | gzip >raw/map1024_flag0.jsonl.gz
build/cfg_a_s6_dual --m 960 --n 2816 --k 4096 --lda 4096 --ldb 2816 --ldd 2816 --storage-m 1024 --storage-n 2816 --zero-m 960 --zero-n -1 --swizzle 1 --input-mode dyadic --seed 17 --input-map-m 1024 --warmup-policy common-sw1-16 --mode warm --first-output-observer 1 2>raw/map1024_flag1.stderr | gzip >raw/map1024_flag1.jsonl.gz
cmp full-d/map960.bin full-d/map1024.bin
