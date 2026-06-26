# TC4/TC5 SM120 FP8 GEMM NCU Summary

Date: 2026-06-26

GPU: NVIDIA GeForce RTX 5070 Laptop GPU, `sm_120`

Target path: SM120a FP8 e4m3 GEMM using `mma.sync.aligned.kind::f8f6f4`, not `tcgen05/TMEM`.

## Version Scope

| Version | Purpose | Main idea |
| --- | --- | --- |
| `tc4a` | 3-stage FP8 TMA mainloop experiment | TMA A/B into SMEM, CTA-wide B operand prepack, then SM120 FP8 MMA. |
| `tc4b` | TMA swizzle experiment | Same as `tc4a`, but B tensor map uses 64B TMA swizzle for large N and no-swizzle fallback for small N. |
| `tc5a` | Static CLC fallback | Persistent worker grid, static grid-stride tile assignment, reuses `tc4b` mainloop. |
| `tc5b` | Dynamic CLC fallback | Persistent worker grid, global atomic work counter for dynamic tile assignment, reuses `tc4b` mainloop. |

`tc5a/tc5b` are CLC / work-tile scheduling experiments. They are not producer/consumer warp-specialized mainloops yet.

## Sweep Result

Source: `results/gemm/tensor_core/gemm_tensor_core_sweep.csv`, 10 trials per size.

GFLOPS mean:

| N | cuBLAS TC | tc2 | tc3 | tc4a | tc4b | tc5a | tc5b |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 635.3 | 677.2 | 496.8 | 511.1 | 510.2 | 510.7 | 342.9 |
| 256 | 5033.2 | 3261.1 | 2336.6 | 3267.1 | 3265.2 | 3263.1 | 2324.7 |
| 512 | 26041.2 | 14514.8 | 11703.7 | 16352.0 | 16348.3 | 16352.4 | 13068.1 |
| 1024 | 39916.3 | 21679.6 | 16615.1 | 31241.4 | 31405.7 | 30483.1 | 28649.2 |
| 2048 | 42812.9 | 23162.6 | 19941.3 | 41131.3 | 39222.7 | 37485.3 | 39544.3 |
| 4096 | 38863.9 | 21857.2 | 19852.3 | 33365.6 | 37921.0 | 34953.8 | 35382.6 |

Observations:

- `tc4a/tc4b` are a large step over `tc2/tc3` from `N >= 1024`, mainly due to the FP8 SM120 MMA path and B operand prepack.
- `tc4a` is best at `N=2048` in the sweep: about `41.1 TFLOPS`, close to cuBLAS Tensor Core.
- `tc4b` is best handwritten at `N=4096`: about `37.9 TFLOPS`; this is where 64B TMA swizzle starts to help.
- `tc5b` dynamic work queue helps relative to `tc5a` at `N=2048`, but still does not remove the main CTA barrier bottleneck.
- `tc5b` is not universally better. At `N=4096`, software atomic scheduling overhead and tile-order effects make it slower than `tc4b`.

## NCU 2048 Summary

Reports:

- `reports/tc4a_2048_fast.ncu-rep`
- `reports/tc4b_2048_fast.ncu-rep`
- `reports/tc5a_2048_fast.ncu-rep`
- `reports/tc5b_2048_fast.ncu-rep`

All four reports use one profiled kernel at `N=2048`.

| Metric | tc4a | tc4b | tc5a | tc5b |
| --- | ---: | ---: | ---: | ---: |
| Duration | 951.90 us | 941.70 us | 964.90 us | 943.52 us |
| Compute throughput | 42.88% | 43.01% | 41.96% | 43.15% |
| Memory throughput | 57.83% | 58.01% | 56.41% | 58.02% |
| DRAM throughput | 10.45% | 11.10% | 10.39% | 10.51% |
| L1/TEX throughput | 61.02% | 61.08% | 61.00% | 61.72% |
| L2 throughput | 47.81% | 48.30% | 47.21% | 48.30% |
| L1/TEX hit rate | 39.41% | 39.44% | 40.37% | 39.48% |
| L2 hit rate | 93.09% | 92.87% | 93.03% | 92.99% |
| Active warps / scheduler | 5.67 | 5.67 | 5.67 | 5.77 |
| Eligible warps / scheduler | 0.35 | 0.35 | 0.34 | 0.34 |
| Issued warp / scheduler | 0.22 | 0.22 | 0.22 | 0.22 |
| No eligible | 77.70% | 77.69% | 78.11% | 78.26% |
| Warp cycles / issued inst | 25.44 | 25.44 | 25.91 | 26.52 |
| Barrier stall | 9.5 cycles | 9.5 cycles | 9.9 cycles | 9.8 cycles |
| Barrier stall share | 37.43% | 37.52% | 38.23% | 37.14% |
| Theoretical occupancy | 50.0% | 50.0% | 50.0% | 50.0% |
| Achieved occupancy | 47.26% | 47.27% | 47.07% | 48.16% |

## Bottleneck Interpretation

The four kernels have almost identical NCU signatures:

- Compute throughput is only about `42-43%`.
- Memory throughput is about `56-58%`, but DRAM throughput is only about `10-11%`.
- This is not a raw DRAM bandwidth limit. Most traffic is served by L1/L2/SMEM-side pipelines.
- Scheduler issue rate is the real problem: only about `0.22` issued warp per scheduler, and `~78%` cycles have no eligible warp.
- The largest visible stall is CTA barrier stall: about `9.5-9.9` cycles, `37-38%` of warp cycles.

Current mainloop still has CTA-wide synchronization:

1. Wait TMA stage.
2. CTA-wide B operand prepack into SMEM.
3. CTA-wide barrier after pack.
4. MMA consumes the packed B.
5. CTA-wide barrier before refill / stage reuse.

This structure improves operand layout, but it serializes the whole CTA around pack/refill barriers. That is why swizzle and CLC scheduling help only partially: they do not remove the CTA-wide mainloop barrier.

## TC4A vs TC4B

`tc4b` adds 64B TMA swizzle for B on large N. In the NCU 2048 report, `tc4b` is slightly faster than `tc4a`:

- `tc4a`: `951.90 us`
- `tc4b`: `941.70 us`

However, the 10-trial sweep shows `tc4a` winning at `N=2048`, while `tc4b` wins at `N=4096`.

Conclusion:

- At `N=2048`, `tc4a` and `tc4b` are close enough that run-to-run variation and profiler overhead can reorder them.
- At `N=4096`, `tc4b` is the better mainline because the swizzled B map pays off more consistently.
- Keep both for now: `tc4a` as no-swizzle prepack baseline, `tc4b` as large-N swizzle path.

## TC5A vs TC5B

`tc5a` and `tc5b` change work-tile scheduling, not the per-tile mainloop.

`tc5b` improves over `tc5a` in the 2048 sweep:

- `tc5a`: `37.5 TFLOPS`
- `tc5b`: `39.5 TFLOPS`

But both have the same low eligible warp issue:

- `tc5a`: `0.34` eligible warps/scheduler
- `tc5b`: `0.34` eligible warps/scheduler

Conclusion:

- Dynamic tile scheduling can reduce some persistent-worker load imbalance.
- It does not solve the CTA barrier stall inside each tile.
- Software atomic work queue is a CLC behavior probe, not the final high-performance CLC implementation.

## Current Ranking

For `N=2048`:

1. `cuBLAS TC`: `42.8 TFLOPS`
2. `tc4a`: `41.1 TFLOPS`
3. `tc5b`: `39.5 TFLOPS`
4. `tc4b`: `39.2 TFLOPS`
5. `tc5a`: `37.5 TFLOPS`
6. `tc2`: `23.2 TFLOPS`
7. `tc3`: `19.9 TFLOPS`

For `N=4096`:

1. `cuBLAS TC`: `38.9 TFLOPS`
2. `tc4b`: `37.9 TFLOPS`
3. `tc5b`: `35.4 TFLOPS`
4. `tc5a`: `35.0 TFLOPS`
5. `tc4a`: `33.4 TFLOPS`
6. `tc2`: `21.9 TFLOPS`
7. `tc3`: `19.9 TFLOPS`

## Next Steps

Priority order:

1. Keep `tc4b` as the large-N performance baseline.
2. Keep `tc4a` as the non-swizzle prepack baseline for isolating TMA swizzle effects.
3. Keep `tc5b` as a CLC scheduling probe, but do not treat software atomic queue as the final CLC path.
4. The next real performance step is not another scheduler variant. It is reducing CTA-wide barrier pressure inside the tile mainloop.
5. Move toward producer/consumer warp specialization:
   - producer warp/group handles TMA and B prepack,
   - consumer warps issue MMA,
   - use smaller synchronization domains instead of full CTA `__syncthreads()`,
   - overlap stage refill with MMA consumption.
6. If staying in the current CTA-cooperative design, optimize the two barrier points first:
   - barrier after `tc4_pack_b_stage`,
   - barrier before stage refill / SMEM stage reuse.

## Useful Commands

Open the reports:

```bash
ncu-ui /home/jianyeshi/Note/CUDA/CUDA_optimazation/reports/tc4a_2048_fast.ncu-rep
ncu-ui /home/jianyeshi/Note/CUDA/CUDA_optimazation/reports/tc4b_2048_fast.ncu-rep
ncu-ui /home/jianyeshi/Note/CUDA/CUDA_optimazation/reports/tc5a_2048_fast.ncu-rep
ncu-ui /home/jianyeshi/Note/CUDA/CUDA_optimazation/reports/tc5b_2048_fast.ncu-rep
```

Regenerate a focused report:

```bash
cd /workspace
mkdir -p /workspace/reports

ncu -f --target-processes all \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section SchedulerStats \
  --section WarpStateStats \
  --section Occupancy \
  --kernel-name-base demangled \
  --kernel-name 'regex:^hgemm_tc4b_sm120a_fp8_tma_3stage_swizzle_mma_128x64x32.*' \
  -c 1 \
  --export /workspace/reports/tc4b_2048_fast \
  /workspace/build_docker/gemm_bench 2048 tc4b
```

