# Benchmark Guide

This project treats benchmarking as part of the implementation. A kernel version is useful only when it is correct, timed consistently, and tied to a clear hardware bottleneck.

## Rules

1. Warm up before measurement.
2. Time only kernel or library execution with CUDA events.
3. Do not include host/device copies in kernel timing.
4. Verify every custom kernel against a trusted reference.
5. Save machine-readable CSV output for later plotting.
6. Compare each optimization against the version immediately before it and against the library baseline.

## REDUCE Metrics

Primary output:

```txt
Version,N,TimeMs,BandwidthGBps,Result,Matched
```

Bandwidth formula:

```txt
BandwidthGBps = N * sizeof(float) / TimeMs / 1e6
```

Useful Nsight Compute metrics:

- Warp divergence: active threads per warp, predicated-off threads.
- Barrier cost: `smsp__warp_issue_stalled_barrier`.
- Memory stalls: `smsp__warp_issue_stalled_long_scoreboard`.
- Global-memory throughput.
- Shared-memory load/store instruction count.

Suggested profiling order:

```bash
ncu --set full ./reduce_bench 16777216
```

Read the versions as a story:

- `v0` shows the cost of interleaved addressing.
- `v1` isolates the benefit of contiguous active threads.
- `v4` shows the reduction in shared-memory traffic from warp shuffle.
- `v5` shows whether vectorized loads improve bandwidth.
- CUB shows the production-quality target.

## GEMM Metrics

Primary output:

```txt
Version,N,TimeMs,GFLOPS,RatioToCuBLAS,Matched
```

GFLOPS formula:

```txt
GFLOPS = 2 * M * N * K / TimeMs / 1e6
```

Useful Nsight Compute metrics:

- Compute throughput and memory throughput.
- `smsp__warp_issue_stalled_long_scoreboard`.
- `smsp__warp_issue_stalled_mio_throttle`.
- Eligible warps per scheduler.
- Register count and occupancy.
- Shared-memory bank conflicts.
- Global load/store instruction count.

Suggested profiling order:

```bash
ncu --set full ./gemm_bench 1024
```

Read the versions as a story:

- `v1` exposes global-memory reuse problems.
- `v2` introduces shared-memory data reuse.
- `v3` increases arithmetic intensity through per-thread output tiles.
- `v4` reduces memory instruction count with `float4` and improves A-tile access.
- `v5` tries to overlap tile loading and compute.
- `v6` improves warp-level locality with hierarchical tiling.
- cuBLAS remains the reference for both correctness and performance.

## Result Interpretation

For small matrices, cuBLAS may win by a large margin because it dispatches specialized kernels. For larger square matrices, handwritten kernels should show a clear progression from v1 to v6. If an optimized version is slower, inspect:

- Whether the tile size increases register pressure too much.
- Whether shared memory reduces occupancy.
- Whether vectorized loads are aligned.
- Whether the benchmark size actually exercises the high-performance path.
