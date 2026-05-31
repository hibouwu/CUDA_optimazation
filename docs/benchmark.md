# Benchmark Guide

This project treats benchmarking as part of the implementation. A kernel version is useful only when it is correct, timed consistently, and tied to a clear hardware bottleneck.

## Rules

1. Warm up before measurement.
2. Time only kernel or library execution with CUDA events.
3. Do not include host/device copies in kernel timing.
4. Verify every custom kernel against a trusted reference.
5. Save machine-readable CSV output for later plotting.
6. Compare each optimization against the version immediately before it and against the library baseline.

GEMM policy: each backend is measured independently. The harness warms up that
backend first, then records one CUDA event interval containing 100 consecutive
launches of the same backend, averages the elapsed time, and only then moves to
the next backend. This avoids interleaving different GEMM kernels inside one
timed region.

For REDUCE, the CPU reference uses double accumulation. A plain sequential
`float` sum can undercount large inputs because the fractional increments become
smaller than the FP32 spacing near the accumulated value.

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

- `v1` exposes the uncoalesced naive access pattern.
- `v2` isolates the benefit of coalesced global-memory access.
- `v3` introduces shared-memory data reuse.
- `v4` shows 1D block indexing and shared-memory padding.
- `v5` improves warp-level locality with hierarchical tiling.
- `v6` increases arithmetic intensity through per-thread output tiles.
- `v9` reduces memory instruction count with `float4` and improves A-tile access.
- `v10` tries to overlap tile loading and compute.
- cuBLAS remains the reference for both correctness and performance.

## Result Interpretation

For small matrices, cuBLAS may win by a large margin because it dispatches specialized kernels. For larger square matrices, handwritten kernels should show a clear progression across the documented versions. If an optimized version is slower, inspect:

- Whether the tile size increases register pressure too much.
- Whether shared memory reduces occupancy.
- Whether vectorized loads are aligned.
- Whether the benchmark size actually exercises the high-performance path.

## Sweep Script

Use the repository scripts to run repeatable size sweeps and generate figures:

```bash
./scripts/run_reduce_experiments.sh
./scripts/run_gemm_experiments.sh
```

The REDUCE script writes `results/reduce/reduce_sweep.csv`, keeps per-size raw
outputs under `results/reduce/raw/`, and generates
`results/reduce/figures/reduce_bandwidth.svg`.

The GEMM script writes `results/gemm/sgemm_sweep.csv`, keeps per-size raw
outputs under `results/gemm/raw/`, and generates `results/gemm/figures/`
charts for key-kernel GFLOPS, all-kernel log-y GFLOPS, and cuBLAS ratio.
The key-kernel view includes cuBLAS, v2, v3, v5, v6, v9, and v10; v1 and v4
remain in the CSV and all-kernel log figure as teaching baselines.

You can override the default workload without editing the script:

```bash
REDUCE_SIZES="1048576 4194304" ./scripts/run_reduce_experiments.sh
GEMM_SIZES="512 1024" ./scripts/run_gemm_experiments.sh
```

The scripts also support presets:

```bash
PRESET=quick ./scripts/run_reduce_experiments.sh
PRESET=default ./scripts/run_reduce_experiments.sh
PRESET=full TRIALS=5 ./scripts/run_reduce_experiments.sh
```

Default REDUCE sizes are powers of two from 256K to 64M elements. Default GEMM
sizes are 128-aligned from 256 to 2048 so the high-performance GEMM paths are
included at every point.

Each size uses `TRIALS=3` independent runs by default. The aggregate CSV keeps a
`Trial` column, and the SVG plotter draws mean values with standard-deviation
error bars. The x-axis uses log2(size) spacing, keeps every measured point, but
only labels powers of two and endpoints to avoid crowded matrix-size ticks.
Increase trials for final measurements:

```bash
TRIALS=5 GEMM_SIZES="1024 2048" ./scripts/run_gemm_experiments.sh
```

The plotting step is intentionally dependency-light: `scripts/plot_benchmarks.py`
uses only the Python standard library.

## GEMM Blocksize Tuning

Use the tuning script to compare candidate tile shapes:

```bash
BUILD_DIR=/workspace/build_cuda13 TUNE_SIZES="256 512 1024 2048" ./scripts/run_gemm_tuning.sh
```

The tuning CSV is written to `results/gemm_tuning/blocksize_tuning.csv`.
Figures are written to `results/gemm_tuning/figures/`:

- `tuning_thread_tile.svg`: only thread-tile configs.
- `tuning_vectorized.svg`: only vectorized-load configs.
- `tuning_double_buffer.svg`: only double-buffer configs.
- `tuning_best_by_kernel.svg`: best configuration per kernel family and size.

Important fields:

- `BM`, `BN`: output tile size per block.
- `BK`: K-axis tile depth.
- `TM`, `TN`: output tile size per thread.
- `Threads`: `(BM / TM) * (BN / TN)`.
- `SharedMemoryBytes`: static shared-memory footprint per block.
- `GFLOPS`, `RatioToCuBLAS`, `Matched`: performance and correctness.

The best block size is not simply the largest block. A good configuration needs
enough data reuse while keeping register pressure, shared-memory use, and active
warps in a reasonable range. Validate candidates across several matrix sizes
before moving one into the main benchmark.
