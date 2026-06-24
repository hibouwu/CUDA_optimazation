# CUDA Optimization

CUDA Optimization is a compact CUDA kernel optimization lab. It keeps each operator as a versioned benchmark, starting from a readable naive implementation and moving step by step toward production-grade library baselines such as CUB and cuBLAS.

The project is intended for learning and profiling. Each version answers one question: which bottleneck does this optimization remove, and how does that show up in CUDA event timing or Nsight Compute metrics?

## Highlights

- Reduce kernels from interleaved shared-memory reduction to warp shuffle, vectorized loads, and CUB baseline.
- SGEMM kernels following the local Matmul note path: naive, coalesced access, shared-memory tiling, thread tiling, vectorized loads, double buffering, plus 1D/padding and warp-tiling branch experiments.
- Transformer optimization workspace for LayerNorm, Softmax, QKV projection, Attention, FFN, and KV cache experiments.
- Reproducible benchmark harnesses with warmup, repeated timing, correctness checks, and CSV output.
- GEMM timing runs each backend independently: warmup first, then 100 timed launches before moving to the next backend.
- Chinese source comments explaining the purpose of each optimization stage.
- CMake and direct `nvcc` build paths.

## Repository Layout

```txt
CUDA_optimazation/
├── CHANGELOG.md
├── CMakeLists.txt
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── WORKLOG.md
├── docs/
│   ├── benchmark.md
│   └── toolchain.md
├── scripts/
│   ├── plot_benchmarks.py
│   ├── run_gemm_experiments.sh
│   ├── run_gemm_tuning.sh
│   ├── run_reduce_experiments.sh
│   └── run_transformer_experiments.sh
├── GEMM/
│   ├── include/
│   │   ├── gemm_benchmark.cuh
│   │   ├── gemm_common.cuh
│   │   └── sgemm_kernels.cuh
│   └── src/main.cu
├── TRANSFORMER/
│   ├── README.md
│   ├── include/
│   │   ├── transformer_benchmark.cuh
│   │   ├── transformer_common.cuh
│   │   └── transformer_kernels.cuh
│   └── src/main.cu
└── REDUCE/
    ├── include/
    │   ├── reduce_benchmark.cuh
    │   ├── reduce_common.cuh
    │   └── reduce_kernels.cuh
    └── src/main.cu
```

## Kernel Matrix

### REDUCE

`REDUCE/src/main.cu` benchmarks one-dimensional FP32 sum reduction.

| Version | Implementation | Main idea | What to inspect |
| --- | --- | --- | --- |
| v0 | `reduce_v0_interleaved` | Interleaved shared-memory tree reduction | warp divergence, shared-memory bank conflicts |
| v1 | `reduce_v1_sequential` | Sequential addressing with contiguous active threads | warp execution efficiency |
| v2 | `reduce_v2_first_add` | Each thread loads and sums two input elements first | partial write count, memory throughput |
| v3 | `reduce_v3_unroll_last_warp` | Unroll the last warp and remove final barriers | barrier stalls, instruction count |
| v4 | `reduce_v4_shuffle` | Warp shuffle reduction plus one shared result per warp | shared-memory traffic, latency |
| v5 | `reduce_v5_vectorized` | `float4` vectorized global loads | memory instruction count, effective bandwidth |
| baseline | `cub::DeviceReduce::Sum` | CUB production reduction | handwritten kernel / CUB bandwidth ratio |

Output CSV:

```txt
reduce_benchmark.csv
Version,N,TimeMs,BandwidthGBps,Result,Matched
```

### GEMM

`GEMM/src/main.cu` benchmarks row-major FP32 square SGEMM:

```txt
C = alpha * A @ B + beta * C
```

| Version | Implementation | Main idea | What to inspect |
| --- | --- | --- | --- |
| baseline | `cublasSgemm` | Default cuBLAS performance baseline plus FP32 pedantic correctness/performance reference | target throughput, TF32 impact |
| v1 | `sgemm_v1_naive_uncoalesced` | Naive mapping with poor coalescing | global load pattern |
| v2 | `sgemm_v1_naive` | Coalesced naive mapping | memory transaction efficiency |
| v3 | `sgemm_v2_smem` | Shared-memory tile reuse | shared-memory throughput |
| v3a | `sgemm_v3a_smem_1d` | 1D thread block shared-memory branch without padding | indexing overhead |
| v3b | `sgemm_v4_smem_1d_padded` | 1D thread block plus shared-memory padding branch | bank conflicts, indexing overhead |
| v4 | `sgemm_v3_thread_tile` | one thread computes a `TM x TN` output tile | arithmetic intensity, register use |
| v5 | `sgemm_v4_vectorized` | `float4` loads and transposed A tile in shared memory | memory instruction count |
| v6 | `sgemm_v5_double_buffer` | Shared-memory ping-pong buffers plus register prefetch | long scoreboard stalls |
| v7 | `sgemm_v7_warp_tiling_double_buffer` | v6 plus block / warp / thread hierarchical tiling | eligible warps, occupancy |
| v8a | `sgemm_v8a_cp_async_b_tile` | v7 plus `cp.async` for B tile staging only | async copy overhead, B load latency |
| v8b | `sgemm_v8b_cp_async_ab_2stage` | A/B `cp.async` staging with cp.async-friendly shared layout | shared layout tradeoff, 2-stage pipeline |
| v8c | `sgemm_v8c_cp_async_ab_3stage` | v8b with a 3-stage async copy pipeline and smaller `BK=8` | latency hiding vs shared-memory footprint |

Output CSV:

```txt
sgemm_benchmark.csv
Version,N,TimeMs,GFLOPS,RatioToCuBLAS,Matched
```

The local note jumps from Kernel 6 to Kernel 9; this repository mirrors the
documented stages. `v5`, `v6`, `v7`, and `v8*` are intentionally written as
high-performance paths and require `N` to be a multiple of 128. Other sizes
still run `v1` to `v4`, `v3a`, `v3b`, and cuBLAS.

### TRANSFORMER

`TRANSFORMER/src/main.cu` currently benchmarks LayerNorm and is the workspace for
Transformer-specific optimization work. It is separate from GEMM because
Transformer performance depends on whole operator pipelines and tensor layouts,
not only a single matrix multiply.

Current LayerNorm versions:

| Version | Implementation | Main idea | What to inspect |
| --- | --- | --- | --- |
| v1 | `layernorm_v1_naive` | One thread handles one row serially | baseline latency, no parallel reduction |
| v2 | `layernorm_v2_block_reduce` | One block handles one row with shared-memory reduction | row reduction bandwidth |
| v3 | `layernorm_v3_vectorized` | `float4` vectorized row loads/stores | memory instruction count, alignment |

Output CSV:

```txt
transformer_benchmark.csv
Operator,Version,Batch,SeqLen,Hidden,NumHeads,HeadDim,TimeMs,BandwidthGBps,Matched
```

Planned sequence:

| Stage | Operator | Focus |
| --- | --- | --- |
| v1 | LayerNorm / RMSNorm | row reductions, vectorized memory traffic |
| v2 | Softmax | max/sum reductions and numerical stability |
| v3 | QKV projection | cuBLAS / GEMM integration |
| v4 | Attention score | batched or strided GEMM layout |
| v5 | Online softmax attention | FlashAttention-style streaming |
| v6 | FFN / MLP | GEMM + activation fusion |
| v7 | KV cache decode | cache layout and decode latency |

## Quick Start

### Option A: Docker / CUDA Container

This is the recommended path on very new Linux systems where local CUDA headers
may conflict with the host compiler or glibc. The verified image for this
project is:

```txt
nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu24.04
```

First verify that Docker can see the GPU:

```bash
docker run --rm --gpus all nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu24.04 nvidia-smi
```

Then enter the project container:

```bash
docker run --gpus all --rm -it \
  --security-opt label=disable \
  -v /home/jianyeshi/Note/CUDA/CUDA_optimazation:/workspace \
  -w /workspace \
  nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu24.04 bash
```

If the container reports `Permission denied` when reading the mounted build
directory on an SELinux-enabled host, add `--security-opt label=disable` before
`-v`.

The NGC CUDA devel image contains `nvcc`, but may not contain CMake. Install the
build tools once inside the container:

```bash
apt-get update
apt-get install -y cmake build-essential python3
```

Build and run:

```bash
cmake -S . -B build_cuda13 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build_cuda13 -j

./build_cuda13/reduce_bench 16777216
./build_cuda13/gemm_bench 1024
./build_cuda13/transformer_bench
```

For a quick REDUCE-only check without CMake:

```bash
nvcc -O3 -std=c++17 -arch=sm_120 -IREDUCE/include REDUCE/src/main.cu -o /tmp/reduce_bench
/tmp/reduce_bench 16777216
```

### Option B: Local CMake

```bash
cmake -S . -B build_cuda13 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build_cuda13 -j

./build_cuda13/reduce_bench 16777216
./build_cuda13/gemm_bench 1024
```

### Option C: Local Direct nvcc

```bash
nvcc -O3 -std=c++17 -arch=sm_120 -IREDUCE/include REDUCE/src/main.cu -o reduce_bench
nvcc -O3 -std=c++17 -arch=sm_120 -IGEMM/include GEMM/src/main.cu -lcublas -o gemm_bench

./reduce_bench 16777216
./gemm_bench 1024
```

Change `sm_120` to your GPU architecture, for example `sm_75`, `sm_80`,
`sm_86`, `sm_89`, or `sm_90`. The experiment scripts auto-detect this from
`nvidia-smi` and fall back to `86` only when no GPU is visible.

## Profiling

Use Nsight Compute after correctness passes:

```bash
ncu --set full ./reduce_bench 16777216
ncu --set full ./gemm_bench 1024
```

Recommended first comparisons:

- Reduce: v0 vs v1 vs v4 vs v5 vs CUB.
- GEMM mainline: v1 vs v2 vs v3 vs v4 vs v5 vs v6 vs v7 vs v8a/v8b/v8c vs cuBLAS FP32 Pedantic.
- GEMM branches: v3a and v3b.

See [docs/benchmark.md](docs/benchmark.md) for measurement rules and metric suggestions.

## Extended Experiments

Run the sweep script after the basic single-size checks pass:

```bash
./scripts/run_reduce_experiments.sh
./scripts/run_gemm_experiments.sh
./scripts/run_transformer_experiments.sh
```

Default sweep sizes:

```txt
REDUCE default: 256K, 512K, 1M, 2M, 4M, 8M, 16M, 32M, 64M elements
GEMM default:   128, 256, 512, 1024, 2048
TRANSFORMER default:
  LayerNorm B:S:H:heads:head_dim =
  1:128:768:12:64, 1:512:768:12:64, 1:1024:768:12:64,
  1:512:4096:32:128, 1:1024:4096:32:128, 1:2048:4096:32:128
```

Outputs:

```txt
results/
├── reduce/
│   ├── reduce_sweep.csv
│   ├── raw/
│   └── figures/reduce_bandwidth.svg
├── gemm/
│   ├── sgemm_sweep.csv
│   ├── raw/
│   └── figures/
│       ├── sgemm_gflops.svg
│       ├── gemm_fp32_best_backend_gflops.svg
│       ├── sgemm_gflops_log.svg
│       └── sgemm_ratio_to_cublas.svg
└── transformer/
    ├── transformer_sweep.csv
    ├── raw/
    └── figures/
        ├── layernorm_bandwidth_H768.svg
        ├── layernorm_time_H768.svg
        ├── layernorm_bandwidth_H4096.svg
        └── layernorm_time_H4096.svg
```

Each experiment script automatically calls `scripts/plot_benchmarks.py` at the end. The
plotter writes SVG files with Python's standard library, so it does not require
`matplotlib`.

Override sizes when needed:

```bash
REDUCE_SIZES="1048576 16777216" ./scripts/run_reduce_experiments.sh
GEMM_SIZES="512 1024" ./scripts/run_gemm_experiments.sh
TRANSFORMER_SHAPES="1:1024:4096:32:128" ./scripts/run_transformer_experiments.sh
```

Use presets for different experiment budgets:

```bash
BUILD_DIR=/workspace/build_cuda13 PRESET=quick ./scripts/run_gemm_experiments.sh
BUILD_DIR=/workspace/build_cuda13 PRESET=default ./scripts/run_gemm_experiments.sh
BUILD_DIR=/workspace/build_cuda13 PRESET=full TRIALS=5 ./scripts/run_gemm_experiments.sh
BUILD_DIR=/workspace/build_cuda13 PRESET=quick ./scripts/run_transformer_experiments.sh
```

Each size runs `TRIALS=3` independent process-level trials by default. The
figures plot the mean value with standard-deviation error bars. The GEMM x-axis
uses linear matrix-size spacing. GEMM also writes a log-y GFLOPS chart so slow
and fast kernels can be inspected in the same figure:

```bash
BUILD_DIR=/workspace/build_cuda13 TRIALS=5 REDUCE_SIZES="16777216" ./scripts/run_reduce_experiments.sh
BUILD_DIR=/workspace/build_cuda13 TRIALS=5 GEMM_SIZES="1024 2048" ./scripts/run_gemm_experiments.sh
```

For report-quality CUDA 13 data on RTX 50 / `sm_120`, use a clean build and
fresh result directory:

```bash
rm -rf build build_cuda13 results
cmake -S . -B build_cuda13 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build_cuda13 -j

BUILD_DIR=/workspace/build_cuda13 PRESET=full TRIALS=5 ./scripts/run_reduce_experiments.sh
BUILD_DIR=/workspace/build_cuda13 PRESET=full TRIALS=5 ./scripts/run_gemm_experiments.sh
BUILD_DIR=/workspace/build_cuda13 PRESET=full TRIALS=5 ./scripts/run_transformer_experiments.sh
```

The default GEMM linear figures keep only key milestones to reduce visual
clutter: cuBLAS FP32 Pedantic, v2, v3, v4, v5, v6, v7, v8a, v8b, and v8c. Branch kernels v3a and v3b remain
available for focused sweeps and appendix-style inspection.
The plotter also writes `gemm_fp32_best_backend_gflops.svg`, which keeps only
the fastest handwritten FP32 backend at each matrix size.

## GEMM Parameter Tuning

The benchmark versions are fixed configurations. To search for better block
tile parameters on a specific GPU, run the tuning script:

```bash
BUILD_DIR=/workspace/build_cuda13 TUNE_SIZES="256 512 1024 2048" ./scripts/run_gemm_tuning.sh
```

It writes:

```txt
results/gemm_tuning/blocksize_tuning.csv
results/gemm_tuning/figures/
├── tuning_thread_tile.svg
├── tuning_vectorized.svg
├── tuning_double_buffer.svg
└── tuning_best_by_kernel.svg
```

The CSV records the kernel family, config name, `BM/BN/BK/TM/TN`, thread count,
shared-memory bytes, average time, GFLOPS, cuBLAS ratio, and correctness flag.
The tuning script automatically plots one figure per backend family and a
best-per-kernel-family summary.

When tuning block size, do not optimize only for the largest `BM x BN`. Watch:

- Thread count: `(BM / TM) * (BN / TN)` must stay within the CUDA block limit.
- Shared memory: larger `BM/BN/BK` can reduce occupancy.
- Register pressure: larger `TM/TN` can increase per-thread registers and lower active warps.
- Alignment: vectorized and double-buffer kernels assume tile-aligned sizes.
- Shape balance: square `128x128` is not always better than rectangular `64x128` or `128x64`.
- Stability: use multiple `TUNE_SIZES`, not only one matrix size.

## Toolchain Notes

CUDA is sensitive to the host compiler and glibc version. On very new Linux distributions, `nvcc` may fail before compiling project code because CUDA headers and system math headers disagree on C23 symbols such as `rsqrt` / `rsqrtf`.

If a minimal file fails:

```cpp
#include <cuda_runtime.h>
int main() { return 0; }
```

the issue is the local CUDA toolchain, not this repository. See [docs/toolchain.md](docs/toolchain.md) for known fixes and container-based workflows.

## Roadmap

- Add benchmark plotting scripts for CSV output.
- Add transpose, scan, histogram, and softmax benchmark modules.
- Add im2col / convolution path with cuDNN comparison.
- Add PyTorch extension examples for selected kernels.
- Add CI build checks once a stable CUDA container image is selected.

## License

MIT License. See [LICENSE](LICENSE).
