# CUDA Optimization

CUDA Optimization is a compact CUDA kernel optimization lab. It keeps each operator as a versioned benchmark, starting from a readable naive implementation and moving step by step toward production-grade library baselines such as CUB and cuBLAS.

The project is intended for learning and profiling. Each version answers one question: which bottleneck does this optimization remove, and how does that show up in CUDA event timing or Nsight Compute metrics?

## Highlights

- Reduce kernels from interleaved shared-memory reduction to warp shuffle, vectorized loads, and CUB baseline.
- SGEMM kernels from one-thread-one-output naive code to shared memory tiling, thread tiling, vectorized loads, double buffering, warp tiling, and cuBLAS baseline.
- Reproducible benchmark harnesses with warmup, repeated timing, correctness checks, and CSV output.
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
├── GEMM/
│   └── src/main.cu
└── REDUCE/
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
| baseline | `cublasSgemm` | cuBLAS reference result and performance baseline | target throughput |
| v1 | `sgemm_v1_naive` | One thread computes one output element | global load pressure |
| v2 | `sgemm_v2_smem` | Shared-memory tile reuse | shared-memory throughput |
| v3 | `sgemm_v3_thread_tile` | One thread computes a `TM x TN` output tile | arithmetic intensity, register use |
| v4 | `sgemm_v4_vectorized` | `float4` loads and transposed A tile in shared memory | memory instruction count |
| v5 | `sgemm_v5_double_buffer` | Shared-memory ping-pong buffers plus register prefetch | long scoreboard stalls |
| v6 | `sgemm_v6_warp_tiling` | block / warp / thread hierarchical tiling | eligible warps, occupancy |

Output CSV:

```txt
sgemm_benchmark.csv
Version,N,TimeMs,GFLOPS,RatioToCuBLAS,Matched
```

`v4` to `v6` are intentionally written as high-performance paths and require `N` to be a multiple of 128. Other sizes still run `v1` to `v3` and cuBLAS.

## Quick Start

### Option A: CMake

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j

./build/reduce_bench 16777216
./build/gemm_bench 1024
```

### Option B: Direct nvcc

```bash
nvcc -O3 -std=c++17 -arch=sm_86 REDUCE/src/main.cu -o reduce_bench
nvcc -O3 -std=c++17 -arch=sm_86 GEMM/src/main.cu -lcublas -o gemm_bench

./reduce_bench 16777216
./gemm_bench 1024
```

Change `sm_86` to your GPU architecture, for example `sm_75`, `sm_80`, `sm_89`, or `sm_90`.

## Profiling

Use Nsight Compute after correctness passes:

```bash
ncu --set full ./reduce_bench 16777216
ncu --set full ./gemm_bench 1024
```

Recommended first comparisons:

- Reduce: v0 vs v1 vs v4 vs v5 vs CUB.
- GEMM: v1 vs v2 vs v4 vs v5 vs v6 vs cuBLAS.

See [docs/benchmark.md](docs/benchmark.md) for measurement rules and metric suggestions.

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
