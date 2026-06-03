# Worklog

This file records the current implementation state and packaging decisions for the repository.

## 2026-05-30

### Project Packaging

- Reworked `README.md` into a public project landing page.
- Added `docs/benchmark.md` for measurement rules and Nsight Compute metric suggestions.
- Added `docs/toolchain.md` for CUDA / host compiler compatibility notes.
- Added `.gitignore` for build outputs, profiler reports, generated CSV files, and editor files.
- Added `LICENSE` using the MIT License.
- Improved `CMakeLists.txt` with a default Release build, default CUDA architecture, and compile command export.
- Split GEMM and REDUCE into common definitions, kernels, benchmark helpers, and small `main.cu` entry points.
- Added separate `scripts/run_reduce_experiments.sh` and `scripts/run_gemm_experiments.sh` sweep scripts plus `scripts/plot_benchmarks.py` for SVG figures with standard-deviation error bars.
- Added `GEMM/src/tune_blocksize.cu` and `scripts/run_gemm_tuning.sh` for block tile / thread tile parameter sweeps.
- Added `TRANSFORMER/` scaffold with roadmap, common CUDA helpers, and a
  `transformer_bench` LayerNorm benchmark target.
- Added `scripts/run_transformer_experiments.sh` and Transformer plotting support
  in `scripts/plot_benchmarks.py`; LayerNorm sweep figures are split by hidden
  size into latency and effective-bandwidth charts.

### REDUCE

Files:

- `REDUCE/include/reduce_common.cuh`
- `REDUCE/include/reduce_kernels.cuh`
- `REDUCE/include/reduce_benchmark.cuh`
- `REDUCE/src/main.cu`

Implemented and documented:

| Version | Kernel / Baseline | Purpose |
| --- | --- | --- |
| v0 | `reduce_v0_interleaved` | Naive interleaved shared-memory reduction. |
| v1 | `reduce_v1_sequential` | Sequential addressing with contiguous active threads. |
| v2 | `reduce_v2_first_add` | First add during global load to reduce partial outputs. |
| v3 | `reduce_v3_unroll_last_warp` | Manual last-warp unroll to reduce barrier overhead. |
| v4 | `reduce_v4_shuffle` | Warp shuffle reduction to reduce shared-memory traffic. |
| v5 | `reduce_v5_vectorized` | `float4` vectorized loads for fewer global-memory instructions. |
| baseline | `cub::DeviceReduce::Sum` | Production baseline for comparison. |

Benchmark behavior:

- Uses double-precision CPU sum as correctness reference. This avoids false
  failures when sequential FP32 accumulation loses small increments at large N.
- Uses CUDA events for timing.
- Uses ping-pong partial buffers for multi-pass reductions.
- Writes `reduce_benchmark.csv`.
- Includes Chinese comments explaining the optimization intent of each version.

### GEMM

Files:

- `GEMM/include/gemm_common.cuh`
- `GEMM/include/sgemm_kernels.cuh`
- `GEMM/include/gemm_benchmark.cuh`
- `GEMM/src/main.cu`

Implemented and documented:

| Version | Kernel / Baseline | Purpose |
| --- | --- | --- |
| baseline | `cublasSgemm` | Correctness and performance reference. |
| v1 | `sgemm_v1_naive_uncoalesced` | Naive mapping with poor coalescing. |
| v2 | `sgemm_v1_naive` | Coalesced naive mapping. |
| v3 | `sgemm_v2_smem` | Shared-memory tile reuse. |
| v4 | `sgemm_v4_smem_1d_padded` | 1D thread block plus shared-memory padding. |
| v5 | `sgemm_v6_warp_tiling` | Block / warp / thread hierarchical tiling. |
| v6 | `sgemm_v3_thread_tile` | Per-thread output tile for higher arithmetic intensity. |
| v9 | `sgemm_v4_vectorized` | `float4` loads and transposed A tile. |
| v10 | `sgemm_v5_double_buffer` | Shared-memory double buffering and register prefetch. |

Note: `notes/CUDA/5_4_总结_CUDA_MATAMUL优化.md` documents Kernel 1, 2, 3, 4, 5, 6, 9, and 10. Kernel 7 and 8 are not expanded in that note, so the benchmark mirrors the documented stages instead of inventing placeholder kernels.

Benchmark behavior:

- Uses cuBLAS as reference output.
- Uses CUDA events for timing. Each GEMM backend warms up independently and then
  runs 100 timed launches before the benchmark moves to the next backend.
- Reports GFLOPS and ratio to cuBLAS.
- Writes `sgemm_benchmark.csv`.
- Includes Chinese comments explaining row-major cuBLAS adaptation and each optimization stage.

### Verification Status

Container verification with `nvcr.io/nvidia/cuda:12.4.1-devel-ubuntu22.04`:

```bash
nvcc -O3 -std=c++17 -arch=sm_86 -IREDUCE/include REDUCE/src/main.cu -o /tmp/reduce_bench
/tmp/reduce_bench 16777216
```

Result: all REDUCE versions and `cub::DeviceReduce::Sum` report `matched=1`.

For RTX 50 / `sm_120`, the recommended benchmark container is now
`nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu24.04` so nvcc and cuBLAS target the
newer architecture directly.

The current local machine fails to compile even a minimal CUDA include:

```cpp
#include <cuda_runtime.h>
int main() { return 0; }
```

Observed toolchain:

```txt
CUDA Toolkit: 13.0
GCC/G++:      15.2.1
glibc:        2.42
```

Failure:

```txt
exception specification is incompatible with previous function "rsqrt"
exception specification is incompatible with previous function "rsqrtf"
```

This is a CUDA / host toolchain compatibility issue. See `docs/toolchain.md` for recommended container-based builds and alternatives.

## Next Work

- Add plotting scripts for benchmark CSV output.
- Add CI using a pinned NVIDIA CUDA container image.
- Add transpose, softmax, scan, histogram, and convolution modules.
- Add PyTorch extension examples for selected kernels.
