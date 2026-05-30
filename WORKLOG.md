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

### REDUCE

File: `REDUCE/src/main.cu`

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

- Uses CPU sum as correctness reference.
- Uses CUDA events for timing.
- Uses ping-pong partial buffers for multi-pass reductions.
- Writes `reduce_benchmark.csv`.
- Includes Chinese comments explaining the optimization intent of each version.

### GEMM

File: `GEMM/src/main.cu`

Implemented and documented:

| Version | Kernel / Baseline | Purpose |
| --- | --- | --- |
| baseline | `cublasSgemm` | Correctness and performance reference. |
| v1 | `sgemm_v1_naive` | One thread computes one output element. |
| v2 | `sgemm_v2_smem` | Shared-memory tile reuse. |
| v3 | `sgemm_v3_thread_tile` | Per-thread output tile for higher arithmetic intensity. |
| v4 | `sgemm_v4_vectorized` | `float4` loads and transposed A tile. |
| v5 | `sgemm_v5_double_buffer` | Shared-memory double buffering and register prefetch. |
| v6 | `sgemm_v6_warp_tiling` | Block / warp / thread hierarchical tiling. |

Benchmark behavior:

- Uses cuBLAS as reference output.
- Uses CUDA events for timing.
- Reports GFLOPS and ratio to cuBLAS.
- Writes `sgemm_benchmark.csv`.
- Includes Chinese comments explaining row-major cuBLAS adaptation and each optimization stage.

### Verification Status

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
