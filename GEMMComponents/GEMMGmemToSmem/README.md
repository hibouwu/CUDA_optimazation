# GEMMGmemToSmem Demos

These demos isolate the `global memory -> shared memory` staging step used in
GEMM kernels. Each subdirectory contains a standalone `demo.cu` that measures
copy latency and effective bandwidth with CUDA events.

## Layout

| Folder | Focus |
| --- | --- |
| `00_naive_gmem_to_smem` | Uncoalesced baseline |
| `01_coalesced_gmem_to_smem` | Coalesced row-major global loads |
| `02_vectorized_float4_load` | `float4` / 16-byte vectorized loads |
| `03_row_col_major_addressing` | Row-major gmem to column-major smem addressing |
| `03_transposed_smem_store` | Direct transposed shared-memory store |
| `04_coalesced_load_transposed_store` | Classic transpose staging pattern |
| `04_smem_padding_bank_conflict` | Shared-memory padding to remove bank conflicts |
| `05_smem_swizzle_store` | XOR swizzle instead of padding |
| `05_transpose_padding_bank_conflict` | Padded transpose with tail predication |
| `06_predicated_tile_load` | Safe tail handling for non-multiple tile sizes |
| `07_double_buffer_gmem_to_smem` | Shared-memory ping-pong buffering |
| `08_cp_async_gmem_to_smem` | `cp.async` 16-byte lane copies |
| `09_tma_gmem_to_smem` | Tensor map setup plus device-side TMA scaffold |

## Runtime Arguments

All demos use the same positional arguments:

```bash
demo [rows] [cols] [iters] [warmup]
```

Defaults:

- Most demos: `4096 4096 200 20`
- `06_predicated_tile_load`: `4093 4091 200 20`
- `05_transpose_padding_bank_conflict`: `4093 4091 200 20`

The program prints:

- demo name and short summary
- GPU name and compute capability
- average kernel time in milliseconds
- effective bandwidth in GB/s
- `max_abs_diff` against a host reference

## Build

Targets are wired into the top-level [CMakeLists.txt](/home/jianyeshi/Note/CUDA/CUDA_optimazation/CMakeLists.txt).

Example:

```bash
cmake -S CUDA_optimazation -B build_cuda13 -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build_cuda13 -j --target gmem_to_smem_08_cp_async
./build_cuda13/gmem_to_smem_08_cp_async 4096 4096 200 20
```

If local `nvcc` fails even on a minimal file due to `rsqrt/rsqrtf` header
conflicts, use the container workflow documented in
[toolchain.md](/home/jianyeshi/Note/CUDA/CUDA_optimazation/docs/toolchain.md).

## Notes

- `02_vectorized_float4_load` and `08_cp_async_gmem_to_smem` require
  `cols % 4 == 0`.
- `09_tma_gmem_to_smem` encodes a real `CUtensorMap` on the host side, but the
  device kernel is still a scaffold. Replace `kernel_tma_scaffold` in
  [gmem_to_smem_demo.cuh](/home/jianyeshi/Note/CUDA/CUDA_optimazation/GEMMComponents/GEMMGmemToSmem/common/gmem_to_smem_demo.cuh)
  with a `cp.async.bulk.tensor` path when you want a true Hopper/Blackwell TMA
  benchmark.
