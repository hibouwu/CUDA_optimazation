# Changelog

## Unreleased

### Added

- REDUCE benchmark with v0-v5 handwritten kernels and CUB baseline.
- GEMM benchmark with v1-v6 mainline kernels, v3b/warp1 branch kernels, and cuBLAS baseline.
- Changed `cublas` to use `CUBLAS_PEDANTIC_MATH` as the single strict FP32 GEMM reference.
- CSV output for reduce and SGEMM benchmark results.
- Extended experiment script with automatic SVG plotting.
- Transformer optimization scaffold with CMake target, roadmap, and LayerNorm v1-v3 benchmark.
- Transformer LayerNorm sweep script and SVG plots split by hidden size.
- CMake build entry points for `reduce_bench` and `gemm_bench`.
- Chinese source comments for kernel optimization stages.
- Split GEMM and REDUCE source into `include/` helpers plus compact `src/main.cu` entry points.
- Benchmark and toolchain documentation under `docs/`.
- MIT license, contributing guide, and git ignore rules.

### Notes

- Current local verification is blocked by a CUDA 13.0 / GCC 15 / glibc 2.42 header compatibility issue. See `docs/toolchain.md`.
