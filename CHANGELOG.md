# Changelog

## Unreleased

### Added

- REDUCE benchmark with v0-v5 handwritten kernels and CUB baseline.
- GEMM benchmark with v1-v6 handwritten kernels and cuBLAS baseline.
- CSV output for reduce and SGEMM benchmark results.
- CMake build entry points for `reduce_bench` and `gemm_bench`.
- Chinese source comments for kernel optimization stages.
- Benchmark and toolchain documentation under `docs/`.
- MIT license, contributing guide, and git ignore rules.

### Notes

- Current local verification is blocked by a CUDA 13.0 / GCC 15 / glibc 2.42 header compatibility issue. See `docs/toolchain.md`.
