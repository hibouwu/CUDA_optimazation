# Notices and provenance

This repository contains original guide code and documentation by Jianye Shi.
The repository is licensed under BSD-3-Clause; individual files adapted from
NVIDIA CUTLASS retain NVIDIA's copyright and SPDX headers.

The pinned dependency under `third_party/cutlass` is NVIDIA CUTLASS v4.6.1 at
commit `e05f953a5b3d38adc240df2ff928e0421c2abba3`. CUTLASS is distributed under
its own BSD-3-Clause license in `third_party/cutlass/LICENSE.txt`.

Concepts were re-derived from the following source material. None of these
paths is a runtime dependency of this repository:

- NVIDIA CUTLASS Blackwell C++ tutorials and unit tests.
- `CUDA_optimazation/GEMMsm110/include/cutlass_sm110_backends.cuh` for the
  application-side `GemmUniversalAdapter` lifecycle.
- `CUDA_optimazation/Docs/cutlass/cute_layout/` for shape-ledger and dataflow
  presentation.
- `CUDA_optimazation/microbench/sm110_full_gemm_campaign/` for function-local
  SASS attribution and evidence-state rules.

Research artifacts, archived performance CSV files, generated binaries, and
machine-specific paths from `CUDA_optimazation` are intentionally not copied.
