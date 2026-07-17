# GEMMquant_sm110 - SM110 Quantized GEMM Roadmap

This directory is the isolated workspace for 1024x1024x1024 quantized GEMM
experiments on Thor/SM110. It is intentionally separate from `GEMMsm110`
because the output semantics, reference checks, scaling metadata, and
performance targets differ from the FP32-output GEMM line.

The target is strict: each precision family must eventually have staged
backends and a best implementation that reaches at least 90% of the relevant
cuBLAS/cuBLASLt reference on the 1024 square problem.

## Precision Tracks

The planned tracks are:

| Precision | Initial reuse | Required final evidence |
| --- | --- | --- |
| NVFP4 | Existing `GEMMsm110` `tc6` fused epilogue | 10/10 matched, best backend >= 0.90x reference |
| MXFP4 | New SM110 quantized GEMM path | 10/10 matched, best backend >= 0.90x reference |
| FP8 | `q0..q3` scalar CUDA, `q4..q7` FP8 MMA, and `q8` cuBLASLt backend | 10/10 matched, best backend >= 0.90x reference |
| INT8 | `q0..q3` scalar CUDA, `q4..q17` WMMA, `q18` inline MMA, and `q19` cuBLAS backend | 10/10 matched, best backend >= 0.90x reference |

The current first step is a runnable harness and plotting path. It does not
pretend that missing precision backends are done. Missing rows are recorded as
`Status=missing`, leave timing/performance fields empty, and are excluded from
best-performance plots.

## Current Entry Points

From the repository root:

```bash
python3 GEMMquant_sm110/scripts/run_quant_gemm_1024.py --trials 10
python3 GEMMquant_sm110/scripts/plot_quant_gemm.py \
  --csv results/quant_gemm_sm110/sm110_quant_gemm_1024_sweep.csv \
  --out results/quant_gemm_sm110/figures/sm110_quant_gemm_1024_best_by_precision.svg
```

FP4/cuBLASLt support probe:

```bash
GEMMquant_sm110/build_and_run.sh fp4-probe 128
```

Build the CUTLASS FP4 backends directly:

```bash
GEMMquant_sm110/build_and_run.sh build-fp4-cutlass
```

The runner reuses `GEMMsm110/build/gemm_sm110_bench` for the first NVFP4 stage.
If that binary is missing, build `GEMMsm110` first. FP8 and INT8 staged CUDA
baselines use the local `GEMMquant_sm110/build/quant_gemm_sm110_bench`; the
Python runner builds it automatically when it is absent. NVFP4 and MXFP4
CUTLASS FP4 backends are generated from CUTLASS example 72b/72a sources and
default to `CUTLASS_FP4_ROOT=/xplorer/op630/_deps/flash-attention/csrc/cutlass`.
Set `CUTLASS_FP4_ROOT` if that checkout is not available.

Current 10-trial status: the CSV has 340 rows. FP8 `q0..q8`, INT8 `q0..q19`,
NVFP4 `q0/q2`, and MXFP4 `q0/q1` are implemented and matched. `NVFP4 q1`
remains a planned native-mainloop row and is explicitly marked
`Status=missing`; it is not plotted as zero and does not participate in best
selection.

Best 10-trial means:

| Precision | Best backend | GFLOP/s | Ratio |
| --- | --- | ---: | ---: |
| NVFP4 | `nvfp4_q2_cutlass_72b_nvfp4` | 129072.3 | 1.261171x |
| MXFP4 | `mxfp4_q1_cutlass_72a_swizzle1` | 125623.5 | 1.229672x |
| FP8 | `fp8_q8_cublaslt_matmul` | 134273.0 | 1.002803x |
| INT8 | `int8_q19_cublas_gemmex` | 122770.5 | 1.011083x |

Additional FP4 probe evidence is kept under
`results/quant_gemm_sm110/raw/cutlass_fp4_probes/`. Earlier CUTLASS 4.3.3
MXFP4 and PyTorch NVFP4 probes remain there as bring-up history; the formal
CSV now uses CUTLASS 4.3.4 example-derived backends with host-reference
comparison and 100 internal timing iterations per trial.

The CUDA 13 headers expose `CUDA_R_4F_E2M1` and `CUDA_R_8F_UE8M0`, so a
cuBLASLt FP4 descriptor probe has been added. On the current Thor runtime the
probe tries row-major and column-major/swap descriptors, `VEC32_UE8M0` and
`VEC16_UE4M3` scale modes, FP16 and FP32 outputs, and both
`CUBLAS_COMPUTE_32F_FAST_16F` and `CUBLAS_COMPUTE_32F`. All 12 probed
descriptors fail at `cublasLtMatmulAlgoGetHeuristic`; `VEC32_UE8M0` returns
`status=7, returned=0`, and `VEC16_UE4M3` returns `status=15, returned=0`.
The captured log is
`results/quant_gemm_sm110/raw/fp4_cublaslt_probe_128_descriptor_sweep.txt`.
This is recorded as probe evidence, not a working MXFP4/NVFP4 backend.

Do not treat `/xplorer/op630/test/tcgen05_fp4_gemm` as a formal matched
backend: its binary reports high FP4 performance, but the visible source only
checks CUDA error in `verify()` and does not compare against a host reference.
The formal NVFP4/MXFP4 CUTLASS backends restore the CUTLASS host reference path
and only mark rows `Matched=1` when `Disposition: Passed` is present.

## Naming

Canonical artifacts:

```text
results/quant_gemm_sm110/sm110_quant_gemm_1024_sweep.csv
results/quant_gemm_sm110/figures/sm110_quant_gemm_1024_best_by_precision.svg
GEMMquant_sm110/SM110_QUANT_GEMM_OPTIMIZATION_REPORT.md
```

The naming includes `sm110`, `quant_gemm`, and `1024` so that it does not get
confused with the FP32-output `GEMMsm110` sweep.
