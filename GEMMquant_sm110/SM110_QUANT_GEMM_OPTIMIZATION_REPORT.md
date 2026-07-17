# SM110 Quantized GEMM Optimization Report

This directory is the isolated quantized GEMM line for 1024x1024x1024 on
Thor/SM110. The formal harness is
`GEMMquant_sm110/scripts/run_quant_gemm_1024.py`; it writes one CSV covering
NVFP4, MXFP4, FP8, and INT8 staged backends. Rows carry `Precision`, `Stage`,
`BackendId`, `GFLOPS`, `RatioToReference`, `Matched`, `Status`, and `Reason`.
Missing or planned paths remain `Status=missing` with empty timing fields and
are excluded from plots.

Latest formal artifacts:

```text
results/quant_gemm_sm110/sm110_quant_gemm_1024_sweep.csv
results/quant_gemm_sm110/figures/sm110_quant_gemm_1024_best_by_precision.svg
```

The latest formal sweep has 340 rows:

| Precision | Implemented rows | Missing rows | Best backend | 10-trial mean GFLOP/s | Mean ratio |
| --- | ---: | ---: | --- | ---: | ---: |
| NVFP4 | 20 | 10 | `nvfp4_q2_cutlass_72b_nvfp4` | 129072.3 | 1.261171x |
| MXFP4 | 20 | 0 | `mxfp4_q1_cutlass_72a_swizzle1` | 125623.5 | 1.229672x |
| FP8 | 90 | 0 | `fp8_q8_cublaslt_matmul` | 134273.0 | 1.002803x |
| INT8 | 200 | 0 | `int8_q19_cublas_gemmex` | 122770.5 | 1.011083x |

All four precision families now have a 10/10 `Status=ok, Matched=1` best
backend above the 0.90x target. `NVFP4 q1` remains an explicit planned row for
a native FP4 mainloop variant and is not used as completion evidence.

## FP4 Backends

The formal NVFP4 backend `nvfp4_q2_cutlass_72b_nvfp4` is generated from CUTLASS
example 72b (`72b_blackwell_nvfp4_nvfp4_gemm.cu`) with an SM110 runtime guard.
The original CUTLASS host reference comparison is preserved. A generated
`float8.h` overlay restricts `CUDA_PTX_UE8M0_CVT_ENABLED` to device
compilation so the host reference uses the portable C++ UE8M0 conversion path.
The backend is compiled with SM110A enabled and timed with 100 internal
iterations per trial.

The formal MXFP4 backends are generated from CUTLASS example 72a
(`72a_blackwell_nvfp4_bf16_gemm.cu`) by replacing `nv_float4_t` input types with
`mx_float4_t`. `mxfp4_q0_cutlass_72a_mxfp4_bf16` uses the default swizzle 2
bring-up, and `mxfp4_q1_cutlass_72a_swizzle1` records the swizzle-tuned stage.
Both preserve CUTLASS host reference comparison.

Build entry point:

```bash
GEMMquant_sm110/build_and_run.sh build-fp4-cutlass
```

The build defaults to:

```text
CUTLASS_FP4_ROOT=/xplorer/op630/_deps/flash-attention/csrc/cutlass
```

## FP8 And INT8

FP8 has staged CUDA scalar baselines, inline MMA baselines, and the final
cuBLASLt backend `fp8_q8_cublaslt_matmul`. The best backend uses FP8 E4M3
inputs, FP32 output, and cuBLASLt matmul descriptors. It is 10/10 matched and
above the 0.90x target.

INT8 has staged scalar, WMMA, inline MMA, and cuBLAS backends. The best backend
is `int8_q19_cublas_gemmex`, using int8 inputs, int32 output, and
`cublasGemmEx`. It is 10/10 matched and above the 0.90x target.

## Probe Notes

The cuBLASLt FP4 descriptor probe remains negative on this runtime. It covers
row-major and column-major/swap descriptors, `VEC32_UE8M0` and `VEC16_UE4M3`
scale modes, FP16 and FP32 outputs, and relevant compute modes. No runnable
algorithm is returned. The log is:

```text
results/quant_gemm_sm110/raw/fp4_cublaslt_probe_128_descriptor_sweep.txt
```

Do not treat `/xplorer/op630/test/tcgen05_fp4_gemm` as formal evidence. Its
visible source only synchronizes output and checks CUDA error in `verify()`; it
does not compare against a reference. The formal FP4 entries use restored
CUTLASS host-reference checks and only set `Matched=1` when the CUTLASS run
prints `Disposition: Passed`.
