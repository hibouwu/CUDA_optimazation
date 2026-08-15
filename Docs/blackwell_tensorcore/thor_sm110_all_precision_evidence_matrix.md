# Thor/SM110 全精度 GEMM 证据矩阵

本表由可执行模型的 numeric coverage 与 full-GEMM support manifest 合并生成。
`implementation_ready` 只表示实现、数值参考和同精度性能 denominator 已经
具备采集条件；`numeric_closure` 只表示所需 Thor 证据已经回传并通过审计；
只有二者同时成立，`end_to_end_closed` 才为真。

- closure suite：`thor-t5000-tma-ingress-supplement-maxn-20260814-c`
- evidence commit：`25d8cf71fa566150b64f2eb1dc7f814ce70fa354`
- precision count：`12`
- implementation ready：`5`
- numeric closed：`4`
- end-to-end closed：`4`
- all precisions end-to-end closed：`false`

| precision | strict upper | compute shapes | implementation | full-GEMM shapes | numerical | denominator | end-to-end |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fp16_f32` | yes | 3/3 | yes | 3/3 | yes | yes | yes |
| `bf16_f32` | yes | 3/3 | yes | 3/3 | yes | yes | yes |
| `tf32_f32` | NO | 3/3 | yes | 3/3 | yes | yes | NO |
| `e4m3_f32` | yes | 3/3 | yes | 3/3 | yes | yes | yes |
| `e5m2_f32` | yes | 3/3 | NO | 0/3 | NO | NO | NO |
| `e3m2_f32` | NO | 3/3 | NO | 0/3 | NO | NO | NO |
| `e2m3_f32` | NO | 3/3 | NO | 0/3 | NO | NO | NO |
| `e2m1_f32` | NO | 3/3 | NO | 0/3 | NO | NO | NO |
| `mxfp4_f32` | NO | 3/3 | NO | 0/3 | NO | NO | NO |
| `nvfp4_f32` | yes | 3/3 | NO | 0/3 | NO | NO | NO |
| `s8_s32` | yes | 3/3 | yes | 3/3 | yes | yes | yes |
| `u8_s32` | yes | 3/3 | NO | 0/3 | NO | NO | NO |

## 未闭环项

### `tf32_f32`

- support gaps：`none`
- numeric gaps：`strict_compute_upper`
- missing compute shapes：`none`
- missing full-GEMM shapes：`none`

### `e5m2_f32`

- support gaps：`implementation_status, closure_candidate_backend, same_contract_numerical_reference, same_precision_performance_denominator_impl`
- numeric gaps：`full_gemm_observed, closure_qualified_full_gemm_shape_matrix, full_gemm_numerical_validation, same_precision_performance_denominator`
- missing compute shapes：`none`
- missing full-GEMM shapes：`1024, 2048, 4096`
- blocker：The cuBLASLt FP8 support table does not list E5M2 times E5M2 as a supported A/B pair, so the attempted library reference is not a valid closure contract.
- blocker：A captured CUTLASS or other independent same-contract full-output reference and performance denominator is required.

### `e3m2_f32`

- support gaps：`implementation_status, native_mainloop, closure_candidate_backend, implementation_source, same_contract_numerical_reference, same_precision_performance_denominator_impl`
- numeric gaps：`strict_compute_upper, full_gemm_observed, closure_qualified_full_gemm_shape_matrix, full_gemm_numerical_validation, same_precision_performance_denominator`
- missing compute shapes：`none`
- missing full-GEMM shapes：`1024, 2048, 4096`
- blocker：No packed FP6 E3M2 full-GEMM implementation, unpack path, or same-contract reference.

### `e2m3_f32`

- support gaps：`implementation_status, native_mainloop, closure_candidate_backend, implementation_source, same_contract_numerical_reference, same_precision_performance_denominator_impl`
- numeric gaps：`strict_compute_upper, full_gemm_observed, closure_qualified_full_gemm_shape_matrix, full_gemm_numerical_validation, same_precision_performance_denominator`
- missing compute shapes：`none`
- missing full-GEMM shapes：`1024, 2048, 4096`
- blocker：No packed FP6 E2M3 full-GEMM implementation, unpack path, or same-contract reference.

### `e2m1_f32`

- support gaps：`implementation_status, native_mainloop, closure_candidate_backend, implementation_source, same_contract_numerical_reference, same_precision_performance_denominator_impl`
- numeric gaps：`strict_compute_upper, full_gemm_observed, closure_qualified_full_gemm_shape_matrix, full_gemm_numerical_validation, same_precision_performance_denominator`
- missing compute shapes：`none`
- missing full-GEMM shapes：`1024, 2048, 4096`
- blocker：No raw unscaled E2M1 full-GEMM path; NVFP4/MXFP4 block-scaled results are different contracts.

### `mxfp4_f32`

- support gaps：`implementation_status, closure_candidate_backend, same_contract_numerical_reference, same_precision_performance_denominator_impl`
- numeric gaps：`strict_compute_upper, full_gemm_observed, closure_qualified_full_gemm_shape_matrix, full_gemm_numerical_validation, same_precision_performance_denominator`
- missing compute shapes：`none`
- missing full-GEMM shapes：`1024, 2048, 4096`
- blocker：Current CUTLASS 72a path outputs BF16, not the model's FP32 output contract.
- blocker：Generated external source is not captured in the historical result bundle.
- blocker：Historical ratio denominator is FP16 cuBLAS, not MXFP4.

### `nvfp4_f32`

- support gaps：`implementation_status, closure_candidate_backend, same_contract_numerical_reference, same_precision_performance_denominator_impl`
- numeric gaps：`full_gemm_observed, closure_qualified_full_gemm_shape_matrix, full_gemm_numerical_validation, same_precision_performance_denominator`
- missing compute shapes：`none`
- missing full-GEMM shapes：`1024, 2048, 4096`
- blocker：Current CUTLASS 72b path outputs narrow NVFP4 rather than FP32.
- blocker：Generated external source is not captured in the historical result bundle.
- blocker：Historical ratio denominator is FP16 cuBLAS, not NVFP4.

### `u8_s32`

- support gaps：`implementation_status, native_mainloop, closure_candidate_backend, implementation_source, same_contract_numerical_reference, same_precision_performance_denominator_impl`
- numeric gaps：`full_gemm_observed, closure_qualified_full_gemm_shape_matrix, full_gemm_numerical_validation, same_precision_performance_denominator`
- missing compute shapes：`none`
- missing full-GEMM shapes：`1024, 2048, 4096`
- blocker：The official cuBLAS GemmEx support table allows CUBLAS_COMPUTE_32I with CUDA_R_8I inputs, not CUDA_R_8U inputs.
- blocker：A same-contract U8 numerical reference and performance denominator must be implemented before closure collection.

## 关闭条件

每个 precision 必须同时满足：

1. 有条件可证明的 compute rate upper；
2. M128N64、M128N128、M128N256 三个 closure-qualified compute 点；
3. 有仓库内可复现的 native full-GEMM candidate；
4. N=1024、2048、4096 三个完整输出数值验证；
5. 三个 shape 都有同输入精度、同输出类型的 performance denominator；
6. trial、源码、编译命令、binary hash、function-scoped SASS、NCU、环境和
   硬件身份通过独立 auditor。
