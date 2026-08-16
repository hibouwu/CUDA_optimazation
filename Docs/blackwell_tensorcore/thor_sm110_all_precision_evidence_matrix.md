# Thor/SM110 全精度 GEMM 证据矩阵

本表由可执行模型的 numeric coverage 与 full-GEMM support manifest 合并生成。
`implementation_ready` 只表示实现、数值参考和同精度性能 denominator 已经
具备采集条件；`numeric_closure` 只表示所需 Thor 数值证据已经回传并通过
审计。最终 `end_to_end_closed` 还要求六个 residency/shape resource
envelope 只使用精确合同的 closure-qualified capacity，并要求 causal
pipeline DAG 已实现和闭环。

- closure suite：`thor-t5000-tma-ingress-supplement-maxn-20260814-c`
- composition：`base_compute_full_plus_component_supplement`
- base compute/full-GEMM：`thor-t5000-closure-maxn-20260814-d382b57-a` @ `d382b57eae289b458c5290e3d2b7e0daf1b7d7c8`
- component supplement：`thor-t5000-tma-ingress-supplement-maxn-20260814-c` @ `25d8cf71fa566150b64f2eb1dc7f814ce70fa354`
- composite qualification：`closure_qualified`
- precision count：`12`
- implementation ready：`6`
- numeric closed：`4`
- closure-qualified resource envelopes：`0`
- causal pipeline closed：`0`
- integrated empirical ideal envelopes：`0`
- end-to-end closed：`0`
- all precisions end-to-end closed：`false`

| precision | strict upper | compute shapes | implementation | full-GEMM shapes | numerical | denominator | resource envelope | causal DAG | integrated ideal | end-to-end |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fp16_f32` | yes | 3/3 | yes | 3/3 | yes | yes | NO | NO | NO | NO |
| `bf16_f32` | yes | 3/3 | yes | 3/3 | yes | yes | NO | NO | NO | NO |
| `tf32_f32` | NO | 3/3 | yes | 3/3 | yes | yes | NO | NO | NO | NO |
| `e4m3_f32` | yes | 3/3 | yes | 3/3 | yes | yes | NO | NO | NO | NO |
| `e5m2_f32` | yes | 3/3 | yes | 0/3 | NO | NO | NO | NO | NO | NO |
| `e3m2_f32` | NO | 3/3 | NO | 0/3 | NO | NO | NO | NO | NO | NO |
| `e2m3_f32` | NO | 3/3 | NO | 0/3 | NO | NO | NO | NO | NO | NO |
| `e2m1_f32` | NO | 3/3 | NO | 0/3 | NO | NO | NO | NO | NO | NO |
| `mxfp4_f32` | NO | 3/3 | NO | 0/3 | NO | NO | NO | NO | NO | NO |
| `nvfp4_f32` | yes | 3/3 | NO | 0/3 | NO | NO | NO | NO | NO | NO |
| `s8_s32` | yes | 3/3 | yes | 3/3 | yes | yes | NO | NO | NO | NO |
| `u8_s32` | yes | 3/3 | NO | 0/3 | NO | NO | NO | NO | NO | NO |

## 未闭环项

### `fp16_f32`

- support gaps：`none`
- numeric gaps：`none`
- model gaps：`closure_qualified_empirical_envelope_matrix, closure_qualified_causal_pipeline_profile_matrix, integrated_empirical_ideal_envelope_matrix`
- missing compute shapes：`none`
- missing full-GEMM shapes：`none`
- missing empirical envelope scenarios：`n1024.hot_l2, n1024.cold_hbm, n4096.hot_l2, n4096.cold_hbm`
- missing causal profile scenarios：`n1024.hot_l2, n1024.cold_hbm, n2048.hot_l2, n2048.cold_hbm, n4096.hot_l2, n4096.cold_hbm`
- missing integrated ideal scenarios：`n1024.hot_l2, n1024.cold_hbm, n2048.hot_l2, n2048.cold_hbm, n4096.hot_l2, n4096.cold_hbm`

### `bf16_f32`

- support gaps：`none`
- numeric gaps：`none`
- model gaps：`closure_qualified_empirical_envelope_matrix, closure_qualified_causal_pipeline_profile_matrix, integrated_empirical_ideal_envelope_matrix`
- missing compute shapes：`none`
- missing full-GEMM shapes：`none`
- missing empirical envelope scenarios：`n1024.hot_l2, n1024.cold_hbm, n4096.hot_l2, n4096.cold_hbm`
- missing causal profile scenarios：`n1024.hot_l2, n1024.cold_hbm, n2048.hot_l2, n2048.cold_hbm, n4096.hot_l2, n4096.cold_hbm`
- missing integrated ideal scenarios：`n1024.hot_l2, n1024.cold_hbm, n2048.hot_l2, n2048.cold_hbm, n4096.hot_l2, n4096.cold_hbm`

### `tf32_f32`

- support gaps：`none`
- numeric gaps：`strict_compute_upper`
- model gaps：`closure_qualified_empirical_envelope_matrix, closure_qualified_causal_pipeline_profile_matrix, integrated_empirical_ideal_envelope_matrix`
- missing compute shapes：`none`
- missing full-GEMM shapes：`none`
- missing empirical envelope scenarios：`n1024.hot_l2, n1024.cold_hbm, n2048.hot_l2, n2048.cold_hbm, n4096.hot_l2, n4096.cold_hbm`
- missing causal profile scenarios：`n1024.hot_l2, n1024.cold_hbm, n2048.hot_l2, n2048.cold_hbm, n4096.hot_l2, n4096.cold_hbm`
- missing integrated ideal scenarios：`n1024.hot_l2, n1024.cold_hbm, n2048.hot_l2, n2048.cold_hbm, n4096.hot_l2, n4096.cold_hbm`

### `e4m3_f32`

- support gaps：`none`
- numeric gaps：`none`
- model gaps：`closure_qualified_empirical_envelope_matrix, closure_qualified_causal_pipeline_profile_matrix, integrated_empirical_ideal_envelope_matrix`
- missing compute shapes：`none`
- missing full-GEMM shapes：`none`
- missing empirical envelope scenarios：`n1024.hot_l2, n1024.cold_hbm, n2048.hot_l2, n2048.cold_hbm, n4096.hot_l2, n4096.cold_hbm`
- missing causal profile scenarios：`n1024.hot_l2, n1024.cold_hbm, n2048.hot_l2, n2048.cold_hbm, n4096.hot_l2, n4096.cold_hbm`
- missing integrated ideal scenarios：`n1024.hot_l2, n1024.cold_hbm, n2048.hot_l2, n2048.cold_hbm, n4096.hot_l2, n4096.cold_hbm`

### `e5m2_f32`

- support gaps：`none`
- numeric gaps：`full_gemm_observed, closure_qualified_full_gemm_shape_matrix, full_gemm_numerical_validation, same_precision_performance_denominator`
- model gaps：`closure_qualified_empirical_envelope_matrix, closure_qualified_causal_pipeline_profile_matrix, integrated_empirical_ideal_envelope_matrix`
- missing compute shapes：`none`
- missing full-GEMM shapes：`1024, 2048, 4096`
- missing empirical envelope scenarios：`n1024.hot_l2, n1024.cold_hbm, n2048.hot_l2, n2048.cold_hbm, n4096.hot_l2, n4096.cold_hbm`
- missing causal profile scenarios：`n1024.hot_l2, n1024.cold_hbm, n2048.hot_l2, n2048.cold_hbm, n4096.hot_l2, n4096.cold_hbm`
- missing integrated ideal scenarios：`n1024.hot_l2, n1024.cold_hbm, n2048.hot_l2, n2048.cold_hbm, n4096.hot_l2, n4096.cold_hbm`

### `e3m2_f32`

- support gaps：`implementation_status, native_mainloop, closure_candidate_backend, implementation_source, same_contract_numerical_reference, same_precision_performance_denominator_impl`
- numeric gaps：`strict_compute_upper, full_gemm_observed, closure_qualified_full_gemm_shape_matrix, full_gemm_numerical_validation, same_precision_performance_denominator`
- model gaps：`closure_qualified_empirical_envelope_matrix, closure_qualified_causal_pipeline_profile_matrix, integrated_empirical_ideal_envelope_matrix`
- missing compute shapes：`none`
- missing full-GEMM shapes：`1024, 2048, 4096`
- missing empirical envelope scenarios：`n1024.hot_l2, n1024.cold_hbm, n2048.hot_l2, n2048.cold_hbm, n4096.hot_l2, n4096.cold_hbm`
- missing causal profile scenarios：`n1024.hot_l2, n1024.cold_hbm, n2048.hot_l2, n2048.cold_hbm, n4096.hot_l2, n4096.cold_hbm`
- missing integrated ideal scenarios：`n1024.hot_l2, n1024.cold_hbm, n2048.hot_l2, n2048.cold_hbm, n4096.hot_l2, n4096.cold_hbm`
- blocker：No packed FP6 E3M2 full-GEMM implementation, unpack path, or same-contract reference.

### `e2m3_f32`

- support gaps：`implementation_status, native_mainloop, closure_candidate_backend, implementation_source, same_contract_numerical_reference, same_precision_performance_denominator_impl`
- numeric gaps：`strict_compute_upper, full_gemm_observed, closure_qualified_full_gemm_shape_matrix, full_gemm_numerical_validation, same_precision_performance_denominator`
- model gaps：`closure_qualified_empirical_envelope_matrix, closure_qualified_causal_pipeline_profile_matrix, integrated_empirical_ideal_envelope_matrix`
- missing compute shapes：`none`
- missing full-GEMM shapes：`1024, 2048, 4096`
- missing empirical envelope scenarios：`n1024.hot_l2, n1024.cold_hbm, n2048.hot_l2, n2048.cold_hbm, n4096.hot_l2, n4096.cold_hbm`
- missing causal profile scenarios：`n1024.hot_l2, n1024.cold_hbm, n2048.hot_l2, n2048.cold_hbm, n4096.hot_l2, n4096.cold_hbm`
- missing integrated ideal scenarios：`n1024.hot_l2, n1024.cold_hbm, n2048.hot_l2, n2048.cold_hbm, n4096.hot_l2, n4096.cold_hbm`
- blocker：No packed FP6 E2M3 full-GEMM implementation, unpack path, or same-contract reference.

### `e2m1_f32`

- support gaps：`implementation_status, native_mainloop, closure_candidate_backend, implementation_source, same_contract_numerical_reference, same_precision_performance_denominator_impl`
- numeric gaps：`strict_compute_upper, full_gemm_observed, closure_qualified_full_gemm_shape_matrix, full_gemm_numerical_validation, same_precision_performance_denominator`
- model gaps：`closure_qualified_empirical_envelope_matrix, closure_qualified_causal_pipeline_profile_matrix, integrated_empirical_ideal_envelope_matrix`
- missing compute shapes：`none`
- missing full-GEMM shapes：`1024, 2048, 4096`
- missing empirical envelope scenarios：`n1024.hot_l2, n1024.cold_hbm, n2048.hot_l2, n2048.cold_hbm, n4096.hot_l2, n4096.cold_hbm`
- missing causal profile scenarios：`n1024.hot_l2, n1024.cold_hbm, n2048.hot_l2, n2048.cold_hbm, n4096.hot_l2, n4096.cold_hbm`
- missing integrated ideal scenarios：`n1024.hot_l2, n1024.cold_hbm, n2048.hot_l2, n2048.cold_hbm, n4096.hot_l2, n4096.cold_hbm`
- blocker：No raw unscaled E2M1 full-GEMM path; NVFP4/MXFP4 block-scaled results are different contracts.

### `mxfp4_f32`

- support gaps：`implementation_status, closure_candidate_backend, same_contract_numerical_reference, same_precision_performance_denominator_impl`
- numeric gaps：`strict_compute_upper, full_gemm_observed, closure_qualified_full_gemm_shape_matrix, full_gemm_numerical_validation, same_precision_performance_denominator`
- model gaps：`closure_qualified_empirical_envelope_matrix, closure_qualified_causal_pipeline_profile_matrix, integrated_empirical_ideal_envelope_matrix`
- missing compute shapes：`none`
- missing full-GEMM shapes：`1024, 2048, 4096`
- missing empirical envelope scenarios：`n1024.hot_l2, n1024.cold_hbm, n2048.hot_l2, n2048.cold_hbm, n4096.hot_l2, n4096.cold_hbm`
- missing causal profile scenarios：`n1024.hot_l2, n1024.cold_hbm, n2048.hot_l2, n2048.cold_hbm, n4096.hot_l2, n4096.cold_hbm`
- missing integrated ideal scenarios：`n1024.hot_l2, n1024.cold_hbm, n2048.hot_l2, n2048.cold_hbm, n4096.hot_l2, n4096.cold_hbm`
- blocker：Current CUTLASS 72a path outputs BF16, not the model's FP32 output contract.
- blocker：Generated external source is not captured in the historical result bundle.
- blocker：Historical ratio denominator is FP16 cuBLAS, not MXFP4.

### `nvfp4_f32`

- support gaps：`implementation_status, closure_candidate_backend, same_contract_numerical_reference, same_precision_performance_denominator_impl`
- numeric gaps：`full_gemm_observed, closure_qualified_full_gemm_shape_matrix, full_gemm_numerical_validation, same_precision_performance_denominator`
- model gaps：`closure_qualified_empirical_envelope_matrix, closure_qualified_causal_pipeline_profile_matrix, integrated_empirical_ideal_envelope_matrix`
- missing compute shapes：`none`
- missing full-GEMM shapes：`1024, 2048, 4096`
- missing empirical envelope scenarios：`n1024.hot_l2, n1024.cold_hbm, n2048.hot_l2, n2048.cold_hbm, n4096.hot_l2, n4096.cold_hbm`
- missing causal profile scenarios：`n1024.hot_l2, n1024.cold_hbm, n2048.hot_l2, n2048.cold_hbm, n4096.hot_l2, n4096.cold_hbm`
- missing integrated ideal scenarios：`n1024.hot_l2, n1024.cold_hbm, n2048.hot_l2, n2048.cold_hbm, n4096.hot_l2, n4096.cold_hbm`
- blocker：Current CUTLASS 72b path outputs narrow NVFP4 rather than FP32.
- blocker：Generated external source is not captured in the historical result bundle.
- blocker：Historical ratio denominator is FP16 cuBLAS, not NVFP4.

### `s8_s32`

- support gaps：`none`
- numeric gaps：`none`
- model gaps：`closure_qualified_empirical_envelope_matrix, closure_qualified_causal_pipeline_profile_matrix, integrated_empirical_ideal_envelope_matrix`
- missing compute shapes：`none`
- missing full-GEMM shapes：`none`
- missing empirical envelope scenarios：`n1024.hot_l2, n1024.cold_hbm, n2048.hot_l2, n2048.cold_hbm, n4096.hot_l2, n4096.cold_hbm`
- missing causal profile scenarios：`n1024.hot_l2, n1024.cold_hbm, n2048.hot_l2, n2048.cold_hbm, n4096.hot_l2, n4096.cold_hbm`
- missing integrated ideal scenarios：`n1024.hot_l2, n1024.cold_hbm, n2048.hot_l2, n2048.cold_hbm, n4096.hot_l2, n4096.cold_hbm`

### `u8_s32`

- support gaps：`implementation_status, native_mainloop, closure_candidate_backend, implementation_source, same_contract_numerical_reference, same_precision_performance_denominator_impl`
- numeric gaps：`full_gemm_observed, closure_qualified_full_gemm_shape_matrix, full_gemm_numerical_validation, same_precision_performance_denominator`
- model gaps：`closure_qualified_empirical_envelope_matrix, closure_qualified_causal_pipeline_profile_matrix, integrated_empirical_ideal_envelope_matrix`
- missing compute shapes：`none`
- missing full-GEMM shapes：`1024, 2048, 4096`
- missing empirical envelope scenarios：`n1024.hot_l2, n1024.cold_hbm, n2048.hot_l2, n2048.cold_hbm, n4096.hot_l2, n4096.cold_hbm`
- missing causal profile scenarios：`n1024.hot_l2, n1024.cold_hbm, n2048.hot_l2, n2048.cold_hbm, n4096.hot_l2, n4096.cold_hbm`
- missing integrated ideal scenarios：`n1024.hot_l2, n1024.cold_hbm, n2048.hot_l2, n2048.cold_hbm, n4096.hot_l2, n4096.cold_hbm`
- blocker：The official cuBLAS GemmEx support table allows CUBLAS_COMPUTE_32I with CUDA_R_8I inputs, not CUDA_R_8U inputs.
- blocker：A same-contract U8 numerical reference and performance denominator must be implemented before closure collection.

## 关闭条件

每个 precision 必须同时满足：

1. 有条件可证明的 compute rate upper；
2. M128N64、M128N128、M128N256 三个 closure-qualified compute 点；
3. 有仓库内可复现的 native full-GEMM candidate；
4. N=1024、2048、4096 三个完整输出数值验证；
5. 三个 shape 都有同输入精度、同输出类型的 performance denominator；
6. hot-L2/cold-HBM × N=1024/2048/4096 六个 resource envelope 都只选择
   closure-qualified 且与 schedule 显式匹配的 capacity；
7. latency、initiation interval、TMA/MMA/TMEM 依赖和 startup/drain 的
   causal pipeline DAG 已实现，并且每个选中 schedule 有独立审计通过的
   closure-qualified joint profile；
8. trial、源码、编译命令、binary hash、function-scoped SASS、NCU、环境和
   硬件身份通过独立 auditor。
