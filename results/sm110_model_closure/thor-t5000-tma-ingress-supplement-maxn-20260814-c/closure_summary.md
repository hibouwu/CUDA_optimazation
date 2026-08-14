# Thor/SM110 GEMM closure 数值摘要

- suite：`thor-t5000-tma-ingress-supplement-maxn-20260814-c`
- commit：`25d8cf71fa566150b64f2eb1dc7f814ce70fa354`
- composition：`base_compute_full_plus_component_supplement`
- campaign sources：`{"base": {"expected_commit": "d382b57eae289b458c5290e3d2b7e0daf1b7d7c8", "provides": ["epilogue_preflight", "compute", "full_gemm"], "suite_id": "thor-t5000-closure-maxn-20260814-d382b57-a"}, "component_supplement": {"expected_commit": "25d8cf71fa566150b64f2eb1dc7f814ce70fa354", "provides": ["component"], "suite_id": "thor-t5000-tma-ingress-supplement-maxn-20260814-c"}}`
- qualification：`closure_qualified`
- audit pass：`True`
- campaign measurement closed：`True`
- all precisions closed：`False`
- all common resources closed：`True`
- capacity：54 项
- base/profile capacity：19 项
- full-GEMM observation：15 项
- overcurrent delta：`{"thor-t5000-closure-maxn-20260814-d382b57-a": {"/sys/class/hwmon/hwmon5/oc1_event_cnt": 0, "/sys/class/hwmon/hwmon5/oc2_event_cnt": 0, "/sys/class/hwmon/hwmon5/oc3_event_cnt": 179}, "thor-t5000-tma-ingress-supplement-maxn-20260814-c": {"/sys/class/hwmon/hwmon5/oc1_event_cnt": 0, "/sys/class/hwmon/hwmon5/oc2_event_cnt": 0, "/sys/class/hwmon/hwmon5/oc3_event_cnt": 0}}`

## Closure-qualified compute/component capacities

| Resource | Case | Median rate | Trials | Evidence | Qualification | Source |
| --- | --- | ---: | ---: | --- | --- | --- |
| `epilogue.nvfp4_requant` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.nvfp4_requant_4096x1024_constant` | 2.460 Gelement/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |
| `epilogue.nvfp4_requant` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.nvfp4_requant_4096x1024_normal` | 2.460 Gelement/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |
| `epilogue.nvfp4_requant` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.nvfp4_requant_4096x1024_outlier` | 2.460 Gelement/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |
| `hbm.read` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.hbm_read_aggregate` | 253.588 GB/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |
| `hbm.write` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.hbm_write_aggregate` | 201.158 GB/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |
| `l2.read` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.l2_read_aggregate` | 1505.112 GB/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |
| `l2.write` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.l2_write_aggregate` | 545.416 GB/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |
| `tensor.bf16.m128n128` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.bf16_f32.m128n128` | 256.984 TFLOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.bf16.m128n256` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.bf16_f32.m128n256` | 229.422 TFLOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.bf16.m128n64` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.bf16_f32.m128n64` | 171.052 TFLOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.e2m1.m128n128` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.e2m1_f32.m128n128` | 513.988 TFLOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.e2m1.m128n256` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.e2m1_f32.m128n256` | 516.044 TFLOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.e2m1.m128n64` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.e2m1_f32.m128n64` | 342.059 TFLOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.e2m3.m128n128` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.e2m3_f32.m128n128` | 411.739 TFLOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.e2m3.m128n256` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.e2m3_f32.m128n256` | 413.059 TFLOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.e2m3.m128n64` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.e2m3_f32.m128n64` | 342.041 TFLOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.e3m2.m128n128` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.e3m2_f32.m128n128` | 411.739 TFLOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.e3m2.m128n256` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.e3m2_f32.m128n256` | 413.059 TFLOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.e3m2.m128n64` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.e3m2_f32.m128n64` | 342.019 TFLOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.e4m3.m128n128` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.e4m3_f32.m128n128` | 411.733 TFLOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.e4m3.m128n256` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.e4m3_f32.m128n256` | 413.056 TFLOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.e4m3.m128n64` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.e4m3_f32.m128n64` | 342.126 TFLOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.e5m2.m128n128` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.e5m2_f32.m128n128` | 464.766 TFLOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.e5m2.m128n256` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.e5m2_f32.m128n256` | 458.843 TFLOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.e5m2.m128n64` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.e5m2_f32.m128n64` | 342.068 TFLOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.fp16.m128n128` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.fp16_f32.m128n128` | 205.871 TFLOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.fp16.m128n256` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.fp16_f32.m128n256` | 206.530 TFLOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.fp16.m128n64` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.fp16_f32.m128n64` | 171.030 TFLOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.mxfp4.m128n128` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.mxfp4_f32.m128n128` | 1027.975 TFLOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.mxfp4.m128n256` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.mxfp4_f32.m128n256` | 1032.093 TFLOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.mxfp4.m128n64` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.mxfp4_f32.m128n64` | 684.217 TFLOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.nvfp4.m128n128` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.nvfp4_f32.m128n128` | 1027.975 TFLOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.nvfp4.m128n256` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.nvfp4_f32.m128n256` | 1032.093 TFLOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.nvfp4.m128n64` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.nvfp4_f32.m128n64` | 684.342 TFLOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.s8.m128n128` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.s8_s32.m128n128` | 513.983 TOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.s8.m128n256` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.s8_s32.m128n256` | 516.047 TOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.s8.m128n64` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.s8_s32.m128n64` | 342.184 TOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.tf32.m128n128` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.tf32_f32.m128n128` | 128.494 TFLOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.tf32.m128n256` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.tf32_f32.m128n256` | 114.711 TFLOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.tf32.m128n64` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.tf32_f32.m128n64` | 85.525 TFLOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.u8.m128n128` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.u8_s32.m128n128` | 513.988 TOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.u8.m128n256` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.u8_s32.m128n256` | 516.047 TOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.u8.m128n64` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.u8_s32.m128n64` | 342.175 TOP/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tma.hbm` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.tma_dram_stream_tc5a_ab_inflight8` | 185.509 GB/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |
| `tma.hbm.diagnostic.serial32k` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.tma_dram_stream_32k` | 261.556 GB/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |
| `tma.hbm.inflight4` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.tma_dram_stream_32k_inflight4` | 259.193 GB/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |
| `tma.smem_ingress.diagnostic.serial32k.per_sm` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.tma_l2_hit_32k` | 68.615 GB/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |
| `tma.smem_ingress.per_sm` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.tma_l2_hit_tc5a_ab_inflight8` | 193.366 GB/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |
| `tma.smem_ingress.per_sm.inflight4` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.tma_l2_hit_32k_inflight4` | 129.398 GB/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |
| `tmem.readback` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.tmem_ld_32x32b_x16_warps4` | 34270.415 GB/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |
| `tmem.readback.x16.warps1` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.tmem_ld_32x32b_x16_warps1` | 1343.479 GB/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |
| `tmem.readback.x8.warps1` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.tmem_ld_32x32b_x8_warps1` | 686.033 GB/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |
| `tmem.readback.x8.warps4` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.tmem_ld_32x32b_x8_warps4` | 19768.340 GB/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |
| `tmem.scale_ingress` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.tmem_scale_ingress_32x128b_warpx4` | 239.259 GB/s | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |

## Base/profile capacities

这些参数参与严格上界或 HBM/L2 经验场景，但没有因本次 closure 自动升级；其 `snapshot_only`/`profiler_model_peak` 等证据等级必须保留。

| Resource | Case | Rate | Evidence | Qualification | Source |
| --- | --- | ---: | --- | --- | --- |
| `hbm.read` | `hbm_read_stream_measured` | 198.467 GB/s | `measured_sustained` | `snapshot_only` | `microbench/05_gmem_dram_bandwidth/results/gmem_dram_bandwidth.csv` |
| `hbm.total` | `hbm_total_vendor_bandwidth` | 273.000 GB/s | `specified_upper` | `snapshot_only` | `Docs/blackwell_tensorcore/thor_sm110_gemm_stage_model.md` |
| `hbm.write` | `hbm_write_stream_measured` | 110.926 GB/s | `measured_sustained` | `snapshot_only` | `microbench/05_gmem_dram_bandwidth/results/gmem_dram_bandwidth.csv` |
| `l2.read` | `l2_read_ncu_model_peak` | 1612.800 GB/s | `profiler_model_peak` | `snapshot_only` | `microbench/L2throughtput/README.md` |
| `l2.read` | `l2_read_unique_measured` | 1491.054 GB/s | `measured_sustained` | `snapshot_only` | `microbench/L2throughtput/results/l2_throughput.csv` |
| `l2.write` | `l2_write_ncu_model_peak` | 806.400 GB/s | `profiler_model_peak` | `snapshot_only` | `microbench/L2throughtput/README.md` |
| `l2.write` | `l2_write_unique_measured` | 471.512 GB/s | `measured_sustained` | `snapshot_only` | `microbench/L2throughtput/results/l2_throughput.csv` |
| `tensor.bf16` | `bf16_vendor_peak_snapshot` | 258.500 TFLOP/s | `derived_upper` | `snapshot_only` | `microbench/mma_compute_only/plots/benchmark_results.csv` |
| `tensor.bf16.m128n256` | `bf16_compute_measured` | 258.030 TFLOP/s | `measured_sustained` | `snapshot_only` | `microbench/mma_compute_only/plots/benchmark_results.csv` |
| `tensor.e4m3` | `e4m3_vendor_peak_snapshot` | 517.000 TFLOP/s | `specified_upper` | `snapshot_only` | `microbench/mma_compute_only/plots/benchmark_results.csv` |
| `tensor.e4m3.m128n256` | `e4m3_compute_measured` | 516.059 TFLOP/s | `measured_sustained` | `snapshot_only` | `microbench/mma_compute_only/plots/benchmark_results.csv` |
| `tensor.e5m2` | `e5m2_vendor_peak_snapshot` | 517.000 TFLOP/s | `specified_upper` | `snapshot_only` | `microbench/mma_compute_only/plots/benchmark_results.csv` |
| `tensor.fp16` | `fp16_vendor_peak_snapshot` | 258.500 TFLOP/s | `derived_upper` | `snapshot_only` | `microbench/mma_compute_only/plots/benchmark_results.csv` |
| `tensor.nvfp4` | `nvfp4_vendor_peak_snapshot` | 1035.000 TFLOP/s | `specified_upper` | `snapshot_only` | `microbench/mma_compute_only/plots/benchmark_results.csv` |
| `tensor.nvfp4.m128n256` | `nvfp4_compute_measured` | 1032.111 TFLOP/s | `unknown` | `quarantined` | `microbench/mma_compute_only/plots/benchmark_results.csv` |
| `tensor.s8` | `s8_vendor_peak_conditional` | 517.500 TOP/s | `derived_upper` | `snapshot_only` | `Docs/blackwell_tensorcore/thor_sm110_gemm_performance_bounds.md` |
| `tensor.u8` | `u8_vendor_peak_conditional` | 517.500 TOP/s | `derived_upper` | `snapshot_only` | `Docs/blackwell_tensorcore/thor_sm110_gemm_performance_bounds.md` |
| `tma.hbm.diagnostic.serial32k` | `tma_hbm_stream_measured` | 245.352 GB/s | `measured_sustained` | `snapshot_only` | `microbench/07_tma_gmem_smem_bandwidth/results/tma_gmem_smem_bandwidth.csv` |
| `tma.smem_ingress.diagnostic.serial32k.per_sm` | `tma_l2_hit_measured` | 60.909 GB/s | `measured_sustained` | `snapshot_only` | `microbench/07_tma_gmem_smem_bandwidth/results/tma_gmem_smem_bandwidth.csv` |

## Full-GEMM 与模型

1024/2048 是预声明的 calibration，4096 是 holdout；该划分不证明 cache residency。报告同时计算 hot-L2 和 cold-HBM：严格上界采用两者中更松的 performance upper，经验包络保留两场景区间。
条件上界反证容差为 2.00%，经验重校准容差为 2.00%。

| Precision | N | Split | Candidate | Observed | Reference | Cand/ref | Upper status (L2/HBM) | Conditional upper range | Median/max upper | Max trial/max upper | Empirical range | Median/empirical range |
| --- | ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bf16_f32` | 1024 | calibration | `bf16_q0_wmma_m128n64k16` | 8.515 TFLOP/s | `cublas_bf16_gemmex` | 8.77% | `ok/ok` | 69.888 TFLOP/s–258.500 TFLOP/s | 3.29% | 3.30% | 69.888 TFLOP/s–128.436 TFLOP/s | 6.63%–12.18% |
| `bf16_f32` | 2048 | calibration | `bf16_q0_wmma_m128n64k16` | 9.175 TFLOP/s | `cublas_bf16_gemmex` | 7.05% | `ok/ok` | 139.776 TFLOP/s–258.500 TFLOP/s | 3.55% | 3.57% | 128.436 TFLOP/s–128.436 TFLOP/s | 7.14%–7.14% |
| `bf16_f32` | 4096 | holdout | `bf16_q0_wmma_m128n64k16` | 8.481 TFLOP/s | `cublas_bf16_gemmex` | 13.14% | `ok/ok` | 258.500 TFLOP/s–258.500 TFLOP/s | 3.28% | 3.40% | 128.436 TFLOP/s–128.436 TFLOP/s | 6.60%–6.60% |
| `e4m3_f32` | 1024 | calibration | `fp8_q7_mma_m16n8k32_smem128x64` | 5.314 TFLOP/s | `fp8_q8_cublaslt_matmul` | 3.96% | `ok/ok` | 93.184 TFLOP/s–412.877 TFLOP/s | 1.29% | 1.29% | 93.184 TFLOP/s–256.872 TFLOP/s | 2.07%–5.70% |
| `e4m3_f32` | 2048 | calibration | `fp8_q7_mma_m16n8k32_smem128x64` | 6.034 TFLOP/s | `fp8_q8_cublaslt_matmul` | 2.67% | `ok/ok` | 186.368 TFLOP/s–517.000 TFLOP/s | 1.17% | 1.17% | 186.368 TFLOP/s–256.872 TFLOP/s | 2.35%–3.24% |
| `e4m3_f32` | 4096 | holdout | `fp8_q7_mma_m16n8k32_smem128x64` | 6.196 TFLOP/s | `fp8_q8_cublaslt_matmul` | 2.92% | `ok/ok` | 372.736 TFLOP/s–517.000 TFLOP/s | 1.20% | 1.20% | 256.872 TFLOP/s–256.872 TFLOP/s | 2.41%–2.41% |
| `fp16_f32` | 1024 | calibration | `tc5b` | 90.752 TFLOP/s | `cublas_tc` | 88.35% | `ok/ok` | 69.888 TFLOP/s–258.500 TFLOP/s | 35.11% | 35.81% | 69.888 TFLOP/s–128.436 TFLOP/s | 70.66%–129.85% |
| `fp16_f32` | 2048 | calibration | `tc5a` | 120.039 TFLOP/s | `cublas_tc` | 91.89% | `ok/ok` | 139.776 TFLOP/s–258.500 TFLOP/s | 46.44% | 46.82% | 128.436 TFLOP/s–128.436 TFLOP/s | 93.46%–93.46% |
| `fp16_f32` | 4096 | holdout | `tc5a` | 62.868 TFLOP/s | `cublas_tc` | 97.88% | `ok/ok` | 258.500 TFLOP/s–258.500 TFLOP/s | 24.32% | 25.63% | 128.436 TFLOP/s–128.436 TFLOP/s | 48.95%–48.95% |
| `s8_s32` | 1024 | calibration | `int8_q15_wmma_m128n64k16_4warp_reuse_a_bcol` | 17.333 TOP/s | `int8_q19_cublas_gemmex` | 14.05% | `ok/ok` | 93.184 TOP/s–412.877 TOP/s | 4.20% | 4.29% | 93.184 TOP/s–256.872 TOP/s | 6.75%–18.60% |
| `s8_s32` | 2048 | calibration | `int8_q15_wmma_m128n64k16_4warp_reuse_a_bcol` | 19.566 TOP/s | `int8_q19_cublas_gemmex` | 9.54% | `ok/ok` | 186.368 TOP/s–517.500 TOP/s | 3.78% | 3.78% | 186.368 TOP/s–256.872 TOP/s | 7.62%–10.50% |
| `s8_s32` | 4096 | holdout | `int8_q15_wmma_m128n64k16_4warp_reuse_a_bcol` | 20.743 TOP/s | `int8_q19_cublas_gemmex` | 9.19% | `ok/ok` | 372.736 TOP/s–517.500 TOP/s | 4.01% | 4.01% | 256.872 TOP/s–256.872 TOP/s | 8.08%–8.08% |
| `tf32_f32` | 1024 | calibration | `tf32_q0_wmma_m64n64k8` | 2.606 TFLOP/s | `cublas_tf32_gemmex` | 6.30% | `partial/partial` | 46.592 TFLOP/s–412.877 TFLOP/s | 0.63% | 0.63% | 46.592 TFLOP/s–64.218 TFLOP/s | 4.06%–5.59% |
| `tf32_f32` | 2048 | calibration | `tf32_q0_wmma_m64n64k8` | 2.630 TFLOP/s | `cublas_tf32_gemmex` | 4.37% | `partial/partial` | 93.184 TFLOP/s–825.754 TFLOP/s | 0.32% | 0.32% | 64.218 TFLOP/s–64.218 TFLOP/s | 4.09%–4.09% |
| `tf32_f32` | 4096 | holdout | `tf32_q0_wmma_m64n64k8` | 2.337 TFLOP/s | `cublas_tf32_gemmex` | 12.61% | `partial/partial` | 186.368 TFLOP/s–1651.507 TFLOP/s | 0.14% | 0.14% | 64.218 TFLOP/s–64.218 TFLOP/s | 3.64%–3.64% |

## Findings

- **warning `overcurrent_events_observed`**：{"deltas": {"/sys/class/hwmon/hwmon5/oc1_event_cnt": 0, "/sys/class/hwmon/hwmon5/oc2_event_cnt": 0, "/sys/class/hwmon/hwmon5/oc3_event_cnt": 179}, "interval": "thor-t5000-closure-maxn-20260814-d382b57-a"}
