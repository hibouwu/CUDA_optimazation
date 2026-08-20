# Thor/SM110 GEMM closure 数值摘要

> **Historical replay notice（2026-08-18）**：本报告基于 `25d8cf7...`/`ba651f0...` 及迁移前独立 read/write empirical schema，不再代表 current f06 模型。当前重放边界见 [`gemm/appendices/current_model_replay.md`](./gemm/appendices/current_model_replay.md)。

本报告由当前模型离线重放已导入的 campaign 证据生成；报告生成动作本身不代表
重新运行了 GPU。是否属于 fresh acquisition，应以 campaign source、run ID、
commit、raw trial 和环境 artifact 为准。

- suite：`thor-t5000-tma-ingress-supplement-maxn-20260814-c`
- commit：`25d8cf71fa566150b64f2eb1dc7f814ce70fa354`
- composition：`base_compute_full_plus_component_supplement`
- campaign sources：`{"base": {"expected_commit": "d382b57eae289b458c5290e3d2b7e0daf1b7d7c8", "provides": ["epilogue_preflight", "compute", "full_gemm"], "suite_id": "thor-t5000-closure-maxn-20260814-d382b57-a"}, "component_supplement": {"expected_commit": "25d8cf71fa566150b64f2eb1dc7f814ce70fa354", "provides": ["component"], "suite_id": "thor-t5000-tma-ingress-supplement-maxn-20260814-c"}}`
- qualification：`closure_qualified`
- audit pass：`False`
- campaign measurement closed：`True`
- all precisions closed：`False`
- all common resources closed：`True`
- capacity：54 项
- base/profile capacity：19 项
- full-GEMM observation：15 项
- causal DAG solver implemented：`True`
- loaded pipeline profiles：0 项
- closure-qualified pipeline profiles：0 项
- resource/causal/integrated complete observations：2/0/0
- overcurrent delta：`{"thor-t5000-closure-maxn-20260814-d382b57-a": {"/sys/class/hwmon/hwmon5/oc1_event_cnt": 0, "/sys/class/hwmon/hwmon5/oc2_event_cnt": 0, "/sys/class/hwmon/hwmon5/oc3_event_cnt": 179}, "thor-t5000-tma-ingress-supplement-maxn-20260814-c": {"/sys/class/hwmon/hwmon5/oc1_event_cnt": 0, "/sys/class/hwmon/hwmon5/oc2_event_cnt": 0, "/sys/class/hwmon/hwmon5/oc3_event_cnt": 0}}`

## Closure-qualified compute/component capacities

定义每周期 rate 为 `rate_per_second / clock_hz`；这里的 `cycle` 是本报告绑定的 GPC 时钟周期（1.575 GHz）。该列只是单位归一化，不改变证据等级。`/GPU` 表示整卡聚合或共享资源，`/SM` 只用于 resource ID 明确标注 `.per_sm` 的独立每-SM 出口。

| Resource | Case | Median rate | Per cycle | Trials | Evidence | Qualification | Source |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `epilogue.nvfp4_requant` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.nvfp4_requant_4096x1024_constant` | 2.460 Gelement/s | 1.562 element/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |
| `epilogue.nvfp4_requant` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.nvfp4_requant_4096x1024_normal` | 2.460 Gelement/s | 1.562 element/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |
| `epilogue.nvfp4_requant` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.nvfp4_requant_4096x1024_outlier` | 2.460 Gelement/s | 1.562 element/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |
| `hbm.read` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.hbm_read_aggregate` | 253.588 GB/s | 161.008 B/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |
| `hbm.write` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.hbm_write_aggregate` | 201.158 GB/s | 127.719 B/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |
| `l2.read` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.l2_read_aggregate` | 1505.112 GB/s | 955.626 B/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |
| `l2.write` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.l2_write_aggregate` | 545.416 GB/s | 346.296 B/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |
| `tensor.bf16.m128n128` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.bf16_f32.m128n128` | 256.984 TFLOP/s | 163.164 kFLOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.bf16.m128n256` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.bf16_f32.m128n256` | 229.422 TFLOP/s | 145.665 kFLOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.bf16.m128n64` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.bf16_f32.m128n64` | 171.052 TFLOP/s | 108.604 kFLOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.e2m1.m128n128` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.e2m1_f32.m128n128` | 513.988 TFLOP/s | 326.341 kFLOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.e2m1.m128n256` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.e2m1_f32.m128n256` | 516.044 TFLOP/s | 327.647 kFLOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.e2m1.m128n64` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.e2m1_f32.m128n64` | 342.059 TFLOP/s | 217.181 kFLOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.e2m3.m128n128` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.e2m3_f32.m128n128` | 411.739 TFLOP/s | 261.422 kFLOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.e2m3.m128n256` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.e2m3_f32.m128n256` | 413.059 TFLOP/s | 262.260 kFLOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.e2m3.m128n64` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.e2m3_f32.m128n64` | 342.041 TFLOP/s | 217.169 kFLOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.e3m2.m128n128` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.e3m2_f32.m128n128` | 411.739 TFLOP/s | 261.422 kFLOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.e3m2.m128n256` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.e3m2_f32.m128n256` | 413.059 TFLOP/s | 262.260 kFLOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.e3m2.m128n64` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.e3m2_f32.m128n64` | 342.019 TFLOP/s | 217.155 kFLOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.e4m3.m128n128` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.e4m3_f32.m128n128` | 411.733 TFLOP/s | 261.418 kFLOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.e4m3.m128n256` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.e4m3_f32.m128n256` | 413.056 TFLOP/s | 262.258 kFLOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.e4m3.m128n64` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.e4m3_f32.m128n64` | 342.126 TFLOP/s | 217.223 kFLOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.e5m2.m128n128` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.e5m2_f32.m128n128` | 464.766 TFLOP/s | 295.090 kFLOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.e5m2.m128n256` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.e5m2_f32.m128n256` | 458.843 TFLOP/s | 291.329 kFLOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.e5m2.m128n64` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.e5m2_f32.m128n64` | 342.068 TFLOP/s | 217.186 kFLOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.fp16.m128n128` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.fp16_f32.m128n128` | 205.871 TFLOP/s | 130.712 kFLOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.fp16.m128n256` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.fp16_f32.m128n256` | 206.530 TFLOP/s | 131.130 kFLOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.fp16.m128n64` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.fp16_f32.m128n64` | 171.030 TFLOP/s | 108.590 kFLOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.mxfp4.m128n128` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.mxfp4_f32.m128n128` | 1027.975 TFLOP/s | 652.683 kFLOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.mxfp4.m128n256` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.mxfp4_f32.m128n256` | 1032.093 TFLOP/s | 655.297 kFLOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.mxfp4.m128n64` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.mxfp4_f32.m128n64` | 684.217 TFLOP/s | 434.423 kFLOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.nvfp4.m128n128` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.nvfp4_f32.m128n128` | 1027.975 TFLOP/s | 652.683 kFLOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.nvfp4.m128n256` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.nvfp4_f32.m128n256` | 1032.093 TFLOP/s | 655.297 kFLOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.nvfp4.m128n64` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.nvfp4_f32.m128n64` | 684.342 TFLOP/s | 434.503 kFLOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.s8.m128n128` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.s8_s32.m128n128` | 513.983 TOP/s | 326.338 kOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.s8.m128n256` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.s8_s32.m128n256` | 516.047 TOP/s | 327.649 kOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.s8.m128n64` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.s8_s32.m128n64` | 342.184 TOP/s | 217.260 kOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.tf32.m128n128` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.tf32_f32.m128n128` | 128.494 TFLOP/s | 81.584 kFLOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.tf32.m128n256` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.tf32_f32.m128n256` | 114.711 TFLOP/s | 72.833 kFLOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.tf32.m128n64` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.tf32_f32.m128n64` | 85.525 TFLOP/s | 54.302 kFLOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.u8.m128n128` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.u8_s32.m128n128` | 513.988 TOP/s | 326.341 kOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.u8.m128n256` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.u8_s32.m128n256` | 516.047 TOP/s | 327.649 kOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tensor.u8.m128n64` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.compute.u8_s32.m128n64` | 342.175 TOP/s | 217.254 kOP/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_campaign/thor-t5000-closure-maxn-20260814-d382b57-a-compute/summary.json` |
| `tma.hbm` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.tma_dram_stream_tc5a_ab_inflight8` | 185.509 GB/s | 117.784 B/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |
| `tma.hbm.diagnostic.serial32k` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.tma_dram_stream_32k` | 261.556 GB/s | 166.067 B/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |
| `tma.hbm.inflight4` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.tma_dram_stream_32k_inflight4` | 259.193 GB/s | 164.567 B/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |
| `tma.smem_ingress.diagnostic.serial32k.per_sm` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.tma_l2_hit_32k` | 68.615 GB/s | 43.565 B/cycle/SM | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |
| `tma.smem_ingress.per_sm` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.tma_l2_hit_tc5a_ab_inflight8` | 193.366 GB/s | 122.772 B/cycle/SM | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |
| `tma.smem_ingress.per_sm.inflight4` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.tma_l2_hit_32k_inflight4` | 129.398 GB/s | 82.157 B/cycle/SM | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |
| `tmem.readback` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.tmem_ld_32x32b_x16_warps4` | 34270.415 GB/s | 21,758.994 B/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |
| `tmem.readback.x16.warps1` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.tmem_ld_32x32b_x16_warps1` | 1343.479 GB/s | 853.003 B/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |
| `tmem.readback.x8.warps1` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.tmem_ld_32x32b_x8_warps1` | 686.033 GB/s | 435.576 B/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |
| `tmem.readback.x8.warps4` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.tmem_ld_32x32b_x8_warps4` | 19768.340 GB/s | 12,551.327 B/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |
| `tmem.scale_ingress` | `thor-t5000-tma-ingress-supplement-maxn-20260814-c.component.tmem_scale_ingress_32x128b_warpx4` | 239.259 GB/s | 151.910 B/cycle/GPU | 10 | `measured_sustained` | `closure_qualified` | `results/sm110_gemm_component_campaign/thor-t5000-tma-ingress-supplement-maxn-20260814-c-components/summary.json` |

## Base/profile capacities

这些参数参与严格上界或 HBM/L2 经验场景，但没有因本次 closure 自动升级；其 `snapshot_only`/`profiler_model_peak` 等证据等级必须保留。

| Resource | Case | Rate | Per cycle | Evidence | Qualification | Source | Artifacts |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| `hbm.read` | `hbm_read_stream_measured` | 198.467 GB/s | 126.011 B/cycle/GPU | `measured_sustained` | `snapshot_only` | `microbench/05_gmem_dram_bandwidth/results/gmem_dram_bandwidth.csv` | — |
| `hbm.total` | `hbm_total_vendor_bandwidth` | 273.000 GB/s | 173.333 B/cycle/GPU | `specified_upper` | `snapshot_only` | `Docs/blackwell_tensorcore/thor_sm110_gemm_stage_model.md` | — |
| `hbm.write` | `hbm_write_stream_measured` | 110.926 GB/s | 70.429 B/cycle/GPU | `measured_sustained` | `snapshot_only` | `microbench/05_gmem_dram_bandwidth/results/gmem_dram_bandwidth.csv` | — |
| `l2.read` | `l2_read_ncu_model_peak` | 1612.800 GB/s | 1,024.000 B/cycle/GPU | `profiler_model_peak` | `snapshot_only` | `microbench/L2throughtput/README.md` | — |
| `l2.read` | `l2_read_unique_measured` | 1491.054 GB/s | 946.701 B/cycle/GPU | `measured_sustained` | `snapshot_only` | `microbench/L2throughtput/results/l2_throughput.csv` | `microbench/L2throughtput/plots/l2_capacity_staircase.svg`<br>`microbench/L2throughtput/plots/l2_concurrency_saturation.svg` |
| `l2.write` | `l2_write_ncu_model_peak` | 806.400 GB/s | 512.000 B/cycle/GPU | `profiler_model_peak` | `snapshot_only` | `microbench/L2throughtput/README.md` | — |
| `l2.write` | `l2_write_unique_measured` | 471.512 GB/s | 299.373 B/cycle/GPU | `measured_sustained` | `snapshot_only` | `microbench/L2throughtput/results/l2_throughput.csv` | `microbench/L2throughtput/plots/l2_capacity_staircase.svg`<br>`microbench/L2throughtput/plots/l2_concurrency_saturation.svg` |
| `tensor.bf16` | `bf16_vendor_peak_snapshot` | 258.500 TFLOP/s | 164.127 kFLOP/cycle/GPU | `derived_upper` | `snapshot_only` | `microbench/mma_compute_only/plots/benchmark_results.csv` | — |
| `tensor.bf16.m128n256` | `bf16_compute_measured` | 258.030 TFLOP/s | 163.829 kFLOP/cycle/GPU | `measured_sustained` | `snapshot_only` | `microbench/mma_compute_only/plots/benchmark_results.csv` | — |
| `tensor.e4m3` | `e4m3_vendor_peak_snapshot` | 517.000 TFLOP/s | 328.254 kFLOP/cycle/GPU | `specified_upper` | `snapshot_only` | `microbench/mma_compute_only/plots/benchmark_results.csv` | — |
| `tensor.e4m3.m128n256` | `e4m3_compute_measured` | 516.059 TFLOP/s | 327.657 kFLOP/cycle/GPU | `measured_sustained` | `snapshot_only` | `microbench/mma_compute_only/plots/benchmark_results.csv` | — |
| `tensor.e5m2` | `e5m2_vendor_peak_snapshot` | 517.000 TFLOP/s | 328.254 kFLOP/cycle/GPU | `specified_upper` | `snapshot_only` | `microbench/mma_compute_only/plots/benchmark_results.csv` | — |
| `tensor.fp16` | `fp16_vendor_peak_snapshot` | 258.500 TFLOP/s | 164.127 kFLOP/cycle/GPU | `derived_upper` | `snapshot_only` | `microbench/mma_compute_only/plots/benchmark_results.csv` | — |
| `tensor.nvfp4` | `nvfp4_vendor_peak_snapshot` | 1035.000 TFLOP/s | 657.143 kFLOP/cycle/GPU | `specified_upper` | `snapshot_only` | `microbench/mma_compute_only/plots/benchmark_results.csv` | — |
| `tensor.nvfp4.m128n256` | `nvfp4_compute_measured` | 1032.111 TFLOP/s | 655.309 kFLOP/cycle/GPU | `unknown` | `quarantined` | `microbench/mma_compute_only/plots/benchmark_results.csv` | — |
| `tensor.s8` | `s8_vendor_peak_conditional` | 517.500 TOP/s | 328.571 kOP/cycle/GPU | `derived_upper` | `snapshot_only` | `Docs/blackwell_tensorcore/thor_sm110_gemm_performance_bounds.md` | — |
| `tensor.u8` | `u8_vendor_peak_conditional` | 517.500 TOP/s | 328.571 kOP/cycle/GPU | `derived_upper` | `snapshot_only` | `Docs/blackwell_tensorcore/thor_sm110_gemm_performance_bounds.md` | — |
| `tma.hbm.diagnostic.serial32k` | `tma_hbm_stream_measured` | 245.352 GB/s | 155.779 B/cycle/GPU | `measured_sustained` | `snapshot_only` | `microbench/07_tma_gmem_smem_bandwidth/results/tma_gmem_smem_bandwidth.csv` | — |
| `tma.smem_ingress.diagnostic.serial32k.per_sm` | `tma_l2_hit_measured` | 60.909 GB/s | 38.672 B/cycle/SM | `measured_sustained` | `snapshot_only` | `microbench/07_tma_gmem_smem_bandwidth/results/tma_gmem_smem_bandwidth.csv` | — |

### 相关 microbenchmark diagnostics（未导入 current capacity selector）

下表记录对未来 GEMM schedule 有用、但合同不等于当前 SS/TMA/CTA-group-1
manifest 的实验。它们不能与上表 capacity 混选；否则会把 TS、DSMEM 或不完整
pipeline 的内点错误用于当前 GEMM。

| 实验 | Snapshot | 可能关闭的未来参数 | 当前证据边界 | Source |
| --- | ---: | --- | --- | --- |
| L2 baseline + capacity/concurrency sweep | 16 MiB baseline 代表点：`read-unique=946.701`、`write-unique=299.373 B/cycle/GPU`；validation SVG 显示约 32 MiB 阶跃 | `hot_l2` capacity 与 saturation | 两个 baseline rate 已在上表；SVG 中位曲线只作 artifact，不从像素反推或重复导入 | [`l2_capacity_staircase.svg`](../../microbench/L2throughtput/plots/l2_capacity_staircase.svg)、[`l2_concurrency_saturation.svg`](../../microbench/L2throughtput/plots/l2_concurrency_saturation.svg) |
| generic `tcgen05.cp` | 859.024 B/cycle/GPU，2.384 cycle/cp | future TS schedule 的 SMEM→TMEM operand ingress | 不适用于当前 SS/TMA schedule，也不替代 512-B scale `warpx4` atom | [`cp_only_results.csv`](../../microbench/mma_with_cp/plots/cp_only_results.csv) |
| TS MMA TMEM consume | `ts-mma-only=115.699 B/cycle/GPU`；CP+MMA A2=`103.011 B/cycle/GPU` | future TS schedule 的 TMEM operand consume | 2048 B/MMA 是估算分子，不是 raw TMEM port upper | [`08_tmem_consume_bandwidth`](../../microbench/08_tmem_consume_bandwidth/README.md) |
| CP/MMA overlap | FP4 M128N256：serial 214.210、A2 overlap 331.938 TFLOP/s，1.55x | future TS causal/joint profile | 输入已在 SMEM/TMEM；不包含 GMEM/TMA、完整 epilogue 或 exact GEMM residency | [`pipeline_results.csv`](../../microbench/mma_with_cp/plots/pipeline_results.csv) |
| DSMEM topology/contention | ring 113.15–141.87、fan-in 52.76–72.09 B/cycle/GPU | future CTA-group-2 cluster/DSMEM schedule | current v1 拒绝 `cta_group=2`；不是 physical interconnect upper | [`09_dsmem_topology_contention`](../../microbench/09_dsmem_topology_contention/README.md) |

L1 global path、local SMEM bank/stride 和 DSMEM baseline runner 也可能服务未来
non-TMA 或 cluster schedule；当前 checkout 没有与本报告同等级的 committed raw
summary，因此不从 README 推测或抄录额外 capacity。

## Full-GEMM 与模型

1024/2048 是预声明的 calibration，4096 是 holdout；该划分不证明 cache residency。报告同时计算 hot-L2 和 cold-HBM：严格上界采用两者中更松的 performance upper；resource envelope、causal profile 和二者合并后的最终经验理想包络分别报告，不能互相顶替。
条件上界反证容差为 2.00%，经验重校准容差为 2.00%。

| Precision | N | Split | Candidate | Candidate median | Reference | Reference median | Observed-best backend | Cand/ref | Upper status (L2/HBM) | Conditional upper range | Candidate median/max upper | Observed-best max trial/max upper | Resource status | Resource range | Causal status | Causal range | Integrated ideal range | Candidate/integrated | Observed-best/integrated |
| --- | ---: | --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bf16_f32` | 1024 | calibration | `bf16_q0_wmma_m128n64k16` | 8.515 TFLOP/s | `cublas_bf16_gemmex` | 97.118 TFLOP/s | `cublas_bf16_gemmex` | 8.77% | `ok/ok` | 69.888 TFLOP/s–258.500 TFLOP/s | 3.29% | 37.94% | `insufficient_evidence/insufficient_evidence` | —–— | `insufficient_evidence/insufficient_evidence` | —–— | —–— | —–— | —–— |
| `bf16_f32` | 2048 | calibration | `bf16_q0_wmma_m128n64k16` | 9.175 TFLOP/s | `cublas_bf16_gemmex` | 130.215 TFLOP/s | `cublas_bf16_gemmex` | 7.05% | `ok/ok` | 139.776 TFLOP/s–258.500 TFLOP/s | 3.55% | 50.63% | `ok/ok` | 128.436 TFLOP/s–128.436 TFLOP/s | `insufficient_evidence/insufficient_evidence` | —–— | —–— | —–— | —–— |
| `bf16_f32` | 4096 | holdout | `bf16_q0_wmma_m128n64k16` | 8.481 TFLOP/s | `cublas_bf16_gemmex` | 64.559 TFLOP/s | `cublas_bf16_gemmex` | 13.14% | `ok/ok` | 258.500 TFLOP/s–258.500 TFLOP/s | 3.28% | 25.63% | `insufficient_evidence/insufficient_evidence` | —–— | `insufficient_evidence/insufficient_evidence` | —–— | —–— | —–— | —–— |
| `e4m3_f32` | 1024 | calibration | `fp8_q7_mma_m16n8k32_smem128x64` | 5.314 TFLOP/s | `fp8_q8_cublaslt_matmul` | 134.204 TFLOP/s | `fp8_q8_cublaslt_matmul` | 3.96% | `ok/ok` | 93.184 TFLOP/s–412.877 TFLOP/s | 1.29% | 32.88% | `insufficient_evidence/insufficient_evidence` | —–— | `insufficient_evidence/insufficient_evidence` | —–— | —–— | —–— | —–— |
| `e4m3_f32` | 2048 | calibration | `fp8_q7_mma_m16n8k32_smem128x64` | 6.034 TFLOP/s | `fp8_q8_cublaslt_matmul` | 226.013 TFLOP/s | `fp8_q8_cublaslt_matmul` | 2.67% | `ok/ok` | 186.368 TFLOP/s–517.000 TFLOP/s | 1.17% | 44.66% | `insufficient_evidence/insufficient_evidence` | —–— | `insufficient_evidence/insufficient_evidence` | —–— | —–— | —–— | —–— |
| `e4m3_f32` | 4096 | holdout | `fp8_q7_mma_m16n8k32_smem128x64` | 6.196 TFLOP/s | `fp8_q8_cublaslt_matmul` | 211.940 TFLOP/s | `fp8_q8_cublaslt_matmul` | 2.92% | `ok/ok` | 372.736 TFLOP/s–517.000 TFLOP/s | 1.20% | 41.97% | `insufficient_evidence/insufficient_evidence` | —–— | `insufficient_evidence/insufficient_evidence` | —–— | —–— | —–— | —–— |
| `fp16_f32` | 1024 | calibration | `tc5b` | 90.752 TFLOP/s | `cublas_tc` | 102.721 TFLOP/s | `cublas_tc` | 88.35% | `ok/ok` | 69.888 TFLOP/s–258.500 TFLOP/s | 35.11% | 39.88% | `insufficient_evidence/insufficient_evidence` | —–— | `insufficient_evidence/insufficient_evidence` | —–— | —–— | —–— | —–— |
| `fp16_f32` | 2048 | calibration | `tc5a` | 120.039 TFLOP/s | `cublas_tc` | 130.633 TFLOP/s | `cublas_tc` | 91.89% | `ok/ok` | 139.776 TFLOP/s–258.500 TFLOP/s | 46.44% | 50.74% | `ok/ok` | 128.436 TFLOP/s–128.436 TFLOP/s | `insufficient_evidence/insufficient_evidence` | —–— | —–— | —–— | —–— |
| `fp16_f32` | 4096 | holdout | `tc5a` | 62.868 TFLOP/s | `cublas_tc` | 64.231 TFLOP/s | `cublas_tc` | 97.88% | `ok/ok` | 258.500 TFLOP/s–258.500 TFLOP/s | 24.32% | 25.63% | `insufficient_evidence/insufficient_evidence` | —–— | `insufficient_evidence/insufficient_evidence` | —–— | —–— | —–— | —–— |
| `s8_s32` | 1024 | calibration | `int8_q15_wmma_m128n64k16_4warp_reuse_a_bcol` | 17.333 TOP/s | `int8_q19_cublas_gemmex` | 123.408 TOP/s | `int8_q19_cublas_gemmex` | 14.05% | `ok/ok` | 93.184 TOP/s–412.877 TOP/s | 4.20% | 30.32% | `insufficient_evidence/insufficient_evidence` | —–— | `insufficient_evidence/insufficient_evidence` | —–— | —–— | —–— | —–— |
| `s8_s32` | 2048 | calibration | `int8_q15_wmma_m128n64k16_4warp_reuse_a_bcol` | 19.566 TOP/s | `int8_q19_cublas_gemmex` | 205.030 TOP/s | `int8_q19_cublas_gemmex` | 9.54% | `ok/ok` | 186.368 TOP/s–517.500 TOP/s | 3.78% | 41.56% | `insufficient_evidence/insufficient_evidence` | —–— | `insufficient_evidence/insufficient_evidence` | —–— | —–— | —–— | —–— |
| `s8_s32` | 4096 | holdout | `int8_q15_wmma_m128n64k16_4warp_reuse_a_bcol` | 20.743 TOP/s | `int8_q19_cublas_gemmex` | 225.688 TOP/s | `int8_q19_cublas_gemmex` | 9.19% | `ok/ok` | 372.736 TOP/s–517.500 TOP/s | 4.01% | 44.42% | `insufficient_evidence/insufficient_evidence` | —–— | `insufficient_evidence/insufficient_evidence` | —–— | —–— | —–— | —–— |
| `tf32_f32` | 1024 | calibration | `tf32_q0_wmma_m64n64k8` | 2.606 TFLOP/s | `cublas_tf32_gemmex` | 41.347 TFLOP/s | `cublas_tf32_gemmex` | 6.30% | `partial/partial` | 46.592 TFLOP/s–412.877 TFLOP/s | 0.63% | 10.06% | `insufficient_evidence/insufficient_evidence` | —–— | `insufficient_evidence/insufficient_evidence` | —–— | —–— | —–— | —–— |
| `tf32_f32` | 2048 | calibration | `tf32_q0_wmma_m64n64k8` | 2.630 TFLOP/s | `cublas_tf32_gemmex` | 60.197 TFLOP/s | `cublas_tf32_gemmex` | 4.37% | `partial/partial` | 93.184 TFLOP/s–825.754 TFLOP/s | 0.32% | 7.50% | `insufficient_evidence/insufficient_evidence` | —–— | `insufficient_evidence/insufficient_evidence` | —–— | —–— | —–— | —–— |
| `tf32_f32` | 4096 | holdout | `tf32_q0_wmma_m64n64k8` | 2.337 TFLOP/s | `cublas_tf32_gemmex` | 18.541 TFLOP/s | `cublas_tf32_gemmex` | 12.61% | `partial/partial` | 186.368 TFLOP/s–1651.507 TFLOP/s | 0.14% | 1.13% | `insufficient_evidence/insufficient_evidence` | —–— | `insufficient_evidence/insufficient_evidence` | —–— | —–— | —–— | —–— |

## Findings

- **warning `overcurrent_events_observed`**：{"deltas": {"/sys/class/hwmon/hwmon5/oc1_event_cnt": 0, "/sys/class/hwmon/hwmon5/oc2_event_cnt": 0, "/sys/class/hwmon/hwmon5/oc3_event_cnt": 179}, "interval": "thor-t5000-closure-maxn-20260814-d382b57-a"}
- **error `residency_empirical_resource_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.bf16_f32_n1024_q0: missing resource-layer scenario prediction
- **error `residency_causal_pipeline_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.bf16_f32_n1024_q0: missing causal-profile scenario prediction
- **error `residency_empirical_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.bf16_f32_n1024_q0: missing integrated resource-plus-causal scenario prediction
- **error `residency_causal_pipeline_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.bf16_f32_n2048_q0: missing causal-profile scenario prediction
- **error `residency_empirical_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.bf16_f32_n2048_q0: missing integrated resource-plus-causal scenario prediction
- **error `residency_empirical_resource_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.bf16_f32_n4096_q0: missing resource-layer scenario prediction
- **error `residency_causal_pipeline_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.bf16_f32_n4096_q0: missing causal-profile scenario prediction
- **error `residency_empirical_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.bf16_f32_n4096_q0: missing integrated resource-plus-causal scenario prediction
- **error `residency_empirical_resource_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.e4m3_f32_n1024_q7: missing resource-layer scenario prediction
- **error `residency_causal_pipeline_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.e4m3_f32_n1024_q7: missing causal-profile scenario prediction
- **error `residency_empirical_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.e4m3_f32_n1024_q7: missing integrated resource-plus-causal scenario prediction
- **error `residency_empirical_resource_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.e4m3_f32_n2048_q7: missing resource-layer scenario prediction
- **error `residency_causal_pipeline_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.e4m3_f32_n2048_q7: missing causal-profile scenario prediction
- **error `residency_empirical_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.e4m3_f32_n2048_q7: missing integrated resource-plus-causal scenario prediction
- **error `residency_empirical_resource_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.e4m3_f32_n4096_q7: missing resource-layer scenario prediction
- **error `residency_causal_pipeline_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.e4m3_f32_n4096_q7: missing causal-profile scenario prediction
- **error `residency_empirical_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.e4m3_f32_n4096_q7: missing integrated resource-plus-causal scenario prediction
- **error `residency_empirical_resource_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.fp16_f32_n1024_tc5b: missing resource-layer scenario prediction
- **error `residency_causal_pipeline_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.fp16_f32_n1024_tc5b: missing causal-profile scenario prediction
- **error `residency_empirical_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.fp16_f32_n1024_tc5b: missing integrated resource-plus-causal scenario prediction
- **error `residency_causal_pipeline_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.fp16_f32_n2048_tc5a: missing causal-profile scenario prediction
- **error `residency_empirical_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.fp16_f32_n2048_tc5a: missing integrated resource-plus-causal scenario prediction
- **error `residency_empirical_resource_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.fp16_f32_n4096_tc5a: missing resource-layer scenario prediction
- **error `residency_causal_pipeline_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.fp16_f32_n4096_tc5a: missing causal-profile scenario prediction
- **error `residency_empirical_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.fp16_f32_n4096_tc5a: missing integrated resource-plus-causal scenario prediction
- **error `residency_empirical_resource_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.s8_s32_n1024_q15: missing resource-layer scenario prediction
- **error `residency_causal_pipeline_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.s8_s32_n1024_q15: missing causal-profile scenario prediction
- **error `residency_empirical_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.s8_s32_n1024_q15: missing integrated resource-plus-causal scenario prediction
- **error `residency_empirical_resource_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.s8_s32_n2048_q15: missing resource-layer scenario prediction
- **error `residency_causal_pipeline_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.s8_s32_n2048_q15: missing causal-profile scenario prediction
- **error `residency_empirical_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.s8_s32_n2048_q15: missing integrated resource-plus-causal scenario prediction
- **error `residency_empirical_resource_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.s8_s32_n4096_q15: missing resource-layer scenario prediction
- **error `residency_causal_pipeline_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.s8_s32_n4096_q15: missing causal-profile scenario prediction
- **error `residency_empirical_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.s8_s32_n4096_q15: missing integrated resource-plus-causal scenario prediction
- **error `residency_empirical_resource_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.tf32_f32_n1024_q0: missing resource-layer scenario prediction
- **error `residency_causal_pipeline_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.tf32_f32_n1024_q0: missing causal-profile scenario prediction
- **error `residency_empirical_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.tf32_f32_n1024_q0: missing integrated resource-plus-causal scenario prediction
- **error `residency_empirical_resource_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.tf32_f32_n2048_q0: missing resource-layer scenario prediction
- **error `residency_causal_pipeline_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.tf32_f32_n2048_q0: missing causal-profile scenario prediction
- **error `residency_empirical_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.tf32_f32_n2048_q0: missing integrated resource-plus-causal scenario prediction
- **error `residency_empirical_resource_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.tf32_f32_n4096_q0: missing resource-layer scenario prediction
- **error `residency_causal_pipeline_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.tf32_f32_n4096_q0: missing causal-profile scenario prediction
- **error `residency_empirical_prediction_incomplete`**：thor-t5000-tma-ingress-supplement-maxn-20260814-c.full.tf32_f32_n4096_q0: missing integrated resource-plus-causal scenario prediction
