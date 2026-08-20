# Microbenchmark 与完整 GEMM 来源

本附录按 experiment ID 汇总模型参数、源码、runner、auditor 和当前结果边界。它不复制 raw trial；结果必须通过 commit/run ID 追踪。

## 1. 结果提交

| evidence | branch / commit | 说明 |
| --- | --- | --- |
| 2026-08-14 composite closure | `ba651f0ebddd0983ceca5b352e65aa7ed5b7f32c` | compute/component/full-GEMM 历史证据；旧 applicability/independent-resource envelope |
| 2026-08-17 parameter supplement | `thor-results/thor-t5000-parameter-plots-maxn-20260817-i` / `aa845dd9e70e2c541ae3a7d5293bf8de4bd55092` | 10 TMA payload + 21 memory duplex cases |
| parameter suite-log follow-up | `78e09488c51b3d81ac2ec9596630f238af11ad91` | completion/audit log follow-up，不改变 GPU trial |

## 2. 参数到实验

| 参数或证据 | experiment | source | runner / auditor | 当前边界 |
| --- | --- | --- | --- | --- |
| shape-qualified compute rate | [EXP-01](../experiments/EXP-01-compute-surface.md) | [run_compute_campaign.py](../../../../microbench/sm110_gemm_campaign/run_compute_campaign.py) 生成 CUDA | [compute auditor](../../../../microbench/sm110_gemm_campaign/audit_campaign.py) | 12 precision / 36 full-SM points；measured，不是 upper |
| L2 1024/512 B/cycle 与 capacity | [EXP-02](../experiments/EXP-02-l2-physical-bounds.md) | [L2 README](../../../../microbench/L2throughtput/README.md)、[memory_path_bandwidth.cu](../../../../microbench/14_memory_path_bandwidth/memory_path_bandwidth.cu) | component runner/auditor | GPU-wide strict conditional upper；无 joint outer proof |
| TMA payload surface | [EXP-03](../experiments/EXP-03-tma-payload-surface.md) | [tma_gmem_smem_bandwidth.cu](../../../../microbench/07_tma_gmem_smem_bandwidth/tma_gmem_smem_bandwidth.cu) | [runner](../../../../microbench/sm110_tma_payload_campaign/run_tma_payload_campaign.py) / [auditor](../../../../microbench/sm110_tma_payload_campaign/audit_campaign.py) | 4/8/16/32/64 KiB × hot/cold；缺 512 B/1 KiB scale request |
| L2 duplex / cold proxy | [EXP-04](../experiments/EXP-04-memory-duplex-surface.md) | [memory_path_bandwidth.cu](../../../../microbench/14_memory_path_bandwidth/memory_path_bandwidth.cu) | [runner](../../../../microbench/sm110_memory_duplex_campaign/run_memory_duplex_campaign.py) / [auditor](../../../../microbench/sm110_memory_duplex_campaign/audit_campaign.py) | 14 `l2.duplex` + 7 `hbm.duplex.proxy`；physical HBM duplex 为 0 |
| exact TMA topology | [EXP-05](../experiments/EXP-05-exact-tma-topology.md) | [tma_ab_contract_bandwidth.cu](../../../../microbench/15_tma_ab_contract_bandwidth/tma_ab_contract_bandwidth.cu) | [runner](../../../../microbench/sm110_gemm_resource_campaign/run_resource_campaign.py) / [auditor](../../../../microbench/sm110_gemm_resource_campaign/audit_campaign.py) | 54-case static contract；Thor exact coverage 当前 2/28 pair |
| TMEM readback | [EXP-06](../experiments/EXP-06-tmem-readback-and-scale.md) | [tmem_readback_bandwidth.cu](../../../../microbench/12_tmem_readback_bandwidth/tmem_readback_bandwidth.cu) | component runner/auditor | x8/x16 × 1/4 warps 已测 |
| TMEM scale ingress | [EXP-06](../experiments/EXP-06-tmem-readback-and-scale.md) | [tmem_scale_ingress_bandwidth.cu](../../../../microbench/13_tmem_scale_ingress_bandwidth/tmem_scale_ingress_bandwidth.cu) | component runner/auditor | 512-B source atom service 已测 |
| NVFP4 epilogue diagnostic | [EXP-06](../experiments/EXP-06-tmem-readback-and-scale.md) | [requant_epilogue_benchmark.cu](../../../../GEMMsm110/tests/requant_epilogue_benchmark.cu) | component runner/auditor | distribution diagnostics；不推广到其它 output semantic |
| causal profile | [EXP-07](../experiments/EXP-07-causal-pipeline.md) | [tc5a_pipeline_dag.cu](../../../../microbench/16_tc5a_pipeline_dag/tc5a_pipeline_dag.cu) | [runner](../../../../microbench/sm110_gemm_causal_campaign/run_causal_campaign.py) / [auditor](../../../../microbench/sm110_gemm_causal_campaign/audit_campaign.py) | FP16/BF16 182-case contract 已冻结；Thor timing 未回传 |
| full-GEMM observation | [EXP-08](../experiments/EXP-08-full-gemm-validation.md) | [main.cu](../../../../GEMMsm110/src/main.cu)、[extended](../../../../GEMMquant_sm110/src/extended_gemm_bench.cu)、[quant](../../../../GEMMquant_sm110/src/quant_gemm_bench.cu) | [runner](../../../../microbench/sm110_full_gemm_campaign/run_full_gemm_campaign.py) / [auditor](../../../../microbench/sm110_full_gemm_campaign/audit_campaign.py) | 历史 Thor 5 precision；current runner 6 precision |

## 3. 相关诊断实验：有用，但不作为 current capacity

下表只收录有已提交数值或可定位绘图工件、并且可能扩展未来 GEMM schedule
模型的实验。它们没有满足 current schedule 的 exact applicability 时，不进入
capacity selector。

| 实验 | 已提交 snapshot | 对模型的潜在用途 | 当前处理 | 来源 |
| --- | ---: | --- | --- | --- |
| L2 baseline + capacity/concurrency sweep | 16 MiB baseline 代表点：`read-unique=946.701`、`write-unique=299.373 B/cycle/GPU`；validation SVG 显示约 32 MiB 阶跃 | `hot_l2` capacity 与 saturation 佐证 | 两个 baseline rate 已在 base capacity；SVG 中位曲线只作 artifact，不从像素反推或重复导入 | [capacity staircase](../../../../microbench/L2throughtput/plots/l2_capacity_staircase.svg)、[concurrency](../../../../microbench/L2throughtput/plots/l2_concurrency_saturation.svg) |
| generic `tcgen05.cp` | 859.024 B/cycle/GPU，2.384 cycle/cp | future TS schedule 的 SMEM→TMEM operand ingress | 不用于当前 SS/TMA schedule；也不替代 512-B scale `warpx4` atom | [cp_only_results.csv](../../../../microbench/mma_with_cp/plots/cp_only_results.csv)、[06 README](../../../../microbench/06_tmem_cp_bandwidth/README.md) |
| TS MMA TMEM consume | `ts-mma-only=115.699 B/cycle/GPU`；CP+MMA A2=`103.011 B/cycle/GPU` | future TS schedule 的 TMEM operand-consume demand | 2048 B/MMA 是估算分子，不是 raw TMEM port upper | [08 README](../../../../microbench/08_tmem_consume_bandwidth/README.md)、[mma_only_results.csv](../../../../microbench/mma_with_cp/plots/mma_only_results.csv) |
| CP/MMA overlap | FP4 M128N256：serial 214.210、A2 overlap 331.938 TFLOP/s，1.55x | future TS causal/joint profile | 输入已在 SMEM/TMEM；没有 GMEM/TMA、完整 epilogue 或 exact GEMM residency，不能导入 current causal profile | [pipeline_results.csv](../../../../microbench/mma_with_cp/plots/pipeline_results.csv)、[11 README](../../../../microbench/11_pipeline_overlap/README.md) |
| DSMEM topology/contention | ring 113.15–141.87、fan-in 52.76–72.09 B/cycle/GPU | future CTA-group-2 cluster/DSMEM schedule | current v1 拒绝 `cta_group=2`；不是物理 interconnect upper | [09 README](../../../../microbench/09_dsmem_topology_contention/README.md) |

L1 global path、local SMEM bank/stride 和 DSMEM baseline runner 也可能服务未来
non-TMA 或 cluster schedule；当前 checkout 没有与本报告同等级的 committed raw
summary，因此只保留 source index，不抄 README 之外的推测数字进入 capacity 表。

## 4. 模型导入器

| importer | 责任 |
| --- | --- |
| [evidence_import.py](../../../../scripts/sm110_gemm_model/evidence_import.py) | compute/component/payload/duplex campaign 重审计与 scoped capacity |
| [resource_import.py](../../../../scripts/sm110_gemm_model/resource_import.py) | 54-case exact TMA resource import |
| [causal_import.py](../../../../scripts/sm110_gemm_model/causal_import.py) | FP16/BF16 causal profile import |
| [observations.py](../../../../scripts/sm110_gemm_model/observations.py) | full-GEMM observation import与 suite qualification |
| [suite.py](../../../../scripts/sm110_gemm_model/suite.py) | compute/component/full-GEMM host/GPU/commit linkage |

## 5. Non-claims

- 本附录中的路径存在不等于对应 Thor runtime 已完成；
- static-only、SASS presence、runner-defined 和 measured 是四种不同状态；
- microbenchmark rate 不自动成为 physical upper；
- result commit 的历史 envelope 不自动升级到当前模型 schema；
- 没有 raw bundle/hash/auditor 时不能人工抄 rate 进入 profile。
