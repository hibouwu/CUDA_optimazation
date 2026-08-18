# 08 当前覆盖与缺口

本章是截至指定 code/evidence commit 的状态快照，不重新定义模型公式。任何新采集或模型修改都必须重新生成本章对应的 coverage JSON。

## 1. 基线

| 项目 | 值 |
| --- | --- |
| 模型代码 commit | `f06f2cd917a4cb23806b5e1be06120be9152ed7b` |
| parameter supplement code commit | `0c42cbb7987e204a2c8f78f17e4cce0096fbdef0` |
| parameter supplement GPU data commit | `aa845dd9e70e2c541ae3a7d5293bf8de4bd55092` |
| parameter supplement result branch | `thor-results/thor-t5000-parameter-plots-maxn-20260817-i` |
| hardware | `thor_t5000_sm110_20sm` |
| operating mode | MAXN |
| target completion | `false` |

旧 `ba651f0e...` closure 仍是有效历史采集，但它的独立 read/write 经验组合和旧 TMA applicability 不能代表当前 f06 模型。详见 [历史结果](../appendices/historical_results.md)。

## 2. Runner 定义覆盖

当前 `runner_coverage.py` 机械报告：

| surface | 状态 |
| --- | --- |
| compute surface | 12 precision、36 full-SM shape，定义完整 |
| common component surface | 定义完整 |
| TMA serialized payload | 4/8/16/32/64 KiB × hot/cold，定义完整 |
| memory duplex | 21 case，hot-L2 与 cold proxy ratio matrix 定义完整 |
| exact TMA topology | 2/28 schedule/precision pair |
| independent joint-pipeline runner | 未定义为全目标 runner |
| full-GEMM runner | 6/12 precision path |
| physical cold external write bytes | 未闭合 |
| all performance parameter runners | `false` |

exact TMA topology 当前只覆盖：

- `tc5a_m128n256k64_stage4 × fp16_f32`；
- `tc5a_m128n256k64_stage4 × bf16_f32`。

其余 generic、FP6 direct-SMEM、block-scale 和其它 shape 共 26 个 pair 未闭合。

## 3. `-i` parameter supplement 导入结果

真实结果经过当前 importer 与独立 auditor 重放：

| 导入对象 | 数量 | 资源语义 |
| --- | ---: | --- |
| TMA payload capacity | 10 | 5 payload × hot/cold |
| hot-L2 duplex capacity | 14 | `l2.duplex`，ratio-qualified |
| cold memory proxy | 7 | `hbm.duplex.proxy` |
| physical HBM duplex | 0 | 未产生，也不得由 proxy 升级 |

当前 required-ratio coverage：

| 门禁 | 值 |
| --- | --- |
| `all_required_l2_duplex_ratios_measured` | true |
| `all_required_hbm_duplex_proxies_measured` | true |
| `all_required_hbm_duplex_ratios_measured` | false |
| `all_required_tma_payloads_planned` | false |
| `all_required_tma_payloads_measured` | false |

TMA payload 缺口来自 block-scale 独立 scale requests：

- 512 B；
- 1024 B。

4/8 KiB value request 已覆盖，但不能把 value+scale 合并成一条虚构的 4.5/9 KiB request。

## 4. 严格 compute upper

当前约 7/12 precision 有可进入 `tensor_core_classical` 的产品级或条件 compute upper：

- FP16；
- BF16；
- E4M3；
- E5M2；
- NVFP4；
- S8；
- U8。

仍缺：

- TF32；
- E3M2；
- E2M3；
- raw E2M1；
- MXFP4。

此外，`all_classical` 需要 `compute.total.<precision_id>` aggregate upper；当前没有完整证据，不能把 Tensor Core upper 自动推广到所有经典实现。

## 5. Causal 与 cache residency

| 门禁 | 当前值 | 原因 |
| --- | --- | --- |
| dependency-span solver exists | true | persistent-worker DAG 已实现并测试 |
| closure-qualified Thor causal profile | 0 | tc5a FP16/BF16 runner 已冻结但尚无 Thor timing bundle |
| `dependency_span_model_complete` | false | 全目标 profile/freeze/evidence 不完整 |
| `cache_residency_model_complete` | false | exact workload/schedule residency 合同未覆盖全部场景 |
| `joint_overlap_model_complete` | false | exact joint contracts 未覆盖全部 required workload/schedule |

求解器存在不等于 profile 存在；静态 SASS 或 synthetic fit 也不等于 fresh Thor timing。

## 6. 完整 GEMM 与全精度

| 门禁 | 当前值 |
| --- | --- |
| precision contracts present | true |
| compute campaigns planned | true |
| complete data paths modeled for current manifest | true |
| all full-GEMM campaigns planned | false |
| all full-GEMM scenarios planned | false |
| all-precision absolute three-layer closure | false |

当前历史 observation 覆盖 FP16、BF16、TF32、E4M3 和 S8；E5M2 runner 已加入但尚无对应 fresh Thor observation。其余 precision 仍缺 candidate/reference、same-contract correctness 或同精度 denominator。

## 7. 仍需 Thor 的实验

按依赖顺序：

1. 512 B 与 1 KiB block-scale scale TMA payload；
2. 26 个缺失 exact TMA schedule/precision pair；
3. 能证明 physical external write bytes 的 HBM/LPDDR joint capacity，或有一手来源的等价 MCC 绝对字节合同；
4. tc5a FP16/BF16 causal 182-case 实跑；
5. 其它 schedule/precision 的 causal/joint campaign；
6. 剩余精度完整 GEMM；
7. 缺失 strict compute upper；
8. residency/cache reuse 的独立证明。

## 8. Source freeze 与附录

当前重构后机械状态：

| 门禁 | 值 | 说明 |
| --- | --- | --- |
| `duplex_campaign_frozen` | false | 旧 freeze basis/hash 不自动覆盖 current 文档/模型提交，需要按 current basis 重新生成 |
| `epilogue_campaign_frozen` | false | current completion 要求的 frozen artifact set 尚未重新绑定 |
| `joint_pipeline_campaign_frozen` | false | independent all-target joint campaign 尚未冻结 |
| `final_source_appendix_generated` | true | current source appendix 已迁移到 `gemm/appendices/` |

freeze=false 不否定已返回 raw trial；它表示 future campaign 的设计 basis 和源码 hash 还没有按 current commit 重新冻结。

## 9. 文档可以先完成

上述硬件缺口不妨碍规范模型文档完成。严谨文档的正确输出可以是 `insufficient_evidence`；不能为了让表格全绿而填入邻近 rate、proxy 或静态结果。

当前模型重放和历史结果分别见：

- [当前模型重放](../appendices/current_model_replay.md)
- [历史结果](../appendices/historical_results.md)
- [审计与复现](../appendices/audit_and_reproduction.md)
