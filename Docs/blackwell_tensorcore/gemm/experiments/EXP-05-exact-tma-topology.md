# EXP-05：exact TMA topology

## 1. 研究问题

serialized payload surface 不能证明具体 GEMM schedule 的多 request/stage topology。该实验测量 precision、tile、value/scale request、stage、thread、SMEM layout、row stride 和 residency 完全冻结的 TMA service。

## 2. 对应模型参数

```text
tma.smem_ingress.contract.<family>.stride<ld>.per_sm
tma.hbm.contract.<family>.stride<ld>
```

## 3. Formal family contract

每个 family 冻结：

- BM/BN/BK；
- value bits 与 physical transport layout；
- scale block；
- stages；
- threads 与 controller thread；
- A/B value bytes；
- A/B scale bytes；
- requests/stage；
- dynamic SMEM footprint；
- precision IDs。

每个 family × row stride {1024,2048,4096} × residency {hot-L2,cold-DRAM} 形成 54 个 case。

## 4. Residency 与作用域

hot-L2：

- 单 CTA；
- 单 observed SM；
- B/s/SM；
- `tma.smem_ingress...per_sm`。

cold-DRAM：

- 20 CTA；
- one CTA/SM；
- B/s/GPU；
- `tma.hbm.contract...`。

## 5. 工作量与计时

runner 根据 formal family contract 重建每 stage payload：

\[
q_{\mathrm{stage}}
=q_{A,v}+q_{B,v}+q_{A,s}+q_{B,s}.
\]

按 blocks × iterations × stage bytes 得到 requested numerator，并使用单 CTA 或 full-grid `%globaltimer` 区间计算 rate。

## 6. 接受门禁

- 54/54 static contract；
- `sm_110a` binary；
- function-scoped `UTMALDG.2D`；
- run spec 与 contract manifest hash；
- 10 external trials；
- row_stride=2048 的预声明 NCU selection；
- hot/cold SM coverage；
- source、compile、binary、SASS、environment、artifact SHA-256；
- campaign auditor 与 platform auditor 双层通过。

## 7. 当前状态

当前 source/static/auditor contract 已冻结，54/54 本地静态合同与 SM110 SASS 检查通过；正式 54-case Thor capacity bundle 尚未回传。

现有历史 tc5a experiment 只覆盖：

- tc5a FP16 stride 2048；
- tc5a BF16 stride 2048。

因此 runner coverage 为 2/28 schedule/precision pair，其余 26 个 pair 缺失。

## 8. 进入模型

模型从 schedule 的 `tma_contract_family_by_precision` 和 workload 的 packed leading dimensions 生成 exact resource ID。以下任一不匹配都 fail closed：

- precision；
- family；
- K≠N 导致 A/B stride 不同；
- stride 未测；
- payload/request/stage/thread/layout；
- residency；
- hardware/mode/clock。

## 9. 不能证明什么

- exact TMA service 仍不是 TMA+MMA+epilogue causal profile；
- hot per-SM rate 不是 GPU-wide L2 capacity；
- cold TMA read path 不证明 output write joint service；
- static 54/54 不是 Thor runtime capacity；
- tc5a stride2048 不能外推到 1024/4096。

## 10. 源码与工件

- formal manifest：[contract_manifest.json](../../../../microbench/sm110_gemm_resource_campaign/contract_manifest.json)
- CUDA source：[tma_ab_contract_bandwidth.cu](../../../../microbench/15_tma_ab_contract_bandwidth/tma_ab_contract_bandwidth.cu)
- runner：[run_resource_campaign.py](../../../../microbench/sm110_gemm_resource_campaign/run_resource_campaign.py)
- campaign auditor：[audit_campaign.py](../../../../microbench/sm110_gemm_resource_campaign/audit_campaign.py)
- platform auditor：[audit_resource_suite.py](../../../../microbench/sm110_gemm_resource_campaign/audit_resource_suite.py)
- importer：[resource_import.py](../../../../scripts/sm110_gemm_model/resource_import.py)
