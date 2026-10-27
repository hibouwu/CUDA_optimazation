# EXP-04：访存基线与计时空窗口（已并入其他页面）

[结果总览](../README.md)

本页内容于 2026-10-08 按测量对象迁出，原始 run `memory_baseline/formal-v3-a` 与探针 `probes/memory_baseline.cu` 不变。

| 原内容 | 现位置 |
|---|---|
| SMEM stride 读写，含 stride=1 不稳定的记录 | [EXP-09 前序测量](../Q07-smem/EXP-09-smem.md#exp-04) |
| global 读、写与依赖复制 | [EXP-13 前序测量](EXP-13-global-rw.md#exp-04) |
| 单 CTA 与整卡计时空窗口 | [R09 计时空窗口](../Q09-frequency/R09-inkernel-clock-stages.md#empty-window) |

迁出前的全文见 Git 历史。

## a044 复测（2026-10-23～26）

171 次原调用全部通过原数值与观测审计，11 个原发布条件通过统计门槛。SMEM B/cycle 中位变化最大 0.00213%，5 个 global 条件 GB/s 变化为 −0.805%～+0.531%。原来因波动未发布的 SMEM stride1 读，CV 从 9.85% 降到 0.000389%。见[访存报告](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/resource-memory-report.md)。

NCU 补上了缓存层级：8 MiB `.cg` 读的单 pass L2 读为 17,302,682 sector（逻辑量/32 多 1,178），DRAM 读 768 B，说明已预热的小工作集由 L2 供给；10-23 的 3-pass 测到 152,320 B，来自 replay 之间的缓存状态。256 MiB 时 L2 总读约为逻辑量的 1.5 倍，按来源拆分后，多出的部分是 LTCFABRIC 跨分区转发。见[L2 来源分析](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/ncu-256m-interpretation.md)、[单 pass 复核](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/ncu-pending-20261026-report.md)。
