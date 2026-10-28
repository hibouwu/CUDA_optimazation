# 多 CTA 的 GPC 共享通路

7 个目录（10-15）：SM 分组拓扑、共享通路对照与每 GPC 服务槽数的两次冻结留出。本地干扰和发出位置见 [issue-placement-interference](issue-placement-interference.md)。

## 结论

- SM 之间存在按组共享的源服务通路：同组配对使窗口增加约 21.8%–22.1%，跨组接近零（746555）。分组来自 cluster 共驻留核对给出的八个主要连通分量，以及竞争对照（746548）；文中“GPC”沿用模型的分组名，未确认与物理 GPC 一一对应。
- 每 GPC 服务槽数：第一次冻结留出（746563）中 8 槽低估 21.12%，2/4 槽最大误差 5.29%／5.47%；第二次（746570）2 槽版本全部通过，ns/cycle 最大误差 3.64%／2.32%。现用组合以“每 GPC 两槽”为工作假设。
- 服务结构与接入方式见 [TMA-SHARED-SERVICE](../../../model/mechanism/hardware_rules/TMA-SHARED-SERVICE.md)。

## 轮次

按日期排列；全部结果目录见[Q02-tma-request/multi-cta-gpc-sharing](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/multi-cta-gpc-sharing/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| cluster 核对 SM 同组关系（746548） | 利用同 cluster CTA 的同 GPC 共调度关系，改变 cluster 大小 2–8 | 确认八个主要连通分量及 124–131 的四个内部配对；后四对与主要分量的关系留待共享服务对照 | [本轮记录](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/multi-cta-gpc-sharing/20261015-gpc-topology-001/README.md)、[拓扑结果](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/multi-cta-gpc-sharing/20261015-gpc-topology-001/topology.json)；首次 API 编译失败留在同轮执行历史 |
| 未解析 SM 配对的共享通路对照（746555） | 固定标量加载与原六个 SM，分别加入已知同组、跨组及四对未知 SM | 未知四对与同组对照均使窗口增加约 21.8%–22.1%，跨组接近零；支持当前共享服务分组 | [本轮记录](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/multi-cta-gpc-sharing/20261015-gpc-contention-002/README.md)、[分析](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/multi-cta-gpc-sharing/20261015-gpc-contention-002/analysis.json)；[加载被移出循环的两次失败](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/multi-cta-gpc-sharing/20261015-gpc-contention-001/README.md)作为执行历史保留 |
| GPC 服务第一次冻结相位留出（746563） | 固定 N32/Q1 与 2/4/8 槽参数，改变 CTA 起步间隔 0/16/32 ns | 8 槽提前下降、低估 21.12%；2/4 槽最大误差仍为 5.29%／5.47%，全部未通过；下一轮保留 2/4 槽结构 | [本轮记录](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/multi-cta-gpc-sharing/20261015-tma-gpc-heldout-001/README.md)、[评分](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/multi-cta-gpc-sharing/20261015-tma-gpc-heldout-001/score.json)；[有限槽开发与条件筛选](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/multi-cta-gpc-sharing/20261015-tma-gpc-development-001/README.md)属于本轮开发 |
| GPC 服务第二次冻结相位／Q2 留出（746570） | 固定 N32/G64、输入与四个物理槽，检验 Q1 新相位及 Q2 的 6 条新条件 | 2 槽版本本轮全部通过，ns/cycle 最大误差 3.64%／2.32%；4/8 槽失败，暂以 2 槽进入迁移 | [本轮记录](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/multi-cta-gpc-sharing/20261015-tma-gpc-heldout-002/README.md)、[评分](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/multi-cta-gpc-sharing/20261015-tma-gpc-heldout-002/score.json)、[预测](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/multi-cta-gpc-sharing/20261015-tma-gpc-heldout-002/predictions.json)；[同结构单坐标再校准](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/multi-cta-gpc-sharing/20261015-tma-gpc-development-002)、[逐点筛选](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/multi-cta-gpc-sharing/20261015-tma-gpc-development-002/screen.json)作为本轮开发依据。 |
