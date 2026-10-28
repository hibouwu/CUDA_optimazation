# 存储局部域

2 个目录（10-15）：用页对依赖加载的延迟识别共享存储的两域，并在新分配上冻结留出。

## 结论

- 两个存储域 GPC{0,4,5,6}/{1,2,3,7}，每域 66 SM，三次留出全部通过（746596）。
- 这是源状态模型使用的工作映射；源状态的其余判别见 [source-state-candidates](source-state-candidates.md)。

## 轮次

按日期排列；全部结果目录见[Q03-source-service/storage-domain-numa](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/storage-domain-numa/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| 存储两域独立页对留出（746596） | 冻结两域映射与相关性门槛，在三次新分配的新页对上检验 | 三次均通过；支持 GPC{0,4,5,6}/{1,2,3,7} 两域工作映射，每域 66 SM | [本轮记录](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/storage-domain-numa/20261015-numa-affinity-heldout-001/README.md)、[评分](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/storage-domain-numa/20261015-numa-affinity-heldout-001/score.json)、[机器可读映射](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/storage-domain-numa/20261015-numa-affinity-heldout-001/candidate_map.json) |
| 源页延迟是否存在共享两域（746593，开发） | 固定64 MiB分配，每个GPC选两个SM，比较同一页对经 discard 后的冷／热依赖加载。 | 页对响应给出 GPC{0,4,5,6}/{1,2,3,7} 的两域候选、每域66 SM，随后冻结新页留出。 | [先导及746591无效热加载历史](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/storage-domain-numa/20261015-numa-latency-development-001/README.md)、[有效加载分析](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/storage-domain-numa/20261015-numa-latency-development-001/attempt002/analysis.json)、[地址低位开发推断](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/storage-domain-numa/20261015-numa-latency-development-001/attempt002/address-parity-development.json) |
