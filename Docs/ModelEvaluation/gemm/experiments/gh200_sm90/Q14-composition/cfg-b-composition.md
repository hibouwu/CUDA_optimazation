# cfg_b 同版组合

5 个目录（10-15 至 10-20）：原 12 条件同版回放、计算活跃对 producer 发出的影响与同版系统。状态化源与信用候选见 [gemm-input-network](../Q02-tma-request/gemm-input-network.md)，经验前缀见 [startup-prefix](../Q01-scheduling/startup-prefix.md)。

## 结论

**现行默认（10-16 前缀版）**
- 原 12 条件的 plain 绝对误差中位 10.225%、最大 25.422%。
- stage 响应预测 3.903%，实测 10.467%。
- A16预测+0.0935%，B80、B112预测0%，实测分别为+28.9%、+18.1%、+18.6%。
- G1/G2/G4 等短条件误差 −11%～−20%。
- 其余长条件误差在 6% 以内。

**G3 S6 主循环已接近计算受限**
- 746824 原始程序的 dual 记录中，主循环（FIRST_MMA→MAIN_END）为 33144／33352 cycle，纯计算 32768 cycle；组合模型在这里误差为 first_L −1.2%、later_L −0.7%。剩余误差在满载供给（stage、行距）与启动段。

**排除的固定形式**
- 固定生产者延迟。源信用、分道、轮转额度、统一提价与回传信用等输入服务候选见 [gemm-input-network](../Q02-tma-request/gemm-input-network.md)。

<a id="跨cta指标的可复现性"></a>
**跨 CTA 指标的可复现性**
- “最晚 CTA 命中 1/120”不适合作失败指标：每个条件重复 10 次，实测最晚 CTA 有 3–8 个不同身份；总猜各条件出现最多的那个，上限也只有 45/120。
- 模型选出的关键 CTA，在实测中的跨度比真正最晚者中位只短 1.65%，90% 在 3.85% 以内（由 full-matrix 的 comparison-final.json 计算）。
- 跨 CTA 推进应改用跨度短缺或结束时刻分布评价。

**入口**
- 组合入口与模块资格见[组合](../../../model/mechanism/composition.md)。

## 轮次

按日期排列；全部结果目录见[Q14-composition/cfg-b-composition](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| 状态化源服务能否联合解释G3两种stage（CPU开发回放） | 固定原状态价格、阶段频率与220cycle复用候选，把G3 S6/S4请求接回共同程序并对照746679 | stage变慢预测4.04%、实测12.10%，第二MAIN跨度仅30.45/46.01ns；将CTA空间推进确定为后续约束 | [组合与逐端点对照](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-combination-regression-001/README.md)、[内部跨度诊断](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-combination-regression-001/output/internal-diagnosis.json) |
| 计算活跃期的固定生产者延迟能否补足反例（CPU筛查） | 固定源、目的、计算和频率，只比较每条活跃期发出增加40／80cycle，S6/S4共用价格 | S4平均L改善但第二MAIN跨度仍约51ns，S6仍30.45ns；两版本均停止，不接入组合 | [候选、反例与复现](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-local-producer-candidate-001/README.md)、[40cycle报告](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-local-producer-candidate-001/delay40/report.json)、[80cycle报告](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-local-producer-candidate-001/delay80/report.json) |
| 同一状态化版本能否覆盖原12条件的stage与行距响应（CPU开发回放） | 固定同一规则和价格，按实际工作列表、Ktile与行距扩至原12条件，统一对照739011 | plain绝对误差中位17.378%、最大39.797%；B行距的候选可见输入相同，地址无关规则仍给零响应，后续须让地址参与服务 | [全矩阵与执行恢复](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-full-matrix-001/README.md)、[三类模型与端点对照](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-full-matrix-001/comparison-final.md)；[共同程序地址适配与原G3逐位回归](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/gemm-input-network/20261015-cfg-b-program-source-adapter-001)为接入产物 |
| 当前局部接口能否固化为同版完整系统 | 当前源码、原profile与协议；G3 S6／S4／A16三条件33调用，其他九项复用保存评分 | 396个CTA、152064个请求、772596个节点时刻及plain逐值保持；交付55份源码和12份输入，54项CPU测试通过；原10.225%／25.422%与C2/C3失败保持 | [当前版本与同条件评分](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261020-Q14-current-system-001/README.md) |
