# 相同 plain 重复与起步相位

2 个目录（10-18），在多 CTA 共享源探针上进行。

## 结论

- 753467：三组相同 plain 配对，同 binary、同源、同 SM 与起步偏移。包络变化在 ±3.3% 内，逐 SM 完整窗口变化 11.3%–17.4%，中位频率变化小于 0.03%。
- 实际起步只变化 0–32 ns，固定起步平移的形式违反 4320–7136 ns，需要动态状态来解释。
- 含义：逐 SM 端点和“最晚 CTA 身份”在相同运行之间不稳定，模型应按包络和分布评价。完整 cfg_b 的同类核对见 [cfg-b-composition](../Q14-composition/cfg-b-composition.md#跨cta指标的可复现性)。

## 轮次

按日期排列；全部结果目录见[Q10-time-mapping/plain-repeat](../../../../../../results/gh200_resource_campaign/access_rules/Q10-time-mapping/plain-repeat/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| 相同plain的自然逐调用变化 | 同binary/固定每SM源及起步，原十次暖机；三进程各两个相同plain | 三对逐SM差11.300153%–17.395530%，包络差≤3.313253%；此前观察差不能全归给打点，保留原资格失败 | [执行](../../../../../../results/gh200_resource_campaign/access_rules/Q10-time-mapping/plain-repeat/20261018-Q10-tma-plain-repeat-001/EXECUTION-v2.md) |
| 相同plain的实际起步界 | 三个A/A配对、全部32SM，GT begin/end | 起步变化≤32ns，固定成本/次序图违反达4320–7136ns；起步平移形式停止 | [执行](../../../../../../results/gh200_resource_campaign/access_rules/Q10-time-mapping/plain-repeat/20261018-Q10-begin-phase-bound-001/EXECUTION.md) |
