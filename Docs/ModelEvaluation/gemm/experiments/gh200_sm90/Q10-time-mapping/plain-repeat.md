# 相同 plain 重复与起步相位

2 个目录（10-18），在多 CTA 共享源探针上进行。

## 结论

- 753467：三组相同 plain 配对，同 binary、同源、同 SM 与起步偏移。包络变化在 ±3.3% 内，逐 SM 完整窗口变化 11.3%–17.4%，中位频率变化小于 0.03%。
- 实际起步只变化 0–32 ns，固定起步平移的形式违反 4320–7136 ns，需要动态状态来解释。
- 含义：逐 SM 端点和“最晚 CTA 身份”在相同运行之间不稳定，模型应按包络和分布评价。完整 cfg_b 的同类核对见 [cfg-b-composition](../Q14-composition/cfg-b-composition.md#跨cta指标的可复现性)。

- 同二进制负对照 bridge-null（由 Q14 移入，777026）：R 同二进制重复 3/3、N 完整配对 2/2 通过原门，旧／新二进制等价桥接仍未通过；同二进制对照可用。

## 轮次

按日期排列；全部结果目录见[Q10-time-mapping/plain-repeat](../../../../../../results/gh200_resource_campaign/access_rules/Q10-time-mapping/plain-repeat/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| 相同plain的自然逐调用变化 | 同binary/固定每SM源及起步，原十次暖机；三进程各两个相同plain | 三对逐SM差11.300153%–17.395530%，包络差≤3.313253%；此前观察差不能全归给打点，保留原资格失败 | [执行](../../../../../../results/gh200_resource_campaign/access_rules/Q10-time-mapping/plain-repeat/20261018-Q10-tma-plain-repeat-001/EXECUTION-v2.md) |
| 相同plain的实际起步界 | 三个A/A配对、全部32SM，GT begin/end | 起步变化≤32ns，固定成本/次序图违反达4320–7136ns；起步平移形式停止 | [执行](../../../../../../results/gh200_resource_campaign/access_rules/Q10-time-mapping/plain-repeat/20261018-Q10-begin-phase-bound-001/EXECUTION.md) |
| 等值A源身份折叠是否改变持续行距增长 | 132CTA输入／计算、仅CTA0写及16次原源全输出前缀保持，只折叠m17–21到m0–4，CTA0地址不变；先原源桥接 | 775812零GPU构建及实际SASS通过，REG168、无溢出、无修复；[001](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/gemm-source-reuse/20261022-Q14-equal-value-source-001/README.md)的775816未启动即取消、测量0进程，775818回收已取消；002同型号132SM GH200迁移，UUID执行前绑定，三臂同批同卡，原区间、5%桥接与其余门槛保持；776745在romeo-a021完成14秒，24/24进程数值、暖机及观察合格；桥接仅trial0通过，未满足至少两项共同trial，Nfold未采，信息不足收口；[003邻接交换顺序](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/gemm-source-reuse/20261022-Q14-equal-value-source-003/README.md)按原最多48进程／5分钟GPU、0构建执行，24进程桥接后收口；模型／拟合0，原12及独立C2/C3状态保持 | [冻结计划、源等值与重叠依据](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/gemm-source-reuse/20261022-Q14-equal-value-source-002/README.md)  003的776879同卡24/24有效但0组过桥，折叠未采；[同binary负对照](../../../../../../results/gh200_resource_campaign/access_rules/Q10-time-mapping/plain-repeat/20261022-Q14-bridge-null-001/README.md)777026在romeo-a056完成21秒，47/48进程有效；N1 A16 plain trial0暖机CV超2%保留，未补跑。R同二进制重复3/3、N完整配对2/2通过原门；旧、新二进制等价桥接仍未通过，同二进制对照可用。 |
