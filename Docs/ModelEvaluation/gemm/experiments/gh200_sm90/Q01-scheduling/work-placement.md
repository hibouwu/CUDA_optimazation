# 工作放置与执行位置

2 轮（10-10、10-15）检验行距与 stage 响应是否只随逻辑工作序列移动。

## 结论

- 交换整条工作序列后，响应更接近原物理执行位置，只按逻辑 B/D 工作收费的形式被反对（739608）。
- 冻结判别 747022：只改工作放置，S6 响应由 +8.657% 变为 −10.198%，S4 控制接近零；支持局部放置响应，未移动 CTA 的推进也随之变化。
- SM 硬件位置与该位置形成的时间状态仍未分开；工作映射本身已在[程序图](../../../model/mechanism/program/cutlass_program_graph.md)核对，放置相关的服务没有定价。

## 轮次

按日期排列；全部结果目录见[Q01-scheduling/work-placement](../../../../../../results/gh200_resource_campaign/access_rules/Q01-scheduling/work-placement/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| 739608：交换整条工作序列：行距响应是否只随逻辑B/D工作序列迁移？ | 同卡、几何、容量、B0/B32和逐轮工作集合固定；交换物理CTA55/91整条持久工作序列，实际SM放置保留。 | 响应更接近原物理执行位置，反对仅按该逻辑工作收费；SM硬件与该位置形成的时间状态仍未分开。 | [原结果目录](../../../../../../results/gh200_resource_campaign/access_rules/Q01-scheduling/work-placement/20261010-R10-work-swap)、[原页判定](../Q03-source-service/R10-layout-cache.md#work-swap) |
| 工作放置是否改变奇偶响应（747022，冻结判别） | 固定d226、暖机和rank配对，仅在目标轮换前四个cluster的工作，以S4作控制。 | S6响应由+8.657%反转为−10.198%，S4控制接近零，支持局部工作放置响应；未移动CTA的推进也变化。 | [结果](../../../../../../results/gh200_resource_campaign/access_rules/Q01-scheduling/work-placement/20261015-d226-work-placement-discrimination-001/RESULTS.md)、[评分](../../../../../../results/gh200_resource_campaign/access_rules/Q01-scheduling/work-placement/20261015-d226-work-placement-discrimination-001/score.json) |
