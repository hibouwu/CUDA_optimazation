# 首输出填零（R18 首填）

2 轮（10-09）检验首输出越界填零能否在 cfg_a 与 cfg_c 共用一个条件价。

## 结论

- cfg_a K4096 的 plain 下降 12.78%，cfg_c 只变化 +0.32%，不能共用条件价。
- 新点只补 A/C 的 first 参数，配置差异和不可辨识范围保留；边界与填零规则的主记录见 [R18](../Q01-scheduling/R18-cluster-boundary.md)。

## 轮次

按日期排列；全部结果目录见[Q02-tma-request/R18-first-fill](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/R18-first-fill/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| 首输出填零能否共用条件价（738707／738865，开发） | 固定输出、工作列表、分配与输入数值，扩展cfg_a/c的输入map，在K1024/4096比较partial／whole-fill与有效地址零。 | cfg_a K4096的plain下降12.78%，cfg_c仅变化+0.32%；新点只补A/C的first参数，保留配置差异及不可辨识范围。 | [738707原始目录](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/R18-first-fill/20261009-R18-first-fill-job738707)、[738865原始目录](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/R18-first-fill/20261009-R18-first-fill-job738865)；[R18配对与决定](../Q01-scheduling/R18-cluster-boundary.md#first-fill) |
