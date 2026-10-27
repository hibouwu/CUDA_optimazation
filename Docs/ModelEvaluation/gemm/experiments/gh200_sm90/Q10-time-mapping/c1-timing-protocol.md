# C1 计时协议

5 个目录（10-15）：主机提交、CUDA event 与 graph 计时对单次波动的影响。

## 结论

- 长条件改用 graph 后，两种计时的宽度都减半以上，设备量变化小于 3%（746805、746820）。
- graph 暖机把首次尝试失败从 13 次降到 3 次，但四个齐备 case 的设备量变化超过 3%，完整等价主张未通过（746848、746850）。
- C1 正式开发据此固定为 graph 暖机与目标、plain 7 次、dual 2 次；协议与评分见 [C1-01](../Q16-config-selection/C1-01-selection.md)。

## 轮次

按日期排列；全部结果目录见[Q10-time-mapping/c1-timing-protocol](../../../../../../results/gh200_resource_campaign/access_rules/Q10-time-mapping/c1-timing-protocol/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| 主机提交／event协议是否放大单次计时波动（746805、746820，开发） | 固定cfg_b的短、长两个开发条件及GEMM SASS，改变frozen／legacy／非阻塞stream／单kernel graph目标调用 | 长条件graph把两种计时宽度均减半以上，设备量变化小于3%；转入固定graph目标的暖机对照，短条件保留缺失 | [判据与首作业](../../../../../../results/gh200_resource_campaign/access_rules/Q10-time-mapping/c1-timing-protocol/20261015-c1-timing-protocol-development-001)、[原清单续跑与结果](../../../../../../results/gh200_resource_campaign/access_rules/Q10-time-mapping/c1-timing-protocol/20261015-c1-timing-protocol-development-002/RESULTS.md)、[两作业分析](../../../../../../results/gh200_resource_campaign/access_rules/Q10-time-mapping/c1-timing-protocol/20261015-c1-timing-protocol-development-002/analysis.json) |
| graph暖机能否等价且降噪（746848、746850，开发） | 固定三形状×八候选、graph目标与原数值规则，只改变legacy／graph暖机 | 首尝试失败13→3，但四个齐备case的设备量超过3%，完整等价主张未通过；正式开发固定graph暖机／目标及plain7、dual2 | [冻结对照](../../../../../../results/gh200_resource_campaign/access_rules/Q10-time-mapping/c1-timing-protocol/20261015-c1-graph-protocol-expansion-001)、[补齐NumPy后的续跑与结果](../../../../../../results/gh200_resource_campaign/access_rules/Q10-time-mapping/c1-timing-protocol/20261015-c1-graph-protocol-expansion-002/RESULTS.md)、[逐case分析](../../../../../../results/gh200_resource_campaign/access_rules/Q10-time-mapping/c1-timing-protocol/20261015-c1-graph-protocol-expansion-002/analysis.json)；[d058固定调用次数计划](../../../../../../results/gh200_resource_campaign/access_rules/Q10-time-mapping/c1-timing-protocol/20261015-d058-fixed-history-discrimination-001)及[提交身份](../../../../../../results/gh200_resource_campaign/access_rules/Q10-time-mapping/c1-timing-protocol/20261015-d058-fixed-history-discrimination-001/submission.json)为已取消的747558执行历史，见[停止记录](../access_rules/EXECUTION.md#当前状态) |
| a044 graph 参照（780670，54 槽） | 当前 C1 graph 协议的六参照 | 有效频率 1.648–1.772 GHz；同卡 graph 时间比旧协议低 1.598%–5.116%，两协议的 plain SASS 逐字节相同；主机、预热与作业漂移尚未隔离 | [graph 桥接](../../../../../../results/gh200_resource_campaign/access_rules/Q10-time-mapping/c1-timing-protocol/20261023-Q10-a044-graph-reference-001/README.md)、[代码与输入核对](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/plain-kernel-comparison/README.md) |
