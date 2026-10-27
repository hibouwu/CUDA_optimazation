# cfg_b 源订阅与共享关系

7 轮（10-15）：在多 CTA 探针上检验源共享关系与订阅服务。

## 结论

- 同源被多个 CTA 读取时窗口明显缩短：746797 中 Q1/Q4 未打点周期约缩短 19.54%／28.58%。独立固定收费失配，旧在途合并也不能同时解释绝对窗口与降幅；保留“共享关系存在”这一事实。
- cfg_b 匹配源订阅的四个版本在冻结判别 746586 中全部被反证，ns/cycle 最大误差 45.14%／44.47%。
- 请求服务与在途源服务串联的 CPU 开发最大误差 28.20%，停止扩参，不接入 GEMM。
- 现行源服务假设与反例见 [TMA-GEMM-HYPOTHESES](../../../model/mechanism/hardware_rules/TMA-GEMM-HYPOTHESES.md)。

## 轮次

按日期排列；全部结果目录见[Q02-tma-request/gemm-source-candidates](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/gemm-source-candidates/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| cfg_b 匹配源订阅冻结判别（746586） | 固定 G3 实际两轮工作坐标和 34 MiB A/B，改变活动组与相位；6 个无先前写入条件检验四种订阅版本，另 6 个写入条件作开发 | 四版本最大 ns/cycle 误差 45.14%／44.47%，全部被反证；转向近期源状态 | [本轮记录](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/gemm-source-candidates/20261015-tma-gemm-source-discrimination-001/README.md)、[评分](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/gemm-source-candidates/20261015-tma-gemm-source-discrimination-001/score.json)；[匹配探针、控制校准与候选筛选](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/gemm-source-candidates/20261015-tma-gemm-source-development-001/README.md)、[此前 GPC→GEMM 迁移](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/multi-cta-gpc-sharing/20261015-tma-gpc-development-001/README.md)挂在此迁移链 |
| 源关系对照开发（746797） | 固定探针、总分配与活动组织，改 CTA→源关系：独立源、每 GPC 四源／一源、全卡一源 | Q1/Q4 未打点周期分别缩短约 19.54%／28.58%；独立固定收费失配，旧在途合并也未同时解释绝对窗口与降幅；保留共享关系，另检验固定源集合重排 | [本轮结果](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/gemm-source-candidates/20261015-tma-source-sharing-development-001/RESULTS.md)、[逐进程分析](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/gemm-source-candidates/20261015-tma-source-sharing-development-001/analysis.json)、[执行方案](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/gemm-source-candidates/20261015-tma-source-sharing-development-001/README.md) |
| 请求服务与在途源服务串联（CPU 开发） | 固定 746797 的端点、控制和目的价格，共同拟合两级服务及完成延迟 | 最大误差 28.20%；不同 GPC 被模型视为相同程序产生至少 5.707% 的对称下界；停止该结构扩参，不接入 GEMM | [本轮记录](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/gemm-source-candidates/20261015-tma-request-source-split-development-001/README.md)、[拟合](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/gemm-source-candidates/20261015-tma-request-source-split-development-001/fit.json)、[反例与辨识诊断](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/gemm-source-candidates/20261015-tma-request-source-split-development-001/diagnostics.json) |
