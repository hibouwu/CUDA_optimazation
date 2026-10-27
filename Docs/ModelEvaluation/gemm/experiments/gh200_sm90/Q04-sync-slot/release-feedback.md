# 释放与反馈观察（R13 衍生）

10 个目录（10-10、10-11）：在真实 GEMM 内观察首代 ready、槽释放与累计反馈。

## 结论

- 用槽释放界绑定首代后，20 个起步配对的 issue→ready 增量下界均为正，排除“ready 不变、只有 consumer 晚返回”的解释。具体的搬运、通知、调度价格仍未辨识（739653）。
- 首代只读观察只有 10/40 条记录可绑定，信息不足（739645）。
- 完整 128 次 arrival 观察的两 stage 上游最大差 848/1020 cycle，超过 192 cycle 的资格门（740061）。
- 累计反馈：stage6 的 F 中位 19 cycle，stage4 的区间含 0，“两 stage 共同固定负补偿”不成立（740370）。
- 在真实 GEMM 内观察释放与反馈，扰动超出资格门；能稳定保留的只有上面第一条否定结论。

## 轮次

按日期排列；全部结果目录见[Q04-sync-slot/R13-release-feedback](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/R13-release-feedback/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| 739645：首代ready只读观察：独立test_wait能否把首代ready与consumer返回分开？ | 固定四组起步×行距对照；同binary启闭stage0 phase0只读观察，以exit≤FIRST绑定首代。 | 仅10/40开启记录可绑定，30/40检测界跨FIRST；该批首代辨识不足，保留原判定，不用后批释放界回填。 | [原结果目录](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/R13-release-feedback/20261010-R13-ready-observer)、[原页判定](../Q02-tma-request/R13-async-retirement.md#ready-observer) |
| 739653：用槽释放界绑定ready代次：ready不变、只有consumer晚返回能否解释起步响应？ | 沿原四条件，增加首个stage0 release前界；必须在slot可复用前成功退出，保留区间而非取中点。 | 40/40可绑定首代，20个起步配对的issue→ready增量下界均正；仅晚返回解释不相容，具体搬运/通知/调度价格未辨识。 | [原结果目录](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/R13-release-feedback/20261010-R13-ready-release)、[原页判定](../Q02-tma-request/R13-async-retirement.md#ready-observer) |
| <a id="round-release-threads"></a>740061：全部128次arrival的冻结观察：完整必要arrival覆盖能否取得释放→再取得的观察资格？ | 同ef卡、同binary、16次off前序；stage4/6×off/on共40study，加20plain，观察全部128线程原arrival前后。 | 完整D、身份和local/event门槛通过，但两stage上游最大差848/1020cycle超192；观察资格失败，不采用post-arrival机制分类或价格。 | [原结果目录](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/R13-release-feedback/20261011-R13-release-threads-ef-study)、[原页判定](../Q02-tma-request/R13-async-retirement.md#ef卡冻结研究未通过上游门槛) |
| <a id="round-wait-feedback"></a>740370：累计观察反馈：此前14组观察的反馈能否视为两stage共同的固定负补偿？ | 同binary、16次off前序；16随机块，各比较stage4/6的ALL15、ONLY2、ONLY16，共96目标，联合计算F=L16−L2。 | stage6 F中位19cycle、条件区间[3,37]；stage4区间含0，共同固定负补偿不成立；未恢复无观察Q/R或授予定价资格。 | [原结果目录](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/R13-release-feedback/20261011-R13-wait-feedback-diagnostic)、[本轮判定](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/R13-release-feedback/20261011-R13-wait-feedback-diagnostic/README.md) |
