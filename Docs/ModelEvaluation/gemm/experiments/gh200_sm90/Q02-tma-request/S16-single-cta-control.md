# S16 单 CTA 控制成本

7 个目录（10-13）：输入事件模型 v1/v2、S16 stage 成本 v1–v3、原二进制静态窗口与历史复用盘点。

## 结论

- S16 的额外成本不能只由完成尾延迟解释；token 控制与组返回两个候选未区分，转入一次控制判别设计后暂停独立收尾。
- 候选与控制成本区分见 [TMA-S16-STAGE-COST](../../../model/mechanism/hardware_rules/TMA-S16-STAGE-COST.md)，首版输入事件模型见 [TMA-INPUT-EVENT-MODEL](../../../model/mechanism/hardware_rules/TMA-INPUT-EVENT-MODEL.md)，历史证据盘点见 [TMA-HISTORY-REUSE](../../../model/mechanism/hardware_rules/TMA-HISTORY-REUSE.md)。

## 轮次

按日期排列；全部结果目录见[Q02-tma-request/S16-single-cta-control](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/S16-single-cta-control/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| <a id="round-s16-control"></a>S16额外成本能否只由完成尾延迟解释（CPU开发） | 固定S1/S2已有价格，用S4R1校准新增成本，再比较S4R2；核对原二进制的token选择与保存代码。 | 只增加完成尾延迟的形式失败；token控制与组返回候选未区分，转入一次控制判别设计。 | [v3分析目录](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/S16-single-cta-control/20261013-tma-s16-stage-cost-v3)；[候选与原判定](../../../model/mechanism/hardware_rules/TMA-S16-STAGE-COST.md#31-已完成静态核对与历史一致性检查)、[中间版本](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/S16-single-cta-control/README.md) |
