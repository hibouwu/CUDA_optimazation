# GH200：经过留出验证的规则

当前没有满足完整时间、局部分项、关键 CTA 和适用范围联合要求、可由本页收录为完整 GEMM 规则的候选。V09 r2 未通过；m11 有开发条件关系，两次冻结留出均未通过。整理不撤销已保存的局部或完整时间成绩，也不扩大它们的资格。

半经验公式与参数绑定维护在[MODEL-CANDIDATES](../../../model/semi_empirical/MODEL-CANDIDATES.md#v09-model)，当前硬件假设与反证入口见[硬件路径目录](../../../model/mechanism/hardware_rules/README.md)，具体实测保留在原页。规则的采用要求见[留出与采用细则](RESEARCH-GUIDE.md#validation)。

| 留出记录 | 已取得的成绩 | 未取得的资格 |
|---|---|---|
| [V01](V01-validation.md) | 异步组合的单 CTA 总窗口成绩通过 | 分项误差抵消、计时探针改动及整卡/完整 kernel 失败保留，不授予各组件机制资格 |
| [V07](V07-rule-validation.md) | 24条件完整时间3.06%/8.71% | 分项和关键 CTA 定位未全部通过，限原卡、原范围 |
| [V09](V09-component-validation.md) | 原冻结29个受支持预测4.11%/35.46%，另1个原拒绝 | 整体未通过，冻结身份和失败成绩不变 |
| [m11](R03-access-demand.md#m11-current) | 开发数据、兼容窗口复测和两轮反证；[v2 留出](R03-access-demand.md#m11-holdout-v2-result) 47 个受支持条件中位 0.44%，8 warp 独立读写 ≤1.83% | v1 最大 34.82%、v2 最大 11.60%（1 warp 独立读），两次留出均未通过，不授予规则资格 |

## 原入口迁移

下列旧链接保留，只跳转候选或开发关系，不表示验证通过。

| 原入口 | 维护位置 |
|---|---|
| <a id="v09-model"></a>v09-model | [候选原文](../../../model/semi_empirical/MODEL-CANDIDATES.md#v09-model) |
| <a id="source-sharing-dev"></a>source-sharing-dev | [候选原文](../../../model/semi_empirical/MODEL-CANDIDATES.md#source-sharing-dev) |
