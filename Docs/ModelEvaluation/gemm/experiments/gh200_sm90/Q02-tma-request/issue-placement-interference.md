# TMA 发出位置与本地计算干扰

7 个目录（10-15）：固定其他 131 个 SM 的 WGMMA 背景，开关或改变目标 SM 上的本地计算（SS/RS）与 TMA 发出 warp，另含 TMA 与同 CTA WGMMA 并发的开发对照。

## 结论

- 固定远端背景时，本地计算使周期增加 12.10%／20.50%，支持局部干扰并排除仅频率解释（746720）；发出、SMEM 与反压未分离。
- 发出 warp 位置会移动慢位置，但约 2.25% 的周期差低于冻结下界，三个定量候选均未通过（746785）。
- 分组共享通路与每 GPC 服务槽数见 [multi-cta-gpc-sharing](multi-cta-gpc-sharing.md)。

## 轮次

按日期排列；全部结果目录见[Q02-tma-request/issue-placement-interference](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/issue-placement-interference/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| 固定远端背景的本地计算干扰留出（746720） | 其他 131 个 SM 的 WGMMA 固定，只开关目标 SM 的第二 WG；使用新 Q1/Q4 源环 | 周期增加 12.10%／20.50%，支持冻结的局部干扰响应并排除仅频率解释；未分离发出、SMEM 与反压 | [本轮记录](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/issue-placement-interference/20261015-tma-local-interference-heldout-001/README.md)、[评分](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/issue-placement-interference/20261015-tma-local-interference-heldout-001/score.json)；[原同 CTA 计算开发](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/issue-placement-interference/20261015-tma-compute-development-001/README.md)为预测来源 |
| 发出 warp 位置冻结留出（746785） | 固定远端计算背景，issuer 从 warp0 改为 1/2，背景在 warp4–7 轮换，并使用新源环 | 慢位置随 issuer 移动，但未打点周期差约 2.25% 低于冻结下界；三种定量候选均未通过，撤下固定干扰幅度 | [本轮记录](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/issue-placement-interference/20261015-tma-issue-placement-heldout-002/README.md)、[评分](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/issue-placement-interference/20261015-tma-issue-placement-heldout-002/score.json)；[位置开发](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/issue-placement-interference/20261015-tma-issue-placement-development-001/README.md)、[首次绑定失败](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/issue-placement-interference/20261015-tma-issue-placement-heldout-001/README.md)分别为开发与执行历史，非另一次留出 |
| 本地 WGMMA 的 SS/RS 开发对照 | 746721 完成60个开发进程，固定其他131个SM为SS WGMMA；目标SM比较空转、SS、RS三种背景。RS把A改为四个32-bit寄存器，B仍读SMEM；形状均为m64n128k16，每组16条，数值0.0625与FP32累加保持相同。源请求、记录开关和暖机与对应基线配对。 | 八项计时外TMA数值检查及全部背景累加值检查通过，memcheck与synccheck均零错误。正确性明细见preflight-summary.json，资源和SASS见构建目录。原始结果、重复进程与控制分项见分析。归档SHA256：`9d8b43a8ffde0f8add1e97e787af99bda8a38c24cb | [原记录](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/issue-placement-interference/20261015-tma-local-rs-development-001/README.md) |
| 复用 A 寄存器后的 SS/RS 对照 | 746723 完成60个开发进程。条件与上一轮相同，RS的A改为循环前读取四个32-bit寄存器，并在16条指令中复用；采样前的SASS检查确认16条RS指令均使用R88开始的同一组寄存器。 | 八项计时外数值检查、全部背景累加值、memcheck和synccheck通过，见正确性明细。源代码、二进制、逐进程结果和分项保留在returned与分析。归档SHA256：`4dd386585d6e5f8905907dea4b2886ac688c6c91e0b3814c60ae0566ad84507f`。 | [原记录](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/issue-placement-interference/20261015-tma-local-rs-development-002/README.md) |
