# G3 stage 响应与输入信用

4 个目录（10-15）：在真实 G3 上用同一物理 S6 构建改变活动槽数和每 Ktile 计算量。

## 结论

- 746729：同一物理 S6 构建，活动 4 槽相对 6 槽 plain 增加 9.85%；活动 5 槽时第二段 L 只增 0.99%。stage 响应不是编译或 SMEM 分配造成的。
- 746824（冻结判别）：每 Ktile 计算加到 3 遍后，两段的 4 槽−6 槽差与单遍之比为 −0.0015／0.0126。支持有限输入信用（stage 代价是可被计算隐藏的供给延迟），排除固定的、不可隐藏的 stage 附加周期。
- 这是关于完整模型 stage 残差最直接的证据。它与“满载时的有效供给延迟落在 4 槽盖不住、6 槽盖得住的区间”这一解释相容，但原判定只支持有限输入信用，未给出延迟数值。组合模型方向正确，幅度为 3.9%，实测 10.5%。
- 重复计算改造程序另有约 1.25 万 cycle 的额外控制（[gemm-connection](../Q14-composition/gemm-connection.md)），其绝对时间不迁移到原 cfg_b。

## 轮次

按日期排列；全部结果目录见[Q04-sync-slot/g3-stage-credit](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/g3-stage-credit/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| stage反例是否只由编译或SMEM分配造成（746729，开发） | 固定同一S6二进制、215040 B SMEM、描述符与输出，只改变运行时活动槽4/5/6。 | 4槽相对6槽的plain增加9.85%，5槽第二L只增0.99%，编译或分配变化不足，保留缓冲额度与供给／释放交互。 | [干预与结果](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/g3-stage-credit/20261015-g3-active-stages-development-001/README.md)、[有限递推审计](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/g3-stage-credit/20261015-g3-active-stages-development-001/analysis-stage-audit/README.md)、[阶段时钟](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/g3-stage-credit/20261015-g3-active-stages-development-001/phase-clock.json) |
| 计算能否隐藏槽数附加成本（746824，冻结判别） | 固定同一S6布局及每Ktile的TMA、commit、wait、release次数，交叉改变活动4/6槽与1/3遍WGMMA。 | 两段4槽−6槽差值的三遍／单遍比为−0.00150／0.01260，支持有限输入信用、排除固定不可隐藏的stage附加周期。 | [冻结与范围](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/g3-stage-credit/20261015-g3-compute-credit-discrimination-003/README.md)、[结果](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/g3-stage-credit/20261015-g3-compute-credit-discrimination-003/RESULTS.md)、[评分](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/g3-stage-credit/20261015-g3-compute-credit-discrimination-003/score.json)；[746816 alpha审核失败](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/g3-stage-credit/20261015-g3-compute-credit-discrimination-001)、[746817 grid审核失败](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/g3-stage-credit/20261015-g3-compute-credit-discrimination-002)归本轮执行历史。 |
