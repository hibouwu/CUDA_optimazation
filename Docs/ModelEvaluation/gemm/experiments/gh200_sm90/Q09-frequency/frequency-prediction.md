# 运行前频率预测

7 个目录（10-14、10-15）：静态频率预测器的开发与冻结留出、阶段频率模型及其在 C1 宽范围条件上的扩展，以及 G3 阶段频率复测。由 frequency-mechanism 移入。

## 结论

- 静态频率预测迁移到新几何：8/8 通过，最大频率误差 4.518%（746491）。饱和响应式在完整时间留出中采用，见 [V09](../Q15-full-time/V09-component-validation.md)。
- 阶段频率模型：新几何与 stage 的冻结留出 8/10 端点通过，整轮失败（746814）；扩到 C1 宽范围条件后最大误差仍 40.25%，未接入正式预测。
- G3 S4 慢 12.10% 时阶段频率反而更高，新增时间需要新增周期来解释（746679）。

## 轮次

按日期排列；全部结果目录见[Q09-frequency/frequency-prediction](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-prediction/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| 活动SM与输入是否影响频率（746447，GPU开发） | 固定cfg_b原生程序、M/N及协议，改变K、请求SM66/132与dyadic/zero输入 | 活动SM与输入均影响有效频率，纳入运行前预测器并进入冻结 | [开发](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-prediction/20261014-frequency-development-001/README.md)、[筛选](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-prediction/20261014-frequency-development-001/offline-static-screen-v2.json) |
| 静态频率预测能否迁移新几何（746491，首次冻结留出） | 固定频率模型，在两个新几何的不同SM数与dyadic/zero输入上检验 | 8/8通过，最大频率误差4.518%，采用冻结范围内的频率模块 | [冻结与范围](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-prediction/20261015-frequency-heldout-001/README.md)、[评分](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-prediction/20261015-frequency-heldout-001/score.json) |
| 活动描述能否解释首次C3失败（CPU开发） | 复用已观察数据，比较RBF与总工作量／工作列表利用率的饱和响应式 | 按分组RMS选饱和响应式，交给新条件独立验证 | [开发](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-prediction/20261015-frequency-development-002/README.md)、[筛选](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-prediction/20261015-frequency-development-002/screen.json) |
| stage反例是否由阶段频率解释（746679，已知条件复测） | 固定G3形状、输入、原冻结二进制与参考卡，比较S6/S4 | 本批S4慢12.10%但阶段频率更高，新增时间仍需新增周期解释；原评分保留 | [复测](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-prediction/20261015-g3-phase-reproduction-001/README.md)、[阶段分析](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-prediction/20261015-g3-phase-reproduction-001/phase-clock.json) |
| 静态输出进度能否预测阶段频率（CPU开发／后验测试） | 固定静态问题与工作列表，增加输出序号、进度及活动特征，比较全调用与阶段频率 | 后验最大误差3.045%，但geometry分组最大18.692%；作为开发经验替代，另取新几何冻结验证 | [开发与范围](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-prediction/20261015-phase-frequency-development-001/README.md)、[开发／后验评分](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-prediction/20261015-phase-frequency-development-001/fit-001/report.json) |
| 阶段模型能否迁移新几何与stage（746814，冻结留出） | 固定阶段模型与逐阶段5%标准，检验新两／三输出几何的S6/S4 | 8/10端点通过，整轮失败；S4三输出前两阶段超标，下一版检查活动与频率联立 | [冻结](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-prediction/20261015-phase-frequency-heldout-001/README.md)、[结果与评分](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-prediction/20261015-phase-frequency-heldout-001/score.json) |
| 阶段状态能否扩展到宽范围C1条件（CPU开发） | 复用C1开发数据，同一学习器按条件分组，比较静态字段与新增推进／活动特征 | 阶段版改善误差，但最大仍40.25%，未接入正式预测 | [标签](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-prediction/20261015-phase-frequency-development-002/README.md)、[开发比较](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-prediction/20261015-phase-frequency-development-002/fit-001/README.md)、[评分](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-prediction/20261015-phase-frequency-development-002/fit-001/report.json) |
| 低比值是否服从共同周期（CPU诊断） | 固定d044/d045/d114既有记录，比较实测时间与输出序号轴上的周期解释 | 未得到可采用的共同周期，下一步先检验双时钟读取间隔 | [时间轴诊断](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-prediction/20261015-phase-frequency-development-002/temporal-001/README.md)、[结果](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-prediction/20261015-phase-frequency-development-002/temporal-001/report.json) |
