# 有效频率与阶段频率

9 个目录（10-14、10-15）：频率状态式、空闲历史、双时钟包围与远端负载下的 TMA 时钟。运行前预测见 [frequency-prediction](frequency-prediction.md)，完整时间留出见 [V09](../Q15-full-time/V09-component-validation.md)。

## 结论

**冻结留出获支持**
- 驻留 TMA 服务按固定 SM 周期换算，而不是固定 ns（746665）。

**未通过**
- 驱动状态式与阶段活动式：最大误差 24%–88%。
- 空闲历史使 d044 升频 52%–54%，排除“只依赖目标输入”的候选；d114 重复范围超限。

**现状**
- 频率独立研究按 GOALS 暂停。结论入口为 [FREQUENCY-MECHANISM](../../../model/mechanism/hardware_rules/FREQUENCY-MECHANISM.md)。

## 轮次

按日期排列；全部结果目录见[Q09-frequency/frequency-mechanism](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-mechanism/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| 驻留TMA服务应固定cycle还是ns（746665，关联C2冻结留出） | 固定目标SM的驻留程序、源与目的，在其余131个SM增减WGMMA背景 | 六个主端点支持固定SM周期、排除固定ns，驻留路径与控制按cycle换算 | [冻结与结果](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-mechanism/20261015-tma-clock-scaling-heldout-001/README.md)、[评分](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-mechanism/20261015-tma-clock-scaling-heldout-001/score.json)；[746653远端计算开发](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-mechanism/20261015-tma-remote-compute-development-001)、[开发分析](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-mechanism/20261015-tma-remote-compute-development-001/analysis.json)作为冻结前依据。 |
| 低比值是否仅由读取间隔造成（747169，C2冻结判别） | 固定六个random条件及原程序，只增加clock64前后包围读数，并与原版配对 | d044/d045 S6及d114 S6/S4排除读取间隔单独解释；短K S4重复不足，保留低频标签并研究状态 | [冻结](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-mechanism/20261015-frequency-clock-bracket-002/README.md)、[结果与评分](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-mechanism/20261015-frequency-clock-bracket-002/score.json)；[747125存储耗尽、未测量](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-mechanism/20261015-frequency-clock-bracket-001/RESULTS.md)归本轮执行历史 |
| 相同目标是否依赖此前空闲（747202，C2冻结判别） | 固定d044/d114、S6/S4、random输入及30次graph预热，仅改变目标前空闲0/1/10ms | d044升频52.10%–53.92%，排除目标输入独立候选；d114重复范围超限，信息不足 | [冻结](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-mechanism/20261015-frequency-idle-history-002/README.md)、[结果与评分](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-mechanism/20261015-frequency-idle-history-002/score.json)；[747193误含overlay、采样前停止](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-mechanism/20261015-frequency-idle-history-001/RESULTS.md)归本轮执行历史 |
| 固定驱动状态式能否解释空闲响应（CPU开发） | 复用空闲干预观测，比较一阶、双松弛、阻尼二阶，共享配置参数并按空闲拟合初态 | 三形式均未达逐阶段5%标准，不接入组合，下一形式改用程序活动驱动 | [方程与失败](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-mechanism/20261015-frequency-state-development-001/README.md)、[参数与残差](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-mechanism/20261015-frequency-state-development-001/fit.json) |
| CTA阶段活动能否驱动状态式（CPU开发） | 复用同批观测，加入主循环／输出区间活动比例，并扩大系数范围复核 | S6/S4最大误差仍88.36%/24.07%，不接入服务，下一步区分服务与等待或检验前序调用状态 | [方程与核对](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-mechanism/20261015-frequency-activity-development-001/README.md)、[首搜索](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-mechanism/20261015-frequency-activity-development-001/report.json)、[扩大范围复核](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/frequency-mechanism/20261015-frequency-activity-development-001/extended-report.json) |
| a044 频率迁移（组 00–07，494 进程） | 原冻结预测与门槛，不重拟合；1980 MHz 目标设置 | 开发组有效频率 1.5224–1.8293 GHz，zero 高于 dyadic；纯频率 8 点留出 8/8 通过（最大 3.7966%）；两组 cfg_b 完整时间各 7/8 通过（最大 5.2356%、6.6015%）；S6/S4 阶段频率 6/10 通过（最大 10.7073%）；空闲 1 ms 后阶段频率 +45.69%、plain −10.95%。组 08–09 与 12 项预检见下一行 | [频率报告](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/frequency-stage-report.md) |
| a044 频率尾组（08–09＋12 项预检，10-27） | 原评分器，只换 GPU 身份或视图根目录；不重拟合 | 08：后台 WGMMA 使 TMA CTA 频率下降 9.1%–9.6%，SM 周期变化 ≤1.72%，ns 增加 8.4%–10.6%；“固定SM周期”仍获支持，“固定纳秒”仍被反证。09：TMA CTA=1 时远端计算使频率下降约 9%，周期基本不变；TMA CTA=16 时响应 ≤1.2%，与旧卡一致。预检 12/12 通过原数值检查，memcheck/synccheck 错误为 0 | [频率报告](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/frequency-stage-report.md#08tma时钟缩放留出) |
