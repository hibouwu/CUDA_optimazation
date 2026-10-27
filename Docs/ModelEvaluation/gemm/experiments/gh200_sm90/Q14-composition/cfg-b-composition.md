# cfg_b 同版组合

19 个目录（10-11 至 10-20）：状态化源、异步接口、原 12 条件回放、经验前缀、同版系统与九方向交付包。

## 结论

**现行默认（10-16 前缀版）**
- 原 12 条件的 plain 绝对误差中位 10.225%、最大 25.422%。
- stage 响应预测 3.903%，实测 10.467%。
- A16预测+0.0935%，B80、B112预测0%，实测分别为+28.9%、+18.1%、+18.6%。
- G1/G2/G4 等短条件误差 −11%～−20%。
- 其余长条件误差在 6% 以内。

**G3 S6 主循环已接近计算受限**
- 746824 原始程序的 dual 记录中，主循环（FIRST_MMA→MAIN_END）为 33144／33352 cycle，纯计算 32768 cycle；组合模型在这里误差为 first_L −1.2%、later_L −0.7%。剩余误差在满载供给（stage、行距）与启动段。

**排除的固定形式**
- 双粒度源信用、按请求大小分道、固定生产者延迟、固定轮转额度、统一提价、地址无关的回传信用。

**747595 源重排**
- Q1 主周期响应 +24.8%，三类预测均失配，Q4 控制不成立，信息不足。

<a id="跨cta指标的可复现性"></a>
**跨 CTA 指标的可复现性**
- “最晚 CTA 命中 1/120”不适合作失败指标：每个条件重复 10 次，实测最晚 CTA 有 3–8 个不同身份；总猜各条件出现最多的那个，上限也只有 45/120。
- 模型选出的关键 CTA，在实测中的跨度比真正最晚者中位只短 1.65%，90% 在 3.85% 以内（由 full-matrix 的 comparison-final.json 计算）。
- 跨 CTA 推进应改用跨度短缺或结束时刻分布评价。

**入口**
- 组合入口与模块资格见[组合](../../../model/mechanism/composition.md)。

## 轮次

按日期排列；全部结果目录见[Q14-composition/cfg-b-composition](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| V09冻结观测联合审计：完整时间或宽关键CTA并列集合命中，能否代替分项与真实最后完成者验收？ | 冻结模型/29支持+1拒绝保持；复放300 dual分项，并扩展到同30条件四变体1200旧进程，不新增GPU或阈值。 | 有符号聚合会掩盖局部误差，cfg_c partial/sw8关键CTA集合0/10命中；cfg_b两项plain失败超本批波动，原V09失败保持。 | [原结果目录](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261011-V09-gate-audit)、[原页判定](../Q15-full-time/V09-component-validation.md#总时间中的误差抵消测后复核) |
| 固定源多重集合的重排冻结判别（747595） | 保持源全局多重集合、分配、请求数和暖机，只改变 CTA→源映射，Q1 主对照、Q4 控制 | 主周期响应 +24.817%，三类预测均失配；Q4 控制也不成立，正式结论为信息不足，不选择共享范围或截止规则 | [本轮结果](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-source-scope-discrimination-001/RESULTS.md)、[冻结评分](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-source-scope-discrimination-001/score.json)、[测后等待／跨度分解](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-source-scope-discrimination-001/diagnosis.json) |
| 双粒度源信用候选（CPU 筛查） | 固定规则与价格，只比较 2048/4096/8192 条行信用，先回放旧必过 `q1-skew8` | 三容量均违反原 ns/cycle 门槛，停止该结构，不进入 GEMM | [本轮结果](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-dual-granule-service-001/RESULTS.md)、[逐条件状态](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-dual-granule-service-001/summary.json)、[绑定与复现入口](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-dual-granule-service-001/README.md) |
| 按请求大小分道（CPU 筛查） | 固定两条 GPC 前沿、4096 行信用及原价，只比较源就绪／最后目的完成归还信用 | 两版本在首项均预测 403.955 ns、实测 512 ns，违反原门槛；无信用等待，归还时点未区分，停止两版 | [本轮结果](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-size-lanes-service-001/RESULTS.md)、[逐条件状态](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-size-lanes-service-001/summary.json)、[执行前绑定](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-size-lanes-service-001/binding.json) |
| 状态化源服务能否联合解释G3两种stage（CPU开发回放） | 固定原状态价格、阶段频率与220cycle复用候选，把G3 S6/S4请求接回共同程序并对照746679 | stage变慢预测4.04%、实测12.10%，第二MAIN跨度仅30.45/46.01ns；将CTA空间推进确定为后续约束 | [组合与逐端点对照](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-combination-regression-001/README.md)、[内部跨度诊断](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-combination-regression-001/output/internal-diagnosis.json) |
| 计算活跃期的固定生产者延迟能否补足反例（CPU筛查） | 固定源、目的、计算和频率，只比较每条活跃期发出增加40／80cycle，S6/S4共用价格 | S4平均L改善但第二MAIN跨度仍约51ns，S6仍30.45ns；两版本均停止，不接入组合 | [候选、反例与复现](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-local-producer-candidate-001/README.md)、[40cycle报告](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-local-producer-candidate-001/delay40/report.json)、[80cycle报告](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-local-producer-candidate-001/delay80/report.json) |
| 现有状态化服务是否产生源关系下的CTA空间差异（CPU开发回放） | 复用746797独立／共享源与Q1/Q4记录，固定旧价格并将组合的状态化源接入探针，另筛查K起步相位 | 暖数据接法退化为旧GPC服务，独立源跨度519.82/352.68ns仍低于2576/3792ns；相位也分不开候选，转查驻留竞争 | [空间与相位复核](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-sharing-spatial-audit-001)、[状态化探针接线与结果](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-stateful-probe-replay-001/README.md)、[逐端点回放](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-stateful-probe-replay-001/output/report.json) |
| 固定轮转32KiB额度能否解释等待差（CPU筛查） | 固定GPC两通道、原服务价及四种源关系×Q1/Q4，只新增按CTA固定轮转且不借空窗的32KiB额度 | 独立源跨度不足且共享源时间被破坏；停止该具体轮转形式及其额度／相位扩扫 | [等待分解与候选](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-resident-arbitration-001/README.md)、[候选筛查](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-resident-arbitration-001/screen.json) |
| 提高统一请求单价能否保留旧约束并补足跨度（CPU筛查） | 固定共享规则及其他价格，只把每16KiB服务价乘1／1.25／1.5／1.75／2，同时检查旧留出与新源关系 | 五价格均未补足跨度，提高价格又破坏原局部条件；停止单价扩扫 | [联合价格检查](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-service-price-check-001/README.md)、[逐条件报告](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-service-price-check-001/report.json) |
| 同一状态化版本能否覆盖原12条件的stage与行距响应（CPU开发回放） | 固定同一规则和价格，按实际工作列表、Ktile与行距扩至原12条件，统一对照739011 | plain绝对误差中位17.378%、最大39.797%；B行距的候选可见输入相同，地址无关规则仍给零响应，后续须让地址参与服务 | [全矩阵与执行恢复](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-full-matrix-001/README.md)、[三类模型与端点对照](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-full-matrix-001/comparison-final.md)；[共同程序地址适配与原G3逐位回归](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-program-source-adapter-001)为接入产物 |
| 逐目的完成归还源信用能否解决现有反例（CPU结构筛查） | 保留两条读取通道及原价，候选新增四条回传通道和8／16／32源信用，先检查实际G5输入同一性与完成接口 | 地址无关输入仍对四行距生成同一轨迹，取消拟议的78次扫描；完成回调接通后保留原同步预测，作为后续接入能力 | [候选与停止依据](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-return-credit-candidate-001/README.md)、[异步源／目的完成接口](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-async-source-interface-001)、[接口与原同步结果核对](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261015-cfg-b-async-source-interface-001/verification-final.json) |
| 固定入口前缀能否改善短条件（CPU接入） | 固定全部服务与映射，只在 entry→producer_first_work 加入 V09 冻结的 1608 ns。 | 原12条件 plain 误差中位／最大由17.378%／39.797%降至10.225%／25.422%，采用该经验前缀，最晚CTA命中仍为1/120。 | [接入与范围](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261016-cfg-b-prefix-001/README.md)、[同条件结果](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261016-cfg-b-prefix-001/RESULTS.md)、[逐进程比较](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261016-cfg-b-prefix-001/comparison.json) |
| 当前局部接口能否固化为同版完整系统 | 当前源码、原profile与协议；G3 S6／S4／A16三条件33调用，其他九项复用保存评分 | 396个CTA、152064个请求、772596个节点时刻及plain逐值保持；交付55份源码和12份输入，54项CPU测试通过；原10.225%／25.422%与C2/C3失败保持 | [当前版本与同条件评分](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261020-Q14-current-system-001/README.md) |
