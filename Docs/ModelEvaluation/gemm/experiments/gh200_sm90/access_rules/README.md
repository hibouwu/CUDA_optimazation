# GH200 访问规则与 GEMM 时间预测

## 当前结论

**[V09](V09-component-validation.md) 整体未通过。** 同卡30条件全部测完，29个受支持预测为 **4.11% / 35.46%**，另1个保持冻结前拒绝；不能把子集当作整组通过。最大误差来自cfg_b长K有效频率迁移，另有A/B行距响应失配；总时间还存在分项误差抵消，不能单凭4.11%的中位数判断机制准确；分项与关键CTA仍有错误。

此前 **V08 整体未通过。** 同一张 GH200、固定 CUTLASS cfg_a/b/c 配置的 36 个新留出条件，总时间绝对相对误差中位数 **2.61%**、最大 **35.34%**，未达到 5% / 10% 目标。失败主要涉及 A/B 行距未按 128 B 对齐、swizzle=8 的整 cluster 补齐 tile，以及 cfg_b 大足迹长 K。跨卡尚未验证。

测后按 swizzle=1、A/B 行距为 128 B 倍数、K ≤ 16384 限定出的 21 个条件为 **1.90% / 6.04%**；这是完整留出集的子集分析，不是独立验证，也不代表整轮通过。逐例结果及后续对照见 [V08](V08-wider-validation.md)。

[V07](V07-rule-validation.md) 在同卡、cfg_a/b/c、swizzle=1、K 位于校准范围内的 24 个条件（12 个不同 M×N）上为 **3.06% / 8.71%**，总时间通过；供给、末次输出和关键 CTA 定位仍失配，20/24 预测偏长。它使用 V06 事件递推、同卡重新校准与 R18 边界修正；R19 只作诊断，R02/R11/R12/R16/B01 未用于 V07。

现行V09候选的公式、输入与限制维护在 [MODEL-CANDIDATES](../../../model/semi_empirical/MODEL-CANDIDATES.md#v09-model)；[RULES](RULES.md)只收通过独立留出的规则。[离线消融](V01-validation.md#v09-ablation)暂不支持整体退回聚合模型，但不覆盖行距、边界和分项失败。两项后续测量已结束：主线只完成参照，必要长K开发数据未采到；m11 随后已完成两次冻结留出，均未通过；当前候选与评分入口见[M11-SMEM](../../../model/mechanism/hardware_rules/M11-SMEM.md)。[一页计划](PLAN.md)列出当前结论和开放问题；任务细节见[主线](../../../model/semi_empirical/MAINLINE-LONG-K.md)、[m11](../../../model/mechanism/hardware_rules/M11-SMEM.md)，已执行的范围与预算并入各自任务文件。实际分工与状态见 [EXECUTION](EXECUTION.md#status)，V09原采样和失败评分保持。

<a id="当前状态2026-10-07"></a>

## 实测与验证索引

按模型阅读与文档职责见[模型入口](../../../model/README.md)。类别是本页的主要用途，“供谁使用”可跨类；V 页是实测评分记录，模型定义另在 MODEL-CANDIDATES。

“已测”表示有测量记录；预测是否通过以及可用范围另列。误差均为绝对相对误差中位数 / 最大值；各轮目标和测试集不同，不直接据此比较模型优劣。

| 实验 | 类别 | 供谁使用 | 状态 | 主要结果与限制 |
|---|---|---|---|---|
| [R00](R00-anchor-target.md) GEMM 锚点 | 实测 | 半经验、机制-程序/硬件 | 已测 | 所测 WGMMA 形状达到指令峰值；建立完整 GEMM 基线 |
| [R01](R01-readiness.md) 依赖与就绪 | 实测 | 半经验、机制-硬件 | 已测 | 依赖链与独立流服务；窗口含地址、消费和循环成本 |
| [R02](R02-register-service.md) 寄存器服务 | 实测 | 半经验、机制-硬件 | 已测，未用于 V07 | 固定 40 寄存器后轮换源慢 5.26%；RF 端口未归因 |
| [R03](R03-access-demand.md) 片上访问 | 实测 | 半经验、机制-硬件 | 已测 | 8 warp 的所测 16 B SMEM/ldmatrix 服务约 124–127 B/cycle |
| [R04](R11-mixed-issue.md#r04) 联合服务 | 实测 | 半经验、机制-硬件 | 已并入 R11 | WGMMA/FFMA 接近重叠；LDS 配对只作诊断 |
| [R05](R05-async-lifecycle.md) 异步生命周期 | 实测 | 半经验、机制-硬件 | 已测，范围受限 | 等待、stage 与复用服务；简单探针的流水重叠受限 |
| [R06](R06-issue-residency.md) 发射与驻留 | 实测 | 半经验、机制-硬件 | 已测 | FFMA 接近发射上限；未得到统一驻留惩罚 |
| R07 频率与固定项 | 实测 | 半经验、机制-硬件 | 已并入 [R00](R00-anchor-target.md#ndebug)、[R09](R09-inkernel-clock-stages.md#r07) | NDEBUG 锚点；调用后探针不能代替调用内频率 |
| [R08](R08-waves-l2-reuse.md) 波次与遍历 | 实测 | 半经验、机制-程序/硬件 | 已测 | 所列形状按离散波次计费；遍历收益依条件变化 |
| [R09](R09-inkernel-clock-stages.md) 调用内分段 | 实测 | 半经验、机制-硬件 | 已测，输入扩展已完成 | 周期与时间须分开；三档输入已完成长窗口同卡对照，频率依赖输入数值 |
| [R10](R10-layout-cache.md) 行距与尺寸尾部 | 实测 | 半经验、机制-硬件 | 已测，局部覆盖候选未通过 | 匹配容量后 A/B 联合行距代价仍存在；五档 B 行距的冻结留出否定了本批两个简单覆盖候选 |
| [R11](R11-mixed-issue.md) 混合发射 | 实测 | 半经验、机制-硬件 | 已测，未用于 V07 | 消费者匹配后，各配对的重叠程度不同 |
| [R12](R12-smem-path-contention.md) SMEM 多路竞争 | 实测 | 半经验、机制-硬件 | 已测，未用于 V07 | TMA/WGMMA 接近重叠；与 STS 并发有额外代价 |
| [R13](R13-async-retirement.md) 供给与退役 | 实测 | 半经验、机制-硬件 | 已测，条件候选待转移 | 行距代价随活动规模和输出位置变化；有效地址/填零候选已进入开发递推；尚无新的完整留出成绩 |
| [R14](R14-stage-handoff.md) 阶段交接 | 实测 | 半经验、机制-程序/硬件 | 已测，迁移受限 | 四 CTA 校准未能直接迁移供给与交接到整卡 |
| [R15](R15-output-service.md) 输出服务 | 实测 | 半经验、机制-程序/硬件 | 已测，条件对照 | 标量探针不能直接作为 CUTLASS 输出常数 |
| [R16](R16-residency-quota.md) 驻留与配额 | 实测 | 半经验、机制-硬件 | 已测，未用于 V07 | 驻留收益依供给变化；寄存器申请等待取决于释放时序 |
| [R17](R18-cluster-boundary.md#r17) 越界 tile | 实测 | 半经验、机制-程序/硬件 | 已并入 R18 | 改用匹配位置的窗口增量；旧两例修正只作诊断 |
| [R18](R18-cluster-boundary.md) cluster 边界 | 实测 | 半经验、机制-程序/硬件 | 旧规则用于V07；新配对已接开发模型 | cfg_b 同逻辑输出下，有效地址零比 M/N 向 TMA 越界填零快约17%–20%；分项仍随位置变化 |
| [R19](R19-critical-cta-tail.md) 关键 CTA 与尾部 | 实测 | 半经验、机制-程序/硬件 | 已测，只作诊断 | swizzle 改变工作与尾部；V07 尾差项选择 none |
| [B01](B01-bandwidth-cache.md) 带宽与缓存 | 实测 | 半经验、机制-硬件 | 已测，未用于 V07 | 条件请求速率；不等同物理 HBM 流量或缓存收益 |
| [V01](V01-validation.md) 组合验证 | 实测 | 半经验（C1/C3）；分项供机制诊断 | 部分通过 | 单 CTA 通过；整卡与完整 kernel 未通过 |
| [V02](V02-kernel-prediction.md) 完整 kernel | 实测 | 半经验（C1/C3）；分项供机制诊断 | 初版目标通过 | 11 个新尺寸为 6.0% / 10.3%，按当时 10% / 20% 目标 |
| [V03](V03-clock-rule.md) 频率规则 | 实测 | 半经验（C1/C3）；分项供机制诊断 | 同卡预测通过 | 11 个新尺寸为 2.6% / 5.0%；跨卡常数有差异 |
| [V04](V04-config-transfer.md) 配置迁移 | 实测 | 半经验（C1/C3）；分项供机制诊断 | 部分通过 | cfg_c 迁移通过；cfg_a/b 仍失配 |
| [V05](V05-rule-transfer.md) 规则迁移 | 实测 | 半经验（C1/C3）；分项供机制诊断 | 未通过 | 18 例为 10.7% / 31.9%；部分打点不作定量判定 |
| [V06](V06-revised-transfer.md) 递推修正 | 实测 | 半经验（C1/C3）；分项供机制诊断 | 未通过 | 18 个留出条件为 5.25% / 19.0%；边界与尾部未解决 |
| [V07](V07-rule-validation.md) 边界迁移 | 实测 | 半经验（C1/C3）；分项供机制诊断 | 总时间通过，分项受限 | 24 条件为 3.06% / 8.71%；范围与分项限制见上 |
| [V08](V08-wider-validation.md) 更宽留出 | 实测 | 半经验（C1/C3）；分项供机制诊断 | 整体未通过 | 36 条件为 2.61% / 35.34%；21 条件子集见上 |
| [V09](V09-component-validation.md) 组件组合 | 实测 | 半经验（C1/C3）；分项供机制诊断 | 整体未通过 | 30条件实测；29预测为4.11%/35.46%，另1拒绝 |

## 模型、机制研究与公共入口

| 文档 | 类别 | 职责与供谁使用 | 状态入口 |
|---|---|---|---|
| [规格计算](../../../model/spec/spec_baseline.md) | 规格 | 理想参照；供 C1-01 的规格比较与 C3 差距分析 | 独立可运行计算器 |
| [MODEL-CANDIDATES](../../../model/semi_empirical/MODEL-CANDIDATES.md) | 半经验 | 当前公式、版本与参数绑定；供预测器 | 各版评分另见 V 页 |
| [RULES](RULES.md) | 半经验 | 采用资格入口，引用限定范围内的成绩 | 当前无完整联合资格规则 |
| [CUTLASS 程序图](../../../model/mechanism/program/cutlass_program_graph.md) | 机制-程序 | 调度、依赖、槽代次；供机制组合 | 已整理源码对应，整图验收未完成 |
| [TMA-HISTORY-REUSE](../../../model/mechanism/hardware_rules/TMA-HISTORY-REUSE.md) | 机制-硬件 | 历史证据盘点；供 TMA 研究选点 | 复用与缺口清单 |
| [TMA-INPUT-EVENT-MODEL](../../../model/mechanism/hardware_rules/TMA-INPUT-EVENT-MODEL.md) | 机制-硬件 | 单 CTA 输入路径的可执行候选，含探针图 | 离线候选；不代表 CUTLASS 图 |
| [TMA-S16-STAGE-COST](../../../model/mechanism/hardware_rules/TMA-S16-STAGE-COST.md) | 机制-硬件 | 控制成本与硬件成本区分；供 TMA 接入判断 | 限额判别设计 |
| [TMA-SHARED-SERVICE](../../../model/mechanism/hardware_rules/TMA-SHARED-SERVICE.md) | 机制-硬件 | 实际并发 CTA×请求组织；供 GEMM 供给组合 | 主线设计 |
| [TMA-GEMM-HYPOTHESES](../../../model/mechanism/hardware_rules/TMA-GEMM-HYPOTHESES.md) | 机制-硬件 | 当前硬件假设、已完成干预与反例 | 开发候选及排除范围 |
| [FREQUENCY-MECHANISM](../../../model/mechanism/hardware_rules/FREQUENCY-MECHANISM.md) | 机制-硬件 | C2 机制辨识，同时供半经验频率预测开发 | 与 TMA 并行 |
| [M11-SMEM](../../../model/mechanism/hardware_rules/M11-SMEM.md) | 机制-硬件 | LDS/STS 当前候选及冻结判定 | 两次冻结均未通过 |
| [其他硬件路径](../../../model/mechanism/hardware_rules/README.md) | 机制-硬件 | RF、联合服务、驻留与配额 | 储备；原始实测仍在 R 页 |
| [MICROARCH-REVIEW](../../../model/mechanism/hardware_rules/MICROARCH-REVIEW.md) | 机制-硬件 | 证据审阅与机制选题储备 | 不等于已立项或已通过 |
| [MAINLINE-LONG-K](../../../model/semi_empirical/MAINLINE-LONG-K.md) | 半经验 | 长 K 开发任务与未采数据；供条件预测 | 该轮测量已结束，任务页保留缺项 |
| [C1-01](C1-01-selection.md) | 实测 | 三类比较对象的 C1/C3 协议、冻结与评分入口 | 依该任务原执行安排 |
| [RESEARCH-GUIDE](RESEARCH-GUIDE.md) | 实测 | 采样、预测边界与评分协议，供所有模型 | 公共细则 |

计划、分工与启动记录不是额外模型类别：[PLAN](PLAN.md)保存实验进度，[EXECUTION](EXECUTION.md)保存协作记录，10-11 两个有界目标的[启动记录](../../gh200_sm90_archive/access_rules/PARALLEL-START-20261011.md)已归档。总推进顺序以[项目 PLAN](../../../PLAN.md)为准。

已合并页面的原路径继续保留：[R04](R04-joint-service.md)、[R07](R07-anchor-clock-fixedcost.md)、[R17](R17-oob-tile.md)；上述索引指向其当前证据位置。

## 数据与代码

- [运行、拟合与分析代码](../../../../../../microbench/gh200_resource_campaign/access_rules/)：预测入口为v09_model，供给/输出/时钟是独立纯模块；原分析CLI保留复现用途。
- 两组R18方向配对统一由[r18_cases.py](../../../../../../microbench/gh200_resource_campaign/access_rules/r18_cases.py)生成（`--family first-fill|input-map --axis m|n`）；四份已有JSON作为历史展开快照保留，`--check`核对全部字段。
- [原始结果归档](../../../../../../results/gh200_resource_campaign/access_rules/)按 run-id 保存源码、构建、环境、样本与分析；冻结预测及绑定记录见各 V 页。
- [规则说明](RULES.md)保留计量约定、手算例子与 2026-10-08 一次性离线复现记录。
- R09、R10、R13 的逐问题推导与证据在 [reanalysis/20261012-consolidation](reanalysis/20261012-consolidation/)，由对应 R 页逐条链接，是这三页的附录，不另立结论。
- [历史过程文件](../../gh200_sm90_archive/access_rules/)保留原样；不作为当前执行要求。
