# GH200 访问规则与 GEMM 时间预测

## 当前结论

更新：2026-10-28。

- **完整 cfg_b 机制组合未通过。** 现行默认为 10-16 前缀版，原12条件 plain 绝对误差中位 10.225%、最大 25.422%。误差集中在行距响应（A16／B80／B112 实测 +18%～+29%，模型 0）、stage 响应（预测 3.9%，实测 10.5%）和短 GEMM 启动段；G3 S6 主循环已接近计算受限。见 [cfg_b 组合](../Q14-composition/cfg-b-composition.md)。
- **半经验 [V09](../Q15-full-time/V09-component-validation.md) 整体未通过**（29 个受支持预测 4.11%／35.46%）；[V07](../Q15-full-time/V07-rule-validation.md) 在限定范围内总时间通过（3.06%／8.71%）。现行公式见 [MODEL-CANDIDATES](../../../model/semi_empirical/MODEL-CANDIDATES.md#v09-model)。
- **获有限支持的局部规则：** 单 CTA 计算—等待—释放递推（冻结外推 ≤1.5%）、每 GPC 两槽共享服务（3.64%）、运行前频率模块（限定范围 C3 2.96%）、普通 L1 快复用（限定条件）。都不能单独改善完整模型。
- **当前缺口：** 满载多 CTA 下的源服务，包括 stage 延迟、行距与 CTA 结束时刻分布。逐 SM 端点在相同运行间不可复现，跨 CTA 推进应按分布评价。
- **a044 重测按原资格逐轮复核。** 前 11 个资源家族的 3,951 次调用通过原审计，378 个原发布条件满足原统计门槛；部分局部机制结论因新卡上的观察扰动、暖机或配对资格不足转为信息不足。1980 MHz 是设置目标，周期与时间换算仍使用实际执行窗口的有效频率。见 [a044 阶段说明](../Q10-time-mapping/A044-REMEASUREMENT.md)和[机制原评分复核](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/mechanism-repair-r2-report.md)。
- **V09 a044 的 34 条件子集未达标。** 36 个声明条件中，34 个取得四变体完整观测；其中 27 个有数值预测的留出条件误差中位／最大为 5.01%／42.6%，超过原 5%／10% 门槛。两个 G1_base 条件未取得完整观测，本次不作原 36 条件整体验收。见[子集评分](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/v09-complete-subset-report.md)。
- **C1 a044 已取得全部 30 个开发块，模型比较、冻结与 120 条件独立留出尚未执行。** 360 条件中 359 个可用、356 个观测完整。两卡共同完整的 304 条件中，242 个最快候选相同；沿用旧卡实测最快候选的 regret 中位为 0、最大为 8.1%。这是卡与频率同时变化后的开发数据和实测排序对照。见 [C1 开发结果](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261023-Q16-a044-development-001/RESULTS.md)。

## 实测与验证索引

按模型阅读与文档职责见[模型入口](../../../model/README.md)。类别是本页的主要用途，“供谁使用”可跨类；V 页是实测评分记录，模型定义另在 MODEL-CANDIDATES。

“已测”表示有测量记录；预测是否通过以及可用范围另列。误差均为绝对相对误差中位数 / 最大值；各轮目标和测试集不同，不直接据此比较模型优劣。

| 实验 | 类别 | 供谁使用 | 状态 | 主要结果与限制 |
|---|---|---|---|---|
| [L2-01](../Q03-source-service/L2-01-saturation.md) 并发服务曲线 | 实测 | 机制-硬件 | 开发合格，六形式按预置单调反例停止 | 无参数拟合／冻结，4留出未采；实际活动、目标机器码及完整回传已核对 |
| [COMPUTE-01](../Q14-composition/COMPUTE-01-resident-chain.md) cfg_b短链A–E | 实测 | 机制-硬件 | A长度留出通过；B/C具体形式停止；D/E开发数值与路径完成 | A的q512局部支持；B/C、D/E端点仅解释trace程序，无新组合价格或全GEMM独立验收 |
| [SYNC-01](../Q04-sync-slot/SYNC-01-replay.md) 同步观测回放 | 实测 | 机制-硬件 | 旧证据重算及CPU接线完成 | 46条件／138进程／4416轮与原分析一致，21项CPU回归通过；后续B/C、D路径与零控制形式反例见COMPUTE-01，独立返回价格未辨识 |
| [TRANSFER-01](../Q02-tma-request/TRANSFER-01-replay.md) 异步搬运回放 | 实测 | 机制-硬件 | 旧证据重算及CPU接线完成 | tensor168份raw、S16的120发布样本、cp.async120份raw，原资格保留；63项CPU检查通过；D原生TMA开发数值／路径已核对，服务价格迁移未辨识 |
| [HBM-01](../Q03-source-service/HBM-01-replay.md) 外存联合窗口与共享服务 | 实测 | 机制-硬件、半经验对照 | 旧18条件／180进程回放；固定1:1两组织6进程完成 | +4.145%在预定5%内，保留无组织修正工作假设；机制信息不足 |
| [SMEM-01](../Q07-smem/SMEM-01-replay.md) 完整SMEM需求与旧回放；[准备完成后的联合窗口](../../../../../../results/gh200_resource_campaign/access_rules/Q07-smem/input-output-joint/20261018-Q07-prepared-joint-001/EXECUTION.md)；[bank排列修复](../../../../../../results/gh200_resource_campaign/access_rules/Q07-smem/input-output-joint/20261018-Q07-bank-layout-001/EXECUTION.md)；[输出相位判别](../../../../../../results/gh200_resource_campaign/access_rules/Q07-smem/input-output-joint/20261018-Q07-output-phase-001/EXECUTION.md)（753514桥接失败，信息不足收口） | 实测 | 机制-程序、机制-硬件 | 768份旧记录与实际地址核对，真实四组织12进程完成 | 并发优于串行但max失配，计时含共有fragment准备，保留联合窗口；原两留出失败保持 |
| [REG-01](../Q08-register-residency/REG-01-replay.md) 寄存器条件与启动配额 | 实测 | 机制-程序、机制-硬件 | 539份R01/R02/R16旧记录、S4/S6实际资源与12warp配额 | 条件窗口保持，软件依赖与资源池分别接入，时间未新增定价 |
| [L1-01](../Q13-l1-lsu/L1-01-replay.md) 普通LSU与缓存探针 | 实测 | 机制-程序、机制-硬件 | 117份旧记录与cfg_b普通访存分支核对，原12开发及8留出完成 | cfg_b有效普通global／local需求为0；8留出支持限定group服务响应，实际carveout未知，方向收口 |
| [SM-01](../Q08-register-residency/SM-01-replay.md) 发射联合窗口与驻留 | 实测 | 机制-程序、机制-硬件 | 267份旧记录回放、174份R12复用，容量及模型活动接线 | cfg_b S4/S6均1 CTA/SM；36份原条件联合合同，目标相位价格待测 |
| [COMPOSITION-01](../Q14-composition/COMPOSITION-01-replay.md) 同版系统交付 | 实测 | 三类模型对照、机制组合 | 三条件请求级CPU精确回归、原12开发评分复核，67项CPU检查通过 | 完整输入→时间线→plain可重放；后续模块观测未改原12行误差10.225%／25.422%，独立C2/C3未通过 |
| [R00](../Q05-wgmma/R00-anchor-target.md) GEMM 锚点 | 实测 | 半经验、机制-程序/硬件 | 已测 | 所测 WGMMA 形状达到指令峰值；建立完整 GEMM 基线 |
| [R01](../Q08-register-residency/R01-readiness.md) 依赖与就绪 | 实测 | 半经验、机制-硬件 | 已测 | 依赖链与独立流服务；窗口含地址、消费和循环成本 |
| [R02](../Q08-register-residency/R02-register-service.md) 寄存器服务 | 实测 | 半经验、机制-硬件 | 已测，未用于 V07 | 固定 40 寄存器后轮换源慢 5.26%；RF 端口未归因 |
| [R03](../Q07-smem/R03-access-demand.md) 片上访问 | 实测 | 半经验、机制-硬件 | 已测 | 8 warp 的所测 16 B SMEM/ldmatrix 服务约 124–127 B/cycle |
| [R04](../Q07-smem/R11-mixed-issue.md#r04) 联合服务 | 实测 | 半经验、机制-硬件 | 已并入 R11 | WGMMA/FFMA 接近重叠；LDS 配对只作诊断 |
| [R05](../Q02-tma-request/R05-async-lifecycle.md) 异步生命周期 | 实测 | 半经验、机制-硬件 | 已测，范围受限 | 等待、stage 与复用服务；简单探针的流水重叠受限 |
| [R06](../Q08-register-residency/R06-issue-residency.md) 发射与驻留 | 实测 | 半经验、机制-硬件 | 已测 | FFMA 接近发射上限；未得到统一驻留惩罚 |
| R07 频率与固定项 | 实测 | 半经验、机制-硬件 | 已并入 [R00](../Q05-wgmma/R00-anchor-target.md#ndebug)、[R09](../Q09-frequency/R09-inkernel-clock-stages.md#r07) | NDEBUG 锚点；调用后探针不能代替调用内频率 |
| [R08](../Q01-scheduling/R08-waves-l2-reuse.md) 波次与遍历 | 实测 | 半经验、机制-程序/硬件 | 已测 | 所列形状按离散波次计费；遍历收益依条件变化 |
| [R09](../Q09-frequency/R09-inkernel-clock-stages.md) 调用内分段 | 实测 | 半经验、机制-硬件 | 已测，输入扩展已完成 | 周期与时间须分开；三档输入已完成长窗口同卡对照，频率依赖输入数值 |
| [R10](../Q03-source-service/R10-layout-cache.md) 行距与尺寸尾部 | 实测 | 半经验、机制-硬件 | 已测，局部覆盖候选未通过 | 匹配容量后 A/B 联合行距代价仍存在；五档 B 行距的冻结留出否定了本批两个简单覆盖候选 |
| [R11](../Q07-smem/R11-mixed-issue.md) 混合发射 | 实测 | 半经验、机制-硬件 | 已测，未用于 V07 | 消费者匹配后，各配对的重叠程度不同 |
| [R12](../Q07-smem/R12-smem-path-contention.md) SMEM 多路竞争 | 实测 | 半经验、机制-硬件 | 已测，未用于 V07 | TMA/WGMMA 接近重叠；与 STS 并发有额外代价 |
| [R13](../Q02-tma-request/R13-async-retirement.md) 供给与退役 | 实测 | 半经验、机制-硬件 | 已测，条件候选待转移 | 行距代价随活动规模和输出位置变化；有效地址/填零候选已进入开发递推；尚无新的完整留出成绩 |
| [R14](../Q06-handoff-output/R14-stage-handoff.md) 阶段交接 | 实测 | 半经验、机制-程序/硬件 | 已测，迁移受限 | 四 CTA 校准未能直接迁移供给与交接到整卡 |
| [R15](../Q06-handoff-output/R15-output-service.md) 输出服务 | 实测 | 半经验、机制-程序/硬件 | 已测，条件对照 | 标量探针不能直接作为 CUTLASS 输出常数 |
| [R16](../Q08-register-residency/R16-residency-quota.md) 驻留与配额 | 实测 | 半经验、机制-硬件 | 已测，未用于 V07 | 驻留收益依供给变化；寄存器申请等待取决于释放时序 |
| [R17](../Q01-scheduling/R18-cluster-boundary.md#r17) 越界 tile | 实测 | 半经验、机制-程序/硬件 | 已并入 R18 | 改用匹配位置的窗口增量；旧两例修正只作诊断 |
| [R18](../Q01-scheduling/R18-cluster-boundary.md) cluster 边界 | 实测 | 半经验、机制-程序/硬件 | 旧规则用于V07；新配对已接开发模型 | cfg_b 同逻辑输出下，有效地址零比 M/N 向 TMA 越界填零快约17%–20%；分项仍随位置变化 |
| [R19](../Q01-scheduling/R19-critical-cta-tail.md) 关键 CTA 与尾部 | 实测 | 半经验、机制-程序/硬件 | 已测，只作诊断 | swizzle 改变工作与尾部；V07 尾差项选择 none |
| [B01](../Q03-source-service/B01-bandwidth-cache.md) 带宽与缓存 | 实测 | 半经验、机制-硬件 | 已测，未用于 V07 | 条件请求速率；不等同物理 HBM 流量或缓存收益 |
| [V01](../Q15-full-time/V01-validation.md) 组合验证 | 实测 | 半经验（C1/C3）；分项供机制诊断 | 部分通过 | 单 CTA 通过；整卡与完整 kernel 未通过 |
| [V02](../Q15-full-time/V02-kernel-prediction.md) 完整 kernel | 实测 | 半经验（C1/C3）；分项供机制诊断 | 初版目标通过 | 11 个新尺寸为 6.0% / 10.3%，按当时 10% / 20% 目标 |
| [V03](../Q15-full-time/V03-clock-rule.md) 频率规则 | 实测 | 半经验（C1/C3）；分项供机制诊断 | 同卡预测通过 | 11 个新尺寸为 2.6% / 5.0%；跨卡常数有差异 |
| [V04](../Q15-full-time/V04-config-transfer.md) 配置迁移 | 实测 | 半经验（C1/C3）；分项供机制诊断 | 部分通过 | cfg_c 迁移通过；cfg_a/b 仍失配 |
| [V05](../Q15-full-time/V05-rule-transfer.md) 规则迁移 | 实测 | 半经验（C1/C3）；分项供机制诊断 | 未通过 | 18 例为 10.7% / 31.9%；部分打点不作定量判定 |
| [V06](../Q15-full-time/V06-revised-transfer.md) 递推修正 | 实测 | 半经验（C1/C3）；分项供机制诊断 | 未通过 | 18 个留出条件为 5.25% / 19.0%；边界与尾部未解决 |
| [V07](../Q15-full-time/V07-rule-validation.md) 边界迁移 | 实测 | 半经验（C1/C3）；分项供机制诊断 | 总时间通过，分项受限 | 24 条件为 3.06% / 8.71%；范围与分项限制见上 |
| [V08](../Q15-full-time/V08-wider-validation.md) 更宽留出 | 实测 | 半经验（C1/C3）；分项供机制诊断 | 整体未通过 | 36 条件为 2.61% / 35.34%；21 条件子集见上 |
| [V09](../Q15-full-time/V09-component-validation.md) 组件组合 | 实测 | 半经验（C1/C3）；分项供机制诊断 | 整体未通过 | 30条件实测；29预测为4.11%/35.46%，另1拒绝 |

全部实验记录见[按研究问题阅读](../README.md#research-questions)。

## 非实验结果目录

<a id="non-experiment-results"></a>

重测批次、存储恢复与整理记录统一列在 [non-experiment/records 索引](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/README.md)。

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
| [C1-01](../Q16-config-selection/C1-01-selection.md) | 实测 | 三类比较对象的 C1/C3 协议、冻结与评分入口 | 依该任务原执行安排 |
| [RESEARCH-GUIDE](RESEARCH-GUIDE.md) | 实测 | 采样、预测边界与评分协议，供所有模型 | 公共细则 |

计划、分工与启动记录不是额外模型类别：[PLAN](PLAN.md)保存实验进度，[EXECUTION](EXECUTION.md)保存协作记录，10-11 两个有界目标的[启动记录](../../gh200_sm90_archive/access_rules/PARALLEL-START-20261011.md)已归档。总推进顺序以[项目 PLAN](../../../PLAN.md)为准。

已合并页面的原路径继续保留：[R04](../Q07-smem/R04-joint-service.md)、[R07](../Q09-frequency/R07-anchor-clock-fixedcost.md)、[R17](../Q01-scheduling/R17-oob-tile.md)；上述索引指向其当前证据位置。

## 数据与代码

- [运行、拟合与分析代码](../../../../../../microbench/gh200_resource_campaign/access_rules/)：预测入口为v09_model，供给/输出/时钟是独立纯模块；原分析CLI保留复现用途。
- 两组R18方向配对统一由[r18_cases.py](../../../../../../microbench/gh200_resource_campaign/access_rules/r18_cases.py)生成（`--family first-fill|input-map --axis m|n`）；四份已有JSON作为历史展开快照保留，`--check`核对全部字段。
- [原始结果](../../../../../../results/gh200_resource_campaign/access_rules/README.md)按 `问题/合并实验/原轮次` 保存源码、构建、环境、样本与分析；每个合并实验的 README 列出全部轮次（含中间版本与失败尝试），冻结预测及绑定记录见各 V 页。10-20 前的平铺路径见 [MOVED-PATHS-20261020.tsv](../../../../../../results/gh200_resource_campaign/access_rules/MOVED-PATHS-20261020.tsv)；10-28 重组前的三层路径见 [MOVED-PATHS-20261028.tsv](../../../../../../results/gh200_resource_campaign/access_rules/MOVED-PATHS-20261028.tsv)。两表均指向当前目录。
- [规则说明](RULES.md)保留计量约定、手算例子与 2026-10-08 一次性离线复现记录。
- R09、R10、R13 的逐问题推导与证据在 2026-10-12 整合附录，随各页放在所属问题目录：[R09](../Q09-frequency/reanalysis/20261012-consolidation/R09/README.md)、[R10](../Q03-source-service/reanalysis/20261012-consolidation/R10/README.md)、[R13](../Q02-tma-request/reanalysis/20261012-consolidation/R13/README.md)；由对应 R 页逐条链接，是这三页的附录，不另立结论。
- [历史过程文件](../../gh200_sm90_archive/access_rules/)保留原样；不作为当前执行要求。
