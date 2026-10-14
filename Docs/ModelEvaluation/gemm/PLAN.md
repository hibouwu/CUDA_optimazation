# 三类模型与机制研究推进计划

更新：2026-10-13。目标、选题与验收规则见[GOALS](GOALS.md)，模型差别见[对照表](mechanistic_modeling.md)。本页维护顺序和依赖；具体测量状态与资源安排见[实验执行计划](experiments/gh200_sm90/access_rules/PLAN.md)。

## 1. 近期顺序

| 优先级 | 任务 | 当前阶段与下一步 | 依赖与收口 |
|---|---|---|---|
| 小任务，限额收口 | [S16 探针控制成本](model/mechanism/hardware_rules/TMA-S16-STAGE-COST.md) | 历史离线诊断完成；先准备同一二进制、活动S2/S4的A/B；C/D仅作必要复核 | 核对动态控制路径后冻结预测；做一次小规模判别，决定这批证据是否仍要求硬件组成本，然后归档 |
| 主线，并列 | [多 CTA 的 TMA 共享服务](model/mechanism/hardware_rules/TMA-SHARED-SERVICE.md) | 复用多CTA记录；以实际并发CTA数×每CTA请求并发组织为主轴，先稀疏后按分歧加点 | 不等待S16完整归因；交付服务规则、共享范围约束及对GEMM供给/stage响应的组合诊断 |
| 主线，并列 | [有效频率预测与机制](model/mechanism/hardware_rules/FREQUENCY-MECHANISM.md) | 复用V09/R09，接入合格的C1-01开发数据；建立运行前预测与可区分的频率状态候选 | 不等待TMA完成；分别交付C3所需预测器和C2干预结论 |
| 主线，并列 | [cfg_b 组合](model/mechanism/composition.md) | (1) 将 cfg_b 的程序事实做成可执行输入：工作列表、操作顺序、槽代次，与源码、SASS、trace 核对。<br>(2) 接通聚合模式：交付可逐个替换模块的组合框架和逐端点输出，数值上等同于 V09；对 cfg_b 冻结预测逐位复现 V09，作为后续每次替换模块的回归基线。<br>(3) 接入请求级服务：TMA tensor 请求、计算完成与槽释放，每次只替换一组服务。<br>(4) 冻结后按 C2/C3 检验。<br>(5) 扩展到其他配置，进入候选集合比较后再做 C1。 | 第 1、2 步只用 CPU 和已有源码、SASS、trace，现在即可开展；第 3 步依赖 TMA 与频率两条主线。多 CTA 实验不以单 CTA 细节为前置条件。 |
| 独立按既定协议推进 | [C1-01 模型比较](experiments/gh200_sm90/access_rules/C1-01-selection.md) | 既定八候选开发/留出设计，分别比较C1与C3 | 条件清单、冻结规则、测量安排由原任务维护；数据合格入库后供两条主线复用开发部分 |

## 2. 三类模型与程序图的接入依赖

| 工作 | 推进顺序 | 接口或交付入口 |
|---|---|---|
| 规格基线 | 独立计算入口已完成；比较时记录产品/参考频率口径 | [规格基线计算](model/spec/spec_baseline.md)；C1-01继续使用原冻结口径 |
| 半经验模型 | 保留V09冻结版本；由C1-01比较现有水平；频率开发先进入独立版本 | C1-01已经分别列出纯规格、wave_roofline、structural与TileSight，完整时间归C3 |
| 程序图 | 已整理源码与递推对应；下一步导出静态图并做逐配置SASS/trace核对，再接TileLang前端 | [CUTLASS程序图](model/mechanism/program/cutlass_program_graph.md)、[操作与事件接口](model/common/interfaces.md) |
| 硬件服务候选 | 多CTA供给与频率并行建立候选及限定参数 | 两项主线任务的现象、假设、实验验证、结论 |
| 机制组合 | 所需局部规则和程序图可用即接入；逐项检查GEMM响应，再做完整时间与配置比较 | cfg_a/b/c与TileLang变体，保留旧版预测作为对照 |

频率预测同时为多CTA供给实验解释cycle/ns变化提供条件；TMA组合的活动描述可作为频率候选输入。两线先使用各自固定的观测与开发版本，之后检验耦合，避免互相等待。

## 3. 后续任务的进入条件

| 候选方向 | 何时重新选题 |
|---|---|
| 多播、DSM、WGMMA其他路径 | 当前C1/C3残差需要该规则，或形成具有新增区分价值且可执行的独立C2课题 |
| FlashAttention、MoE | GEMM阶段已有可复用接口，且明确新算子需要增加或检验的机制 |
| 算子图、端到端与其他硬件 | 单kernel组合有可用范围，目标工作负载或新设备已明确 |

以上为候选方向，不是自动展开的任务队列。具体预算与终止条件放在各任务登记中。

## 4. 协作与资料入口

- TileLang方向合作者提供降级IR到程序图的前端；本仓库维护服务候选、参数证据与组合验证。首轮人工核对图不等待自动前端。
- 历史复用入口：[TMA盘点](model/mechanism/hardware_rules/TMA-HISTORY-REUSE.md)、[首版输入模型](model/mechanism/hardware_rules/TMA-INPUT-EVENT-MODEL.md)、[模型候选与反例](model/semi_empirical/MODEL-CANDIDATES.md)。
- 用户有两次向 NVIDIA 微架构研究人员请教的机会；保留为候选分歧明确后的咨询资源，不作为当前工作的前置条件。
