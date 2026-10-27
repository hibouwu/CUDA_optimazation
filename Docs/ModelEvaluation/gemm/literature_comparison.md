# 本仓库 GEMM 建模与参考论文对照

更新：2026-10-13。对照对象包括 `ModelEvaluation/gemm` 的通用接口与三类模型（[模型入口](model/README.md)）、GH200 实际预测代码与留出结果，以及 `blackwell_tensorcore/gemm` 的 Thor 上界和经验包络。论文范围为 [references](../references/README.md) 中 36 份 PDF、35 项研究；Modern GPU Cores 的正式稿和预印本属于同一研究。

当前 GH200 模型是一套**固定执行组织下、由测量校准的 CTA 事件递推模型**（半经验模型 V09），已经超过简单 Roofline，但尚未形成可跨 kernel 结构迁移的完整资源模型，整体留出也未通过。机制模型（程序图＋硬件服务规则）正在建立，主线是多 CTA TMA 共享服务与有效频率。与论文最接近的分别是 Warp Specialization Model 的事件递推、TileSight 的 tile 结构与资源组合，以及 Sim-FA 的异步生命周期；现有证据不足以给这些模型排精度名次。

本仓库的主要差别在于：把输入数值及调用条件对有效频率的影响、局部分项误差和关键 CTA 定位纳入检验；实际薄弱点在于供给、交接、输出及频率参数的迁移。用统一 DAG 表达这些对象有价值，但图结构本身不能替代服务参数与仲裁规则的验证。

## 仓库中三类模型与实测各自回答什么

| 类别与对象 | 当前实际内容 | 与论文对照时的边界 |
| --- | --- | --- |
| [L0–L4 通用接口](model/common/interfaces.md) | 操作、资源需求、存储生存期、eligible/issue/complete/visible/release；区分时间下界和条件预测 | 是模型规范，不能把接口覆盖等同于实际预测覆盖。层号表示建模职责，不是 cache 层级。 |
| 规格：[规格基线](model/spec/spec_baseline.md) | max(F/P, Q/B) 的理想参照，规格与来源在硬件配置文件 | 对应 Roofline；不承诺实际耗时，暖 L2 下不是严格下界 |
| 半经验：[V09](model/semi_empirical/MODEL-CANDIDATES.md#v09-model) | cfg_a/b/c 静态软件工作映射，first/later 主循环、输出递推、输入相关时钟、plain 时间映射 | 是当前完成新留出评分的组合候选，整体未通过。它不是通用指令级模拟器。 |
| 半经验：[开发模块](../../../microbench/gh200_resource_campaign/access_rules/supply_model.py) | 有限槽、共享请求、源身份、cluster 依赖的纯事件模块；[输出模块](../../../microbench/gh200_resource_campaign/access_rules/output_model.py)有 rank 依赖图 | “没有任何 DAG 实现”不准确；但这些模块不能继承 V09 的成绩，也没有组成通过新留出的替代模型。 |
| 比较：[C1-01](experiments/gh200_sm90/Q16-config-selection/C1-01-selection.md) | 八个 CUTLASS 候选上，规格、半经验、经验模型与 TileSight 在同一数据上比较 C1 与 C3 | 已登记，开发测量尚未执行，目前没有同条件成绩 |
| 机制-程序：[CUTLASS 程序图](model/mechanism/program/cutlass_program_graph.md) | cooperative/pingpong 的操作、依赖与槽代次，已与源码和现有递推对应 | 尚无独立图导出器和整图 SASS/trace 验收 |
| 机制-硬件：[硬件路径](model/mechanism/hardware_rules/README.md) | TMA 单 CTA（S16）与多 CTA 共享服务、有效频率、m11 等路径的候选、反例与重启条件 | m11 两次留出未通过；TMA 与频率尚无通过干预留出的规则 |
| 机制组合 | 程序图接入版本化服务规则，见[接口](model/common/interfaces.md)与 [PLAN](PLAN.md) | 尚未形成可独立验收的组合模型 |
| [跨算子扩展](PLAN.md#3-后续任务的进入条件) | 从已确定执行策略的降级 IR 提取程序，逐节点模拟，再扩展 FlashAttention、MoE 和算子图 | 整体路线仍是规划。代码中的局部事件图不等于通用前端和跨算子预测器已经完成。 |
| 教学：[FP32 分块示例](schemes/fp32_simt_tiled.md) | 给定串行阶段、固定服务份额和一 CTA/SM 假设下的单/双缓冲递推 | 参数是教学假设；不能用其公式成立证明 GH200 达到相同服务。与经典 latency hiding 和流水模型对照。 |
| [Thor 三层模型](../../blackwell_tensorcore/gemm/README.md) | 条件性能上界、经验理想包络、已观测完整 GEMM；persistent worker 因果递推 | 目标是“声明实现域最快能到哪里”；文档所绑定历史快照仍为 complete=false，不当作当前 GH200 预测成绩；未重新采集 Thor 数据。 |

Thor 的 `measured_sustained` 不是物理速率上界，独立 component 峰值也不证明联合可达。这是它与直接用校准吞吐率预测时间的论文之间的重要主张差别，不能因为二者都含 `max(work/rate)` 就视为同一种结论。详见 [Thor 范围](../../blackwell_tensorcore/gemm/model/01_scope_and_claims.md)、[因果流水线](../../blackwell_tensorcore/gemm/model/06_causal_pipeline_model.md)。

## 当前 V09 真正计算了什么

代码入口是 [`predict_components`](../../../microbench/gh200_resource_campaign/access_rules/v09_model.py)，配合 [supply_model](../../../microbench/gh200_resource_campaign/access_rules/supply_model.py)、[output_model](../../../microbench/gh200_resource_campaign/access_rules/output_model.py)、[clock_model](../../../microbench/gh200_resource_campaign/access_rules/clock_model.py)。它接受尺寸、行距、输入模式、swizzle、构建得到的静态资源和冻结参数。校准阶段使用实测；预测入口不读取目标计时。

对 CTA i 的第 j 个输出 tile，主循环仍被压缩为：

\[
L_{i,j}(f)=b_{cfg,p}+K_t\max\left(q_{cfg}/f,\theta_{cfg,p}^{\mathsf T}x_{i,j}\right),\qquad K_t=K/64.
\]

p 区分首个和后续输出 tile。q 为 cfg_a/b/c 的 512/512/1024 cycle/Ktile；f 用 GHz，L 与 b 用 ns。x 包含有效源请求、A/B 地址覆盖、SMEM 填零等逻辑特征。它不是物理 L2/HBM 事务测量。

这些 L 与输出窗口进入 cooperative/pingpong 的 CTA 递推，再取最慢 CTA 的包络 W。时钟由名义计算和源请求活动与 W 联立求解；最终 `T_plain = F + κW`。F/κ 是观察协议间的经验映射，不是独立测出的纯 launch 成本。完整关系见 [现行公式](model/semi_empirical/MODEL-CANDIDATES.md#v09-model)。

**结构差距就在这条 L 公式里：** TMA 发出、目的槽就绪、WGMMA commit/wait、槽释放和不同请求之间的仲裁，在正式 V09 路径中还没有全部逐事件展开。部分影响被吸收进 b、θ 和预填权重。因而改变 stage、布局或背景 CTA 后，即使数学工作量不变，原价格也可能不再适用。

2026-10-13 用当前代码和原冻结参数重放 29 个受支持条件，plain 时间、频率以及关键 CTA 集合与冻结结果完全一致。原拒绝条件保持拒绝；未重跑支持性线性规划，也未重拟合或采样。由原评分逐行重算：绝对相对误差**中位 4.11%、均值 6.00%、最大 35.46%**。均值是补算值，不能把原先的中位数称为 MAPE。[V09 原始评分](../../../results/gh200_resource_campaign/access_rules/Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011/score.json)保持不变。

## 最接近的论文与可直接借鉴的部分

### TileSight 的结构更通用 但论文包络比当前实验问题更粗

[TileSight](../references/papers/2026-tilesight/2026-tilesight.md) §3.1–3.5 将 tile 执行计划分成资源向量、依赖、并发和遍历顺序。论文用启动—稳态—排空包络组合时间，以 tile reuse distance 估计缓存命中，再按完整波和尾波组合。它已经把缓存复用和候选配置选择放到统一模型中，覆盖范围比当前固定 cfg_a/b/c 更广。

本仓库 L0–L4 与它有明显共性，但不能直接按层号对应：TileSight Figure 3 的 L0–L4 是**存储位置层次**，本仓库的 L0–L4 是**操作到算子图的建模职责**。

论文 §3.4 的有效深度近似为 `stages × resident_tiles_per_SM − 1`；§7 明确限制了未公开 warp/CTA 调度、延迟受限情况，并假设不同 SM 的 tile 以一致速率推进。这些近似不能直接解释本仓库已有的 stage4 early、关键 CTA 和起步干预反例。论文没有提供一条经 GH200 输入模式与调用历史验证的有效频率规则。[原 PDF 第 5、13 页](../references/papers/2026-tilesight/2026-tilesight.pdf)。

还要区分论文和本地代码快照。[归档接口](../references/repositories/tilesight/tilesight/modeling/program/contract.py)已有 `Actor`、`Carry`、`BufferSlots` 的 acquire/release，以及 alias/ring 存储组；[分析入口](../references/repositories/tilesight/tilesight/modeling/program/analysis.py)已有 resource/recurrence/credit II 和完成事件输出。**不能把论文的短公式当作该源码全部能力的上限，也不能把源码新增能力自动视为已被论文实验验证。**

我们已在 C1-01 的 480 个条件×8 个候选上运行公开 TileSight（commit 75428c4，H200 microbench profile，只把 HBM 带宽与 L2 容量改为 GH200 规格），[预测与映射说明](../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261013-C1-baselines/tilesight/README.md)。它不能表达 pingpong，stage 6/4/3 的预测相同，cluster 只进入派发顺序与多播近似；这些预测尚未与实测比较。

最短借鉴路径是复用其程序描述、缓存访问身份和结果接口，保留本仓库自己的 GH200 服务条件与拒绝规则。对“固定已编译 kernel 的预测”，还必须保留实际 issue 顺序；若求解器可以自由选择最好的合法排序，得到的可能是优化方案或理想包络，而非这个既定实现的运行时间。

### Warp Specialization Model 与现有递推最接近

[A Performance Model for Warp Specialization Kernels](../references/papers/2025-warp-specialization-model/2025-warp-specialization-model.md) §2 用 A/B 加载和 MATH 开始时刻的 max 递推表达生产者、消费者和环形缓冲，再组合为 wave 时间与整体时间。形式上与本仓库 FP32 示例和 GH200 CTA 递推最接近；tile DAG 和 max 递推本身并不是本项目独有的方法。

差别在服务价格和执行对象：该文在 A6000 上用两个 tile 尺寸估计 DMA/MATH 的启动与吞吐，假定 wave 时间一致；没有覆盖本项目的 Hopper cluster、TMA 填零、输入相关频率与跨 CTA 背景效应。其 2048 配置平均误差 4.5%、最大 21.5% 属于这套执行组织。[原 PDF 第 4–5、7–8 页](../references/papers/2025-warp-specialization-model/2025-warp-specialization-model.pdf)。

可借鉴的是递推表达和 simulation/solver 的一致性检查；不能直接移植其常数，也不能把两点线性定价视为已经解决本项目行距迁移。

### Sim-FA 展开了异步状态 也保留了经验修正

[Sim-FA](../references/papers/2026-sim-fa/2026-sim-fa.md) §IV–V 显式建模 warpgroup、TMA、WGMMA 组、mbarrier、有限请求资源和内存系统。它比 V09 的聚合 L 更接近下一步所需的异步状态表示。TTGIR 插桩产生运行 trace，再翻译成模拟指令；采集的时间戳用于程序顺序和去重，不等于把目标时长直接作为模拟结果。

但“输入需要执行”仍与本仓库“运行前只使用静态程序及冻结参数”的目标不同。更关键的是，论文 GEMM 验证核是 **128×128×64、每 CTA 一个 warpgroup、没有 warp specialization**；FA3 另有专门前端。不能把这 23 个 GEMM 的成绩看作 cfg_a/b/c cooperative/pingpong 的验证。

第 7–8 页还说明：H800 锁定 1830 MHz，采用冷 L2/CUPTI kernel duration；GEMM 校准加入 **4314 cycle 固定项和每 BK64 迭代 330 cycle 残差**，最后达到 5.49% MAPE。更细的模拟器仍有校准项，其完整时间边界也不同于本项目 plain CUDA event 及暖机协议。[原 PDF 第 7–8 页](../references/papers/2026-sim-fa/2026-sim-fa.pdf)。

值得采用的是异步组与 barrier 状态机、请求背压和分项验证；H800 的延迟、MSHR 数与经验残差需要在本项目协议下重新辨识，不能当作 GH200 已知硬件常数。

### WaveTune 是配置选择基线 Twill 是生成更好调度的方法

[WaveTune](../references/papers/2026-wavetune/2026-wavetune.md) §4.3 按宏配置与 wave 数分桶，用稀疏实测拟合 `T(G,L)=αGL+βG+γL+δ`，并缓存微配置锚点。它适合比较“较简单的统计模型是否已经足够选好配置”。它不提供 barrier、关键 CTA 或硬件内部原因；每个桶的系数也不是可迁移的资源价格。当前 V09 的局部主循环已经含有线性 K 假设，加入双线性拟合本身不能解决长 K 的频率迁移。

[Twill](../references/papers/2025-twill/2025-twill.md) §3–4 将依赖、资源容量、寄存器和跨 warp 通信约束纳入调度求解。第 8 页对 streaming 可变延迟操作采用零延迟，由外部调优选择流水深度。它的最优性相对于给定成本和约束成立，不是完整 GEMM 时间或真实硬件全局最优的保证。

WaveTune 形式已作为 C1-01 的 wave_bilinear 基线；Twill 用于检查程序图的可实现性和调度约束。二者都不能直接替代局部服务的定价。

### 更细的微架构和缓存模型解决不同层的问题

[Volkov](../references/papers/2016-volkov-latency-hiding/2016-volkov-latency-hiding.md) 第 III–IV 章适合检查延迟、吞吐、并发和 occupancy 的极限关系；不能从分配槽数直接推导平均在途数。[Warpspeed](../references/papers/2023-warpspeed/2023-warpspeed.md)从地址表达式估计数据量，值得借鉴其区分请求、复用与实际层级流量的方式。[Modern GPU Cores 正式稿](../references/papers/2025-modern-gpu-cores-final/2025-modern-gpu-cores-final.md) §5–7深入 SASS 控制字段、发射、RF 和片上访存，但没有为本项目建立 Hopper TMA/WGMMA 验证。

本仓库普通 SMEM 的结果说明“层级更细”不等于“机制已经确定”：m11 首次随机留出 50 个受支持条件中位 4.55%、最大 34.82%；原 588 个结构化开发条件无法区分两种需求规则。测后改用“子组 bank 两两不相交才合并”的规则重新冻结，v2 留出 47 个受支持条件中位 0.44%、最大 11.60%，仍未通过，失败集中在 1 warp 独立读。[v2 结果](experiments/gh200_sm90/Q07-smem/R03-access-demand.md#m11-holdout-v2-result)。不能用其它架构的 bank 规则或开发集上的吻合代替留出。

## 哪些差异已经造成实际预测失败

| 现象 | 当前 V09 与实测差异 | 论文能提供什么 | 仍需要本项目回答什么 | 当前负责的任务 |
| --- | --- | --- | --- | --- |
| cfg_b 长 K | 预测有效频率 0.9000 GHz，观测约 1.2000 GHz；总时间 +35.46%，代入观测频率后 +1.71% | 输入功率、DVFS、功率上限、GPU 个体差异给出机制背景；锁频模拟便于隔离周期成本 | 只用允许的静态输入、冻结参数和声明历史，能否预测调用内频率；观测频率替换只作诊断 | [有效频率](model/mechanism/hardware_rules/FREQUENCY-MECHANISM.md)；剩余局部 L 残差在[长 K](model/semi_empirical/MAINLINE-LONG-K.md) |
| cfg_b A+16 B | 实测相对基线慢 28.94%，模型只给 1.40% | 地址形成、缓存流量与分区/互扰方法有帮助 | 同样逻辑地址覆盖在不同全卡组织下为何有不同服务；没有物理流量证据时保留条件服务名称 | 半经验支持范围外；服务归因见 [TMA 候选与反例](model/mechanism/hardware_rules/TMA-GEMM-HYPOTHESES.md) |
| cfg_b stage6→4 | 实测慢 10.47%，当前冻结候选响应为零 | producer/consumer 递推、BufferSlots 和异步组状态能表达更多依赖 | 哪些槽释放、初态、非同质服务导致响应；现有开发事件图尚未通过组合验证 | [多 CTA TMA 共享服务](model/mechanism/hardware_rules/TMA-SHARED-SERVICE.md)与[程序图](model/mechanism/program/cutlass_program_graph.md)的槽代次；[S16](model/mechanism/hardware_rules/TMA-S16-STAGE-COST.md)只检查单 CTA 探针的控制成本 |
| cfg_c partial/sw8 | 真正最后完成 CTA 在 10/10 次均不属于预测并列集合 | 逐 tile 调度、边界工作与尾波模拟 | 正确的关键 CTA、跨 rank 约束及入口分散；相近总时间不能代替这一检验 | [程序图](model/mechanism/program/cutlass_program_graph.md)的调度列表（含补齐 tile）接入机制组合后检验 |

这些数值均来自 [V09 失败分析](experiments/gh200_sm90/Q15-full-time/V09-component-validation.md) 和 [现行候选](model/semi_empirical/MODEL-CANDIDATES.md#v09-model)，不把后续开发修正回填为原成绩。

其中 stage 和布局问题并不能只靠“画出更细 DAG”解决。已有 `shared_stage_events` 候选曾在联合窗口上得到较小残差，却被 CTA 完成分布和干预响应否定；固定边成本、固定前驱的图也不能解释全部起步干预。新结构必须同时约束局部端点、整卡分布和干预方向。[开发候选及反例](model/semi_empirical/MODEL-CANDIDATES.md)。

## 误差数字怎样比较才有意义

| 模型与当前版本 | 论文或本地记录中的指标 | 为什么不能直接排名 |
| --- | --- | --- |
| GH200 V09 | 29 个支持条件：中位 4.11%、MAPE 6.00%、最大 35.46%；另 1 个拒绝 | 同一参考卡、三个固定 CUTLASS 配置，含布局、输入、stage 与边界迁移；整体未通过 |
| TileSight | 703 个 GEMM 形状，四架构 pooled MAPE 12.35% | 规模、架构、软件配置与计时不同；既无相同留出，也不检验本项目同一关键 CTA |
| Warp Specialization Model | A6000，2048 配置，平均 4.5%、最大 21.5% | 执行组织、服务模型和硬件不同 |
| Sim-FA GEMM | H800，23 形状，MAPE 5.49%，19/23 在 ±10% 内 | 单 WG、无 WS 的 GEMM；锁频、冷缓存、CUPTI 边界，含固定与每迭代校准项 |
| Microbenchmark-driven | 摘要报告 B200 21 个 kernel 为 1.31%；正文另报告 H200 应用迁移较大误差 | §VI 的 H200 Rodinia 为 43.6%、SPEChpc 为 555%；部分工作量依赖 profiler，不能拿微基准成绩代表静态迁移 |

论文数字均是作者在其评测协议下报告的结果，本次没有运行其完整 artifact。同条件比较要等 C1-01 的开发与留出测完：届时规格基线、wave 量化、WaveTune 形式、TileSight 与半经验模型在同一留出上分别报告 C1 与 C3。最后一行尤其说明，应同时核对工作量如何取得、参数如何校准、时间边界和测试集，而不能只抄摘要中的最佳数字。[TileSight §5.1](../references/papers/2026-tilesight/2026-tilesight.md)、[Sim-FA §V-B](../references/papers/2026-sim-fa/2026-sim-fa.md)、[Microbenchmark-driven 应用结果](../references/papers/2026-microbenchmark-driven/2026-microbenchmark-driven.md)。

## 对当前路线的建议

**保留当前静态输入和分项检验目标，先把一组已经测过的执行组织表达准确，再扩展算子和硬件。** 无需重新建一套资源大表，也不必从通用逐指令模拟器起步。

1. **先对齐事件语义和程序接口。** 用现有 cfg_b 两输出或 cfg_a/c rank 图映射到统一程序描述，明确输入 ready、WGMMA 组退役、SMEM 源可复用和全局输出完成。优先检查现有 TileSight `Program/BufferSlots` 接口能否承载这些语义，避免重复实现同类图基础设施。不能把 `done` 一律当作消费者可见或最终输出完成。
2. **用现有反例筛选服务结构。** stage4/6、A16/B32、cluster 边界和 producer 起步干预已经提供约束。先检查一个参数集能否同时解释响应方向与端点；参数不可辨识时输出范围或拒绝，不靠按 case 增加补偿项。这一步可先离线进行。
3. **分开比较周期、时间和选择效果。** 周期用于定位资源与依赖；完整时间另验证频率及计时映射；若用途是选 tile，再加入 WaveTune 风格基线和排序指标。诊断时可以代入观测频率，但正式静态预测不能使用该输入。待候选确定后再申请新的 GPU 留出，旧留出已经用于诊断的部分不再计作新验证。

能够合理主张的潜在区别是：**在无目标运行计时输入的条件下，对异步 GEMM 的局部事件、全卡关键路径及输入相关频率做联合预测，并能说明支持范围。** 当前证据尚未证明这套能力已经完成；仅有 L0–L4 分层、tile DAG 或 max 递推不足以构成新的方法贡献。

## 与当前计划的对应

| 计划中的任务 | 可直接借鉴的文献 | 本项目仍需自己回答的部分 |
| --- | --- | --- |
| [多 CTA TMA 共享服务](model/mechanism/hardware_rules/TMA-SHARED-SERVICE.md) | Real GPU NoC 的源端／目的端／互扰探针；Hopper 微基准的 TMA、DSM 测量；Sim-FA 的请求背压；Volkov 的并发与吞吐关系；Warpspeed 区分请求与物理流量 | 并发 CTA 数×每 CTA 在途请求下的共享范围与释放事件；H800 等常数不能移植 |
| [有效频率](model/mechanism/hardware_rules/FREQUENCY-MECHANISM.md) | Input-Dependent Power、H100/H200 Power Capping、Not All GPUs Are Created Equal；AccelWattch 与多域 DVFS 模型（依赖计数器） | 只用运行前输入预测调用内频率；状态规则需干预区分；新颖性按该任务列出的文献逐项核对后再主张 |
| [S16 控制成本](model/mechanism/hardware_rules/TMA-S16-STAGE-COST.md) | Dissecting Volta/Turing 的 SASS 控制字段与寄存器依赖探针 | token 控制与组状态阻塞的判别；结果只适用于该探针 |
| [程序图](model/mechanism/program/cutlass_program_graph.md) | TileSight 的 Program/BufferSlots 接口、Twill 的依赖与资源约束、Warp Specialization 的递推、TileLang 的降级语义 | 固定已编译 kernel 的实际 issue 顺序；不让求解器自由选择“最好”的合法排序 |
| [C1-01](experiments/gh200_sm90/Q16-config-selection/C1-01-selection.md) | TileSight、WaveTune、tritonBLAS、Habitat、Stream-K 的波次量化 | 同一留出上的 regret 与完整时间误差，及计时噪声下限 |

## 全部论文在本项目中的位置

下表逐项覆盖 35 项研究、36 份 PDF，定位其与实际模型的关系。深入比较集中在上述直接相关方法；其余项目用于界定可借鉴的局部能力，不表示全部公式都已复现。

| 本地论文 | 主要对照对象 | 与本项目的具体区别或用途 |
| --- | --- | --- |
| [Roofline 2008](../references/papers/2008-roofline-report/2008-roofline-report.md) | 时间下界／Thor 上界 | 提供计算与带宽瓶颈基线；不生成依赖、启动、排空或关键 CTA。使用实测速率时只得到条件估计。 |
| [Hong–Kim MWP/CWP](../references/papers/2009-hong-kim-mwp-cwp/2009-hong-kim-mwp-cwp.md) | L0–L1 并行隐藏 | 显式考虑访存与线程并行度；早期 SIMT 架构和聚合 warp 模型不能直接覆盖异步 warpgroup。 |
| [GPUMech](../references/papers/2014-gpumech/2014-gpumech.md) | 停顿与资源竞争 | 基于指令 trace 和功能模拟做区间分析；比 V09 更细，但输入取得方式不同。 |
| [Volkov 学位论文](../references/papers/2016-volkov-latency-hiding/2016-volkov-latency-hiding.md) | L0 延迟／L1 吞吐；多 CTA TMA 并发 | 区分延迟、吞吐和所需并发；适合检验模型极限，不给 GH200 的具体参数。 |
| [Dissecting GPU Memory Hierarchy](../references/papers/2017-gpu-memory-hierarchy/2017-gpu-memory-hierarchy.md) | 缓存与 SMEM 探针 | 提供 pointer-chasing 和冲突实验方法；Fermi–Maxwell 的常数不移植。 |
| [Multi-Domain DVFS Power Model](../references/papers/2018-gpgpu-power-dvfs/2018-gpgpu-power-dvfs.md) | 频率与功率 | 核心／显存双域功率模型，依赖性能计数器；不能直接填入无计数器的 R09。 |
| [Dissecting Volta](../references/papers/2018-volta-microbenchmark/2018-volta-microbenchmark.md) | SASS／RF／存储层次；S16 控制路径 | 适合借鉴探针与控制字段核查；硬件是 V100。 |
| [PPT-GPU CAL 2019](../references/papers/2019-ppt-gpu/2019-ppt-gpu.md) | warp／CTA 到 kernel | 模拟粒度更细，并使用 trace／缓存信息；本地 PDF 不是后来的 SC 2021 全文。 |
| [Dissecting Turing T4](../references/papers/2019-turing-microbenchmark/2019-turing-microbenchmark.md) | RF reuse／指令服务 | 补充 T4 探针；不能当作 SM90 端口或延迟定值。 |
| [Accel-Sim](../references/papers/2020-accelsim/2020-accelsim.md) | 逐周期模拟与验证 | 原生 ISA trace 和详细后端，验证基础设施值得借鉴；目标输入和运行成本不同。 |
| [Hierarchical Roofline](../references/papers/2020-hierarchical-roofline-dl/2020-hierarchical-roofline-dl.md) | 分层带宽限制 | 多层流量依赖 profiler；本仓库逻辑字节不能冒充计数器测得流量。 |
| [AccelWattch](../references/papers/2021-accelwattch/2021-accelwattch.md) | 功率／DVFS | 周期级功率模型，Volta 验证；本项目 R09 是有效频率经验模型，不是功率预测器。 |
| [Habitat](../references/papers/2021-habitat/2021-habitat.md) | 跨 GPU 时间迁移 | 先在已有 GPU 上运行再做 wave scaling／学习预测；与只用静态程序及硬件校准的目标不同。 |
| [Demystifying Ampere](../references/papers/2022-ampere-microbenchmark/2022-ampere-microbenchmark.md) | PTX–SASS 与指令周期 | 用作方法和架构对照；A100 指令价格不直接给 GH200。 |
| [Not All GPUs Are Created Equal](../references/papers/2022-gpu-variability/2022-gpu-variability.md) | 跨卡与环境差异 | 支持对同型号个体、功率和频率变化做控制；不能代替本项目跨卡验证。本项目 10-13 已见同卡不同作业的参照差异。 |
| [Shared/Local Memories of Ampere](../references/papers/2023-ampere-shared-local/2023-ampere-shared-local.md) | LDS／LDGSTS／spill | A100 局部路径测量；没有替本项目验证 SM90 宽广播与 lane 排列规则。 |
| [Stream-K](../references/papers/2023-stream-k/2023-stream-k.md) | L3 工作划分与尾波 | 改变按 tile 分配的策略，是 GEMM 调度方法；不是固定 cfg_a/b/c 的现成时间预测器。 |
| [Dissecting Tensor Cores](../references/papers/2023-tensor-core-microbenchmark/2023-tensor-core-microbenchmark.md) | ldmatrix／mma／稀疏路径 | 提供至 Ampere 的局部服务测量；没有 Hopper WGMMA 完整生命周期。 |
| [Warpspeed](../references/papers/2023-warpspeed/2023-warpspeed.md) | 静态地址与流量 | 缓存与地址估计比 V09 简单覆盖特征更丰富；验证以访存型 kernel 为主。 |
| [Input-Dependent Power Usage](../references/papers/2024-input-dependent-power/2024-input-dependent-power.md) | 输入数值对功率的影响；有效频率 | 解释输入内容不可忽略；论文特意选择不降频条件，不给出 R09 的时间／频率映射。 |
| [LLMCompass](../references/papers/2024-llmcompass/2024-llmcompass.md) | L3–L4 映射和组合 | tile 级模拟与 mapper 用于 LLM 硬件评估；不提供本项目逐 CTA 异步端点的同条件验证。 |
| [Real GPU NoC Characteristics](../references/papers/2024-real-gpu-noc/2024-real-gpu-noc.md) | 全卡竞争与拓扑；多 CTA TMA | 源端、目的端及互扰探针可帮助区分局部成本和背景竞争；当前 GH200 条件服务尚未完成该物理归因。 |
| [Hopper 微基准](../references/papers/2025-hopper-microbenchmark/2025-hopper-microbenchmark.md) | TMA／WGMMA／DSM | 架构更接近，但主要为 H800；必须对齐 tile、等待方式、驻留与计时窗口。 |
| [Modern GPU Cores 正式稿](../references/papers/2025-modern-gpu-cores-final/2025-modern-gpu-cores-final.md) / [预印本](../references/papers/2025-modern-gpu-cores-preprint/2025-modern-gpu-cores-preprint.md) | 发射／RF／LSU | 增强 trace 与微架构模拟；可借鉴约束，不能据其验证范围宣称 Hopper 完整支持。 |
| [NeuSight](../references/papers/2025-neusight/2025-neusight.md) | tile 学习模型与 L4 | 学习 tile 利用率再组合到算子和模型，目标侧不必实跑；需要训练数据，解释边界与事件模型不同。 |
| [TileLang](../references/papers/2025-tilelang/2025-tilelang.md) | 程序前端 | 提供 tile 运算和降级语义；编译器本身不提供本项目所需的可信服务价格。 |
| [tritonBLAS](../references/papers/2025-tritonblas/2025-tritonblas.md) | 层级 tiling 与配置选择 | 按 wave 和 tile 成本选配置；MI300X 上配置选择效果不等同绝对时间预测误差。 |
| [Twill](../references/papers/2025-twill/2025-twill.md) | 可实现流水与调度优化 | 联合资源／寄存器／warp 约束；给定成本下的最优调度不等于硬件完整时间预测。 |
| [Warp Specialization Model](../references/papers/2025-warp-specialization-model/2025-warp-specialization-model.md) | L2 生产消费递推；程序图 | 与现有方法最接近；A6000 和简单 DMA/MATH 定价没有覆盖 GH200 的全卡条件服务。 |
| [Accel-Sim 2.0](../references/papers/2026-accelsim2/2026-accelsim2.md) | 异步与分布式周期模拟 | 支持新异步特征和复杂内存系统；动态 trace 驱动，不能直接满足静态输入约束。 |
| [H100/H200 Power Capping](../references/papers/2026-h100-h200-power-capping/2026-h100-h200-power-capping.md) | 功率上限与频率 | 可借鉴频率和显存功率对照；设备条件不是 GH200 的完整平台功率合同。 |
| [Microbenchmark-Driven Model](../references/papers/2026-microbenchmark-driven/2026-microbenchmark-driven.md) | 跨架构参数化时间模型 | 摘要低误差与应用迁移不同，部分需求来自 profiler；不能作为无需重新辨识即可迁移的证据。 |
| [Sim-FA](../references/papers/2026-sim-fa/2026-sim-fa.md) | TMA／WGMMA／barrier 状态 | 异步状态更显式，但需动态 trace；GEMM 验证核、冷缓存计时和残差校准均与 V09 不同。 |
| [TileSight](../references/papers/2026-tilesight/2026-tilesight.md) | tile 结构／流水／缓存／L4；C1-01 公开基线 | 最适合对齐程序接口和广域基线；不能用均匀 tile 推进假设覆盖本项目已见的局部响应。已在 C1-01 条件上给出静态预测，不能表达 pingpong 与 stage 差别。 |
| [WaveTune](../references/papers/2026-wavetune/2026-wavetune.md) | 经验延迟与调优；C1-01 基线 | 按宏配置／wave 分桶拟合，适合排序基线；不承担微架构因果解释。 |

## 复核入口

本次没有重新拟合模型、修改原成绩或启动 GPU 实验。V09 重放只读冻结参数与评分；TileSight 基线不使用本项目实测。

论文版本以本地归档为准；作者记录另见 [TileSight v1](https://arxiv.org/abs/2607.22432v1)、[Sim-FA v4](https://arxiv.org/abs/2605.00555v4)、[Warp Specialization Model](https://arxiv.org/abs/2506.11209)。论文阅读版仍受 [阅读版说明](../references/README.md#论文阅读版) 约束，关键公式和计时协议应以原 PDF 为准。
