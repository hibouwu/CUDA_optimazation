# GEMM 设计与性能建模

已有 L0–L4 接口和限定执行组织的 FP32 时间模型；GH200 的固定 CUTLASS 配置已完成 FP16 GEMM 时间预测与 V01–V09 留出验证，最新 V09 整体未通过；当前结论与适用范围见 [GH200 预测实验入口](experiments/gh200_sm90/access_rules/README.md)。模型分规格基线、半经验和机制三类，共用实机验证；机制模型由程序图、硬件服务规则及二者的组合构成。GH200 上：规格基线有独立计算入口；九个硬件服务方向与同版cfg_b组合已形成离线交付，多CTA TMA与有效频率的既有模块进入组合，新曲线与真实短链仍待测，S16单CTA判别限额收口；八候选的C1-01已有开发比较，剩余测量及独立留出暂停。目标与验收见 [GOALS](GOALS.md)；顺序、依赖及跨算子、跨硬件的后续方向见 [PLAN](PLAN.md)；实验状态见 [GH200 执行计划](experiments/gh200_sm90/access_rules/PLAN.md)。

当前基础问题范围：**单 GPU 的 Single GEMM**。Batched / Grouped 暂不考虑，Distributed GEMM 暂不实现；具体标记见[审查范围](modeling_review.md#当前范围与暂缓项)。`problem.md` 保留完整候选域。

三类模型、机制子模块与实机证据统一从[模型与证据入口](model/README.md)查阅；该页同时列出各类内容的唯一维护位置。

## 阅读顺序

日常按以下顺序阅读：

1. 本页：总问题、当前范围与模型层级。
2. [模型入口](model/README.md)：共同输入、三个模型分支、机制组合与实机验证怎样组成整个模型，以及各模型的文件夹。
3. [模型目标](GOALS.md)：按规格基线、半经验模型、机制模型分节写问题、子问题与验收，另有实机验证和共用的 C1/C3、选题与停止规则；[模型对照](mechanistic_modeling.md)只比较当前与理想模型。
4. [计划](PLAN.md)：列出九个模块阶段、同版cfg_b组合、待测依赖与暂停支线。
5. 当前模块：对应的模型或研究页，入口见[模型目录](model/README.md#目录)。
6. 实验证据：[实机验证入口](experiments/README.md)；[GH200 实验](experiments/gh200_sm90/README.md)与[时间预测验证](experiments/gh200_sm90/access_rules/README.md)：查阅实测依据、预测误差和适用范围。

实现某个模块时按需查阅：

- [问题定义](problem.md)与[问题实例](workloads.yaml)：确定计算、输入输出与性能指标；case 字段、默认值与来源见[共同输入](model/common/inputs.md)。
- [统一接口](model/common/interfaces.md)与下方 L0–L4：确定操作、资源、事件及层间组合。
- [Thor/SM110](hardware/thor_sm110/README.md)或 [GH200/SM90](hardware/gh200_sm90/README.md)：查阅目标硬件的参数、支持条件与来源。
- [FP32 分块模型](schemes/fp32_simt_tiled.md)：查看配置如何转为需求、事件和周期预测；[tc5a](schemes/tc5a.md)单独保留为后续阶段的历史候选。
- [性能建模参考资料](../references/README.md)：查阅论文、模型源码与官方研究材料，按模型层级和输入要求选择对照。

## 模型层级

以下层级用于 Thor 和 GH200 的性能建模；L0–L4 是模型层级，与硬件 L1/L2 cache 无关。

| 模型层级 | 定义 | 输入与输出边界 | 主要回答的问题 |
|---|---|---|---|
| [**L0：基础操作服务**](model/common/L0.md) | 计算、数据搬运、同步和辅助操作的基本服务 | 所需输入与状态满足条件 → 操作约定的完成、可见或释放事件 | 资源需求、完成延迟、启动间隔、在途与依赖限制，以及服务适用条件 |
| [**L1：CTA / SM 片上计算能力**](model/common/L1.md) | 给定片上数据供给、计算组织与资源配置，建立单 CTA 执行模型及单 SM 上驻留 CTA 的并发服务模型 | 输入块在 SMEM 就绪 → 结果在约定片上位置就绪；明确 CTA 或 SM 范围及驻留配置 | 单 CTA 的完成时间与吞吐量；单 SM 在指定驻留配置下的计算吞吐能力，以及缓冲、同步和资源竞争的限制 |
| [**L2：协作流水线（Collective）**](model/common/L2.md) | 一组协作线程的数据搬运、计算与同步；分别描述 mainloop 和 epilogue | mainloop：所需 GMEM 输入与初始累加状态满足条件 → 所分配 K 范围的累加结果就绪；epilogue：累加结果及所需输入就绪 → 当前输出 tile 可用 | 分块供给、缓冲复用、启动、稳态、排空及条件化阶段时间 |
| [**L3：完整 kernel / GEMM**](model/common/L3.md) | 组合协作阶段，并组织所有工作 tile 与 CTA/cluster 的执行 | 输入在 GMEM → 完整输出在 GMEM | 阶段组合、缓存复用、共享带宽、任务分配、波次、负载不均衡和完整完成时间 |
| [**L4：算子图执行**](model/common/L4.md) | 多个算子的依赖、数据传递与调度 | 算子图输入就绪 → 算子图输出完成 | 融合是否减少数据传输，额外调度和资源占用是否抵消收益；MegaKernel 是一种实现方案 |

CTA 是工作单元，SM 是承载工作单元的硬件资源。L1 分别描述单 CTA 与指定驻留配置下的单 SM 能力；两者不能直接等同。L0 的操作可由上层直接引用，不要求逐层调用。

## 文件职责

| 位置 | 维护内容 |
|---|---|
| `GOALS.md` | 按三类模型分节的问题、子问题与验收，实机验证，以及共用的有效性、C1/C2/C3、选题优先级、预算、停止与汇报规则；不含具体实验 |
| `mechanistic_modeling.md` | 当前半经验模型与理想机制模型的差别表 |
| `problem.md`、`workloads.yaml` | 问题定义与具体实例；保留旧模型的 2048³ FP16→FP32 实例作为参考 |
| `PLAN.md` | 当前优先级、推进顺序、依赖与任务入口；GH200实验状态另见access_rules/PLAN |
| `model/README.md` | 模型结构（共同输入→三个模型→组合与实机验证）、按模型导航、当前产物资格与文档维护位置 |
| `model/spec/`、`model/semi_empirical/`、`model/mechanism/program/`、`model/mechanism/hardware_rules/` | 按模型分的文件夹：规格计算入口、半经验公式与长 K 残差、程序依赖与硬件候选路径；分别链接硬件参数和原实测，目录见[模型入口](model/README.md#目录) |
| `model/common/inputs.md` | 三类模型共用的 case 字段：含义、单位、缺省与推导、来源和读取方；解析实现为 `case_inputs.py` |
| `model/common/interfaces.md` | 操作、资源、事件、计账、时间下界与程序图组织 |
| `model/common/L0.md`–`L4.md` | 各层边界、接口、通用约束与时间组合关系 |
| `model/mechanism/README.md` | 机制分解：TMA＋WGMMA GEMM 按机制的定价事件、证据类型、已辨识规则与缺口 |
| `model/mechanism/composition.md` | 机制组合首版：模块来源与资格、连接方式、输出与完成判据 |
| `hardware/<设备>/README.md`、`L0.md`–`L3.md` | 设备条件、分类资源参数、架构约束与证据；按实际新增内容分层 |
| `schemes/<方案>.md` | 适用范围、配置、工作量、资源使用、事件关系与方案推导；教学参数和硬件事实分开 |
| `experiments/README.md` | 三类模型共用的实机验证入口：证据的组织与协议位置 |
| `experiments/<设备>/` | 实验条件、接受条件、实测分析与 run 归档入口；GH200 的实验页按 16 个研究问题分在 `Qnn-*/` 目录 |
| [建模审查](modeling_review.md) | 历史审查与阶段性缺口记录；当前 GH200 状态以实验入口为准 |

同一硬件资源只定义一次，上层引用其参数；L4 当前没有独立新增的硬件参数。具体 tile、缓冲数和任务映射由方案给定，通用模型不固定 tc5a 等候选的选择。

[GH200 实验入口](experiments/gh200_sm90/README.md)保存资源微基准与完整 GEMM 预测验证入口。预测保留问题、方案、硬件和运行条件；实验记录实际条件与观测，原始 run 由实验文档链接，不把实测写成通用规则。
