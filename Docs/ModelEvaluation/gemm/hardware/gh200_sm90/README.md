# GH200/SM90

[问题与层级定义](../../README.md)

[ROMEO GH200 访问说明](../../../../../docs/romeo_gh200_access.md)

## 分层参数索引

| 参数文件 | 本文件维护的资源与约束 | 通用定义 |
|---|---|---|
| [L0](L0.md) | 计算规格、支持的操作路径及指令约束 | [基础操作服务](../../model/L0.md) |
| [L1](L1.md) | 片上容量、访问服务、分配、驻留与同步限制 | [CTA / SM 片上计算](../../model/L1.md) |
| [L2](L2.md) | 全局存储、缓存、搬运带宽及路径条件 | [Collective 流水线](../../model/L2.md) |
| [L3](L3.md) | 设备规模与实际可用资源范围 | [完整 kernel](../../model/L3.md) |

资源按计算、容量、带宽、驻留与同步等类别分别建表；同一资源只定义一次，上层引用。通用模型使用 [操作、资源与事件接口](../../model/interfaces.md)，硬件文件维护数值、支持条件及证据。当前 [L4 算子图模型](../../model/L4.md)复用已有资源，没有单独的硬件 L4 参数文件。

产品标称值、架构限制和历史测量分别记录。历史参数保留访问形状、统计范围与并发条件；方案专用的联合服务参数写入方案文件。

[微基准实验与后续计划](../../experiments/gh200_sm90/README.md) · [旧数据独立复审](../../experiments/gh200_sm90_archive/EXP-01-compute-audit.md)

## 已测条件服务与覆盖

17 组资源实验共有 754 条条件观测，另有 24 条排除记录保留。各实验的结果、图、真实手算和限制统一见[结果总览](../../experiments/gh200_sm90/README.md)，结构化数值与来源见[参数来源索引](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s21-seventeen-family-index-v1/index.json)。原参数文件中的历史 draft 标签保留，不影响数值使用。

后续[访问与供给规则](../../experiments/gh200_sm90/access_rules/README.md)补充163个默认资源条件及20个组合观测，包含旧坐标重测，不与上述754条直接相加。展开版依赖序列、WGMMA major对照、异步完成与完整输出边界均保留具体条件；固定CUTLASS的[聚合阶段模型](../../../../../results/gh200_resource_campaign/access_rules/20261006-cutlass-wait-matching/published/parameters.json)属于特定实现/输入下的条件预测，未填成通用硬件端口或裸延迟参数。

| 资源 | 代表实验 | 使用时必须保留的条件 |
|---|---|---|
| 计算 | FMA、mma.sync、WGMMA、FP8/INT8 | 精度、指令形状、依赖链、warp/warpgroup数、提交与等待 |
| 片上访问与同步 | SMEM、矩阵搬运、shuffle、barrier/fence | 地址和访问宽度、参与范围、校验与同步、完整循环及排空 |
| 全局与异步搬运 | global联合读写、cp.async、TMA 1D/2D | 工作集、读写比例、payload、方向、stride/swizzle、完成与复用事件 |
| 分配与辅助操作 | 驻留上限、local/spill、地址、转换、原子 | 实际编译资源、资源预留、争用条件与各自计量单位 |

这些结果是固定实验条件下的服务观测。单CTA的clock64周期与全GPU的globaltimer完成时间分开；逻辑运输字节不自动等于物理HBM流量。访问、校验、同步或drain在主窗口内的实验不能当作裸指令延迟。occupancy API是上限，不单独证明实际驻留；SMEM预留量也不等于搬运带宽。

TMA 并发、DSM/多播和两套受控组合的 GPU 采样及 CPU 重算均已完成，已计入 754 条（S16 32 条，S17、S19、S20 共 252 条）。原始 summary 中的 stable 标记只表示样本稳定，不代表结果可用于其他条件。

本轮采用单GPU问题范围，Batched/Grouped及Distributed暂不实施。cluster/DSM是单GPU内部路径。GH200不继承Thor的TMEM/TCGen05、tc5a缓冲配置或调度假设。

物理在途容量、裸指令延迟、更广泛共享通路服务和完整kernel预测仍需专门证据。软件stage扫描不证明物理队列深度；本轮微基准不宣称完整GEMM预测已经闭合。

## 产品与架构条件

官方规格核对日期：2026-09-30。产品数值来自 NVIDIA GH200 数据表（3773000，MAR25，第 4 页），架构限制采用 CUDA 13.1 的 CC 9.0 表和 Hopper 调优指南。表中单 GH200 与双 GH200 NVL2 分开，本文采用单 GH200 列。

| 项目 | 官方规格或范围 | 来源 |
|---|---|---|
| 组成 | 一颗 Grace CPU 与一颗 Hopper GPU | [产品页][product] |
| CUDA 架构 | Compute Capability 9.0；架构专用代码使用 `sm_90a` | [官方 GPU 列表][cuda-gpus] |
| CPU | 72 核 Arm Neoverse V2 | [产品数据表][datasheet] |
| 模组功耗范围 | 450–1000 W，包含 CPU、GPU 和内存 | [产品数据表][datasheet] |
| GPU 内存版本 | 96 GB HBM3 或 144 GB HBM3e，容量与带宽配对见 L2 | [产品数据表][datasheet] |

GPU 产品算力见 [L0](L0.md)，片上容量及驻留上限见 [L1](L1.md)，内存与 NVLink-C2C 规格见 [L2](L2.md)。未公开的指令延迟和有效并发服务率继续保留为待确定。

## 设备与参数状态

本地已有 2026-09-30、`romeo-a057` 的[设备归档][device-archive]：设备名 `NVIDIA GH200 120GB`、CC 9.0、132 SM、CUDA Runtime 可见全局内存 102005473280 B（95 GiB）。同次 [nvidia-smi 记录][device-env]的 FB Memory Total 为 97871 MiB。设备名中的 120GB 不能作为 GPU HBM 容量；官方数据表把 120/240/480 GB 列为 Grace LPDDR5X 容量选项。

2026-10-01 在同一节点的新[CUDA Runtime 查询](../../../../../results/gh200_resource_campaign/20261001-memory-c/device.jsonl)复核了 132 SM、95 GiB 可见内存，并补得 L2 cache 60 MiB、每 SM SMEM 228 KiB、CTA opt-in SMEM 227 KiB。归档容量与 96 GB HBM3 产品档接近，但不能仅由名称或可见容量确定所有 SKU 参数；HBM 带宽先保留产品档条件。动态频率保留在新 run 遥测中；普通 global 混合请求已有测量，更广泛联合服务与实际物理流量仍需证据。教学算例不代替本设备的完整 GEMM 预测。

[FP32 分块模型](../../schemes/fp32_simt_tiled.md)提供组合推导示例；GH200 不继承 Thor 上 tc5a 的缓冲配置和调度假设。

[product]: https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/
[datasheet]: https://dam-cdn.nvd.orangelogic.com/AssetLink/tkno03fa645gb7f33w60rf200d330420.pdf
[cuda-gpus]: https://developer.nvidia.com/cuda/gpus
[device-archive]: ../../../../../microbench/gh200_l0/results/20260930-initial/summary.json
[device-env]: ../../../../../microbench/gh200_l0/results/20260930-initial/environment.txt

[audit]: ../../../../../microbench/gh200_l0/AUDIT.md
