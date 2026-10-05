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

[微基准实验与后续计划](../../experiments/gh200_sm90/README.md) · [旧数据独立复审](../../experiments/gh200_sm90/EXP-01-compute-audit.md)

## 已审条件化服务观测

六个正式家族（访存基线、FMA、MMA、WGMMA、cp.async、全局读写联合服务）已有 279 条条件化观测取得有限发布资格，10 个排除项继续保留。具体配置、单位、资源与完成边界见[参数使用说明](../../experiments/gh200_sm90/EXP-21-parameter-usage.draft.md)，结构化数值见[参数文件](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s21-parameter-export-B/all-six-draft-r4/parameters.json)。资格由[独立发布桥接审查](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/reviews/S21-publication-bridge-B-review.json)和完整来源提供，原冻结文件的 draft 标签保留。

S18 又完成27个单CTA驻留与辅助资源配置，已通过独立C、原ARM严格封存和换目录重放；见[实验结果与手算](../../experiments/gh200_sm90/EXP-18-auxiliary-results-v1.md)和[27条条件参数](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/parallel/compute_onchip/s18-integration-v2/operations-v1/job733338-results/published-v1/qualified-parameters.json)。原六家族279条、S18的27配置与S14的24配置加上 S09 的15配置与 S10 的22配置、S11 的11条可导出配置合计378条；S18的C/N和逻辑Q/C保持自身单位。SMEM预留不等于SMEM搬运带宽，occupancy仍是API上限，实际carveout未知。

S14完成24个1D TMA bulk双向配置，见[正式实验与手算](../../experiments/gh200_sm90/EXP-14-tma-bulk-results-v4.md)及[24条条件参数](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/parallel/tma_cluster/analysis/s14-formal-733392-qualified/parameter-candidates.json)。参数内部已获得C后资格；单CTA为B_transport/clock64_cycle/CTA，全GPU为GB_transport/s/GPU。它们包含提交、等待和CTA控制，保留32-slot、网格与资源条件，不解释为物理HBM峰值或缓存命中证明。

S09 完成15个 SMEM scalar/vector、stride、广播和独立读写配置，见[正式结果与手算](../../experiments/gh200_sm90/EXP-09-shared-memory-results-v1.md)及[条件参数](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s09-main-execution/device-revalidation-v2/published-v1/qualified-parameters.json)。固定256线程、单CTA、64 KiB动态SMEM、8192轮；访问、checksum和同步均在clock64窗口内。每条保留中位数、范围、CV和批次数；广播全部30样本的CV约4.8%。这组结果不替代物理SMEM端口带宽、多CTA竞争或Tensor Core取数能力。

这些是固定实验条件下的服务观测。选择时同时匹配指令、类型、依赖与并发、线程/CTA 范围、循环长度、设备和工具链；逻辑字节不自动转换为物理 HBM 流量。FP8/INT8、张量映射或并发TMA、DSM和受控组合仍需各自完成测量与审查，不能从已测家族补出未知参数。

## 参数覆盖状态

当前范围为单 GPU 的 Single GEMM；Batched / Grouped 暂不考虑，Distributed 暂不实现。cluster / DSM 属于单 GPU 内部路径，按方案需要使用。

- [L0](L0.md)：已补 local、矩阵搬运、warp 交换、辅助及原子操作路径，明确异步分组与完成条件，并引用 7 个代表计算配置的条件化复测值。
- [L1](L1.md)：已补 CUDA occupancy 分配舍入、`setmaxnreg`、溢出路径，以及命名 barrier 与 `mbarrier` 的不同资源约束。
- [L2](L2.md)：已补 tiled TMA 的请求/描述符约束、输出完成、DSM 与多播需求；实际 L2 cache 为 60 MiB；新增普通 global 请求服务，物理 HBM/TMA/DSM 带宽仍待确认。
- [L3](L3.md)：已补 cluster 的共同调度、大小及驻留约束，区分参考频率、局部周期和整卡完成时间。

规则依据为按 SM90 筛选的 PTX ISA 8.8、CUDA 13.1 文档和本机 CUDA 13.0 occupancy 头文件；经验数据来自已有 CUDA 12.9 [审查复测][audit]。2026-10-01 新增 [EXP-02](../../experiments/gh200_sm90/EXP-02-memory-paths.md) 的 120 次独立进程资源测量及旧三批独立复审；物理在途容量、裸指令延迟、共享通路服务和完整 kernel 预测仍未闭合。

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

S10 完成22个矩阵片段搬运与warp交换配置，见[正式结果与手算](../../experiments/gh200_sm90/EXP-10-matrix-exchange-results-v1.md)及[条件参数](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s10-main-execution/runtime-r2/published-v1/qualified-parameters.json)。矩阵是单CTA、32线程、8192轮、8 KiB动态SMEM；shuffle比较32/128线程与1/4条独立流。逻辑B/cycle与warp指令/cycle分开，窗口包含本循环的地址、消费/生成、同步和drain；不能作为裸指令延迟、物理端口峰值或全GPU能力。

S11完成同步与fence的13个单CTA配置，见[结果与手算](../../experiments/gh200_sm90/EXP-11-synchronization-results-v1.md)及[11条条件参数和2个排除项](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s11-main-execution/published-v1/qualified-parameters.json)。CTA/mbarrier按整个CTA共同完成的phase计量，fence采用零未完成请求条件下的warp指令序列；偏斜点包含每选定线程每阶段256条依赖MAD。完整窗口不是裸指令延迟，独立arrival launch的时间差不从正式循环扣除；fence也不作为TMA/cp.async/WGMMA统一完成事件。原ARM严格封存/同leaf重放通过，本机3.14的5处CV末位差异失败明确保留。
