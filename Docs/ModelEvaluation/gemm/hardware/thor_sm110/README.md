# Thor/SM110

[问题与层级定义](../../README.md)

## 分层参数索引

| 参数文件 | 本文件维护的资源与约束 | 通用定义 |
|---|---|---|
| [L0](L0.md) | 计算规格、支持的操作路径及指令约束 | [基础操作服务](../../model/L0.md) |
| [L1](L1.md) | 片上容量、访问服务、分配、驻留与同步限制 | [CTA / SM 片上计算](../../model/L1.md) |
| [L2](L2.md) | 全局存储、缓存、搬运带宽及路径条件 | [Collective 流水线](../../model/L2.md) |
| [L3](L3.md) | 设备规模与实际可用资源范围 | [完整 kernel](../../model/L3.md) |

资源按计算、容量、带宽、驻留与同步等类别分别建表；同一资源只定义一次，上层引用。通用模型使用 [操作、资源与事件接口](../../model/interfaces.md)，硬件文件维护数值、支持条件及证据。当前 [L4 算子图模型](../../model/L4.md)复用已有资源，没有单独的硬件 L4 参数文件。

产品标称值、架构限制和历史测量分别记录。历史参数保留访问形状、统计范围与并发条件；方案专用的联合服务参数写入方案文件。

## 公共设备条件

官方规格核对日期：2026-09-30。本目录针对 Jetson T5000；官网上 T4000 等其他模组的数值不混入。产品标称值、架构上限与历史运行条件分别记录。

| 项目 | 值 | 来源 |
|---|---|---|
| 模组 | Jetson T5000，Blackwell 架构，第五代 Tensor Core | [产品规格][product] |
| CUDA 架构 | Compute Capability 11.0；架构专用代码使用 `sm_110a` | [官方 GPU 列表][cuda-gpus]、[SM110 源码][source] |
| GPU 最大频率 | 官网标称 1.57 GHz | [产品规格][product] |
| 模组功耗范围 | 40–130 W | [产品规格][product] |
| CPU | 14 核 Arm Neoverse-V3AE，最大 2.6 GHz | [产品规格][product] |
| 历史采集条件 | MAXN，SM 核心时钟记录为 1.575 GHz | [硬件配置][profile] |

官网频率按其显示精度记录为 1.57 GHz；上表历史采集采用 1.575 GHz，保留原归一化口径，不因产品页的显示精度而重算历史数据。

运行条件来自仓库已有采集，不代表当前设备状态。各层引用同一设备与频率条件，原始统计范围、适用条件和证据随参数保留。

## 关联方案

[tc5a](../../schemes/tc5a.md) 为历史候选参考，当前尚未选用；[FP32 分块模型](../../schemes/fp32_simt_tiled.md)中的教学参数不作为本设备数值。

[product]: https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/
[profile]: ../../../../../scripts/sm110_gemm_model/profiles/thor_sm110.json
[source]: ../../../../../GEMMsm110/include/
[cuda-gpus]: https://developer.nvidia.com/cuda/gpus
