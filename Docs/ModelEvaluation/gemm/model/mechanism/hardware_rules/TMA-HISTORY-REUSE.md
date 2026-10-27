# TMA 输入路径：历史证据复用与扩展

更新：2026-10-13。首轮离线盘点，服务 [微架构研究计划](../../../PLAN.md) 的单目的 TMA 输入路径。结论是复用现有实验，按硬件问题组织共同证据，再补必要的衔接对照。原始目录、失败样本、冻结成绩和发布身份保持原样。

## 本次完成与证据边界

已阅读 EXP-14/15/16、R05、R12、R13 及 R10 的相关记录，抽读搬运与消费者源码，核对发布文件，并重算部分已导出的逐进程统计。没有执行 GPU、新拟合或完整原始数组/SASS 重验。

| 家族 | global→shared 已发布条件 | 历史独立进程样本 | 本次核对 |
|---|---:|---:|---|
| EXP-14 / S14 | 12 | 120 | CSV 与已发布记录的数值、样本数、单位一致；读取原独立审查与交付身份，本次未逐原始进程重算 |
| EXP-15 / S15 | 34 | 340 | 从 samples.csv 的工作量/分母重算速率与各条件中位数，对照 qualified-parameters；核对发布链 |
| EXP-16 / S16 | 16 | 160 | 同上；输入另保留 2 个 S4R4 容量拒绝条件，不计入有效条件 |
| 合计 | 62 | 620 | 只统计这三个家族的输入方向，不包含 R05/R12/R13，也不代表一个可混合拟合的数据集 |

S15/S16 的算术重算实际覆盖两方向共 100 个有效条件、1000 个 samples.csv 行；发布文件及审查链接共 13 项 SHA 检查通过。S14 是发布摘要间核对。具体路径与检查范围见 [本次核对结果](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/S16-single-cta-control/20261013-tma-history-reuse/audit.json)。samples.csv 重算不等于原始二进制数据重验。

S15/S16 的原 CSV 保留 `qualified=false`，不能仅据此否定后续发布，也不能直接改成 true。S15 由 [发布桥接](../../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/node-sampling-v1/published-all68-v1/publication-bridge.json)绑定合格参数；S16 由 [最终状态](../../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s16-main-execution/published-v1/publication-final-state.json)绑定独立桥接审查与 32 个两方向条件。它们的资格均为限定条件下的运输服务，不是物理 HBM 带宽、内部队列深度或通用机制资格。

## 统一的硬件对象

源是 GPU 设备分配中的 `.global` 数据，目的为发起 CTA 的本地 SMEM。抽读的 R05、R13 公共 `DeviceBuffer` 与 S16 正式入口均使用 `cudaMalloc`；逐 run 的冻结源码身份仍随原记录核验。小工作集、反复访问和工作集超过 L2 容量分别是运行条件，不直接证明 L2 命中率或 HBM 流量。

证据表须保留以下字段：源分配及实际地址集合、CTA 间共享关系、复访距离、缓存准备、请求指令与形状、目的布局、消费者、软件槽数、线程/CTA/SM 组织、资源占用、计时端点、单位、设备与 run 身份、观察扰动资格。

静态源码入口：[R05](../../../../../../microbench/gh200_resource_campaign/access_rules/probes/r05.cu)、[R13](../../../../../../microbench/gh200_resource_campaign/access_rules/probes/r13.cu)、[设备分配](../../../../../../microbench/gh200_resource_campaign/access_rules/probes/r00_common.hpp)、[S16 正式入口](../../../../../../microbench/gh200_resource_campaign/families/s16/microbench/gh200_resource_campaign/probes/tma_stage_request_formal.cu)。

## 可复用证据

| 历史研究 | 已研究的条件与端点 | 对新研究的作用 | 不能直接合并的部分 |
|---|---|---|---|
| [EXP-14：1D bulk](../../../experiments/gh200_sm90/Q02-tma-request/EXP-14-tma-1d.md) | 1–64 KiB；单 CTA/整卡；单槽单请求，完成等待后 CTA 会合 | 单请求尺寸响应与基础完成协议 | 包含控制、等待和同步，不是裸请求延迟；多 CTA 速率不是单 CTA 参数 |
| [EXP-15：2D tensor](../../../experiments/gh200_sm90/Q02-tma-request/EXP-15-tma-2d.md) | 连续、SW128、padding；每 CTA 轮换 32 个 global 槽；每轮单请求 | 请求形状、源行距、目的布局对照 | 与 1D bulk 的指令、shape、迭代次数不同，不能从两条曲线之差直接定价 tensor 开销 |
| [EXP-16：stage×请求数](../../../experiments/gh200_sm90/Q02-tma-request/EXP-16-tma-pipeline.md) | 16 KiB/request；S/R=1/2/4；单 CTA/整卡；完成与槽释放，无正式逐 item 全值消费 | 已覆盖软件容量和分组，不必再做同一轮 S×R 扫描 | SMEM 用量、驻留和 CTA 数随配置变化；不是硬件在途容量扫描 |
| [R05-A 修正版/B](../../../experiments/gh200_sm90/Q02-tma-request/R05-async-lifecycle.md) | TMA 输入等待位置；B 为 stage1/2/4×轻重 FFMA 消费者，固定四个物理槽 | 控制消费工作与活动槽数，复用合法等待/发布/消费/复用链 | B 的 TMA 请求为16 KiB，cp.async 对照为2 KiB；不同载荷不作硬件路径等价比较。原 A 的 invalid_sass_schedule 数据排除，使用独立修正版 |
| [R05-D](../../../experiments/gh200_sm90/Q02-tma-request/R13-async-retirement.md#r05-d) | 48 KiB/代、160线程、132 CTA、stage2/4；共享15 MiB源 vs 独立241 MiB源；立即释放消费者 | 已有源复访与多 CTA 供给对照，无须从零重建 | 工作集大小、复访距离与跨 CTA 共享同时变化，不能只归因于缓存层次；与 R13 的消费者及分母不同 |
| [R13 序列与事件](../../../experiments/gh200_sm90/Q02-tma-request/R13-async-retirement.md) | 32/48 KiB×stage1/2/4×直接退休/FFMA/WGMMA；固定四槽，384线程，共享小源；ready/retire/reuse | 最接近当前需要的消费与释放证据，可复用 plain 与有资格的局部事件 | 三类事件来自不同调用，不能拼绝对时间线；部分观察扰动超限，只保留定性顺序 |
| [R12 多路径竞争](../../../experiments/gh200_sm90/Q07-smem/R12-smem-path-contention.md) | 独立 SMEM 区域中的 TMA、WGMMA、STS；独立/串行/并发及阶段位置控制 | 扩展目的端共享服务的竞争问题，复用受控的角色与布局设计 | 仅单 CTA、小源、固定工作量；额外窗口不唯一证明共享端口；v1/v2 消费归约缺陷不沿用 |
| [R10/R13 后续定位](../../../experiments/gh200_sm90/Q03-source-service/R10-layout-cache.md) | 行距、源分配/执行方向、背景 CTA、producer延迟、首次ready | 给机制候选提供必须满足的反例与共享范围线索 | 是完整 GEMM 背景；不把局部端点当裸 TMA 事件，也不反向混入旧冻结预测 |

R05-C/E 属于输出源复用与 SMEM→global 写回，EXP-17 属于 cluster/DSM/多播，保留为后续独立方向；它们可约束组合研究，但不混入当前输入路径服务价格。

## 已有证据已经限制了什么

1. **缓冲容量不能独自描述推进。** EXP-16 同为64 KiB缓冲，S1R4约28.5、S2R2约39.8 B/cycle。它提示请求分组与软件推进必须保留，不证明内部队列容量。两种配置各自的循环与资源条件仍须跟随数值引用。
2. **消费者会改变完整序列率。** R13 的stage4、32 KiB输入，直接退休与WGMMA条件报告约83.077与58.046 B/SM-covered-cycle；48 KiB下约85.031与45.337。不能挑一个数作为所有程序的每SM供给上限。
3. **源复用对照已有，但归因仍混合。** R05-D 的共享小源与独立大源约55.45与15.26 B/cycle/CTA；两者同时改变多个源条件，未证明“纯L2”和“纯HBM”服务。
4. **观察软件返回不足以命名硬件完成。** R13 的reuse窗口从producer成功acquire之后开始，到下次A请求前结束，不包含先前empty等待。早期事件是分调用观察，不能组合出一条实测完整时间线。
5. **后续ready定位已有反例价值。** job739653的释放边界观察排除了本协议下“ready不变、仅consumer晚返回”的解释；请求排队、数据服务及完成可见性仍未分开。该批观察也有局部扰动，不转成裸延迟参数。[事件边界与原报告](../../../experiments/gh200_sm90/Q02-tma-request/R13-async-retirement.md#ready-observer)

## 合并方式与研究扩展

合并研究问题、观察接口和证据索引；不同 run 的原始数据保持独立。每项结论先在原条件下成立，再通过匹配对照判断参数能否迁移。

| 合并后的子题 | 复用来源 | 最小扩展方向 |
|---|---|---|
| 请求服务粒度与分组 | EXP-14/15/16 | 优先从相同16 KiB载荷找衔接点，统一源访问、目的容量、线程与完成窗口；bulk/tensor保留独立参数，验证后再决定能否共享 |
| 请求准入与容量回收 | EXP-16＋R05-B＋R13 | 分开控制软件槽容量、请求发出组织和消费者推进；先复放等待/补发关系，再判断缺哪个发出或释放端点 |
| 源复访与共享范围 | R05-D＋R10 | 分开检查总地址并集、每CTA复访距离和CTA间共享关系；无法同时控制的因素明确保留，不把小/大工作集差命名为缓存命中差 |
| 目的端竞争与完成可见性 | R12＋R13 ready定位 | 复用独立地址区域与阶段匹配控制，比较目标操作在不同背景下的完成响应；可作为下一层扩展，不立即扫描所有组合 |

不同卡、不同指令形式、不同消费者、不同流量分子或时间分母的数值不混合定值。即使同卡，也需核对编译资源、缓存准备、时钟条件与观察窗口。S14 与 R05 记录同一 GPU UUID，也不足以单凭同卡认定可直接拼接；S16、R12、R13 还涉及其他 GPU 身份。

## 接下来先做什么

先对 **16 KiB 1D bulk global→本地SMEM** 提取 EXP-14、EXP-16 和 R05-B 的最小衔接条件表：单请求、请求分组、活动stage与消费者分别记账。这个载荷三组都已有记录，适合检查能否减少新测量；它不代表已测得16 KiB的统一硬件价格。

完成端点和条件核对后，先用现有记录约束“软件发出/等待组织已足以解释”和“还需有限共享服务状态”这类候选。若相同预测量对应不同结果，先排除该模型形式；若剩余机制需要新观察才可区分，再列少量实机衔接对照。R13的32/48 KiB真实消费者与R12竞争留作组合迁移约束。

当前可直接开始的是条件表、端点图与离线候选反例检查。完整原始数据重验按实际需要选择受影响的run；没有新证据需求时不重跑整套历史实验。下一次GPU任务应补“哪两个候选仍无法区分”，而不是扩大已有参数矩阵。

后续进展：已完成上述15个单CTA衔接条件的首版[输入事件模型与离线比较](TMA-INPUT-EVENT-MODEL.md)。R05的容量候选仍未区分，S16的stage4响应存在缺口；没有新增GPU数据。本文件的盘点统计与核对范围保持原口径。
