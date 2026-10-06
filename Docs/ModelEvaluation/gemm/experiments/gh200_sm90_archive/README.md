# GH200 微基准：目录与测量计划

本轮测量单 GPU GEMM 建模所需的资源与受控组合，不实施完整 GEMM、tile 搜索、Batched / Grouped 或 Distributed。cluster / DSM 属于单 GPU 内部通信。

## 当前 v2 计划与阅读入口

先读[结果总览](RESULTS.md)：已发布家族、最后四个家族的实际审查状态、参数使用边界和仍缺的证据。

- [PLAN.md](PLAN.md)：唯一完整计划，统一目标、S00–S22 范围、执行与测量规则及交付标准。
- [剩余六个实验的批量方案](RUN-REMAINING.md)：保留各实验的固定配置和连续运行方案；这些家族已完成测量与发布，当前接续最终离线验收。
- [FP8 / INT8 正式结果](EXP-08-low-precision-results-v2.md)：32 配置、320 正式进程已通过独立实际 B/C；24 条原生条件参数已发布，8 个 FP8 MMA 编译展开点保留逻辑工作率对照。两图、完整值来源、真实手算和[合格参数](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s08-main-execution/full-output-v1/runtime-v1/published-v1/qualified-parameters.json)齐全。
- [TMA 二维搬运完整结果](EXP-15-tma-2d-results-v3.md)：68 配置、680 正式进程及 68 个自身 pilot 已完成独立数据与发布审查，覆盖 34 单 CTA 和 34 全 GPU 配置。[68 条条件参数](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/node-sampling-v1/published-all68-v1/qualified-parameters.json)、两图及真实手算已发布；归档超时恢复保留原作业终态，未重新采样。
- [当前合格参数来源索引](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s21-seventeen-family-index-v1/index.json)：17 个正式结果家族、754 条条件观测，24 个排除项保留。S16、S17、S19、S20 的 GPU 采样与原环境重算已完成，各家族实际C与发布桥接均已通过；S21综合文档审查与S22完整重放仍在进行。
- [DSM / cluster / 多播实测](EXP-17-cluster-results-v1.md)：42个条件、420个正式进程已通过实际C和发布桥接，源请求、接收量、写后读回及同步阶段分别计量。
- [FP32受控组合实测](EXP-19-fp32-combination-results-v1.md)：105个条件、1050个正式进程已通过实际C，五类对照、stage与K曲线、真实手算及拟合误差齐全；发布已复核，拟合不授物理服务系数资格。
- [BF16 WGMMA受控组合实测](EXP-20-bf16-combination-results-v1.md)：105个条件、1050个正式进程已通过实际C，metadata、两种映射witness及图表已核对；发布已复核，9.53%拟合失配继续保留。
- [TMA并发与缓冲实测](EXP-16-tma-concurrency-results-v1.md)：32个合法条件、320个正式进程通过最终C r2，四个容量排除保留；原归档ACK失败终态不改，图布局修复与发布桥接均已复核，32条已接入公共索引。
- [GUIDE.md](GUIDE.md)：阅读归档、手算一条真实正式样本及离线重放。
- [COVERAGE.md](COVERAGE.md)：冻结的资源覆盖范围。
- [六类条件化参数的使用与真实 raw 手算](EXP-21-parameter-usage.draft.md)：279 条观测已通过独立发布桥接审查，10 个排除项保留。原四份导出和教学文档保留受审前的 draft / 候选标签；当前资格见 [发布桥接 B](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/reviews/S21-publication-bridge-B-review.json)。[参数文件](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s21-parameter-export-B/all-six-draft-r4/parameters.json)、[排除项](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s21-parameter-export-B/all-six-draft-r4/excluded.json)与[来源封套](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s21-publication-bridge-v1/envelope.json)共同使用；只适用于记录中的单位、工作量、完成边界和设备条件，不是硬件峰值或完整 GEMM 预测。
- [FMA 实测与固定长度分析](EXP-05-legacy-fma-results-v3.md)：93 点、930 样本已通过独立 C 审查并封存；90 个 payload 与 3 个计时控制分开，计数器缺口保留。正文的待审标签是受审前冻结版本，当前资格见 [C r2](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/reviews/S05-C-review-r2.json) 与 [COMPLETE](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/legacy_fma/formal-v3-b/COMPLETE)。
- [MMA 实测与固定长度分析](EXP-06-legacy-mma-results-v3.md)：55 点、550 样本已通过独立 C 审查并封存，见 [C 审查](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/reviews/S06-C-review.json) 和 [COMPLETE](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/legacy_mma/formal-v3-b/COMPLETE)。
- [WGMMA 实测、等待深度与 SS/RS 对照](EXP-07-legacy-wgmma-results-v3.md)：85 点、850 样本已通过独立 C 审查并封存，见 [C 审查](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/reviews/S07-C-review.json) 和 [COMPLETE](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/legacy_wgmma/formal-v3-b/COMPLETE)。MMA/WGMMA 正文和图表保留受审前待审标签；当前资格以这些审查与完成记录为准。
- [访存基线实测与图表](EXP-04-memory-baseline-results-v3.md)：S04 已通过独立 C 审查并封存，11 个 payload 配置稳定，两个不稳定配置和计数器权限缺口保留。正文与图表的“待 C 审查”标签是受审前冻结版本；当前资格见 [C 审查](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/reviews/S04-C-review.json) 和 [COMPLETE](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/memory_baseline/formal-v3-a/COMPLETE)。
- [cp.async 有限流水线实测](EXP-12-async-copy-results-v1.md)：24 点、240 样本已通过独立 C 审查、原运行环境严格封存及封存后重放，见 [C 审查](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/reviews/S12-C-review.json) 与 [COMPLETE](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/async_copy/formal-b3-v2/COMPLETE)。报告保留受审前待审字样；有效运输字节不等于物理 HBM 流量。
- [全局读写联合服务实测](EXP-13-global-duplex-results-v1.md)：18 点、180 样本已通过独立 C 审查、原运行环境严格封存和封存后重放；完整原包与精确依赖组装已交叉核对。见 [C 审查](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/reviews/S13-C-review.json) 与 [COMPLETE](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/global_duplex/formal-b3-v1/COMPLETE)。冻结报告保留受审前待审字样；速率为条件化逻辑读写请求量，不证明物理 HBM 流量或缓存命中。
- [同步与完成事件正式结果](EXP-11-synchronization-results-v1.md)：13配置、130正式进程，独立C与原ARM封存/同leaf重放通过；11条[条件参数](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s11-main-execution/published-v1/qualified-parameters.json)可导出，2个无目标WARPSYNC的逻辑warp点保留为排除项。三图、逐线程数值重算和手算已交付；严格summary重放需原ARM解释器条件。
- [矩阵搬运与 warp 交换正式结果](EXP-10-matrix-exchange-results-v1.md)：22配置、220正式进程已通过独立C、原ARM封存和同leaf换目录重放；分别给出矩阵逻辑B/cycle和shuffle的warp指令/cycle，两图、手算、CSV及[条件参数](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s10-main-execution/runtime-r2/published-v1/qualified-parameters.json)已交付。限定单CTA完整循环。
- [SMEM 访问正式结果](EXP-09-shared-memory-results-v1.md)：15 配置、170 个正式进程样本已通过独立 C、原 ARM 封存和同 leaf 换目录重放；两图、逐样本 CSV、手算及[条件参数](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s09-main-execution/device-revalidation-v2/published-v1/qualified-parameters.json)已交付。广播保留全部三批，CV 约 4.8%；数据是单 CTA 完整访问循环的逻辑 B/clock64 cycle，不能用作物理端口峰值。
- [TMA bulk双向正式结果](EXP-14-tma-bulk-results-v4.md)：S14的24配置、240正式样本已通过独立C、原ARM封存及换目录审查，参数、四图和中文说明均已交付；分别采用单CTA clock64和全GPU globaltimer包络，逻辑运输量不代表物理HBM流量。
- [驻留与辅助资源正式结果](EXP-18-auxiliary-results-v1.md)：S18 的27配置、270正式进程已通过独立C、ARM严格封存及换目录重放，参数、四组图及中文说明已发布；限定单CTA完整循环。见[条件参数](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/parallel/compute_onchip/s18-integration-v2/operations-v1/job733338-results/published-v1/qualified-parameters.json)和[最终接入验收](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/parallel/acceptance-20261005/compute_onchip/S18-final-integration/acceptance.json)。原六家族279条、S18的27配置和S14的24配置加上 S09 的15配置与 S10 的22配置、S11 的11条可导出配置，共378条条件观测，按原文件的单位和条件分别使用。
- [FP8 长累加差异观察](EXP-08-long-accumulation-result-v1.md)：8192 个输出全为 64，与数学参考 16384 不同。诊断观察与说明已独立审查，见 [结果文档复审](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/reviews/S08-long-accumulation-result-doc-review-r2.json)；不授数值正确性或性能资格。
- [FP8 四点边界观察与图表](EXP-08-boundary-result-v1.md)：31/32/33/64轮分别全为62/64/64/64，结果和教学图表已独审；[报告审查](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/reviews/S08-accumulation-boundary-report-review.json)绑定可逐字重生的r4。正文待审标签保留为受审版本，不授内部机制、数值B3或性能资格。
- [旧配置映射](EXP-03-compute-baseline.md)、[访存基线](EXP-04-memory-baseline-v2.md)、[FMA](EXP-05-legacy-fma.md)、[MMA](EXP-06-legacy-mma.md)、[WGMMA](EXP-07-legacy-wgmma.md)、[FP8/INT8](EXP-08-low-precision.md)、[SMEM 补充](EXP-09-shared-memory.md)、[矩阵搬运与 warp 交换草案](EXP-10-matrix-exchange.md)。

v2 总控为 `run_suite.py`，新结果按 `<suite-id>/<family>/<run-id>/` 归档，审查在 `<suite-id>/reviews/`。当前 suite 是 `20261001-resource-suite-v2`；其 `implementation_status.json` 记录推进状态。实验 A 通过、目标编译成功、GPU preflight、独立 B 和正式 C 是不同阶段，不能互相替代。目前尚未完成全轮交付，历史 v1 结果也不能当作 v2 已重测。

## 存放位置

| 内容 | 位置 |
|---|---|
| 已有计算探针与历史归档 | [microbench/gh200_l0](../../../../../microbench/gh200_l0/README.md)，保留原路径 |
| 新资源探针、运行和审查脚本 | [microbench/gh200_resource_campaign](../../../../../microbench/gh200_resource_campaign/README.md) |
| v2 原始数据、报告和图表 | `results/gh200_resource_campaign/<suite-id>/<family>/<run-id>/` |
| v1 历史资源运行 | `results/gh200_resource_campaign/<run-id>/`，保留原路径 |
| 实验设计与接受条件 | 当前目录，按资源家族分文件 |
| 硬件参数 | [hardware/gh200_sm90](../../hardware/gh200_sm90/README.md)，引用通过相应检查的参数 |
| 通用模型 | [model](../../model/interfaces.md)，保留 L0–L4 定义 |

沿用 Thor 的“源码、run 归档、实验说明”分工。不另建 `GH200/microbench/` 或第二套 Hopper 通用模型；未来完整算子实现可以放 `GH200/kernels/`，继续引用这些组件测量。

## v1 历史结果与原测量顺序

| 顺序 | 内容 | 状态与边界 |
|---|---|---|
| 0 | 旧计算测量的配置、工作量、周期、统计、SASS 与工件 | [EXP-01](EXP-01-compute-audit.md)：三批独立重算完成，资格缺口保留 |
| 1 | 设备属性、SMEM scalar stride、global 读写与 1:1 duplex | [EXP-02](EXP-02-memory-paths.md)：12×10 次独立进程实测完成；SMEM stride=1 波动待复测，NCU 无权限 |
| 2 | 矩阵搬运、warp 交换、同步和分配条件 | 后续按指令形式、线程范围与寄存器需求分别测量 |
| 3 | TMA payload、stride、并发请求与缓冲复用 | 先验证合法请求和完成，再测单 CTA 与全 GPU 路径 |
| 4 | DSM、多播与 cluster 驻留 | 按选定方案启用，不凭 cluster 数量推断收益 |
| 5 | 计算、供给与 epilogue 的重叠和竞争 | 前述组件条件匹配后测联合服务 |
| 6 | 完整 GEMM 预测对照 | 待实际方案及正确性要求明确后进行 |

上表保留历史结果和当时的顺序；当前执行范围和顺序以 v2 的 PLAN 为准。完整 GEMM 预测对照留作后续项目。

## 与 Thor 对齐的检查项

每批检查配置、作用范围、数值、工作量、计时与排空、独立重复、源码/二进制/SASS、环境、原始样本、审查结果和图表来源。

ROMEO 使用 Slurm、节点身份、CUDA/驱动及实际遥测描述环境，不照搬 Thor 的 MAXN。无 NCU 权限时仍保留有用的执行与计时记录，但不根据工作集大小宣称缓存命中或实际 HBM 流量，不补造动态指令计数和历史冻结记录。
