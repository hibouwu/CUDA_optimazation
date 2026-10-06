# GH200 微基准结果总览

截至2026-10-06，17个实验家族已发布754条条件观测，24个排除项保留。各家族实际C和发布桥接均已通过；最终S21综合审查与S22完整换目录重放仍在进行。

## 较早发布的13个家族

| 实验 | 可引用条件数 | 能回答的问题 | 结果说明 |
|---|---:|---|---|
| 访存基线 | 11 | 指定访问循环的逻辑工作率及不稳定条件 | [S04](EXP-04-memory-baseline-results-v3.md) |
| FMA | 90 | 精度、依赖链与线程规模下的计算服务 | [S05](EXP-05-legacy-fma-results-v3.md) |
| mma.sync | 54 | 给定形状和参与 warp 下的计算服务 | [S06](EXP-06-legacy-mma-results-v3.md) |
| WGMMA | 82 | 提交、等待、warpgroup 与 SS/RS 条件的影响 | [S07](EXP-07-legacy-wgmma-results-v3.md) |
| FP8 / INT8 | 24 | 选定原生低精度形式的工作率 | [S08](EXP-08-low-precision-results-v2.md) |
| SMEM 访问 | 15 | 访问宽度、跨步、广播与读写组织的影响 | [S09](EXP-09-shared-memory-results-v1.md) |
| 矩阵搬运与交换 | 22 | 矩阵片段搬运及 shuffle 循环服务 | [S10](EXP-10-matrix-exchange-results-v1.md) |
| 同步与完成 | 11 | 指定参与范围及到达条件下的完整阶段成本 | [S11](EXP-11-synchronization-results-v1.md) |
| cp.async | 24 | 请求形式、有限流水与完成边界下的搬运服务 | [S12](EXP-12-async-copy-results-v1.md) |
| 全局读写组合 | 18 | 工作集和读写比例对逻辑工作率的影响 | [S13](EXP-13-global-duplex-results-v1.md) |
| TMA 1D bulk | 24 | payload、方向及单 CTA / 全 GPU 条件下的服务 | [S14](EXP-14-tma-bulk-results-v4.md) |
| TMA 2D tensor | 68 | 连续、padding、合法 swizzle 条件的影响 | [S15](EXP-15-tma-2d-results-v3.md) |
| 驻留与辅助资源 | 27 | 有限资源条件及辅助操作循环服务 | [S18](EXP-18-auxiliary-results-v1.md) |

[正式参数来源索引](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s21-seventeen-family-index-v1/index.json)引用原参数与审查记录，不统一换算不同单位，也不将不同作用范围的结果混合。

24 个排除项不是同一种原因：包含不稳定、没有目标原生指令或未取得所需资格的条件。其具体判定见各实验说明和原排除文件；仍保留结果与原因，不能作为合格标量参数使用。

## 旧三批配置的承接与历史保全

旧三批共有81个配置行、62个基本指令/启动组合，但保留输入、执行范围、循环长度和计时差异后是229个完整测量条件，其中222个计算条件、7个空窗口控制。229个条件均能对应到本轮正式summary，每项10次独立进程；另有4个新增WGMMA代表点。来源键完整保留，本轮计时与采样按对应v3合同执行。

FMA、MMA、WGMMA的实际配置数分别为93、55、85，总计233。计算参数与空窗口单位分开，计时控制不导出为FLOP工作率。旧L0及v1历史归档、旧入口共878个受保护文件的SHA保持一致；新结果没有改写旧记录。核对入口见[旧配置承接与历史保全检查](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s21-final-delivery-candidate-v1/legacy-and-history-final-check.json)，此项本地核对仍需最终独立验收。

## 最近完成采样的四个家族

| 实验 | 已测合法条件 | 实际 C | 公共发布 | 结果说明 | 脚本入口 |
|---|---:|---|---|---|---|
| TMA 并发与缓冲 S16 | 32；另 4 项容量排除 | 已通过C r2 | 已发布 | [S16结果](EXP-16-tma-concurrency-results-v1.md) | [S16 脚本](../../../../../microbench/gh200_resource_campaign/families/s16/README.md) |
| DSM / cluster / 多播 S17 | 42 | 已通过 | 已发布 | [S17结果](EXP-17-cluster-results-v1.md) | [S17 脚本](../../../../../microbench/gh200_resource_campaign/families/s17/README.md) |
| FP32 受控组合 S19 | 105 | 已通过 | 已发布 | [S19结果](EXP-19-fp32-combination-results-v1.md) | [S19 脚本](../../../../../microbench/gh200_resource_campaign/families/s19/README.md) |
| BF16 WGMMA 受控组合 S20 | 105 | 已通过 | 已发布 | [S20结果](EXP-20-bf16-combination-results-v1.md) | [S20 脚本](../../../../../microbench/gh200_resource_campaign/families/s20/README.md) |

S19 的[中文结果说明](EXP-19-fp32-combination-results-v1.md)已经整理。105个条件工作率已通过实际C和发布桥接；当前资格由准确签录及最终发布状态绑定。15 组直线拟合只作为诊断，不导出物理启动成本或每 tile 服务系数。transport、stage=4 的样本内最大拟合相对误差是 8.60%；正式样本稳定不能替代拟合准确性。

S16 原作业完成全部采样后因归档 ACK 等待超时而失败。原失败终态保留，数据随后完整保全并完成原环境重算，没有重新采样。该经历不改变数据资格要求，最终C r2和发布桥接均已通过。

## 怎样用于性能建模

引用一个值时，至少保留操作形式、tile 或 payload、参与线程及执行范围、资源占用、循环长度、输入、计时单位和完成边界。单 CTA 的 clock64 工作率与全 GPU 的 globaltimer 工作率回答不同问题；搬运的逻辑字节与实际 HBM 流量也需要分开。

例如，S19 的 overlap、stage=2、K=8，一个正式样本执行 N=305 个完整 K 序列：

- FMA 工作量：65536×8×305 = 159907840 FLOP。
- 本 CTA 主窗口：9084453 个 clock64 周期。
- 工作率：159907840÷9084453 = 17.6023631 FLOP/cycle。
- 每个完整 K 序列的周期：9084453÷305 = 29785.0918 cycle。

这里的 17.60 是该固定配置的经验工作率。改变 tile、线程布局、缓冲占用、同步或输出处理后，需要新的组合证据，不能直接当作通用 CUDA Core 峰值。原样本、哈希及完整条件见 S19 结果说明。

## 仍缺的证据与未纳入范围

计数器权限不足的实验没有取得物理 HBM 流量、实际缓存命中或动态指令计数资格。occupancy API 给出上限，不能单独证明实际驻留或全 SM 覆盖。预热收敛和样本稳定也不等于长期热稳态。

本轮不实施完整 GEMM、完整预测器、最优 tile 搜索、稀疏、Batched / Grouped 或 Distributed GEMM。Thor 的 TMEM / TCGen05 不属于 GH200 实验。

完整换目录重放入口为 [replay_frozen_family.py](../../../../../microbench/gh200_resource_campaign/replay_frozen_family.py)，r3入口已通过独立增量审查，完整重放正在执行。已验证原统计源码能精确复现最后四家族的284个条件，以及S08/S15的100个条件，共384个条件的六项统计。最终S22核查还需补S08最终32点和S15全部68点的换目录完整数组重算；它们已有原环境重算，但不能仅据此替代换目录证据。上述统计检查不代表完整数组、报告、图表和链接重放已经完成。可机读的当前交付清单见 [delivery-status.json](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s21-final-delivery-candidate-v1/delivery-status.json)。
