# GH200 TMA 2D：68 配置的完成服务

本次有限矩阵已全部测量：34 个单 CTA 配置、34 个全 GPU 配置，共 68 个自身 pilot 和 680 个正式进程。前 56 点已发布；最后 12 个全 GPU padding 配置已完整收集并通过原生重算，独立结果审查正在进行。

## 实验问题与配置

给定二维 16-bit payload，GMEM 行跨度和 SMEM 布局怎样影响完成搬运的速率？输入块准备完成之后，循环发起 TMA，等待约定的完成事件，再进入下一次搬运。

| 条件 | 本轮配置 |
|---|---|
| 方向 | GMEM→SMEM、SMEM→GMEM |
| 有效 payload | 1/4/8/16/32/64 KiB |
| 布局 | continuous-none、padding-none；SW128 采用 1/4/8/16/32 KiB 子集 |
| 数据 | 非均匀 UINT16 模式；各 CTA 使用专属的 32 个全局槽 |
| 范围 | 单 CTA、全 GPU；实际 CTA 数来自资源查询 |
| 完成 | G2S 的 mbarrier 完成和 CTA 会合；S2G 的 commit、完整 wait0 和 CTA 会合 |

同一轮只有一个请求，完成后再发下一次。该实验测完整完成服务；软件 stage 和多请求并发由 S16 另测。

## 计量与真实算例

二维块宽高为 W、H 时，有效 payload 为 `Q=2WH` 字节。B 个 CTA 各完成 N 次搬运，主计时窗口内工作量为 `X=BNQ`。padding 占用地址空间，但不计入有效 payload。

单 CTA 用自身 clock64 差，结果为 B_transport/clock64 cycle/CTA；全 GPU 用全部 CTA 的 globaltimer 完成包络，结果为 GB_transport/s/GPU。这里的 B/ns 在数值上等于十进制 GB/s。

最后一组的真实样本 `gmem_to_smem_1kib_padding_none_all_gpu`，第 0 批第 0 次进程：

`B=528，N=16501，Q=1024 B`，因此 `X=528×16501×1024=8921628672 B`。

包络为 `5249312 ns`，得到 `8921628672/5249312=1699.580568 GB_transport/s/GPU`。该配置十进程中位数为 `1713.153341`。计时结束后导出的两份 tile 合计 `1081344 B`，不计入主循环搬运量。原始字段和 SHA 见[算例](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/node-sampling-v1/reports-last12-v1/padding_through16kib/worked-example.json)。

CUDA event 包含初始化和计时后的导出，用于校准与一致性检查；主指标仍取设备内部的循环完成窗口。

## 结果与建模含义

68 点均在第一批十进程后稳定，最大 CV 为 1.58%。预热采用 8–30 次有界窗口，末五窗口 CV≤2%；它不证明长期热稳态。图中误差条为十个进程的样本标准差。

![单 CTA 完成服务](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/node-sampling-v1/report-all68-v1/one_cta.png)

![全 GPU 完成服务](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/node-sampling-v1/report-all68-v1/all_gpu.png)

本次全 GPU 结果中，padding 的 S2G 速率在 4–32 KiB 区间约为 1.33–1.45 TB_transport/s，连续布局约为 3.81–3.83 TB_transport/s；G2S 的布局间差异较小。两个方向不能共用一个与 stride 无关的 TMA 速率常数。比较跨 payload 的点时还需保留各自的 CTA 数、N 和资源条件。

这些曲线描述给定输入、布局、循环及等待方式的联合服务。它们没有测出物理 HBM 字节数、缓存命中率或 TMA 队列深度，不能直接替代完整 GEMM 的搬运阶段模型。

## 数据来源与复现

两部分使用同一 GPU UUID `GPU-7c184a2e-41ea-3d2b-df5b-1699c95fc1fe`、SM90a/CUDA 12.9，并分别保留自身 pilot 和采样条件。[前 56 点说明](EXP-15-tma-2d-results-v1.md)保留其原数据、资格和解释器证据边界。

最后 12 点由 GPU 作业 734312 完成。采样之后，归档确认超过等待窗口，该作业以 FAILED 75:0 结束；CPU 734378 保留原节点目录并完成收集，未重测 GPU。完整收集核对了 943 个文件，约 1.79 GB 外层归档；CPU 734445 对 12 个 pilot 和 120 个正式包完成原生逐值重算。采样有效性、原作业的归档超时和后续保全恢复分别记录。

- [68 点结果表](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/node-sampling-v1/report-all68-v1/results.csv)、[680 进程 CSV](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/node-sampling-v1/report-all68-v1/samples.csv)、[最后 12 点中文表](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/node-sampling-v1/report-all68-v1/RESULTS.zh.md)。
- [最后 12 点候选参数](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/node-sampling-v1/report-all68-v1/last12-parameter-candidates.json)、[合并图表来源](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/node-sampling-v1/report-all68-v1/sources.json)。前 56 点原有参数资格不变，最后 12 点等待独立 C。
- [最后 12 点完整收集索引](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/node-sampling-v1/collection-job734312-r2-CPU734378/collection-index.json)、[原始归档](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/node-sampling-v1/collection-job734312-r2-CPU734378/actual.tar)、[原生重算记录](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/node-sampling-v1/native-replay/complete734445/collection-index.json)。

单组图表由公共 [report_tma_tensor_2d.py](../../../../../microbench/gh200_resource_campaign/report_tma_tensor_2d.py)读取真实 raw 和封存重算结果生成。以新增的前八点为例：

```sh
python3 -B /home/jianyeshi/Note/CUDA/CUDA_optimazation/microbench/gh200_resource_campaign/report_tma_tensor_2d.py \
  --run /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/node-sampling-v1/collection-job734312-r2-CPU734378/actual/formal-padding_through16kib \
  --replay /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/node-sampling-v1/native-replay/complete734445/native-replay-provisional-r3/padding_through16kib.json \
  --output /tmp/gh200-s15-padding-report
```

输出目录须尚不存在。移动完整归档后，保持 `repo/` 与四个 `formal-*` 目录同级，再修改命令前缀。图表生成不运行 GPU；完整数组重放使用归档内的 `repo/audit_formal.py`。重新测量的顺序与存储条件见[批量方案](RUN-15-tma-2d.md)。
