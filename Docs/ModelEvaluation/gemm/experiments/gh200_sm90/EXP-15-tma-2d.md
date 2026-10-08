# EXP-15：TMA 2D tensor

[结果总览](README.md)

## 结论

- 连续布局下，整卡读写都在 payload ≥8 KiB 时达 3.8–3.9 TB/s；**2D tensor 写不像 1D bulk 写（[EXP-14](EXP-14-tma-1d.md)，2.7 TB/s）那样慢**，写 4 KiB 起即达 3.8 TB/s。
- SW128 swizzle 与无 swizzle 结果相同，在本实验范围内没有代价。
- **本实验的 padding 布局下写大幅下降**：每行 128 B 有效数据后留 16 B，行距 144 B（64 KiB 时 256 B 数据、行距 272 B），整卡 4–32 KiB 写只有 1.33–1.45 TB/s，约为连续布局的 35–38%；读降到约 3.36 TB/s（降 13%）。1 KiB（8 行）时写反而不降。
- 原因见下方“后续结果”：行起点只按 16 B 对齐；行距 160 B 或 256 B 时不下降。
- 单 CTA 一次一个请求：连续读 64 KiB 达 55 B/cycle（1D bulk 同 payload 为 31.6，但 box 形状、循环次数不同）；padding 读 34 B/cycle。

## 配置

16-bit 数据（UINT16 位模式），rank 2 tiled tensor map，elementStrides=(1,1)，interleave、L2 promotion、OOB fill 均为 none。每 CTA 128 线程、thread0 发起，独占 32 个全局槽轮流使用。

| payload Q | box `(W,H)` 元素 | 连续行距 | padding 行距 | 测试布局 |
|---|---|---:|---:|---|
| 1 KiB | (64, 8) | 128 B | 144 B | 连续 none、连续 SW128、padding none |
| 4 KiB | (64, 32) | 128 B | 144 B | 同上 |
| 8 KiB | (64, 64) | 128 B | 144 B | 同上 |
| 16 KiB | (64, 128) | 128 B | 144 B | 同上 |
| 32 KiB | (64, 256) | 128 B | 144 B | 同上 |
| 64 KiB | (128, 256) | 256 B | 272 B | 连续 none、padding none（SW128 最内维不能超过 128 B） |

- 工作量：\(B\times N\times Q\)，\(Q=2WH\) 只计有效数据，padding 字节不计。
- 读（GMEM→SMEM）：每轮发一次 tensor copy，mbarrier 按 Q 字节完成后 CTA 会合，再发下一次。
- 写（SMEM→GMEM）：每轮 commit 后 `wait_group 0` 等完整写入，再 CTA 会合。
- 每轮只有一个请求在途；多请求见 [EXP-16](EXP-16-tma-pipeline.md)。整卡 CTA 数由资源查询决定。计时后的导出不计入。

## 结果（中位数）

整卡 GB/s：

| 方向 / 布局 | 1 | 4 | 8 | 16 | 32 | 64 KiB |
|---|---:|---:|---:|---:|---:|---:|
| 读，连续 | 1755 | 3012 | 3816 | 3877 | 3873 | 3873 |
| 读，SW128 | 1747 | 3014 | 3815 | 3874 | 3872 | — |
| 读，padding | 1713 | 2741 | 3384 | 3362 | 3361 | 3551 |
| 写，连续 | 2885 | 3806 | 3810 | 3815 | 3837 | 3828 |
| 写，SW128 | 2889 | 3838 | 3824 | 3820 | 3839 | — |
| 写，padding | 2967 | 1451 | 1369 | 1340 | 1330 | 1658 |

单 CTA B/cycle：

| 方向 / 布局 | 1 | 4 | 8 | 16 | 32 | 64 KiB |
|---|---:|---:|---:|---:|---:|---:|
| 读，连续 | 1.74 | 6.53 | 12.72 | 21.38 | 36.93 | 55.46 |
| 读，padding | 1.73 | 6.31 | 10.69 | 18.10 | 25.73 | 34.21 |
| 写，连续 | 3.46 | 10.45 | 17.00 | 22.20 | 26.21 | 28.82 |
| 写，padding | 3.32 | 9.31 | 14.17 | 17.62 | 20.05 | 21.54 |

最大 CV 1.6%。

![整卡](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/node-sampling-v1/published-all68-v1/all_gpu.png)

<a id="r05-e"></a>

## 后续结果：输出行距（原 R05-E）

2026-10-08 由 [R05](access_rules/R05-async-lifecycle.md) 迁入，job735059，2026-10-06。目的是区分本页写回下降的原因：128 B 连续；144 B 只按 16 B 对齐，即本页的 padding 条件；160 B 按 32 B 对齐；256 B 是 128 B 倍数的 padding。TMA 2D 输出，box (64,128) 16-bit 共 16 KiB，整卡；基地址 128 B 对齐，非均匀数据，只计有效 16 KiB；完成边界同本页，commit 后 `wait_group 0`，再 CTA 会合。

| GMEM 行距 | 128 B | 144 B | 160 B | 256 B |
|---|---:|---:|---:|---:|
| 整卡完整窗口 ns | 950714 | 1314624 | 966899 | 898909 |

144 B 比 128 B 慢 38%。行距 160 B（行起点 32 B 对齐）和 256 B 都不慢，所以本页 padding 写回下降的原因是 144 B 行距只按 16 B 对齐，而不是行间有空隙。[R05 B/C/D/E 报告](../../../../../results/gh200_resource_campaign/access_rules/20261006-c-job735059/r05-formal-v2/report.md)。

## 数据

[results.csv](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/node-sampling-v1/published-all68-v1/results.csv)（68 点）、[samples.csv](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/node-sampling-v1/published-all68-v1/samples.csv)、[条件参数](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/node-sampling-v1/published-all68-v1/qualified-parameters.json)、[单 CTA 图](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s15-main-execution/formal-sampling/node-sampling-v1/published-all68-v1/one_cta.png)。
