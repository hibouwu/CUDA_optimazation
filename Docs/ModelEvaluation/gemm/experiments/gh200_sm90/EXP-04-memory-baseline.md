# EXP-04：访存基线与计时空窗口

[结果总览](README.md)

本实验是 v1 访存测量（EXP-02）的重测，后续由 [EXP-09](EXP-09-smem.md)（SMEM）和 [EXP-13](EXP-13-global-rw.md)（global）展开。

## 结论

- 单 CTA 空计时窗口为 34 cycle，可作为 `clock64` 计时的固定开销参考。整卡空窗口约 128 ns，但跨进程 CV 12.7%，不稳定。
- SMEM 4 B 读 stride=1 在两档之间跳动（约 95 与 127 B/cycle，CV 9.9%），三批后仍不稳定；EXP-09 改为两数组、独立读写后得到稳定的 87 B/cycle。stride 2–32 稳定为 128/stride 量级。
- 整卡 256 MiB 读 3.52 TB/s、写 3.84 TB/s、1:1 依赖复制 3.41 TB/s，与 EXP-13 一致。

## 结果（中位数）

| 配置 | 单位 | 值 | CV |
|---|---|---:|---:|
| SMEM 读 stride 1 / 2 / 4 / 8 / 16 / 32 | B/cycle/CTA | 95.2 / 64.0 / 32.0 / 16.0 / 8.0 / 4.0 | 9.9% / <0.001% |
| SMEM 写 stride 1 | B/cycle/CTA | 127.9 | <0.001% |
| global 读 `.ca` / `.cg`，8 MiB | GB/s | 13371 / 7834 | 0.2% / 1.0% |
| global 读 `.cg`，256 MiB | GB/s | 3515 | 0.3% |
| global 写，256 MiB | GB/s | 3838 | 0.7% |
| global 依赖复制，各 128 MiB | GB/s（读+写） | 3411 | 0.2% |
| 空窗口：单 CTA / 整卡 | cycle / ns | 34 / 128 | 0 / 12.7% |

SMEM：1 CTA、256 线程、每线程每轮 8 次 4 B；global：528 CTA、256 线程、每线程 16 B。

![SMEM stride=1 三批样本](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/memory-baseline-formal-v3-a/stride1.png)

## 数据

[samples.csv](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/memory-baseline-formal-v3-a/samples.csv)、[SMEM 图](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/memory-baseline-formal-v3-a/smem.png)、[global 图](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/memory-baseline-formal-v3-a/global.png)。原始 run：`memory_baseline/formal-v3-a`。
