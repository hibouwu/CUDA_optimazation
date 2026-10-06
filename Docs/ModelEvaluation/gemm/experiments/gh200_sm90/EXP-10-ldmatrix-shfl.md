# EXP-10：ldmatrix / stmatrix 与 shfl

[结果总览](README.md)

## 结论

- 本实验只用 1 个 warp（矩阵）或 1–4 个 warp（shfl），**不是 SM 的吞吐上限**。shfl 的速率随独立流数和 warp 数线性增长，说明这些配置仍受依赖延迟限制；`ldmatrix` 只测了 1 个 warp，推断同样如此。
- `ldmatrix` x1/x2/x4 为 16.5 / 28.4 / 35.3 B/cycle，远低于 SMEM 的 128 B/cycle；`stmatrix` 略高（19.0 / 33.0 / 41.8）。`.trans` 与普通形式没有差别。
- shfl：单 warp 单依赖链 0.0385 条/cycle，即约 26 cycle/条；独立流数或 warp 数加 4 倍，速率也正好加 4 倍，128 线程 × 4 流时仍未饱和（0.615 条/cycle）。

## 配置

矩阵：1 CTA、32 线程、`m8n8.b16`，load、store、load→store 往返；每轮 8 个位置，8192 轮。shfl：32/128 线程，1/4 条独立依赖流，`(lane+1)%32` 交换。

## 结果（中位数）

| 矩阵搬运（B/cycle/CTA） | x1 | x2 | x4 |
|---|---:|---:|---:|
| `ldmatrix` | 16.5 | 28.4 | 35.3 |
| `stmatrix` | 19.0 | 33.0 | 41.8 |
| 往返（读+写） | 8.8 | 14.8 | 22.1 |

| shfl（warp 指令/cycle/CTA） | 1 流 | 4 流 |
|---|---:|---:|
| 32 线程 | 0.0385 | 0.1538 |
| 128 线程 | 0.1538 | 0.6153 |

![矩阵搬运](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s10-main-execution/runtime-r2/analysis-v2/matrix.png)

## 待补

4–8 warp 的 `ldmatrix` 吞吐，以及 `ldmatrix` 供给 `mma.sync` 的组合；这是 SIMT/mma.sync 方案的片上供给参数。

## 数据

[samples.csv](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s10-main-execution/runtime-r2/analysis-v2/samples.csv)、[parameters.json](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s10-main-execution/runtime-r2/analysis-v2/parameters.json)、[shfl 图](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s10-main-execution/runtime-r2/analysis-v2/shuffle.png)。原始 run：`implementation/s10-main-execution/runtime-r2/`。
