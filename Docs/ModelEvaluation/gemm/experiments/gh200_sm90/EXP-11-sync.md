# EXP-11：同步与 fence

[结果总览](README.md)

## 结论

- 齐步到达时，CTA barrier 每阶段 21 cycle（128 线程）/ 29 cycle（256 线程）；mbarrier（arrive + 等待相位）218 / 226 cycle，约为 CTA barrier 的 8–10 倍。
- 到达偏斜时（最后一个 warp 多做 256 条依赖 IMAD），阶段时间由最慢参与者决定：CTA barrier 约 1080 cycle，mbarrier 约 1294；mbarrier 比 barrier 多出的约 200 cycle 不被偏斜掩盖。
- 无未完成访存时，GPU 范围 fence 267 cycle/条，CTA 范围 fence 与 `fence.proxy.async` 约 11 cycle/条。流水线中应避免每个 stage 都执行 GPU 范围 fence。
- 写成 `bar.warp.sync` 的 warp 点在 SASS 中没有生成 WARPSYNC，只得到逻辑循环时间（3.6 cycle/阶段），不能当作 warp barrier 的成本。

## 配置

1 CTA；同步点 2048 轮 × 每轮 8 阶段，fence 点 8192 轮 × 8 条。偏斜条件为最后一个 warp 每阶段多执行 256 条依赖 IMAD。

## 结果（中位数）

| 同步（cycle/phase） | 128 线程齐步 | 128 线程偏斜 | 256 线程齐步 | 256 线程偏斜 |
|---|---:|---:|---:|---:|
| CTA barrier | 21.3 | 1080.0 | 29.3 | 1082.4 |
| mbarrier | 217.9 | 1293.5 | 225.9 | 1294.8 |

| fence，无未完成请求（cycle/条） | 值 |
|---|---:|
| CTA 范围 | 10.9 |
| GPU 范围 | 267.4 |
| `fence.proxy.async` | 10.9 |

![CTA 与 mbarrier](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s11-main-execution/analysis-v2/collective.png)

## 限制

fence 只测了没有未完成访存的情形；有在途 store 或 TMA 时的等待未测。mbarrier 的 TMA 事务计数路径见 [EXP-14](EXP-14-tma-1d.md)–[EXP-16](EXP-16-tma-pipeline.md)。

## 数据

[samples.csv](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s11-main-execution/analysis-v2/samples.csv)、[parameters.json](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s11-main-execution/analysis-v2/parameters.json)、[fence 图](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s11-main-execution/analysis-v2/fence.png)。原始 run：`implementation/s11-main-execution/`。
