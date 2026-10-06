# EXP-12：cp.async

[结果总览](README.md)

## 结论

- 单 CTA（128 线程）的速率随 stage 数增加：16 B 请求从 1 stage 的 4.0 B/cycle 增到 4 stage 的 7.5 B/cycle，远低于 TMA 的单 CTA 速率（[EXP-14](EXP-14-tma-1d.md)）。
- 请求宽度影响很大：4 stage 时 4 / 8 / 16 B 分别为 2.8 / 5.3 / 7.5 B/cycle（单 CTA），整卡 2.0 / 3.9 / 4.7–5.3 TB/s。应使用 16 B 请求。
- 整卡 16 B、4 stage 时 `.cg` 5.3 TB/s，高于 `.ca` 的 4.7 TB/s；源只有 8 MiB，小于 L2，速率高于 HBM 产品值，推断数据主要来自 L2（无计数器证明）。
- 2→4 stage 的整卡增益只有 3–4%，单 CTA 增益 17%：整卡在 2 stage 时已接近本条件下的上限。

## 配置

每 CTA 128 线程，每线程每轮 8 次请求，每次独立 commit；先填满 stage 个槽，消费最老槽（读相邻线程写入的数据做 checksum）后补入新请求。8192 轮，整卡 528 CTA，共享一份 8 MiB 只读源。窗口含 prefill、稳态、排空、消费者读取与 CTA 同步。

## 结果（中位数）

| 请求 | 单 CTA B/cycle：stage 1 / 2 / 4 | 整卡 GB/s：stage 1 / 2 / 4 |
|---|---|---|
| `.ca` 4 B | 1.29 / 2.26 / 2.82 | 1267 / 1749 / 2015 |
| `.ca` 8 B | 2.32 / 4.05 / 5.30 | 2237 / 3268 / 3937 |
| `.ca` 16 B | 3.96 / 6.46 / 7.53 | 3688 / 4614 / 4734 |
| `.cg` 16 B | 4.08 / 6.46 / 7.53 | 3850 / 5098 / 5304 |

![整卡各 stage](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/async-copy-formal-b3-v2/stages-all_gpu.png)

## 限制

消费者读取和每 stage 的 CTA 同步都在窗口内，结果是"复制 + 消费"协议的速率；没有测大于 L2 的源工作集。

## 数据

[cases.csv](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/async-copy-formal-b3-v2/cases.csv)、[trials.csv](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/async-copy-formal-b3-v2/trials.csv)、[stage 比较](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/async-copy-formal-b3-v2/comparisons.csv)、[手算](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/async-copy-formal-b3-v2/manual-examples.md)。原始 run：`async_copy/formal-b3-v2`。
