# EXP-13：global 读写联合服务

[结果总览](README.md)

## 结论

- 大工作集（每数组 4×L2 = 241 MiB）：读 3.52 TB/s，写 3.79 TB/s，读写混合 3.33–3.67 TB/s，为 HBM3 产品值 4 TB/s 的 83–95%。
- 混合时读写合计不超过单独读写：1:1 为 3.43 TB/s，**不能把读、写速率相加**。读占比越高越快（4:1 为 3.67，1:4 为 3.33）。
- 依赖复制与独立 1:1 读写几乎相同（3.43 vs 3.43 TB/s），依赖关系不是限制。
- 小工作集（两数组合计 < L2）：`.ca` 读 13.5 TB/s、`.cg` 读 8.3 TB/s，写 4.5 TB/s。推断 `.ca` 的差额来自 L1 命中、`.cg` 只经 L2（无计数器证明）；写在小工作集下也只有 4.5 TB/s，明显低于读。

## 配置

528 CTA × 256 线程，每线程 16 B 访问，16 轮。small 每数组 L2/4（16.5 MiB），large 每数组 4×L2（241 MiB）。读写比 1:0、0:1、1:1、2:1、4:1、1:2、1:4；指标为读字节 + 写字节，除以整卡包络时间。

## 结果（GB/s，均值）

| 模式 | small | large |
|---|---:|---:|
| 读 `.ca` | 13465 | 3522 |
| 读 `.cg` | 8293 | 3524 |
| 写 `.wb` | 4531 | 3790 |
| 依赖复制 1:1 | 7787 | 3434 |
| 独立 1:1 | 7748 | 3428 |
| 独立 2:1 | 10787 | 3591 |
| 独立 4:1 | 9033 | 3671 |
| 独立 1:2 | 6399 | 3387 |
| 独立 1:4 | 5521 | 3327 |

最大 CV 1.6%。

![large](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/global-duplex-formal-b3-v1-r2/large.svg)

## 限制

速率为逻辑请求量；小工作集的层级（L1/L2）由工作集大小和 `.ca/.cg` 差别推断，没有计数器证明。

## 数据

[cases.csv](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/global-duplex-formal-b3-v1-r2/cases.csv)、[trials.csv](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/global-duplex-formal-b3-v1-r2/trials.csv)、[small 图](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/global-duplex-formal-b3-v1-r2/small.svg)、[手算](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/global-duplex-formal-b3-v1-r2/manual-example.md)。原始 run：`global_duplex/formal-b3-v1`。
