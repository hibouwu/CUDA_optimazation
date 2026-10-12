# EXP-09：SMEM 访问

[结果总览](README.md)

## 结论

- 8 B 和 16 B 访问的读、写都达到 128 B/cycle/CTA，与每 SM 32 bank × 4 B/cycle 的数值一致。
- **读写同时进行时合计仍约 128 B/cycle**：在这些条件下建模时不能把读、写峰值相加；内部端口结构不能由此唯一确定。
- 4 B 读只有 87 B/cycle，4 B 写却有 128；读路径还要做 checksum 累加，4 B 读可能受发射或依赖限制而非 bank 限制。GEMM 的 SMEM 读应使用 ≥8 B 的访问宽度。
- stride=k（4 B）时速率约为 128/k，与 bank 冲突随跨步加倍的预期一致（无计数器验证）。
- 广播（warp 内同址）约 89 B/cycle，样本在 88.5–110.7 之间分两档，CV 4.8%。

## m11 扩展的资格

原 EXP-09 与 job740961 兼容复测支持特定窗口的 stride 条件关系；它们没有证明普通 LDS/STS 的通用 wavefront 时间模型。原128 B子组需求把宽广播与连续访问算成同样需求，已由同组织反例否定。后续全 warp 形式也未满足原绝对周期与留布局要求，宽广播和依赖执行仍缺可迁移描述；没有独立留出成绩，不向 RULES 收录为已验证机制。

[第一轮与后续离线结论](access_rules/R03-access-demand.md#m11-current)、[第一轮完整交付](../../../../../results/gh200_resource_campaign/access_rules/20261011-m11-lds-sts-bounded/run-20261011T210745Z/continuation-20261012/README.md)、[两形式反证](../../../../../results/gh200_resource_campaign/access_rules/20261012-m11-demand-offline-v1/README.md)。

## 配置

1 CTA、256 线程、两个 32 KiB 数组；每线程每轮 8 次访问，8192 轮。窗口含循环、读 checksum 和 CTA 同步。

## 结果（B/cycle/CTA，中位数）

| 访问 | 4 B | 8 B | 16 B |
|---|---:|---:|---:|
| 读，连续 | 87.1 | 128.0 | 128.0 |
| 写，连续 | 127.9 | 128.0 | 128.0 |
| 读写各半，连续 | 106.4 | 128.0 | 128.0 |

| 4 B 读 stride | 2 | 4 | 8 | 16 | 32 | 广播 |
|---|---:|---:|---:|---:|---:|---:|
| B/cycle | 64.0 | 32.0 | 16.0 | 8.0 | 4.0 | 88.8 |

![全部配置](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s09-main-execution/device-revalidation-v2/analysis-v1/services.png)

## 限制

只测单 CTA 的线程 load/store；`ldmatrix` 见 [EXP-10](EXP-10-ldmatrix-shfl.md)，WGMMA 的 SMEM 取数和 TMA 写 SMEM 未单独测，多 CTA 竞争也未测。

## 数据

[samples.csv](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s09-main-execution/device-revalidation-v2/analysis-v1/samples.csv)、[parameters.json](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s09-main-execution/device-revalidation-v2/analysis-v1/parameters.json)、[广播三批](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s09-main-execution/device-revalidation-v2/analysis-v1/broadcast.png)。原始 run：`implementation/s09-main-execution/device-revalidation-v2/`。

<a id="exp-04"></a>

## 前序测量（原 EXP-04 的 SMEM 部分）

2026-10-08 由 EXP-04 迁入。EXP-04 是 v1 访存测量（EXP-02）的重测；1 CTA、256 线程、每线程每轮 8 次 4 B，原始 run `memory_baseline/formal-v3-a`。

| 配置 | B/cycle/CTA | CV |
|---|---:|---:|
| 读 stride 1 / 2 / 4 / 8 / 16 / 32 | 95.2 / 64.0 / 32.0 / 16.0 / 8.0 / 4.0 | 9.9% / <0.001% |
| 写 stride 1 | 127.9 | <0.001% |

stride=1 读在约 95 与 127 B/cycle 两档之间跳动，三批后仍不稳定；本页改为两数组、独立读写后得到稳定的 87 B/cycle。stride 2–32 稳定为 128/stride 量级，与本页一致。

![SMEM stride=1 三批样本](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/memory-baseline-formal-v3-a/stride1.png)

[samples.csv](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/memory-baseline-formal-v3-a/samples.csv)、[SMEM 图](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/memory-baseline-formal-v3-a/smem.png)。计时空窗口迁至 [R09](access_rules/R09-inkernel-clock-stages.md#empty-window)，global 部分迁至 [EXP-13](EXP-13-global-rw.md#exp-04)。

## m11 兼容窗口对照准备（2026-10-11）

[R03 的普通 LDS/STS 扩展](access_rules/R03-access-demand.md#m11-普通-ldssts-有界扩展2026-10-11)绑定本页原 probe 与 `probe_runtime.cuh`，原样保留 256 线程、两数组、8192 轮、每轮八次访问、checksum 与 CTA 同步，预列 stride、广播和连续向量对照。旧广播 88.5–110.7 B/cycle 与 CV 4.8% 保留为旧窗口证据，不将 89 B/cycle 移植给新窗口。

首次提交 740708 在新 probe 的 SASS 门槛处失败，尚未运行本页兼容对照；没有新增本页数值或复现资格。新窗口的实际地址依赖、独立流和 warp 同步另记，不能仅因 checksum 正确就将编译器移出循环的 LDS 算作每轮工作。

2026-10-12 人类修订预算后，CPU CUDA 12.9 完整 SASS 预检通过，第二提交 740961 在 romeo-a048/GPU-009a8880-f309-a55d-d256-748fb9adad74 完成兼容窗口复测。旧 probe、参数和窗口保持原样：

| 本轮兼容窗口 | 中位 B/cycle/CTA | 进程数 |
|---|---:|---:|
| 4 B 读 stride 1 / 2 / 4 / 8 / 16 / 32 | 87.1312 / 63.9924 / 31.9989 / 16.0002 / 8.00009 / 4.00009 | 各 3 |
| 8 B / 16 B 连续读 | 127.9745 / 127.9878 | 各 3 |
| 4 B 连续写 | 127.9466 | 3 |
| 4 B 同址广播 | 88.78126 | 10 |

预写的九个兼容门槛全部通过。旧广播范围 88.5–110.7 和 CV 4.8% 保留为历史观测，不由本轮十个进程替换。新 probe 的同步/消费组织另列，其候选需求模型被向量广播反例否定；本页兼容复测不授予新模型留出资格。见 [m11 开发反证](access_rules/R03-access-demand.md#开发反证2026-10-12)及[本轮完整数据](../../../../../results/gh200_resource_campaign/access_rules/20261011-m11-lds-sts-bounded/run-20261011T210745Z/continuation-20261012/development-analysis/old-compatibility.json)。
