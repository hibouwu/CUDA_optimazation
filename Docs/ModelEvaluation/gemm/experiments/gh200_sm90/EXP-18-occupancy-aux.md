# EXP-18：驻留资源与辅助操作

[结果总览](README.md)

## 结论

- 一个单 CTA 依赖循环在 64 和 128 线程时每轮都是 84 cycle，256 线程 136、512 线程 261：4 个 warp 以内每轮时间不变，再多则近似线性变长；推断与每 SM 4 个调度分区有关。
- 每线程 live 变量从 48 增到 80、168 个寄存器时，每轮 148、277 cycle（48 寄存器时 84）。
- SMEM 预留 32/64/128 KiB、carveout 提示 0/100 对单 CTA 速率没有影响；它们只改变 occupancy 上限（6/3/1 CTA/SM）。
- **local 访问很贵**：显式 local 数组每字约 59 cycle（32 字 1898 cycle/轮，128 字 7473）；寄存器溢出到 local 时 32 字 224 cycle/轮，128 字 1874。GEMM 内核应以无溢出为约束。
- 依赖链每轮（单流）：`add.u64` 29 cycle，`mad.wide.u32` 31，`cvt.rn.f16.f32` 37，`cvt.f32.f16` 43；4 条独立流时分别为 30、44、61、55，即 64 位加法几乎完全并行，转换指令的吞吐有限。
- global 原子：同址每轮 73 cycle，各线程不同地址 28；SMEM 原子同址 74，不同地址 31。同址争用使原子变慢约 2.5 倍。

## 配置

1 CTA，每配置固定一种资源或操作；单位为完整循环的 cycle/轮。线程/资源组每轮的工作相同，只改变线程数、寄存器或 SMEM 占用。`setmaxnreg` 只做合法性检查，不在本表。

## 结果（cycle/轮，中位数）

| 配置 | cycle/轮 | 寄存器/线程 | occupancy 上限 CTA/SM |
|---|---:|---:|---:|
| 64 / 128 / 256 / 512 线程 | 84.0 / 84.1 / 136.0 / 261.5 | 48 | 20 / 10 / 5 / 2 |
| live 64 / 128（128 线程） | 148.2 / 276.5 | 80 / 168 | 6 / 3 |
| SMEM 32 / 64 / 128 KiB | 84.0 / 84.1 / 84.1 | 48 | 6 / 3 / 1 |
| local 数组 32 / 128 字 | 1898 / 7473 | 14 | 16 |
| 溢出压力 32 / 128 字 | 224.1 / 1873.5 | 32 | 16 |
| `add.u64` 1 / 4 流 | 29.0 / 30.0 | 16 / 23 | 16 |
| `mad.wide.u32` 1 / 4 流 | 31.0 / 44.0 | 14 / 28 | 16 |
| `cvt.rn.f16.f32` 1 / 4 流 | 37.0 / 61.0 | 14 | 16 |
| `cvt.f32.f16` 1 / 4 流 | 43.0 / 55.0 | — | 16 |
| global 原子：同址 / 不同址 | 73.0 / 28.0 | — | — |
| SMEM 原子：同址 / 不同址 | 74.0 / 31.0 | — | — |

![辅助操作](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/parallel/compute_onchip/s18-integration-v2/operations-v1/job733338-results/published-v1/03-auxiliary.png)

## 限制

每轮 cycle 包含循环控制；单流结果接近依赖延迟，但不是裸指令延迟。occupancy 为 API 上限，没有测多个 CTA 同时驻留时的速率。

## 数据

[完整表](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/parallel/compute_onchip/s18-integration-v2/operations-v1/job733338-results/published-v1/TABLE.md)、[条件参数](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/parallel/compute_onchip/s18-integration-v2/operations-v1/job733338-results/published-v1/qualified-parameters.json)、[四组图说明](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/parallel/compute_onchip/s18-integration-v2/operations-v1/job733338-results/published-v1/EXPERIMENT.md)、[复现](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/parallel/compute_onchip/s18-integration-v2/operations-v1/job733338-results/published-v1/REPRODUCE.md)。
