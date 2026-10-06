# EXP-08：FP8 与 INT8

[结果总览](README.md)

## 结论

- WGMMA `m64n64k32` 的 FP8（E4M3、E5M2，FP32 累加）与 INT8（S8、U8，S32 累加）两个 warpgroup 时达 8179–8181 FLOP(OP)/cycle，即满峰值；一个 warpgroup 为 7716–7731（94%）。整卡均约 1.96 PFLOP/s（POPS）。
- INT8 `mma.sync m16n8k32` 每 warp 672 OP/cycle，1→2 warp 线性增长；本实验只测到 2 warp，没有测到饱和点。
- FP8 `mma.sync` 在 `sm_90a` 上被编译为转换加 FP16 MMA，不是原生路径；表中只按逻辑 FLOP 记录，不作为 FP8 指令参数。
- **FP8 WGMMA 的累加精度有限**：A=B=1/16、初值 0，每条 WGMMA 给每个输出加 \(32\times2^{-8}=1/8\)。累加到 62、64 都精确，但从 64 起再也不增长（33 轮得 64，参考 66；8192 轮仍为 64，参考 16384）。单个乘积为 \(2^{-8}\)，累加器为 \(2^{6}\)，二者相差 14 个二进制量级：结果与“乘积按累加器指数对齐、只保留约 14 bit”一致，与 DeepSeek‑V3 技术报告对 Hopper FP8 GEMM 的描述相符。
- 这说明某些输入下长累加会产生明显误差；是否需要处理取决于问题实例的输入范围和误差要求。若直接累加不能满足要求，可按 K 分段提升到 CUDA Core 的 FP32 累加（DeepGEMM 取 128），并计入额外计算、寄存器和依赖成本。正式性能测量因此改用正负抵消的有界输入。

## 配置

每组 2 条累加链，每链每轮 16 条指令，8192 轮；WGMMA 为 SS、wait0。参与组为 1/2 个 warp 或 warpgroup，单 CTA 与整卡（528 或 396 CTA）各测一次。

## 结果（中位数）

| 路径 | 编码 | 组数 | 1 CTA /cycle | 整卡 G/s |
|---|---|---:|---:|---:|
| WGMMA | E4M3 | 1 / 2 | 7716 / 8179 | 1961738 / 1959979 |
| WGMMA | E5M2 | 1 / 2 | 7716 / 8179 | 1960985 / 1961339 |
| WGMMA | S8 | 1 / 2 | 7731 / 8181 | 1978658 / 1963339 |
| WGMMA | U8 | 1 / 2 | 7731 / 8181 | 1978654 / 1975943 |
| mma.sync | S8 / U8 | 1 / 2 | 672 / 1344 | 644028 / 1249517 |
| mma.sync（编译展开） | E4M3 / E5M2 | 1 / 2 | 1285 / 2570 | 约 1.24 / 1.45 P（逻辑） |

![FP8 WGMMA](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s08-main-execution/full-output-v1/runtime-v1/published-v1/fp8_service.png)

### 累加边界

| 轮数 | 参考值 | 全部 8192 个输出 |
|---:|---:|---:|
| 31 | 62 | 62 |
| 32 | 64 | 64 |
| 33 | 66 | 64 |
| 64 | 128 | 64 |
| 8192 | 16384 | 64 |

![累加边界](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/boundary31-64-v1-r4/boundary.png)

## 限制

- 只测 N=64、wait0 的 WGMMA；没有 FP8 输出、混合编码或稀疏形式。
- 累加边界只用了均匀输入；14 bit 是由一个边界点推出的解释，要确定对齐与舍入规则，需要用不同量级的乘积和累加器初值再测。

## 数据

[results.csv](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s08-main-execution/full-output-v1/runtime-v1/published-v1/results.csv)、[samples.csv](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s08-main-execution/full-output-v1/runtime-v1/published-v1/samples.csv)、[条件参数](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s08-main-execution/full-output-v1/runtime-v1/published-v1/qualified-parameters.json)。累加边界：[observations.csv](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/boundary31-64-v1-r4/observations.csv)，原始 run `low_precision/boundary31-64-v1` 与 `low_precision/long-uniform8192-v1`。
