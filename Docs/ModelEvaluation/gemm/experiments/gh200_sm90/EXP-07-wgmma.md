# EXP-07：WGMMA

[结果总览](README.md)

## 结论

- FP16/BF16 `m64n64k16`（FP32 累加，SS）在一个 warpgroup、2 条链、每轮 16 条后 commit 的条件下，wait 3/7 达 **4095.9 FLOP/cycle，即满峰值**；wait 0 为 3858（94%）。一个 warpgroup 足以占满单 SM 的 Tensor Core。
- 整卡 960–984 TFLOP/s，对应平均 SM 时钟约 1.81 GHz；整卡下 wait 0/3/7 差别 <3%；推断是每 SM 驻留的多个 CTA 互相掩盖了等待。
- 2 个 warpgroup 时 wait0 也到 4090；推断 wait0 的等待被另一组的发射掩盖。
- BF16 RS（A 在寄存器）与 SS 相差 <1%（单 CTA 3890 vs 3858，整卡 980 vs 984 TFLOP/s）。
- 依赖链：1 链、每轮 1 条 WGMMA + commit + wait0 时 \(T(n)=555+105n\) cycle，每轮增量约 105 cycle，包含一条 WGMMA 的发出到完成、commit、wait0 与循环控制，可作为单条依赖 WGMMA 往返的上界。

## 配置

`m64n64k16`，每条 131072 FLOP；A/B 在 SMEM 中反复使用，无 GMEM 或 TMA 供给。每轮发出 batch×chains 条后一次 `commit_group`，再 `wait_group 0/3/7`；循环后 wait0 并排空全部累加器。128 线程整卡 396 CTA（occupancy 3/SM），256 线程 132 CTA。资源：SS 148 寄存器/线程、5136 B SMEM；RS 146 寄存器、3088 B。

## 结果（2 链、batch 16，中位数）

| 输入 | warpgroup/CTA | wait | 1 CTA FLOP/cycle | 整卡 GFLOP/s |
|---|---:|---:|---:|---:|
| FP16 | 1 | 0 | 3858.6 | 956267 |
| FP16 | 1 | 3 | 4095.9 | 970373 |
| FP16 | 1 | 7 | 4095.9 | 941383 |
| FP16 | 2 | 0 | 4090.0 | 960506 |
| FP16 | 2 | 3 | 4095.9 | 971382 |
| BF16 | 1 | 0 | 3858.6 | 976072 |
| BF16 | 1 | 3 | 4095.9 | 975360 |
| BF16 | 2 | 0 | 4090.0 | 970595 |
| BF16 RS | 1 | 0 | 3890.4 | 980376 |

![BF16 整卡](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/legacy-wgmma-formal-v3-b-r2/bf16-all_gpu-1.png)

## 限制

- 本页只测了 N=64；N=128/256、1/2 个 warpgroup、SS/RS 见 [R00-B](access_rules/R00-anchor-target.md)，都达到 4095 FLOP/cycle。
- 没有 TF32、FP16 累加形式，也没有与 FFMA 同时执行的情形。
- wait 值控制的是未完成 group 数，不能推出硬件队列深度。

## 数据

[cases.csv](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/legacy-wgmma-formal-v3-b-r2/cases.csv)（85 点）、[samples.csv](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/legacy-wgmma-formal-v3-b-r2/samples.csv)、[固定长度拟合](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/legacy-wgmma-formal-v3-b-r2/fixed_length_fits.json)、[全部图](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/legacy-wgmma-formal-v3-b-r2/index.md)、[SASS 审查](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/legacy_wgmma/formal-v3-b/build/sass_audit.json)。原始 run：`legacy_wgmma/formal-v3-b`。
