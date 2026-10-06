# EXP-06：warp 级 mma.sync

[结果总览](README.md)

## 结论

- FP16/BF16 `m16n8k16`（FP32 累加）单 CTA 最高 2731 FLOP/cycle，整卡 657 TFLOP/s，约为 Tensor Core 理论值的 2/3；要到满峰值需改用 WGMMA（[EXP-07](EXP-07-wgmma.md)）。
- TF32 `m16n8k8` 1344 FLOP/cycle、整卡 325 TFLOP/s，同样约为理论值的 65%。
- **FP64 `m8n8k4` 只有 128 FLOP/cycle，整卡 33.5 TFLOP/s**，与 FP64 FMA（[EXP-05](EXP-05-fma.md)）相同，是数据表 FP64 Tensor Core 67 TFLOP/s 的一半。本实验只测了旧形状；PTX 为 sm_90 新增的 f64 形状（待按 PTX ISA 核实 `m16n8k4/k8/k16`）需另测。
- 1 warp 时 FP16 只有 667 FLOP/cycle，4 warp 才到 2668；单 warp 无法占满 SM。
- 依赖链：FP16、1 warp、1 链、每轮 16 个依赖 MMA 时 \(T(n)=199+384n\) cycle，约 24 cycle/MMA（含循环控制）。

## 配置

每 warp 8 条独立累加链，每链每轮 16 个 MMA；操作数在计时前放入寄存器，无 GMEM/SMEM 供给。整卡 CTA 数为 132×min(4, occupancy)。每个 MMA 的工作量为 \(2MNK\)，不乘线程数。

## 结果（8 链，中位数）

| 形式 | 形状 | 1 CTA 32 线程 | 1 CTA 128 线程 | 1 CTA 256 线程 | 整卡 128 线程 GFLOP/s |
|---|---|---:|---:|---:|---:|
| FP16 | 16×8×16 | 667.0 | 2668.1 | 2730.6 | 656831 |
| BF16 | 16×8×16 | 667.0 | 2668.1 | 2730.6 | 653779 |
| TF32 | 16×8×8 | 328.1 | 1312.3 | 1344.3 | 323537 |
| FP64 | 8×8×4 | 32.0 | 127.9 | 128.0 | 33454 |

单 CTA 单位为 FLOP/cycle/CTA。

![FP16 单 CTA](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/legacy-mma-formal-v3-b/f16-one_cta-1.png)

## 限制

- 窗口含循环控制、累加值的 SMEM 排空与 CTA barrier；不是裸 MMA 延迟。
- 没有测 FP16 累加形式，也没有测 `ldmatrix` 供给下的 MMA。

## 数据

[cases.csv](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/legacy-mma-formal-v3-b/cases.csv)（55 点）、[samples.csv](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/legacy-mma-formal-v3-b/samples.csv)、[固定长度拟合](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/legacy-mma-formal-v3-b/fixed_length_fits.json)、[全部图](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/legacy-mma-formal-v3-b/index.md)、[SASS 审查](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/legacy_mma/formal-v3-b/build/sass_audit.json)。原始 run：`legacy_mma/formal-v3-b`。
