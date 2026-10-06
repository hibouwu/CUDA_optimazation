# EXP-05：CUDA Core FMA

[结果总览](README.md)

## 结论

- 整卡 FP32 FMA 65.7 TFLOP/s，约为数据表 67 TFLOP/s 的 98%；FP64 33.5 TFLOP/s，与数据表 34 一致。
- 单 SM 要 4 个 warp、每线程 8 条独立链才接近理论值：FP32 从 1 warp 的 58 FLOP/cycle 到 4 warp 的 232、8 warp 的 250（理论 256）。
- FP16x2 / BF16x2 每条指令两个 lane，单 CTA 415–437 FLOP/cycle，整卡约 117 TFLOP/s。
- 标量 FP16/BF16 被编译器部分合并成 HFMA2，速率介于 FP32 与 x2 之间，且 FP16 与 BF16 不同（256 线程 315 vs 284）；逻辑指令数不等于 SASS 指令数。
- 依赖链的每轮增量：FP32、1 warp、1 链、每轮 16 个 FMA 时 \(T(n)=159+71n\) cycle，即约 4.4 cycle/FMA，包含循环控制。

## 配置

每线程 \(c\) 条独立累加链，每链每轮 16 个 FMA；6 种形式 FP32、FP64、FP16、FP16x2、BF16、BF16x2。整卡为 528 CTA（132×4）。另有 1/8 链、128/512/2048 轮的固定长度点用于拟合每轮增量。

## 结果（8 链，中位数）

| 形式 | 1 CTA 32 线程 | 1 CTA 128 线程 | 1 CTA 256 线程 | 整卡 256 线程 GFLOP/s |
|---|---:|---:|---:|---:|
| FP32 | 58.1 | 232.4 | 250.1 | 65702 |
| FP64 | 30.6 | 122.3 | 127.2 | 33458 |
| FP16 | 57.7 | 230.8 | 315.1 | 84566 |
| FP16x2 | 103.7 | 414.8 | 436.9 | 117165 |
| BF16 | 52.2 | 208.7 | 283.7 | 80619 |
| BF16x2 | 103.7 | 414.8 | 436.9 | 116614 |

单 CTA 单位为 FLOP/cycle/CTA。

![FP32 单 CTA](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/legacy-fma-formal-v3-b-r3/f32-one_cta-1.png)

## 限制

- 计时含循环控制、末尾归约和 SMEM 排空；拟合截距与斜率不是单条 FMA 的延迟或发射间隔。
- 输入为固定简单值；没有测 FMA 与 LDS、IMAD 或 MMA 同时执行。

## 数据

[cases.csv](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/legacy-fma-formal-v3-b-r3/cases.csv)（93 点）、[samples.csv](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/legacy-fma-formal-v3-b-r3/samples.csv)、[固定长度拟合](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/legacy-fma-formal-v3-b-r3/fixed_length_fits.json)、[全部图](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/legacy-fma-formal-v3-b-r3/index.md)。原始 run：`legacy_fma/formal-v3-b`。

```bash
python -B microbench/gh200_resource_campaign/report_compute_v2.py \
  results/gh200_resource_campaign/20261001-resource-suite-v2/legacy_fma/formal-v3-b \
  --replay <audit_replay.py 输出> --output <新目录>
```
