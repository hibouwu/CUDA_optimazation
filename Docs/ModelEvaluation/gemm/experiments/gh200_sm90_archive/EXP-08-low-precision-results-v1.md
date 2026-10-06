# GH200 FP8 与 INT8：指令序列的完成速率

32 个配置已完成实际数值检查、320 个正式进程和原 ARM 环境的完整重算。当前进入独立结果审查：24 个原生计算配置为参数候选，8 个 FP8 `mma.sync` 编译展开配置作为逻辑工作率对照。审查通过前不把候选写入合格参数库。

## 测量对象

固定指令形状、参与组数和累加链，重复执行计算并等待结果完成。它给 L0/L1 提供特定计算序列的服务速率，包含循环控制与规定的提交、等待和排空成本。

| 路径 | 输入与累加 | 形状 | 参与组与执行范围 |
|---|---|---|---|
| `mma.sync` INT8 | S8×S8 / U8×U8，S32 累加 | M16N8K32 | 1/2 warp；单 CTA / 全 GPU |
| WGMMA FP8 | E4M3×E4M3 / E5M2×E5M2，FP32 累加 | M64N64K32 | 1/2 warpgroup；单 CTA / 全 GPU |
| WGMMA INT8 | S8×S8 / U8×U8，S32 累加 | M64N64K32 | 1/2 warpgroup；单 CTA / 全 GPU |
| FP8 `mma.sync` 编译展开对照 | E4M3 / E5M2，逻辑 FP32 结果 | 逻辑 M16N8K32 | 1/2 warp；单 CTA / 全 GPU；不导出原生 FP8 参数 |

每组有两条累加链，每条链每迭代执行 16 条逻辑 MMA/WGMMA，正式循环固定 8192 次。WGMMA 使用 SS 操作数和 `wait 0`。完整线程数、CTA 数、寄存器和共享内存随配置记录，不能只按形状比较。

## 数值检查与性能输入

每配置先运行 1/2 次迭代的均匀、非均匀四个短 kernel，共 128 次短 kernel。行列、lane、组和累加链对应的完整输出由独立参考核对。随后每配置运行十个独立进程，按固定随机顺序采样；每进程执行 8–30 次有界预热，末五窗口 CV≤2% 才算预热收敛。

FP8 WGMMA 正式输入采用独立的正负乘积抵消方案，初始累加值 D0 区分组和链。该方案保持长循环有界；此前正输入长累加得到错误结果的[诊断记录](EXP-08-long-accumulation-result-v1.md)和[边界观察](EXP-08-boundary-result-v1.md)继续保留。INT8 的正式正输入累加值在 S32 范围内。父路径正式进程中的非均匀辅助检查不计入主测量工作量。

## 计量与手算

设指令形状为 M×N×K，参与组数为 G，累加链数为 C，每链每迭代指令数为 U，循环数为 I，CTA 数为 B，则逻辑工作量为：

`W = 2MNK × G × C × U × I × B`。

浮点使用 FLOP，整数使用 OP。单 CTA 用自身 `clock64` 的停止值减开始值；整卡用所有 CTA 的 `globaltimer` 完成包络，不跨 SM 相减 `clock64`。工作量除以 ns 的数值等于 GFLOP/s 或 GOP/s。

真实 FP8 样本 `bounded_v1_wgmma_e4m3_g2_one_cta`：

`2×64×64×32×2×2×16×8192×1 = 137438953472 FLOP`；
`137438953472 / 16803110 = 8179.37593 FLOP/clock64 cycle/CTA`。

真实 INT8 整卡样本 `mma_s8_g2_all_gpu`：

`2×16×8×32×2×2×16×8192×528 = 2267742732288 OP`；
`2267742732288 / 1800032 ns = 1259834.68 GOP/s/GPU`。

这两个算例各用第 0 批第 0 次进程；结果表报告全部十进程的中位数，两者不必相等。完整字段与 raw SHA 见[手算来源](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s08-main-execution/full-output-v1/runtime-v1/report-candidates-v1/worked-examples.json)。

## 结果

32 点均在第一批十进程后稳定，没有追加采样。浮点和整数分别画图，误差条表示十进程的最小值到最大值。单 CTA 表中 CV 显示为 0.000% 的点可能仍有低于显示精度的变化，应读取 CSV 原值。

![FP8 WGMMA 完成速率](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s08-main-execution/full-output-v1/runtime-v1/report-candidates-v1/fp8_service.png)

![INT8 完成速率](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s08-main-execution/full-output-v1/runtime-v1/report-candidates-v1/int8_service.png)

在本组条件下，INT8 `mma.sync` 从一组增加到两组后，单 CTA 速率约翻倍；WGMMA 单 CTA 的提升较小。整卡 WGMMA 在一组和两组配置间接近。参与组数同时改变线程数和驻留条件，因此这些对照不能单独确定硬件队列深度或最佳 tile。

FP8 编译展开对照保留 `logical_FLOP` 单位，描述编译后整个序列完成逻辑运算的速率。它不能作为 Hopper 原生 FP8 `mma.sync` 的单指令服务参数。

## 条件、原始数据与复现

本次 GPU 作业 734492 正常结束，用时 9 分 28 秒；设备 UUID 为 `GPU-ec947ba1-3e15-9860-163d-330d6c66d1b4`，目标为 SM90a/CUDA 12.9。32 次短检查进程和 320 次正式进程的完整原始输出保全为 911 个文件。ARM CPU 734502 使用与采样相同路径、SHA 和版本的 Python 3.9.21 完整重算，用时 2 分 56 秒。

- [完整结果表](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s08-main-execution/full-output-v1/runtime-v1/report-candidates-v1/RESULTS.zh.md)、[逐进程 CSV](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s08-main-execution/full-output-v1/runtime-v1/report-candidates-v1/samples.csv)、[配置 CSV](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s08-main-execution/full-output-v1/runtime-v1/report-candidates-v1/results.csv)。
- [候选参数与 8 个排除项](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s08-main-execution/full-output-v1/runtime-v1/report-candidates-v1/parameter-candidates.json)、[报告来源](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s08-main-execution/full-output-v1/runtime-v1/report-candidates-v1/sources.json)。
- [完整归档索引](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s08-main-execution/full-output-v1/runtime-v1/collection-job734492/collection-index.json)、[原始归档](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s08-main-execution/full-output-v1/runtime-v1/collection-job734492/actual.tar)、[原生重算与解释器身份](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s08-main-execution/full-output-v1/runtime-v1/native-replay-v1/complete734502/collection-index.json)。

换目录重生图表时保留归档 `actual/` 下的 `repo/` 和 `packs/`。下面只读取既有证据，不运行 GPU，也不授予新参数资格：

```sh
python3 -B /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s08-main-execution/full-output-v1/runtime-v1/report_low_precision.py \
  --run /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s08-main-execution/full-output-v1/runtime-v1/collection-job734492/actual \
  --replay /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s08-main-execution/full-output-v1/runtime-v1/native-replay-v1/complete734502/native-replay-v1/replay.json \
  --output /tmp/gh200-s08-new-report
```

输出目录须尚不存在。追加 `--tables-only` 可不依赖 Matplotlib。实际探针、完整矩阵运行器和收集器位于[冻结源码](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s08-main-execution/full-output-v1/runtime-v1/collection-job734492/actual/repo/)，测量执行说明见[批量方案](RUN-08-low-precision.md)。
