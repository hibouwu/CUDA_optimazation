# EXP-19：FP32 固定 tile 组合

[结果总览](README.md)

## 结论

- 32×32×32 tile、1 CTA、128 线程的纯计算只有 18.4 FLOP/cycle，约为 FP32 理论值的 7%。每 Ktile 65536 FLOP 用 3548 cycle；按 4 warp 发射 1024 条 warp 级 FFMA 只需约 256 cycle；推断其余时间花在 SMEM 读取、依赖和同步上（未分别测量）。**这个配置是演示组合关系的小例子，不代表实际 SIMT GEMM 的 mainloop。**
- 搬运与计算的组合关系在该配置下清楚可见（每 Ktile 斜率）：纯计算 3548，纯搬运 760–926，串行 4022，重叠 3647–3769。串行比计算多 474 cycle/Ktile，重叠后多出部分剩 221 / 135 / 99（stage 1/2/4），即收回 53% / 72% / 79%。
- stage 从 1 增到 2、4 时重叠斜率只从 3769 降到 3683、3647：计算远慢于搬运，加 stage 收益很小。
- 输出处理在每个序列上增加约 800 cycle 的固定项（拟合截距之差），斜率不变。
- 直线拟合在 K=64 检查点上误差 <0.5%；transport stage 4 的样本内最大误差 8.6%。

## 配置

1 CTA、128 线程、FP32；tile 32×32×32，每 Ktile 输入 8 KiB。五种模式：compute（输入已在 SMEM）、transport（只搬运与校验）、serial、overlap、output（overlap + 输出处理）。stage 1/2/4；K = 1, 2, 4, 8, 16, 32 个 Ktile 拟合，64 为检查点。所有模式预留同样 4 个物理槽（32 KiB）。

每个完整 K 序列的主窗口工作量（样本总量再乘重复次数 N=305）：

| 模式 | 计入的 FLOP | 窗口内 GMEM 输入 | 窗口内 GMEM 输出 |
|---|---:|---:|---:|
| compute | \(65536K\) | 0（输入在窗口前预取） | 0 |
| transport | 0 | \(8192K\) B | 0 |
| serial / overlap | \(65536K\) | \(8192K\) B | 0 |
| output | \(65536K+2048\) | \(8192K\) B | 4096 B |

output 对 32×32 个累加结果各做一次 `0.5*C+bias`（2 FLOP/元素），**变换、写回和 `threadfence` 都在主计时窗口内**，所以 output 的 FLOP/cycle 含输出 FMA，不是纯 mainloop 工作率。其他模式的最终结果导出在停止计时之后。每个序列开始时累加器清零在窗口内；输入与缓冲初始化、正确性回读在窗口外。

## 结果

K=8、stage=2：

| 模式 | cycle/序列 | 工作率 |
|---|---:|---|
| compute | 28535 | 18.37 FLOP/cycle |
| transport | 6620 | 9.90 B/cycle |
| serial | 32331 | 16.22 FLOP/cycle |
| overlap | 29785 | 17.60 FLOP/cycle |
| output | 30584 | 17.21 FLOP/cycle |

拟合 \(T(K)=a+bK\)（cycle/序列）：

| 模式 | stage 1：a / b | stage 2：a / b | stage 4：a / b |
|---|---|---|---|
| compute | 151 / 3548 | 151 / 3548 | 151 / 3548 |
| transport | 191 / 926 | 287 / 792 | 447 / 760 |
| serial | 149 / 4022 | 151 / 4022 | 148 / 4022 |
| overlap | 186 / 3769 | 311 / 3683 | 456 / 3647 |
| output | 975 / 3771 | 1072 / 3684 | 1215 / 3649 |

![overlap](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s19-main-execution/actual-report-v1/overlap.png)

## 数据

[results.csv](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s19-main-execution/actual-report-v1/results.csv)、[samples.csv](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s19-main-execution/actual-report-v1/samples.csv)、[拟合](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s19-main-execution/actual-report-v1/sequence-fits.json)、[完整报告目录](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s19-main-execution/actual-report-v1/RESULTS.zh.md)。脚本：[families/s19](../../../../../microbench/gh200_resource_campaign/families/s19/README.md)。
