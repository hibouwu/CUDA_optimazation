# EXP-20：BF16 WGMMA 固定 tile 组合

[结果总览](README.md)

## 结论

- 64×64×64 tile、1 个 warpgroup、每 Ktile 4 条依赖的 `m64n64k16` 后 wait0：纯计算每 Ktile 606 cycle，只有 830 FLOP/cycle（理论值的 20%）。平均到每条 WGMMA 约 150 cycle，这包含 `wgmma.fence`、commit、wait0、同步和循环控制，不能当作 L0 的裸延迟。**这是依赖与等待受限的配置**；[EXP-07](EXP-07-wgmma.md) 中多链、wait≥3 时同一指令可达满峰值。
- 加入 TMA 供给后，overlap 每 Ktile 1002–1123 cycle，串行 1330–1358，纯搬运 1338–1472：单 CTA、16 KiB/Ktile 的 TMA 供给（约 12 B/cycle）比计算慢一倍以上；推断 overlap 主要受搬运限制，与 [EXP-16](EXP-16-tma-pipeline.md) 中单 CTA 速率随缓冲容量上升一致。
- overlap 比纯搬运还快；两者的消费者不同（transport 由线程读校验全部数据，overlap 由 WGMMA 读 SMEM），推断差别来自消费侧。
- 输出处理在每个序列上增加约 2500 cycle 的固定项（拟合截距之差），相当于 2.5 个 Ktile 的 overlap 时间。
- stage 1→2 的 overlap 斜率从 1123 降到 1002，stage 4 没有进一步改善（1033）；拟合最大误差 9.5%，K=64 检查点误差 <0.3%。

## 配置

1 CTA、128 线程、BF16 输入、FP32 累加，A/B 在 SMEM。每 Ktile 两块 64×64 输入共 16 KiB，由 TMA 载入，mbarrier 判定就绪，`wgmma.fence`/commit/wait0 后释放槽。五种模式同 [EXP-19](EXP-19-fp32-tile.md)；4 个物理槽（64 KiB）+ 4 个 mbarrier。

每个完整 K 序列的主窗口工作量（样本总量再乘重复次数 N=733）：

| 模式 | 计入的 FLOP | 窗口内 GMEM 输入 | 窗口内 GMEM 输出 |
|---|---:|---:|---:|
| compute | \(524288K\) | 0（输入在窗口前预取） | 0 |
| transport | 0 | \(16384K\) B | 0 |
| serial / overlap | \(524288K\) | \(16384K\) B | 0 |
| output | \(524288K+8192\) | \(16384K\) B | 16384 B |

output 对 64×64 个 FP32 累加结果各做一次 SIMT `0.5*C+bias`（2 FLOP/元素），**变换、写回和 `threadfence` 都在主计时窗口内**，所以 output 的 FLOP/cycle 含输出 FMA，不是纯 WGMMA 工作率。其他模式的最终结果导出在停止计时之后。每个序列开始时累加器清零在窗口内。

## 结果

K=8、stage=2：

| 模式 | cycle/序列 | 工作率 |
|---|---:|---|
| compute | 5052 | 830.2 FLOP/cycle |
| transport | 11058 | 11.85 B/cycle |
| serial | 10773 | 389.3 FLOP/cycle |
| overlap | 8357 | 501.9 FLOP/cycle |
| output | 10888 | 386.0 FLOP/cycle |

拟合 \(T(K)=a+bK\)（cycle/序列）：

| 模式 | stage 1：a / b | stage 2：a / b | stage 4：a / b |
|---|---|---|---|
| compute | 204 / 606 | 204 / 606 | 204 / 606 |
| transport | 309 / 1472 | 369 / 1339 | 272 / 1370 |
| serial | 164 / 1330 | 87 / 1349 | 132 / 1358 |
| overlap | 285 / 1123 | 364 / 1002 | 262 / 1033 |
| output | 2768 / 1127 | 2849 / 1004 | 2751 / 1035 |

![overlap](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s20-main-execution/actual-report-v1/overlap.png)

## 数据

[results.csv](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s20-main-execution/actual-report-v1/results.csv)、[samples.csv](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s20-main-execution/actual-report-v1/samples.csv)、[拟合](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s20-main-execution/actual-report-v1/sequence-fits.json)、[完整报告目录](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s20-main-execution/actual-report-v1/RESULTS.zh.md)。脚本：[families/s20](../../../../../microbench/gh200_resource_campaign/families/s20/README.md)。
