# R00：完整 GEMM 基线与目标 WGMMA 形状

[实验索引](README.md)。状态：30 点完成（2026-10-06）。代码与完整命令见[运行入口](../../../../../../microbench/gh200_resource_campaign/access_rules/README.md)。

## 问题

1. GH200 上完整 GEMM 实测能到多少？现有微基准都是孤立资源，没有完整 kernel 的完成时间可对照。
2. 实际 Hopper GEMM 用的 WGMMA 形状（`m64n256`、2 个 consumer warpgroup）能否达到 EXP-07 中 `m64n64` 的峰值？这决定后续实验和方案的 tile。

## R00-A：完整 GEMM 基线（15 点）

| 实现 | 精度（A/B → 累加 → D） | 条件 | 数量 |
|---|---|---|---:|
| cuBLASLt | FP16→FP32→FP32（主实例）、BF16→FP32→FP32、FP8 E4M3→FP32→BF16、FP32→FP32→FP32（`CUBLAS_COMPUTE_32F_PEDANTIC`，不用 TF32） | 2048³ 驱逐准备后调用、2048³ 相同输入重复调用、8192³ 相同输入重复调用 | 12 |
| cuBLASLt | FP16→FP32→FP32 | 2000³（不整除常见 tile） | 1 |
| CUTLASS sm90 | FP16→FP32→FP32；tile128×256×64、cluster2×1×1、TMA warp-specialized cooperative、4 stage；无C输入，epilogue tile128×32 | 8192³、4096³，相同输入重复调用 | 2 |

**布局与计时**

- A/B/D 行主序，α=1、β=0，不读 C。库若要求预打包，布局适配单列，不计入主窗口。
- 相同输入重复调用：预热后取单次调用时间。驱逐准备：每次调用前写一个 2×L2 大小的独立缓冲，完成后再开始计时。
- CUDA event 包围同一 stream 中的单次 matmul；分配、描述符、启发式查询、workspace、驱逐操作在窗口外。报告 \(T\)、\(2MNK/T\) 及其与产品规格的比例。
- 频率：记录 GEMM 运行期间的遥测（如 NVML 采样），注明采样间隔；取不到就记为未知，不用其他 kernel 代替。

**配置记录**

- cuBLASLt：采样前完成启发式查询，固定库版本、workspace 上限，取第一个合法候选；同一尺寸的两种缓存准备用同一算法。记录 algo 及可查询的 tile、stage、cluster、split-K，以及实际 kernel 名（`nsys` 可用时）。
- CUTLASS：固定版本与提交，记录调度器、mainloop/epilogue 类型、线程数、寄存器、SMEM。4 stage 不能启动时记录原因，不换配置。

**正确性**

- 每个运行尺寸抽 4096 个输出：随机内部位置、首末行列、tile 边界与尾块；参考为按实际存储输入解码后的 FP64 点积。另对一个小尺寸做全量检查。
- 报告最大绝对误差、抽样相对L2误差，以及非近零参考上的最大相对误差；抽样范数不能称全矩阵Frobenius误差。`workloads.yaml`的容差为空，本组只报告任务误差；有限精确输入的实现见证单独检查。
- FP8：参考按量化后的 A/B、缩放因子与 BF16 输出舍入计算；量化本身引入的误差另列。

CUTLASS 的 2048³ 与 2048×2048×8192 留给 [V01](V01-validation.md) 作完整 kernel 留出点，本组不运行。

## R00-B：目标 WGMMA 形状（15 点）

计时与输出检查沿用 [EXP-07](../EXP-07-wgmma.md)。A/B 在 SMEM，K-major、128B swizzle 描述符，FP32 累加。

- **提交与等待**：每条累加链每轮 16 条 WGMMA；一个 warpgroup 每轮的 \(16\times\text{链数}\) 条一起 commit 一次，然后 `wait_group 1`；循环后 wait0 并排空累加器。
- **链数**：为使每线程累加寄存器都是 128 个，N=64 用 4 条链、N=128 用 2 条、N=256 用 1 条。N=256 单链即实际 GEMM 中一个 warpgroup 累加一个输出 tile 的情形。改变 N 时链数和每组指令数随之变化，比较的是"固定累加容量"下的配置。

| 子集 | 坐标 | 数量 |
|---|---|---:|
| FP16 SS，单 CTA | `m64nNk16`，N=64/128/256 × 每 CTA 1/2 个 warpgroup | 6 |
| FP16 SS，整卡 | N=128/256，2 个 warpgroup | 2 |
| FP16 RS，单 CTA | N=256 × 1/2 个 warpgroup | 2 |
| BF16 SS 迁移对照，单 CTA | N=256 × 1/2 个 warpgroup，其余与 FP16 点相同 | 2 |
| FP8 E4M3 SS | `m64n256k32`；单 CTA 1/2 个 warpgroup，整卡 2 个 warpgroup | 3 |

工作量：每条 \(2\times64\times N\times K\)，乘每链指令数、链数、warpgroup 数、轮数；单 CTA 报 FLOP/cycle/CTA，整卡报 GFLOP/s。记录寄存器、SMEM、occupancy、spill 和整卡 grid；有 spill 的点单列。FP8 长循环沿用有界输入与全输出参考。

本实现用CUTLASS v3.9.2的CuTe指令/描述符封装。128B swizzle行按128B补齐，描述符只覆盖K=16/32；实际分配记录。短检查用非均匀坐标输入和1/2条操作；正式FP16/BF16均匀输入，FP8在16条批内交替正/负A抵消，保持累加有界。性能适用范围随这些输入条件引用。

## 实现要点

- R00-A：cuBLASLt 固定首个合法启发式候选、64 MiB workspace；FP8 用 TN 布局适配、scale=1、关闭 fast accumulation，适配在窗口外。CUTLASS 为 v3.9.2，128×256×64、cluster 2×1×1、384 线程、cooperative、4 stage、`ElementC=void`、epilogue tile 128×32，SMEM 231424 B（带 C 输入的默认 epilogue 需 264192 B，超过 232448 B 上限，因此不读 C）。
- CUTLASS 的布局是 A K-major、B MN-major（SASS 带 `.tnspB`），producer/consumer 寄存器经 `setmaxnreg` 分为 40/232；R00-B 与 V01 的手写探针 A/B 均为 K-major。比较两者时这些条件不同。
- R00-B：CuTe 描述符封装，128B swizzle 行按 128 B 补齐；正式 FP16/BF16 为均匀输入，FP8 在批内交替正负 A 以保持累加有界。
- 2048³ 的两种缓存准备在原运行中没有相邻成对执行，只作各自的单点统计；运行入口已改为相邻成对。

## 条件扩展：FP64 新形状（4 点）

需要解释 EXP-06 FP64 只到产品值一半时启用：按 PTX ISA 核实 sm_90 的 f64 `mma.sync` 形状（如 `m16n8k4/k8/k16`），4 warp、8 链各测单 CTA，最快形状再测整卡。

## 结果（2026-10-06，job734996，GPU-ec947ba1…）

**R00-A：完整 GEMM**（CUDA event 单次调用，中位数，10 个进程）

| 实现 / 精度 | 2048³ 驱逐准备后 | 2048³ 重复调用 | 8192³ 重复调用 |
|---|---:|---:|---:|
| cuBLASLt FP16→FP32 | 537 TFLOP/s | 555 | 715 |
| cuBLASLt BF16→FP32 | 532 | 556 | 698 |
| cuBLASLt FP8→BF16 | 772 | 825 | 1127 |
| cuBLASLt FP32（PEDANTIC） | 48.9 | 48.7 | 51.6 |
| 固定 CUTLASS FP16 | — | — | 643（4096³：663） |

cuBLASLt FP16 2000³ 为 517 TFLOP/s。固定 CUTLASS 编译时没有 `-DNDEBUG`，ptxas 因断言代码（C7510）把每条 MMA 改成提交后 wait0，R07 在另一张卡上同次对比，加 `NDEBUG` 快 5.5–8%，所以 643/663 TFLOP/s 偏低；见 [R07](R07-anchor-clock-fixedcost.md)。8192³ 时 FP16 达数据表 990 TFLOP/s 的 72%，FP8 为 1979 的 57%，FP32 为 67 的 77%。

**R00-B：目标 WGMMA 形状**

| 配置 | 结果 |
|---|---|
| FP16 SS `m64n{64,128,256}k16`，1/2 个 warpgroup，单 CTA | 4095.1–4095.6 FLOP/cycle |
| FP16 RS N=256、BF16 SS N=256，1/2 个 warpgroup，单 CTA | 4095.1–4095.6 FLOP/cycle |
| FP8 SS `m64n256k32`，1/2 个 warpgroup，单 CTA | 8190–8191 FLOP/cycle |
| FP16 SS N=128/256，2 个 warpgroup，整卡 | 989 TFLOP/s |
| FP8 SS N=256，2 个 warpgroup，整卡 | 1978 TFLOP/s |

**结论**：目标形状在任意 N、warpgroup 数、SS/RS 下都达到峰值，整卡也达到数据表值。完整 GEMM 与指令峰值之间的差距（8192³ FP16 约 28%，2048³ 约 44%）来自供给、流水、epilogue、波次与尾部，而不是 WGMMA 本身。FP8 的差距更大（43%）。

手算：8192³ 工作量 1099511627776 FLOP，除以 1.538160 ms 得 714.8 TFLOP/s。

[原始报告](../../../../../../results/gh200_resource_campaign/access_rules/20261006-r00-job734996/formal-v1/report.md) · [CPU 重算与图](../../../../../../results/gh200_resource_campaign/access_rules/20261006-r00-job734996/offline-replay-v1/report.md)。频率遥测未取得，按未知处理。

## 输出

- R00-A：各精度、尺寸的实测基线，所选实现配置，两种缓存准备的差别。
- R00-B：FP16 目标形状与 warpgroup 数的服务，BF16 与 FP16 的差别，FP8 代表点；据此选定 R01-C、R04、R05、V01 使用的形状。

依据：[cuBLAS](https://docs.nvidia.com/cuda/archive/12.9.1/cublas/index.html)、[PTX ISA 8.8](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html)。


## K/MN布局成对补充（2026-10-06）

[结果与复现报告](../../../../../../results/gh200_resource_campaign/access_rules/20261006-v01-followup/report.md)。补充4个条件、26个正式进程，不扩大默认矩阵：FP16 SS m64n256k16、2 warpgroup、128个累加寄存器/线程，分别用B Major::K和Major::MN，单CTA各3进程、整卡各10进程。每相邻一对只改变B布局，公共源入口为`r00_wgmma.cu --b-major K/MN`，配对采样入口为`run_layout_pair.py`。

两版均154 registers/thread、动态SMEM49152 B、无local；正式循环为1024轮，每轮16条MMA、commit/wait1、最后wait0。K/MN的单CTA中位数分别4195097/4195107 cycle，整卡中位数均2311808 ns。MN机器码的16条HGMMA均带`.tnspB`；非均匀1/2轮短检查通过，26个正式输出已逐值独立复核。

当前条件没有观测到足以解释完整CUTLASS约24%预测偏差的孤立MN服务差异。这个对照未包含完整kernel的TMA输入、cluster/多播、寄存器重分配和调度，不把结论迁移成这些机制无成本。
