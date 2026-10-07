# GH200 资源微基准：结果与结论

设备：ROMEO GH200（CC 9.0，132 SM，L2 60 MiB，CUDA 可见 95 GiB HBM3），CUDA 12.9，`sm_90a`。采样时间 2026-10-01 至 10-05，suite `20261001-resource-suite-v2`。本轮只测单 GPU 资源与两套固定 tile 组合，不含完整 GEMM。

## 测量口径

| 项目 | 约定 |
|---|---|
| 单 CTA | 同一 CTA 的 `clock64` 差，单位为 X/cycle/CTA |
| 整卡 | 所有 CTA 的 `globaltimer` 最早起点到最晚终点，单位为 GFLOP/s 或 GB/s；不乘 SM 数换算 |
| 工作量 | FLOP、OP、逻辑字节分别计；逻辑字节不等于 HBM 物理流量（无 NCU 权限，缓存命中与物理流量均未测） |
| 计时窗口 | 包含循环控制、规定的等待、同步与结果排空；不是裸指令延迟 |
| 采样 | 每配置 10 个独立进程，8–30 次预热；默认报告中位数（例外在实验页注明），CV 普遍 <1% |
| 正确性 | 每次启动做完整输出校验，另有非均匀短输入检查；SASS 确认目标指令在循环内 |

例：FP32 FMA，1 CTA、32 线程、1 条链、每轮 16 个 FMA、128 轮，工作量 \(2\times32\times16\times128=131072\) FLOP，用时 9247 cycle，得 14.17 FLOP/cycle/CTA。

## 主要结论

下文"理论值"按 CC 9.0 每 SM 每周期：FP32 FMA 256 FLOP，FP16/BF16 Tensor Core 4096 FLOP，FP8/INT8 8192，SMEM 128 B。整卡 WGMMA 980 TFLOP/s 对应平均 SM 时钟约 1.81 GHz。

| 资源 | 结论 | 详见 |
|---|---|---|
| CUDA Core | FP32 整卡 65.7 TFLOP/s、FP64 33.5、FP16x2 117；单 SM 需 4 warp × 8 条独立链才接近 256 FLOP/cycle | [EXP-05](EXP-05-fma.md) |
| mma.sync | FP16/BF16 约 2700 FLOP/cycle，TF32 约 1340，均约为理论值的 65%；**FP64 `m8n8k4` 只有 128 FLOP/cycle，整卡 33.5 TFLOP/s，与 FP64 FMA 相同，仅为数据表 67 TFLOP/s 的一半** | [EXP-06](EXP-06-mma.md) |
| WGMMA | m64n64k16 单个 warpgroup 在 wait≥3 时达 4095.9 FLOP/cycle（满峰值），整卡 960–984 TFLOP/s；wait0 降到 3858；RS 与 SS 相差 <1% | [EXP-07](EXP-07-wgmma.md) |
| FP8 / INT8 | WGMMA 两个 warpgroup 达 8180 FLOP(OP)/cycle，整卡约 1.96 PFLOP/s；**均匀输入下，FP8 WGMMA 的 FP32 累加器到 64 后不再增长**，单个乘积为 \(2^{-8}\)；结果与约 14 bit 对齐窗口的解释相符；FP8 `mma.sync` 被编译为转换加 FP16 MMA，不是原生路径 | [EXP-08](EXP-08-fp8-int8.md) |
| SMEM | 8/16 B 访问的读、写、读写合计都约 128 B/cycle，建模时不能把读、写峰值相加；4 B 读只到 87；stride=k 时速率为 128/k 量级 | [EXP-09](EXP-09-smem.md) |
| ldmatrix / shfl | 单 warp 下 `ldmatrix.x4` 35 B/cycle，shfl 依赖链约 26 cycle/条；[R03](access_rules/R03-access-demand.md) 测得 8 warp 时 `ldmatrix.x4` 127、`stmatrix.x4` 116 B/cycle | [EXP-10](EXP-10-ldmatrix-shfl.md) |
| 同步 | CTA barrier 21–29 cycle/phase；mbarrier 218–226；GPU 范围 fence 267 cycle，CTA 范围 fence 与 `fence.proxy.async` 约 11 | [EXP-11](EXP-11-sync.md) |
| cp.async | 单 CTA 128 线程最多 7.5 B/cycle；整卡 16 B、4 stage 达 5.3 TB/s（源仅 8 MiB，小于 L2，速率高于 HBM 产品值） | [EXP-12](EXP-12-cp-async.md) |
| global | 大工作集读 3.52 TB/s、写 3.79、1:1 读写 3.43（HBM3 产品值 4 TB/s）；小工作集 `.ca` 读 13.5 TB/s | [EXP-13](EXP-13-global-rw.md) |
| TMA 1D bulk | 整卡读 ≥8 KiB 时 3.8 TB/s；**bulk 写只有 2.6–2.7 TB/s**；单 CTA 单请求受往返时间限制 | [EXP-14](EXP-14-tma-1d.md) |
| TMA 2D tensor | 连续布局读写整卡均 3.8–3.9 TB/s，SW128 无代价；行距 144 B（只按 16 B 对齐）时写降到 1.33–1.45 TB/s；[R05-E](access_rules/R05-async-lifecycle.md) 证实行距 160 B（32 B 对齐）和 256 B 都正常，原因是行起点对齐 | [EXP-15](EXP-15-tma-2d.md) |
| TMA 流水 | 单 CTA 读速率随缓冲容量增加：16 KiB 时 14.7 B/cycle，64 KiB 时 28.5–39.8；S1R1 时每个 16 KiB 请求的完整周期约 1110 cycle，是单请求往返时间的上界，不是实测延迟；整卡与 S、R 无关，约 3.7 TB/s | [EXP-16](EXP-16-tma-pipeline.md) |
| cluster / DSM | 整卡 DSM 读约 2.2–2.5 TB/s，本地 SMEM 读 10–12 TB/s；TMA 多播在 C=2/4/8 时接收量均约 5 TB/s，逻辑源请求按 1/C 下降 | [EXP-17](EXP-17-cluster.md) |
| 驻留与辅助 | local 访问每字约 59 cycle；同址原子约 73 cycle/轮，无争用 28；cvt、IMAD.WIDE 等依赖链 29–43 cycle/轮 | [EXP-18](EXP-18-occupancy-aux.md) |
| FP32 tile 组合 | 32×32×32、128 线程只有 18 FLOP/cycle（峰值的 7%）；搬运与计算重叠后每 Ktile 从 4022 降到 3683 cycle | [EXP-19](EXP-19-fp32-tile.md) |
| BF16 WGMMA 组合 | 64×64×64、1 warpgroup、每 Ktile wait0：纯计算 830 FLOP/cycle（20%），平均每条 WGMMA 约 150 cycle（含 fence、commit、wait0 与循环，不是裸延迟）；加入 TMA 供给后每 Ktile 约 1000–1120 cycle，搬运未被隐藏 | [EXP-20](EXP-20-bf16-wgmma-tile.md) |

## 锚点与访问规则实验（access_rules，2026-10-06 起）

FP16 Tensor Core GEMM 的后续实验见 [access_rules](access_rules/README.md)。已完成部分的主要结论：

| 实验 | 结论 |
|---|---|
| [R00](access_rules/R00-anchor-target.md) 完整 GEMM | cuBLASLt FP16 8192³ 715 TFLOP/s（数据表的 72%）、2048³ 555（56%）；FP8 8192³ 1127（57%）；固定 CUTLASS 128×256×64 为 643 |
| [R00](access_rules/R00-anchor-target.md) 目标 WGMMA | `m64n{64,128,256}k16`、1/2 个 warpgroup、SS/RS、FP16/BF16 都达 4095 FLOP/cycle；整卡 FP16 989、FP8 1978 TFLOP/s。完整 GEMM 的差距不在 WGMMA 指令本身 |
| [R05-D](access_rules/R05-async-lifecycle.md) 每 SM 供给 | 目标在途量（48 KiB/stage，1 CTA/SM）下，所有 CTA 共享 L2 中的小源时每 SM 55.5 B/cycle，高于满速 WGMMA 需要的 48；独立大源只有 15.3。满速需要依靠跨 CTA 的 L2 复用 |
| [R05-E](access_rules/R05-async-lifecycle.md) TMA 写回 | 行距 144 B 慢 38%，160/256 B 正常：行起点 32 B 对齐即可 |
| [R03](access_rules/R03-access-demand.md) 片上并发 | 8 warp 时 LDS.128/STS.128/`ldmatrix.x4` 达 124–127 B/cycle，`stmatrix.x4` 116 |
| [R01](access_rules/R01-readiness.md) 依赖时间 | 每步：FFMA 4.0–4.3 cycle、add 5.0、LEA+LDS 29、global L2/HBM 288/630、`ldmatrix.x4` 34；WGMMA 发出到 wait0 返回约 36+0.62·N cycle（N=256 为 195） |
| [R06](access_rules/R06-issue-residency.md) 资源 | 128 线程 FFMA 单 CTA 241 FLOP/cycle、整卡 64–65 TFLOP/s，63–128 寄存器之间差别 <2%；窗口内 clock64/globaltimer 比值约 1.97–1.98 GHz |
| [V01](access_rules/V01-validation.md) 规则预测 | 异步目标组合主循环 1026 cycle/Ktile（理想 1024）；单 CTA 预测误差 −0.4%~−1.7%；整卡 −7%~−10%（频率假设偏高）；完整 CUTLASS −23%~−28%（漏了固定开销） |
| [R07](access_rules/R07-anchor-clock-fixedcost.md) 频率与固定开销 | 持续 GEMM 时 SM 频率 1.40–1.45（8192³）、约 1.75 GHz（2048³），几乎所有样本只报 SW Power Cap，模块功率上限 680 W；CUTLASS 需 `-DNDEBUG`（否则每条 MMA 后 wait0，慢 5.5–8%）；M=N=2048 单调用经验关系 T≈9.45+0.637·Ktile µs |


## 尚未覆盖

- 基于规则的完整 GEMM 预测：规则需要加入每 kernel 固定开销和负载频率，并解释完整 kernel 每 Ktile 比纯计算多出的约 20%（按约 1.95 GHz 换算的估计）后，再按先预测后测量重新验证。直接使用 R07 的经验关系时，这部分已包含在斜率中。
- TF32 WGMMA、FP16 累加形式；FP64 `mma` 的 sm_90 新形状。
- 多 CTA 竞争下的片上服务。
- 物理 HBM 流量、缓存命中、动态指令计数（需 NCU 权限）。
- 物理在途队列深度（软件 stage 扫描不能给出）。

## 后续实验约定

每个新实验交付一个脚本、一份原始数据目录和一页本目录下的结论文档。正文须让读者答出：具体配置、分子怎么算、计时窗口到哪里结束、结论适用的条件、数据与脚本在哪；由数据推出的解释写明是推断。上线前检查四项：SASS 含目标指令、输出数值正确、样本稳定（CV 超过 5% 时保留全部样本并注明）、结果与已知物理量级相符。原 A/B/C 审查、发布桥接和 S21/S22 换目录验收冻结在现状，不再推进。

## 数据与代码

- 原始数据：`results/gh200_resource_campaign/20261001-resource-suite-v2/`（不入 git），各实验文档给出具体目录。
- 全部 754 条条件观测的索引：[index.json](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s21-seventeen-family-index-v1/index.json)。
- 探针与运行脚本：[microbench/gh200_resource_campaign](../../../../../microbench/gh200_resource_campaign/README.md)。
- 实验设计、审查流程、旧版结果和 v1 历史记录：[gh200_sm90_archive](../gh200_sm90_archive/ARCHIVE.md)。
