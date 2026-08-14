# Thor SM110 GEMM 阶段化解析模型：从数据流到可验证的性能预测

> **目标对象**：NVIDIA Jetson AGX Thor / SM110 上的 FP16 × FP16 → FP32 GEMM  
> **具体实现**：`GEMMsm110` 的 `tc3`，CTA tile 为 \(128\times128\times64\)，2-stage TMA pipeline  
> **模型定位**：解释性能由哪些阶段构成，并给出可校准、可证伪的预测公式；不是用一个经验系数拟合所有 GEMM  
> **当前状态**：模型结构完整；计算阶段已有同 shape、同 `kind::f16` 指令族的 BF16 Thor 实测代理值，但仍需 FP16 同构复测；TMA 和 epilogue 也需补齐后才能声称“数值预测已闭环”

## 1. 先说结论

对当前 `tc3` kernel，一个足够简单、同时又有代码依据的总时间模型是：

\[
T_{\text{pred}} =
T_{\text{launch}}
+N_{\text{wave}}
\left[
T_{\text{fixed}}
+t_L
+(N_K-1)\max(t_L,t_C)
+t_C
+t_E
\right].
\]

这条式子不是从经验曲线猜出来的。它逐项对应 `tc3` 的执行顺序：

1. kernel 启动一次，因此加 \(T_{\text{launch}}\)；
2. 每一批 SM 并行处理一组输出 CTA，因此乘 CTA service wave 数 \(N_{\text{wave}}\)；
3. 第一个 K-stage 必须先由 TMA 装入，所以先付出 \(t_L\)；
4. 中间 \(N_K-1\) 个间隔中，下一次 TMA load 与当前 MMA 确实可以双缓冲重叠，所以每个间隔由较慢者决定；
5. 最后一次 MMA 后面已经没有下一次 load，但它仍必须完成，所以再加 \(t_C\)；
6. accumulator 完成后才能从 TMEM 回读并写回 C，所以 epilogue 时间 \(t_E\) 与 mainloop 相加；
7. TMEM 分配、barrier 初始化和不能归入数据量的控制成本放入 \(T_{\text{fixed}}\)。

预测性能随后只是“总计算量除以总时间”：

\[
P_{\text{pred}}=\frac{2MNK}{T_{\text{pred}}}.
\]

这里的系数 2 也有明确含义：一次乘加 \(a\times b+c\) 按一次乘法和一次加法计为 2 FLOP。

## 2. 模型边界：先固定研究对象

### 2.1 本文建模哪一条 kernel 路径

本文不建模一个抽象的“任意 Blackwell GEMM”，而是先固定仓库中已经实现并测过的 `tc3`：

| 项目 | `tc3` 的实际取值 | 依据 |
| --- | ---: | --- |
| 输入与累加 | FP16 × FP16 → FP32 | `Tc3Runner` 与 `mma_f16` |
| CTA tile | \(B_M=128, B_N=128, B_K=64\) | kernel 模板常量 |
| MMA atom | \(128\times128\times16\) | 每个 K-stage 发射 4 条 `tcgen05.mma` |
| CTA threads | 128 | launch 配置 |
| SMEM stages | 2 | `kStages = 2` |
| 每 stage SMEM | 32 KiB | A 16 KiB + B 16 KiB |
| CTA 动态 SMEM | 64 KiB | 2 个 stage |
| accumulator | TMEM | `tcgen05.alloc`、`tcgen05.mma` |
| 写回 | TMEM → registers → GMEM | `tcgen05.ld` 后 `float4` store |
| 支持的尺寸 | \(M,N\) 是 128 的倍数，\(K\) 是 64 的倍数 | runner 的显式检查 |

代码依据见 [`tc3_pipeline.cuh`](../../GEMMsm110/include/backends/tc3_pipeline.cuh) 和 [`sm110_ptx_helpers.cuh`](../../GEMMsm110/include/sm110_ptx_helpers.cuh)。

### 2.2 三类信息必须分开

为了避免“公式看起来完整，但参数没有来源”，本文把输入分成三类：

| 类别 | 例子 | 使用方式 |
| --- | --- | --- |
| 硬件或代码事实 | Thor 实机报告 20 SM；`tc3` 使用 \(128\times128\times64\) tile | 可以直接进入离散计数和数据量公式 |
| 实测参数 | BF16 K4 mainloop 为 113.015 TFLOP/s | 可作同指令族代理值；FP16 最终参数仍应同构复测 |
| 模型假设 | 满载 wave 的有效速率在所验证尺寸范围内近似稳定 | 必须通过多尺寸误差验证，不能写成硬件定律 |

官方资料给出 Jetson T5000 为 10 TPC、2560 CUDA cores、273 GB/s LPDDR5X；仓库中的目标设备记录为 20 SM。本文使用 20 SM 做调度计数，但不使用有歧义的“Tensor Core 个数”推导吞吐，而直接使用微基准标定计算速率。官方规格可参见 [NVIDIA Jetson Thor 产品页](https://www.nvidia.com/en-gb/autonomous-machines/embedded-systems/jetson-thor/)；本地设备记录见 [`sm110_gemm_bank_conflict_research.md`](../cutlass/sm110_gemm_bank_conflict_research.md)。

## 3. 从代码得到阶段依赖图

当前 kernel 的单个 CTA 数据流是：

```text
A/B in GMEM
    │
    ├─ TMA load stage i ───────────────┐
    │                                  ▼
    └─ TMA load stage i+1      tcgen05.mma stage i
             │                 SMEM → TMEM accumulator
             └──── 可重叠 ─────────────┘
                                      │
                                      ▼
                             tcgen05.ld: TMEM → registers
                                      │
                                      ▼
                                vector store → C in GMEM
```

NVIDIA 的 PTX 文档把 `cp.async.bulk.tensor` 定义为非阻塞的 tensor copy；`tcgen05.mma` 也是异步操作，accumulator 位于 TMEM。`tc3` 又通过两个 SMEM stage、TMA mbarrier 和 MMA completion barrier 明确实现了“装入下一 stage、计算当前 stage”的顺序。因此，TMA 与 MMA 的稳态部分可以取最大值，而不是相加。相关指令语义见 [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) 和 [CUTLASS tcgen05 MMA Programming Guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/tcgen05_programming.html)。

相反，`tc3` 在所有 K-stage 完成并等待 MMA barrier 之后才执行 `tcgen05.ld` 和全局写回。这里存在数据依赖，所以第一版模型不能把 epilogue 与本 CTA 的 mainloop 取最大值。

## 4. 第一层：只计算必做的工作量

### 4.1 输出 CTA 数与 K-stage 数

一般写法使用向上取整：

\[
N_M=\left\lceil\frac{M}{B_M}\right\rceil,
\qquad
N_N=\left\lceil\frac{N}{B_N}\right\rceil,
\qquad
N_{\text{CTA}}=N_MN_N.
\]

\(N_M\) 表示 M 方向需要多少个输出 tile，\(N_N\) 表示 N 方向需要多少个输出 tile；二者相乘就是输出 CTA 总数。

K 方向的 stage 数为：

\[
N_K=\left\lceil\frac{K}{B_K}\right\rceil.
\]

当前 `tc3` 不实现边界补零，而是直接要求整除，所以在本文的实测范围内，上述三个向上取整都可以换成普通整数除法。保留一般写法，是为了以后扩展 predication 时不必重写模型。

### 4.2 一个 CTA 的一个 K-stage 搬多少输入

A tile 的形状是 \(B_M\times B_K\)，B tile 的形状是 \(B_K\times B_N\)。若 A、B 每个元素分别占 \(s_A,s_B\) 字节，则：

\[
d_L=B_MB_Ks_A+B_NB_Ks_B.
\]

式子的第一项是 A tile 的元素数乘元素字节数，第二项同理对应 B tile。因此 \(d_L\) 的量纲是 byte，它表示一次 K-stage 的**逻辑 TMA 搬运量**，不是 DRAM 实际流量。

对当前 FP16、\(128\times128\times64\) tile：

\[
\begin{aligned}
d_L
&=128\times64\times2+128\times64\times2 \\
&=32768\ \text{B}=32\ \text{KiB}.
\end{aligned}
\]

这也与代码中的两个 16 KiB stage buffer 一致。

### 4.3 一个 CTA 的一个 K-stage 做多少计算

一个输出 tile 有 \(B_MB_N\) 个元素，每个元素在该 stage 内做 \(B_K\) 次乘加，因此：

\[
f_C=2B_MB_NB_K.
\]

量纲是 FLOP。代入当前 tile：

\[
\begin{aligned}
f_C
&=2\times128\times128\times64 \\
&=2097152\ \text{FLOP}.
\end{aligned}
\]

代码每个 stage 发射 4 条 \(128\times128\times16\) MMA；用指令数复核也得到：

\[
4\times\left(2\times128\times128\times16\right)
=2097152\ \text{FLOP}.
\]

两条独立路径得到同一个数，是对 tile 解释和指令解释的一次交叉校验。

### 4.4 一个 CTA 的 epilogue 写多少输出

当前 kernel 每个输出只写一次 FP32 值。若输出元素占 \(s_C\) 字节，则：

\[
d_E=B_MB_Ns_C.
\]

对 FP32 输出：

\[
d_E=128\times128\times4=65536\ \text{B}=64\ \text{KiB}.
\]

这里把 \(d_E\) 定义成最终输出字节数。后面用到的 \(B_E\) 因而是“每秒完成多少输出字节”的复合 epilogue 速率，内部已经包括 TMEM read、寄存器搬运和 global store；它不是某条总线的物理带宽。

## 5. 第二层：把工作量换成阶段时间

### 5.1 为什么使用 SM service wave

当前路径是 `cta_group::1`，一个输出 CTA 只在一个 SM 上执行。Thor 目标设备有 \(S=20\) 个 SM，所以一次最多有 20 个 SM 同时为 20 个不同输出 CTA 提供 Tensor/TMA 服务。

定义 service wave 数：

\[
N_{\text{wave}} =
\left\lceil\frac{N_{\text{CTA}}}{S}\right\rceil.
\]

这里的 wave 不是说 GPU 只能驻留 20 个 CTA。多个 CTA 可以同时 resident，用来隐藏等待；但同一 SM 上的 resident CTA 仍然共享该 SM 的 Tensor、TMA、SMEM 和寄存器资源。模型把这些并发效果吸收到“满 SM 的有效阶段速率”里，再用 service wave 表示每个 SM 总共要服务多少份 CTA 工作。

这个写法还自然表达最后一波不满：例如 64 个 CTA 在 20 个 SM 上至少需要 4 份 CTA service time，最后只有 4 个 SM 有工作，不能把总 FLOP 简单除以满芯片吞吐而忽略尾波。

### 5.2 Load stage 时间

L2 共享总线和每个 SM 的 TMA→SMEM 出口是两个资源，不能压成一个
aggregate \(B_L\)。定义 \(B_{\mathrm{L2}}\) 为整 GPU 共享 L2 read rate，单位
byte/s/GPU；定义 \(C_{\mathrm{TMA,SM}}\) 为用单 CTA 隔离测得的一个 SM 的
TMA→SMEM ingress，单位 byte/s/SM。一个满 service wave 包含 \(S\) 份 CTA
load，所以理想重叠下至少需要：

\[
t_L=\max\left(
\frac{Sd_L}{B_{\mathrm{L2}}},
\frac{d_L}{C_{\mathrm{TMA,SM}}}
\right).
\]

每一项都是 byte 除以 byte/s，结果为 s。Thor 上的 1024 B/cycle L2 read
model peak 是整 GPU 共享值，不能乘以 SM 数；反过来，也不能用 20-SM TMA
aggregate 除以 20 来声称已经隔离了每 SM 出口，因为该 aggregate 测量本身
可能先被共享 L2 限制。这里仍不能直接把 273 GB/s LPDDR5X 峰值代入 L2-hit
项，因为 TMA 请求可能命中 L2，且仓库 benchmark 在 warmup 后重复使用相同
A/B；逻辑 TMA 字节与 DRAM 实际字节不是同一个量。冷入口还必须额外与共享
LPDDR total/read 约束取最大值。通用、可执行的资源公式以性能上限文档为准。

### 5.3 Compute stage 时间

令 \(P_C\) 为 20 个 SM 满载，并且 MMA shape、每 4 条 MMA 的 stage 边界、TMEM 占用和 resident CTA 压力都与 `tc3` 匹配时的计算阶段吞吐，则：

\[
t_C=\frac{Sf_C}{P_C}.
\]

分子是 FLOP，分母是 FLOP/s，结果也是 s。`P_C` 不应直接使用官方 258.5 TFLOP/s 峰值，因为峰值微基准可以连续发射更多独立 MMA，而 `tc3` 每个 \(B_K=64\) stage 只有 4 条 MMA，随后必须等待该 stage 完成才能安全复用 SMEM。

仓库中与这一路径最接近的 BF16 `M128N128 K4` mainloop 微基准实测为 113.015 TFLOP/s。它与 `tc3` 都使用 `tcgen05.mma.kind::f16` / `UTCHMMA` 指令族、相同 M/N/K atom 和 K4 等待节奏，但仍有两个重要差别：输入格式 descriptor 不同；该微基准每 CTA 分配 512 个 TMEM columns，而 `tc3` 只分配 128 个，因而 resident CTA 压力不等价。本文只能把 113.015 TFLOP/s 当作**单 CTA/SM 条件下的 BF16 代理值**。正式闭环必须生成同构 FP16、128-column TMEM、相同 SMEM 和 grid 压力的 case，不能把“同指令族”直接写成“吞吐必然相同”。数据见 [`mma_mainloop_sweep_results.csv`](../../microbench/mma_with_cp/plots/mma_mainloop_sweep_results.csv)，微基准资源配置见 [`tcgen05_ss_mma_mainloop_k4_m128n128_bf16_benchmark.cu`](../../microbench/mma_with_cp/benchmark_src/tcgen05_ss_mma_mainloop_k4_m128n128_bf16_benchmark.cu)。

### 5.4 双缓冲 mainloop 时间是怎样推出来的

若完全串行，\(N_K\) 个 stage 每次都先 load 再 compute：

\[
t_{\text{main,serial}}=N_K(t_L+t_C).
\]

但 `tc3` 有两个 SMEM buffer。时间线是：

```text
先装入 L0
随后 (C0 与 L1 重叠)
随后 (C1 与 L2 重叠)
...
最后完成 C(NK-1)
```

重叠区间只有在 load 和 compute 都结束后才能前进，所以单个稳态间隔是：

\[
t_{\text{steady}}=\max(t_L,t_C).
\]

共有 \(N_K-1\) 个这样的相邻间隔，再加不能隐藏的第一次 load 和最后一次 compute：

\[
t_{\text{main}}
=t_L+(N_K-1)\max(t_L,t_C)+t_C.
\]

该式也通过两个边界情况：

- 当 \(N_K=1\) 时，没有可重叠的相邻 stage，得到 \(t_L+t_C\)；
- 当 \(N_K\) 很大时，首尾成本被摊薄，单 stage 平均时间趋近 \(\max(t_L,t_C)\)。

因此这里的 `max` 来自双缓冲依赖关系，而不是为了让预测值更接近实测。

### 5.5 Epilogue 时间

令 \(B_E\) 为匹配 `tc3` 写回路径的满载复合 epilogue 速率，则一个 service wave 的 epilogue 时间是：

\[
t_E=\frac{Sd_E}{B_E}.
\]

`tc3` 必须在 MMA 完成后才从 TMEM 读取 accumulator，所以第一版模型使用 \(t_{\text{main}}+t_E\)，而不是 \(\max(t_{\text{main}},t_E)\)。不同 resident CTA 之间可能发生 mainloop/epilogue 交错；这部分只能由同占用配置的微基准和多尺寸验证确认，不能凭空假设完全重叠。

### 5.6 固定成本与最终公式

\(T_{\text{fixed}}\) 收纳每个 service wave 中不随 \(K\)-stage 数据量线性增长的成本，例如 TMEM alloc/dealloc、barrier 初始化、同步和地址控制。kernel launch 是整个 grid 只发生一次，所以单独记为 \(T_{\text{launch}}\)。

于是得到本文的最终时间模型：

\[
T_{\text{pred}} =
T_{\text{launch}}
+N_{\text{wave}}
\left[
T_{\text{fixed}}
+t_L
+(N_K-1)\max(t_L,t_C)
+t_C
+t_E
\right].
\]

再把三个阶段时间展开，可得到直接用于计算的形式：

\[
\begin{aligned}
T_{\text{pred}}
=T_{\text{launch}}
+N_{\text{wave}}
\Bigg[
&T_{\text{fixed}}
+\frac{Sd_L}{B_L} \\
&+(N_K-1)
\max\left(
\frac{Sd_L}{B_L},
\frac{Sf_C}{P_C}
\right) \\
&+\frac{Sf_C}{P_C}
+\frac{Sd_E}{B_E}
\Bigg].
\end{aligned}
\]

这一版模型只有四个需要测量的性能参数：\(B_L,P_C,B_E,T_{\text{fixed}}\)，外加一次 launch 时间。每个参数都对应一条可隔离的数据路径，没有“冲突惩罚系数”“架构代数”之类无法独立辨识的自由参数。

## 6. 一个很有用的诊断量：TMA 与 MMA 的平衡点

当 \(t_L=t_C\) 时，load 与 compute 恰好平衡。把两个阶段公式相等：

\[
\frac{Sd_L}{B_{L,\text{balance}}} =
\frac{Sf_C}{P_C}.
\]

两边的 \(S\) 消掉，得到：

\[
B_{L,\text{balance}}
=P_C\frac{d_L}{f_C}.
\]

也可以先定义 K-stage 的逻辑算术强度：

\[
I_{\text{stage}}=\frac{f_C}{d_L}.
\]

那么平衡带宽就是：

\[
B_{L,\text{balance}}=\frac{P_C}{I_{\text{stage}}}.
\]

当前 tile 的逻辑算术强度为：

\[
I_{\text{stage}}
=\frac{2097152}{32768}
=64\ \text{FLOP/B}.
\]

代入实测 \(P_C=113.015\ \text{TFLOP/s}\)：

\[
B_{L,\text{balance}}
=\frac{113.015}{64}
\approx1.766\ \text{TB/s}.
\]

这个 1.766 TB/s 是 TMA **逻辑有效带宽**的平衡点，不是说 Thor 的 LPDDR 带宽有 1.766 TB/s。若同构 TMA 微基准测得 \(B_L<1.766\ \text{TB/s}\)，模型判定稳态由 load 主导；反之才由 MMA 主导。因为 A/B 可从 L2 重用，逻辑有效带宽高于 273 GB/s 并不矛盾。

## 7. \(M=N=K=1024\) 的完整算例

### 7.1 离散计数

当前 tile 为 \(128\times128\times64\)：

\[
\begin{aligned}
N_{\text{CTA}}
&=\frac{1024}{128}\times\frac{1024}{128}=64, \\
N_K
&=\frac{1024}{64}=16, \\
N_{\text{wave}}
&=\left\lceil\frac{64}{20}\right\rceil=4.
\end{aligned}
\]

最后一个 service wave 只有 4 个 CTA。这正是只使用 \(2MNK/P_C\) 会漏掉的尾波量化。

### 7.2 计算阶段的代理标定结果

暂时采用 BF16 同指令族代理值时，一个满 wave 的单 K-stage 计算时间是：

\[
\begin{aligned}
t_C
&=\frac{20\times2097152}{113.015\times10^{12}}
&\approx0.371\ \mu\text{s}.
\end{aligned}
\]

如果暂时假设 `tc3` 的有效 compute service rate 就等于这个单 CTA/SM 的 BF16 代理值，那么 4 个 wave、每个 16 个 K-stage 对应的计算时间估计为：

\[
T_{C,\text{one-CTA proxy}}
=4\times16\times0.371
\approx23.75\ \mu\text{s}.
\]

这里暂用 113.015 TFLOP/s，是因为它至少匹配 `tc3` 每 4 条 MMA 等待一次的 stage 边界。23.75 µs 只是“FP16 与该 BF16 case 具有相同 stage completion throughput，并且额外 resident CTA 没有改变有效吞吐”这两个显式假设下的代理估计，**不是严格下界**。额外 CTA 可能隐藏 stage wait，使实际 \(P_C\) 高于 113.015 TFLOP/s。

若使用仓库纯吞吐微基准的 BF16 258.030 TFLOP/s，考虑同样的最后一波后得到 10.40 µs。把该实测峰值视为 FP16/BF16 `kind::f16` 路径的设备上限时，10.40 µs 才是当前证据下更保守的 compute-only 下界；它只回答“硬件最快可能多快”，也不能直接预测 `tc3`。

### 7.3 与现有整核结果对照

仓库记录 `tc3` 在 \(N=1024\) 时为 37,410 GFLOP/s。由 GEMM 总 FLOP 数反推时间：

\[
\begin{aligned}
T_{\text{meas}}
&=\frac{2\times1024^3}{37410\times10^9} \\
&\approx57.404\ \mu\text{s}.
\end{aligned}
\]

因此，单 CTA/SM 的 K4 代理计算时间占实测时间的比例为：

\[
\frac{23.75}{57.404}\approx41.4\%.
\]

正确解释是：如果 113.015 TFLOP/s 代理值能代表 `tc3` 的 compute service rate，那么计算部分约对应 23.75 µs，整核还存在约 33.65 µs 的差额。因为 `tc3` 的 resident CTA 可能隐藏 K4 微基准中的等待，这个差额也不能被视为非计算阶段的严格时间。

**不能**把这 33.65 µs 直接命名为“内存时间”。TMA 与 MMA 部分重叠，epilogue、同步、TMEM 管理和最后一波利用率也都在总时间中。只有补齐 \(B_L,B_E,T_{\text{fixed}}\) 后，模型才能把差值分解到具体阶段。

### 7.4 为什么不能直接套 273 GB/s

该问题的实际逻辑 TMA 输入量为：

\[
D_{L,\text{logical}}
=N_{\text{CTA}}N_Kd_L
=64\times16\times32768
=32\ \text{MiB}.
\]

但不同输出 CTA 会重复请求相同 A 行块或 B 列块；warmup 后这些输入还可能位于 32 MiB L2 中。所以 32 MiB 是 TMA 看到的逻辑请求量，不等于 LPDDR 传输量。

若讨论一个冷启动、每个 A/B 元素至少从 DRAM 读取一次、C 写一次的理论下界，最少字节数才是：

\[
D_{\text{cold,min}}
=MKs_A+KNs_B+MNs_C
=8\ \text{MiB}.
\]

用 273 GB/s 得到约 30.73 µs，但仓库的计时策略先 warmup，再对相同矩阵重复 launch。这个冷启动下界与当前 hot-cache benchmark 不是同一实验口径，不能拿来充当 \(t_L\)。模型应直接测同口径的 \(B_L\)，而不是猜 L2 hit rate。

## 8. 怎样把模型真正校准闭环

下面五个实验都应使用相同功耗模式、SM 时钟、矩阵布局、SW128 descriptor、CTA threads、grid 规模、warmup 和计时方法。

| 参数 | 所需微基准 | 计算方法 | 为什么现有数据不能替代 |
| --- | --- | --- | --- |
| \(P_C\) | FP16 M128N128 K4 issue-and-wait；128-column TMEM、64 KiB SMEM、grid 压力匹配 `tc3` | \(P_C=20f_C/t_C\) | 现有 512-column TMEM 的 BF16 113.015 TFLOP/s 只作代理 |
| \(B_L\) | 每 CTA 同时发 A/B 两个 16 KiB TMA load，2-stage，至少 20 CTA | \(B_L=20d_L/t_L\) | 现有 TMA 测试是单 CTA、4 KiB tile，几何和并发度不匹配 |
| \(B_E\) | 预置 TMEM accumulator，仅执行与 `tc3` 相同的 `tcgen05.ld` 和 FP32 store | \(B_E=20d_E/t_E\) | 普通 memcpy 带宽不包含 TMEM read 和 lane ownership |
| \(T_{\text{fixed}}\) | 保留 TMEM/barrier/控制结构，去掉按 K 重复的 load/MMA 和输出搬运 | 直接测每个满 wave 的时间 | 不能用 kernel launch 代替设备端控制成本 |
| \(T_{\text{launch}}\) | 同 stream 的空 kernel，多次 launch | event 总时间除以次数 | 只发生一次，不应乘 \(N_{\text{wave}}\) |

补齐参数后，对每个尺寸按以下固定顺序计算：

1. 用 \(M,N,K\) 和 tile 得到 \(N_{\text{CTA}},N_K,N_{\text{wave}}\)；
2. 用数据类型和 tile 得到 \(d_L,f_C,d_E\)；
3. 用同口径实测速率得到 \(t_L,t_C,t_E\)；
4. 代入总时间公式；
5. 用 \(2MNK/T_{\text{pred}}\) 得到 TFLOP/s；
6. 与 event 实测时间比较，而不是只比较吞吐图形是否相似。

## 9. 验证标准与何时允许扩展模型

### 9.1 最小验证矩阵

建议至少测以下规则尺寸：

| 目的 | 尺寸示例 |
| --- | --- |
| 小 grid 与尾波 | \(512^3\) |
| 当前锚点 | \(1024^3\) |
| 中等规模 | \(2048^3\) |
| 长 K、摊薄首尾 | \(1024\times1024\times4096\) |
| 改变输出 CTA 数但保持 K | \(2048\times1024\times1024\) |

时间相对误差定义为：

\[
\varepsilon_T =
\frac{\left|T_{\text{pred}}-T_{\text{meas}}\right|}
{T_{\text{meas}}}.
\]

第一阶段目标可设为：校准点误差不超过 5%，同一 tile 与 cache 口径下的未参与校准尺寸误差不超过 10%。超过 10% 时先查看误差是否随 \(K\)、CTA wave 或输出字节数呈系统趋势，再决定增加哪一个可测阶段。

### 9.2 只有出现对应证据才扩展

- 误差主要随 \(N_K\) 增长：检查 \(B_L\)、\(P_C\) 是否随 K 或 cache 状态变化；
- 只在 \(N_{\text{CTA}}\bmod20\neq0\) 时变大：改进最后一波的 partial-wave 标定；
- 误差主要随 \(MN\) 增长：检查 epilogue 的 \(B_E\)；
- 小尺寸都有近似固定偏差：检查 \(T_{\text{launch}}\) 与 \(T_{\text{fixed}}\)；
- NCU 显示 TMA/MMA 不能达到各自微基准速率：再引入一个有对照实验支持的资源竞争项。

不应一开始就加入任意的 bank-conflict、cache miss、occupancy 或“架构惩罚”系数。若一个新项不能由独立微基准或 profiler counter 标定，它只会提高拟合自由度，不会提高解释力。

## 10. 这个模型能回答什么，不能回答什么

### 能回答

- 当前 tile 的 load、compute、epilogue 各自有多少工作量；
- 为什么 2-stage pipeline 的稳态时间取 \(\max(t_L,t_C)\)；
- 某次测试是 load-bound 还是 compute-bound；
- CTA 尾波为什么使小 grid 低于简单的 \(2MNK/P_C\)；
- 优化 tile、K-stage 或 epilogue 后，应该改动哪个参数和哪一项公式；
- 模型缺少哪一个微基准，因而当前能否做可信的数值预测。

### 暂时不能回答

- 任意 CUTLASS schedule、2-SM cluster、persistent/CLC kernel 的性能；
- 非整 tile 边界、split-K、batched GEMM；
- cache 冷热状态变化下的 DRAM/L2 精确流量；
- 不同功耗模式、DVFS 或并发 workload 下的速率变化；
- bank conflict 对 `tcgen05.mma` operand consumption 的独立惩罚。

这些问题不是永远不能建模，而是不能在没有新测量的情况下沿用本文参数。`tc4b/tc4c` 的 `cta_group::2` 应把 20 个 SM 改成 10 个 SM pair，并重新测 \(P_C,B_L,B_E\)；persistent kernel 则需要把 service-wave 调度换成持久化 work-tile 调度。

## 11. 当前可复现的证据索引

- `tc3` 实现、tile、2-stage、TMA/MMA/epilogue 顺序：[`tc3_pipeline.cuh`](../../GEMMsm110/include/backends/tc3_pipeline.cuh)
- `tcgen05`、TMA、TMEM PTX wrapper：[`sm110_ptx_helpers.cuh`](../../GEMMsm110/include/sm110_ptx_helpers.cuh)
- `tc3` 的 \(N=1024\) 阶段性结果 37,410 GFLOP/s：[`GEMMsm110/README.md`](../../GEMMsm110/README.md)
- BF16 M128N128 K4 mainloop 的 113.015 TFLOP/s：[`mma_mainloop_sweep_results.csv`](../../microbench/mma_with_cp/plots/mma_mainloop_sweep_results.csv)
- MMA 微基准的计时口径、forced-wait 与 mainloop 区别：[`分析报告.txt`](<../../microbench/mma_with_cp/分析报告.txt>)
- BF16 纯 MMA 峰值实测 258.030 TFLOP/s：[`benchmark_results.csv`](../../microbench/mma_compute_only/plots/benchmark_results.csv)
- Thor 设备资源与数据流记录：[`sm110_gemm_bank_conflict_research.md`](../cutlass/sm110_gemm_bank_conflict_research.md)
- 官方 `tcgen05` 数据流说明：[CUTLASS tcgen05 MMA Programming Guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/tcgen05_programming.html)
- 官方异步指令语义：[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)

## 12. 最终判断

这份模型刻意停在“五个可测参数”处，而没有用 \(N=1024\) 的整核时间反向拟合一组看似完整的 TMA、epilogue 和同步参数。原因是：一个总时间点无法唯一分解多个会重叠的阶段；强行分解只会得到数学上能拟合、物理上不可辨识的答案。

当前已经可以严谨地得到三点：

1. 10.40 µs 是依据现有纯 MMA 峰值得到的 compute-only 下界；23.75 µs 是依据单 CTA/SM BF16 K4 case 得到的代理估计，不能混称为下界；
2. 实测整核约 57.404 µs，说明 load、epilogue、控制和跨 CTA 调度仍有显著可见成本；
3. 要把“解释模型”升级成“预测模型”，下一步不是增加公式，而是补齐与 `tc3` 同构的满载 TMA 和 epilogue 微基准。

这正是阶段化解析模型应有的边界：每个公式都能回到数据量、指令依赖或调度事实；每个未知参数都有明确实验；数据不够时明确说不能唯一判断，而不是用额外系数掩盖证据缺口。
