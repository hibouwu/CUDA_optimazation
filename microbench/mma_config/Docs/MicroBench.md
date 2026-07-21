# tcgen05 MMA 延迟、吞吐与 operand 供数模型

本文使用一个三阶段抽象模型分析 `tcgen05.mma` 的延迟、稳态吞吐和流水重叠，并据此设计 `microbench/mma_config` 中的实验。

这里的 stage、work tile 和相关时间项都是性能建模概念，不代表 NVIDIA 已公开确认了对应的物理流水级、端口或内部切分方式。实验首先测量可观察量，再判断数据是否支持模型。

## 术语与边界

### MMA instruction

本文中的 MMA 指一条 `tcgen05.mma` 描述的矩阵乘累加操作：

$$
D \leftarrow A B + D
$$

其中 A 可能来自 SMEM 或 TMEM，B 通常来自 SMEM，D 是位于 TMEM 中的 accumulator。软件发射一条 MMA，硬件内部如何拆分和调度并未完全公开。

### Operand

Operand 包括 MMA 所需的 A/B 数据以及描述 shape、dtype、layout 和地址的 descriptor/config。本文的 operand feed 主要指硬件通过 async proxy 从 SMEM 获取 A/B，并送入后续矩阵计算数据路径。

使用 collector 的 `fill/use/lastuse` 时，后续 MMA 可能复用 collector 中的 operand。因此：

```text
logical operand bytes/MMA != 必然发生的 SMEM read bytes/MMA
```

只有 collector discard 且地址复用效应受控的实验，才适合用逻辑 operand 字节数估计有效 SMEM ingress。

### Work tile

Work tile 是为了表达 MMA 内部流水而引入的抽象粒度。一条 MMA 可以建模为：

$$
W_0,W_1,\ldots,W_{q-1}
$$

每个 work tile 包含部分 operand 读取、Tensor Core 计算和 TMEM accumulator 更新。这里的 $q$ 不是 PTX 暴露的参数，也不能仅凭一条吞吐曲线唯一反演。

### Latency 与 initiation interval

Latency $L_{\mathrm{mma}}$ 表示 MMA 从发射到其结果满足完成条件的总时间。Initiation interval $II_{\mathrm{mma}}$ 表示流水进入稳态后，相邻 MMA 开始处理的最小有效间隔。

必须区分：

$$
L_{\mathrm{mma}} \ne II_{\mathrm{mma}}
$$

一条 MMA 的完成延迟可以很长，但硬件仍可能以较小的 $II_{\mathrm{mma}}$ 接收后续工作。

### Completion

Completion 是同步语义下的完成条件，而不是必须独立存在的第四个物理流水级。它表示此前被跟踪的 MMA 已完成必要的 TMEM 更新，后续依赖操作可以按同步规则继续。

`tcgen05.commit` 让 mbarrier 跟踪执行线程发出的所有先前 async-tcgen05 操作。因此软件通常观察到的是某个累计 instruction prefix 的 completion，而不是内部每个 work tile 的 retire 时间。

## 三阶段 MMA 抽象模型

将一条 `tcgen05.mma` 抽象为：

```text
Issue / Setup
      ↓
Operand Feed + MMA Compute
      ↓
TMEM Accumulation / Retire
      ● completion point
```

三个逻辑阶段之间可以重叠，不能把它们的持续时间无条件串行相加。

### Stage 1：Issue / Setup

第一阶段表示把 MMA 请求送入执行流水线，可抽象地包含：

- warp 发射、指令解码和分派；
- descriptor/config 解析；
- operand 地址或访问模式初始化；
- 内部执行状态建立；
- 第一批 operand 请求准备。

将其首部成本记为：

$$
T_{\mathrm{issue}}
$$

这一项可能更接近固定开销，因此对小 shape 的占比更明显，并容易被大 shape 的稳态工作摊薄。但不能直接断言它对应独立硬件单元或固定 1 cycle。

### Stage 2：Operand Feed + MMA Compute

第二阶段是主要执行阶段。对于抽象 work tile，可以画成：

```text
time →

SMEM feed:   W0-read | W1-read | W2-read | W3-read
Tensor Core:         W0-MMA  | W1-MMA  | W2-MMA  | W3-MMA
```

模型假设不同 work tile 的 operand feed 与 Tensor Core 计算可以流水重叠。定义：

$$
II_{\mathrm{operand}}
$$

为供应一个抽象 work tile operand 的稳态间隔，$II_{\mathrm{TC}}$ 为 Tensor Core 处理一个 work tile 的稳态间隔。在简单 roofline 近似下：

$$
II_{\mathrm{body}}
\approx
\max\left(II_{\mathrm{operand}},II_{\mathrm{TC}}\right)
$$

若：

$$
II_{\mathrm{operand}} \le II_{\mathrm{TC}}
$$

则数据支持 compute-limited 解释；若：

$$
II_{\mathrm{operand}} > II_{\mathrm{TC}}
$$

则数据支持 operand-feed-limited 解释。

这个 `max` 模型是假设而不是测量事实。descriptor、collector、TMEM update、内部仲裁或依赖也可能改变观察到的间隔。

固定 M、K、dtype、operand 来源和 layout 时，改变 N 通常会同时改变 B 数据量、计算量和可能的内部 work tile 数，因此可以尝试：

$$
T_{\mathrm{body}}(N)
\approx q(N) II_{\mathrm{body}}
$$

但“body 只与 N 有关”不构成通用硬件规律。

### Stage 3：TMEM Accumulation / Retire

第三阶段表示部分结果更新到 TMEM accumulator：

$$
D_{\mathrm{TMEM}}
\leftarrow
D_{\mathrm{TMEM}} + \Delta D
$$

TMEM accumulation 可能与 Stage 2 的后续 work tile 计算重叠。最后一个 work tile 更新、内部流水排空以及完成状态传播形成尾部成本：

$$
T_{\mathrm{retire}}
$$

completion point 位于这个逻辑阶段结束处，但不能据此断言存在一个固定长度的物理 retire stage。

## 单条 MMA 的逻辑时间模型

若一条 MMA 被抽象为 $q$ 个 work tile，则：

$$
T_{\mathrm{mma}}
\approx
T_{\mathrm{issue}}
+T_{\mathrm{first}}
+(q-1)II_{\mathrm{body}}
+T_{\mathrm{retire}}
$$

令：

$$
S=T_{\mathrm{issue}}+T_{\mathrm{first}},
\qquad
D=T_{\mathrm{retire}}
$$

可简写为 startup、steady-state、drain 模型：

$$
T_{\mathrm{mma}}
\approx
S+(q-1)\max\left(II_{\mathrm{operand}},II_{\mathrm{TC}}\right)+D
$$

这里的 $S$、$q$ 和 $D$ 通常不能从单个 forced-wait 测量中分别识别；它们用于组织假设，而不是预设的硬件常数。

## 多条 MMA 的流水模型

对于连续发射的 Q 条 MMA，一般不能使用 `Q+2`，除非已经证明每个阶段及启动间隔都恰好为 1 cycle。忽略外部计时开销时，更一般的模型是：

$$
T(Q)
\approx
L_{\mathrm{first}}+(Q-1)II_{\mathrm{mma}}
$$

```text
mma0: [Issue] [ Operand Feed + MMA ] [TMEM / Complete]
mma1:         [Issue] [ Operand Feed + MMA ] [TMEM / Complete]
mma2:                 [Issue] [ Operand Feed + MMA ] [TMEM / Complete]
```

实际的 $II_{\mathrm{mma}}$ 可以建模为：

$$
II_{\mathrm{mma}}
\approx
\max\left(
II_{\mathrm{issue}},
II_{\mathrm{body}},
II_{\mathrm{TMEM}},
II_{\mathrm{dependency}}
\right)
$$

如果 timed region 包含一次最终 `tcgen05.commit` 和 mbarrier wait，实际测量值更接近：

$$
T_{\mathrm{measured}}(Q)
\approx
T_{\mathrm{fixed}}
+L_{\mathrm{first}}
+(Q-1)II_{\mathrm{mma}}
$$

因此实验采用回归：

$$
T_{\mathrm{measured}}(Q)=\alpha+\beta Q
$$

其中 $\beta$ 近似长 batch 的边际 cycles/MMA；$\alpha$ 混合了 loop、commit、最终 drain、wait 和首条 MMA 延迟，不能直接命名为纯 startup 或纯同步开销。

per-MMA forced `commit/wait` 则会把 completion latency、同步开销和等待空泡重复加入每次迭代，只适合作为隔离完成延迟的诊断，不代表稳态 MMA 吞吐。

## Tensor Core operand 供数、吞吐和 SMEM 带宽

以 BF16、`M=128, K=16` 为例，若 A、B 都来自 SMEM：

$$ \text{SMEM bytes/MMA} = 2(MK+NK) $$

其中每个 BF16 是 2 bytes。

若采用当前 Thor stage model 中的 BF16 Tensor Core 峰值假设：

$$
P_{\mathrm{TC}}=8192\ \text{FLOP/cycle/SM}
$$

则理想计算时间为：

$$ T_{\mathrm{TC}} = \frac{2 MNK}{8192} = \frac N2 $$

得到 MMA 为维持 Tensor Core 峰值所需的平均 SMEM 读取速率：

$$
BW_{\mathrm{required}} = \frac{2K(M+N)}{N/2}
= 64\frac{128+N}{N}\quad\text{bytes/cycle}
$$

| BF16 MMA shape | A+B 数据量 |     理想计算时间 | 所需 SMEM 平均带宽 |
| -------------- | ------: | ---------: | -----------: |
| `m128n64k16`   |    6 KB |  32 cycles |  192 B/cycle |
| `m128n128k16`  |    8 KB |  64 cycles |  128 B/cycle |
| `m128n256k16`  |   12 KB | 128 cycles |   96 B/cycle |

这揭示了一个重要现象：

`N` 越小，每条 MMA 的计算时间下降得比 A 操作数的数据量更快，因此单位 cycle 的 SMEM 供数压力反而更高。

这与你之前的单 warp 结果方向一致：

```text
N64   → 约 42% peak
N128  → 约 81% peak
N256  → 接近 100% peak
```

因此，`N64` 和 `N128` 性能较低，确实可能包含以下原因：

```text
SMEM → Tensor Core operand path 供数不足
operand collector / descriptor processing 吞吐不足
同一线程连续发射 MMA 的最小间隔
Tensor Core 小 shape 的固定启动开销
```

但仅凭吞吐曲线，不能断言就是 SMEM 带宽。

## 2026-07-20 静态校准结果

旧版 `mma_config` 使用 runtime-dispatch kernel，timed loop 内包含 descriptor、operand slot、SMEM 地址、D ring 地址、collector/protocol 分派、wait 和 CTA 同步，因此不再用于推断硬件路径。新的静态校准为每个 case 使用独立的 compile-time binary，并将 descriptor、SMEM 地址和 TMEM setup 移到 timed region 之外。

BF16 full-grid、collector discard、`wait_hint=0`、same-D、`input_d=0` 的静态 Q sweep 如下：

| Shape | Q1 | Q2 | Q4 | Q8 | Q16 | Q32 | Q64 | fitted beta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `m128n128k16` | 450.270 | 247.145 | 145.581 | 102.079 | 86.768 | 75.333 | 68.937 | 63.747 |
| `m128n256k16` | 450.262 | 308.800 | 212.660 | 181.482 | 154.818 | 141.364 | 135.092 | 129.381 |

`m128n128k16` Q4 的 `145.581 cycles/MMA` 与已有可信 BF16 K4 mainloop 的 `146.132 cycles/MMA` 同量级且几乎一致，因此静态 harness 可以作为后续分析的主校准口径。

使用 fitted beta 计算的 logical operand service rate 为：

| Shape | Logical operand bytes/MMA | fitted beta | logical bytes/cycle |
| --- | ---: | ---: | ---: |
| `m128n128k16` | 8192 B | 63.747 | 128.5 |
| `m128n256k16` | 12288 B | 129.381 | 95.0 |

这些值只表示 software-visible logical service rate，不能命名为物理 SMEM → Tensor Core 端口宽度，也不能据此推出每 cycle 读取多少 SMEM bank 或写入多少 TMEM bank。

关键控制项为：

| Control | BF16 N128 full-grid observation |
| --- | ---: |
| empty/control loop | ~264 cycles/iteration |
| commit + already-completed wait | ~258 cycles/iteration |
| forced single-MMA wait, hint=0 | ~450 cycles/MMA |
| forced single-MMA wait, hint=32 或 `0x989680` | ~431 cycles/MMA |
| CTA-wide `__syncthreads()` | 20.828 cycles/sync |

因此，`Q=1` 是 forced-completion diagnostic，不应通过减去 empty commit/wait 直接解释为纯 MMA latency；长 batch 回归的 beta 才是当前静态 harness 下更稳健的边际成本。

## 模型参数与实验的对应关系

三阶段模型中的参数不能全部从单条曲线直接分离，需要由 `ExperimentPlan.md` 中的不同控制实验共同约束：

| 模型项 | 主要实验 | 可观测量 |
| --- | --- | --- |
| collector reuse | `01_collector_protocol` | discard 与 fill/use/lastuse 的差值 |
| $L_{\mathrm{first}}$ 与同步尾部 | `02_latency_throughput` | Q=1 forced-completion 与回归截距 |
| $II_{\mathrm{mma}}$ | `02_latency_throughput` | 长 batch 回归斜率 $\beta$ |
| $II_{\mathrm{operand}}$ 是否主导 | `03_effective_smem_ingress` | collector-discard 下 bytes/cycle roofline |
| layout/address 对 operand feed 的影响 | `04_smem_layout_address` | 合法 descriptor pair 的相对退化 |
| LSU shared-path 竞争 | `05_ldshared_contention` | 控制组下的双路径归一化吞吐 |
| $II_{\mathrm{TMEM}}$ 与 $II_{\mathrm{dependency}}$ | `06_tmem_dependency` | D alias、`input_d` 和 reuse-distance 曲线 |

只有多个实验同时支持时，才把性能限制归因于某个逻辑阶段。若参数不可识别，应保留为组合项，不把回归截距或吞吐拐点强行解释成单一硬件常数。
