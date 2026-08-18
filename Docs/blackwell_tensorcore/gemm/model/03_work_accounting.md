# 03 工作量与数据流计账

本章只回答“任何实现或某个 schedule 必须做多少工作”，不在这里代入服务率。分子边界错误时，即使带宽测得完全准确，最终时间也没有物理意义。

## 1. Useful 与 issued compute

定义用户要求的经典 GEMM 工作量：

\[
W_{\mathrm{use}}=2MNK.
\]

定义 CTA tile 尺寸为 \(B_M,B_N,B_K\)，定义 tile 数：

\[
N_M=\left\lceil\frac{M}{B_M}\right\rceil,
\quad
N_N=\left\lceil\frac{N}{B_N}\right\rceil,
\quad
N_K=\left\lceil\frac{K}{B_K}\right\rceil.
\]

`tail_policy=pad` 时，定义 issued dimensions：

\[
M'=N_MB_M,
\quad N'=N_NB_N,
\quad K'=N_KB_K,
\]

并定义：

\[
W_{\mathrm{issue}}=2M'N'K'+W_{\mathrm{reduce}}.
\]

当前 `split_k=1`，所以 \(W_{\mathrm{reduce}}=0\)。定义 shape efficiency：

\[
\eta_{\mathrm{shape}}=\frac{W_{\mathrm{use}}}{W_{\mathrm{issue}}}.
\]

非整除 shape 的 `exact` schedule 只有在专用 tail kernel 的指令粒度和路径已进入 manifest 时才合法；当前模型直接拒绝未经证明的 exact tail。

## 2. 最小输入与输出 I/O

定义逻辑输入 value 最小字节数：

\[
Q_{\mathrm{in,val}}^{\mathrm{LB}}
=(MK+KN)s_{\mathrm{in}}.
\]

对 block-scaled precision，定义一个 scale 覆盖 \(b_s\) 个 K 元素，每个 scale 占 \(s_s\) B。scale block 不跨 K vector：

\[
Q_{\mathrm{in,scale}}^{\mathrm{LB}}
=
\left(
M\left\lceil\frac{K}{b_s}\right\rceil
+N\left\lceil\frac{K}{b_s}\right\rceil
\right)s_s.
\]

定义 C read 最小字节数：

\[
Q_C^{\mathrm{LB}}=
\begin{cases}
0,&\beta=0,\\
MN s_{\mathrm{acc}},&\beta\ne0.
\end{cases}
\]

当前 accumulator output 的最小写回为：

\[
Q_D^{\mathrm{LB}}=MN s_{\mathrm{out}}.
\]

这些量是所有经典实现不可避免的逻辑边界，主要进入严格层。

## 3. Value transport 与 scale transport

定义 `input_transport_layout` 的物理 value bytes：

- `logical_packed`：使用 precision 的逻辑 byte width；
- `byte_padded`：每个值使用 1 B container；
- `b6x16_p32` / `b4x16_p64`：每 16 个值使用 16 B 物理 transport atom。

对 block scale，定义外维为 \(X\)、K tile 为 \(B_K\) 的 scale transport：

\[
S(X,B_K,b_s,s_s)
=
\left\lceil\frac{X}{128}\right\rceil128
\left\lceil
\frac{\left\lceil B_K/b_s\right\rceil}{4}
\right\rceil4s_s.
\]

128-vector 和 4-scale-group padding 来自当前 Blackwell scale-factor storage atom。逻辑 scale bytes 与 TMA/S2T 物理 transport bytes 必须分别记录。

## 4. 单 tile 的独立 TMA request

定义：

\[
q_{A,v}=B_MB_Ks_{\mathrm{transport}},
\qquad
q_{B,v}=B_KB_Ns_{\mathrm{transport}},
\]

\[
q_{A,s}=S(B_M,B_K,b_s,s_s),
\qquad
q_{B,s}=S(B_N,B_K,b_s,s_s).
\]

四个量对应 A value、B value、A scale、B scale 的独立 request payload。没有 scale 的精度令 \(q_{A,s}=q_{B,s}=0\)。

不得把 \(q_{A,v}+q_{A,s}\) 冒充一条 TMA request；模型保留：

- `tma_a_value_bytes`；
- `tma_b_value_bytes`；
- `tma_a_scale_bytes`；
- `tma_b_scale_bytes`；

四项独立计账。`tma_a_input_bytes` / `tma_b_input_bytes` 只是矩阵侧汇总。

例如当前 block-scaled M128N256K64 tile 的独立 payload 包含：

- A value：4096 B；
- B value：8192 B；
- A scale：512 B；
- B scale：1024 B。

已有 4/8 KiB value surface 不能替代 512 B/1 KiB scale request。

## 5. Schedule-issued TMA bytes

定义 output-tile task 数：

\[
N_{\mathrm{task}}=N_MN_N.
\]

定义 tile visit 数：

\[
N_{\mathrm{visit}}=N_MN_NN_K.
\]

于是：

\[
Q_{A,v}^{\mathrm{TMA}}=N_{\mathrm{visit}}q_{A,v},
\quad
Q_{B,v}^{\mathrm{TMA}}=N_{\mathrm{visit}}q_{B,v},
\]

\[
Q_{A,s}^{\mathrm{TMA}}=N_{\mathrm{visit}}q_{A,s},
\quad
Q_{B,s}^{\mathrm{TMA}}=N_{\mathrm{visit}}q_{B,s}.
\]

定义总 issued TMA payload：

\[
Q_{\mathrm{TMA,issued}}
=Q_{A,v}^{\mathrm{TMA}}
+Q_{B,v}^{\mathrm{TMA}}
+Q_{A,s}^{\mathrm{TMA}}
+Q_{B,s}^{\mathrm{TMA}}.
\]

它是 schedule 请求给 TMA/L2 路径的 payload，不自动等于 DRAM physical bytes。

## 6. Unique cold-entry bytes

定义 \(Q_{\mathrm{TMA,unique}}\) 为在理想跨 output-tile L2 reuse 下，补齐后的 A/B value 与 scale 物理输入并集。它对 A 和 B 各计一次，不乘 \(N_MN_N\)。

因此一般有：

\[
Q_{\mathrm{TMA,unique}}
\le
Q_{\mathrm{TMA,issued}}.
\]

左侧适合描述理想 cold-entry unique input；右侧描述具体 schedule 对共享 L2/TMA 路径发出的 request payload。二者不能互换。

## 7. TMEM 与 reduction

定义 accumulator readback issued bytes：

\[
Q_{\mathrm{TMEM,readback}}=
\begin{cases}
M'N's_{\mathrm{acc}},&\text{pad},\\
MNs_{\mathrm{acc}},&\text{合法 exact}.
\end{cases}
\]

定义 scale-to-TMEM ingress 为 schedule-issued scale source payload：

\[
Q_{\mathrm{TMEM,scale}}
=Q_{A,s}^{\mathrm{TMA}}+Q_{B,s}^{\mathrm{TMA}}.
\]

对 split-K，定义 reduction I/O：

\[
Q_{\mathrm{reduction}}
=2(\mathrm{split\_k}-1)MNs_{\mathrm{acc}}.
\]

当前 v1 该项为 0。

## 8. 工作量到资源的映射

| 资源层 | 分子 |
| --- | --- |
| strict compute | \(W_{\mathrm{use}}\) |
| empirical compute | \(W_{\mathrm{issue}}\) |
| strict external memory | minimum input + C read + minimum output |
| strict L2 read/write | minimum read 与 minimum write 分开作为 GPU-wide 外约束 |
| empirical physical HBM duplex | unique cold-entry read + output write；必须 exact ratio-matched physical joint capacity |
| empirical L2 duplex | issued TMA read + C read + output write；必须 exact ratio-matched joint capacity |
| per-SM TMA ingress | 每 task issued bytes 与 task waves |
| TMEM scale/readback | 对应 schedule-issued scale 与 accumulator bytes |

下一章只使用不可避免的最低工作和 rate upper 构造条件性能上界。
