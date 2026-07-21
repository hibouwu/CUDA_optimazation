#

## Tensor Core operand 供数、吞吐和 SMEM 带宽

以 BF16、`M=128, K=16` 为例，若 A、B 都来自 SMEM：

$$ \text{SMEM bytes/MMA} = 2(MK+NK) $$

其中每个 BF16 是 2 bytes。

加上前文中NVIDIA给出的 Thor 上 BF16 Tensor Core 峰值是 8192 FLOP / cycle / SM

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

旧 `mma_config` runtime-dispatch kernel 在 timed loop 内包含 descriptor、
operand slot、SMEM 地址、D ring 地址、collector/protocol 分派、wait 和
CTA 同步，因此不再用于硬件路径推断。新的静态校准使用每个 case 一个
compile-time binary，并把 descriptor、SMEM 地址和 TMEM setup 移到 timed
region 外。

BF16 full-grid、collector discard、wait_hint=0、same-D、`input_d=0` 的静态
Q sweep：

| Shape | Q1 | Q2 | Q4 | Q8 | Q16 | Q32 | Q64 | fitted beta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `m128n128k16` | 450.270 | 247.145 | 145.581 | 102.079 | 86.768 | 75.333 | 68.937 | 63.747 |
| `m128n256k16` | 450.262 | 308.800 | 212.660 | 181.482 | 154.818 | 141.364 | 135.092 | 129.381 |

`m128n128k16` Q4 的 `145.581 cycles/MMA` 与已有可信 BF16 K4 mainloop
`146.132 cycles/MMA` 同量级且几乎一致，因此静态 harness 可以作为后续
分析的主校准口径。

用 fitted beta 计算的 logical operand service rate：

| Shape | Logical operand bytes/MMA | fitted beta | logical bytes/cycle |
| --- | ---: | ---: | ---: |
| `m128n128k16` | 8192 B | 63.747 | 128.5 |
| `m128n256k16` | 12288 B | 129.381 | 95.0 |

这个表只说明 software-visible logical service rate。它不能命名为物理
SMEM->Tensor Core 端口宽度，也不能推出每 cycle 读取多少 SMEM bank 或
写入多少 TMEM bank。

关键控制项：

| Control | BF16 N128 full-grid observation |
| --- | ---: |
| empty/control loop | ~264 cycles/iteration |
| commit + already-completed wait | ~258 cycles/iteration |
| forced single-MMA wait, hint=0 | ~450 cycles/MMA |
| forced single-MMA wait, hint=32 或 0x989680 | ~431 cycles/MMA |
| CTA-wide `__syncthreads()` | 20.828 cycles/sync |

因此，`Q=1` 是 forced-completion diagnostic，不应通过空 commit/wait 相减来
解释为纯 MMA latency；长 batch 的 beta 才是当前 harness 下更稳的边际成本。
