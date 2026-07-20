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
