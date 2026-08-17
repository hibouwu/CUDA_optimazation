# GEMM Shape Ledger

## 需要分开的八层 shape

1. `problem_mnkl`：数学问题；
2. CTA/CTA-pair tile；
3. `MmaTileShape`：collective 一次负责的 tile；
4. 单条 `tcgen05.mma` instruction shape；
5. TMA transfer tile；
6. 每个 SMEM pipeline stage；
7. epilogue tile；
8. scale-vector 和 physical scale chunk。

## Dense FP16 `128³`

```text
problem              128 x 128 x 128
MMA tile             128 x 128 x 64
instruction          128 x 128 x 16
K tiles per problem  2
instructions/K tile  4
cluster              1 x 1 x 1
CTA group            1
```

`problem K=128` 不能推出 `MmaTile K=128`，也不能推出单条 instruction K=128。

## MXFP8 block-scaled `128³`

```text
problem/tile K       128
instruction K        32
MMA K steps          4
scale vector         32
logical SFA          128 x 4
logical SFB          128 x 4
physical chunk       128 x 4 = 512 scale bytes
```

独立 closed-form scale offset 为：

```text
block = block_sf * ceil(MN / 128) + block_mn
offset = block * 512
       + (mn % 32) * 16
       + ((mn % 128) / 32) * 4
       + (sf % 4)
```

Host test会把该公式与 CUTLASS `Sm1xxBlockScaledConfig` layout逐坐标比较；CPU GEMM
reference 使用原始 logical scales，不从被测 physical buffer反解。

## 2SM

`dense_f16_2sm_p256x128x128` 使用 `MmaTile M=256` 和 cluster `2×1×1`。
它与单 SM 的 `problem M=128` 不是同一 case；不能为了复用文件名而混淆。
