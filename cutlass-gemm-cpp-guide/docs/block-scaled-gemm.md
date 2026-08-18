# Block-scaled GEMM

## 数学语义

对 scale vector size `SV`：

```text
D[i,j] = Σk (
  decode(Aq[i,k]) * SFA[i, floor(k / SV)]
) * (
  decode(Bq[j,k]) * SFB[j, floor(k / SV)]
)
```

完整类型名称必须同时包含 value、scale、SV、accumulator 和 output；“FP4 GEMM”不完整。

## MXFP8 `128³` flagship

[`bs_mxfp8_1sm_p128`](../cases/bs_mxfp8_1sm_p128/)：

```text
value       E4M3
scale       E8M0
SV          32
problem     128 x 128 x 128
tile        128 x 128 x 128
accumulator FP32
output      BF16
```

该 case在固定工具链的同一 PTX function block中已经静态观测到：

```text
cp.async.bulk.tensor.3d / .4d
tcgen05.cp.cta_group::1.32x128b.warpx4
tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale
tcgen05.ld.sync.aligned...
```

这证明静态 lowering，不证明 Thor runtime correctness。

## MXFP4 与 NVFP4

| Format | Value | Scale | SV | v0.1 tile K |
|---|---|---|---:|---:|
| MXFP4 | E2M1 | E8M0 | 32 | 256 |
| NVFP4 | E2M1 | UE4M3 | 16 | 256 |

两者不能借用 MXFP8 的 `128×128×128` tile contract。sub-byte input由 CUTLASS
`HostTensor<..., PackedVectorLayout>` 打包；独立 CPU oracle保存转换后的 logical values，
而不是从打包后的 device buffer反解。

## Scale layout验证

每个 logical scale同时经过：

1. 独立 closed-form offset；
2. CUTLASS `Sm1xxBlockScaledConfig` layout；
3. physical tensor写入。

1 与 2 不一致时在 launch前失败。数值 reference使用 logical scale array，所以 physical
layout错误会导致 GPU output与 reference不一致。
