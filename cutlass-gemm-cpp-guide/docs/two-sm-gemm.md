# 2SM / CTA-pair GEMM

[`dense_f16_2sm_p256x128x128`](../cases/dense_f16_2sm_p256x128x128/) 固定：

```text
problem       256 x 128 x 128
MMA tile      256 x 128 x 64
cluster       2 x 1 x 1
CTA group     2
schedule      KernelTmaWarpSpecialized2SmSm100
```

## 角色

- 两个 peer CTA构成 CTA pair；
- 只有 leader CTA发出 `tcgen05.mma.cta_group::2`；
- TMA/multicast和 barrier transaction count必须覆盖 CTA pair；
- TMEM accumulator的可见性与每个 CTA的 epilogue partition需要分别处理。

## 验收

PTX contract要求同一 kernel block中出现：

```text
cp.async.bulk.tensor
tcgen05.mma.cta_group::2.kind::f16
tcgen05.ld
```

SASS contract要求 2CTA MMA family，而不是在另一个 1SM kernel里找到普通 `UTCHMMA`。

## 不做的推断

2CTA SASS只能证明该 binary选择了 2SM 指令形态，不能单独证明：

- 两个 SM总是并行达到峰值；
- multicast没有争用；
- runtime数值正确；
- 2SM 对 `128³` problem有意义。
