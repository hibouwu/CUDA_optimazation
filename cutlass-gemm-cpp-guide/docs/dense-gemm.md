# Dense GEMM

## FP16 control

[`dense_f16_1sm_p128`](../cases/dense_f16_1sm_p128/) 是所有后续 case 的 control：

```cpp
using Config = guide::DenseGemmConfig<
    cutlass::half_t, cutlass::half_t, float,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_1, cute::_1, cute::_1>,
    cutlass::gemm::KernelTmaWarpSpecialized1SmSm100,
    cutlass::epilogue::NoSmemWarpSpecialized1Sm,
    8, 8, 4>;
```

A 和 B 在 host 上以 logical `(M,K)` / `(N,K)` 生成，再通过 CUTLASS stride layout填入
实际物理 storage。CPU reference只读 logical arrays，因此 layout 错误不会在 reference 端被复制。

## 数据流

```text
A/B GMEM
  -> TMA / cp.async.bulk.tensor
A/B SMEM descriptors
  -> tcgen05.mma.kind::f16
TMEM accumulator
  -> tcgen05.ld
registers
  -> direct/no-SMEM epilogue -> D
```

普通 dense SS 路径不需要 `tcgen05.cp`。本仓库不通过插入无效 copy 来制造“三指令齐全”的
dense case；三路径在 block-scaled 章节自然出现。

## BF16 与 FP8

- BF16 使用相同 `kind::f16` family，但 descriptor type不同；
- 非 block-scaled E4M3 使用 `f8f6f4` family，不带 SFA/SFB；
- FP8 case的容差单独记录，不能复用 FP16 threshold；
- 三者都做完整输出和 output canary，不只比较若干 sample。

## 错误判断

- binary里任意位置出现 `UTCCP`，不能证明 dense mainloop用了它；
- 只能检查被测 kernel function block；
- `can_implement` success仍不是数值 PASS；
- `128³` 结果不用于性能比较。
