# TMA、TMEM 与 TCGen05

## 三个不同的数据移动概念

### TMA / `cp.async.bulk.tensor`

负责 GMEM↔SMEM tensor transfer，由 tensor map、坐标、swizzle 和 mbarrier描述。它不是
`tcgen05.cp`，也不按普通 per-thread LDG/STS copy理解。

### `tcgen05.cp`

负责 SMEM→TMEM。本仓库有两种用途：

- block-scaled collective把 SFA/SFB 搬入 TMEM；
- mechanism-only TS test把 A operand搬入 TMEM，再由 TS MMA消费。

### `tcgen05.ld`

负责 TMEM accumulator→register。它不是 SMEM load；在使用 accumulator前必须满足相应
completion/wait contract。

## MXFP8 的 source-to-instruction 路径

固定 CUTLASS v4.6.1 中：

1. `sm100_blockscaled_mma_warpspecialized.hpp` 为 A/B/SFA/SFB 建 TMA copy；
2. collective选择 `SM100_UTCCP_4x32dp128bit_1cta`；
3. `copy_sm100.hpp` wrapper发出
   `tcgen05.cp.cta_group::1.32x128b.warpx4`；
4. `cute::gemm` 带 TMEM SFA/SFB fragment发出
   `tcgen05.mma...mxf8f6f4.block_scale`；
5. epilogue使用 TMEM-load atom读回 accumulator。

`tests/mechanism/test_cutlass_source_contracts.py` 检查上述源码链；最终 PTX/SASS 仍由每个
case 的实际 binary验证。

## 异步边界

教程把下面事件分开画和测试：

```text
TMA transaction complete
SMEM stage available to MMA
MMA complete / accumulator ready
TMEM read complete
SMEM stage reusable
TMEM allocation releasable
```

不能因为源码调用顺序相邻，就推断硬件调度或物理传输顺序。
