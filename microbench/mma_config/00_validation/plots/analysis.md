# 00_validation 分析

## 观察

- 有效 cases: 294
- 无效 cases: 96
- 最快 median cycles case: `val_fp16_m128n128k16_sw32_c128_d0_discard_q1` = 1812.000 cycles
- 最高 TFLOP/s case: `val_fp16_m128n256k16_sw32_c256_d0_fill_use_lastuse_q4` = 1.663568
- 后续 stage 运行前已经覆盖 descriptor、TMA load、MMA issue、commit/wait、完整 D readback、guard columns 和 CUDA error 检查。
- tested shape 下 M128 FP32 accumulator 的 TMEM D footprint：

| Shape | D footprint columns | 512 columns 中最大不重叠 D tiles |
| --- | ---: | ---: |
| m128n64k16 | 64 | 8 |
| m128n128k16 | 128 | 4 |
| m128n256k16 | 256 | 2 |

- 按 dtype/N 统计的 valid descriptor rows: bf16/N64:55, bf16/N128:55, bf16/N256:37, fp16/N64:55, fp16/N128:55, fp16/N256:37.
- invalid reason counts: max_abs_error>0.05:48, max_abs_error>0.25:48
- 全部 96 个 invalid row 都是 `smem_layout=none` / `swizzle=none`。CUDA launch、wait 和 TMEM guard 检查完成，但 D 值超出 tolerance。这些 row 被分类为“可执行但数值错误”的 descriptor/layout case，并从所有性能结论中排除。
- 静态性能实验使用通过各自 full-D 和 guard validation 的 swizzled descriptor case。

## 推断

- row 只报告软件可见行为。`pending_mbarriers` 被视为累计 completion-prefix tracking，而不是独立 async group queue。
- 如存在 effective SMEM rate，它只表示 collector-discard 条件下 logical operand bytes / measured cycle，不是物理 port width。
- `layout=none` 不作为当前性能矩阵支持的 descriptor，因为 validation 显示它有确定性数值错误。

## 不支持的说法

- 这些结果不能识别物理 SMEM bank count、物理 TMEM bank width 或 hidden collector depth。
