# 07_config_matrix 分析

## 观察

- 有效 cases: 192
- 无效 cases: 64
- 最快 median cycles case: `cfg_bf16_m128n256k16_c256_sw32_discard_same` = 2891088.000 cycles
- 最高 TFLOP/s case: `cfg_bf16_m128n256k16_c256_sw32_discard_same` = 46.796040
- Top config `cfg_bf16_m128n256k16_c256_sw32_discard_same`: 46.796040 TFLOP/s, beta 705.832 cycles/MMA.
- Top config `cfg_fp16_m128n256k16_c256_sw32_discard_pingpong`: 46.796040 TFLOP/s, beta 705.832 cycles/MMA.
- Top config `cfg_bf16_m128n256k16_c256_sw32_discard_pingpong`: 46.795991 TFLOP/s, beta 705.833 cycles/MMA.
- Top config `cfg_fp16_m128n256k16_c256_sw32_discard_same`: 46.795975 TFLOP/s, beta 705.833 cycles/MMA.
- Top config `cfg_fp16_m128n256k16_c256_sw32_fill_use_lastuse_same`: 45.225567 TFLOP/s, beta 730.342 cycles/MMA.
- invalid reason counts: max_abs_error>0.05:32, max_abs_error>0.25:32

## 推断

- row 只报告软件可见行为。`pending_mbarriers` 被视为累计 completion-prefix tracking，而不是独立 async group queue。
- 如存在 effective SMEM rate，它只表示 collector-discard 条件下 logical operand bytes / measured cycle，不是物理 port width。

## 不支持的说法

- 这些结果不能识别物理 SMEM bank count、物理 TMEM bank width 或 hidden collector depth。
