# 07_config_matrix analysis

## Observation
- valid cases: 192
- invalid cases: 64
- fastest median cycles case: `cfg_bf16_m128n256k16_c256_sw32_discard_same` = 2891088.000 cycles
- best TFLOP/s case: `cfg_bf16_m128n256k16_c256_sw32_discard_same` = 46.796040
- Top config `cfg_bf16_m128n256k16_c256_sw32_discard_same`: 46.796040 TFLOP/s, beta 705.832 cycles/MMA.
- Top config `cfg_fp16_m128n256k16_c256_sw32_discard_pingpong`: 46.796040 TFLOP/s, beta 705.832 cycles/MMA.
- Top config `cfg_bf16_m128n256k16_c256_sw32_discard_pingpong`: 46.795991 TFLOP/s, beta 705.833 cycles/MMA.
- Top config `cfg_fp16_m128n256k16_c256_sw32_discard_same`: 46.795975 TFLOP/s, beta 705.833 cycles/MMA.
- Top config `cfg_fp16_m128n256k16_c256_sw32_fill_use_lastuse_same`: 45.225567 TFLOP/s, beta 730.342 cycles/MMA.
- invalid reason counts: max_abs_error>0.05:32, max_abs_error>0.25:32

## Inference
- Rows report software-visible behavior only. `pending_mbarriers` is treated as cumulative completion-prefix tracking, not as an independent async group queue.
- Effective SMEM rates, when present, are logical operand bytes per measured cycle under collector-discard conditions and are not physical port widths.

## Unsupported Claim
- These results do not identify physical SMEM bank count, physical TMEM bank width, or hidden collector depth.
