# 04_smem_layout_address analysis

## Observation
- valid cases: 54
- invalid cases: 90
- fastest median cycles case: `layout_fp16_m128n256k16_sw32_off0` = 3387342.000 cycles
- best TFLOP/s case: `layout_fp16_m128n256k16_sw32_off0` = 39.940304
- Valid SMEM base offsets in this run: 0, 128, 256.
- Invalid SMEM base offsets/descriptors are isolated in invalid_cases.csv: 0, 16, 32, 64, 128, 256.
- invalid reason counts: max_abs_error>0.05:9, max_abs_error>0.25:9, misaligned address:72

## Inference
- Rows report software-visible behavior only. `pending_mbarriers` is treated as cumulative completion-prefix tracking, not as an independent async group queue.
- Effective SMEM rates, when present, are logical operand bytes per measured cycle under collector-discard conditions and are not physical port widths.

## Unsupported Claim
- These results do not identify physical SMEM bank count, physical TMEM bank width, or hidden collector depth.
