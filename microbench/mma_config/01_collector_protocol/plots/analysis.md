# 01_collector_protocol analysis

## Observation
- valid cases: 90
- invalid cases: 0
- fastest median cycles case: `collector_bf16_m128n256k16_same_fill_lastuse_r0` = 1028366.000 cycles
- best TFLOP/s case: `collector_fp16_m128n256k16_pingpong_discard_r0` = 36.784602
- discard median cycles/MMA range: 897.934 to 898.132.
- fill/use/lastuse median cycles/MMA range: 988.076 to 1383.852.
- weights-stationary B collector cases executed for b1, b2, b4.

## Inference
- Rows report software-visible behavior only. `pending_mbarriers` is treated as cumulative completion-prefix tracking, not as an independent async group queue.
- Effective SMEM rates, when present, are logical operand bytes per measured cycle under collector-discard conditions and are not physical port widths.

## Unsupported Claim
- These results do not identify physical SMEM bank count, physical TMEM bank width, or hidden collector depth.
