# 05_ldshared_contention analysis

## Observation
- valid cases: 42
- invalid cases: 0
- fastest median cycles case: `ldcont_bf16_interference_only_ops0` = 3212295.000 cycles
- best TFLOP/s case: `ldcont_bf16_none_ops0` = 18.387145
- Fixed active interference warp count; modes present: interference_only, l1_hit_global, ld_shared, none, predicated_off_load, register_alu.
- none TFLOP/s median range: 18.387015 to 18.387145.
- register_alu TFLOP/s median range: 1.340939 to 17.012779.
- predicated_off_load TFLOP/s median range: 1.275401 to 17.012770.
- l1_hit_global TFLOP/s median range: 0.912927 to 17.012766.
- ld_shared TFLOP/s median range: 1.048234 to 17.012753.

## Inference
- Rows report software-visible behavior only. `pending_mbarriers` is treated as cumulative completion-prefix tracking, not as an independent async group queue.
- Effective SMEM rates, when present, are logical operand bytes per measured cycle under collector-discard conditions and are not physical port widths.

## Unsupported Claim
- These results do not identify physical SMEM bank count, physical TMEM bank width, or hidden collector depth.
