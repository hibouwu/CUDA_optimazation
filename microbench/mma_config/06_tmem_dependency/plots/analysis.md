# 06_tmem_dependency analysis

## Observation
- valid cases: 172
- invalid cases: 0
- fastest median cycles case: `tmem_bf16_m128n128k16_c512_full_in0_rd4` = 3181149.000 cycles
- best TFLOP/s case: `tmem_fp16_m128n256k16_c512_full_in1_rd2` = 42.523881
- D alias classes present: full, none, partial.
- TMEM column allocations present: 128, 256, 512; independent_d_count is clamped to actual capacity.

## Inference
- Rows report software-visible behavior only. `pending_mbarriers` is treated as cumulative completion-prefix tracking, not as an independent async group queue.
- Effective SMEM rates, when present, are logical operand bytes per measured cycle under collector-discard conditions and are not physical port widths.

## Unsupported Claim
- These results do not identify physical SMEM bank count, physical TMEM bank width, or hidden collector depth.
