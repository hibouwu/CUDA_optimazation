# 03_effective_smem_ingress analysis

## Observation
- valid cases: 18
- invalid cases: 0
- fastest median cycles case: `ingress_bf16_m128n256k16_rotating` = 7255257.000 cycles
- best TFLOP/s case: `ingress_bf16_m128n256k16_rotating` = 37.294742
- collector-discard logical effective SMEM operand rate range: 6.935 to 13.875 bytes/cycle.

## Inference
- Rows report software-visible behavior only. `pending_mbarriers` is treated as cumulative completion-prefix tracking, not as an independent async group queue.
- Effective SMEM rates, when present, are logical operand bytes per measured cycle under collector-discard conditions and are not physical port widths.

## Unsupported Claim
- These results do not identify physical SMEM bank count, physical TMEM bank width, or hidden collector depth.
