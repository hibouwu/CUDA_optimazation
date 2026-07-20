# 02_latency_throughput analysis

## Observation
- valid cases: 300
- invalid cases: 0
- fastest median cycles case: `lat_bf16_m128n128k16_same_d_in0_q48` = 369114.000 cycles
- best TFLOP/s case: `lat_fp16_m128n256k16_same_d_in0_q64` = 43.174675
- Fitted beta range over latency rows: 752.564 to 874.469 cycles/MMA.
- Q=1 forced-completion diagnostic cycles/MMA range: 1521.723 to 1643.221.
- commit-prefix scan rows: 12; `pending_mbarriers` is recorded as completion-prefix tracking count.

## Inference
- Rows report software-visible behavior only. `pending_mbarriers` is treated as cumulative completion-prefix tracking, not as an independent async group queue.
- Effective SMEM rates, when present, are logical operand bytes per measured cycle under collector-discard conditions and are not physical port widths.

## Unsupported Claim
- These results do not identify physical SMEM bank count, physical TMEM bank width, or hidden collector depth.
