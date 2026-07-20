# 00_validation analysis

## Observation
- valid cases: 294
- invalid cases: 96
- fastest median cycles case: `val_fp16_m128n128k16_sw32_c128_d0_discard_q1` = 1812.000 cycles
- best TFLOP/s case: `val_fp16_m128n256k16_sw32_c256_d0_fill_use_lastuse_q4` = 1.663568
- Descriptor, TMA load, MMA issue, commit/wait, full D readback, guard columns, and CUDA error checks are all exercised before later stages run.
- TMEM D footprint for M128 FP32 accumulators under the tested shapes:

| Shape | D footprint columns | Max non-overlap D tiles in 512 columns |
| --- | ---: | ---: |
| m128n64k16 | 64 | 8 |
| m128n128k16 | 128 | 4 |
| m128n256k16 | 256 | 2 |
- Valid descriptor rows by dtype/N: bf16/N64:55, bf16/N128:55, bf16/N256:37, fp16/N64:55, fp16/N128:55, fp16/N256:37.
- invalid reason counts: max_abs_error>0.05:48, max_abs_error>0.25:48

## Inference
- Rows report software-visible behavior only. `pending_mbarriers` is treated as cumulative completion-prefix tracking, not as an independent async group queue.
- Effective SMEM rates, when present, are logical operand bytes per measured cycle under collector-discard conditions and are not physical port widths.

## Unsupported Claim
- These results do not identify physical SMEM bank count, physical TMEM bank width, or hidden collector depth.
