# TMEM cp ingress bandwidth summary

This directory consolidates the existing `tcgen05.cp` measurements from
`../mma_with_cp`.  It is intentionally scoped to the SMEM-to-TMEM ingress path
exercised by `tcgen05.cp`, not to a raw TMEM bank/load/store bandwidth model.

Run:

```bash
./build_and_run.sh summarize
```

Generated files:

- `results/tmem_cp_only_summary.csv`: cp-only app timing copied from
  `../mma_with_cp/plots/cp_only_results.csv`.
- `results/tmem_cp_interference_summary.csv`: cp traffic observed while adding
  cp noise to SS/TS MMA mainloops.
- `results/ncu/tmem_cp_only_ncu_summary.csv`: selected counters exported from
  the existing key NCU report when present.
- `results/adversarial_review.md`: boundary and validity review.

Current headline:

| Path | Sustained app throughput | Latency proxy |
|---|---:|---:|
| `tcgen05.cp` only, SMEM-to-TMEM ingress | 859.024 B/cycle/GPU | 2.384 cycles/cp |

Interpretation:

- This is meaningful for TS pipelines that stage an operand in TMEM with
  `tcgen05.cp`.
- It should be reported as `tcgen05.cp` ingress throughput, not as pure TMEM
  write-port peak.
- Existing dependency experiments in `../mma_config/06_tmem_dependency` do not
  identify TMEM bank count, bank width, raw write bandwidth, or raw read
  bandwidth.
