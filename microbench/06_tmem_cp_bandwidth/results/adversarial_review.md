# TMEM cp ingress adversarial review

Pass with scope limitation: existing data measures the `tcgen05.cp` SMEM-to-TMEM ingress path, not raw TMEM bank bandwidth.

Checked evidence:

- App cp-only timing reports 859.024 B/cycle/GPU and 2.384 cycles/cp.
- Rough cp ingress upper for this directory is the cp-only sustained value, 859.024 B/cycle/GPU; no dedicated NCU UTCCP byte peak is exposed.
- The same cp-only row is stable across FP4, FP8, and BF16 generated shapes because the measured instruction suffix/effective bytes are the same.
- Interference sweep reaches up to 773.397 B/cycle/GPU of cp traffic while showing MMA slowdown, so the path is performance-visible.
- `../mma_config/06_tmem_dependency/plots/analysis.md` explicitly says those dependency rows cannot identify TMEM bank count, bank width, write bandwidth, or hidden dependency scoreboard size.
- Representative cp-only SASS contains 29 static `UTCCP.T.S.128dp128bit` instructions, confirming the intended tcgen05.cp path.
- Existing key NCU report is exportable and reports memory throughput at 8.300% peak active.
- The same NCU report gives zero for `sm__inst_executed_pipe_tmem.*`; this is treated as metric coverage limitation for UTCCP, not as evidence that cp did not execute.

Conclusion boundary:

- OK to report: `tcgen05.cp` ingress throughput to TMEM, and cp interference with SS/TS MMA.
- Not OK to report from this directory alone: raw TMEM read bandwidth, raw TMEM write-port peak, bank count, or bank width.
