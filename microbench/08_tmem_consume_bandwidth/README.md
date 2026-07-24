# TMEM consume via TS MMA microbenchmark

This directory measures the observed rate at which TS `tcgen05.mma` kernels
consume an A operand from TMEM.  It wraps representative generated binaries from
`../mma_with_cp/build` and collects fresh app timing, SASS evidence, and NCU
counters in this directory.

Cases:

- `ts-mma-only`: TS MMA-only FP4 M128N256.  The timed loop issues TS MMA using
  an A operand already staged in TMEM.
- `ts-cp-mma-a2-k16`: steady-state TS CP+MMA pipeline.  Each K tile issues
  `tcgen05.cp` for the next A panel and TS MMA consuming the current A panel
  from TMEM.
- `ss-mma-mainloop-k16`: SS MMA baseline with SMEM descriptors, not a TMEM-A
  consumer.

Run:

```bash
./build_and_run.sh run
./build_and_run.sh validate
./build_and_run.sh ncu
```

Interpretation boundary:

- `estimated_tmem_consume_bytes_per_cycle` assumes one 2048 B A operand is
  consumed per TS MMA instruction for this generated FP4 M128N256 shape.
- This is a TS-MMA demand-rate measurement, not a raw TMEM read-port bandwidth
  peak or a TMEM bank-width detector.

Current representative results:

| Case | Role | TFLOP/s | Estimated TMEM consume | NCU SM %peak | Rough consume upper |
|---|---|---:|---:|---:|---:|
| `ts-mma-only` | TMEM consume | 373.198 | 115.699 B/cycle/GPU | 44.77% | 258.429 B/cycle/GPU |
| `ts-cp-mma-a2-k16` | consume + cp pipeline | 332.272 | 103.011 B/cycle/GPU | 51.28% | 200.879 B/cycle/GPU |
| `ss-mma-mainloop-k16` | SMEM baseline | 921.885 | 0 B/cycle/GPU | 91.78% | N/A |

NCU `sm__inst_executed_pipe_tmem.*` reports zero for these UTCOMMA cases on
this toolchain, so it is not used as the consume-path proof.  SASS evidence is
`UTCOMMA... tmem[...]`; utilization is reported through SM/tensor/TC counters.
