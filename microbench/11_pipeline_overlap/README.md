# TCGen05 CP/MMA pipeline overlap microbenchmark

This directory measures end-to-end overlap between `tcgen05.cp` SMEM-to-TMEM
traffic and TS MMA consumption on the FP4 M128N256 path.

Cases:

- `cp-only`: component baseline for `tcgen05.cp` ingress.
- `ts-mma-only`: component baseline for TS MMA consuming A from TMEM.
- `serial-a1`: one cp and one MMA serialized through one TMEM slot.
- `overlap-a2`: double-buffered cp/MMA overlap.
- `warp-split-a2`: two issuer groups split across cp/MMA work.
- `mainloop-a2-k16`: longer steady-state A2 mainloop with 16 K blocks.

```bash
./build_and_run.sh run
./build_and_run.sh validate
./build_and_run.sh ncu
./build_and_run.sh review
```

Interpretation boundaries:

- This is not a GMEM/TMA pipeline. It starts from data already resident for the
  generated TCGen05 microbenchmarks and measures the `tcgen05.cp`/TMEM/MMA part.
- `bytes_per_cycle` from the binaries is cp payload traffic. TMEM consume demand
  is estimated as `2048 B` per TS MMA instruction.
- Nsight Compute on this platform does not expose a reliable UTCCP byte counter;
  NCU is used for SM/tensor pipe utilization, scheduler stalls, and rough
  bandwidth upper estimates.
