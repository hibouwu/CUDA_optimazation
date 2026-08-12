# SM110 TMEM accumulator readback bandwidth

This benchmark measures the explicit `tcgen05.ld` path used to move an FP32
accumulator tile from TMEM into warp registers. It is deliberately distinct
from `08_tmem_consume_bandwidth`, which measures a TS MMA instruction consuming
an operand resident in TMEM.

The four cases cross `32x32b.x8`/`32x32b.x16` load forms with one/four active
warps per CTA. Every load is followed by `tcgen05.wait::ld`; the reported byte
count is the architectural destination payload:

```text
32 lanes * registers_per_lane * 4 bytes
```

The full-GPU interval is the earliest CTA `%globaltimer` start through the
latest CTA stop. The 20-block Thor/T5000 run is accepted only when the result
auditor sees all 20 SM IDs. TMEM initialization, allocation, deallocation, and
the final checksum are outside the reported read loop except for two CTA
barriers that bracket it; using 10,000 iterations makes their contribution
measurable and negligible rather than silently subtracting it.

Compile-only check:

```bash
bash microbench/12_tmem_readback_bandwidth/build_and_run.sh build-only
```

On the repository author's Fedora/CUDA 13.0 host only, the known host-header
compatibility workaround can be enabled with
`NVCC_HOST_UNDEF_GNU_SOURCE=1`; it is omitted on Thor by default.

The unified component campaign is the canonical hardware collection entry;
the local `run` action is only a convenient manual probe.
