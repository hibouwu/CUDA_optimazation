# TMA GMEM-to-SMEM bandwidth microbenchmark

This directory measures `cp.async.bulk.tensor.3d.shared::cta.global` copy
throughput from global memory into CTA shared memory on Thor.

Modes:

- `l2-hit`: 16 MiB tensor backing storage, intended to fit in L2.
- `dram-stream`: backing storage is rounded up to at least one unique tile per
  CTA iteration, intended to exceed L2 and avoid timed-loop reuse.

Each CTA owns one tensor-map tile per iteration.  One issuer thread launches one
TMA load into shared memory, arrives with expected transaction bytes, waits on
the shared-memory mbarrier, then advances to the next slot.  Reported bandwidth
is requested TMA payload bytes divided by the maximum per-CTA `clock64()` span.

```bash
./build_and_run.sh run
./build_and_run.sh validate
./build_and_run.sh ncu
```

Interpretation boundaries:

- This measures the end-to-end TMA ingress path, including issue, completion,
  mbarrier wait, and shared-memory destination effects.
- It is not a pure DRAM pin bandwidth test and not a pure shared-memory write
  port peak.
- NCU direct `dram__bytes*` counters may be unavailable on this machine; when
  missing, the NCU script reports LTS miss-sector bytes as the DRAM-path proxy.

Current 32 KiB tile baseline:

| Mode | App throughput | NCU TMA throughput | NCU TMA %peak | DRAM proxy/expected |
|---|---:|---:|---:|---:|
| `l2-hit` | 773-776 B/cycle/GPU | 753.86 B/cycle/GPU | 29.45% | 0.006 |
| `dram-stream` | 156-164 B/cycle/GPU | 157.32 B/cycle/GPU | 6.15% | 1.008 |

The NCU TMA `%peak` normalization implies a rough TMA read-byte model peak of
about 2.56 KiB/cycle/GPU.  LTS `%peak` normalization implies a rough LTS model
peak of about 1.024 KiB/cycle/GPU.
