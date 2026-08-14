# TMA GMEM-to-SMEM bandwidth microbenchmark

This directory measures `cp.async.bulk.tensor.3d.shared::cta.global` copy
throughput from global memory into CTA shared memory on Thor.

Modes:

- `l2-hit`: 16 MiB tensor backing storage, intended to fit in L2.
- `dram-stream`: backing storage is rounded up to at least one unique tile per
  CTA iteration, intended to exceed L2 and avoid timed-loop reuse.

Each CTA owns one tensor-map tile per iteration. `--inflight 1` preserves the
serial baseline: one issuer thread launches one TMA load into shared memory,
arrives with expected transaction bytes, waits on the slot's shared-memory
mbarrier, then advances. `--inflight 4` gives four destination slots independent
mbarriers and uses a prefill/rolling-reclaim/drain window, so a CTA can keep four
TMA requests outstanding like the four-stage tc5a mainloop instead of measuring
the latency of an issue-immediately-wait loop. The legacy
`bytes_per_cycle` field remains requested payload divided by the maximum
per-CTA `clock64()` span. The closure-qualified whole-GPU rate is instead
`globaltimer_gbytes_per_second`: total requested payload divided by the interval
from the earliest CTA `%globaltimer` start to the latest CTA stop.

```bash
./build_and_run.sh run
./build_and_run.sh validate
./build_and_run.sh ncu
```

Interpretation boundaries:

- This measures the end-to-end TMA ingress path, including issue, completion,
  mbarrier wait, and shared-memory destination effects.
- `inflight=1` and `inflight=4` are separate empirical conditions. The latter
  is the saturation candidate used by the GEMM model; retaining the serial case
  makes the concurrency gain auditable instead of silently changing the old
  capacity's meaning.
- A Thor/T5000 full-GPU observation is accepted only when
  `unique_smid_count == sm_count == 20`; otherwise its issued-byte accounting
  is not promoted into the empirical envelope.
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

Formal closure measures both concurrency contracts with ten external trials:

```bash
./tma_gmem_smem_bandwidth --mode l2-hit --bytes $((16 << 20)) \
  --tile-bytes 32768 --slots 4 --inflight 1 --iters 4096 \
  --warmup-iters 32 --blocks-per-sm 1 --threads 128
./tma_gmem_smem_bandwidth --mode l2-hit --bytes $((16 << 20)) \
  --tile-bytes 32768 --slots 4 --inflight 4 --iters 4096 \
  --warmup-iters 32 --blocks-per-sm 1 --threads 128
```

The formal campaign also includes `16 KiB × inflight=8`; tc5a has four stages
and issues separate A/B TMA requests per stage, so this point checks whether
eight outstanding requests are needed to saturate the per-SM exit. It does not
claim that equal 16 KiB probe requests reproduce tc5a's exact 16 KiB/32 KiB
pair. The DRAM-stream cases use the same three concurrency points with a 256 MiB
unique working-set contract. A pipeline rate is not a specification upper: the
empirical layer still intersects it with all applicable hard ceilings.
