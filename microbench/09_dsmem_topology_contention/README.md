# DSMEM topology and contention microbenchmark

This directory measures DSMEM remote access patterns that are not covered by
the simple local-vs-neighbor baseline in `../03_dsmem_bandwidth`.

Modes:

- `ring-read-d1`, `ring-write-d1`: every CTA accesses `(rank + 1) % cluster`.
- `ring-read-d2`, `ring-write-d2`: every CTA accesses `(rank + 2) % cluster`.
- `fanin-read-root0`, `fanin-write-root0`: ranks 1..N-1 access rank 0 in each
  cluster; rank 0 participates in barriers but does not issue timed traffic.

The default cluster size is 4 and the default launch is one cluster per four
SMs, avoiding multi-wave cluster scheduling.  Reported bytes count only active
remote-access CTAs, so fan-in modes count `(cluster_size - 1)` active CTAs per
cluster.

```bash
./build_and_run.sh run
./build_and_run.sh validate
./build_and_run.sh ncu
```

Interpretation boundaries:

- These are remote DSMEM end-to-end throughput measurements, not a physical
  interconnect topology reverse-engineering proof.
- Fan-in write intentionally creates multi-writer pressure into one target CTA's
  DSMEM.  The payload is partitioned by source rank to avoid every source
  writing the exact same addresses.
- NCU validation uses `l1tex__t_bytes_pipe_lsu_mem_dshared*` counters when
  available and estimates a rough dshared model peak from NCU `%peak`.

Current cluster-size 4 results:

| Mode | App throughput | NCU dshared throughput | NCU dshared %peak |
|---|---:|---:|---:|
| `ring-read-d1` | 113.15 B/cycle/GPU | 116.30 B/cycle/GPU | 1.14% |
| `ring-read-d2` | 119.63 B/cycle/GPU | 122.66 B/cycle/GPU | 1.20% |
| `fanin-read-root0` | 52.76 B/cycle/GPU | 53.31 B/cycle/GPU | 0.52% |
| `ring-write-d1` | 127.25 B/cycle/GPU | 131.20 B/cycle/GPU | 1.28% |
| `ring-write-d2` | 141.87 B/cycle/GPU | 146.41 B/cycle/GPU | 1.43% |
| `fanin-write-root0` | 72.09 B/cycle/GPU | 72.85 B/cycle/GPU | 0.71% |

NCU `%peak` implies a rough dshared model peak of about 10.2 KiB/cycle/GPU for
these counters.  The measured remote topology paths are far below that model
peak, especially many-to-one fan-in.
