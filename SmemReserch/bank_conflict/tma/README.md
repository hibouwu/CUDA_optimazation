# TMA copy, shared-memory consumer/producer, and round-trip benchmark

This benchmark separates four questions that should not be mixed:

1. Can the profiler observe a copy performed by the TMA async proxy?
2. After a TMA load, what happens when an ordinary LSU consumer reads the
   resulting shared-memory layout?
3. Before a TMA store, what happens when an ordinary LSU producer writes that
   layout?
4. What is the end-to-end cost of a TMA load followed by a TMA store-back?

TMA itself does not issue one ordinary `ld.shared` or `st.shared` per lane.
Consequently, T0 and T3 must not be described as N-way bank-conflict tests.
Shared-memory bank-conflict counters are interpreted primarily for the
ordinary consumer in T1 and producer in T2.

## Controlled tensor geometry

Every case transfers the same 4096-byte tile and uses the same tensor box:

```text
data type          = uint8
box                = 32 bytes × 128 rows
total bytes        = 4096
shared alignment   = 1024 bytes
threads per block  = 256
consumer/producer  = one 32-lane warp
```

The 32-byte innermost box span is legal for none, 32B, 64B, and 128B tensor-map
swizzle modes. Keeping the logical geometry fixed avoids the old benchmark's
`128×32`, `32×128`, and `64×64` tensor-box change.

When the swizzle width is larger than the 32-byte inner dimension, the CUDA
guide requires shared memory to accommodate the complete swizzle width.
Physical shared-memory footprints are therefore:

| Swizzle | Physical row span | Shared footprint |
| --- | ---: | ---: |
| none | 32 bytes | 4096 bytes |
| 32B | 32 bytes | 4096 bytes |
| 64B | 64 bytes | 8192 bytes |
| 128B | 128 bytes | 16384 bytes |

This footprint difference is an inherent resource cost and is emitted as
`shared_bytes` in the CSV.

Swizzle operates on 16-byte atoms. With a 1024-byte-aligned shared base, the
matched ordinary LSU address calculation uses:

```text
row             = logical_byte_offset / 32
atom_in_row     = (logical_byte_offset % 32) / 16
atoms_per_span  = swizzle_width / 16
physical_atom   = (row % atoms_per_span) XOR atom_in_row
physical_offset = row * swizzle_width + physical_atom * 16
```

Bytes inside each 16-byte atom retain their order. The consumer and producer
use explicit 16-byte `ld.shared.v4.u32` and `st.shared.v4.u32` operations.

## T0: copy-only and profiler observability

| Case | Operation | Purpose |
| --- | --- | --- |
| `T0a_gmem_to_smem_no_swizzle_copy` | TMA GMEM → SMEM | Baseline load throughput and profiler visibility |
| `T0b_smem_to_gmem_no_swizzle_copy` | TMA SMEM → GMEM | Baseline store throughput and bulk-group completion |

T0 has no timed ordinary shared-memory consumer/producer inside the iteration.
The shared tile initialization and final checksum are fixed per-kernel setup
and validation work, amortized by `--iters`.

## T1: consumer reads after TMA load

| Case | TMA load layout | Ordinary consumer |
| --- | --- | --- |
| `T1a_load_no_swizzle_column_consumer` | none | Column-wise 16-byte load |
| `T1b_load_32b_swizzle_matched_consumer` | 32B | Matched swizzled column load |
| `T1c_load_64b_swizzle_matched_consumer` | 64B | Matched swizzled column load |
| `T1d_load_128b_swizzle_matched_consumer` | 128B | Matched swizzled column load |

Each iteration waits for TMA load completion and then one warp reads one
16-byte logical atom per row. The no-swizzle case uses the linear address. The
swizzled cases apply the matching 16-byte-atom mapping before issuing ordinary
LSU shared loads.

T1 is the stage where shared-load bank-conflict metrics are meaningful.

## T2: producer writes before TMA store

| Case | Ordinary producer | TMA store layout |
| --- | --- | --- |
| `T2a_column_producer_store_no_swizzle` | Column-wise 16-byte store | none |
| `T2b_matched_producer_store_32b_swizzle` | Matched swizzled store | 32B |
| `T2c_matched_producer_store_64b_swizzle` | Matched swizzled store | 64B |
| `T2d_matched_producer_store_128b_swizzle` | Matched swizzled store | 128B |

One warp writes one logical 16-byte atom per row. After CTA synchronization,
the issuing thread executes:

```text
fence.proxy.async.shared::cta
TMA SMEM → GMEM
cp.async.bulk.commit_group
cp.async.bulk.wait_group 0
```

The proxy fence makes generic-proxy shared writes visible to the TMA async
proxy. T2 is the stage where shared-store bank-conflict metrics are meaningful.

## T3: TMA load plus TMA store-back

| Case | Round-trip layout |
| --- | --- |
| `T3a_load_store_no_swizzle` | none |
| `T3b_load_store_32b_swizzle` | 32B |
| `T3c_load_store_64b_swizzle` | 64B |
| `T3d_load_store_128b_swizzle` | 128B |

Each iteration performs:

```text
GMEM input
  → TMA load
  → swizzled or linear SMEM tile
  → TMA store with the matching tensor map
  → GMEM output
```

There is no ordinary per-lane consumer or producer in T3. It measures the
round-trip TMA path and verifies that store-back reconstructs the original
global-memory byte order.

## Correctness and CSV

All cases report `PASS` or `FAIL`:

- T0a and T1 validate the final shared-tile checksum.
- T0b validates the stored copy.
- T2 simulates the matched producer layout on the host and compares every
  output byte.
- T3 compares every output byte with the input.

CSV fields:

```text
experiment,case,direction,swizzle,box_x_bytes,box_y,shared_bytes,
consumer,producer,tma_operations,iters,avg_ms,min_ms,
tma_bytes,effective_GBps,correctness
```

`effective_GBps` counts only TMA transfer bytes. T0–T2 count 4096 bytes per
iteration; T3 counts 8192 bytes because it performs one load and one store.

## Build and run

```bash
CUDA_ARCH=110 ./scripts/build.sh
./scripts/run_basic.sh
./scripts/run_basic.sh --case T0
./scripts/run_basic.sh --case T1 --iters 1000
./scripts/run_basic.sh --case T2c_matched_producer_store_64b_swizzle
```

`run_basic.sh` writes `results/basic_results.csv` and generates:

- `results/avg_ms.png`
- `results/effective_gbps.png`

The CLI supports `--case all`, stage selectors `T0` through `T3`, exact case
names, `--iters`, `--warmups`, `--repeats`, and `--list-cases`.

## Nsight Compute

```bash
./scripts/run_ncu.sh
./scripts/run_ncu.sh --case T1
./scripts/run_ncu.sh --case T2
```

The script expands a stage through `--list-cases` and profiles each exact case
separately. Candidate metrics include:

- ordinary shared-load/store bank conflicts and requests for T1/T2;
- shared bank read/write activity;
- MIO throttle stalls.

Metric availability varies by GPU and Nsight Compute version:

```bash
ncu --query-metrics | grep -Ei 'tma|tensor|shared|bank|mio'
```

Do not interpret an ordinary LSU bank-conflict counter as a complete measure
of TMA async-proxy behavior. Compare TMA throughput, elapsed time, and available
TMA/tensor metrics as a separate layer.

## References

- [PTX `cp.async.bulk.tensor`](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor)
- [PTX bulk async-group completion](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-wait-group)
- [PTX tensor swizzling modes](https://docs.nvidia.com/cuda/parallel-thread-execution/#swizzling-modes)
- [CUDA TMA swizzle guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
