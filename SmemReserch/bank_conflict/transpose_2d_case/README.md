# Two-dimensional transpose `ld.shared` bank-conflict microbenchmarks

This directory isolates the ordinary shared-memory load path for the classic
transpose access pattern. It does not study `st.shared`, TMA, descriptor
swizzle, `tcgen05`, TMEM, CUTLASS/CUTE, or full GEMM kernels.

The core transpose-style mapping is:

```text
linear_index = row * pitch + col
bank = linear_index % 32
```

For the scalar column load used in E0 and E1:

```text
row = lane
col = warp
bank(lane) = (lane * pitch + warp) % 32
```

When `gcd(pitch, 32) = d`, a warp touches `32 / d` banks and the worst per-bank
fan-in is `d`. That is the theory behind the pitch sweep.

## Scope

This directory studies only `ld.shared` behavior.

- Included: padding, pitch sweep, broadcast/multicast controls, vectorized
  shared loads, and a software XOR swizzle.
- Excluded: `st.shared`, TMA, TMA descriptor swizzle, `tcgen05`, TMEM,
  CUTLASS/CUTE, and end-to-end GEMM conclusions.

The reason for keeping only the load path here is that store-path behavior,
async copy paths, and descriptor-driven paths have different hardware rules and
would blur the interpretation of the bank-conflict results.

## Experiments

### E0 Classic Transpose Baseline

| Case | Access | Purpose | Expected bank behavior |
| --- | --- | --- | --- |
| `E0_load_pitch32` | `tile[lane * 32 + warp]` | Classic transpose worst case | All 32 lanes of a warp target one bank |

### E1 Pitch Sweep

| Case | Pitch | `gcd(pitch, 32)` | Theoretical unique banks | Theoretical conflict degree |
| --- | ---: | ---: | ---: | ---: |
| `E1_load_pitch1` | 1 | 1 | 32 | 1 |
| `E1_load_pitch2` | 2 | 2 | 16 | 2 |
| `E1_load_pitch4` | 4 | 4 | 8 | 4 |
| `E1_load_pitch8` | 8 | 8 | 4 | 8 |
| `E1_load_pitch16` | 16 | 16 | 2 | 16 |
| `E1_load_pitch31` | 31 | 1 | 32 | 1 |
| `E1_load_pitch32` | 32 | 32 | 1 | 32 |
| `E1_load_pitch33` | 33 | 1 | 32 | 1 |

These representative pitches cover the power-of-two conflict progression and
the boundary behavior around pitches 31, 32, and 33.
For very small pitches such as 1, 2, and 4, the logical rows intentionally
overlap in the backing array. That is acceptable here because E1 is about the
shared-memory bank mapping implied by `lane * pitch + warp`, not about modeling
a production transpose tile shape.

### E2 Broadcast And Multicast Controls

| Case | Access | Purpose | Interpretation |
| --- | --- | --- | --- |
| `E2_load_broadcast_same_addr` | All lanes read `tile[0]` | Isolate broadcast | Same bank and same address |
| `E2_load_multicast_2addr` | Lanes 0-15 read `tile[0]`, lanes 16-31 read `tile[1]` | Isolate 2-address multicast | Two repeated addresses |
| `E2_load_multicast_4addr` | Each eight-lane group reads one of `tile[0..3]` | Isolate 4-address multicast | Four repeated addresses |
| `E2_load_conflict_same_bank_diff_addr` | Lanes read `tile[lane * 32]` | Contrast with broadcast | Same bank and different addresses |

This stage separates two concepts that are often conflated:

- Same bank plus same address can be served as broadcast or multicast.
- Same bank plus different addresses is the ordinary bank-conflict situation.

### E3 Vector Width

| Case | Operation | Pitch | Vector width | Note |
| --- | --- | ---: | ---: | --- |
| `E3_load_f32_pitch32` | `ld.shared.f32` | 32 | 1 | Scalar transpose load |
| `E3_load_f32_pitch33` | `ld.shared.f32` | 33 | 1 | Scalar transpose load with padding |
| `E3_load_f32x2_pitch32` | `ld.shared.v2.f32` | 32 | 2 | Aligned `float2` load |
| `E3_load_f32x2_pitch33` | `ld.shared.v2.f32` | 33 | 2 | Uses column adjustment to keep alignment |
| `E3_load_f32x4_pitch32` | `ld.shared.v4.f32` | 32 | 4 | Aligned `float4` load |
| `E3_load_f32x4_pitch33` | `ld.shared.v4.f32` | 33 | 4 | Uses column adjustment to keep alignment |

The vector cases use explicit volatile PTX:

- scalar: `ld.volatile.shared.f32`
- `float2`: `ld.volatile.shared.v2.f32`
- `float4`: `ld.volatile.shared.v4.f32`

For odd pitch plus vector width, the benchmark applies a small per-row column
adjustment so the `v2` and `v4` loads remain naturally aligned. That keeps the
instruction legal, but it also means the pitch-33 vector cases are not a
byte-for-byte copy of the scalar `tile[lane * 33 + warp]` pattern. The README
states that explicitly because otherwise the comparison would be misleading.

To verify the generated width on a target machine:

```bash
cuobjdump --dump-sass build/transpose_2d_bench | grep -E "LDS|LDGSTS"
```

`LDS`, `LDS.64`, and `LDS.128` are the SASS patterns worth checking for the
scalar, `v2`, and `v4` cases respectively.

### E4 Software Swizzle

| Case | Layout rule | Goal | Contrast target |
| --- | --- | --- | --- |
| `E4_load_xor_swizzle_pitch32` | `physical_col = warp ^ (lane & 31)` | Reduce transpose-style conflict without padding | `E0_load_pitch32` |

This case keeps `pitch=32` but changes the logical-to-physical mapping:

```c++
physical_col = warp ^ (lane & 31);
index = lane * 32 + physical_col;
```

That is a software swizzle, not a TMA descriptor swizzle. The tradeoff is
different from padding:

- Padding changes row stride and increases shared-memory footprint.
- Software swizzle keeps the stride but requires both producer and consumer to
  agree on the permuted layout.

This XOR pattern is a targeted microbenchmark, not a claim that one swizzle is
universally optimal.

## CSV Output

| Field | Meaning |
| --- | --- |
| `experiment` | Stage name such as `E0_basic_pitch_effect` |
| `case` | Concrete case name such as `E1_load_pitch16` |
| `operation` | Shared-load instruction form, for example `ld.shared.f32` or `ld.shared.v4.f32` |
| `pitch` | Logical row stride used by the case |
| `vector_width` | `1`, `2`, or `4` for `f32`, `f32x2`, or `f32x4` |
| `theoretical_unique_banks` | Number of banks touched by one warp-level shared-load instruction |
| `theoretical_conflict_degree` | Largest number of distinct word addresses requested from any one bank |
| `iters` | Loop count inside one kernel launch |
| `avg_ms` | Mean kernel time over timed repeats |
| `min_ms` | Minimum kernel time over timed repeats |
| `effective_GBps` | Requested bytes divided by `avg_ms`; benchmark-local derived rate |

The theoretical fields are computed per warp by enumerating the word addresses
that a single shared-memory instruction requests. That keeps broadcast,
multicast, scalar loads, and vector loads on one consistent definition:

- `theoretical_unique_banks`: how many banks the instruction touches
- `theoretical_conflict_degree`: the largest number of distinct word addresses
  requested from any one bank

## Build

```bash
CUDA_ARCH=110 ./scripts/build.sh
```

## Run

Run everything:

```bash
./scripts/run_basic.sh
```

This writes `results/basic_results.csv` and then invokes `parse_results.py`,
which generates `results/avg_ms.png` and `results/effective_gbps.png`.

Run one experiment stage:

```bash
./scripts/run_basic.sh --case E0
./scripts/run_basic.sh --case E3
```

Run one exact case:

```bash
./build/transpose_2d_bench --case E2_load_conflict_same_bank_diff_addr --iters 100000
```

Useful environment overrides:

```bash
ITERS=1000 WARMUPS=3 REPEATS=10 ./scripts/run_basic.sh --case E1
```

## Nsight Compute

```bash
./scripts/run_ncu.sh
./scripts/run_ncu.sh --case E2
```

The script uses `--list-cases` to expand `all`, `E0`, `E1`, `E2`, `E3`, `E4`,
or a concrete case name into per-case NCU runs. After profiling, it invokes
`parse_ncu_results.py`, prints one table per collected metric, and writes one
PNG bar chart per metric into `results/ncu/`. Query metrics available on the
target machine with:

```bash
ncu --query-metrics | grep -Ei "bank|shared|l1tex|sass_inst_executed"
```

## Interpretation Notes

- This is still a microbenchmark. The accumulator dependency and loop body are
  there to preserve the loads, so `effective_GBps` is only a derived,
  benchmark-local rate.
- Vector-load cases span multiple words per instruction. Their bank behavior is
  better interpreted from the enumerated theoretical fields and Nsight Compute
  counters than from a scalar `gcd(pitch, 32)` mental model alone.
- Results in this directory should not be projected onto TMA, descriptor
  swizzle, `tcgen05`, or full GEMM behavior.
