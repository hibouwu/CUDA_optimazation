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

### E0 Basic Pitch Effect

- `E0_load_pitch32`: `tile[lane * 32 + warp]`. Classic worst case. Every lane in
  a warp lands on the same bank.
- `E0_load_pitch33`: `tile[lane * 33 + warp]`. `+1` padding rotates lanes across
  all banks.

### E1 Pitch Sweep

Cases:

- `E1_load_pitch1`
- `E1_load_pitch2`
- `E1_load_pitch4`
- `E1_load_pitch8`
- `E1_load_pitch16`
- `E1_load_pitch31`
- `E1_load_pitch32`
- `E1_load_pitch33`
- `E1_load_pitch34`
- `E1_load_pitch35`
- `E1_load_pitch36`
- `E1_load_pitch40`
- `E1_load_pitch64`

These are the systematic check of the `gcd(pitch, 32)` rule.
For very small pitches such as 1, 2, and 4, the logical rows intentionally
overlap in the backing array. That is acceptable here because E1 is about the
shared-memory bank mapping implied by `lane * pitch + warp`, not about modeling
a production transpose tile shape.

### E2 Broadcast And Multicast Controls

- `E2_load_broadcast_same_addr`: all lanes read `tile[0]`.
- `E2_load_multicast_2addr`: lanes 0-15 read `tile[0]`, lanes 16-31 read `tile[1]`.
- `E2_load_multicast_4addr`: each eight-lane group reads one of `tile[0..3]`.
- `E2_load_conflict_same_bank_diff_addr`: lanes read `tile[lane * 32]`.

This stage separates two concepts that are often conflated:

- Same bank plus same address can be served as broadcast or multicast.
- Same bank plus different addresses is the ordinary bank-conflict situation.

### E3 Vector Width

- `E3_load_f32_pitch32`
- `E3_load_f32_pitch33`
- `E3_load_f32x2_pitch32`
- `E3_load_f32x2_pitch33`
- `E3_load_f32x4_pitch32`
- `E3_load_f32x4_pitch33`

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

- `E4_load_xor_swizzle_pitch32`

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

Each row includes:

- `experiment`
- `case`
- `operation`
- `pitch`
- `vector_width`
- `theoretical_unique_banks`
- `theoretical_conflict_degree`
- `iters`
- `avg_ms`
- `min_ms`
- `effective_GBps`

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
or a concrete case name into per-case NCU runs. Query metrics available on the
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
