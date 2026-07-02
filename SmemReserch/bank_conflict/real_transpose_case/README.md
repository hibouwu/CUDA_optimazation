# Real matrix transpose: coalescing, tiling, and bank conflicts

This directory measures complete CUDA matrix-transpose kernels: data is read
from global memory, transposed, and written back to global memory. It studies
the end-to-end path from global-memory coalescing to shared-memory tiling and
bank-conflict removal.

This is intentionally different from
[`transpose_2d_case`](../transpose_2d_case/README.md), which is a load-only
`ld.shared` microbenchmark. This directory does not contain TMA, TMA descriptor
swizzle, `tcgen05`, TMEM, CUTLASS, or CUTE experiments.

## Relationship to the reference implementation

The optimization sequence follows the algorithmic intent of the
[reference blog](https://www.wingedge777.com/en/article/49c4e15376366f8d),
not its code line by line. R0 represents the two one-sided coalescing choices,
R1 introduces the shared-memory corner turn, R2 adds padding, and R3 adds
128-bit global-memory accesses while keeping shared memory scalar. R4 isolates
the scalar XOR-swizzled layout; unlike the blog's final packed-swizzle example,
this backend does not claim vectorized global or shared-memory accesses. R5 is
an additional local copy reference rather than a transpose optimization from
the blog.

## What is being separated

Global-memory coalescing and shared-memory bank conflicts are different
hardware effects:

- Coalescing combines a warp's global-memory accesses into as few memory
  transactions as possible. A transpose written directly in input order has
  contiguous loads but output stores separated by `height * sizeof(float)`.
- Shared-memory tiling first loads a contiguous 32x32 block and then exchanges
  the block's row/column roles. This makes both global sides contiguous.
- With `tile[32][32]`, the transpose-side shared-memory read addresses are
  separated by 32 words. They map to the same bank and introduce a 32-way
  conflict.
- `tile[32][33]` changes the bank mapping without changing the logical tile.
  XOR swizzle instead permutes the physical column while retaining a 32x32
  allocation.

All timed cases use CUDA events. Correctness is checked after timing with exact
float equality:

```text
output[col * height + row] == input[row * width + col]
```

Effective bandwidth counts one input read and one output write:

```text
bytes = width * height * sizeof(float) * 2
effective_GBps = bytes / avg_seconds / 1e9
```

## Cases

| Experiment | Case | Global-memory behavior | Shared-memory behavior |
| --- | --- | --- | --- |
| R0 | `R0_transpose_naive` | Linear input traversal: coalesced loads, strided stores | None |
| R0 | `R0_transpose_coalesced_read` | Explicit 32x32 traversal: coalesced loads, strided stores | None |
| R0 | `R0_transpose_coalesced_write` | Output-order traversal: strided loads, coalesced stores | None |
| R1 | `R1_transpose_smem_pitch32` | Coalesced loads and stores | `float tile[32][32]`; transpose read is bank-conflicted |
| R2 | `R2_transpose_smem_pitch33` | Coalesced loads and stores | `float tile[32][33]`; padding removes the column conflict |
| R3 | `R3_transpose_smem_packed_pitch33` | 128-bit `float4` loads and stores when dimensions permit | Padded tile; shared-memory accesses remain scalar |
| R4 | `R4_transpose_smem_xor_swizzle` | Coalesced loads and stores | 32x32 tile with consistent XOR mapping |
| R5 | `R5_transpose_copy_baseline` | Coalesced copy, no transpose | None; bandwidth reference, not a strict theoretical upper bound |

The two coalesced-read R0 cases intentionally have the same fundamental memory
transaction pattern. The naive case uses a one-dimensional input-order launch;
the named coalesced-read case uses the same 32x8/32x32 traversal as the tiled
cases, making comparisons easier.

### Padding versus software swizzle

Padding changes each shared row's stride from 32 to 33 words. It is simple but
adds one column of storage. XOR swizzle keeps a 32x32 allocation and maps
`physical_col = logical_col ^ logical_row`. Both the write and transpose read
must apply that mapping; applying it on only one side produces wrong output.
For a fixed warp-local column, padding changes the transpose-read bank mapping
from `(lane * 32 + column) % 32` to
`(lane * 33 + column) % 32`, distributing the lanes across all 32 banks.

### Vector alignment

R3 vectorizes only global memory. A pitch of 33 floats means consecutive shared
rows start 132 bytes apart, so row starts are not consistently 16-byte aligned.
Forcing shared `float4` instructions would therefore be invalid or would make
the layout substantially more complex.

`cudaMalloc` supplies adequately aligned base pointers. R3 uses global
`float4` accesses when both width and height are divisible by four, ensuring
the input and output row strides are 16-byte compatible. Otherwise it falls
back to scalar global accesses and records `vector_width=1` in the CSV.
Boundary tiles that are not full 32x32 are handled correctly.

### Current SM110 observation

In current SM110 runs, R2, R3, and R4 generally remain within a few percent of
one another, and their exact times move with clocks and system load. The packed
and swizzled variants therefore do not demonstrate a stable, meaningful speedup
over padding in this implementation; their purpose here is to compare access
strategies. The optimization ordering and speedups reported by the reference
blog are motivation, not performance claims for this benchmark or GPU.

R5 moves the same logical byte count with coalesced loads and stores, but it
does not transpose and has different address arithmetic and kernel behavior.
It is a useful bandwidth reference, not a guaranteed upper bound for every
transpose backend.

## Build and run

From this directory:

```bash
CUDA_ARCH=110 ./scripts/build.sh
./scripts/run_basic.sh
./scripts/run_basic.sh --case R0 --width 4096 --height 4096
./scripts/run_basic.sh --case R3 --width 8192 --height 8192 \
  --iters 20 --warmups 3 --repeats 10
./scripts/run_basic.sh --case R2_transpose_smem_pitch33
```

`run_basic.sh` writes `results/basic_results.csv`, prints a summary table, and
generates:

- `results/avg_ms.png`
- `results/effective_gbps.png`

The CSV schema is:

| Field | Meaning |
| --- | --- |
| `experiment`, `case` | Experiment stage and exact backend name |
| `width`, `height`, `dtype` | Input matrix description |
| `tile_dim`, `block_rows` | Kernel traversal geometry; zero for untiled baselines |
| `smem_pitch` | Shared row pitch in floats; zero when shared memory is unused |
| `vector_width` | Actual scalar elements per global memory instruction path |
| `swizzle` | `none` or `xor` |
| `avg_ms`, `min_ms` | Per-kernel time across CUDA-event repeats |
| `effective_GBps` | Requested input plus output bytes divided by average time |
| `correctness` | `PASS` or `FAIL` from exact transpose/copy validation |

Defaults are 4096x4096, 10 timed iterations, 2 warmups, and 10 repeats. The
CLI accepts `--case all`, stage selectors `R0` through `R5`, exact case names,
`--width`, `--height`, `--iters`, `--warmups`, and `--repeats`.

## Nsight Compute

Run all cases or one stage:

```bash
./scripts/run_ncu.sh
./scripts/run_ncu.sh --case R1 --width 4096 --height 4096
```

The script expands stage selectors through `--list-cases`, profiles every case
separately, stores CSV files in `results/ncu/`, and generates one bar chart per
available metric. Each bar is annotated with its metric value. If performance
counters are admin-only, the script automatically uses passwordless `sudo` and
restores CSV ownership to the current user. Default metrics cover:

- global load/store sector counts;
- global load/store request counts;
- shared load/store bank conflicts.

Metric availability and exact names vary by GPU and Nsight Compute release.
If Thor rejects one, query the installed profiler:

```bash
ncu --query-metrics | grep -Ei 'global|sector|request|bank_conflict'
```

For coalescing, compare sectors per global request rather than a sector count
alone: all cases move the same logical bytes, but a strided warp access needs
more sectors. For shared-memory behavior, compare R1 with R2 and R4.

To verify R3 code generation:

```bash
cuobjdump --dump-sass build/real_transpose_bench |
  grep -E 'LDG|STG|128|LDS|STS'
```

Look for 128-bit global loads/stores in R3. Do not expect vectorized shared
loads/stores: they are deliberately scalar because pitch 33 does not preserve
16-byte row alignment.
