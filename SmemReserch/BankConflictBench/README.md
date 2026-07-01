# Shared-memory bank-conflict microbenchmarks

This directory contains a standalone CUDA microbenchmark for ordinary
`ld.shared` instructions. It uses one block of `dim3(32, 8, 1)`: `threadIdx.x`
is the lane and `threadIdx.y` is the warp.

## Cases

For 32-bit words, the simplified mapping is:

```text
linear_index = row * pitch + col
bank = linear_index % 32
```

| CLI case | Per-lane mapping within each warp | Purpose |
|---|---|---|
| `baseline` | `s[warp][lane]` | 32 distinct banks |
| `stride` | `s[lane * stride + offset]` | stride 1/2/4/8/16/32 gives nominal 1/2/4/8/16/32-way conflicts |
| `same_bank_32way_2d` | `s[lane][0]`, pitch 32 | 32 distinct words in bank 0 |
| `broadcast` | `s[warp][0]` | all lanes load the same word |
| `multicast_hash` | `s[warp][hash(lane) % 32]` | subsets of lanes may share addresses |
| `v4_contiguous` | `s128[warp][lane * 4 + 0..3]` | one volatile `ld.shared.v4.f32` per lane |
| `v2_multicast_pairs` | `s32[warp][(lane / 2) * 2 + 0..1]` | each lane pair loads the same `float2` |
| `v4_multicast_quads` | `s32[warp][(lane / 4) * 4 + 0..3]` | each four-lane group loads the same `float4` |

The two-dimensional form is only an explanatory view. Bank selection always
uses the flattened linear index. In particular,
`s[lane][0]` with `pitch=32` gives
`index = lane * 32 + 0`, hence `bank = (lane * 32) % 32 = 0`.

A bank conflict is a property of addresses requested by one shared-memory
instruction in one warp. Activity from different warps is better described as
shared-memory contention or throughput pressure; it is not part of the
same-warp bank-conflict definition.

Broadcast and multicast requests have repeated addresses. Hardware can serve
same-address requests differently from distinct-word requests that map to the
same bank, so they should not be interpreted as ordinary N-way conflicts.

These experiments cover conventional `ld.shared` instructions on the LSU path.
Do not directly extrapolate their results to TMA, descriptor-based operations,
or the `tcgen05` MMA path.

## Build

From `BankConflictBench`:

```bash
./build.sh
CUDA_ARCH=80 ./build.sh
```

The environment value is passed to `CMAKE_CUDA_ARCHITECTURES`; values such as
80, 90, 100, and 110 are accepted when supported by the installed toolkit.
Equivalent direct use is:

```bash
cmake -S ../src -B ../build -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build ../build --parallel
```

## Run

```bash
../build/smem_bank_bench --case baseline --iters 100000
../build/smem_bank_bench --case stride --stride 8 --offset 0 --iters 100000
../build/smem_bank_bench --case all --iters 100000

./run_basic.sh
./parse_results.py
```

`run_basic.sh` writes `results/basic_results.csv` and then invokes
`parse_results.py` to print summary tables plus PNG charts, including
`all_cases_avg_ms_bar.png` and `all_cases_effective_gbps_bar.png`.
Set `ITERS` to shorten or lengthen a run, for example `ITERS=1000 ./run_basic.sh`.
Set `WARMUPS` and `REPEATS` to change the measurement loop, for example
`WARMUPS=10 REPEATS=50 ./run_basic.sh`. Direct invocation also accepts
`--warmups N --repeats N`.
By default each case has five warmups and twenty timed repetitions.
`effective_GBps` counts requested bytes and uses average elapsed time; it is a
microbenchmark-derived effective rate, not necessarily physical shared-memory
traffic.

`run_ncu.sh` now builds the benchmark, profiles every case and each stride
separately into `results/ncu/`, and then invokes `parse_ncu_results.py` to
print metric tables and generate one PNG bar chart per collected metric.
It defaults to zero warmups and one measured launch because profiler replay
already collects the hardware counters; set `WARMUPS` or `REPEATS` only when
additional launches are intentional.
Metric availability varies by architecture and Nsight Compute release. Override
the comma-separated `METRICS` environment variable when needed. If a metric is
rejected, inspect available names with:

```bash
ncu --query-metrics | grep -Ei "bank|shared|l1tex"
```

If the running NVIDIA driver restricts GPU performance counters to
administrators, `run_ncu.sh` automatically uses passwordless `sudo` for `ncu`
and restores ownership of the generated CSV files. This machine also has
`NVreg_RestrictProfilingToAdminUsers=0` configured for the next driver reload,
so `sudo` will no longer be needed after a reboot.

The benchmark uses volatile inline PTX (`ld.volatile.shared.f32`,
`ld.volatile.shared.v2.f32`, and `ld.volatile.shared.v4.f32`) and writes each
thread's accumulator to global memory to preserve the measured loads.
