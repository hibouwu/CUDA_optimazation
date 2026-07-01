# One-dimensional `st.shared` benchmark

This benchmark isolates ordinary LSU-path volatile shared-memory stores. One
block uses `dim3(32, 8)`, with `threadIdx.x` as lane and `threadIdx.y` as warp.

For scalar stores:

```text
linear_index = lane * stride + offset
bank = linear_index % 32
```

| Case | Mapping | Intent |
|---|---|---|
| `baseline` | `s[warp][lane]` | One scalar word per bank |
| `stride` | `s[warp][lane * stride + offset]` | 1/2/4/8/16/32-way destination-bank sweep |
| `same_bank_32way_2d` | equivalent to `s[lane][0]`, pitch 32 | Distinct words in one bank |
| `same_address` | every lane in a warp stores the same bits to one word | Same-address diagnostic; not a load-style broadcast |
| `v2_contiguous` / `v4_contiguous` | one contiguous vector per lane | Vector store throughput |
| `v2_multicast_pairs` / `v4_multicast_quads` | lane groups store identical vectors to the same destination | Same-destination vector diagnostic |

Same-address stores must not be described as multicast or broadcast: those are
load-service behaviors. Multiple writers are retained only as a hardware
diagnostic and may be reported as races by sanitizers.

Build and run from this directory:

```bash
CUDA_ARCH=110 ./scripts/build.sh
./scripts/run_basic.sh
./scripts/run_ncu.sh
```

`run_basic.sh` writes `results/basic_results.csv`. `run_ncu.sh` uses editable
candidate metrics through `METRICS=...`; query the Thor installation before
drawing conclusions from a metric name.

