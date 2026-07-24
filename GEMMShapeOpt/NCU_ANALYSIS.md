# GEMMShapeOpt NCU Analysis Plan and Metric Interpretation

## Current Status

Epilogue performance gates have been added for:

- `none`
- `bias`
- `relu`
- `gelu`
- `residual`

The gate is still `>= 0.90x` versus `cuBLASLt Matmul heuristic`, with
`cublas_tc` treated only as the reference backend. `shapeopt` is the
non-reference validation backend and currently routes to the same cuBLASLt
heuristic fallback used by `GEMMsm110`.

Verified shape-epilogue cases:

| Shape set | Cases | Result |
| --- | ---: | --- |
| `target_shapes.csv` | 45 | PASS |
| `smoke_shapes.csv` | 20 | PASS |
| `default_shapes.csv` | 70 | PASS |
| `core_shapes.csv` | 100 | PASS |
| `extended_shapes.csv` | 70 | PASS |

Result directories:

- `results/gemm_shape_opt/target_epilogue_90_gate_shapeopt_final/`
- `results/gemm_shape_opt/target_epilogue_90_gate_shapeopt_latest/`
- `results/gemm_shape_opt/smoke_epilogue_90_gate_shapeopt/`
- `results/gemm_shape_opt/default_epilogue_90_gate_shapeopt/`
- `results/gemm_shape_opt/core_epilogue_90_gate_shapeopt/`
- `results/gemm_shape_opt/extended_epilogue_90_gate_shapeopt/`

Latest line plots:

- `results/gemm_shape_opt/plots_epilogue/index.html`
- `results/gemm_shape_opt/plots_epilogue/target_epilogue_90_gate_shapeopt_latest_ratio.svg`
- `results/gemm_shape_opt/plots_epilogue/core_epilogue_90_gate_shapeopt_ratio.svg`
- `results/gemm_shape_opt/plots_epilogue/extended_epilogue_90_gate_shapeopt_ratio.svg`

## NCU Profiling Status

NCU profiling script:

```bash
GEMMShapeOpt/scripts/run_ncu_profiles.sh
```

Case list:

```bash
GEMMShapeOpt/profiles/ncu_cases.csv
```

Attempted output:

```bash
results/gemm_shape_opt/ncu/manual/ncu_status.csv
results/gemm_shape_opt/ncu/manual_latest/ncu_status.csv
```

The current run failed for every case with:

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

This is a system permission issue, not a benchmark or kernel failure. NCU can
launch the application, but cannot collect performance counters. After enabling
GPU performance counter access for this user, rerun:

```bash
OUT_DIR=results/gemm_shape_opt/ncu/manual \
GEMMShapeOpt/scripts/run_ncu_profiles.sh
```

or, to reproduce the latest path:

```bash
OUT_DIR=results/gemm_shape_opt/ncu/manual_latest \
NCU_BIN=/usr/local/cuda/bin/ncu \
GEMMShapeOpt/scripts/run_ncu_profiles.sh
```

## NCU Case Coverage

The NCU case list covers both requested scopes:

| Suite | Backend | Case intent |
| --- | --- | --- |
| `shapeopt` | `shapeopt` | GEMMShapeOpt router path, currently cuBLASLt fallback, including epilogue cases |
| `gemm_sm110` | `tc5a` | handwritten TCGen05 square and tail cases |
| `gemm_sm110` | `tc5b` | hybrid 2-SM path or tc5 fallback case |

Representative shapes include square, ragged, GEMV-like, the requested
`13x17x2048` micro GEMV-like case, residual epilogue, and M/N-tail cases.

## What Each Metric Tells Us

| Topic | NCU section / metric family | What it means | How to read it |
| --- | --- | --- | --- |
| Tensor Core utilization | `SpeedOfLight`, `InstructionStats`, tensor pipe metrics when available | Whether the kernel is issuing tensor-core work close to hardware capacity | High tensor throughput with high SM throughput means the mainloop is compute-bound and tensor cores are busy. Low tensor throughput with high memory/stall metrics means tensor cores are starved. |
| SM throughput | `SpeedOfLight`, `sm__throughput.*pct_of_peak*` | Overall SM pipeline utilization | If SM throughput is low while runtime is high, inspect stalls and occupancy. If SM throughput is high but ratio is low, the algorithm may do wasted work, especially tails. |
| Memory throughput | `SpeedOfLight`, `MemoryWorkloadAnalysis`, `sm__memory_throughput.*pct_of_peak*` | Pressure on memory hierarchy | High memory throughput plus low tensor utilization means memory-fed/epilogue-bound behavior. Low memory and low tensor throughput usually points to stalls, launch overhead, or poor occupancy. |
| Shared memory / SMEM | `MemoryWorkloadAnalysis`, shared-memory tables | Whether SMEM traffic or bank behavior limits issue rate | High shared throughput with `mio_throttle` or barrier stalls suggests SMEM/TMA/producer-consumer pressure. Low shared throughput with low tensor utilization suggests the kernel is not feeding tensor cores. |
| Warp stalls | `SchedulerStats`, `WarpStateStats`, `smsp__average_warps_issue_stalled_*` | Why eligible warps are not issuing | `long_scoreboard` often means waiting on memory dependencies; `mio_throttle` points to memory/SMEM instruction queue pressure; `barrier` points to synchronization; `math_pipe_throttle` points to math pipeline saturation; `not_selected` can be normal if there are enough eligible warps. |
| Occupancy | `Occupancy`, `sm__warps_active.*`, `smsp__warps_active.*` | How many warps are active versus the theoretical limit | Low achieved occupancy is not always bad for tensor cores, but if combined with latency stalls it means insufficient latency hiding. Compare theoretical occupancy to achieved occupancy. |
| Registers | `LaunchStats`, `launch__registers_per_thread`, `launch__occupancy_limit_registers` | Whether register count limits resident warps/blocks | High registers/thread can reduce occupancy. If `launch__occupancy_limit_registers` is the limiting factor and stalls are memory-latency-heavy, reducing registers may help. |
| Static/dynamic SMEM | `LaunchStats`, `launch__shared_mem_per_block*`, `launch__occupancy_limit_shared_mem` | Whether SMEM allocation limits resident CTAs | If SMEM is the occupancy limiter, smaller tile/stage count may improve latency hiding but can reduce data reuse. |
| Tail waste | `GEMMShapeOpt/scripts/compute_tail_waste.py` derived CSV | How much padded tile work is done beyond real M/N/K | This is not a direct NCU counter. It is derived from tile shape. High tail waste explains poor ratio even if tensor utilization looks high: the kernel may be efficiently doing useless padded work. |

## Metric To Conclusion Mapping

NCU metric names can vary slightly by CUDA/NCU release and GPU family. In the
raw CSV, search by the metric patterns below if the exact name differs.

| Question | Primary NCU metric / pattern | Conclusion when high | Conclusion when low |
| --- | --- | --- | --- |
| Are tensor cores busy? | `sm__pipe_tensor_throughput.*pct_of_peak*`, `*pipe_tensor*`, `*mma*`, `*wgmma*`, `*tcgen05*` | Tensor pipe is fed. If runtime is still bad, check tail waste and epilogue/memory cost. | Mainloop is starved or the kernel is not mapping work to tensor cores efficiently. Check stalls and memory. |
| Is overall SM issue healthy? | `sm__throughput.*pct_of_peak*`, `smsp__issue_active.*` | SM pipelines are active; remaining gap is likely algorithmic work waste or memory/epilogue mix. | Kernel has idle issue slots. Check eligible warps, occupancy, and stall reasons. |
| Is shared memory a bottleneck? | `l1tex__*mem_shared*`, shared-memory rows in `MemoryWorkloadAnalysis_Tables`, `*bank_conflict*` | SMEM/TMA movement or bank behavior may limit tensor-core feeding, especially with `mio_throttle`/barrier stalls. | SMEM is not the main limiter; look at global memory, occupancy, launch overhead, or tail waste. |
| Is global/cache bandwidth the bottleneck? | `dram__throughput.*pct_of_peak*`, `lts__throughput.*pct_of_peak*`, `sm__memory_throughput.*pct_of_peak*` | Kernel is memory/epilogue traffic sensitive. `residual` should raise read traffic; `bias` should add only small vector traffic. | Not bandwidth-bound; stalls or low parallelism are more likely. |
| Why are warps not issuing? | `smsp__average_warps_issue_stalled_*` | The largest stall reason names the next investigation: `long_scoreboard` memory dependency, `mio_throttle` memory/SMEM queue, `barrier` synchronization, `math_pipe_throttle` math pipe pressure. | If all stall reasons are low but throughput is low, inspect launch size and tail work. |
| Is occupancy enough to hide latency? | `sm__warps_active.*pct_of_peak*`, `smsp__warps_active.*`, NCU Occupancy section | Enough resident warps exist; do not optimize occupancy blindly if tensor utilization is already high. | If paired with memory stalls, latency hiding is insufficient. Check register and SMEM limits. |
| Are registers limiting occupancy? | `launch__registers_per_thread`, `launch__occupancy_limit_registers`, `launch__registers_per_block` | Register pressure may cap resident CTAs/warps; reducing fragments/stages may help latency-bound kernels. | Registers are unlikely to be the first limiter. |
| Is SMEM allocation limiting occupancy? | `launch__shared_mem_per_block*`, `launch__occupancy_limit_shared_mem` | Tile size or stage count is limiting active CTAs. Reduce only if stalls show more latency hiding is needed. | SMEM capacity is not the occupancy limiter. |
| Is tail work wasting tensor-core cycles? | `OutputWastePct`, `ComputeWastePct`, `Mtail`, `Ntail`, `Ktail` from tail-waste CSV | High compute waste means square tiling is a poor fit even if tensor utilization looks good. Use skinny/GEMV or tail-specific kernels. | Low waste means focus on mainloop, memory path, and epilogue. |

For this repo, use tail waste as a first-pass explanation before reading
counter data. The current `tc5a/tc5b` tile family is `M128N256K64`, so skinny
and GEMV-like shapes can report high tensor activity while doing mostly padded
work.

## Tail Waste Evidence

Generated files:

- `results/gemm_shape_opt/tail_waste/target_tail_waste.csv`
- `results/gemm_shape_opt/tail_waste/core_tail_waste.csv`

The default model assumes the current main square tile family:

```text
TileM=128, TileN=256, TileK=64
```

Important target examples:

| Shape | Output waste | Compute waste | Interpretation |
| --- | ---: | ---: | --- |
| `2048x2048x2048` | 0.00% | 0.00% | No tile-tail waste; performance should reflect mainloop efficiency. |
| `1024x1024x1000` | 0.00% | 2.34% | Mostly regular; only K tail cleanup matters. |
| `384x520x300` | 32.29% | 36.52% | M/N/K tail cleanup is material. |
| `4096x64x4096` | 75.00% | 75.00% | Skinny-N wastes most N tile lanes with M128N256 tile. |
| `64x4096x4096` | 50.00% | 50.00% | Skinny-M wastes half of M tile lanes. |
| `1x4096x4096` | 99.22% | 99.22% | GEMV-like decode should not use square GEMM tiling. |
| `13x17x2048` | 99.33% | 99.33% | Micro GEMV-like shape is dominated by padded work under square tiling. |

## How To Draw Conclusions Once NCU Counters Are Enabled

1. Start with `RatioToReference`.
   If ratio is already `>= 0.90`, use NCU to decide whether to replace
   cuBLASLt fallback with a handwritten path. If ratio is below 0.90, inspect
   the bottleneck first.

2. Check tail waste before reading low-level counters.
   A high tensor-core utilization number can be misleading on skinny/GEMV-like
   shapes because tensor cores may be busy computing padded tiles. For these
   cases, `ComputeWastePct` is the first-order explanation.

3. Use Tensor Core and SM throughput together.
   High tensor and high SM throughput means mainloop is efficient. Low tensor
   throughput with high memory or stall metrics means the tensor core pipe is
   starved.

4. Use stall reasons to choose the next optimization.
   Barrier stalls suggest pipeline synchronization or producer/consumer
   imbalance. MIO throttle suggests memory or SMEM instruction pressure. Long
   scoreboard suggests memory dependency latency. Low eligible warps suggests
   occupancy or dependency-chain issues.

5. Use occupancy and register data as constraints, not goals.
   Higher occupancy is useful only if it hides stalls. If tensor utilization is
   already high, reducing registers just to increase occupancy may not improve
   runtime.

6. For epilogues, compare `none` versus `bias/relu/gelu/residual`.
   If the mainloop metrics stay similar but memory throughput or stall metrics
   rise, the epilogue is the bottleneck. `residual` should show extra C input
   read traffic; `bias` should show a small vector read; `gelu` may increase
   scalar/math instruction pressure.

## Current Limitation

This document includes the profiling plan and metric interpretation. Actual NCU
counter conclusions cannot be filled in on this machine until
`ERR_NVGPUCTRPERM` is resolved. The failed NCU logs are intentionally kept under
`results/gemm_shape_opt/ncu/manual/` as evidence of the blocker.
