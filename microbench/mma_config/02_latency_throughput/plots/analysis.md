# 02_latency_throughput 分析

## 静态重新标定（主证据）

- 源码: `benchmark_src/tcgen05_02_static_calibration_bench.cu`.
- CSV: `plots/static_calibration_benchmark.csv`.
- SASS dump: `plots/static_sass/`.
- aggregate rows: 576 valid, 0 invalid.
- BF16 N128 Q4 full-grid same-D `input_d=0` case 为 `145.581 cycles/MMA` 和 `113.44 clock64 full-grid TFLOP/s`，匹配可信 BF16 K4 mainloop reference `~146.132 cycles/MMA`。

BF16 full-grid、wait_hint=0、same-D、`input_d=0`：

| Shape | Q1 | Q2 | Q4 | Q8 | Q16 | Q32 | Q64 | fitted beta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| N128 | 450.270 | 247.145 | 145.581 | 102.079 | 86.768 | 75.333 | 68.937 | 63.747 |
| N256 | 450.262 | 308.800 | 212.660 | 181.482 | 154.818 | 141.364 | 135.092 | 129.381 |

BF16 N128 full-grid 的 completion controls：

| Control | 观察 |
| --- | ---: |
| empty Q64 register-control loop | ~264 cycles/iteration |
| commit + already-completed mbarrier wait | ~258 cycles/iteration |
| forced single-MMA wait, hint=0 | ~450 cycles/MMA |
| forced single-MMA wait, hint=32 or 0x989680 | ~431 cycles/MMA |
| CTA-wide `__syncthreads()` | 20.828 cycles/sync |

## 推断

- 旧 runtime-dispatch beta 是测量污染；静态 Q4 gate 与已知 K4 baseline 一致。
- Q1 是 forced-completion diagnostic。long-batch fitted beta 才是本 harness 中有用的稳态边际成本。
- `tcgen05.commit` 被视为执行线程发出的所有先前 async tcgen05 操作的累计 completion-prefix tracking，不是独立 async group queue。
- `clock64_full_grid_tflops` 和 `event_wall_tflops` 是不同指标。event wall time 包含 setup/readback，不用于 timed-window beta。

## 不支持的说法

- 这些数据不能识别物理 SMEM port width、SMEM bank count、TMEM bank width/count、hidden collector depth 或 hidden async group queue depth。

## 旧 Runtime-Dispatch 负控制

旧 `benchmark_results.csv` 保留用于审计，但其 `752.564-874.469 cycles/MMA` fitted beta 不用于硬件推断，因为 timed loop 中混入 runtime dispatch、address arithmetic、descriptor/control work、wait 和 CTA synchronization。
