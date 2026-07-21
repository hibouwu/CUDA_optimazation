# 03_effective_smem_ingress 分析

## 静态重新标定（主证据）

- CSV: `plots/static_ingress_benchmark.csv`.
- 图: `plots/static_ingress_address.svg`.
- aggregate rows: 12 valid, 0 invalid.
- 固定控制变量：static kernel、Q16、合法 D-ring、`input_d=1`、collector discard、wait_hint=0、full-grid launch blocks。

BF16 address-mode 观察：

| Shape | same | pingpong | rotating | Spread |
| --- | ---: | ---: | ---: | ---: |
| N128 | 86.653 | 86.784 | 86.728 | 0.132 cycles/MMA |
| N256 | 154.810 | 154.752 | 154.785 | 0.059 cycles/MMA |

使用静态 `02_latency_throughput` 的 fitted beta，logical operand bytes/beta 对 N128 约为 `128.5 B/cycle`，对 N256 约为 `95.0 B/cycle`。

## 推断

- same、pingpong 和 rotating operand address 在当前静态 Q16 窗口中没有显著改变 cycles/MMA。
- visible logical service-rate 计算对软件模型有用，但不是物理 SMEM-to-Tensor-Core port width。
- N128/N256 scaling 与测得的 MMA completion envelope 一致；它没有隔离出纯 operand-ingress bottleneck。

## 不支持的说法

- 这些 row 不能识别物理 port width 或 SMEM bank count。
- 不能用 logical bytes 除以受污染 runtime-dispatch cycles 来计算物理带宽。

## 旧 Runtime-Dispatch 负控制

旧 `effective_smem_bytes_per_cycle` 的 `6.935-13.875 B/cycle` 来自受污染 timed loop，不用于物理推断。
