# 05_ldshared_contention 分析

## 静态重新标定（主证据）

- CSV: `plots/static_ldshared_benchmark.csv`.
- 图: `plots/static_ldshared_extra.svg`.
- aggregate rows: 11 valid, 0 invalid.
- 固定控制变量：warp 0 发 MMA；warp 1-3 执行 interference。主比较中的 active warp count 固定。

BF16 N128 Q16, ops=32：

| Mode | cycles/MMA | Delta vs register ALU |
| --- | ---: | ---: |
| register ALU | 163.809 | 0.000 |
| predicated-off ld.shared | 163.882 | +0.073 |
| ld.shared | 163.714 | -0.095 |
| L1-hit ld.global | 171.210 | +7.401 |

MMA-only baseline 是 `86.727 cycles/MMA`。

## 推断

- 本次运行中 ld.shared 不比 register-ALU control 更慢。
- L1-hit global load 比 register ALU 和 ld.shared 更慢，因此观察到的 interference 更适合解释为 active-warp/scheduler/control pressure 和 general load/LSU pressure，而不是已证明的 shared-memory-specific port conflict。
- 这个实验不能证明普通 LSU `ld.shared` 与 tcgen05 operand ingress 完全或部分共享物理资源。

## 不支持的说法

- 只有当 shared-specific degradation 明显超过 register、predicated-off、L1-hit global 和 interference-only controls，并且随 shared address/bank pattern 系统变化时，才能声称 SMEM-port sharing。

## 旧 Runtime-Dispatch 负控制

旧 runtime-dispatch ld.shared sweep 保留用于审计，但它在 warp 0 内使用 lane-level divergence，因此不能隔离 port contention。
