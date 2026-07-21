# tcgen05 MMA 硬件路径标定报告

## 2026-07-20 静态重新标定

之前 `mma_config` 的 runtime-dispatch 测量结果仅保留为负控制，不再用于推断
tcgen05 MMA 本体成本、SMEM operand ingress 带宽、collector 深度、TMEM
dependency 或 ld.shared contention。这些旧 kernel 在 timed loop 内混入了
descriptor 构造、operand slot 选择、D-ring 地址计算、collector protocol 分派
和 CTA 同步。

当前主要证据来自每个 case 一个独立编译 binary 的静态测量：

- dtype、N、Q、D mode、`input_d`、operand address mode、collector protocol、
  wait hint、active-block request 和 interference mode 都是编译期常量。
- descriptor 构造、SMEM 地址设置、TMEM allocation/init/guard 和 TMA prefill
  都在 timed loop 之外。
- timed mode 包括 `empty`、`commit_wait`、`forced_wait`、`batch` 和
  `cta_sync`。
- 每个 single-case binary 都 dump SASS。关键 BF16 N128 Q4 校准 binary 含 4
  条 `UTCHMMA` 指令，Q16 collector binary 含 16 条 `UTCHMMA` 指令，不再是旧
  runtime-dispatch binary 中大量混杂 MMA 指令的形态。
- `clock64` block cycles 是校准用 timed-window 指标。CUDA event wall time
  只作为端到端交叉检查，因为它包含 setup/readback。

## 环境

| 字段 | 值 |
| --- | --- |
| GPU | NVIDIA Thor |
| Compute capability | 11.0 |
| Driver | 580.00 |
| CUDA toolkit/PTX | CUDA 13.0, PTX ISA 9.3 |
| clock64 throughput 使用的 SM clock | 1575.000 MHz，来自 sysfs/default |
| Memory clock | `nvidia-smi` 不可用 |
| Temperature / power | `nvidia-smi` 不可用 |
| 静态 full-grid launch blocks | 20 |

静态 CSV 中的 telemetry audit 状态为
`mem_clock_mhz:unavailable;temperature_c:unavailable;power_w:unavailable;sm_clock_mhz:sysfs_or_default`。

## 产物

| 实验 | 主要源码 | CSV | 图 / SASS | 分析 |
| --- | --- | --- | --- | --- |
| 静态校准 | `02_latency_throughput/benchmark_src/tcgen05_02_static_calibration_bench.cu` | `02_latency_throughput/plots/static_calibration_benchmark.csv` | `02_latency_throughput/plots/static_calibration_q_sweep.svg`, `02_latency_throughput/plots/static_sass/` | `02_latency_throughput/plots/static_calibration_analysis.md` |
| Collector protocol | 同一静态源码 | `01_collector_protocol/plots/static_collector_benchmark.csv` | `01_collector_protocol/plots/static_collector_protocol.svg`, `02_latency_throughput/plots/static_sass/` | `01_collector_protocol/plots/analysis.md` |
| SMEM ingress | 同一静态源码 | `03_effective_smem_ingress/plots/static_ingress_benchmark.csv` | `03_effective_smem_ingress/plots/static_ingress_address.svg` | `03_effective_smem_ingress/plots/analysis.md` |
| ld.shared contention | 同一静态源码 | `05_ldshared_contention/plots/static_ldshared_benchmark.csv` | `05_ldshared_contention/plots/static_ldshared_extra.svg` | `05_ldshared_contention/plots/analysis.md` |
| TMEM dependency | 同一静态源码 | `06_tmem_dependency/plots/static_tmem_benchmark.csv` | `06_tmem_dependency/plots/static_tmem_dependency.svg` | `06_tmem_dependency/plots/analysis.md` |
| CTA sync control | 同一静态源码 | `02_latency_throughput/plots/static_sync_control.csv` | CSV 中记录 SASS hash | 本报告 |

raw CSV 和 invalid CSV 与各 benchmark CSV 放在同一目录。静态 aggregate CSV
包含 p10、p50、p90、变异系数、SASS hash/count、`clock64_full_grid_tflops`
和 `event_wall_tflops`。

## 复现命令

```bash
python3 microbench/mma_config/scripts/run_all.py --static --static-matrix all --quick --repeats 3
python3 microbench/mma_config/02_latency_throughput/scripts/run_static_calibration.py --matrix calibration --repeats 3
python3 microbench/mma_config/02_latency_throughput/scripts/run_static_calibration.py --matrix collector --quick --repeats 3
python3 microbench/mma_config/02_latency_throughput/scripts/run_static_calibration.py --matrix ingress --quick --repeats 3
python3 microbench/mma_config/02_latency_throughput/scripts/run_static_calibration.py --matrix ldshared --quick --repeats 3
python3 microbench/mma_config/02_latency_throughput/scripts/run_static_calibration.py --matrix tmem --quick --repeats 3
cuobjdump --dump-sass microbench/mma_config/02_latency_throughput/build/static_calibration/<case>/bench
```

calibration 矩阵有 576 个 valid aggregate row，0 个 invalid row。collector、
ingress、ld.shared 和 TMEM quick 矩阵分别有 32、12、11 和 8 个 valid
aggregate row，也都是 0 invalid row。

`00_validation` 仍保留 96 个 legacy invalid validation row，全部是
`smem_layout=none` / `swizzle=none`。这些 case 被分类为“可执行但数值错误”的
descriptor/layout 组合：CUDA launch、wait 和 TMEM guard 检查完成，但 D 值超出
tolerance。它们不属于 validated performance configuration。

## 观察

静态 BF16 M128N128 Q4 full-grid case 测得 `145.581 cycles/MMA` 和
`113.44 clock64 full-grid TFLOP/s`。它与已有可信的 `mma_with_cp` BF16
M128N128 K4 mainloop 参考值 `~146.132 cycles/MMA` 匹配，因此作为接受静态
harness 的校准门槛。

BF16、full-grid、wait_hint=0、same-D、`input_d=0`：

| Shape | Q1 | Q2 | Q4 | Q8 | Q16 | Q32 | Q64 | fitted beta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M128N128K16 | 450.270 | 247.145 | 145.581 | 102.079 | 86.768 | 75.333 | 68.937 | 63.747 |
| M128N256K16 | 450.262 | 308.800 | 212.660 | 181.482 | 154.818 | 141.364 | 135.092 | 129.381 |

fitted beta 在 BF16/FP16、same-D `input_d=0`、same-D `input_d=1` 和合法
D-ring mode 之间保持稳定。BF16 N128 的三个 beta 分别是 `63.747`、`63.749`
和 `64.631 cycles/MMA`。BF16 N256 的三个 beta 分别是 `129.381`、
`129.384` 和 `129.354 cycles/MMA`。

BF16 N128 full-grid 的 completion control：

| Control | 观察 |
| --- | ---: |
| empty Q64 register-control loop | ~264 cycles/iteration |
| `tcgen05.commit` + already-completed mbarrier wait | ~258 cycles/iteration |
| one forced-wait MMA, wait_hint=0 | ~450 cycles/MMA |
| one forced-wait MMA, wait_hint=32 or 0x989680 | ~431 cycles/MMA |
| one CTA-wide `__syncthreads()` | 20.828 cycles/sync |

wait hint 对短 batch 的影响大于长 batch。BF16 N128 Q4 same-D full-grid 在
wait hint 0、32 和 0x989680 下分别是 `145.581`、`157.677` 和
`159.803 cycles/MMA`。Q64 下对应为 `68.937`、`69.512` 和
`70.185 cycles/MMA`。

collector protocol 在固定 Q16、合法 D-ring、`input_d=1` 和 wait_hint=0 下重跑。
BF16 N128 discard 是 `86.752-86.778 cycles/MMA`；成对 fill/lastuse 是
`85.471-85.506`；fill/use/lastuse 或 fill/use/discard 是 `89.683-89.777`。
BF16 N256 discard 是 `154.707-154.718`；成对 fill/lastuse 是
`150.838-150.867`；fill/use/lastuse 或 fill/use/discard 是
`148.616-148.654 cycles/MMA`。

SMEM ingress address mode 在固定 Q16、D-ring、`input_d=1` 和 collector
discard 下测试。BF16 N128 的 same/pingpong/rotating 差异在
`0.132 cycles/MMA` 内；BF16 N256 差异在 `0.059 cycles/MMA` 内。用 fitted
beta 计算 logical operand bytes/beta，N128 约为 `128.5 B/cycle`，N256 约为
`95.0 B/cycle`。这是软件可见的 logical service-rate 计算，不是物理端口宽度。

ld.shared contention 实验中 warp 0 发 MMA，warp 1-3 做干扰。ops=32 时，
register ALU 是 `163.809 cycles/MMA`，predicated-off ld.shared 是 `163.882`，
ld.shared 是 `163.714`，L1-hit global load 是 `171.210`。相对 register ALU，
ld.shared 在本次运行中没有更慢，差值为 `-0.095 cycles/MMA`；L1-hit global
load 则是 `+7.401 cycles/MMA`。

TMEM dependency quick test 在当前 Q16 静态窗口中没有看到 same-D 与合法
ring-D、`input_d=0` 与 `input_d=1` 的可测 penalty。BF16 N128 为
`86.721-86.799 cycles/MMA`；BF16 N256 为 `154.746-154.828 cycles/MMA`。

## 数值来源与推导

CUDA kernel 在每个 launched CTA 的 timed region 开始和结束记录 `clock64()`。
host 端把所有 CTA 中最大的 cycle count 作为 `elapsed_cycles`。使用最大值的原因
是 full-grid 完成时间受最慢 CTA 限制，而 descriptor setup、TMA prefill、TMEM
allocation/init/guard、readback 和 deallocation 都在 timed interval 外。

batch MMA row 的计算口径：

```text
cycles_per_mma = median_over_repeats(elapsed_cycles / (Q * iterations))
```

例如 BF16 N128 Q4 full-grid gate 的 p50 `cycles_per_mma = 145.581`，来自 p50
timed cycles 除以每 CTA 的 `4 * 500` 条 MMA。这个 row 与已有可信 BF16 N128 K4
mainloop 对比，是因为两者的主计时窗口都执行 4 条 K16 tcgen05 MMA atom；数值接近
说明静态 harness 可以作为主标定路径。

control row 的口径：

- `empty` 报告 `elapsed_cycles / iterations`；它是不可被编译器消掉的
  register/control loop，但它与 MMA loop 的指令不完全一致，所以不把它简单相减
  称为“纯 overhead”。
- `commit_wait` 报告 `elapsed_cycles / iterations`；每轮执行一次
  `tcgen05.commit` 加一个已经完成的 wait，用来隔离可见 completion-boundary
  成本。
- `forced_wait` 报告 `elapsed_cycles / (Q * iterations)`；报告中的 control
  使用 Q=1，因此每条 MMA 后立刻 commit/wait。它是 forced-completion diagnostic，
  不是稳态 MMA issue 成本。
- `cta_sync` 报告 `elapsed_cycles / iterations`；`20.828 cycles/sync` 来自
  `static_sync_control.csv` 中的 `20828 / 1000`。

fitted beta 使用 batch time per iteration 做普通最小二乘：

```text
T(Q) = cycles_per_mma(Q) * Q
T(Q) = alpha + beta * Q
```

报告中的 beta 只使用 Q>=2，因为 Q=1 被 forced completion 主导。`beta` 因此是
当前 harness 中 long-batch 的边际成本；`alpha` 吸收 loop、commit、final drain
和 wait boundary 成本。这也是报告把 beta 用于稳态比较、把 Q1 单独保留为 latency
diagnostic 的原因。

throughput 字段的公式：

```text
flops_per_mma = 2 * 128 * N * 16
clock64_full_grid_tflops =
  flops_per_mma * Q * iterations * launch_blocks /
  (elapsed_cycles / sm_clock_hz) / 1e12

event_wall_tflops =
  flops_per_mma * Q * iterations * launch_blocks /
  (event_ms / 1e3) / 1e12
```

`clock64_full_grid_tflops` 是校准用 timed-window 指标。CUDA event wall
throughput 更低，是因为它包含 launch、setup、prefill、TMEM readback 和 cleanup；
这里只作为端到端交叉检查。

p10、p50、p90 和变异系数都按同一 `case_id` 的 raw repeat rows 计算。row 只有在
CUDA status OK、`guard_ok=1`、SASS dump 成功且 `max_abs_error` 在 dtype
tolerance 内时才是 valid。SASS count 是对 `cuobjdump --dump-sass` 输出做字符串
计数，用来确认 single-case binary 里目标指令序列符合预期，而不是旧
runtime-dispatch 的混合序列。

collector range 是对同一 dtype/shape/protocol 下 same 和 pingpong operand-address
mode 取 min/max。静态 runner 为不同 slot 使用不同 A/B operand 数据，并使用
protocol-aware CPU reference。`fill_lastuse` 的指令序列是成对 `fill,lastuse`；
`fill_use_lastuse` 是 `fill,use...,lastuse`；`fill_use_discard` 的最后一条是
`discard`，并消费当前 descriptor。因此这些 row 能检查是否消费了预期 collector
slot。

SMEM ingress logical bytes/cycle 只从 collector-discard 静态 row 计算：

```text
logical_operand_bytes_per_mma = sizeof(dtype) * K * (M + N)
logical_bytes_per_cycle = logical_operand_bytes_per_mma / fitted_beta
```

BF16 N128 为 `2 * 16 * (128 + 128) / 63.747 = 128.5 B/cycle`。BF16 N256 为
`2 * 16 * (128 + 256) / 129.381 = 95.0 B/cycle`。它被称为 logical visible
rate，因为分母是完整 MMA completion envelope，不是隔离出的物理 SMEM port
service time。

ld.shared delta 在固定 ops=32 下计算：

```text
extra_cycles_vs_register_alu =
  cycles_per_mma(mode, ops=32) - cycles_per_mma(register_alu, ops=32)
```

register-ALU row 控制额外 active warps 和 scheduler pressure；predicated-off
load 控制没有真实 shared traffic 的 load instruction/control overhead；L1-hit
global row 控制 general load/LSU pressure。因此 ld.shared 不比 register ALU 慢，
只能说明证据不足，不能支持 shared-specific port-conflict claim。

TMEM dependency range 是固定 Q16 下 same-D、合法 D-ring、`input_d=0` 和
`input_d=1` 静态 row 的 min/max。报告说“没有可测 penalty”而不是“没有成本”，
因为观察到的 spread 很小且没有系统性，不能识别 hidden TMEM dependency scoreboard
或 bank 结构。

## 推断

- 旧 runtime-dispatch 的 `750-870 cycles/MMA` beta 是测量污染。静态 Q4 结果与
  可信 K4 mainloop reference 同量级且几乎相同。
- 主要可见固定 completion 成本是约 `258 cycle` 的 commit/already-complete wait
  边界，以及约 `431-450 cycle` 的 forced single-MMA completion 边界。长 batch
  将其摊销后，在本 harness 下 N128 beta 约 `64 cycles/MMA`，N256 beta 约
  `129 cycles/MMA`。
- N128/N256 的 beta 比例更接近 tensor work size，而不是固定 operand-byte
  service limit。当前数据支持 logical visible service-rate range，不支持物理
  SMEM-to-Tensor-Core port width。
- collector protocol 在固定 Q 下有 ISA-visible 性能影响，但当前数据不能识别
  hidden collector depth。
- ld.shared 测试没有显示 shared-specific degradation。证据首先指向
  active-warp/scheduler/control pressure 和 general load/LSU pressure，而不是已证明
  的物理端口共享。
- same-D、合法 D-ring 和 `input_d=1` 在当前 Q16 静态窗口中不可区分。这不证明 TMEM
  dependency 免费，只说明这个测试没有暴露它。

## 直接回答

1. 当前证据可以给出 tested shapes 的 visible logical operand service rate。用 fitted
   beta 计算，BF16 N128 约 `128.5 B/cycle`，BF16 N256 约 `95.0 B/cycle`。不能给出
   物理 SMEM-to-Tensor-Core port width。
2. 证据不足以推断每 cycle 读取多少 SMEM bank 或写入多少 TMEM bank。
3. 不能证明 tcgen05 operand ingress 与普通 LSU `ld.shared` 完全共享或部分共享物理
   port。当前结论是证据不足；ld.shared 不比 register-ALU control 更慢，而 L1-hit
   global load 更慢。
4. hidden operand collector depth 不能识别。数据只能确认 tested Q16 sequence 下的
   ISA-visible collector protocol 效应；排除 wait、dispatch 和 D dependency 后没有
   可复现的 depth kink。
5. 在静态 BF16 N128 full-grid harness 中，commit 加 already-completed wait 约
   `258 cycles`，forced single-MMA completion 约 `431-450 cycles`，CTA
   synchronization 是 `20.828 cycles`，amortized batch MMA beta 对 N128 约
   `63.7 cycles/MMA`，对 N256 约 `129.4 cycles/MMA`。
6. same-D、合法 independent-D ring 和 `input_d=1` 在 tested static Q16/Q-sweep
   window 中没有显著影响吞吐；剩余 TMEM dependency 参数未被这些 run 识别。

## 不支持的说法

这些数据不能识别物理 SMEM port width、物理 SMEM bank count、物理 TMEM bank
count/width、hidden collector depth 或 hidden async group queue depth。它们也不能
证明普通 LSU `ld.shared` 与 async tcgen05 operand ingress 完全或部分共享物理端口。

## 旧 Runtime-Dispatch 负控制

早期各 stage 的 `benchmark_results.csv` 保留用于审计，但不应用于硬件推断。旧
`02_latency_throughput` 的 `752.564-874.469 cycles/MMA` beta、旧
`03_effective_smem_ingress` 的 logical `6.935-13.875 B/cycle`、旧
`05_ldshared_contention` 的 TFLOP/s 曲线，都受到 runtime dispatch、address/descriptor
work、D-ring arithmetic、wait placement 和 CTA synchronization 污染。

## 交付审计

| 要求 | 状态 | 证据 |
| --- | --- | --- |
| 每个实验一个独立子目录 | 完成 | `00_validation` 到 `07_config_matrix` 都有 `benchmark_src`、`scripts` 和 `plots` |
| 用静态校准 kernel 取代 universal timed kernel | 完成 | `02_latency_throughput/benchmark_src/tcgen05_02_static_calibration_bench.cu` |
| 性能推断前完成正确性检查 | 静态校准已完成 | 静态 aggregate CSV 均为 0 invalid row，记录了 guard 和 `max_abs_error` |
| SASS 审计 | 完成 | `02_latency_throughput/plots/static_sass/` 以及每行 SASS hash/count |
| 随机化执行顺序 | 完成 | `run_static_calibration.py` shuffle case 并记录 `run_order` |
| p10/p50/p90/CV timing | 完成 | 静态 aggregate CSV 含 p10、p50、p90 和 CV 字段 |
| CUDA event wall time 交叉检查 | 完成 | `event_wall_tflops` 和 event p10/p50/p90 与 clock64 throughput 分开记录 |
| Telemetry | 除 SM clock 外不可用 | Thor 上 `nvidia-smi` 对 memory clock、temperature 和 power 返回 N/A |
| 物理硬件说法边界 | 完成 | 上文明确列出 unsupported claims |
