# GEMMsm110 — Thor/SM110 Blackwell TCGen05/TMEM GEMM

> **目标架构**：NVIDIA DRIVE AGX Thor (SM110, compute capability 11.0, Blackwell)  
> **路径**：tcgen05/TMEM（非 SM120 的 `mma.sync.aligned.kind::f8f6f4` 路径）  
> **当前状态**：已验证 FP16 × FP16 → FP32 GEMM 路径到 `tc5n`。
> `tc5n` 是当前推荐的 FP32 路径：1024 方阵使用 2-SM overlapped
> cluster path，2048 方阵回退到 `tc5h`，两者均达到同进程 cuBLAS
> Tensor Core 的 90% 以上。`tc6` 是 FP16 × FP16
> → NVFP4 fused epilogue backend。所有有效 FP32 结果都以 cuBLAS
> Tensor Core 输出为参考。

## Backend 实验路线

本目录采用逐阶段、逐变量隔离的 backend 设计。除当前阶段明确声明的
变量外，FP32 backend 必须保持相同的 FP16 × FP16 → FP32 数据类型、
矩阵布局、输入数据、数值误差阈值、warmup/repeat 和计时方式。`tc6`
是例外：它的输出是 packed NVFP4 value buffer 加 E4M3 block scale，
仍使用同一组输入和 cuBLAS FP32 reference 做量化校验。

### 阶段 0：correctness / reference

| Backend | 实验变量 | 目的 |
| --- | --- | --- |
| `tc0` | 传统 CUDA Tensor Core；不使用 TMA、TCGen05 或 TMEM | 自有 correctness/performance baseline |

### 阶段 1：TMA + TCGen05 minimal path

| Backend | 实验变量 | 目的 |
| --- | --- | --- |
| `tc1a` | 2D TMA、16B/linear SMEM、TCGen05/TMEM | 线性 SMEM descriptor bring-up；当前暂停计入结果 |
| `tc1b` | 3D TMA、16B/linear SMEM、TCGen05/TMEM | 线性 SMEM descriptor + 3D TMA bring-up；当前暂停计入结果 |

`tc1a/tc1b` 在修正输入生成 bug 后没有通过 finite-input correctness
校验，主 benchmark 暂时报告 unavailable。后续只有在线性 SMEM
descriptor 的数值映射重新验证后，才能把它们恢复成有效对照。

### 阶段 2：bank conflict isolation

| Backend | 实验变量 | 目的 |
| --- | --- | --- |
| `tc2a` | `tc1a` + 128B SMEM swizzle | 2D TMA 下只改变 SMEM layout/swizzle descriptor |
| `tc2b` | `tc1b` + 128B SMEM swizzle | 3D TMA 下只改变 SMEM layout/swizzle descriptor |

`tc2a/tc2b` 是当前通过校验的最小 TCGen05/SW128 descriptor 路径。
由于 `tc1a/tc1b` 暂停，当前不能再把 `tc1 -> tc2` 差异解释成严格
bank-conflict 收益，只能把 `tc2` 作为后续 pipeline 的正确性基线。

### 阶段 3：latency hiding / overlap

| Backend | 实验变量 | 目的 |
| --- | --- | --- |
| `tc3` | `tc2a` + multi-stage pipeline、double buffer、TMA mbarrier | 用 prefetch 与 load/compute overlap 隔离延迟隐藏收益 |

`tc3` 的直接基线固定为 `tc2a`，不同时引入 3D TMA、warp
specialization 或 cluster MMA。

当前自研 TCGen05 backend（`tc1a` 到 `tc6`）都已改为 raw
CUDA/inline-PTX kernel。kernel 自行完成 `CUtensorMap` 坐标计算、
TMA、mbarrier phase、TCGen05 MMA、TMEM 分配/回读和 epilogue，不使用
CuTe `Tensor`、`TiledMMA`、TMA atom、CUTLASS collective 或 CUTLASS
scheduler。

### 阶段 4：scheduling and cluster

| Backend | 实验变量 | 目的 |
| --- | --- | --- |
| `tc4a` | `tc3` + warp specialization | TMA producer、MMA consumer 和 epilogue/readback 分工 |
| `tc4b` | `tc3` + 2-SM cluster MMA | 使用 cluster dimensions 与 `cta_group::2`，隔离 2-SM MMA 收益 |
| `tc4c` | `tc3` + warp specialization + 2-SM cluster MMA | 验证两种优化能否叠加 |

`tc4b/tc4c` 现在使用 raw CUDA/inline PTX 2-SM 路径：一个 cluster 计算
`M256N256K128` tile；两个 CTA 分别加载 A 的 M half 和 B 的 N half，
leader CTA 发 `tcgen05.mma.cta_group::2`，两个 CTA 各自从 TMEM 读回
本 CTA 的 128 行、完整 256 列。TMEM allocation 按 2SM allocator 的
512 column capacity 申请和释放。这个路径已经通过 finite-input
correctness。去掉每个 K tile 的 cluster-wide sync 后，`tc4c` 明显快于
旧 2-SM raw 路径，但在 1024/2048 上因为 cluster tile 数较少且没有
overlapped epilogue，仍不能替代当前推荐的 hybrid `tc5n` 路径。

### 阶段 5：persistent scheduling

| Backend | 实验变量 | 目的 |
| --- | --- | --- |
| `tc5a` | `tc4a` 计算路径 + resident persistent workers + static grid stride | 静态 persistent 调度 |
| `tc5b` | `tc5a` 的计算路径 + software dynamic work queue | 动态领取 tile 的实验路径；当前暂停计入有效结果 |
| `tc5c/tc5d/tc5e` | `tc5a` 主循环的 TileN/TileK tuning 变体 | 隔离小尺寸 tile 数、resident CTA 和 TMA/wait 开销 |
| `tc5f/tc5g` | stage1 tuning 变体 | 验证少 stage 换更多 resident CTA 是否值得；实测不占优 |
| `tc5h` | 6-warp、4-stage、双 TMEM buffer、overlapped epilogue | 2048+ 推荐 FP32 fallback |
| `tc5i/tc5j` | `tc5h` 的 N128/K128 tuning 变体 | 验证 tile shape 对 1024/2048 的影响 |
| `tc5k` | M64 overlap 实验 | 当前暂停；已能 matched=1，但 M64 panel epilogue 性能不占优 |
| `tc5n` | 1024 专用 2-SM overlapped cluster path + `tc5h` fallback | 当前推荐 FP32 backend |

`tc5a/tc5b` 不依赖 2-SM `cta_group::2` 路径，而是复用已经通过
finite-input correctness 的 `tc4a` 1-SM mainloop。这样阶段 5 先把
persistent 调度做成可验证实现；2-SM 路径作为 stage4 对照单独评估，
不混入 `tc5a` 的静态 persistent 结论。

当前推荐的 FP32 性能 backend 是 `tc5n`。1024 方阵走 2-SM
`M256N256K128` cluster path，并让每个 persistent cluster 最多处理
两个 output tile，使 tile0 的 epilogue 和 tile1 的 mainloop 重叠；
其它尺寸回退到 `tc5h`。`tc5h` 在 `tc5a` 的 raw
TCGen05/TMA 基础上增加 6-warp 分工：4 个 epilogue warp、1 个 TMA
warp、1 个 MMA warp；同时使用双 TMEM accumulator buffer，让上一个
tile 的 TMEM readback/GMEM store 与下一个 tile 的 mainloop 重叠。
FP32 store 使用 `st.global.L1::no_allocate.L2::evict_first.v8.f32`，
避免旧的两个 `float4` store 成为 1024/2048 的主要尾部成本。
`tc5b` 保留为“动态 persistent 调度是否值得”的源码实验；在 `4096`
复测中出现过 timeout，主 benchmark 暂停把它计入有效结果。

`tc4b/tc4c` 的 2-SM accumulator ownership 已重新验证：2SM
`cta_group::2` 的 CTA 分区不能按 N 方向左右切输出，而是每个 CTA
负责本 CTA 的 M half，并通过 B 的 N half 分区共同形成完整 N 输出。
因此它们现在作为 stage4 对照纳入 benchmark；当前推荐 FP32 性能路径
是 `tc5n`。

### 阶段 6：fused NVFP4 epilogue

| Backend | 实验变量 | 目的 |
| --- | --- | --- |
| `tc6` | `tc5a` 同款 resident persistent mainloop + fused NVFP4 store | 在 TMEM readback 阶段直接输出 packed E2M1 和 E4M3 block scale |

### 性能 reference

| Backend | 定位 |
| --- | --- |
| `cublas_tc` | cuBLAS Tensor Core reference，同时生成数值参考 |
| `cutlass` | CUTLASS 官方 Blackwell Auto Schedule reference |

## 实现状态与旧代码迁移

| Backend | 状态 |
| --- | --- |
| `cublas_tc`、`cutlass` | 已实现 |
| `tc0` | 已实现；CUDA WMMA，SASS 为 `HMMA.16816.F32` |
| `tc1a`、`tc1b` | 暂停计入结果；linear/INTER SMEM descriptor 路径未通过修正输入后的 correctness |
| `tc2a`、`tc2b` | 已验证；raw CUDA/inline-PTX，single-stage SW128 descriptor 路径 |
| `tc3` | 已实现；自有 raw CUDA/inline-PTX，固定 128×128×64、2-stage、2D TMA SW128 pipeline |
| `tc4a` | 已实现；raw CUDA/inline-PTX，固定 128×256×128、2-stage，warp0 MMA consumer、warp1 TMA producer |
| `tc4b`、`tc4c` | 已验证；raw inline-PTX 2-SM `cta_group::2`，固定 256×256×128 cluster tile |
| `tc5a` | 已验证；`tc4a` 计算路径 + resident static persistent scheduler |
| `tc5b` | 暂停计入结果；software dynamic work queue 在 4096 复测中出现 timeout |
| `tc5c/tc5d/tc5e` | 已验证；`tc5a` 的 M128N128/N256、K64/K128 tuning 变体 |
| `tc5f/tc5g` | 已验证；stage1 tuning 变体，性能不占优 |
| `tc5h` | 已验证；6-warp 4-stage M128N256K64 + 双 TMEM buffer + overlapped epilogue，2048+ fallback |
| `tc5i/tc5j` | 已验证；`tc5h` 的 N128/K128 tuning 变体，性能不占优 |
| `tc5k` | 暂停计入结果；M64 TCGen05/TMEM layout 已可 matched=1，但 epilogue 性能不占优，主 benchmark 标为 unavailable |
| `tc5n` | 已验证；1024 专用 2-SM overlapped cluster path + `tc5h` fallback，当前推荐 |
| `tc6` | 已验证；复用 `tc5a` 的 128×256×128 resident persistent TCGen05 mainloop，TMEM readback 直接写 packed NVFP4 |

本轮修正了 benchmark 输入生成中的无符号下溢问题：原先
`(i % 17) - 8` 在 `size_t` 上会先发生 unsigned wrap，可能生成
huge/inf/NaN 输入并掩盖错误。现在先转成 `int` 再减偏置，因此
`matched=1` 才能作为有效 correctness 证据。

### CUTLASS 依赖边界

- `cutlass` backend：唯一允许实例化 CUTLASS 官方高层 GEMM 的性能
  reference。
- `tc1/tc2/tc3/tc4/tc5/tc6`：完全自有 kernel，仅使用
  `sm110_ptx_helpers.cuh` 中的薄 PTX/Driver API helper；这些 helper
  的指令写法参考 CUDA PTX 文档、`learn-cuda` 和 CUTLASS 底层 arch
  wrapper，但不包含或实例化 CUTLASS/CuTe。
- `tc0`：CUDA WMMA correctness/performance baseline，不属于 TCGen05
  raw PTX 路线。

raw TCGen05 路径使用 K-major A 与 K-major B。逻辑矩阵
`B[K,N]` 在 benchmark 初始化阶段预排布为物理 `B[N,K]`，该一次性
转换不进入 kernel 计时；cuBLAS/CUTLASS reference 继续使用原始
`B[K,N]`，两份数据逐元素逻辑等价。

### 矩形 shape 与边界处理

benchmark 入口支持两种问题规模：

- `./build/gemm_sm110_bench N backend`：方阵 `M=N=K`。
- `./build/gemm_sm110_bench M N K backend`：矩形 GEMM。

当前边界处理在 `tc5a/tc5b/tc5h/tc5n/tc6` 上保留为有效路径。`tc5a/tc5b`
的完整 tile fast path 要求存在 `128x256x128` 整 tile；`tc5h` 的
完整 tile fast path 是 `128x256x64`，K tail、M/N 边界、小矩阵会由
CUDA cleanup kernel 补齐。`tc5n` 在精确 `1024x1024x1024` 时走
2-SM overlapped cluster path，其它 shape 复用 `tc5h`。`tc6` 的完整 tile fast
path 要求 `M%128==0`、`N%256==0`、`K%128==0` 且 `N%16==0`，对应
128×256×128 raw TMA + TCGen05 + fused NVFP4 epilogue。不满足这些
条件时，`tc6` 会切到 CUDA correctness fallback：每 16 个 row-major
输出为一组，串行计算 FP32 accumulation，生成同样的 packed E2M1 value
和 E4M3 block scale。这些 fallback 只证明边界语义和输出格式正确，
不作为高性能路径。

### tc6 NVFP4 backend 接入点

`tc6` 的目标不是在 GEMM 后另起一个独立量化 kernel，而是在 TMEM
readback 阶段直接完成 NVFP4 store。现有
`include/requant/sm110_tcgen05_epilogue.cuh` 已经提供三块可复用逻辑：

- 从 TMEM accumulator 读取 FP32 pair：`sm110_tcgen05_load_32x32b_x2`。
- 每 16 个输出值生成一个 E4M3 block scale：
  `sm110_make_nvfp4_block_scale`。
- 把两个 FP32 accumulator 按 tensor scale 和 block scale 转为 packed
  E2M1：`sm110_requant_nvfp4_e2m1x2`。

当前实现复用已验证的 `tc5a` resident persistent mainloop：warp0 发
MMA，warp1 发 TMA，所有 128 个线程按行读取 TMEM accumulator。每
16 个连续 row-major 输出生成一个 `uint8` E4M3 block scale，每两个
E2M1 输出 packed 到一个 `uint8` value buffer。整 tile fast path 在
`1024x1024x1024` 上与 CPU NVFP4 reference bit-exact；非整除 fallback
允许 bit 不完全一致，但 dequant 后 RMSE 和最大误差必须不劣于 CPU
reference 口径。

## 验收标准

每个 backend 合入全量 sweep 前必须满足：

1. 对所有支持尺寸输出真实 kernel 时间和 GFLOPS，不使用 probe 值。
2. 与 `cublas_tc` 比较并满足 `matched=1`。
3. SASS 中存在该阶段要求的关键指令；例如 TMA、TCGen05、2-SM
   multicast 或 CLC。
4. NCU 记录与阶段目标相关的指标：bank conflict、barrier stall、
   eligible warps、Tensor Core/TMA 吞吐和 occupancy。
5. 若某尺寸不满足 tile/cluster 约束，必须报告 skipped/unavailable，
   不得静默切换为其他 backend。
6. 最终 backend 需要支持非整除 shape。允许 full-tile fast path 配合
   cleanup kernel，但必须在文档里说明哪些部分是高性能路径，哪些部分
   只是 correctness fallback。

## 当前阶段性结果

下面是修正输入生成 bug 后，在 Thor 上重新跑的结果。`tc5n` 行来自
本轮同进程单次复测；旧图表和旧 sweep 中包含 unsigned wrap 输入，
不能再作为结论引用。

`M=N=K=1024` 主要用于快速 correctness smoke。这个规模的 tile 数少，
TMA、TMEM 分配/回读、barrier 和 kernel 固定开销占比高；因此 `tc5n`
在这个尺寸专门选择 2-SM overlapped cluster path，让 tile0 epilogue
和 tile1 mainloop 重叠。

| Backend | GFLOPS | 配对结论 |
| --- | ---: | --- |
| `cublas_tc` | 102,885 | cuBLAS Tensor Core reference |
| `cutlass` | 104,699 | CUTLASS official Blackwell auto-schedule reference |
| `tc0` | 3,573 | WMMA correctness baseline |
| `tc1a/tc1b` | unavailable | linear SMEM descriptor 路径暂停验证 |
| `tc2a` | 28,283 | raw SW128 single-stage TCGen05，matched=1 |
| `tc2b` | 28,280 | raw 3D TMA + SW128 single-stage TCGen05，matched=1 |
| `tc3` | 37,378 | raw 2D TMA/TCGen05 2-stage pipeline，matched=1 |
| `tc4a` | 45,532 | raw 2D TMA/TCGen05 warp-specialized K8/N256 pipeline，matched=1 |
| `tc4b` | 37,444 | raw 2-SM `M256N256` cluster path，matched=1 |
| `tc4c` | 37,457 | raw 2-SM + warp-specialized cluster path，matched=1 |
| `tc5a` | 69,886 | static persistent 1-SM TCGen05 + v8 no-allocate store，matched=1 |
| `tc5h` | 87,389 | overlapped epilogue M128N256K64 TCGen05，matched=1 |
| `tc5n` | 94,648 | hybrid 2-SM overlapped M256N256K128 TCGen05，matched=1 |
| `tc5b` | unavailable | dynamic work queue 路径暂停稳定性验证 |
| `tc6` | 28,313 | resident persistent mainloop + fused NVFP4 epilogue，matched=1 |

本轮优化后，1024/2048 的 FP32 自研 backend 已从 `tc5a` 的约
0.44x/0.60x 提升到 `tc5n` 的约 0.93x/0.92x。1024 的提升来自
2-SM `M256N256K128` overlapped cluster path；2048 继续使用 `tc5h`
的 M128N256K64 4-stage overlapped epilogue fallback。旧 2-SM
`tc4b/tc4c` raw 路径去掉冗余 cluster-wide sync 后已经从旧的
0.44-0.52x 提升到约 0.78x/0.87x，但非 overlapped epilogue 仍会放大
TMA、TMEM allocation/readback 和固定成本，所以只作为 stage4 对照。

旧的 full-tile 稳态参考是 4096。`tc5a` 在 `4096x4096x4096` 上达到
同进程 cuBLAS Tensor Core 的约 0.82x；2-SM `tc4b/tc4c` 也能达到
约 0.74x/0.76x，但仍低于 `tc5a`。

| Shape | Backend | cuBLAS GFLOP/s | Backend GFLOP/s | Ratio | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| 2048³ | `tc4a` | 130,364 | 74,969 | 0.57x | warp-specialized non-persistent，matched=1 |
| 1024³ | `tc4c` | 102,876 | 80,261 | 0.78x | 旧 2-SM + warp specialization raw path，matched=1 |
| 2048³ | `tc4c` | 130,232 | 113,868 | 0.87x | 旧 2-SM + warp specialization raw path，matched=1 |
| 2048³ | `tc5a` | 130,364 | 78,227 | 0.60x | static persistent，matched=1 |
| 1024³ | `tc5h` | 102,909 | 87,387 | 0.85x | overlapped epilogue，matched=1 |
| 2048³ | `tc5h` | 131,168 | 120,122 | 0.92x | overlapped epilogue，matched=1 |
| 1024³ | `tc5n` | 102,211 | 94,648 | 0.93x | hybrid 2-SM overlapped cluster path，matched=1 |
| 2048³ | `tc5n` | 130,035 | 119,506 | 0.92x | hybrid fallback to `tc5h`，matched=1 |
| 4096³ | `tc3` | 64,420 | 38,664 | 0.60x | multi-stage non-persistent，matched=1 |
| 4096³ | `tc4b` | 64,420 | 48,596 | 0.74x | 2-SM cluster，matched=1 |
| 4096³ | `tc4c` | 64,420 | 47,973 | 0.76x | 2-SM + warp specialization，matched=1 |
| 4096³ | `tc5a` | 64,420 | 53,073 | 0.82x | 旧 large-shape baseline，matched=1 |

`tc5b` 的动态 work queue 在临时复测中可以产出 matched=1 的结果，但
也出现过 timeout，因此不再作为有效性能 backend 统计。这个结论对面试
解释很重要：当前完整调度策略是 `tc5a` 的 resident static persistent，
不是尚未稳定的动态领取。

几条看似能改善 1024/2048 的路线已经实测后否定，当前不作为推荐
backend。第一，split-K 可以把 1024/2048 的 work item 数量放大，但每个
partial K 太短，TMA、TMEM allocation/readback 和 FP32 partial reduction
成本被放大；`SplitK=2/4` 都显著慢于 `tc5a`。第二，把 `tc5a` 从
`TileK=128, Stages=2` 改成 `TileK=256, Stages=1` 虽然能让单次 wait
覆盖更多 K-block，但失去双缓冲后 1024/2048 都变慢。第三，降低
`TileK` 或 `TileN` 以换取 2-3 个 resident CTA/SM 也不占优：更多
CTA 没有抵消更频繁的 TMA/wait 和更低效的 MMA shape。第四，`tc5h`
的 N128/K128 变体和 stage1 变体均不占优；因此 `tc5n` fallback 继续
保留 M128N256K64、4-stage 和 overlapped epilogue。

真正可能增加 1024 小尺寸并行度的路线之一是 `M64`，但当前 `tc5k`
还不是答案。CUTLASS/CuTe 底层 trait 显示 1SM F16/BF16 MMA 支持
`M=64`，但 M64 的 epilogue 不能沿用 M128 的 `32x32b.x8` TMEM load；
当前 `16x256b` panel 读回已经可以在 1024/2048 上 `matched=1`，并且
把 `x16` 读回拆成 `x8` 后资源从 `REG:118` 降到 `REG:80 STACK:0`。
但 `tc5k` 的 1024/2048 仍只有约 0.56x cuBLAS，因此继续在 registry
和 main benchmark 中标为 unavailable，避免 stage5 混入明显慢于
`tc5h` 的结果。后续需要重做 M64 epilogue 的 TMEM panel 到 GMEM
store 打包，而不是只降低寄存器。

`tc6` 的 GFLOP/s 仍按 GEMM 数学量报告，方便观察 fused epilogue 开销；
但它输出 packed NVFP4 value 和 E4M3 block scale，不写 FP32 `D` 矩阵，
因此不能把它的 RatioToReference 当成和 FP32 cuBLAS 完全等价的性能比较。

Thor 的 `l1tex__data_bank_conflicts_pipe_lsu` 计数会被 TMA 写入记账
主导，因此不能单独用该计数解释 SW128 收益；需要同时结合 kernel
时间、descriptor/SASS 和 stall 指标。

## 前置条件

- GPU：Thor/SM110
- CUDA Toolkit：13.0+
- CUTLASS：4.5.2（默认 `/xplorer/shijy/third_party/cutlass`，可通过 `CUTLASS_ROOT` 指定）
- 编译器：需支持 C++17

## 目录结构

```
GEMMsm110/
├── build_and_run.sh            # 编译 + 运行脚本
├── include/
│   ├── gemm_common.cuh         # CHECK_CUDA/CHECK_CUBLAS 宏、kWarmup/kRepeat 等
│   ├── gemm_benchmark.cuh      # 输入构造、精度对比、benchmark_kernel
│   ├── cutlass_sm110_backends.cuh # 仅官方 CUTLASS auto-schedule 基线
│   ├── sm110_backend_registry.cuh # backend 名称、元数据和 runner 注册
│   ├── sm110_ptx_helpers.cuh   # 自有 kernel 使用的薄 PTX/Driver API helper
│   ├── backends/               # 按实验阶段拆分的自有 backend
│   │   ├── tc0_baseline.cuh
│   │   ├── tc1_tc2_tma.cuh     # tc1/tc2 共用模板，避免额外变量
│   │   ├── tc3_pipeline.cuh
│   │   ├── tc4a_warp_specialized.cuh
│   │   ├── tc4bc_cluster.cuh   # tc4b/tc4c 共用模板
│   │   ├── tc5_persistent.cuh  # tc5a static / tc5b dynamic persistent scheduler
│   │   └── tc6_nvfp4.cuh       # fused NVFP4 epilogue backend
│   ├── requant/                # NVFP4/E2M1 尾处理后端和 CPU reference
│   └── tc3_gemm_kernel.cuh     # 独立 TMEM minimal sanity probe
├── tests/
│   ├── requant_epilogue_benchmark.cu
│   └── run_requant_epilogue_benchmark.sh
└── src/
    └── main.cu                 # 入口
```

## 编译和运行

```bash
# 编译 + 运行（默认 N=1024, 全部 backend）
./build_and_run.sh

# 指定尺寸
./build_and_run.sh 2048

# 指定矩形 M N K
./build_and_run.sh 260 132 256 tc6
./build_and_run.sh 260 132 256 all

# 只跑某个 backend
./build_and_run.sh 1024 tc0
./build_and_run.sh 1024 tc1a
./build_and_run.sh 1024 tc1b
./build_and_run.sh 1024 tc2a
./build_and_run.sh 1024 tc2b
./build_and_run.sh 1024 cutlass
./build_and_run.sh 1024 tc3
./build_and_run.sh 1024 tc4a
./build_and_run.sh 1024 tc4b
./build_and_run.sh 1024 tc4c
./build_and_run.sh 1024 tc5a
./build_and_run.sh 1024 tc5b
./build_and_run.sh 1024 tc5h
./build_and_run.sh 1024 tc5n
./build_and_run.sh 1024 tc6

# 仅编译不运行
./build_and_run.sh build-only

# 清理编译产物
./build_and_run.sh clean

# 每个 backend 最长运行 30 秒
./build_and_run.sh 1024 tc6 30
```

编译产物在 `build/` 目录下，运行时在当前目录生成
`sgemm_sm110_benchmark.csv`。结果包含 `TimeMs`、`GFLOPS`、
`RatioToReference` 和 `Matched`。

批量实验脚本默认对每个矩阵尺寸、每个 backend 运行 10 个独立
trial；每个 trial 预热 5 次，并对随后 100 次 kernel launch
计时取平均。可通过 `TRIALS` 临时覆盖独立样本数：

```bash
TRIALS=20 bash scripts/run_gemm_sm110_experiments.sh
```

## NVFP4 尾处理测试

批量测试不同矩阵尺寸和输入分布：

```bash
./tests/run_requant_epilogue_benchmark.sh
```

可通过环境变量调整测试轮数：

```bash
WARMUP=5 ITERATIONS=50 SEED=7 \
  ./tests/run_requant_epilogue_benchmark.sh
```

单独运行一个用例：

```bash
./tests/build/requant_epilogue_benchmark \
  --rows 1024 \
  --cols 1024 \
  --distribution outlier \
  --warmup 10 \
  --iterations 100
```

测试覆盖 `uniform`、`normal`、`laplace`、`outlier`、`lognormal`
和 `constant` 分布。结果包含量化值与 E4M3 block scale 的 CPU
reference 精确匹配、RMSE、最大绝对误差、处理吞吐和有效带宽。

## 相关仓库文档

- `Docs/cutlass/blackwellMMA.md` — UMMA 指令和 SMEM descriptor 参考
- `Docs/cutlass/cutlassSimpleExemple.md` — CUTLASS SM100 UMMA 示例
- `Docs/cutlass/sm110_gemm_bank_conflict_research.md` — SM110 GEMM bank conflict 研究计划
