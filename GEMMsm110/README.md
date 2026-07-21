# GEMMsm110 — Thor/SM110 Blackwell TCGen05/TMEM GEMM

> **目标架构**：NVIDIA DRIVE AGX Thor (SM110, compute capability 11.0, Blackwell)  
> **路径**：tcgen05/TMEM（非 SM120 的 `mma.sync.aligned.kind::f8f6f4` 路径）  
> **当前状态**：FP32 主报告集合已整理为 `GEMM_SUITE=core`，覆盖
> `128 256 512 1024 2048 4096`，每个尺寸 10 轮。1024 方阵的稳定推荐
> 是 `tc5b`，10 轮平均达到同进程 cuBLAS Tensor Core 的 0.927x；
> 2048 方阵的推荐 fast path 是 `tc5a`，10 轮平均达到 0.917x，
> 但本轮最低单轮为 0.898x。
> `tc6` 输出 NVFP4，不再混入 FP32 主图，改由 `GEMM_SUITE=nvfp4`
> 单独复测。所有有效 FP32 结果都以 cuBLAS Tensor Core 输出为参考。

## Backend 实验路线

本目录采用逐阶段、逐变量隔离的 backend 设计。除当前阶段明确声明的
变量外，FP32 backend 必须保持相同的 FP16 × FP16 → FP32 数据类型、
矩阵布局、输入数据、数值误差阈值、warmup/repeat 和计时方式。`tc6`
是例外：它的输出是 packed NVFP4 value buffer 加 E4M3 block scale，
仍使用同一组输入和 cuBLAS FP32 reference 做量化校验。

### 阶段 0：correctness / reference

| Backend | 实验变量 | 目的 |
| --- | --- | --- |
| `tc0` | 传统 CUDA Tensor Core；不使用 TMA、TCGen05 或 TMEM | 自有 correctness baseline；不作为 SM110 性能主图基线 |

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
overlapped epilogue，仍不能替代 1024 的 hybrid `tc5b` 路径或 2048 的
`tc5a` 路径。

### 阶段 5：persistent scheduling

| Backend | 实验变量 | 目的 |
| --- | --- | --- |
| `tc5a` | 6-warp、4-stage、双 TMEM buffer、overlapped epilogue | 2048+ 推荐 FP32 fallback，进入主图 |
| `tc5b` | 1024 专用 2-SM overlapped cluster path + `tc5a` fallback | 1024 推荐 FP32 backend，进入主图 |
| `tc5c` | `tc4a` 计算路径 + resident persistent workers + static grid stride | 静态 persistent 调度 |
| `tc5d/tc5e/tc5f` | `tc5c` 主循环的 TileN/TileK tuning 变体 | 隔离小尺寸 tile 数、resident CTA 和 TMA/wait 开销 |
| `tc5g/tc5h` | stage1 tuning 变体 | 验证少 stage 换更多 resident CTA 是否值得；实测不占优 |
| `tc5i/tc5j` | `tc5a` 的 N128/K128 tuning 变体 | 验证 tile shape 对 1024/2048 的影响 |

`tc5` 公开列表按主图优先排序：进入 `GEMM_SUITE=core` 主图的
`tc5a/tc5b` 放在最前；不进入主图、只用于调参或阶段解释的
`tc5c/tc5d/tc5e/tc5f/tc5g/tc5h/tc5i/tc5j` 放在后面。它们不依赖
2-SM `cta_group::2` 路径作为基础，而是先复用已经通过 finite-input
correctness 的 `tc4a` 1-SM mainloop；2-SM 路径作为 stage4 对照单独评估，
不混入 `tc5c` 的静态 persistent 结论。

命名整理后，旧文档和旧图里的 `tc5h` 对应现在的 `tc5a`，旧 `tc5n`
对应现在的 `tc5b`；原来的 static persistent `tc5a` 后移为 `tc5c`，
其它不进入主图的调参版本按顺序后移。

当前 FP32 性能推荐按尺寸区分：1024 方阵推荐 `tc5b`，2048 方阵推荐
`tc5a`。`tc5b` 在 1024 走 2-SM `M256N256K128` cluster path，并让每个
persistent cluster 最多处理两个 output tile，使 tile0 的 epilogue 和
tile1 的 mainloop 重叠；其它尺寸虽然会回退到 `tc5a`，但主报告按
实测稳定性把 2048 的推荐线直接写成 `tc5a`。`tc5a` 在 `tc5c` 的 raw
TCGen05/TMA 基础上增加 6-warp 分工：4 个 epilogue warp、1 个 TMA
warp、1 个 MMA warp；同时使用双 TMEM accumulator buffer，让上一个
tile 的 TMEM readback/GMEM store 与下一个 tile 的 mainloop 重叠。
FP32 store 使用 `st.global.L1::no_allocate.L2::evict_first.v8.f32`，
避免旧的两个 `float4` store 成为 1024/2048 的主要尾部成本。

`tc4b/tc4c` 的 2-SM accumulator ownership 已重新验证：2SM
`cta_group::2` 的 CTA 分区不能按 N 方向左右切输出，而是每个 CTA
负责本 CTA 的 M half，并通过 B 的 N half 分区共同形成完整 N 输出。
因此它们保留为 stage4/`GEMM_SUITE=unstable` 对照，但不进入
`GEMM_SUITE=core` 主报告集合。当前 FP32 推荐按尺寸区分：1024 方阵
用 `tc5b`，2048 方阵用 `tc5a`。

### 阶段 6：fused NVFP4 epilogue

| Backend | 实验变量 | 目的 |
| --- | --- | --- |
| `tc6` | `tc5c` 同款 resident persistent mainloop + fused NVFP4 store | 在 TMEM readback 阶段直接输出 packed E2M1 和 E4M3 block scale |

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
| `tc5a` | 已验证；6-warp 4-stage M128N256K64 + 双 TMEM buffer + overlapped epilogue，2048+ fallback |
| `tc5b` | 已验证；1024 专用 2-SM overlapped cluster path + `tc5a` fallback，1024 推荐 |
| `tc5c` | 已验证；`tc4a` 计算路径 + resident static persistent scheduler |
| `tc5d/tc5e/tc5f` | 已验证；`tc5c` 的 M128N128/N256、K64/K128 tuning 变体 |
| `tc5g/tc5h` | 已验证；stage1 tuning 变体，性能不占优 |
| `tc5i/tc5j` | 已验证；`tc5a` 的 N128/K128 tuning 变体，性能不占优 |
| `tc6` | 已验证；复用 `tc5c` 的 128×256×128 resident persistent TCGen05 mainloop，TMEM readback 直接写 packed NVFP4 |

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
- `tc0`：CUDA WMMA correctness baseline，不属于 TCGen05
  raw PTX 路线。

raw TCGen05 路径使用 K-major A 与 K-major B。逻辑矩阵
`B[K,N]` 在 benchmark 初始化阶段预排布为物理 `B[N,K]`，该一次性
转换不进入 kernel 计时；cuBLAS/CUTLASS reference 继续使用原始
`B[K,N]`，两份数据逐元素逻辑等价。

### 矩形 shape 与边界处理

benchmark 入口支持两种问题规模：

- `./build/gemm_sm110_bench N backend`：方阵 `M=N=K`。
- `./build/gemm_sm110_bench M N K backend`：矩形 GEMM。

当前边界处理在 `tc5c/tc5a/tc5b/tc6` 上保留为有效路径。`tc5c`
的完整 tile fast path 要求存在 `128x256x128` 整 tile；`tc5a` 的
完整 tile fast path 是 `128x256x64`。这些 raw SW128 TMA fast path 还要求
原始 K leading dimension 满足 64 个 half 的对齐；否则不创建 TMA descriptor，
直接切到 CUDA cleanup kernel 做完整 correctness fallback。K tail、M/N
边界和小矩阵也由 cleanup 补齐。`tc5b` 在精确 `1024x1024x1024` 时走
2-SM overlapped cluster path，其它 shape 复用 `tc5a`。

`tc6` 的完整 tile fast path 要求 `M%128==0`、`N%256==0`、`K%128==0` 且
`N%16==0`，对应 128×256×128 raw TMA + TCGen05 + fused NVFP4 epilogue。
不满足这些条件时，`tc6` 会切到 CUDA correctness fallback：每 16 个
row-major 输出为一组，串行计算 FP32 accumulation，生成同样的 packed E2M1
value 和 E4M3 block scale。这些 fallback 只证明边界语义和输出格式正确，
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

当前实现复用已验证的 `tc5c` resident persistent mainloop：warp0 发
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

下面是修正输入生成 bug 后，在 Thor 上重新跑的结果。当前主图使用
`GEMM_SUITE=core` 的 128 到 4096 全量 sweep；旧图表和旧 sweep 中包含
unsigned wrap 输入和混合 trial，不能再作为结论引用。

`M=N=K=1024` 主要用于快速 correctness smoke。这个规模的 tile 数少，
TMA、TMEM 分配/回读、barrier 和 kernel 固定开销占比高；因此 `tc5b`
在这个尺寸专门选择 2-SM overlapped cluster path，让 tile0 epilogue
和 tile1 mainloop 重叠。

| Backend | 1024 结果 | 配对结论 |
| --- | ---: | --- |
| `cublas_tc` | 102,729 avg GFLOP/s | cuBLAS Tensor Core reference，10/10 matched=1 |
| `cutlass` | 104,757 avg GFLOP/s | CUTLASS official Blackwell auto-schedule reference，10/10 matched=1 |
| `tc0` | 3,575 avg GFLOP/s | WMMA correctness-only baseline，10/10 matched=1；不进入 SM110 性能主图 |
| `tc1a/tc1b` | unavailable | linear SMEM descriptor 路径暂停验证 |
| `tc2a` | 35,439 avg GFLOP/s | raw SW128 single-stage TCGen05，10/10 matched=1 |
| `tc2b` | 35,392 avg GFLOP/s | raw 3D TMA + SW128 single-stage TCGen05，10/10 matched=1 |
| `tc3` | unstable suite | raw 2D TMA/TCGen05 2-stage pipeline，4096 复测出现 timeout |
| `tc4a` | 69,549 avg GFLOP/s | raw 2D TMA/TCGen05 warp-specialized pipeline，10/10 matched=1 |
| `tc4b/tc4c` | unstable suite | raw 2-SM cluster 对照，2048/4096 复测出现 timeout |
| `tc5c` | unstable suite | static persistent 1-SM TCGen05 路径，4096 复测出现 timeout |
| `tc5a` | 87,367 avg GFLOP/s | overlapped epilogue M128N256K64 TCGen05，10/10 matched=1 |
| `tc5b` | 95,183 avg GFLOP/s | hybrid 2-SM overlapped M256N256K128 TCGen05，10/10 matched=1 |
| `tc6` | nvfp4 suite | resident persistent mainloop + fused NVFP4 epilogue，不进入 FP32 主图 |

本轮优化后，1024/2048 的 FP32 自研 backend 已从 `tc5c` 的约
0.44x/0.60x 提升到接近或超过 0.90x cuBLAS Tensor Core。1024 的提升来自
2-SM `M256N256K128` overlapped cluster path；该路径现在补齐
`cta_group::2` 的 TMEM alloc/relinquish/dealloc 全生命周期 cluster
同步，并针对固定 1024 shape 展开 tile 坐标、TMA 和 MMA K loop。
2048 继续使用 `tc5a` 的 M128N256K64 4-stage overlapped epilogue
fallback。旧 2-SM
`tc4b/tc4c` raw 路径去掉冗余 cluster-wide sync 后已经从旧的
0.44-0.52x 提升到约 0.78x/0.87x，但非 overlapped epilogue 仍会放大
TMA、TMEM allocation/readback 和固定成本，所以只作为 stage4 对照。

当前 `core` sweep 的稳定性复测结果：1024 方阵推荐 `tc5b`，10/10 次
`matched=1`，平均 95,183 GFLOP/s，RatioToCuBLAS 平均 0.927、最小
0.925；2048 方阵推荐 fast path 是 `tc5a`，10/10 次 `matched=1`，
平均 119,550 GFLOP/s，RatioToCuBLAS 平均 0.917、最小 0.898。
`tc5b` 在 2048 不是独立新 kernel，而是同一个 `tc5a` fallback 的
单独运行；这轮它平均 120,089 GFLOP/s、RatioToCuBLAS 平均 0.920、
最小 0.909，因此两条 2048 线的差异应解释为运行波动，而不是不同算法收益。

旧的 full-tile 稳态参考是 4096。`tc5c` 在 `4096x4096x4096` 上达到
同进程 cuBLAS Tensor Core 的约 0.82x；2-SM `tc4b/tc4c` 也能达到
约 0.74x/0.76x，但仍低于 `tc5c`。

| Shape | Backend | cuBLAS GFLOP/s | Backend GFLOP/s | Ratio | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| 2048³ | `tc4a` | 130,364 | 74,969 | 0.57x | warp-specialized non-persistent，matched=1 |
| 1024³ | `tc4c` | 102,876 | 80,261 | 0.78x | 旧 2-SM + warp specialization raw path，matched=1 |
| 2048³ | `tc4c` | 130,232 | 113,868 | 0.87x | 旧 2-SM + warp specialization raw path，matched=1 |
| 2048³ | `tc5c` | 130,364 | 78,227 | 0.60x | static persistent，matched=1 |
| 1024³ | `tc5a` | 102,909 | 87,387 | 0.85x | overlapped epilogue，matched=1 |
| 2048³ | `tc5a` | 131,168 | 120,122 | 0.92x | overlapped epilogue，matched=1 |
| 1024³ | `tc5b` | 102,729 avg | 95,183 avg | 0.927x avg | hybrid 2-SM overlapped cluster path，10/10 matched=1 |
| 2048³ | `tc5a` | 130,332 avg | 119,550 avg | 0.917x avg | overlapped epilogue M128N256K64，10/10 matched=1；最低单轮 0.898x |
| 4096³ | `tc3` | 64,420 | 38,664 | 0.60x | multi-stage non-persistent，matched=1 |
| 4096³ | `tc4b` | 64,420 | 48,596 | 0.74x | 2-SM cluster，matched=1 |
| 4096³ | `tc4c` | 64,420 | 47,973 | 0.76x | 2-SM + warp specialization，matched=1 |
| 4096³ | `tc5c` | 64,420 | 53,073 | 0.82x | 旧 large-shape baseline，matched=1 |

几条看似能改善 1024/2048 的路线已经实测后否定，当前不作为推荐
backend。第一，split-K 可以把 1024/2048 的 work item 数量放大，但每个
partial K 太短，TMA、TMEM allocation/readback 和 FP32 partial reduction
成本被放大；`SplitK=2/4` 都显著慢于 `tc5c`。第二，把 `tc5c` 从
`TileK=128, Stages=2` 改成 `TileK=256, Stages=1` 虽然能让单次 wait
覆盖更多 K-block，但失去双缓冲后 1024/2048 都变慢。第三，降低
`TileK` 或 `TileN` 以换取 2-3 个 resident CTA/SM 也不占优：更多
CTA 没有抵消更频繁的 TMA/wait 和更低效的 MMA shape。第四，`tc5a`
的 N128/K128 变体和 stage1 变体均不占优；因此 `tc5b` fallback 继续
保留 M128N256K64、4-stage 和 overlapped epilogue。

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
│   │   ├── tc5_persistent.cuh  # tc5 persistent and overlapped epilogue schedulers
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
# 编译 + 运行（默认 N=1024, core 主报告集合）
./build_and_run.sh

# 指定尺寸
./build_and_run.sh 2048

# 指定矩形 M N K
./build_and_run.sh 260 132 256 tc6
./build_and_run.sh 384 520 300 tc5b
./build_and_run.sh 260 132 256 all
./build_and_run.sh 1024 all

# 只跑某个 backend
./build_and_run.sh 1024 core
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
./build_and_run.sh 1024 tc5c
./build_and_run.sh 1024 tc5d
./build_and_run.sh 1024 tc5e
./build_and_run.sh 1024 tc5f
./build_and_run.sh 1024 tc5g
./build_and_run.sh 1024 tc5h
./build_and_run.sh 1024 tc5i
./build_and_run.sh 1024 tc5j
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

批量实验脚本用于生成主报告 CSV、GFLOP/s 图和 ratio 图。主报告的推荐
入口是 `scripts/run_sm110_gemm_core_sweep.sh`，它固定默认使用
`GEMM_SUITE=core`、`PRESET=full`、`TRIALS=10` 和 120 秒 backend timeout。
`scripts/run_gemm_sm110_experiments.sh` 是底层通用入口，默认集合同样是
`GEMM_SUITE=core`，覆盖
`128 256 512 1024 2048 4096`，每个矩阵尺寸、每个 backend 运行 10 个
独立 trial；每个 trial 预热 5 次，并对随后 100 次 kernel launch
计时取平均。当前环境如果已经有 `GEMMsm110/build/gemm_sm110_bench`，
可以从仓库根目录复用现有二进制全量重跑主图：

```bash
SKIP_BUILD=1 bash scripts/run_sm110_gemm_core_sweep.sh
```

这个入口默认使用 `BACKEND_ATTEMPTS=3`，用于重试偶发的 launch failure
或 timeout；如果要保留一次尝试的原始行为，可以显式设置
`BACKEND_ATTEMPTS=1`。

主报告性能集合包含 `cublas_tc`、`cutlass`、`tc2a`、`tc2b`、`tc4a`、
`tc5a` 和 `tc5b`。`tc0` 保留在 `stage0` 和显式 `tc0` 入口里，只做
WMMA correctness 对照。不稳定但有阶段说明价值的 backend 可单独跑：

```bash
SKIP_BUILD=1 GEMM_SUITE=unstable PRESET=full TRIALS=10 BACKEND_TIMEOUT_SECONDS=120 \
  bash scripts/run_gemm_sm110_experiments.sh
```

`tc6` 输出 packed NVFP4 value 和 E4M3 block scale，不进入 FP32 主图；
需要复测 fused NVFP4 epilogue 时使用：

```bash
SKIP_BUILD=1 GEMM_SUITE=nvfp4 PRESET=full TRIALS=10 BACKEND_TIMEOUT_SECONDS=120 \
  bash scripts/run_gemm_sm110_experiments.sh
```

脚本会覆盖兼容旧入口的 `results/gemm_sm110/gemm_sm110_sweep.csv`、
`results/gemm_sm110/figures/gemm_tensor_core_gflops.svg` 和
`results/gemm_sm110/figures/gemm_tensor_core_ratio_to_cublas_tc.svg`；同时生成
更明确的命名别名：

```text
results/gemm_sm110/sm110_gemm_core_128_4096_10trials.csv
results/gemm_sm110/figures/sm110_gemm_core_128_4096_10trials_gflops.svg
results/gemm_sm110/figures/sm110_gemm_core_128_4096_10trials_ratio_to_cublas_tc.svg
```

原文件名保留，是为了兼容已有引用；新文件名用于报告和归档。可通过
`GEMM_SIZES`、`TRIALS` 或 `RESULT_TAG` 临时覆盖尺寸、独立样本数和命名。

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
