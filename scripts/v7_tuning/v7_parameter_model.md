# v7 Warp Tiling 参数建模

这份文档用于在 benchmark 之前裁剪 `sgemm_v7_warp_tiling_double_buffer`
的候选参数集。目标不是靠模型直接预测最快 kernel，而是先排除明显违反
代码映射、当前实现路径或硬件资源限制的参数组合，再把剩余候选分层排序。

文档主线：

```text
参数定义
-> 参数之间的精确关系
-> 正确性约束
-> 当前实现约束
-> 硬件资源模型
-> 候选生成
-> 实验验证
```

## 参考资料

- Volkov 和 Demmel, ["Benchmarking GPUs to Tune Dense Linear Algebra"](https://mc.stanford.edu/cgi-bin/images/6/65/SC08_Volkov_GPU.pdf), SC 2008.
  对本项目最有用的结论是：GEMM 调优通常不是选最大 tile，而是在寄存器
  blocking、occupancy 和 memory hierarchy 之间找平衡。
- Nath, Tomov, Dongarra, ["An Improved MAGMA GEMM for Fermi Graphics Processing
  Units"](https://www.netlib.org/lapack/lawnspdf/lawn227.pdf), IJHPCA 2010.
  这篇文章的价值在于把 block tile、thread tile、数据搬运和 occupancy 约束
  分开讨论。
- Li, Tomov, Dongarra, ["Preliminary Results of Autotuning GEMM Kernels for the
  NVIDIA Fermi GPU"](https://www.netlib.org/lapack/lawnspdf/lawn267.pdf), LAWN 267.
  这是最接近本项目“先裁剪候选集，再做实测”的参考。
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/):
  用来确认硬件资源限制、memory hierarchy 和 occupancy 规则。
- [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html):
  用来确认 shared memory bank conflict 和 tile-based matrix multiplication 的基本原则。
- [NVIDIA CUTLASS Efficient GEMM 文档](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html):
  用来参考 block tile -> warp tile -> thread tile 的层次化 GEMM 组织方式。
- hibouwu, [BlockedSizeCherche](https://github.com/hibouwu/ppn-mnist-neural-network/blob/main/Docs/report/BlockedSizeCherche.md):
  参考它把候选参数先分层建模、再用硬件资源约束裁剪的写法。本文件采用
  “硬件预算 -> 参数消耗 -> 约束分类 -> 实测验证”的组织方式。

## 本机 GPU 硬件参数

测试环境：

```text
GPU: NVIDIA GeForce RTX 5070 Laptop GPU
Compute Capability: 12.0 (sm_120)
Driver: 580.159.03
CUDA runtime / nvcc: container image CUDA 13.0
```

通过 `nvidia-smi`、CUDA runtime 和 NVIDIA Blackwell tuning guide 确认到的
关键参数：

| 参数 | 数值 | 对 v7 调参的影响 |
| --- | ---: | --- |
| SM 数量 | 36 | 决定全局并行度；大 tile 会减少 block 数，小尺寸下更容易喂不满所有 SM |
| 全局显存 | 7680 MiB runtime 可见 / 8151 MiB nvidia-smi | 不限制当前 benchmark，但影响最大可测矩阵尺寸 |
| Warp size | 32 | v7 的 lane 映射、`WMITER/WSUBM/WSUBN` 都必须以 32 lane 为基本单位 |
| Max threads / block | 1024 | 派生 `NumThreads` 的硬上限 |
| Max threads / SM | 1536 | 和 register/shared memory 一起决定 active warps 上限 |
| Max resident warps / SM | 48 | occupancy 的分母；`active_warps / 48` 才是本机理论 occupancy |
| Max resident blocks / SM | 24 runtime 可见 | block 数上限；occupancy 模型中要和线程、warp、寄存器、shared memory 限制一起取 min |
| Registers / block | 65536 | 限制单 block 总寄存器 |
| Registers / SM | 65536 | 决定 occupancy；每线程寄存器越多，resident warps 越少 |
| Max registers / thread | 255 | 单线程最多可分配 255 个架构寄存器；是否 spill 取决于 live range、`launch_bounds`、`maxrregcount` 和编译器分配，必须以 ptxas 为准 |
| Static shared memory / block | 49152 B | 当前静态 shared memory 编译上限；v7 候选默认不能超过 48 KiB |
| Opt-in shared memory / block | 101376 B | 只有改成 opt-in dynamic shared memory 后才可用；第一轮 v7 调参先不依赖 |
| Shared memory / SM | 102400 B | 决定每个 SM 可以同时放几个 v7 block |
| L2 cache | 32 MiB | 对大矩阵 B tile/A tile 复用和 cuBLAS reference 都有影响 |
| Memory bus width | 128 bit | 限制显存带宽上限；更需要提高 tile 复用和减少无效 global load |
| Max SM clock | 3090 MHz | 理论算力上限会随 laptop 功耗/温度波动 |
| Max memory clock | 12001 MHz | 理论带宽上限会随功耗/温度波动 |

## 参数定义

v7 的分块层次是：

```text
block tile: BM x BN
warp tile:  WM x WN
thread micro-layout: TM x TN, WNITER, WMITER
K tile: BK
```

各参数含义：

| 参数 | 含义 | 是否独立枚举 |
| --- | --- | --- |
| `BM/BN` | 一个 thread block 负责的 C tile 大小 | 是 |
| `BK` | 每次搬入 shared memory 的 K 维深度 | 是 |
| `WM/WN` | 一个 warp 负责的 C tile 大小 | 是 |
| `TM/TN` | 单个 lane 内 micro-layout 的基本形状 | 是 |
| `WNITER` | warp tile 在 N 方向拆成多少个 sub-tile | 是 |
| `WMITER` | warp tile 在 M 方向拆成多少个 sub-tile | 否，由公式派生 |
| `WSUBM/WSUBN` | warp sub-tile 大小 | 否，由公式派生 |
| `NumThreads` | 一个 block 的线程数 | 否，由 `BM/BN/WM/WN` 派生 |

`NumThreads` 不应该作为独立调参变量。它是 block tile 中 warp tile 数量的结果。

## 精确关系

### Block 到 Warp

```text
warp_tiles_m   = BM / WM
warp_tiles_n   = BN / WN
warps_per_block = warp_tiles_m * warp_tiles_n
NumThreads      = 32 * warps_per_block
```

第一轮不枚举 `NumThreads`，只在派生后保留：

```text
NumThreads in {128, 256}
```

这是一条 search-space policy，不是候选空间的独立维度。

### Warp 到 Lane

```text
WMITER = WM * WN / (32 * TM * TN * WNITER)
WSUBM  = WM / WMITER
WSUBN  = WN / WNITER
```

32 个 lane 必须刚好覆盖一个 warp sub-layout：

```text
(WSUBM / TM) * (WSUBN / TN) == 32
```

代码中的 lane 映射对应：

```cpp
thread_col_in_warp = lane % (WSUBN / TN)
thread_row_in_warp = lane / (WSUBN / TN)
```

如果这个覆盖关系不成立，会出现 lane 越界、重复计算或者 warp tile 有区域没人算。

### Accumulator 和 Register Reuse

单线程 accumulator 数：

```text
accumulators = WMITER * TM * WNITER * TN
             = WM * WN / 32
```

所以固定 `WM/WN` 时，`TM/TN/WNITER` 不改变 accumulator 总数；它们改变的是
accumulator 在 M/N 方向上的组织、A/B fragment 的形状和 shared memory 访问形态。

静态寄存器模型：

```text
reg_m    = WMITER * TM
reg_n    = WNITER * TN
R_static = accumulators + reg_m + reg_n
```

寄存器复用指标：

```text
A_register_reuse = reg_n
B_register_reuse = reg_m
combined_register_reuse = accumulators / (reg_m + reg_n)
```

直观理解：

```text
reg_n 大 -> 一个 A 标量能和更多 B 标量做 FMA
reg_m 大 -> 一个 B 标量能和更多 A 标量做 FMA
combined_register_reuse 高 -> 每次 shared->register load 对应更多 FMA
```

这些指标只用于排序或分组。真实 register count、stack frame 和 spill 必须以
`ptxas -v` 为准。

### Shared Memory 和 Arithmetic Intensity

v7 使用 A/B 双 buffer 做 shared-memory ping-pong staging：

```text
shared_bytes = 2 * (BM * BK + BK * BN) * sizeof(float)
             = 8 * BK * (BM + BN)
```

每个 block tile 在每个 `BK` step 中：

```text
global_load_bytes = 4 * BK * (BM + BN)
fma_flops         = 2 * BM * BN * BK
arithmetic_intensity = fma_flops / global_load_bytes
                     = (BM * BN) / (2 * (BM + BN))
```

`BK` 不改变这个 global-memory arithmetic intensity；它主要影响 shared memory
占用、同步次数、copy 组织和每个 staged tile 的计算时间。

## 正确性约束

下面约束直接来自 warp tile 划分和 lane 映射。不满足时会导致越界、漏算、
重复计算或者错误结果。

```text
BM % WM == 0
BN % WN == 0

NumThreads % 32 == 0

WM * WN 能被 32 * TM * TN * WNITER 整除
WMITER >= 1

WM % WMITER == 0
WN % WNITER == 0

WSUBM % TM == 0
WSUBN % TN == 0

(WSUBM / TM) * (WSUBN / TN) == 32
```

这些约束是 correctness constraint，不是性能启发式。

`NumThreads <= max_threads_per_block` 不是数学正确性约束，而是 launch
feasibility。超过设备每 block 最大线程数时 kernel 不能合法 launch。

## 限制审查记录

| Constraint | Category | Evidence level | Code evidence | Failure consequence | Hard reject | Validation method |
| --- | --- | --- | --- | --- | --- | --- |
| `BM % WM == 0` | mapping_correctness | exact_code_derived | `sgemm_kernels.cuh`, `warp_row/warp_col` 和 grid tile 映射 | missing_coverage / out_of_bounds | yes | unit test, static_assert |
| `BN % WN == 0` | mapping_correctness | exact_code_derived | `warp_col = warp_idx % (BN / WN)` | missing_coverage / out_of_bounds | yes | unit test, static_assert |
| `NumThreads % 32 == 0` | mapping_correctness | exact_code_derived | `warp_idx = threadIdx.x / 32` | malformed warp mapping | yes | unit test, static_assert |
| `NumThreads <= max_threads_per_block` | launch_feasibility | official_documentation | CUDA device property and launch rules | launch_failure | yes | unit test, compile/launch test |
| `WM * WN % (32 * TM * TN * WNITER) == 0` | mapping_correctness | exact_code_derived | `constexpr WMITER = ...` | integer truncation would break mapping | yes | unit test, static_assert |
| `WMITER >= 1` | mapping_correctness | exact_code_derived | loops over `WMITER`, `WSUBM = WM / WMITER` | missing_coverage / divide by zero | yes | unit test, static_assert |
| `WM % WMITER == 0` | mapping_correctness | exact_code_derived | `WSUBM = WM / WMITER` | truncated sub-tile | yes | unit test, static_assert |
| `WN % WNITER == 0` | mapping_correctness | exact_code_derived | `WSUBN = WN / WNITER` | truncated sub-tile | yes | unit test, static_assert |
| `WSUBM % TM == 0` | mapping_correctness | exact_code_derived | `thread_row_in_warp * TM + i` | missing/duplicate coverage | yes | unit test, static_assert |
| `WSUBN % TN == 0` | mapping_correctness | exact_code_derived | `thread_col_in_warp * TN + i` | missing/duplicate coverage | yes | unit test, static_assert |
| `(WSUBM / TM) * (WSUBN / TN) == 32` | mapping_correctness | exact_code_derived | lane mapping at `thread_col_in_warp/thread_row_in_warp` | missing/duplicate coverage | yes | unit test, static_assert |
| `BK % 4 == 0` | implementation_constraint | exact_code_derived | `inner_col_a = threadIdx.x % (BK / 4)` and `float4` A load | unsupported_current_implementation | yes | unit test, static_assert |
| `BN % 4 == 0` | implementation_constraint | exact_code_derived | `inner_col_b = threadIdx.x % (BN / 4)` and `float4` B load | unsupported_current_implementation | yes | unit test, static_assert |
| `TN % 4 == 0` | implementation_constraint | exact_code_derived | C store loop `res_idx_n += 4` with `float4` | unsupported_current_implementation | yes | unit test, static_assert |
| `(NumThreads * 4) % BK == 0` | implementation_constraint | exact_code_derived | `RowStrideA = (NumThreads * 4) / BK` | unsupported_current_implementation | yes | unit test, static_assert |
| `(NumThreads * 4) % BN == 0` | implementation_constraint | exact_code_derived | `RowStrideB = NumThreads / (BN / 4)` | unsupported_current_implementation | yes | unit test, static_assert |
| `(BM * BK) % (4 * NumThreads) == 0` | implementation_constraint | exact_code_derived | A copy loop has no remainder path | missing_coverage | yes | unit test, static_assert |
| `(BN * BK) % (4 * NumThreads) == 0` | implementation_constraint | exact_code_derived | B copy loop has no remainder path | missing_coverage | yes | unit test, static_assert |
| `shared_bytes <= 49152` | resource_feasibility | official_documentation / compile path | static shared-memory path | compile_failure / resource_exceeded | yes for current path | compile test, resource probe |
| `NumThreads in {128,256}` | search_space_policy | empirical_policy | generator first-round policy | performance_risk_only | no | generation summary, benchmark |
| `accumulators <= 128` | search_space_policy | empirical_policy | generator first-round policy | performance_risk_only | no | generation summary, ptxas |
| `R_static` ranking | performance_hypothesis | analytical_model | Python model only | performance_risk_only | no | ptxas, resource probe, benchmark |

检查顺序会影响 first-failure `rejection_reason`。当前 Python 生成器先检查映射
正确性，再检查当前实现约束，再检查搜索策略和资源限制。CUDA `static_assert`
覆盖会导致编译失败的模板期条件；runtime shape contract 和 ptxas/API 资源
必须在后续阶段验证。

## 当前实现约束

下面约束不是 GEMM 理论限制，而是当前 v7 的 `float4` copy、`float4` C store
和无 tail path 实现要求。理论上可以通过 scalar tail path 或更通用的 copy
逻辑放宽，但当前 kernel 没有。

### Vectorized Copy / Store

```text
BK % 4 == 0
BN % 4 == 0
TN % 4 == 0
```

原因：

```text
BK % 4 == 0:
  A tile 使用 float4 读取，inner_col_a = threadIdx.x % (BK / 4)。

BN % 4 == 0:
  B tile 使用 float4 读取，inner_col_b = threadIdx.x % (BN / 4)。

TN % 4 == 0:
  C store 的 res_idx_n 每次加 4，并用 float4 读写 C。
```

当前 copy loop 还要求线程集合刚好覆盖 shared tile，没有尾部 copy：

```text
(NumThreads * 4) % BK == 0
(NumThreads * 4) % BN == 0
(BM * BK) % (4 * NumThreads) == 0
(BN * BK) % (4 * NumThreads) == 0
```

更具体地说：

```text
(NumThreads * 4) % BK == 0:
  让 RowStrideA = (NumThreads * 4) / BK 是整数。

(BM * BK) % (4 * NumThreads) == 0:
  等价于 BM 能被 RowStrideA 整除，否则 A tile 最后一段 row 没有 copy。

(NumThreads * 4) % BN == 0:
  让 RowStrideB = NumThreads / (BN / 4) 是整数。

(BN * BK) % (4 * NumThreads) == 0:
  等价于 BK 能被 RowStrideB 整除，否则 B tile 最后一段 K row 没有 copy。
```

### Copy Loop 展开示例

baseline:

```text
BM=128, BN=128, BK=16, NumThreads=128

A:
inner_row_a = threadIdx.x / 4
inner_col_a = threadIdx.x % 4
RowStrideA = (128 * 4) / 16 = 32
offset in {0, 32, 64, 96}

每个 thread 复制一个 row 上的 4 个连续 float。
threadIdx.x 的 inner_row_a 覆盖 0..31，offset 覆盖 4 段 row。
inner_col_a 覆盖 K 方向的 {0..3, 4..7, 8..11, 12..15}。
所以 A tile 的 128 x 16 元素刚好复制一次。

B:
inner_row_b = threadIdx.x / 32
inner_col_b = threadIdx.x % 32
RowStrideB = 128 / 32 = 4
offset in {0, 4, 8, 12}

inner_row_b 覆盖 0..3，offset 覆盖 K 方向 0..15。
inner_col_b 覆盖 N 方向 32 个 float4，即 0..127。
所以 B tile 的 16 x 128 元素刚好复制一次。
```

BK=8 对照：

```text
BM=128, BN=128, BK=8, NumThreads=128

A:
inner_row_a = threadIdx.x / 2
inner_col_a = threadIdx.x % 2
RowStrideA = 64
offset in {0, 64}
覆盖 128 rows x 8 columns。

B:
inner_row_b = threadIdx.x / 32
inner_col_b = threadIdx.x % 32
RowStrideB = 4
offset in {0, 4}
覆盖 8 rows x 128 columns。
```

64-accumulator / 256-thread 对照：

```text
BM=128, BN=128, BK=16, NumThreads=256

A:
inner_row_a = threadIdx.x / 4
inner_col_a = threadIdx.x % 4
RowStrideA = 64
offset in {0, 64}
覆盖 128 rows x 16 columns。

B:
inner_row_b = threadIdx.x / 32
inner_col_b = threadIdx.x % 32
RowStrideB = 8
offset in {0, 8}
覆盖 16 rows x 128 columns。
```

这些推导说明四条 copy divisibility 条件在当前实现中是 coverage 条件。它们
不是 GEMM 数学限制；如果以后增加 remainder loop，可以删除或降级。

### Float4 对齐和运行时输入契约

当前 v7 的准确输入契约：

```text
layout:
  row-major, tightly packed A[M,K], B[K,N], C[M,N]

leading dimension:
  lda = K
  ldb = N
  ldc = N
  当前接口没有显式 lda/ldb/ldc 参数。

tile multiple:
  M % BM == 0
  N % BN == 0
  K % BK == 0

pointer alignment:
  cudaMalloc base pointer 足够满足 float4 对齐。

offset alignment:
  A: row * K + k_tile + inner_col_a * 4
     需要 K % 4 == 0 且 k_tile 是 4 的倍数。
  B: k_row * N + block_col * BN + inner_col_b * 4
     需要 N % 4 == 0 且 BN % 4 == 0。
  C: row * N + block_col * BN + warp_col * WN + sub_col + thread_col * TN + res_idx_n
     需要 N % 4 == 0、BN % 4 == 0，并由 WNITER/WSUBN/TN 约束保证列偏移是 4 的倍数。

tail behavior:
  没有 tail path，没有边界保护。
```

当前 benchmark 使用紧密 row-major 方阵，因此 `M=N=K` 且 leading dimension
就是 `N`。这不是 kernel 自动支持任意 leading dimension。

### C Store 覆盖

C store 使用：

```text
for res_idx_n in 0, 4, 8, ...
  float4 load/store C
```

`TN=4` 对应一次 `float4` store；`TN=8` 对应两次 `float4` store。lane 的 C
起始列由：

```text
w_sub_col_idx * WSUBN + thread_col_in_warp * TN
```

决定。由于 `WSUBN % TN == 0` 且 `(WSUBM/TM) * (WSUBN/TN) == 32`，每个 lane
负责的 micro tile 不重叠且覆盖完整 warp sub-layout。由于 `TN % 4 == 0`，
每个 lane 的列内 store 以 4 个 float 为步长。

### Runtime Input Contract

`M % BM`、`N % BN`、`K % BK` 属于运行时输入契约，不应在纯参数枚举阶段应用。
当前 benchmark 入口只测方阵 `N`，而 v7 baseline launch 只在当前实例需要的
tile multiple 条件满足时运行。

## 硬件资源模型

### Shared Memory

当前静态 shared-memory 编译路径先使用硬过滤：

```text
shared_bytes <= 49152
```

在本机 `Shared memory / SM = 102400 B` 时，如果 shared memory 是主要限制，
单 SM 可容纳 block 数大致是：

```text
actual_shared_bytes <= 48 KiB -> shared memory 允许 2 blocks / SM
actual_shared_bytes <= 32 KiB -> shared memory 允许 3 blocks / SM
actual_shared_bytes <= 24 KiB -> shared memory 允许 4 blocks / SM
```

这里的 `actual_shared_bytes` 应该包含静态 shared memory、动态 shared
memory，以及可能的编译/对齐开销。最终 active block 数仍然可能先被寄存器、
线程数或者 max resident block 数限制。

当前 v7 的 global-to-shared 搬运使用普通 `float4` load/store，没有使用显式
异步 copy。代码没有建立由 `cp.async` 或等价机制保证的 global-load/compute
overlap。不同 warp 的 latency hiding、L1/L2 行为或者编译器有限重排仍可能
带来部分延迟隐藏，但不能因为存在两块 shared buffer 就把它当成真实异步
pipeline。

实际 shared memory 声明是：

```cpp
__shared__ float as[2][BM * BK];
__shared__ float bs[2][BK * BN];
```

因此模型：

```text
shared_bytes = 2 * (BM * BK + BK * BN) * 4
```

对当前 v7 是精确的。baseline 中：

```text
shared_bytes_model = 32768
cudaFuncAttributes.sharedSizeBytes = 32768
ptxas smem = 32768 bytes
```

当前没有 padding、额外 shared-memory 数组或编译器额外静态 shared memory。

### Occupancy

粗模型：

```text
warps_per_block = NumThreads / 32

registers_per_block_est =
    ceil_to_alloc_unit(registers_per_thread * NumThreads)

blocks_limited_by_registers =
    floor(registers_per_sm / registers_per_block_est)

blocks_limited_by_shared_memory =
    floor(max_smem_per_sm / actual_shared_bytes)

active_blocks_per_sm = min(
    max_blocks_per_sm,
    floor(max_threads_per_sm / NumThreads),
    floor(max_warps_per_sm / warps_per_block),
    blocks_limited_by_registers,
    blocks_limited_by_shared_memory
)

active_warps = active_blocks_per_sm * warps_per_block
occupancy    = active_warps / 48
```

本机第一轮模型代入：

```text
max_blocks_per_sm  = 24
max_threads_per_sm = 1536
max_warps_per_sm   = 48
registers_per_sm   = 65536
max_smem_per_sm    = 102400
```

`registers_per_thread` 必须以 `ptxas -v` 输出为准；模型中的
`ceil_to_alloc_unit` 只是提醒寄存器分配有粒度，不能把估算值当成精确值。
最终 occupancy 需要用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor`、
`ptxas` register 输出或者 Nsight Compute 来确认。

baseline resource probe:

```text
numRegs = 166
sharedSizeBytes = 32768
activeBlocksPerSM = 3
activeWarpsPerSM = 12
occupancy = 25%
```

寄存器和 shared memory 的单项估算都可能解释 3 blocks/SM。由于寄存器分配
粒度不能由当前模型精确复现，resource probe 输出应保留多个可能限制来源：

```text
limitingResources = shared_memory|registers_estimate
```

或者在无法唯一归因时标记为：

```text
multiple_or_uncertain
```

不要把单项粗估写成唯一确定的 primary limit；active blocks 以 occupancy API
为准。

### Grid-wave 指标

grid-wave 是性能分析指标，不是正确性约束：

```text
grid_blocks = ceil(M / BM) * ceil(N / BN)
resident_capacity = SM_count * activeBlocksPerSM
grid_waves = grid_blocks / resident_capacity
```

baseline `BM=BN=128`、`SM_count=36`、`activeBlocksPerSM=3`：

```text
N=1024:
  grid_blocks = 8 * 8 = 64
  resident_capacity = 36 * 3 = 108
  grid_waves = 0.59

N=2048:
  grid_blocks = 16 * 16 = 256
  grid_waves = 2.37

N=4096:
  grid_blocks = 32 * 32 = 1024
  grid_waves = 9.48
```

这解释了为什么小尺寸下 v7 可能喂不满 GPU，而大尺寸更能体现 warp tiling 的
复用优势。

### Global-memory Transaction 模型

baseline 中，每个 thread 做 `float4` global load。能从代码精确推出的是地址
模式，不能仅凭 `float4` 断言实际 transaction 完全合并。

A tile 每个 warp：

```text
threadIdx.x = lane
inner_row_a = lane / 4
inner_col_a = lane % 4
address_A = base + (inner_row_a * K + inner_col_a * 4) * sizeof(float)
```

所以每 4 个 lane 访问同一 row 上连续的 4 个 `float4`，即连续 64 bytes；
一个 warp 覆盖 8 个 row 的 64-byte 片段。A 的访问是“每 4 lane 一段连续”，
不是整个 warp 单段连续。

B tile 每个 warp：

```text
inner_row_b = lane / 32 = 0
inner_col_b = lane % 32
address_B = base + (inner_row_b * N + inner_col_b * 4) * sizeof(float)
```

所以一个 warp 访问同一 row 上连续 32 个 `float4`，即连续 512 bytes。

实际 32-byte sector 数、L1/L2 合并和 replay 需要 Nsight Compute 验证：

```text
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
```

### Shared-memory Bank 模型

bank 粗模型：

```text
bank = word_address % 32
```

baseline A fragment load：

```text
address_A_smem =
  dot_idx * BM
  + warp_row * WM
  + w_sub_row_idx * WSUBM
  + thread_row_in_warp * TM
  + i

thread_row_in_warp = lane / 4
```

同一 `thread_row_in_warp` 的 4 个 lane 读取同一地址，这是 broadcast，不是
不同地址 bank conflict。不同 `thread_row_in_warp` 之间可能映射到相同 bank；
例如 `TM=8` 时，`thread_row=0` 和 `thread_row=4` 对同一 `i` 的 bank 相同
但地址不同，存在 shared bank conflict 风险。

baseline B fragment load：

```text
address_B_smem =
  dot_idx * BN
  + warp_col * WN
  + w_sub_col_idx * WSUBN
  + thread_col_in_warp * TN
  + i

thread_col_in_warp = lane % 4
```

相同 `thread_col_in_warp` 的多个 lane 读取同一地址，属于 broadcast。不同列
组通常落在不同 bank 组。是否出现 replay 和实际冲突程度需要 Nsight Compute
验证：

```text
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
smsp__stall_mio_throttle
```

因此 `TN/WNITER/WSUBN` 相关规则目前应作为 performance_hypothesis，而不是硬
过滤。

### Baseline Resource Record

当前 baseline：

```text
BM=128, BN=128, BK=16
WM=64, WN=64, WNITER=4
TM=8, TN=4, WMITER=1
NumThreads=128

accumulators = 128
reg_m        = 8
reg_n        = 16
R_static     = 152

ptxas registers/thread = 166
stack frame            = 0 bytes
spill stores           = 0 bytes
spill loads            = 0 bytes

baseline residual = R_ptxas - R_static = 166 - 152 = 14
```

这个 residual 只记录当前 baseline 的经验结果，不能当作其他候选的固定额外
寄存器开销。完整 baseline 资源记录见
`scripts/v7_tuning/v7_baseline_resources.json`。

## 候选生成

第一轮候选空间：

```text
BM, BN in {64, 128, 256}
BK     in {8, 16, 32}
WM, WN in {32, 64, 128}
TM, TN in {(4,4), (8,4), (4,8), (8,8)}
WNITER in {1, 2, 4, 8}
```

`NumThreads` 不独立枚举。对每个组合先计算：

```text
NumThreads = 32 * (BM / WM) * (BN / WN)
```

然后第一轮只保留：

```text
NumThreads in {128, 256}
accumulators in {32, 64, 128}
shared_bytes <= 49152
```

其中 `accumulators in {32, 64, 128}` 来自当前第一轮生成器的合法结果分布。
`accumulators <= 128` 只是 search-space policy，用来控制候选空间和编译成本；
它不是硬件正确性限制，也不是所有 GPU/所有 kernel 形态的通用上限。

候选裁剪顺序：

```text
1. 派生 NumThreads / WMITER / WSUBM / WSUBN
2. correctness constraints
3. implementation constraints
4. static shared memory <= 48 KiB
5. first-round search-space policy
6. ranking/grouping metrics: R_static, register reuse, occupancy estimate
```

当前生成器输出：

```text
raw combinations: 3888
valid candidates: 500
rejected candidates: 3388

valid accumulator counts:
  32: 28
  64: 160
  128: 312

valid thread counts:
  128: 276
  256: 224
```

## 第一批实例化候选

资源探测阶段只实例化少量候选，不把 500 个合法候选全部编译：

| Name | BM | BN | BK | WM | WN | WNITER | TM | TN | Derived Threads | 选择理由 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 128 | 128 | 16 | 64 | 64 | 4 | 8 | 4 | 128 | 当前基线 |
| bk8 | 128 | 128 | 8 | 64 | 64 | 4 | 8 | 4 | 128 | shared memory 更少，但同步更多 |
| acc64_threads256 | 128 | 128 | 16 | 32 | 64 | 2 | 4 | 4 | 256 | 64 accumulator，更多 warp/block |
| m_wide | 128 | 128 | 16 | 128 | 32 | 2 | 8 | 4 | 128 | 加强 M 方向 warp tile |
| n_wide | 128 | 128 | 16 | 32 | 128 | 4 | 4 | 8 | 128 | 加强 N 方向 warp tile |
| micro_layout_4x4 | 128 | 128 | 16 | 64 | 64 | 4 | 4 | 4 | 128 | accumulator 总数不变，改变 micro-layout |

这些候选由 `scripts/v7_tuning/generate_probe_registry.py` 生成 registry，不复制
kernel 本体。

## 实验验证

参数模型测试：

```text
python3 -m unittest scripts/v7_tuning/test_candidate_model.py
```

候选生成：

```text
python3 scripts/v7_tuning/generate_candidates.py
```

资源探测：

```text
cmake -S . -B build_verify -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build_verify -j --target v7_resource_probe
./build_verify/v7_resource_probe
```

baseline API 资源探测结果：

```text
numRegs = 166
sharedSizeBytes = 32768
localSizeBytes = 0
activeBlocksPerSM = 3
activeWarpsPerSM = 12
theoreticalOccupancy = 25%
limitingResources = multiple_or_uncertain:shared_memory|registers_estimate
```

正式性能测试不要只因为某个候选赢了一个尺寸就选择它。优先选择满足下面条件的
候选：

```text
1. 所有 benchmark 尺寸都 correctness 通过
2. 在 2048 上距离最优不超过 2-3%
3. 在 4096 上最优或者接近最优
4. ptxas / Nsight Compute 给出的资源使用能解释性能表现
```

如果两个候选性能接近，优先选择 register pressure 更低、跨 trial 更稳定的
版本。这样的参数更适合作为后续 v8a/v8b 调参的基础。
