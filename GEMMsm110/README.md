# GEMMsm110 — Thor/SM110 Blackwell TCGen05/TMEM GEMM

> **目标架构**：NVIDIA DRIVE AGX Thor (SM110, compute capability 11.0, Blackwell)  
> **路径**：tcgen05/TMEM（非 SM120 的 `mma.sync.aligned.kind::f8f6f4` 路径）  
> **当前状态**：已验证 FP16 × FP16 → FP32 GEMM 路径到 `tc5a`，
> `tc5a` 是当前推荐的高性能路径；新增 `tc6` FP16 × FP16 → NVFP4
> fused epilogue backend。所有有效 FP32 结果都以 cuBLAS Tensor Core
> 输出为参考。

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

### 阶段 5：persistent scheduling

| Backend | 实验变量 | 目的 |
| --- | --- | --- |
| `tc5a` | `tc4a` 计算路径 + resident persistent workers + static grid stride | 静态 persistent 调度 |
| `tc5b` | `tc5a` 的计算路径 + software dynamic work queue | 动态领取 tile 的实验路径；当前暂停计入有效结果 |

`tc5a/tc5b` 不再依赖未验证的 2-SM `cta_group::2` 路径，而是复用已经
通过 finite-input correctness 的 `tc4a` 1-SM mainloop。这样阶段 5
先把 persistent 调度做成可验证实现；2-SM specialization 后续作为独立
优化恢复。

当前推荐的 FP32 性能 backend 是 `tc5a`。它用 resident persistent
worker 和静态 grid-stride 分配 tile，避免了大量 CTA launch/scheduling
tail，也没有 `tc5b` 每个 tile 动态领取 work item 的原子和同步开销。
`tc5b` 保留为“动态 persistent 调度是否值得”的源码实验；在 `4096`
复测中出现过 timeout，主 benchmark 暂停把它计入有效结果。

`tc4b/tc4c` 当前不计入有效性能结果。修正输入生成 bug 后，2-SM
`cta_group::2` 路径暴露出 TMEM accumulator layout 映射问题；主
benchmark 会把这些 backend 标为 unavailable，直到 2-SM accumulator
行/列 ownership 被重新验证。

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
| `tc4b`、`tc4c` | 暂停计入结果；2-SM `cta_group::2` accumulator layout 需要重新验证 |
| `tc5a` | 已验证；`tc4a` 计算路径 + resident static persistent scheduler |
| `tc5b` | 暂停计入结果；software dynamic work queue 在 4096 复测中出现 timeout |
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

当前边界处理在 `tc5a/tc5b/tc6` 上保留为有效路径。`tc5a/tc5b` 的
完整 tile fast path 要求存在 `128x256x128` 整 tile；K tail、M/N
边界、小矩阵会由 CUDA cleanup kernel 补齐。`tc6` 的完整 tile fast
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

下面是修正输入生成 bug 后，在 Thor 上重新跑的 3-trial 小 sweep
结果。旧图表和旧 sweep 中包含 unsigned wrap 输入，不能再作为结论
引用。

`M=N=K=1024` 主要用于快速 correctness smoke。这个规模的 tile 数少，
TMA、TMEM 分配/回读、barrier 和 kernel 固定开销占比高，不代表最终
高性能 full-tile 路径的上限。

| Backend | GFLOPS | 配对结论 |
| --- | ---: | --- |
| `cublas_tc` | 102,801 | cuBLAS Tensor Core reference |
| `cutlass` | 104,718 | CUTLASS official Blackwell auto-schedule reference |
| `tc0` | 3,574 | WMMA correctness baseline |
| `tc1a/tc1b` | unavailable | linear SMEM descriptor 路径暂停验证 |
| `tc2a` | 28,330 | raw SW128 single-stage TCGen05，matched=1 |
| `tc2b` | 28,267 | raw 3D TMA + SW128 single-stage TCGen05，matched=1 |
| `tc3` | 37,364 | raw 2D TMA/TCGen05 2-stage pipeline，matched=1 |
| `tc4a` | 45,530 | raw 2D TMA/TCGen05 warp-specialized K8/N256 pipeline，matched=1 |
| `tc4b/tc4c` | unavailable | 2-SM `cta_group::2` accumulator layout 暂停验证 |
| `tc5a` | 45,599 | static persistent 1-SM TCGen05，matched=1 |
| `tc5b` | unavailable | dynamic work queue 路径暂停稳定性验证 |
| `tc6` | 28,318 | resident persistent mainloop + fused NVFP4 epilogue，matched=1 |

更能代表最终 FP32 backend 的是大 full-tile 场景。`tc5a` 在
`4096x4096x4096` 上达到同进程 cuBLAS Tensor Core 的约 85%，因此当前
性能目标应按 `tc5a` 静态 persistent 路径陈述，而不是按 `tc5b` 动态
work queue 或 1024 smoke 陈述。

| Shape | Backend | cuBLAS GFLOP/s | Backend GFLOP/s | Ratio | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| 2048³ | `tc4a` | 130,458 | 75,967 | 0.58x | warp-specialized non-persistent，matched=1 |
| 2048³ | `tc5a` | 130,458 | 78,878 | 0.61x | static persistent，matched=1 |
| 4096³ | `tc3` | 63,255 | 38,931 | 0.62x | multi-stage non-persistent，matched=1 |
| 4096³ | `tc5a` | 63,255 | 53,669 | 0.83x | 当前推荐 FP32 backend，matched=1 |

`tc5b` 的动态 work queue 在临时复测中可以产出 matched=1 的结果，但
也出现过 timeout，因此不再作为有效性能 backend 统计。这个结论对面试
解释很重要：当前完整调度策略是 `tc5a` 的 resident static persistent，
不是尚未稳定的动态领取。

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
