# GEMMsm110 — Thor/SM110 Blackwell TCGen05/TMEM GEMM

> **目标架构**：NVIDIA DRIVE AGX Thor (SM110, compute capability 11.0, Blackwell)  
> **路径**：tcgen05/TMEM（非 SM120 的 `mma.sync.aligned.kind::f8f6f4` 路径）  
> **当前状态**：完整 FP16 × FP16 → FP32 GEMM，包含 cuBLAS 校验与 GFLOPS

## Backend 实验路线

本目录采用逐阶段、逐变量隔离的 backend 设计。除当前阶段明确声明的
变量外，所有 backend 必须保持相同的 FP16 × FP16 → FP32 数据类型、
矩阵布局、输入数据、数值误差阈值、warmup/repeat 和计时方式。

### 阶段 0：correctness / reference

| Backend | 实验变量 | 目的 |
| --- | --- | --- |
| `tc0` | 传统 CUDA Tensor Core；不使用 TMA、TCGen05 或 TMEM | 自有 correctness/performance baseline |

### 阶段 1：TMA + TCGen05 minimal path

| Backend | 实验变量 | 目的 |
| --- | --- | --- |
| `tc1a` | 2D TMA、16B/linear SMEM、TCGen05/TMEM | 最小 2D TMA 路径，同时保留 bank conflict 对照 |
| `tc1b` | 3D TMA、16B/linear SMEM、TCGen05/TMEM | 仅改变 TMA descriptor rank，隔离 3D TMA 影响 |

`tc1b` 将 A/B 表示为 rank-3 tensor，第三维为 batch/L；当前方阵测试
使用 `L=1`。它不允许顺带改变 swizzle、stage 数或调度策略。

### 阶段 2：bank conflict isolation

| Backend | 实验变量 | 目的 |
| --- | --- | --- |
| `tc2a` | `tc1a` + 128B SMEM swizzle | 2D TMA 下只改变 SMEM layout/swizzle descriptor |
| `tc2b` | `tc1b` + 128B SMEM swizzle | 3D TMA 下只改变 SMEM layout/swizzle descriptor |

`tc1a` 对 `tc2a`、`tc1b` 对 `tc2b` 是严格配对实验。tile shape、
CTA threads 和单 stage 执行顺序必须相同。

### 阶段 3：latency hiding / overlap

| Backend | 实验变量 | 目的 |
| --- | --- | --- |
| `tc3` | `tc2a` + multi-stage pipeline、double buffer、TMA mbarrier | 用 prefetch 与 load/compute overlap 隔离延迟隐藏收益 |

`tc3` 的直接基线固定为 `tc2a`，不同时引入 3D TMA、warp
specialization 或 cluster MMA。

当前 `tc3` 已改为参考 `learn-cuda/02e_matmul_sm100/matmul_v3.cu`
编写的 raw CUDA/inline-PTX kernel。kernel 自行完成 `CUtensorMap`
坐标计算、TMA、mbarrier phase、TCGen05 MMA、TMEM 分配/回读和
epilogue，不使用 CuTe `Tensor`、`TiledMMA`、TMA atom、CUTLASS
collective 或 CUTLASS scheduler。

### 阶段 4：scheduling and cluster

| Backend | 实验变量 | 目的 |
| --- | --- | --- |
| `tc4a` | `tc3` + warp specialization | TMA producer、MMA consumer 和 epilogue/readback 分工 |
| `tc4b` | `tc3` + 2-SM cluster MMA | 使用 cluster dimensions 与 `cta_group::2`，隔离 2-SM MMA 收益 |
| `tc4c` | `tc3` + warp specialization + 2-SM cluster MMA | 验证两种优化能否叠加 |

### 阶段 5：persistent scheduling

| Backend | 实验变量 | 目的 |
| --- | --- | --- |
| `tc5a` | `tc4c` + persistent kernel、static scheduling、TMEM double buffer、epilogue warp specialization | 静态 persistent 调度 |
| `tc5b` | `tc5a` 的计算路径 + hardware CLC dynamic scheduling | 只比较静态 tile 分配与硬件动态调度 |

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
| `tc1a`、`tc1b` | 已实现；共享单-stage kernel，仅 descriptor rank 不同 |
| `tc2a`、`tc2b` | 已实现；与 tc1 配对，仅 SMEM atom 改为 SW128 |
| `tc3` | 已实现；自有 raw CUDA/inline-PTX，固定 128×128×64、2-stage、2D TMA SW128 pipeline |
| `tc4a` | 已实现；warp0 MMA consumer、warp1 TMA producer |
| `tc4b` | 已实现；固定 256×128×64、2-stage、2×1 cluster |
| `tc4c` | 已实现；与 tc4b 同 kernel，仅启用独立 TMA producer warp |
| `tc5a` | 已实现；继承 tc4c 的 256×128×64 2-SM 计算路径，resident cluster 静态 grid-stride 调度、双 TMEM accumulator slot |
| `tc5b` | 已实现；与 tc5a 共用计算路径，改用 hardware CLC 动态取消/领取未启动 cluster |

`tc5a` 与 `tc5b` 都使用 warp0 MMA、warp1 TMA，warpgroup1
（warp4–7）专门完成 TMEM readback/epilogue；warp2/3 留空。完整
epilogue warpgroup 是 TMEM data-path lane ownership 的要求，不能用两个
warp 伪造四个 logical TMEM copy slice。`tc5b` 的 SASS 包含
`UGETNEXTWORKID.BROADCAST`，`tc5a` 的对应 kernel 不包含该指令。

### CUTLASS 依赖边界

- `cutlass` backend：唯一允许实例化 CUTLASS 官方高层 GEMM 的性能
  reference。
- `tc3`：完全自有 kernel，仅使用
  `sm110_ptx_helpers.cuh` 中的薄 PTX/Driver API helper；这些 helper
  的指令写法参考 CUDA PTX 文档、`learn-cuda` 和 CUTLASS 底层 arch
  wrapper，但不包含或实例化 CUTLASS/CuTe。
- `tc1/tc2/tc4/tc5`：当前 kernel 调度是自有代码，但仍使用部分
  CuTe tensor/atom helper；后续按同样方式逐阶段迁移。迁移完成前，
  不把这些后端描述为“无 CUTLASS/CuTe 依赖”。

raw TCGen05 路径使用 K-major A 与 K-major B。逻辑矩阵
`B[K,N]` 在 benchmark 初始化阶段预排布为物理 `B[N,K]`，该一次性
转换不进入 kernel 计时；cuBLAS/CUTLASS reference 继续使用原始
`B[K,N]`，两份数据逐元素逻辑等价。

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

## 当前阶段性结果

Thor、N=1024 的单次 100-repeat 测量如下；正式结论仍以多 trial sweep
的均值和标准差为准。

| Backend | GFLOPS | 配对结论 |
| --- | ---: | --- |
| `tc0` | 3,576 | WMMA correctness baseline |
| `tc1a` | 8,621 | 2D linear/INTER |
| `tc1b` | 8,621 | 3D descriptor 与 2D 基本持平 |
| `tc2a` | 22,150 | 只改 SW128 后约为 tc1a 的 2.57× |
| `tc2b` | 22,108 | 只改 SW128 后约为 tc1b 的 2.56× |
| `tc3` | 37,410 | raw 2D TMA/TCGen05 2-stage pipeline |
| `tc4a` | 25,582 | 仍为 CuTe-helper 路径；raw 迁移前不再与 tc3 作严格配对结论 |
| `tc4b` | 29,140 | 仍为 CuTe-helper 2-SM cluster 路径 |
| `tc4c` | 29,939 | 仍为 CuTe-helper warp-specialized 2-SM 路径 |
| `tc5a` | 30,828 | 仍为 CuTe-helper static persistent 路径 |
| `tc5b` | 29,939 | 仍为 CuTe-helper hardware CLC 路径 |

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
│   ├── custom_sm110_gemm.cuh   # 迁移期旧实现；完成拆分后删除
│   ├── sm110_backend_registry.cuh # backend 名称、元数据和 runner 注册
│   ├── sm110_ptx_helpers.cuh   # 自有 kernel 使用的薄 PTX/Driver API helper
│   ├── backends/               # 按实验阶段拆分的自有 backend
│   │   ├── tc0_baseline.cuh
│   │   ├── tc1_tc2_tma.cuh     # tc1/tc2 共用模板，避免额外变量
│   │   ├── tc3_pipeline.cuh
│   │   ├── tc4a_warp_specialized.cuh
│   │   ├── tc4bc_cluster.cuh   # tc4b/tc4c 共用模板
│   │   └── tc5_persistent.cuh  # tc5a static / tc5b CLC 共用计算路径
│   ├── requant/                 # NVFP4/E2M1 尾处理后端
│   ├── tc3_gemm_kernel.cuh     # 独立 TMEM minimal sanity probe
│   ├── tc4_gemm_kernel.cuh     # 早期 pipeline 设计稿（benchmark 不使用）
│   └── tc5_gemm_kernel.cuh     # 早期 probe（benchmark 不使用）
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

# 仅编译不运行
./build_and_run.sh build-only

# 清理编译产物
./build_and_run.sh clean

# 每个 backend 最长运行 30 秒
./build_and_run.sh 1024 tc5a 30
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
