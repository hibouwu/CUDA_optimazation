# GEMMsm110 — Thor/SM110 Blackwell TCGen05/TMEM GEMM

> **目标架构**：NVIDIA DRIVE AGX Thor (SM110, compute capability 11.0, Blackwell)  
> **路径**：tcgen05/TMEM（非 SM120 的 `mma.sync.aligned.kind::f8f6f4` 路径）  
> **当前状态**：完整 FP16 × FP16 → FP32 GEMM，包含 cuBLAS 校验与 GFLOPS

## 版本

| 版本 | 定位 | 状态 |
| --- | --- | --- |
| `cutlass` | CUTLASS 官方 Blackwell Auto schedule（示例 71 默认策略） | 可运行、GFLOPS、数值校验 |
| `tc3` | 自有 cooperative-copy + TCGen05/TMEM GEMM | 可运行、GFLOPS、数值校验 |
| `tc4` | 自有 TMA + TCGen05/TMEM GEMM | 可运行、GFLOPS、数值校验 |
| `tc5a` | 自有 static persistent TMA + TCGen05/TMEM GEMM | 可运行、GFLOPS、数值校验 |
| `tc5b` | 自有硬件 CLC persistent TMA + TCGen05/TMEM GEMM | 可运行、GFLOPS、数值校验 |

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
│   ├── custom_sm110_gemm.cuh  # 自有 tc3/tc4/tc5 数值 GEMM kernels
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
./build_and_run.sh 1024 cutlass
./build_and_run.sh 1024 tc3
./build_and_run.sh 1024 tc4
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
