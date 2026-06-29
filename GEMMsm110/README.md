# GEMMsm110 — Thor/SM110 Blackwell TCGen05/TMEM GEMM Probe

> **目标架构**：NVIDIA DRIVE AGX Thor (SM110, compute capability 11.0, Blackwell)  
> **路径**：tcgen05/TMEM（非 SM120 的 `mma.sync.aligned.kind::f8f6f4` 路径）  
> **当前状态**：指令通路 probe，非完整数值 GEMM

## 版本

| 版本 | 定位 | 状态 |
| --- | --- | --- |
| `tc3` | TCGen05/TMEM 最小指令通路 probe (alloc/st/ld/dealloc) | 可运行 |
| `tc4` | Warp-specialized pipeline scaffold (MMA/Scheduler/Load/Epilogue 8-warp 分工) | scaffold（launch disabled） |
| `tc5a` | Static CLC persistent worker + tcgen05 probe | 可运行 |
| `tc5b` | Dynamic CLC (software atomic work queue) + tcgen05 probe | 可运行 |

## 前置条件

- GPU：Thor/SM110（非 SM110 设备上 tc3/tc5 会自动 skip）
- CUDA Toolkit：13.0+
- 编译器：需支持 C++17

## 目录结构

```
GEMMsm110/
├── build_and_run.sh            # 编译 + 运行脚本
├── include/
│   ├── gemm_common.cuh         # CHECK_CUDA/CHECK_CUBLAS 宏、kWarmup/kRepeat 等
│   ├── gemm_benchmark.cuh      # 输入构造、精度对比、benchmark_kernel
│   ├── requant/                 # NVFP4/E2M1 尾处理后端
│   ├── tc3_gemm_kernel.cuh     # TCGen05/TMEM alloc/st/ld/dealloc 最小 probe
│   ├── tc4_gemm_kernel.cuh     # Warp-specialized pipeline scaffold
│   └── tc5_gemm_kernel.cuh     # CLC persistent worker probe (static + dynamic)
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
./build_and_run.sh 1024 tc3
./build_and_run.sh 1024 tc5a
./build_and_run.sh 1024 tc5b

# 仅编译不运行
./build_and_run.sh build-only

# 清理编译产物
./build_and_run.sh clean

# tc5 控制 worker 数量（环境变量）
TC5_SM110_WORKERS_PER_SM=3 ./build_and_run.sh 1024 tc5a
```

编译产物在 `build/` 目录下，运行时在 `build/` 内生成 `sgemm_sm110_benchmark.csv`。

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
