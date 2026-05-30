# CUDA Optimization Roadmap

这个目录把 `/home/jianyeshi/Note/notes/CUDA` 的笔记和 `../cuda_code` 的课程代码整理成一个可逐步扩展的优化练习仓库。目标不是只保存最终答案，而是每个算子都保留从 naive 到库函数基线的版本，方便用 Nsight Compute 对比每一步优化到底解决了什么瓶颈。

## 当前目录

```txt
CUDA_optimazation/
├── README.md
├── CMakeLists.txt
├── REDUCE/
│   └── src/main.cu
└── GEMM/
    └── src/main.cu
```

## 总体路线

| 阶段 | 主题 | 目标 | 对应参考 |
| --- | --- | --- | --- |
| 0 | CUDA 基础 | 线程模型、显存层级、host/device 分配与拷贝、计时与错误检查 | `notes/CUDA/2_*`, `cuda_code/course1`, `course2` |
| 1 | Reduce | 理解树形归约、shared memory、bank conflict、warp shuffle、grid-stride loop | `notes/CUDA/4_*`, `cuda_code/course4` |
| 2 | Softmax / Norm | 把 reduce 模式放入真实算子，处理 max/sum 两段归约 | `notes/CUDA/3_*`, `cuda_code/course3`, `course8` |
| 3 | GEMM v1-v3 | naive、shared memory tiling、thread tile，提高数据复用和计算密度 | `notes/CUDA/5_1_*`, `cuda_code/course5_1` |
| 4 | GEMM v4-v6 | float4 向量化加载、shared memory 转置、双缓冲、warp tiling | `notes/CUDA/5_2_*`, `5_3_*`, `5_4_*` |
| 5 | Memory patterns | transpose、histogram、scan，学习合并访存、冲突规避和多 block 协作 | `cuda_code/course6`, `course7`, `course13` |
| 6 | 高级库/框架 | cuBLAS、cuDNN/im2col、PyTorch extension、FlashAttention | `cuda_code/course9`, `course10`, `course12` |

## 算子优化计划

### 1. REDUCE

目标：把一维 float 数组求和做到正确、可测，并逐步接近显存带宽上限。

| 版本 | 文件中名字 | 优化点 | 重点观察指标 |
| --- | --- | --- | --- |
| v0 | `reduce_v0_interleaved` | 交错寻址 shared-memory 归约，最直观但分支和取模开销大 | warp divergence, shared bank conflict |
| v1 | `reduce_v1_sequential` | 顺序寻址，活跃线程连续，减少分支分化和 bank conflict | warp execution efficiency |
| v2 | `reduce_v2_first_add` | 每个线程先加载两个元素，减少 block 数和全局写回 | memory throughput |
| v3 | `reduce_v3_unroll_last_warp` | 最后一个 warp 手动展开，减少 `__syncthreads()` | instruction count, stall barrier |
| v4 | `reduce_v4_shuffle` | warp shuffle 做两级归约，shared memory 只保存 warp 结果 | shared memory load/store, latency |
| v5 | `reduce_v5_vectorized` | `float4` 向量化读取 + grid-stride loop，减少全局访存指令数 | memory instruction count, bandwidth |
| 库基线 | `cub::DeviceReduce::Sum` | CUB 生产级 reduce 基线 | 与手写 kernel 的带宽比 |

当前 `REDUCE/src/main.cu` 已包含 v0-v5、CUB baseline、CPU 校验和 CUDA event benchmark。

运行示例：

```bash
cd /home/jianyeshi/Note/CUDA/CUDA_optimazation
nvcc -O3 -std=c++17 -arch=sm_86 REDUCE/src/main.cu -o reduce_bench
./reduce_bench 16777216
```

Nsight Compute 示例：

```bash
ncu --set full ./reduce_bench 16777216
```

### 2. GEMM

目标：实现 `C = alpha * A @ B + beta * C`，矩阵按 row-major 存储，并用 cuBLAS 做正确性和性能基线。

| 版本 | 文件中名字 | 优化点 | 重点观察指标 |
| --- | --- | --- | --- |
| v1 | `sgemm_v1_naive` | 每个线程算一个 C 元素，直接读 global memory | global load efficiency, cache pressure |
| v2 | `sgemm_v2_smem` | 32x32 shared memory tiling，复用 A/B tile | shared memory throughput |
| v3 | `sgemm_v3_thread_tile` | 每线程计算 TMxTN 小块，提高算术强度 | FMA utilization, register usage |
| v4 | `sgemm_v4_vectorized` | float4 向量化加载 + A tile 转置存储 | memory instruction count, bandwidth |
| v5 | `sgemm_v5_double_buffer` | 双缓冲 shared memory + 寄存器预取，减少内存等待 | long scoreboard stalls |
| v6 | `sgemm_v6_warp_tiling` | 多级 tile：block/warp/thread，提高 warp 级局部性 | occupancy, eligible warps |
| 库基线 | `cublasSgemm` | NVIDIA 高性能 SGEMM | 手写 kernel / cuBLAS ratio |

当前 `GEMM/src/main.cu` 已包含 v1-v6 和 cuBLAS baseline。为了让边界处理更清楚，默认 benchmark 使用正方阵，尺寸可以从命令行传入。v4-v6 走高性能路径，要求 N 是 128 的倍数；其他尺寸会自动跳过这些版本。

运行示例：

```bash
cd /home/jianyeshi/Note/CUDA/CUDA_optimazation
nvcc -O3 -std=c++17 -arch=sm_86 GEMM/src/main.cu -lcublas -o gemm_bench
./gemm_bench 1024
```

如果显卡不是 Ampere/RTX 30 系列，把 `-arch=sm_86` 换成对应架构，例如 `sm_75`、`sm_80`、`sm_89`。

也可以用 CMake：

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
./build/reduce_bench 16777216
./build/gemm_bench 1024
```

## 推荐补全顺序

1. 先跑通 `REDUCE/src/main.cu`，记录 v0-v5 和 CUB 的时间、带宽。
2. 用 `ncu` 分别抓 v0、v1、v4、v5、CUB，重点看 warp 分化、barrier stall、shared/global memory 指令数。
3. 跑通 `GEMM/src/main.cu` 的 v1-v6 和 cuBLAS，先确认结果一致，再看 GFLOPS。
4. 给 GEMM 增加 CSV 输出和 Python 画图脚本，复用 `cuda_code/course5_1/draw.py` 的格式。
5. 增加 cuDNN/im2col conv、FlashAttention/PyTorch extension，形成从基础算子到高级库的完整链路。

## Benchmark 规范

- 所有 kernel 都先 warmup，再重复计时。
- 计时只包含 kernel/library 调用，不包含 host/device 拷贝。
- 每个版本都必须和 CPU 或 cuBLAS 结果比对。
- Reduce 用有效带宽：`N * sizeof(float) / time`。
- GEMM 用吞吐：`2 * M * N * K / time`。
- 每次优化都记录一个主要瓶颈和一个主要改善指标，避免只看总耗时。
