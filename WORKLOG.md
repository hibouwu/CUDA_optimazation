# CUDA Optimization Worklog

## 目标

根据 `/home/jianyeshi/Note/notes/CUDA` 的笔记和 `../cuda_code` 的课程代码，把 `CUDA_optimazation` 整理成一个从 naive 到高级库基线的 CUDA 算子优化练习仓库。

## 已完成内容

### 1. 项目结构

新增和补全：

```txt
CUDA_optimazation/
├── CMakeLists.txt
├── README.md
├── WORKLOG.md
├── REDUCE/src/main.cu
└── GEMM/src/main.cu
```

`CMakeLists.txt` 生成两个目标：

- `reduce_bench`
- `gemm_bench`

### 2. README 路线

`README.md` 已整理为完整优化路线：

1. CUDA 基础：线程模型、显存模型、host/device 分配与拷贝。
2. Reduce：从 shared memory naive 到 warp shuffle、float4、CUB。
3. Softmax / Norm：把 reduce 模式放进真实算子。
4. GEMM：从 naive 到 shared memory、thread tile、vectorized、double buffer、warp tiling。
5. Memory patterns：transpose、histogram、scan。
6. 高级库/框架：cuBLAS、CUB、cuDNN/im2col、PyTorch extension、FlashAttention。

### 3. REDUCE 实现

文件：`REDUCE/src/main.cu`

已实现版本：

| 版本 | Kernel / Baseline | 说明 |
| --- | --- | --- |
| v0 | `reduce_v0_interleaved` | 交错寻址 shared memory 归约，保留取模和 warp divergence，作为 naive 对照。 |
| v1 | `reduce_v1_sequential` | 顺序寻址，活跃线程连续，降低线程束分化和 shared memory bank conflict。 |
| v2 | `reduce_v2_first_add` | 每个线程先读两个元素再入 shared memory，减少 block 数和中间写回。 |
| v3 | `reduce_v3_unroll_last_warp` | 最后一个 warp 手动展开，减少同步和循环控制开销。 |
| v4 | `reduce_v4_shuffle` | 使用 `__shfl_down_sync` 做 warp 内归约，shared memory 只保存 warp 级结果。 |
| v5 | `reduce_v5_vectorized` | 使用 `float4` 向量化读取，减少 global memory load 指令数。 |
| lib | `cub::DeviceReduce::Sum` | CUB 生产级 reduce 基线，用来衡量手写 kernel 的差距。 |

Benchmark 设计：

- CPU 用 `std::accumulate` 生成参考结果。
- 每个 kernel warmup 后用 CUDA event 重复计时。
- 多 block partial result 使用 ping-pong buffer 循环归约到 1 个元素，避免只做两轮归约导致大输入错误。
- 输出每个版本的平均时间、有效带宽、结果和匹配状态。

### 4. GEMM 实现

文件：`GEMM/src/main.cu`

已实现版本：

| 版本 | Kernel / Baseline | 说明 |
| --- | --- | --- |
| lib | `cublasSgemm` | cuBLAS SGEMM baseline，row-major 输入通过交换 A/B 参数适配 cuBLAS column-major 语义。 |
| v1 | `sgemm_v1_naive` | 每个线程计算一个 C 元素，直接从 global memory 读 A/B。 |
| v2 | `sgemm_v2_smem` | 32x32 shared memory tiling，减少 global memory 重复读取。 |
| v3 | `sgemm_v3_thread_tile` | 每个线程计算 TMxTN 输出小块，提高计算密度和寄存器复用。 |
| v4 | `sgemm_v4_vectorized` | 使用 `float4` 加载，并把 A tile 转置写入 shared memory，让计算阶段读 A 连续化。 |
| v5 | `sgemm_v5_double_buffer` | 使用双 shared-memory buffer 和寄存器预取，减少访存等待。 |
| v6 | `sgemm_v6_warp_tiling` | 引入 block/warp/thread 三级 tiling，每个 warp 负责 64x64 子块。 |

Benchmark 设计：

- 输入为 row-major 方阵，默认 N=1024，可从命令行传入。
- cuBLAS 先生成参考输出。
- 每个手写 kernel 都和 cuBLAS 输出比较。
- 输出平均时间、GFLOPS、匹配状态。
- v4-v6 为高性能路径，要求 N 是 128 的倍数；其他尺寸会跳过这些版本，避免在核心路径塞边界分支。

### 5. 编译方式

手动编译：

```bash
nvcc -O3 -std=c++17 -arch=sm_86 REDUCE/src/main.cu -o reduce_bench
nvcc -O3 -std=c++17 -arch=sm_86 GEMM/src/main.cu -lcublas -o gemm_bench
```

CMake：

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
```

运行：

```bash
./reduce_bench 16777216
./gemm_bench 1024
```

### 6. 本机验证状态

当前机器上 `nvcc` 编译被 CUDA 13.0 和系统 glibc 头文件兼容性阻塞。即使最小程序只包含：

```cpp
#include <cuda_runtime.h>
int main() { return 0; }
```

也会报：

```txt
exception specification is incompatible with previous function "rsqrt"
exception specification is incompatible with previous function "rsqrtf"
```

这说明问题来自本地 CUDA/glibc 组合，不是当前项目代码特有问题。代码层面已按 CUDA C++ 常规写法整理，后续需要在兼容的 CUDA toolkit / host compiler / glibc 组合下实测。

## 后续实测建议

1. 先在 CUDA 12.x 或兼容 CUDA 13 的 host compiler 环境编译。
2. 跑 `reduce_bench`，按 v0-v5/CUB 记录带宽。
3. 跑 `gemm_bench 1024`、`2048`、`4096`，记录 v1-v6/cuBLAS GFLOPS。
4. 用 Nsight Compute 分别分析：
   - REDUCE：v0、v1、v4、v5、CUB。
   - GEMM：v1、v2、v4、v5、v6、cuBLAS。
5. 根据 `smsp__warp_issue_stalled_*`、global/shared memory throughput、eligible warps 再微调 tile size。
