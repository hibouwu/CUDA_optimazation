# GEMM Shared Memory Bank Conflict 问题定义（NVIDIA Thor / SM110 Blackwell）

## 1. 背景

### 1.1 Thor 架构概述

NVIDIA DRIVE AGX Thor 是面向车载边缘计算的 SoC，基于 Blackwell GPU 架构（SM110），关键特性：

| 参数 | 数值 |
| --- | --- |
| GPU 架构 | Blackwell (SM110) |
| CUDA Cores | 2560 |
| 显存 | 128 GB LPDDR5X |
| Dense INT8 TOPS | 500 |
| Dense FP8 TFLOPS | 500 |
| Dense FP16 TFLOPS | 250 |
| FP8 支持 | 是 |
| TMEM（Tensor Memory） | 是（tcgen05.mma） |
| TMA | 是 |
| 典型功耗 | ~100W（SoC） |

与 consumer Blackwell (SM120, RTX 50) 不同，Thor 使用 `tcgen05` / TMEM 路径，属于 datacenter Blackwell family-specific 目标。这意味着：

- MMA 指令使用 `tcgen05.mma`（Unified MMA），而非 `mma.sync.aligned.kind::f8f6f4`
- 引入 TMEM 作为 Tensor Core 专用累加器内存
- 支持 FP4/FP6/FP8 低精度数据类型及块缩放（block scaling）

### 1.2 车载边缘计算特点

Thor 的典型 GEMM workload 与数据中心有显著差异：

| 维度 | Datacenter (H100/B200) | Thor (车载边缘) |
| --- | --- | --- |
| 矩阵尺寸 | 大 (M,N ≥ 4096) | 中小 (M,N 256~2048) |
| Batch size | 大 | 小 |
| 延迟要求 | 吞吐优先 | 实时低延迟（端到端 < 100ms） |
| 功耗约束 | 宽松 (700W+) | 严格 (~100W SoC) |
| 精度 | FP8/FP16 | INT4/FP4/FP8（模型量化） |
| Tile 尺寸 | 大 tile (128×256) | 受 limited occupancy 约束的较小 tile |

边缘场景中的 GEMM 通常来自：
- Transformer 推理的 MatMul（self-attention, FFN）
- 量化模型的 INT4/FP4 GEMM
- 多模态感知（图像+激光雷达融合）的矩阵运算

### 1.3 Shared Memory Bank 基础

NVIDIA GPU 的 shared memory 按 32 个 bank 组织，每个 bank 宽度为 4 字节（32-bit）。同一 bank 的多个访问会序列化（bank conflict）：

```
bank = (byte_address / 4) % 32
```

- **无冲突**：同一 warp 的 32 个 lane 访问 32 个不同 bank，或同一地址（broadcast）
- **2-way conflict**：2 个 lane 访问同一 bank 的不同地址
- **N-way conflict**：N 个 lane 访问同一 bank，指令 replay N 次

Bank conflict 的直接影响：
- MMA operand 从 SMEM 读取时产生 replay，降低有效吞吐
- 增加 Stall on MIO Throttle
- 在高 arithmetic intensity 场景（如 GEMM）中可能成为隐藏瓶颈

---

## 2. 问题定义

### 2.1 核心问题

**在 Thor/SM110 架构的 GEMM kernel 中，shared memory bank conflict 在哪些场景发生？如何准确量化、建模并消除？**

具体场景：

1. **Global-to-Shared 搬运阶段**（GMEM → SMEM copy）
   - 使用 TMA 或 `cp.async` 将 A/B tile 从 global memory 搬到 shared memory
   - SMEM 布局由 UMMA 要求的 swizzle 模式决定
   - 搬运阶段不产生 MMA 级的 bank conflict，但布局决定了后续读取的 bank 分布

2. **MMA Operand 从 SMEM 读取阶段**
   - 当前 SM110 kernel 通过 SMEM descriptor 让 MMA 指令直接读取 SMEM
   - SMEM 的 swizzle 布局 `Layout_K_SW128_Atom` 决定了 operand 元素到 bank 的映射
   - MMA atom 的 shape (如 128×256×16) 和 operand 的 K-major layout 共同影响 bank 访问模式

3. **非 MMA 的 SMEM 读写**（Epilogue、前处理等）
   - Epilogue 阶段从 TMEM→RMEM 后，可能在 SMEM 中做后处理再写回 GMEM
   - Scheduler/CLC warp 对 work tile metadata 的 SMEM 读写

### 2.2 为什么 Thor 上的 Bank Conflict 值得专门研究

**（a）SM110 的新特性引入新问题**

| 特性 | 对 Bank Conflict 的影响 |
| --- | --- |
| TCGen05 MMA 直接从 SMEM 读 operand | SMEM descriptor 的 swizzle 布局由硬件规定，bank 映射不可自由调整 |
| TMEM 替代 RMEM 做累加 | Epilogue 必须 tcgen05.ld TMEM→RMEM，可能存在新 SMEM bank 访问模式 |
| 4-stage pipeline (CLC→Load→MMA→Epi) | 多 warp 角色并发访问 SMEM 的不同区域，bank 冲突可能跨 stage 叠加 |
| 小 tile 场景更多 | Tile 越小，operand 在 SMEM 中的 stride 越小，越容易落入相同 bank |

**（b）边缘计算场景的特殊性**

- 小尺寸 GEMM 下，occupancy 更低，单个 warp 的 bank conflict 影响更大
- 量化模型（INT4/FP4）的 packed operand 改变了 SMEM 访问粒度（8×4bit/32bit），bank 冲突模式不同于 FP16
- 实时性要求下，bank conflict 导致的 MIO stall 是延迟抖动的关键来源

**（c）现有设计的已知瓶颈**

当前 SM110 kernel 设计中，已知存在以下 bank-conflict 相关的设计权衡：

- `tc4a` 的 B operand CTA prepack：将 B 读入 SMEM 后全 CTA 重新排列（pack），目的是优化 MMA 的访问模式。代价是引入额外的 CTA barrier。
- 64B TMA swizzle（`tc4b`）：改变 B tile 在 SMEM 中的布局以匹配 MMA 的 bank 期望，但并非所有 tile 尺寸都受益。
- Swizzle vs padding 的策略选择：UMMA swizzle 由 Layout Atom 决定，无法像传统 GEMM 那样自由选择 padding 消除 bank conflict。

---

## 3. 研究范围与子问题

### 3.1 子问题 1：SM110 SMEM Bank 访问模式建模

- 问题：给定 MMA atom shape、operand datatype、swizzle layout，能否精确推导出每个 warp 中每个 lane 在 SMEM 中的 bank 映射？
- 输入：MMA atom (128×256×16)，operand 类型 (FP16/BF16/FP4/FP8)，swizzle layout `Layout_K_SW128_Atom`
- 输出：按 lane 索引的 bank 编号分布，冲突度（conflict degree）统计
- 方法：解析 SMEM 地址公式 → bank index → 冲突统计

### 3.2 子问题 2：Edge Computing 典型 GEMM Shape 下的冲突模式

- 问题：在车载推理的典型矩阵尺寸下（M=N=512, 1024, 2048），bank conflict 模式是什么？
- 关注点：
  - A operand 读取（K-major swizzle）的 bank 分布
  - B operand 读取（K-major swizzle）的 bank 分布
  - TMEM epilogue load（tcgen05.ld）是否经过 SMEM 产生额外 bank 访问？
- 期望输出：各 shape 下 bank conflict 的理论值和实测值（Nsight Compute）

### 3.3 子问题 3：SM110 UMMA Swizzle 下的冲突消除策略

- 问题：在 UMMA 规定的 swizzle 约束下，能否通过调整 tile shape 减少 bank conflict？
- 探索方向：
  - 改变 `MMA_K` / `Tiles_K` 参数（如 `bK = 16*4 vs 16*2`）对 unrolling 和 bank 访问的影响
  - 在 swizzle layout 固定后，是否还有 padding 的空间
  - A/B operand 的 Major 选择（K-major vs M/N-major）对 bank 分布的影响

### 3.4 子问题 4：低精度（FP4/INT4/FP8）的 Packed Access 冲突

- 问题：量化 GEMM 中 packed operand 的 bank 访问粒度变化如何影响冲突模式？
- FP8：2× 元素/bank（vs FP16 的 1× 元素/bank）
- FP4/INT4：8× 元素/bank（32-bit bank 承载 8 个 4-bit 值）
- 打包后 bank 内的连续访问可能减少 bank 间的冲突，但也可能产生新的 broadcast 或 bank 内 conflict

---

## 4. 当前进展与基线

### 4.1 现有 SM110 Kernel 基线

| 版本 | 定位 | Tile Shape | Bank Conflict 关注点 |
| --- | --- | --- | --- |
| `tc3` | TCGen05/TMEM bring-up (probe) | 128×64×32 | 仅验证 TMEM alloc/ld/dealloc，无完整 GEMM |
| `tc4` | Warp-specialized pipeline scaffold | 128×64×32, 4-stage | MMA warp SMEM descriptor 读取 |
| `tc5` | CLC scheduler + tc4 mainloop | 同 tc4 | 多 warp 并发 SMEM 访问 |

### 4.2 已有参考

- GEMMComponents 下的 `04_smem_padding_bank_conflict` 和 `05_transpose_padding_bank_conflict`：传统 GEMM 的 SMEM bank conflict demo
- `scripts/v7_tuning/v7_parameter_model.md`：WMMA 下的 bank conflict 分析模型（作为方法论参考）
- CUTLASS 示例 `01_mma_sm100.cu`：SM100 UMMA 的 SMEM swizzle layout 定义

### 4.3 实验平台

- 目标芯片：NVIDIA DRIVE AGX Thor (SM110)
- 开发容器：CUDA 13.0, `cutlass-dev:cuda13.0`
- Profiling：Nsight Compute (`ncu`), Nsight Systems (`nsys`)
- 关键 metric：
  ```
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
  smsp__stall_mio_throttle
  smsp__warp_issue_stalled_barrier
  ```

---

## 5. 预期产出

1. **Bank Conflict 模型**：SM110 UMMA swizzle layout 下的 bank 映射公式和验证脚本
2. **冲突热点分析**：各 GEMM shape 下 A/B operand 的 bank 冲突热点图和统计
3. **缓解策略对比**：tile shape 调整、swizzle 选择、precision 影响的对比实验
4. **Thor 优化建议**：面向边缘推理的 GEMM SMEM 布局最佳实践

---

## 6. 与现有工作的关系

- `GEMMComponents/GEMMGmemToSmem/`：提供传统 GEMM 中 GMEM→SMEM 搬运的 bank conflict demo，作为方法论基础
- `GEMMsm110/`：当前 SM110 kernel 的实验平台，bank conflict 分析将基于此进行
- `scripts/v7_tuning/v7_parameter_model.md`：WMMA 的 bank 分析模型，SM110 的 UMMA 分析将借鉴其方法
- `Docs/cutlass/blackwellMMA.md`：UMMA 指令和 SMEM descriptor 的参考文档
