# GEMM Shared Memory Bank Conflict 问题定义：NVIDIA Thor / SM110 Blackwell

> **文档性质**：研究计划（Research plan），不是结论性技术报告。
> **最后更新**：2026-06-26
> **目标架构**：NVIDIA DRIVE AGX Thor (SM110 Blackwell, tcgen05/TMEM 路径)
> **不在本文范围内的路径**：
> 非 Thor/SM110 的 Tensor Core GEMM 路径，包括 SM120 consumer Blackwell、SM100 datacenter Blackwell、SM90 Hopper、SM80 Ampere 等。

---

## A. 背景说明

### A.1 架构边界

Thor / SM110 属于 Blackwell 架构的 datacenter/automotive 分支。与 SM120 (consumer Blackwell) **不同**，SM110 的核心 GEMM 路径是：

```
TMA → SMEM (with descriptor/layout atom)
  → tcgen05.mma → TMEM accumulator
  → tcgen05.ld → RMEM
  → epilogue → GMEM
```

**关键区分**：

本文只研究 Thor/SM110 的 tcgen05/TMEM GEMM 路径。其他架构只作为“不能直接套用”的边界提醒，不展开其 MMA 指令、fragment layout 或 bank conflict 模型。

SM110 的 operand 消费模型：通过 SMEM descriptor 描述 operand 在 SMEM 中的物理布局，tcgen05.mma 根据 descriptor 消费 operand，其内部访问路径和可观测性需要实验验证。本文不把 Ampere/Hopper 的 ldmatrix + mma.sync 的 lane-fragment 模型直接套用到 SM110。

### A.2 硬件参数（待确认）

以下参数来自公开资料和本仓库已有代码中的推断。**需要用官方的 deviceQuery / ncu / datasheet 逐项确认**，不应在实验中当作硬编码常量：

- GPU 架构：Blackwell (SM110, compute capability 11.0)
- SM 数量：待确认
- TMEM 容量：per-SM，待确认（文档参考：256KB per SM，组织为 512 columns × 128 lanes × 32-bit）
- Shared memory / SM：待确认
- Max threads / SM：待确认
- Max registers / SM：待确认

**特别提醒**：不要直接把 Wikipedia 或公开新闻稿中的 TOPS/TFLOPS 写入实验参数。研究中应该用实际测量值替代。

---

## B. 数据流主线

SM110 GEMM kernel 中，一次完整的 K-tile 迭代涉及以下数据流阶段。**每个阶段涉及不同的内存空间、不同的指令类型和不同的 bank conflict 适用性。**

```
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 1: GMEM → SMEM (TMA copy)                                   │
│   - 使用 TMA (cp.async.bulk.tensor) 或 cooperative_copy         │
│   - 搬运 A tile [M×K] 和 B tile [K×N] 到 shared memory           │
│   - SMEM layout 由 Swizzle Layout Atom 决定                      │
│   - TMA 不按普通 `ld.shared` / `st.shared` 的 per-warp LSU bank conflict     │
│     model 直接分析。                                                         │
│   - TMA swizzle 可能影响 SMEM physical layout、TMA throughput 和后续          │
│     `tcgen05.mma` operand consumption。                                      │
│   - 是否存在可观测的 bank-level structural effect，需要通过 microbenchmark    │
│     间接验证。                                                               │
│   - 若使用 manual store (st.shared)，则产生普通 bank conflict，可直接建模     │
├─────────────────────────────────────────────────────────────────┤
│ 阶段 2: SMEM descriptor 构造                                      │
│   - cta_mma.make_fragment_A/B() 从 SMEM tensor 生成 descriptor   │
│   - Descriptor 封装: SMEM base addr, layout, swizzle mode,       │
│     leading dimension, offset                                    │
│   - 这是纯元数据操作, 不产生 memory access                        │
├─────────────────────────────────────────────────────────────────┤
│ 阶段 3: tcgen05.mma (SMEM descriptor → TMEM)                     │
│   - 由指定的 producer / MMA 发起线程或 warp 角色发出               │
│     tcgen05.mma 指令（具体以 PTX 和当前 kernel 实现为准）           │
│   - 输入: a-desc (64-bit), b-desc (64-bit), idesc (32-bit)      │
│   - Accumulator 写入 TMEM (不是 SMEM, 不是 Register)             │
│   - MMA 硬件内部按 descriptor 从 SMEM 读 operand                  │
│   - 这些 SMEM read 是否经过 LSU pipe? 是否被普通 bank conflict    │
│     metric 捕获? —— 待验证，不能直接假设                          │
├─────────────────────────────────────────────────────────────────┤
│ 阶段 4: tcgen05.ld (TMEM → RMEM)                                  │
│   - Warp-level 同步指令                                           │
│   - 从 TMEM 加载 accumulator 到 register                          │
│   - TMEM 不是 SMEM: 有独立的地址空间、寻址方式和物理存储          │
│   - 不产生 SMEM bank conflict                                    │
├─────────────────────────────────────────────────────────────────┤
│ 阶段 5: Epilogue (RMEM → SMEM staging → GMEM store)              │
│   - 在 register 中做 activation/scale/epilogue 处理 (axpby etc.) │
│   - 可能经过 SMEM staging (st.shared → ld.shared) 再 TMA store   │
│   - 此处的 st.shared / ld.shared 产生普通 SMEM bank conflict     │
│   - 可直接被 Nsight Compute 的 shared load/store bank conflict   │
│     metric 捕获                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**重要**：
- 阶段 3 (tcgen05.mma operand consumption) 是本文的核心未知量
- 阶段 5 (epilogue SMEM staging) 是可直接建模和测量的已知量
- 不要将阶段 3 和阶段 5 混在一起分析

---

## C. 需要纠正的概念（强制阅读）

本节是本文档的方法论基础。以下所有约定在研究过程中必须遵守。

### C.1 Bank conflict 是 shared memory 问题，不是 global memory coalescing 问题

- Global memory 的访问效率由 coalescing 决定（warp 内访问是否落在同一 L1 cache line / DRAM segment）
- Shared memory 的 bank conflict 由 `(byte_address / 4) % 32` 决定
- 二者是不同层级的独立问题，不应混用术语

### C.2 Bank conflict 的建模粒度是 per-warp, per-instruction

- 正确的分析单元：**一个 warp 执行一条 `ld.shared` 或 `st.shared` 指令时，32 个 lane 的 byte address 在 32 个 bank 上的分布**
- 不要跨 warp 累加 bank conflict counts 得到一个“总冲突度”
- 不要跨 stage 合并（如“TMA stage + MMA stage 的总 bank conflict”）
- 多个 pipeline stage **并发**访问 SMEM 会增加 SMEM/LSU/MIO 压力，但这应描述为 **SMEM bandwidth contention** 或 **MIO throttle**，而非“跨 stage bank conflict”
- 即使是同一 warp 在执行多条不同的 shared memory 指令，每条指令的 bank conflict 也应独立评估

### C.3 tcgen05.mma operand consumption 不假定被普通 LSU bank conflict metric 捕获

- `tcgen05.mma` 通过 SMEM descriptor 消费 operand
- 其 SMEM 访问可能走 Tensor Core 专用 datapath，不一定经过 LSU pipe
- Nsight Compute 的 `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` 等 metric 标注了 `pipe_lsu`，捕获的是 LSU pipe 上的 shared load
- **不能直接假设**这些 metric 能捕获 tcgen05.mma 内部的 SMEM operand 访问
- 需要做 microbenchmark 对比（详见 F 节阶段 5）

### C.4 TMEM 不是 SMEM

- TMEM 有独立的物理存储、地址空间（32-bit `{lane[31:16], column[15:0]}`）和指令集
- `tcgen05.ld` 明确是 **TMEM → Register**，不是 SMEM load
- 其他 `tcgen05` data movement 指令（如 `tcgen05.cp`, `tcgen05.st`）如果涉及 SMEM source/destination，需要单独按 PTX 语义确认，不在本文中凭直觉归类
- 不要把 TMEM 访问当成 SMEM bank conflict 来分析
- TMEM 内部有 lane/column 组织，但那是 TMEM 的存储结构，不叫 bank conflict

### C.5 FP4/INT4 packed operand 不能简单线性换算 conflict

- "FP16: 1 element per 2 bytes; FP4: 8 elements per 4 bytes, 所以 conflict 降低 8×" —— **这是错误的**
- Shared memory bank conflict 看的是 instruction 的 byte address 和访问宽度，不是 element count
- 如果一条 `ld.shared.b32` 读 4 bytes，它仍然访问 1 个 bank；里面打包了多少个 FP4 元素不等于 bank 访问数变小
- Packed format 改变的是 **element ↔ byte address 的映射**，以及 MMA 硬件如何解释这些 bit
- 正确做法：先弄清楚实际执行的 SMEM instruction 是什么（`ld.shared.b32`? `ld.shared.b128`? TMA? descriptor?），再看它的 byte address 落在哪些 bank

### C.6 Swizzle ≠ Padding

| 方法 | 机制 | 是否改变 bank 映射 | 是否增加 SMEM 占用 |
| --- | --- | --- | --- |
| Padding | 在每行末尾插入额外元素，改变 stride | 间接改变 | 是 |
| TMA swizzle (64B/128B) | TMA 硬件在搬运时对地址做 XOR swizzle | 改变 SMEM physical layout，可能影响后续 bank 分布和吞吐，需要验证 | 否（TMA 内部处理） |
| CuTe layout swizzle (Sw<B,M,S>) | CuTe 在逻辑坐标到物理坐标的映射中插入 XOR | 改变物理地址 | 否 |
| Manual XOR | 手写地址变换 | 改变 bank 映射 | 否 |

这三种 swizzle 是不同的概念和实现层。在研究 bank conflict 时要区分：
- 数据是如何**搬进** SMEM 的（TMA swizzle？manual store？cooperative_copy？）
- 数据在 SMEM 中的**物理 layout** 是什么（CuTe layout swizzle？无 swizzle？）
- MMA 是通过什么**descriptor** 看到这些数据的（descriptor 的 swizzle mode 字段？）

---

## D. 核心研究问题

> 在 Thor/SM110 的 tcgen05 GEMM kernel 中，A/B operand 的 SMEM physical layout、TMA swizzle、SMEM descriptor 和 MMA atom shape 如何共同决定 SMEM 访问效率？其中哪些路径能被 Nsight Compute 的 shared bank conflict metric 直接观测，哪些路径只能通过 MIO stall、TMA throughput、MMA issue/active、microbenchmark 间接验证？

子问题拆解：

1. **可观测路径**：epilogue SMEM staging、manual prepack store/load、scheduler metadata 访问中，`ld.shared` / `st.shared` 的 bank conflict 可被 NCU 直接捕获。能否基于 CuTe layout 写出精确的 byte-address → bank-id 映射公式？

2. **间接验证路径**：tcgen05.mma 的 operand consumption 如果**不**被 `pipe_lsu` metric 捕获，能否通过以下手段间接评估 SMEM layout 的质量：
   - 改变 swizzle mode (no swizzle / 64B / 128B) 对比 runtime 和 MIO stall
   - 改变 tile shape (bM, bN, bK) 对比 MMA issue rate
   - B operand prepack (将 B 在 SMEM 中重新排列后再构造 descriptor) 的收益多少来自 bank conflict 的减少，多少来自其他因素

3. **TMA 路径的干扰**：TMA 搬运走独立 datapath，但如果 TMA 的 write port 和 MMA 的 SMEM read port 共享 SMEM SRAM 的 bank？这是否会导致 bank 级别的 structural hazard？—— 需要查证硬件文档，如果没有公开文档，则需要 microbenchmark。

---

## E. 研究对象

### E.1 GMEM → SMEM 的 TMA copy / manual store 路径

- TMA (`cp.async.bulk.tensor`)：A[B×K] 和 B[K×N] tile
- 如果存在 TMA fallback / manual store，则分析 `st.shared` 的 bank conflict
- 如果使用 `cooperative_copy`，则分析对应的 `ld.global` + `st.shared` 序列

### E.2 tcgen05.mma 使用 a-desc / b-desc 消费 SMEM operand 的路径

- 构造 descriptor 的 CuTe 代码路径（`make_fragment_A/B` → `DescriptorIterator`）
- `Layout_K_SW128_Atom` 的 swizzle 参数 (`Sw<3,4,3>`)
- descriptor 的 swizzle mode 字段如何映射到硬件的 bank 访问模式 **（待验证，假设）**
- 这是本文的核心挑战，因为硬件内部行为不完全公开

### E.3 Epilogue 中的 SMEM staging

- 从 TMEM → RMEM (`tcgen05.ld`) 以后：
  - 将 accumulator 的 register value 写入 SMEM (`st.shared`)
  - 从 SMEM 读取做 epilogue 融合操作 (`ld.shared`)
  - 通过 TMA store 写回 GMEM
- 这些 `st.shared` / `ld.shared` 的 bank conflict 是**可直接建模和观测的**

### E.4 辅助 SMEM 访问

- mbarrier 状态
- TMEM base pointer（32-bit，存储在 SMEM 中）
- Scheduler/CLC warp 的 work tile metadata（如果放在 SMEM）
- Scale factor / block scale metadata：
  - 如果 scale factor 被手动 staging 到 SMEM（如通过 TMA 搬运到 SMEM 再读取），则按普通 SMEM load/store 建模
  - 如果 scale factor 作为 `tcgen05.mma.block_scale` 的 TMEM operand（PTX 文档中 scale-A / scale-B 地址指向 Tensor Memory 中的 scale matrix），则不属于 SMEM bank conflict

---

## F. 不研究或暂缓研究

以下内容明确排除在当前研究范围之外，或者在后续阶段再处理：

1. **不把 Ampere/Hopper 的 ldmatrix + mma.sync 的 lane-fragment 模型直接作为 SM110 结论**
   - 本文不以 `ldmatrix + mma.sync` 的 lane-fragment 模型作为 SM110 GEMM 的分析基础；SM110 目标路径是 `tcgen05.mma + SMEM descriptor + TMEM accumulator`。
   - 只在对比的意义上引用旧架构的数据，不直接套用公式

2. **不把 TMEM access 误归类成 SMEM bank conflict**
   - `tcgen05.ld` 是 TMEM → Register，按 TMEM 的 lane/column 地址空间寻址
   - 不适用 SMEM 的 bank = `(addr/4) % 32` 公式

3. **不把跨 warp、跨 stage 的 SMEM 压力误称为 bank conflict degree**
   - 多个 warp 并发访问 SMEM 导致 MIO throttle 是带宽竞争问题
   - 描述为“SMEM bandwidth contention”或“MIO压力”，不叫“跨 stage bank conflict”

4. **不在没有 descriptor/layout 公式的情况下直接手算 tcgen05.mma operand bank conflict**
   - 如果无法从 CuTe layout / descriptor 推导出确切的 SMEM byte address 映射，就不写“理论 bank conflict = X”
   - 标注为“待从 CuTe 代码反向推导 layout 公式后建模”

5. **不把没有实测支持的结论写成确定事实**
   - 不用“已经证明”“必然发生”
   - 用“假设”“待验证”“可观测路径”“间接指标”

---

## G. 分阶段学习与研究路线

### 阶段 1：Blackwell GEMM 数据流（理解数据路径）

**目标**：画出完整的数据流图，标注每个阶段的内存空间和指令类型。

**内容**：
- TMA (`cp.async.bulk.tensor`) 从 GMEM 搬运 A/B tile 到 SMEM
- `cta_mma.make_fragment_A/B()` 从 SMEM tensor 生成 SMEM descriptor
- `tcgen05.mma` 消费 a-desc / b-desc，累加到 TMEM
- `tcgen05.ld` 从 TMEM 加载到 register (RMEM)
- Epilogue: register 中做 axpby，经 SMEM staging 后 TMA store 回 GMEM

**产出**：一张 ASCII/Mermaid 数据流图，标注每个箭头的指令类型和内存空间（见 B 节已有初版）

**参考文件**：
- `Docs/cutlass/blackwellMMA.md`（UMMA 指令概述）
- `Docs/cutlass/cutlassSimpleExemple.md`（CUTLASS UMMA 示例的数据流）
- `GEMMsm110/include/tc4_gemm_kernel.cuh`（本仓库 SM110 pipeline 骨架）

---

### 阶段 2：SMEM descriptor / instruction descriptor（理解 descriptor 编码）

**目标**：理解 a-desc、b-desc、idesc 的字段结构和 CuTe 中对应的 C++ 类型。

**内容**：
- `a-desc` (64-bit)：封装 A 矩阵在 SMEM 中的 base address、layout、swizzle mode、leading dimension、offset
- `b-desc` (64-bit)：同上，B 矩阵
- `idesc` (32-bit)：数据类型、稀疏性、转置/取反标志、`ScaleOut` (accumulate vs overwrite)
- `cta_group::1` vs `cta_group::2`：1-CTA vs 2-CTA pair
- major mode (K-major / M-major / N-major)
- swizzle mode 字段的编码和可选值

**产出**：descriptor 字段表 + CuTe 类型对应关系

**参考文件**：
- `Docs/cutlass/blackwellMMA.md` 的 "matrix descriptors" 和 "instruction descriptor" 小节
- CUTLASS 源码：`include/cute/atom/mma_traits_sm100.hpp`（如果在容器中可访问）
- PTX ISA 文档：`tcgen05` 章节的 descriptor 编码

---

### 阶段 3：CuTe layout 和 CUTLASS layout atom（理解逻辑→物理映射）

**目标**：从 CuTe layout 推导出给定逻辑坐标 (m, k) 的物理 byte address。

**内容**：
- `Layout_K_SW128_Atom<TypeA>`：`Sw<3,4,3> o smem_ptr[16b] o ((MmaA, K_tiles), ...)`
  - `Sw<3,4,3>` 是 CuTe layout 中的 XOR swizzle 参数组合。其精确定义应以 CuTe 展开结果和 CUTLASS 源码为准。
  - **本文在反推出 `logical coord -> physical offset` 公式之前，不手写未经验证的物理含义。**
- `partition_shape_A/B` → `tile_to_mma_shape` 的推导链
- 逻辑坐标 → 物理坐标 → byte offset → bank id 的完整推导

**产出**：Python 脚本，输入 layout 参数，输出 (m, k) → (byte_addr, bank_id) 的映射

**参考文件**：
- `Docs/cutlass/cutlassSimpleExemple.md` 的 SMEM Layouts 小节（Swizzle 参数和打印输出）
- `GEMMComponents/GEMMGmemToSmem/04_smem_padding_bank_conflict/demo.cu`（传统 padding 方式，用于对比）
- `scripts/v7_tuning/v7_parameter_model.md` 的 "Shared-memory Bank 模型" 小节（旧版 WMMA 的分析方法论）

---

### 阶段 4：对可见的 SMEM 指令先建模 bank conflict

**目标**：先对能被 NCU 直接观测的 `ld.shared` / `st.shared` 建模。这是确定性的，不需要猜测硬件内部行为。

**内容**：
- Manual prepack：CTA 在 `tcgen05.mma` 之前/之后对 SMEM 的读写
- Epilogue SMEM staging：`st.shared` (RMEM → SMEM staging) + `ld.shared` (SMEM → register for epilogue computation)
- Scheduler / metadata：work tile coordinate、barrier phase、TMEM base ptr 的 SMEM 访问
- 对以上每个路径，用阶段 3 的 Python 脚本计算每个 warp 每指令的 bank 冲突期望值
- 用 NCU 实测对比：`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` 等

**产出**：可观测路径的 bank conflict model + NCU 验证数据

---

### 阶段 5：对 tcgen05.mma 路径做 microbenchmark（不假装能手算硬件内部）

**目标**：在不假设能直接建模硬件内部 SMEM 访问的情况下，通过对照实验间接评估 layout 质量。

**实验设计**（每个实验只改变一个变量）：

| 实验 | 改变变量 | 测量指标 |
| --- | --- | --- |
| Swizzle mode 对比 | no swizzle / 64B TMA swizzle / 128B swizzle | Runtime, MIO stall, MMA issue rate |
| K-tile 深度对比 | `bK = 16*2` vs `bK = 16*4` vs `bK = 16*8` | 同上 |
| A/B major 对比 | `UMMA::Major::K` vs `UMMA::Major::MN` | 同上 |
| Tile N 对比 | `bN = 64, 128, 256` | 同上 + NCU shared bank conflict metrics |
| B operand prepack | with vs without B prepack | Runtime, MIO stall, barrier stall |
| Padding 对比 | with vs without manual padding in SMEM | Runtime, SMEM usage, NCU metrics |

**关键原则**：
- 每次只改变一个变量
- 同时记录 NCU 的 LSU bank conflict metrics（即使不确定能否捕获 MMA 的 operand 访问）
- 如果改变 swizzle 后 MIO stall 显著变化但 LSU bank conflict metrics 不变，则支持非普通 LSU shared load/store 路径的假设，但不能单独证明
- **不做无根据的“理论计算”**。在没有公开的 SMEM read datapath 文档的情况下，microbenchmark 的结果比手算更可信

---

## H. Nsight Compute 指标说明

### H.1 可直接解释的指标（LSU pipe, shared memory）

| Metric | 说明 | 适用范围 |
| --- | --- | --- |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | LSU pipe 上 shared load 的 bank conflict 次数 | `ld.shared` 指令 |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` | LSU pipe 上 shared store 的 bank conflict 次数 | `st.shared` 指令 |
| `l1tex__data_bank_reads` | Shared memory bank read 总次数 | 所有 shared read |
| `l1tex__data_bank_writes` | Shared memory bank write 总次数 | 所有 shared write |

### H.2 需要谨慎解释的指标

| Metric | 说明 | 注意事项 |
| --- | --- | --- |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ldgsts.sum` | LSU pipe 上 ldgsts (load generic/store generic to shared) 的冲突 | TMA 走 async copy pipe，通常不在此处 |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_atom.sum` | Atomic 操作的 bank conflict | 通常不在 GEMM critical path 上 |

### H.3 间接指标（现象，不是 bank conflict 的直接证明）

| Metric | 说明 | 解读注意事项 |
| --- | --- | --- |
| `smsp__stall_mio_throttle` | Warp stall due to MIO (memory input/output) throttle | 反映 LSU/TMA/async copy 等 MIO pipe 的整体压力。如果高，可能由 bank conflict、TMA 竞争、或普通的 SMEM bandwidth contention 共同导致。**不是 bank conflict 的直接证明** |
| `smsp__stall_short_scoreboard` | Warp stall waiting for scoreboard (data dependency) | 反映 load-to-use latency。如果 bank conflict 增加了 load 延迟，会间接反映在这里 |
| `smsp__warp_issue_stalled_barrier` | Warp stall waiting at barrier | 反映同步开销。如果改善 layout 后此值下降，可能是 MMA 或 copy 更早完成——间接证据，不是 bank conflict 的证明 |

### H.4 关于 tcgen05.mma operand consumption 的特别说明

- `tcgen05.mma` 指令的 operand 访问可能走 Tensor Core 专用 datapath，不一定经过 LSU pipe
- 如果改变 swizzle 后 `pipe_lsu` bank conflict metrics 基本不变，但 runtime、MIO stall 或 MMA issue rate 显著变化，这**支持**“变化来自非普通 LSU shared load/store 路径”的假设。但它**不能单独证明** `tcgen05.mma` 的 SMEM operand read 不走 LSU pipe，需要结合 SASS、source counters、TMA throughput 和对照实验综合判断。
- 在未确认 datapath 之前，**不在文档中写“tcgen05.mma 的 bank conflict 可以通过 NCU 的 X metric 观察”**

---

## I. 与现有仓库代码的关系

### I.1 直接相关的代码

| 路径 | 内容 | 如何使用 |
| --- | --- | --- |
| `GEMMsm110/` | SM110 GEMM kernel (tc3/tc4/tc5) | 实验平台，bank conflict 分析的目标代码 |
| `GEMMComponents/GEMMGmemToSmem/` | GMEM→SMEM 搬运的独立 demo | 方法论参考，包含 `04_smem_padding_bank_conflict`、`05_transpose_padding_bank_conflict`、`05_smem_swizzle_store` |
| `scripts/v7_tuning/v7_parameter_model.md` | WMMA bank conflict 分析模型 | 方法论参考（注意：是 WMMA 模型，不是 UMMA 模型） |
| `Docs/cutlass/blackwellMMA.md` | UMMA 指令参考 | tcgen05.mma/ld/cp/st 的 PTX 语法和 descriptor 说明 |
| `Docs/cutlass/cutlassSimpleExemple.md` | CUTLASS UMMA 示例 | CuTe layout、SMEM descriptor、swizzle 的代码路径 |

### I.2 接下来最应该读的 3 个文件

1. **`Docs/cutlass/cutlassSimpleExemple.md`**：理解 CuTe 中 `Layout_K_SW128_Atom`、`make_fragment_A/B`、descriptor 构造的完整代码路径
2. **`GEMMsm110/include/tc4_gemm_kernel.cuh`**：理解本仓库 SM110 kernel 的 pipeline 骨架和 warp 角色分工
3. **`GEMMComponents/GEMMGmemToSmem/04_smem_padding_bank_conflict/demo.cu`**：理解传统 padding 改变 stride、缓解普通 SMEM ld/st bank conflict 的基本原理，用于和 SM110 的 swizzle 方案对比

---

## J. 最小可执行产出（TODO）

按优先级排列：

- [ ] **画出 SM110 tcgen05 GEMM 数据流图**，标注每个阶段的内存空间、指令类型、是否涉及 SMEM bank conflict
- [ ] **从 CUTLASS `01_mma_sm100.cu` 或本仓库的代码反推** A/B SMEM layout 的 `logical coordinate → byte offset` 公式。（`01_mma_sm100.cu` 只作为 Blackwell UMMA/CuTe layout 的参考入口；最终结论必须回到 SM110 编译目标和本仓库 kernel 验证。）
- [ ] **写一个 layout-to-bank Python 脚本**，只验证普通 SMEM load/store/prepack 路径的 bank 映射（阶段 4）
- [ ] **做 swizzle vs no-swizzle microbenchmark**，同时采集 NCU 的 LSU bank conflict metrics 和 MIO stall
- [ ] **做 B operand prepack vs no-prepack 对比实验**
- [ ] **不同 tile shape (bN=64/128/256, bK=16×2/16×4/16×8) 的对比实验**
- [ ] **对比以上实验中 `pipe_lsu` bank conflict metrics 和 MIO stall 的变化趋势**，判断 tcgen05.mma 的 operand 访问是否走 LSU pipe
- [ ] **最终讨论**：tcgen05.mma operand consumption 的效率是否能通过 SMEM layout 改进间接归因

---

## K. 附录：术语对照

| 术语 | 含义 | 是否涉及 SMEM bank |
| --- | --- | --- |
| TMA | Tensor Memory Accelerator，异步 tensor copy | 不按普通 ld.shared/st.shared 的 LSU bank conflict 模型直接分析；需用吞吐和 microbenchmark 间接验证 |
| SMEM descriptor (a-desc/b-desc) | 64-bit 元数据，描述 SMEM 中矩阵的物理布局 | 否（元数据） |
| tcgen05.mma | 第 5 代 Tensor Core MMA 指令 | operand 访问 SMEM，但是否走 LSU pipe 待验证 |
| TMEM | Tensor Memory，Tensor Core 专用累加器内存 | 否（独立内存空间） |
| tcgen05.ld | TMEM → Register 的 warp-level load | 否 |
| tcgen05.cp | TMEM data movement 指令，若涉及 SMEM source 需按 PTX 语义分析 | source/destination 涉及 SMEM 时可能涉及 bank |
| Layout_K_SW128_Atom | CuTe layout atom，K-major + 128B XOR swizzle | 是（决定 SMEM 物理地址到 bank 的映射） |
| Sw<3,4,3> | CuTe layout 中的 XOR swizzle 参数组合，精确定义以 CuTe 展开结果为准 | 是（决定 bank 映射，但物理含义待反推验证） |
| Epilogue SMEM staging | 在写回 GMEM 之前用 SMEM 做中间 buffer | 是（st.shared / ld.shared） |
