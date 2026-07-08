# Thor TCGen05 SS/TS 输入准备微基准

## **测试目的与背景**

本文用 Thor/SM110 上更贴近真实 GEMM 主循环的 SS/TS 流水线实验，测量 A 操作数的输入准备对 `tcgen05.mma` 完成吞吐的影响。这里的输入准备指 `tcgen05.cp` 把 A tile 从共享内存（SMEM）搬到张量内存（TMEM），让后面的 TS `tcgen05.mma` 可以从 TMEM 读取 A。真实 GEMM 会先用 TMA 把 A/B tile 搬到 SMEM，再让 Tensor Core 消费这些 tile；[上一篇 compute-only 报告](https://xiaopeng.feishu.cn/wiki/SMCIwsJwaimD6pkqXs4cf1AanXx)已经给出不含这段 A tile 拷贝的 dense SS 基线，例如 `FullSM4WarpBlock M128N256K64 FP4` 达到 `1032.111 TFLOP/s`；本文在 `microbench/mma_with_cp` 中继续测四个核心 MMA 组合：SS 每个线程束（warp）一个 D tile、TS 每个 warp 一个 D tile、SS 每个 warp 多个 D tile、TS 每个 warp 多个 D tile。

本文关注 SMEM/TMEM 输入路径对 `tcgen05.mma` 的影响。全局内存（GMEM）、TMA、尾处理、TMEM 读回和全局写回先不涉及；TMEM 分配、释放和 relinquish 后续需要时再单独测。

当前实验主线覆盖 SS 和 TS 两类操作数来源。SS 表示 A/B 都从 SMEM 描述符读取，TS 表示 A 从 TMEM 读取、B 从 SMEM 描述符读取；TS 的 A tile 由 `tcgen05.cp` 从 SMEM 写入 TMEM。

![SS 路径流水线：tcgen05.mma 直接从 SMEM 读取 A/B tile](图片和附件/img_v3_0213d_d8abef8b-af78-405e-8b4e-ccb3557bcaag.jpg)

![TS 路径流水线：tcgen05.cp 先把 A tile 写入 TMEM，再执行 tcgen05.mma](图片和附件/img_v3_0213d_ab5d3d01-d9b6-4678-828b-01ec28efde9g.jpg)

SS 图展示了真实 GEMM 主循环里最直接的 Tensor Core 消费路径。黑色 `TMA load 0/1/2/3` 先把连续 K 阶段的 A/B tile 放进 SMEM，红色 `tcgen05.mma` 随后直接从 SMEM 描述符读取 A/B tile，并把累加结果写入 TMEM 累加器；粉色括号表示真实 kernel 的总执行时间，绿色 GMEM 方块表示后续全局内存相关操作，这部分先不讨论。

TS 图展示了带输入准备的主循环路径。黑色 `TMA load` 仍然先把 tile 放进 SMEM，蓝色 `tcgen05.cp` 把 A tile 从 SMEM 搬到 TMEM，红色 `tcgen05.mma` 再从 TMEM 读取 A、从 SMEM 读取 B，并更新 TMEM 累加器；这个蓝色拷贝阶段是 `mma_with_cp` 相对[上一篇 compute-only 报告](https://xiaopeng.feishu.cn/wiki/SMCIwsJwaimD6pkqXs4cf1AanXx)新增的核心实验对象。

SS 和 TS 的关键差异在 A 操作数的读取位置。SS 的每条 `tcgen05.mma` 通过 `a_desc` 从 SMEM 读取 A，TS 的每条 `tcgen05.mma` 通过 `[a_tmem]` 从 TMEM 读取 A；因此 TS 需要在 MMA 使用 A 之前安排 `tcgen05.cp` 填充动作。

本轮实验把 SS/TS 输入路径和 D tile 策略交叉测量。SS one-D-per-warp 和 TS one-D-per-warp 都让每个 warp 写自己的一个 D tile，用来对齐真实 GEMM 中常见的输出 tile 归属；SS multi-D-per-warp 和 TS multi-D-per-warp 都让每个 warp 在多个 D tile 间轮转，用来观察连续写同一个 D tile 的依赖延迟能否被隐藏。TS 两组实验还要记录 `tcgen05.cp` 单独拷贝 cycles 和 cp+mma 重叠后暴露出来的 cycles，对应图中蓝色 A tile 输入准备和粉色总执行时间。

## **tcgen05.mma SS/TS 指令解析**

`tcgen05.mma` 执行的数学动作是 `C[M,N] += A[M,K] * B[K,N]`。本文所有 shape 都按 `M*N*K` 顺序写，例如 `M128N256K64` 表示 `C[128,256] += A[128,64] * B[64,256]`。

SS 路径把 A/B 都交给 SMEM 描述符。硬件根据 `a_desc` 读取 A tile，根据 `b_desc` 读取 B tile，把结果累加到 `[d_tmem]` 指向的 TMEM C/D 区域；[上一篇 compute-only 报告](https://xiaopeng.feishu.cn/wiki/SMCIwsJwaimD6pkqXs4cf1AanXx)中的 dense 基线使用这个路径。

```ptx
// SS: A from SMEM, B from SMEM, D/C in TMEM.
// [d_tmem]: TMEM 中的 C/D 累加器地址。
// a_desc: A tile 的 SMEM 描述符。
// b_desc: B tile 的 SMEM 描述符。
// idesc: MMA 的 M/N shape 和数据类型描述符。
tcgen05.mma.cta_group::1.kind::<dtype>
  [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;
```

TS 路径把 A 操作数的来源切到 TMEM。硬件通过 `[a_tmem]` 读取已经写入 TMEM 的 A tile，通过 `b_desc` 继续从 SMEM 读取 B tile；这个形态可以把 A tile 拷贝准备成本和 MMA 使用成本拆开测。

```ptx
// TS: A from TMEM, B from SMEM, D/C in TMEM.
// [a_tmem]: TMEM 中的 A tile 地址，通常由 tcgen05.cp 写入。
// b_desc: B tile 的 SMEM 描述符。
tcgen05.mma.cta_group::1.kind::<dtype>
  [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
```

FP4/block-scale 路径额外传入 `[scale-A-tmem]` 和 `[scale-B-tmem]`。[上一篇 compute-only 报告](https://xiaopeng.feishu.cn/wiki/SMCIwsJwaimD6pkqXs4cf1AanXx)的 FP4 SS 基线使用 `tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16`，对应 SASS 为 `UTCOMMA.4X`；TS FP4 kernel 需要把 A tile、A scale 和 B scale 的 TMEM 布局一起验证。

完整 `tcgen05.mma` 语法以 NVIDIA PTX ISA 文档 `9.7.17.10.9.1` 为准。本文只固定 SS/TS microbenchmark 需要的最小 operand 形态，其他 sparse、2CTA、cluster 路径后续单独展开。

## **tcgen05.cp 指令解析**

`tcgen05.cp` 在 TS 路径里负责把 A tile 从 SMEM 写到 TMEM。硬件先按 `s_desc` 描述的 SMEM 布局读取 A tile，再把这块 A tile 写到 `[taddr]` 指向的 TMEM 位置；下一条 TS `tcgen05.mma` 使用 `[a_tmem]` 读取同一份 A。

```ptx
// cta_group::<1|2>: 选择 1 个 CTA 或 2 个 CTA 协作执行 copy。
// <cp-shape>: 选择本条拷贝指令一次写入 TMEM 的范围。
tcgen05.cp.cta_group::<1|2>.<cp-shape>
  // [taddr]: TMEM 目的地址；s_desc: SMEM 源矩阵描述符。
  [taddr], s_desc;
```

一条 `tcgen05.cp` 指令可以按三个字段阅读。`cta_group::<1|2>` 选择 CTA 组，当前 `mma_with_cp` 先测 `cta_group::1`，与[上一篇 compute-only 报告](https://xiaopeng.feishu.cn/wiki/SMCIwsJwaimD6pkqXs4cf1AanXx)中的 `tcgen05.mma.cta_group::1` 对齐；`<cp-shape>` 选择一次拷贝的 TMEM 写入范围；`[taddr], s_desc` 分别告诉硬件 TMEM 写入地址和 SMEM 读取布局。

`[taddr]` 是 TMEM 目的地址。内联 PTX 中通常用 32-bit 寄存器操作数传入，例如 `tmem_base + a_tmem_offset`；`s_desc` 是 SMEM 矩阵描述符，通常用 64-bit 寄存器操作数传入，例如 `make_smem_desc(smem_a, desc_leading, desc_stride)` 生成的 A 描述符。

```C++
// a_tmem 是 A tile 在 TMEM 中的起始地址，后续 TS MMA 会用 [a_tmem] 读取 A。
uint32_t a_tmem = tmem_base + a_tmem_offset;

// a_desc 描述 A tile 在 SMEM 中的起始地址、leading offset、stride 和交错布局。
uint64_t a_desc = make_smem_desc(smem_a, desc_leading, desc_stride);

asm volatile(
  // cta_group::1 与当前单 CTA MMA benchmark 对齐；128x128b 是本条 cp 的写入范围。
  "tcgen05.cp.cta_group::1.128x128b [%0], %1;"
  // "r" 传 32-bit TMEM 地址；"l" 传 64-bit SMEM 描述符；memory 防止编译器重排内存访问。
  :: "r"(a_tmem), "l"(a_desc) : "memory");
```

CUDA 13.x 的 CCCL PTX wrapper 给出了当前可直接引用的 `tcgen05.cp` 基础形态。宿主机头文件 `/usr/local/cuda-13.0/include/cccl/cuda/__ptx/instructions/generated/tcgen05_cp.h` 中列出 6 种基础后缀，kernel 可以先从 `128x128b` 或 `128x256b` 试编译。

```ptx
// 一次拷贝写入 128x256 bit 范围，适合先试较大的 A tile 输入准备。
tcgen05.cp.cta_group::<1|2>.128x256b [taddr], s_desc;

// 一次拷贝写入 4x256 bit 范围，适合验证窄 M 范围的拷贝路径。
tcgen05.cp.cta_group::<1|2>.4x256b [taddr], s_desc;

// 一次拷贝写入 128x128 bit 范围，适合作为 TS 微基准的首个简单样例。
tcgen05.cp.cta_group::<1|2>.128x128b [taddr], s_desc;

// warpx2::02_13 表示 2 个 warp pair 采用 0/2 与 1/3 的分组映射。
tcgen05.cp.cta_group::<1|2>.64x128b.warpx2::02_13 [taddr], s_desc;

// warpx2::01_23 表示 2 个 warp pair 采用 0/1 与 2/3 的分组映射。
tcgen05.cp.cta_group::<1|2>.64x128b.warpx2::01_23 [taddr], s_desc;

// warpx4 表示 4 个 warp 参与 32x128 bit 范围的拷贝。
tcgen05.cp.cta_group::<1|2>.32x128b.warpx4 [taddr], s_desc;
```

低精度打包输入使用带打包后缀的拷贝形态。`b8x16.b6x16_p32` 和 `b8x16.b4x16_p64` 描述 SMEM 中低比特 A tile 到 TMEM 的重排方式；FP4/FP6 kernel 需要把这些后缀与 MMA 数据类型、A tile 布局和 scale 数据布局一起试编译、反汇编和跑 NCU。

```ptx
// b8x16.b6x16_p32: SMEM 侧按 8-bit lane 组织，TMEM 侧生成 6-bit packed 数据，pack group 为 32。
tcgen05.cp.cta_group::<1|2>.128x256b.b8x16.b6x16_p32 [taddr], s_desc;
tcgen05.cp.cta_group::<1|2>.4x256b.b8x16.b6x16_p32 [taddr], s_desc;
tcgen05.cp.cta_group::<1|2>.128x128b.b8x16.b6x16_p32 [taddr], s_desc;
tcgen05.cp.cta_group::<1|2>.64x128b.warpx2::02_13.b8x16.b6x16_p32 [taddr], s_desc;
tcgen05.cp.cta_group::<1|2>.64x128b.warpx2::01_23.b8x16.b6x16_p32 [taddr], s_desc;
tcgen05.cp.cta_group::<1|2>.32x128b.warpx4.b8x16.b6x16_p32 [taddr], s_desc;

// b8x16.b4x16_p64: SMEM 侧按 8-bit lane 组织，TMEM 侧生成 4-bit packed 数据，pack group 为 64。
tcgen05.cp.cta_group::<1|2>.128x256b.b8x16.b4x16_p64 [taddr], s_desc;
tcgen05.cp.cta_group::<1|2>.4x256b.b8x16.b4x16_p64 [taddr], s_desc;
tcgen05.cp.cta_group::<1|2>.128x128b.b8x16.b4x16_p64 [taddr], s_desc;
tcgen05.cp.cta_group::<1|2>.64x128b.warpx2::02_13.b8x16.b4x16_p64 [taddr], s_desc;
tcgen05.cp.cta_group::<1|2>.64x128b.warpx2::01_23.b8x16.b4x16_p64 [taddr], s_desc;
tcgen05.cp.cta_group::<1|2>.32x128b.warpx4.b8x16.b4x16_p64 [taddr], s_desc;
```

`tcgen05.cp` 后缀和报告里的 `M*N*K` shape 描述不同对象。`128x128b` 描述拷贝指令一次写入 TMEM 的范围；`M128N256K16 BF16` 描述 MMA 指令执行的矩阵乘加 shape，其中 M/N 来自 `idesc`，K 来自 BF16 指令路径，B 的读取布局来自 `b_desc`。

## **测试动作**

`mma_with_cp` 先测四个核心 MMA 组合。四个组合都输出 cycles、TFLOP/s、MMA 计数器和停顿计数器；TS 组合额外输出仅 `tcgen05.cp` cycles 和 cp+mma 重叠 cycles，用来解释 A tile 从 SMEM 写入 TMEM 后还暴露多少成本。

|实验|计时窗口|输出字段|解释口径|
|---|---|---|---|
|SS one-D-per-warp（每 warp 一个 D tile）|SS `tcgen05.mma` 循环 + 提交/等待|cycles、TFLOP/s、MMA 计数器|每个 warp 写自己的一个 D tile，贴近真实 GEMM 的输出 tile 归属|
|TS one-D-per-warp（每 warp 一个 D tile）|`tcgen05.cp` 准备 A tile + TS `tcgen05.mma` 循环 + 提交/等待|仅 cp cycles、TS MMA cycles、重叠 cycles、TFLOP/s、拷贝/MMA 计数器|A 从 SMEM 写入 TMEM 后，每个 warp 写自己的一个 D tile|
|SS multi-D-per-warp（每 warp 多个 D tile）|SS `tcgen05.mma` 循环 + D tile 轮转 + 提交/等待|cycles、TFLOP/s、MMA 计数器、停顿计数器|同一个 warp 在多个 D tile 间轮转，观察是否隐藏连续写同一 D tile 的依赖延迟|
|TS multi-D-per-warp（每 warp 多个 D tile）|`tcgen05.cp` 准备 A tile + TS `tcgen05.mma` 循环 + D tile 轮转 + 提交/等待|仅 cp cycles、TS MMA cycles、重叠 cycles、TFLOP/s、拷贝/MMA 计数器、停顿计数器|A 从 SMEM 写入 TMEM 后，同一个 warp 在多个 D tile 间轮转|

SS 基线的证据来自[上一篇 compute-only 报告](https://xiaopeng.feishu.cn/wiki/SMCIwsJwaimD6pkqXs4cf1AanXx)。例如 `FullSM4WarpBlock M128N256K64 FP4` 为 `1032.111 TFLOP/s`、`5120404 cycles`、峰值比 `99.72%`；后续四个核心 MMA 组合应优先和同 shape、同精度、同 launch 的 SS 基线对比。

D tile 策略需要在 SS 和 TS 源码里同时显式区分。[上一篇 compute-only 报告](https://xiaopeng.feishu.cn/wiki/SMCIwsJwaimD6pkqXs4cf1AanXx)当前每个 CTA 内所有 warp 都使用 `tmem_c = tmem_base`，这会让 4 个 warp 写同一个 D tile 起点；新实验应给 SS/TS 都增加 `tmem_c = tmem_base + warp_id * d_tile_stride` 的每 warp 一个 D tile 版本，再给 SS/TS 都增加 `tmem_c = tmem_base + warp_id * d_tile_stride + (iter % d_tiles_per_warp) * rotate_stride` 的多 D tile 轮转版本。

TS/cp 实验应沿用纯计算基线的用例命名和输出表格顺序。用例名继续用 `m128n256`、`m128n128`、`m128n64`，报告表格继续按 `精度 / WarpNum / 矩阵形状(M*N*K) / 计算量` 排列。

## **实现与验证边界**

实现入口应沿用[上一篇 compute-only 报告](https://xiaopeng.feishu.cn/wiki/SMCIwsJwaimD6pkqXs4cf1AanXx)的脚本结构。`mma_with_cp/run_thor_tcgen05_report.py` 负责生成 `benchmark_src/*.cu`、编译到 `build/`、运行 benchmark 并输出中文报告；`build_and_run.sh`、`run_ncu_reports.sh` 和画图脚本继续使用同类命令入口。

`tcgen05.cp` 的 SASS 名称需要由 kernel 验证。实现 `benchmark_src` 后，用 `cuobjdump --dump-sass build/<case>` 找拷贝指令对应的 SASS，再用 NCU 关键指标记录拷贝管线、张量管线和 stall counter；当前文档只固定 PTX 语法、操作数语义和实验拆分。

NCU 关键指标应优先解释异常。报告先保存小指标集，用于观察 cycles、MMA/cp 指令计数、张量管线、拷贝管线、warp 发射/停顿和启动边界；完整指标可以作为单独脚本或参数保存，避免图形界面报告过大。

TMEM 分配默认放在计时外。kernel 在 `clock64()` 前执行 `tcgen05.alloc.cta_group::1`，在 `clock64()` 后执行 `tcgen05.dealloc` 和 `tcgen05.relinquish_alloc_permit`；分配开销后续单独测。

## **总结**

本文把 `mma_with_cp` 的实验对象收敛到 SS/TS 输入准备流水线。SS 基线已由[上一篇 compute-only 报告](https://xiaopeng.feishu.cn/wiki/SMCIwsJwaimD6pkqXs4cf1AanXx)给出，新实验继续测 SS one-D-per-warp、TS one-D-per-warp、SS multi-D-per-warp、TS multi-D-per-warp 四个核心 MMA 组合；TS 路径额外记录 `tcgen05.cp` 拷贝准备和 cp+mma 重叠后的 cycles。

每 warp 多个 D tile 用来观察依赖延迟隐藏能力。连续多次写同一个 D tile 可能把后续 MMA 绑在同一个累加器依赖链上；让同一个 warp 在多个 D tile 间轮转，可以分别观察 SS 和 TS 的 cycles、TFLOP/s 和 stall counter 是否改善。

`tcgen05.cp` 是 TS 路径的关键新增指令。它用 `[taddr], s_desc` 把 A tile 从 SMEM 描述符写入 TMEM，后续 TS `tcgen05.mma` 用 `[a_tmem]` 读取 A；实现时先验证 `cta_group::1.128x128b` 或 `128x256b` 的 SASS 和 NCU counter。

最终判断 TS 输入准备流水线要看重叠后暴露出来的 cycles。仅 cp 的 cycles 解释 A tile 从 SMEM 搬到 TMEM 的基础成本，仅 TS MMA 的 cycles 解释 A 在 TMEM 时的使用成本，TS one-D-per-warp 和 TS multi-D-per-warp 的 cp+mma 重叠 cycles 对应实际流水线里露出来的成本。
