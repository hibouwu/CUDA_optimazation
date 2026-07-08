# tcgen05 micro\-benchmark

# 引言及tcgen05基础语法

Ampere tensor core（ptx 为 mma）从 register file（RF）中获取输入 A 和 B 矩阵。Hopper tensor core（ptx 为 wgmma）从 RF/SMEM 中获取输入 A 矩阵，从 SMEM 中获取输入 B 矩阵。

而我们关注的 Blackwell\-family Tensor Core 路径对应 PTX `tcgen05.mma`。在该路径中，A 可以来自 SMEM 或 TMEM，B 来自 SMEM，accumulator C/D 保存在 TMEM 中。常见 GEMM 路径是先将 A/B tile 搬入 SMEM，再通过 SMEM descriptor 交给 `tcgen05.mma` 读取（TS：A在TMEM、B在SMEM，A tile 被`tcgen05.cp`存入TMEM；SS：A/B都在SMEM。）。pipeline 如下图所示：

![img\_v3\_0213d\_d8abef8b\-af78\-405e\-8b4e\-ccb3557bcaag\.jpg](图片和附件/img_v3_0213d_d8abef8b-af78-405e-8b4e-ccb3557bcaag.jpg)

![img\_v3\_0213d\_ab5d3d01\-d9b6\-4678\-828b\-01ec28efde9g\.jpg](图片和附件/img_v3_0213d_ab5d3d01-d9b6-4678-828b-01ec28efde9g.jpg)

本文的 SS/TS microbenchmark 主线只需要两种 `tcgen05.mma` 形态。SS 路径让 A/B 都从 SMEM descriptor 读取，TS 路径让 A 从 TMEM 读取、B 从 SMEM descriptor 读取；完整 `tcgen05.mma` 语法见 NVIDIA PTX ISA 文档 `9.7.17.10.9.1`。

```ptx
// SS: A from SMEM, B from SMEM, D/C in TMEM.
tcgen05.mma.cta_group::1.kind::<dtype>
  [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d;

// TS: A from TMEM, B from SMEM, D/C in TMEM.
tcgen05.mma.cta_group::1.kind::<dtype>
  [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d;
```

FP4/block-scale 路径在上述两种形态上额外传入 `[scale-A-tmem]` 和 `[scale-B-tmem]`。TS 路径还需要用 `tcgen05.cp` 把 A tile 写入 TMEM；单独计时 `tcgen05.cp` 只能给出 copy 路径的基础成本，pipeline 计时窗口更应该观察 copy 与 MMA 重叠之后还暴露出来的 cycles。

`tcgen05.cp` 在 TS 路径里负责把 A operand 从 SMEM 喂到 TMEM。硬件先按 `s_desc` 描述的 SMEM layout 读取 A tile，再把这块 A tile 写到 `[taddr]` 指向的 TMEM 位置；下一条 TS `tcgen05.mma` 使用 `[a_tmem]` 读取这份 A，B 继续通过 `b_desc` 从 SMEM 读取。

```ptx
// cta_group::<1|2>: 选择 1 个 CTA 或 2 个 CTA 协作执行 copy。
// <cp-shape>: 选择本条 copy 指令一次写入 TMEM 的 footprint。
tcgen05.cp.cta_group::<1|2>.<cp-shape>
  // [taddr]: TMEM 目的地址；s_desc: SMEM source matrix descriptor。
  [taddr], s_desc;
```

一条 `tcgen05.cp` 指令可以按三个字段阅读。`cta_group::<1|2>` 选择 CTA group，当前 `mma_with_cp` microbenchmark 先测 `cta_group::1`，与 `mma_compute_only` 中的 `tcgen05.mma.cta_group::1` 对齐；`<cp-shape>` 选择一次 copy 的 TMEM 写入 footprint；`[taddr], s_desc` 分别告诉硬件 TMEM 写入地址和 SMEM 读取布局。

`[taddr]` 是 TMEM destination address。inline PTX 中通常用 32-bit register operand 传入，例如 `tmem_base + a_tmem_offset`；`s_desc` 是 SMEM matrix descriptor，inline PTX 中通常用 64-bit register operand 传入，例如 `make_smem_desc(smem_a, desc_leading, desc_stride)` 生成的 A descriptor。

```C++
// a_tmem 是 A tile 在 TMEM 中的起始地址，后续 TS MMA 会用 [a_tmem] 读取 A。
uint32_t a_tmem = tmem_base + a_tmem_offset;

// a_desc 描述 A tile 在 SMEM 中的起始地址、leading offset、stride 和 swizzle layout。
uint64_t a_desc = make_smem_desc(smem_a, desc_leading, desc_stride);

asm volatile(
  // cta_group::1 与当前 single-CTA MMA benchmark 对齐；128x128b 是本条 cp 的 footprint。
  "tcgen05.cp.cta_group::1.128x128b [%0], %1;"
  // "r" 传 32-bit TMEM address；"l" 传 64-bit SMEM descriptor；memory 防止编译器重排内存访问。
  :: "r"(a_tmem), "l"(a_desc) : "memory");
```

CUDA 13.x 的 CCCL PTX wrapper 给出了当前可直接引用的 `tcgen05.cp` 基础形态。宿主机头文件 `/usr/local/cuda-13.0/include/cccl/cuda/__ptx/instructions/generated/tcgen05_cp.h` 中列出了下面 6 种基础 suffix，后续 kernel 可以先从 `128x128b` 或 `128x256b` 这类直观形态开始试编译。

```ptx
// 一次 copy 写入 128x256 bit footprint，适合先试较大的 A tile feed。
tcgen05.cp.cta_group::<1|2>.128x256b [taddr], s_desc;

// 一次 copy 写入 4x256 bit footprint，适合验证窄 M footprint 的 copy 路径。
tcgen05.cp.cta_group::<1|2>.4x256b [taddr], s_desc;

// 一次 copy 写入 128x128 bit footprint，适合作为 TS microbenchmark 的首个简单样例。
tcgen05.cp.cta_group::<1|2>.128x128b [taddr], s_desc;

// warpx2::02_13 表示 2 个 warp pair 采用 0/2 与 1/3 的分组映射。
tcgen05.cp.cta_group::<1|2>.64x128b.warpx2::02_13 [taddr], s_desc;

// warpx2::01_23 表示 2 个 warp pair 采用 0/1 与 2/3 的分组映射。
tcgen05.cp.cta_group::<1|2>.64x128b.warpx2::01_23 [taddr], s_desc;

// warpx4 表示 4 个 warp 参与 32x128 bit footprint 的 copy。
tcgen05.cp.cta_group::<1|2>.32x128b.warpx4 [taddr], s_desc;
```

低精度 packed 输入使用带 pack suffix 的 copy 形态。`b8x16.b6x16_p32` 和 `b8x16.b4x16_p64` 描述 SMEM 中 packed low-bit A tile 到 TMEM 的重排方式；FP4/FP6 kernel 需要把这些 suffix 与 MMA dtype、A tile layout 和 scale 数据布局一起试编译、反汇编和跑 NCU。

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

`tcgen05.cp` suffix 和报告里的 `M*N*K` shape 各自描述不同对象。`128x128b` 描述 copy 指令一次写入 TMEM 的 footprint；`M128N256K16 BF16` 描述 MMA 指令执行的矩阵乘加 shape，其中 M/N 来自 `idesc`，K 来自 BF16 指令路径，B 的读取布局来自 `b_desc`。

TS microbenchmark 需要把 copy 成本和 pipeline 暴露成本分开测。第一组实验只循环 `tcgen05.cp`，记录 A tile feed 的基础 cycles；第二组实验先把 A tile 放进 TMEM，再只计 TS `tcgen05.mma` cycles；第三组实验把 `tcgen05.cp` 放进 MMA loop 或前置 stage，记录 cp 与 MMA 重叠后仍然暴露的 cycles。报告判断 input-feed pipeline 时优先看第三组 cycles，copy-only cycles 用来解释异常。

`tcgen05.cp` 的 SASS 名称和 NCU counter 由后续 kernel 验证。实现 `benchmark_src` 后，用 `cuobjdump --dump-sass build/<case>` 找 copy 指令对应的 SASS，再用 key metrics NCU report 记录 copy pipe、tensor pipe、stall counter；当前文档先固定 PTX 语法、operand 语义和实验拆分。





因此，Blackwell/SM110 上的 swizzle layout 重点是：SMEM 中的 tile 应如何布局，才能让 `tcgen05.mma` 以更硬件友好的方式读取，从而减少 bank conflict 和带宽浪费。最后 epilogue 再将 TMEM accumulator 读回寄存器并写回 GMEM。



# SMEM matrix descriptor

![image\.png](图片和附件/image.png)

`start_address` （比特位 0–13）—— 操作数在 SMEM 中的起始字节地址除以 16 的值。因此粒度为 16 B（一个 `uint128` ），该字段可寻址的 SMEM 大小上限为 `2^14 * 16 = 256 KB` ，远大于 Blackwell 的 232 KB。

`leading_byte_offset` / LBO（比特位 16–29）—— 硬件用来遍历操作数的两个 stride 之一。同样以 16\-B 单位存储。它是沿着连续维度的 stride（对于 K\-major 为 K， 对于 MN\-major 为 M）。

`stride_byte_offset` / SBO（位 32–45）——另一个 stride。同样以 16\-B 为单位。它是沿非连续维度的 stride（K\-major 时为 M，MN\-major 时为 K）。

`layout_type` （比特位 61–63）—— swizzle 模式： `SWIZZLE_NONE=0` 、 `SWIZZLE_128B=2` 、 `SWIZZLE_64B=4` 、 `SWIZZLE_32B=6` 。这与上文中的 8 种 swizzle 布局相同。

四种`primitive building blocks`原始构建块表示一个SMEM tile：

|SMEM tile|M=128 × K=128 \(= 128 × 256 B\)|从 GMEM 进行拷贝的高级别 SMEM 区域的一个 TMA 阶段|
|---|---|---|
|Swizzle atom|8 × 128 B \(= M=8 × K=64 bf16\)|`Swizzle 128B` 布局的构建块。对 TMA 友好：每条 TMA 指令一次加载多个 swizzle atom。|
|MMA subtile|M=64 × K=16 \(= 64 × 32 B\)|一个 `tcgen05.mma` 指令的 A operand。|
|8×16B chunk|8 × 16 B \(= 8 rows of one uint128\)|MMA 硬件访问的原始单位。每个块完全位于一个 swizzle atom内。|

![image\.png](图片和附件/image%201.png)

1. SMEM tile（ `(M=128, K=128)` ）是一个高层级的 SMEM 区域，一个 DMA 阶段从 GMEM 将其拷贝过来。

2. 通过堆叠 swizzle atom来构造用于 TMA 的 SMEM tile。SMEM tile字面上就是 `tile_to_shape(Layout_K_SW128_Atom, (M=128, K=128))` —— 一个由 `8x128B` swizzle atom组成的 16×2 网格（图中蓝色或橙色方框）。单个 TMA 指令（方框）一次加载多个沿 M 堆叠的 swizzle atom。

3. MMA 通过 MMA 子 tile 消费 SMEM  tile ，每个 MMA 子 tile 对应一个 `tcgen05.mma` 指令（图中红色方框）。对于 bf16，一个 `tcgen05.mma` 指令处理一个 `M=64, K=16` A 子 tile 。整个 SMEM 瓷片被划分为 2x8=16 个 MMA 子 tile 。这通常被称为 MMA 分区布局 `tCsA = ((MMA_M, MMA_K), Num_MMA_M, Num_MMA_K) = ((64, 16), 2, 8)` 。

每个 MMA 子块又是一个由 8×16B 块组成的网格。MMA 硬件并不识别 swizzle  atom  —— 它以 `8×16B` 块为单位访问 SMEM。一个 swizzle atom 包含 8 个块；一个 MMA 子块包含 8x2=16 个块。

K\-major SMEM 描述符（ layout\_type 、 SBO 和 LBO ）描述了 8x16B 个 chunk 在 SMEM 中一个 MMA 子瓦片内的布局方式。

MN\-major SMEM 描述符（ layout\_type 、 SBO 和 LBO ）描述了 swizzle atom 在 SMEM 中一个 MMA 子瓦片内的布局方式。

每个 MMA 子子块由一个 SMEM 描述符描述。在不同的 MMA 子子块之间， 8x16B 块/抖动原子以相同的方式布局。唯一的区别是子子块的 start\_address 。因此，MMA 子子块之间的 SMEM 描述符更新就是在描述符中对 start\_address 的更新。

![20260707\-174041\.jpg](图片和附件/20260707-174041.jpg)

# tcgen05
