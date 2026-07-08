# tcgen05 micro\-benchmark

# 引言及tcgen05基础语法

Ampere tensor core（ptx 为 mma）从 register file（RF）中获取输入 A 和 B 矩阵。Hopper tensor core（ptx 为 wgmma）从 RF/SMEM 中获取输入 A 矩阵，从 SMEM 中获取输入 B 矩阵。

而我们关注的 Blackwell\-family Tensor Core 路径对应 PTX `tcgen05.mma`。在该路径中，A 可以来自 SMEM 或 TMEM，B 来自 SMEM，accumulator C/D 保存在 TMEM 中。常见 GEMM 路径是先将 A/B tile 搬入 SMEM，再通过 SMEM descriptor 交给 `tcgen05.mma` 读取（TS：A在TMEM、B在SMEM，A tile 被`tcgen05.cp`存入TMEM；SS：A/B都在SMEM。）。pipeline 如下图所示：

![img\_v3\_0213d\_d8abef8b\-af78\-405e\-8b4e\-ccb3557bcaag\.jpg](图片和附件/img_v3_0213d_d8abef8b-af78-405e-8b4e-ccb3557bcaag.jpg)

![img\_v3\_0213d\_ab5d3d01\-d9b6\-4678\-828b\-01ec28efde9g\.jpg](图片和附件/img_v3_0213d_ab5d3d01-d9b6-4678-828b-01ec28efde9g.jpg)

```Plain Text
// 1. Floating-point type without block scaling:

tcgen05.mma.cta_group.kind   [d-tmem],  a-desc,  b-desc, idesc,
                             { disable-output-lane }, enable-input-d {, scale-input-d};

tcgen05.mma.cta_group.kind   [d-tmem], [a-tmem], b-desc, idesc,
                             { disable-output-lane }, enable-input-d {, scale-input-d};

.kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4 }
.cta_group = { .cta_group::1, .cta_group::2 }

----------------------------------------------------------------------------------

// 2. Floating-point type with block scaling:

tcgen05.mma.cta_group.kind.block_scale{.scale_vectorsize}
                                        [d-tmem],  a-desc,  b-desc, idesc,
                                        [scale-A-tmem], [scale-B-tmem], enable-input-d;

tcgen05.mma.cta_group.kind.block_scale{.scale_vectorsize}
                                        [d-tmem], [a-tmem], b-desc, idesc,
                                        [scale-A-tmem], [scale-B-tmem], enable-input-d;

.kind = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
.cta_group      = { .cta_group::1,   .cta_group::2 }
.scale_vectorsize = { .scale_vec::1X, .scale_vec::2X, .scale_vec::4X, .block16, .block32 }

----------------------------------------------------------------------------------

// 3. Convolution MMA for floating-point type without block scaling:

tcgen05.mma.cta_group.kind.collector_usage [d-tmem],  a-desc,  b-desc, idesc,
                                           { disable-output-lane }, enable-input-d {, scale-input-d};

tcgen05.mma.cta_group.kind{.ashift}.collector_usage [d-tmem], [a-tmem], b-desc, idesc,
                                                    { disable-output-lane }, enable-input-d {, scale-input-d};

tcgen05.mma.cta_group.kind.ashift{.collector_usage} [d-tmem], [a-tmem], b-desc, idesc,
                                                    { disable-output-lane }, enable-input-d {, scale-input-d};

.kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4 }
.cta_group = { .cta_group::1,   .cta_group::2 }
.collector_usage = { .collector::buffer::op }
::buffer         = { ::a }
::op             = { ::fill, ::use, ::lastuse, ::discard* }

----------------------------------------------------------------------------------

// 4. Activation Stationary MMA for floating-point type with block scaling:

tcgen05.mma.cta_group.kind.block_scale{.scale_vectorsize}.collector_usage
                                            [d-tmem],  a-desc,  b-desc, idesc,
                                            [scale-A-tmem], [scale-B-tmem], enable-input-d;

tcgen05.mma.cta_group.kind.block_scale{.scale_vectorsize}.collector_usage
                                            [d-tmem], [a-tmem], b-desc, idesc,
                                            [scale-A-tmem], [scale-B-tmem], enable-input-d;

.cta_group       = { .cta_group::1,   .cta_group::2 }
.scale_vectorsize  = { .scale_vec::1X, .scale_vec::2X, .scale_vec::4X, .block16, .block32 }
.kind            = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
.collector_usage = { .collector::buffer::op }
::buffer         = { ::a }
::op             = { ::fill, ::use, ::lastuse, ::discard* }

----------------------------------------------------------------------------------

// 5. Integer type:

tcgen05.mma.cta_group.kind::i8  [d-tmem],  a-desc,  b-desc, idesc,
                                { disable-output-lane }, enable-input-d;

tcgen05.mma.cta_group.kind::i8  [d-tmem], [a-tmem], b-desc, idesc,
                                { disable-output-lane }, enable-input-d;

.cta_group = { .cta_group::1,   .cta_group::2  }

----------------------------------------------------------------------------------

// 6. Convolution MMA for integer type:

tcgen05.mma.cta_group.kind::i8.collector_usage          [d-tmem],  a-desc,  b-desc, idesc,
                                                        { disable-output-lane }, enable-input-d;

tcgen05.mma.cta_group.kind::i8.ashift{.collector_usage} [d-tmem], [a-tmem], b-desc, idesc,
                                                        { disable-output-lane }, enable-input-d;

tcgen05.mma.cta_group.kind::i8{.ashift}.collector_usage [d-tmem], [a-tmem], b-desc, idesc,
                                                        { disable-output-lane }, enable-input-d;

.cta_group       = { .cta_group::1,   .cta_group::2  }
.collector_usage = { .collector::buffer::op }
::buffer         = { ::a }
::op             = { ::fill, ::use, ::lastuse, ::discard* }
```





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





