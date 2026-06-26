# CUTLASS 示例：简单 UMMA

[https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/blackwell/01_mma_sm100.cu]

了解更基础的讨论，请参阅我们之前的系列博客。[https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/]

## GMEM 分块器与分区

首先，我们需要将全局输入张量按 tile 分区，并将它们分配给 CTA 进行处理。在此示例中，我们没有多个 SM 协同处理同一个 UMMA，因此 CTA 的 tile 与 UMMA 的 tile 相同（在更复杂的设置中并非如此，稍后文章会讨论）。因此我们先创建 tiled_mma 对象，然后根据 tiled_mma 确定分块器的维度。

```cpp
TiledMMA tiled_mma = make_tiled_mma(SM100_MMA_F16BF16_SS<TypeA, TypeB, TypeC,                 
                                                         128, 256,
                                                         UMMA::Major::K, 
                                                         UMMA::Major::K>{});
auto bM = tile_size<0>(tiled_mma);             // 1 MMA per CTA tile
auto bN = tile_size<1>(tiled_mma);             // 1 MMA per CTA tile
auto bK = tile_size<2>(tiled_mma) * Int<4>{};  // 4 MMA per CTA tile
auto mma_tiler = make_shape(bM, bN, bK);       // (MMA_M, MMA_N, MMA_K)
```

这里需要区分的一点是，MMA_K 中的因子 4 表示每个 K-tile 内的 MMA 数量，而不是 K-tile 的数量。这意味着每次从 GMEM 到 SMEM 的拷贝会对应 4 次 UMMA 调用，因为该拷贝是按 tile 进行的。

打印 MMA :

TiledMMA
  ThrLayoutVMNK:  (_1,_1,_1,_1):(_0,_0,_0,_0)
  PermutationMNK: (_,_,_)
MMA_Atom
  ThrID:    _1:_0
  Shape_MNK:  (_128,_256,_16)                   // MmaM, MmaN, MmaK instruction size
  LayoutA_TV: (_1,(_128,_16)):(_0,(_1,_128))    // TV -> MmaCoordinate mapping for A
  LayoutB_TV: (_1,(_256,_16)):(_0,(_1,_256))    // TV -> MmaCoordinate mapping for B
  LayoutC_TV: (_1,(_128,_256)):(_0,(_1,_128))   // TV -> MmaCoordinate mapping for C

正如我们在 MMA atom 中看到的，所有"线程布局"（thread layout）都被重新定义为指代协同执行 MMA 的 CTA 的布局。在此示例中，每个 TiledMMA 仅使用一个 CTA，因此所有这些布局的大小均为 1。A、B 和 C 的值布局按预期显示其形状。然后我们得到每个 CTA 对应的以下 GMEM 张量：

print(gA);   // (_128,_64,4):(256,_1,_64)
print(gB);   // (_256,_64,4):(256,_1,_64)
print(gC);   // (_128,_256):(1024,_1)
print(gD);   // (_128,_256):(1024,_1)

我们看到这些布局的模式中出现了静态整数 bM=128, bN=256, bK=64，以及动态整数 4（因为在此示例中取 K=256）。

由于 thread layout 被重新用于 Peer CTA layout，因此 tiled_mma 现在按 CTA 对等 ID 而不是线程 ID 进行切分。不过在此示例中，我们只有一个 CTA，所以可以直接按 _0{} 切分。

ThrMMA cta_mma = tiled_mma.get_slice(_0{});
Tensor tCgA = cta_mma.partition_A(gA);        // (MmaA, NumMma_M, NumMma_K, Tiles_K)
Tensor tCgB = cta_mma.partition_B(gB);        // (MmaB, NumMma_N, NumMma_K, Tiles_K)
Tensor tCgC = cta_mma.partition_C(gC);        // (MmaC, NumMma_M, NumMma_N)
Tensor tCgD = cta_mma.partition_C(gD);        // (MmaC, NumMma_M, NumMma_N)

print(tCgA); // ((_128,_16),_1,_4,4):((256,_1),_0,_16,_64)
print(tCgB); // ((_256,_16),_1,_4,4):((256,_1),_0,_16,_64)
print(tCgC); // ((_128,_256),_1,_1):((1024,_1),_0,_0)
print(tCgD); // ((_128,_256),_1,_1):((1024,_1),_0,_0)

此更改反映在分片 MMA 的命名中。在针对 Hopper 的 CuTe 示例中，分片 MMA 通常标记为 thr_mma，但现在称为 cta_mma。最后，分区后的 GMEM 张量的布局与从 MMA atom 尺寸 128x256x16 推断出的预期布局一致。

### 处理集群（Clusters）

到目前为止，我们只讨论了针对单个 SM 的 UMMA 情况。然而，当每个 UMMA 涉及 2 个 SM 时，UMMA 的形状与 CTA 的形状不再相同，因此我们需要用 Peer CTA ID（即 CTA 在其配对中的位置，0 或 1）对 tiled_mma 进行切分。我们在此简要介绍如何处理这种情况，更广泛的讨论将在本系列的第 2 部分中展开。

每个 CTA 对必然由集群中相邻的两个 CTA 组成。这意味着我们可以按如下方式提取 Peer CTA ID。

```cpp
Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                         make_tile(typename TiledMMA::AtomThrID{}));
auto mma_coord_vmnk = make_coord(
                   blockIdx.x % size<0>(cluster_layout_vmnk), // Peer CTA coordinate
                   blockIdx.x / size<0>(cluster_layout_vmnk), //    MMA-M coordinate
                   blockIdx.y,                                //    MMA-N coordinate
                   _);                                        //    MMA-K coordinate
 
  auto mma_v = get<0>(mma_coord_vmnk);
  ThrMMA cta_mma = tiled_mma.get_slice(mma_v);   // Use Peer CTA coordinate
```

cluster_layout_vmnk 用于创建感知 CTA 对的 cluster_shape；AtomThrID 为 1 或 2，取决于指定的 UMMA atom 是否使用 CTA 对。然后使用它来计算 CTA 的四维坐标，其中第 0 维是 Peer CTA ID。注意，当 size<0>(cluster_layout_vmnk) 为 1（无 CTA 对）时，该坐标简化为更常见的形式 (1, blockIdx.x, blockIdx.y, _)。

最后，我们可以使用第 0 模式对 tiled_mma 进行切分。再次说明，在这个特定示例中，mma_v 始终为 0，因为只有 1 个 CTA。但在后面的示例中，mma_v 将为 0 或 1。

SMEM 布局与 Swizzle（交错排列）

我们现在已经为全局张量准备好了分块器（tiler），因此拷贝的源端已经就绪。接下来是目标端：SMEM。对于目标张量 A（即 tCsA），应按照形状 (MmaA, NumMma_M, NumMma_K) = ((_128,_16),_1,_4) 进行组织，以与 GMEM 布局保持一致。CUTLASS 提供了一个实用函数来创建所需的形状。

```cpp
auto mma_shape_A = partition_shape_A(tiled_mma, make_shape(size<0>(mma_tiler),
                                                           size<2>(mma_tiler)));
auto mma_shape_B = partition_shape_B(tiled_mma, make_shape(size<1>(mma_tiler), 
                                                           size<2>(mma_tiler)));
```

为了优化 SMEM 访问，布局也应进行 swizzle（交错排列），其做法如下。

```cpp
// Sw<3,4,3> o smem_ptr[16b](unset) o ((_128,_16),_1,_4):((_64,_1),_0,_16)
auto sA_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeA>{},
                                         mma_shape_A);
// Sw<3,4,3> o smem_ptr[16b](unset) o ((_256,_16),_1,_4):((_64,_1),_0,_16)
auto sB_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeB>{}, 
                                         mma_shape_B);
```

这里，Layout_K_SW128_Atom<TypeA> 是针对 K-major A 数据类型 TypeA 的 128 字节宽 swizzle。swizzle 的宽度由连续维度上 tile 的大小决定。在本例中，K 维有 4 个大小为 16 的 tile，半精度为 2 字节，因此宽度为 16×4×2=128 字节。有关 MMA swizzle 的更多细节，请参见这篇文章。

与其他 CUTLASS 代码一样，此示例动态分配 SMEM 并将其作为一个 SharedStorage 结构来管理。在本例中，SharedStorage 保存 A 和 B 的 tile，以及一个用于管理 MMA 异步性的 mbarrier 对象。为处理 TMEM 分配，SharedStorage 还保存了一个用于 TMEM 基址的 32 位地址。

```cpp
template <class TypeA,           // Tensor A data type
          class TypeB,          // Tensor B data type
          class ASmemLayout,    // (MmaA, NumMma_M, NumMma_K, ...)
          class BSmemLayout>     // (MmaB, NumMma_N, NumMma_K, ...)
struct SharedStorage
{
  alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> A;
  alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout>> B;
 
  alignas(16) cute::uint64_t mma_barrier;  // Barrier to track MMA computation on SMEM
  alignas(16) cute::uint32_t tmem_base_ptr;  // Base pointer for TMEM allocation
 
  CUTE_DEVICE constexpr auto tensor_sA() { return make_tensor(make_smem_ptr(A.begin()), ASmemLayout{}); }
  CUTE_DEVICE constexpr auto tensor_sB() { return make_tensor(make_smem_ptr(B.begin()), BSmemLayout{}); }
};
```

此示例使用自动向量化的 cute::cooperative_copy 将数据从 GMEM 复制到 SMEM。我们也可以改用 TiledCopy 或像往常一样使用 TMA。

输入和输出描述符（Input and Output Descriptors）

UMMA 可以将第一个输入来源设为 SMEM 或 TMEM，第二个输入必须位于 SMEM，累加器必须位于 TMEM。示例中使用的特定 atom 变体将两个输入都取自 SMEM。

要创建这些描述符，我们使用与 Hopper 及更早 GEMM 中相同的 cta_mma.make_fragment 方法。

```cpp
// Represent the SMEM buffers for A and B
Tensor tCsA = shared_storage.tensor_sA();      // (MmaA, NumMma_M, NumMma_K)
Tensor tCsB = shared_storage.tensor_sB();      // (MmaB, NumMma_M, NumMma_K)
 
Tensor tCrA = cta_mma.make_fragment_A(tCsA);
Tensor tCrB = cta_mma.make_fragment_B(tCsB);
 
Tensor tCtAcc = cta_mma.make_fragment_C(tCgC); // (MmaC, NumMma_M, NumMma_N)
```

就像在 Hopper 的 WGMMA 中一样，operand tensor 并不是以寄存器为后端的数据张量，而是 SMEM 矩阵描述符的张量。例如，打印 tCrA 显示

tCrA:   UMMA::DescriptorIterator o (_1,_1,_4):(_0,_0,_2)

每个 MMA atom 使用一个描述符，按 (NumMma_M, NumMma_K) = (_1, _4) 平铺。我们之前在关于 WGMMA 的博客中已经介绍过矩阵描述符。

这里的累加器张量是一个普通的以 TMEM 为后端的张量，但它的布局起初可能不容易理解：

tCtAcc: tmem_[32b](TMEM_ADDR) o ((_128,_256),_1,_1):((_65536,_1),_0,_0)


TMEM 地址的步长是 65536；这是因为我们之前讨论过的 TMEM 的 32 位寻址方案。该地址的高 16 位表示 lane，而低 16 位表示列。这里的技巧是 65536 = 1<<16。例如，坐标 (1,1) 会变成：

(1,1) = (1*1<<16) + 1 = x0001.0001

它是对应于第 1 列第 1 条 lane 的 32 位地址（以十六进制表示）。

## GEMM 与同步

就像 Hopper 的 WGMMA 一样，UMMA 是异步的，因此需要同步。示例使用了一些 CUTLASS 的快捷方式以及围绕 mbarrier 的抽象。下面是示例中显示工作流程的摘录。

```cpp
if (elect_one_warp && elect_one_thr) {
  cute::initialize_barrier(shared_storage.mma_barrier, /* num_ctas */ 1);
}
int mma_barrier_phase_bit = 0;  // Each barrier has an associated phase_bit.
__syncthreads();
 
// Initial MMA overwrites the accumulators
tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
for (int k_tile = 0; k_tile < size<3>(tCgA); ++k_tile)
{
  // ... copy data in ...
 
  // Only one warp starts UMMAs
  if (elect_one_warp) {
    // Execute a MmaTile_M x MmaTile_N x MmaTile_K GEMM
    for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
      gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCtAcc);
      // Non-initial MMAs accumulate into the accumulators
      tiled_mma.accumulate_ = UMMA::ScaleOut::One;
    }
    // Ensure MMAs are completed, only then we can reuse the A and B SMEM.
    cutlass::arch::umma_arrive(&shared_storage.mma_barrier);
  }
  // All warps wait for MMAs to complete to avoid overwriting the A and B SMEM.
  cute::wait_barrier(shared_storage.mma_barrier, mma_barrier_phase_bit);
  mma_barrier_phase_bit ^= 1;
}
 
// ... copy data out ...
```


这些同步构造与用于 TMA 的基本相同。如果你想要关于 TMA 和同步的入门教程，请参阅我们之前的博客。值得注意的一点是，mbarrier 由将要发起 UMMA 的那个 warp 中的一个线程进行初始化。

gemm 调用和循环结构应该也和 Hopper 示例类似。主要需要注意的区别是只有一个 warp 发起 UMMA。回想一下，只有一个线程应该发出 PTX UMMA 指令。CUTLASS 在其 UMMA atom 的实现中在底层选择了这个线程，因此如果从多个线程调用 cute::gemm 实际上会导致死锁。

最后值得一提的是 UMMA::ScaleOut::Zero 。这会指示 UMMA 覆写 TMEM，而不是在已有值上累加。经过第一次 k_block 迭代后，这会被设置为 UMMA::ScaleOut::One ，以便结果开始累加。

## 从 TMEM 复制（Copy out of TMEM）

一旦所有 MMA 完成，我们需要将累加器结果从 TMEM 复制到寄存器。这是使用 PTX 的 tcgen05.ld 指令完成的。CUTLASS 将 tcgen05.ld 抽象为一个 copy atom，我们之前看到的不同变体在 cute/atom/copy_traits_sm100.hpp 中定义的 copy traits 中表示。我们的示例使用 SM100_TMEM_LOAD_32dp32b1x atom。我们可以在围绕该 atom 的 PTX 包装器（位于 cute/arch/copy_sm100.hpp）中看到这如何被翻译成正确的变体。

```cpp
// 32 data path lanes, 32-bit pattern, repeated 1 times
struct SM100_TMEM_LOAD_32dp32b1x
{
  using SRegisters = uint32_t[1];
  using DRegisters = uint32_t[1];
 
  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src_addr,
       uint32_t& dst0)
  {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.ld.sync.aligned.32x32b.x1.b32"
                    "{%0},"
                    "[%1];\n"
    :  "=r"(dst0)
    :  "r"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use TMEM_LOAD without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
  }
};
```

使用这个 atom，我们可以设置一个 TiledCopy，将累加器结果从 TMEM 提取到 RMEM。注意，与本示例中其余的 CTA 级操作不同，这里回到了 warp 和线程级操作——因为数据必须移动到寄存器中以执行 epilogue。

```cpp
// Create the tiled copy operation for the accumulator (TMEM -> RMEM)
TiledCopy tiled_t2r_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
ThrCopy   thr_t2r_copy   = tiled_t2r_copy.get_slice(threadIdx.x);
 
//...
 
Tensor tDtAcc = thr_t2r_copy.partition_S(tCtAcc);    
Tensor tDgD   = thr_t2r_copy.partition_D(tCgD);     
using AccType = typename decltype(tCtAcc)::value_type;
Tensor tDrAcc = make_tensor<AccType>(shape(tDgD));   
// Load TMEM -> RMEM
```

在这里我们使用一个专用函数 make_tmem_copy，从 copy atom 和 TMEM 张量推导出一个 TV 布局并创建 TiledCopy。关于此函数需要知道的一点是，它被硬编码为使用 4 个 warp，即 1 个 warpgroup。如前面章节所述，TMEM 的某些区域只能由 warpgroup 中对应的 warp 访问，基于 warp 索引对 4 取模。PTX 手册中的这张图展示了在我们的示例中数据如何分配给各个 warp：

![alt text](image.png)

下面来自 PTX 手册的图表展示了这映射到的 TMEM 地址。

![alt text](image-1.png)

要了解 CuTe 如何处理此复制操作，我们可以查看位于 cute/atom/copy_traits_sm100.hpp 的 traits 结构体。

```cpp
template <>
struct Copy_Traits<SM100_TMEM_LOAD_32dp32b1x>
     : TMEM_LOAD_Unpack<SM100_TMEM_LOAD_32dp32b1x>
{
  using ThrID = Layout<_32>;
  using ValID = Layout<Shape <_32, _32>, Stride< _1,TMEM::DP_b>>;
  using SrcLayout = Layout<Shape <_32, _1024>, Stride< _0, _1>>;
  using DstLayout = Layout<Shape <_32, _32>, Stride<_32, _1>>;
  using RefLayout = SrcLayout;
};
```

布局 ThrID 定义了从逻辑线程 ID 到 warp 中线程索引的映射；值 32 表示这是一个跨整个 warp 的操作。

ValID 给出了从逻辑位 ID 到位地址的映射；例如，位 35 被 ValID 布局映射到 lane 1 的第 3 位。该布局的形状为 (bit, lane)，且 lane 的 stride，即 TMEM::DP_b，是 1<<21；其中 1<<16 来自我们之前看到的 TMEM 寻址方案，额外的 5 是因为单元宽度为 1<<5=32 位。

SrcLayout 给出了从 (src-thread, src-bit) 到 bit-target 的映射。此加载为整 warp 的操作，输入基地址在整个 warp 中相同。因此线程维被抑制（stride 为 0），以便将 src-bit 映射到 bit。

最后，DstLayout 显示了 (dst-thread, dst-bit) 到 bit 的映射。布局的形状 <32,32> 告诉我们每个线程负责写出 32 位（1 个寄存器）。注意，对于 32dp32b 来说这个布局很简单，因为 TMEM 中的 lane 和 column 可以直接映射到输出的行和列。但对于更复杂的加载模式，我们需要这个布局来确定输出 RMEM 位如何映射到逻辑位索引。

现在回到代码中，由该 atom 创建的 TiledCopy 用于对输出矩阵进行分区。然后按线程 ID 切分这些分区以获得每个线程的 Tensor。鉴于我们的 MMA 尺寸为 128×256，下面是为线程 0 打印的 Tensor（为便于参考再次显示 tCtAcc）：

// reproduced from above
tCtAcc: tmem_[32b](0x0000.0000) o ((_128,_256),_1,_1):((_65536,_1),_0,_0)
// new tensors for tmem -> rmem copy
tDtAcc: tmem_[32b](0x0000.0000) o ((_32,_1),_256,_1,_1):((_65536,_0),_1,_0,_0)
tDrAcc: ptr[32b](0x705671fff290) o ((_1,_1),_256,_1,_1):((_0,_0),_1,_0,_0)

我们可以在 tCtAcc 中直接看到 128×256 的 MMA 大小。分区后的 tDtAcc 是一个每线程的张量，映射到 TMEM 地址。再次注意，同一 warp 中的每个线程以统一方式读取相同的 TMEM 地址，这也解释了 value 模式下的子布局 (_32, _1) : (_65536, _1)。在 4 个 warp 中共有 128 个线程，这覆盖了 M 模式。第 1 个模式表示该模式重复 256 次以覆盖 N 模式，因此我们得到 128×256 的 tile。最后两个值为 1 表示 M-tile 和 N-tile，在我们的例子中均为 1。在 tDrAcc 方面，主要区别是这里表示的是寄存器。因此由于每个线程负责 TMEM 中的一个 32 位单元，在 value 模式下我们只看到 (_1, _1)。同样在 4 个 warp 的 128 个线程下，这覆盖了 M 模式。其他模式与 tDtAcc 相同。

最后，一旦累加器被复制到 RMEM，就可以在写回 GMEM 之前对其进行后处理（例如 axpby）。

## 分配和释放 TMEM

对于这个基础示例还有一个额外的话题需要讨论：TMEM 的分配和释放。我们可以使用 CuTe 帮助类 cute::TMEM::Allocator1Sm 来实现，这个类提供了对上文讨论的 tcgen05.alloc 和 tcgen05.dealloc 函数的接口。基本模式如下。

// instantiate the allocator 
cute::TMEM::Allocator1Sm tmem_allocator{};
 
if (elect_one_warp) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);
}
__syncthreads();
tCtAcc.data() = shared_storage.tmem_base_ptr;   // move accumulator offset
 
// rest of kernel
 
if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(shared_storage.tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
  }

如前节所述，一个 warp 执行分配，传入列数和指向共享内存中 32 位值的指针；allocate 方法随后存储已分配 TMEM 起始位置（最低的 (lane, column)）的 32 位地址。尽管此 MMA 指令只需要 256 列，为简化起见内核会分配全部 512 列 TMEM。请注意，虽然只有一个线程将 TMEM 地址传给 MMA 指令，但所有线程在 epilogue 从 TMEM 加载数据时都需要该地址，因此必须通过共享内存传递。最后，调用 allocate 的同一个 warp 也必须调用 free。作为一个稍微进阶的特性，release_allocation_lock 方法是 tcgen05.relinquish_alloc_permit 的包装；显然，这是对 CTA 不再进行任何进一步 TMEM 分配的保证，允许后续 CTA 排队等候同一 SM。你可以在 CUTLASS sm100 GEMM kernel 中看到关于 TMEM 管理的一些更完整的示例。

为便于 TMEM 管理，nvcc 添加了标志 --g-tensor-memory-access-check 。启用此标志后，运行时内核在任何未初始化或越界的 TMEM 访问时都会报错并打印错误信息。