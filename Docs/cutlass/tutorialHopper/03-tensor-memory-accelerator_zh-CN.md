# CUTLASS 教程：掌握 NVIDIA® 张量内存加速器（TMA）

TMA（张量内存加速器）是 NVIDIA Hopper™ 架构引入的新特性，用于在 GPU 全局内存（GMEM）与线程块（即 CTA）的共享内存（SMEM）之间执行异步内存拷贝。与早期方法相比，TMA 提供了多项优势，例如：（1）借助异步性促进 [warp 专门化](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md#warp-specialization)内核调度，从而提高 GPU 利用率；（2）通过 TMA 拷贝描述符，以单线程方式处理地址、步长等拷贝辅助数据的计算。后者更节省寄存器，也必然会处理谓词判定（例如越界检查）。NVIDIA [技术博客](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)和 [Hopper 调优指南](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#tensor-memory-accelerator)对这些优势做了很好的阐述；如果希望理解 TMA 的设计动机，强烈建议阅读这些资料。

与上述资料不同，本文聚焦建立“如何编写使用 TMA 的内核”这一可操作的理解。全文依赖 [CuTe 库](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md)；CuTe 通过封装底层 GPU 指令的 API 暴露 TMA。这些底层指令包括 PTX [`cp.async.bulk.tensor`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor) 和 [`cp.reduce.async.bulk.tensor`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-reduce-async-bulk-tensor)，以及 [`cuTensorMap`](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html) 操作数；本文也会讨论它们。

本文分为三个主要部分：第一部分讨论 TMA 加载，第二部分讨论 TMA 存储，第三部分介绍 TMA 存储归约和 TMA 加载多播等更高级的操作。本质上，TMA 加载把数据从 GPU GMEM 拷贝（“加载”）到某个 CTA 的 SMEM，TMA 存储则把数据从 CTA SMEM 拷贝（“存储”）到 GPU GMEM。TMA 加载、TMA 存储和更高级的变体共享许多概念，因此大部分必要概念将在 TMA 加载部分介绍，后续部分只聚焦剩余差异。

此外，TMA 是在[异步代理](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#async-proxy)中执行的异步操作，因此需要使用某些内存一致性强制工具，例如异步内存屏障（[`mbarrier`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier)）和异步内存栅栏（[`fence.proxy.async`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar-fence)），以保证内核行为正确。同步本身是一个非常广泛的主题，本文只介绍实际使用所需的部分。

最后，如果读者希望查阅覆盖许多相同要点，但不涉及 CUTLASS 或 CuTe 概念的资料，推荐参阅 [CUDA® 编程指南中对 TMA 的介绍](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tensor-memory-access)。

## TMA 加载

TMA 加载将数据从 GMEM 拷贝到 SMEM。本节展示如何编写使用 TMA 加载完成该目标的内核。使用 TMA 加载的内核与使用其他内存拷贝方法的内核存在很大差异，因此先通过一个简单示例任务展示如何编写该内核，然后再解释涉及的概念。

#### 示例任务

为展示 TMA 加载的用法，考虑一个对二维行主序矩阵进行分块的简单任务。给定形状为 `[m,n]` 的矩阵 `A` 和两个正整数 `CTA_M`、`CTA_N`。`CTA_M` 和 `CTA_N` 在编译期已知，`m` 和 `n` 则在运行时由矩阵 `A` 给出。为简化讨论，还假设 `m % CTA_M == n % CTA_N == 0`；下文会看到，该要求可以放宽。

启动一个大小为 `{m/CTA_M, n/CTA_N, 1}` 的 CTA 网格。第 `(i,j)` 个 CTA 的 SMEM 保存来自 `A` 的第 `(i,j)` 个、形状为 `[CTA_M, CTA_N]` 的矩阵块。可以用以下 `numpy` 伪代码描述该分配：

```python
A = np.random.uniform(M, N)
for i in range(M):
  for j in range(N):
    cta_i_j = A.reshape(M // CTA_M, CTA_M, N // CTA_N, N)[i, :, j, :]
```

两步流程。该任务使用 TMA 加载完成。在 CuTe 中，TMA 加载分两步实现：第一步在主机代码中构造 TMA 拷贝描述符；第二步在内核代码中使用该描述符执行实际 TMA 加载。该两步流程与 CuTe `TiledCopy` 的常规用法不同；如[相关教程](https://github.com/NVIDIA/cutlass/blob/637b15906358191cb4238af419d408a65819d7ec/examples/cute/tutorial/tiled_copy.cu#L120-L124)所示，`TiledCopy` 的所有拷贝步骤通常都写在内核代码中。

#### 主机代码

在主机端创建三个对象：作为拷贝源的 GMEM 张量、每个 CTA 上作为拷贝目标的 SMEM 张量布局，以及以这两者为参数的 `tma_load` 对象。由于 SMEM 布局在主机端创建，因此就 TMA 加载而言，所有 CTA 共享同一个 SMEM 布局。创建完这些对象后，可将它们传给设备端内核，并在内核中调用 TMA 加载。

完整的主机代码如下：

```cpp
template <typename T, int CTA_M, int CTA_N>
void host_fn(T* data, int M, int N) {
  using namespace cute;
  // 创建 GMEM 张量
  auto gmem_layout = make_layout(make_shape(M, N), LayoutRight{});
  auto gmem_tensor = make_tensor(make_gmem_ptr(T), gmem_layout);
  // 创建 SMEM 布局
  auto smem_layout = make_layout(make_shape(CTA_M, CTA_N), LayoutRight{});
  // 创建 TMA 对象
  auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, gmem_tensor, smem_layout);
  // 调用内核
  tma_load_kernel<CTA_M, CTA_N>
                 <<<dim3{M / CTA_M, N / CTA_N, 1}, 1>>>
                 (tma_load, gmem_tensor, smem_layout);
}
```

创建 `gmem_layout`、`gmem_tensor` 和 `smem_tensor` 的各行只使用了 [CuTe](https://github.com/NVIDIA/cutlass/blob/637b15906358191cb4238af419d408a65819d7ec/media/docs/cute/02_layout_algebra.md) 基础概念，如需回顾可参阅[相关](https://github.com/NVIDIA/cutlass/blob/637b15906358191cb4238af419d408a65819d7ec/media/docs/cute/01_layout.md) [CuTe 教程](https://github.com/NVIDIA/cutlass/blob/637b15906358191cb4238af419d408a65819d7ec/media/docs/cute/03_tensor.md)。这里重点解释 `tma_load` 对象。该对象是 `cute::TiledCopy` 的实例，它保存信息并实现执行 CTA 级拷贝所需的方法。代码片段使用 `cute::make_tma_copy` 函数的[显式默认形式](https://github.com/NVIDIA/cutlass/blob/637b15906358191cb4238af419d408a65819d7ec/include/cute/atom/copy_traits_sm90_tma.hpp#L1206-L1217)创建 `tma_load`。该函数的完整实现包含一些细微之处，本文稍后讨论 `MULTICAST` 时会深入介绍。但显式默认形式已足以应对大多数用例，包括当前示例任务。建议使用该形式，避免不必要的复杂性和错误。

下面检视本例使用的 `make_tma_copy` 签名：

- 最后两个参数是 `gmem_tensor` 和 `smem_layout`。`make_tma_copy` 在内部使用这些信息创建 `TmaDescriptor`；该类型只是 [`CUtensorMap`](https://github.com/NVIDIA/cutlass/blob/637b15906358191cb4238af419d408a65819d7ec/include/cute/arch/copy_sm90_desc.hpp#L178) 的别名。描述符对象会在 TMA 内核中使用。
- 第一个参数是 [`SM90_TMA_LOAD`](https://github.com/NVIDIA/cutlass/blob/637b15906358191cb4238af419d408a65819d7ec/include/cute/arch/copy_sm90_tma.hpp#L269) 实例。该对象将拷贝操作分派到所需的 PTX `cp.async.bulk.tensor` 调用；本文第三部分会更深入地讨论。

#### 内核代码

相关内核代码片段如下。这些代码行包含多个重要 TMA 概念，下文将依次解释。

```cpp
template <typename T, int CTA_M, int CTA_N, class TmaLoad, class GmemTensor>
void tma_load_kernel(__grid_constant__ const TmaLoad tma_load, GmemTensor gmem_tensor) {
  using namespace cute;
  constexpr int tma_transaction_bytes = CTA_M * CTA_N * sizeof(T);
  __shared__ T smem_data[CTA_M * CTA_N];
  __shared__ uint64_t tma_load_mbar;
  auto smem_layout = make_layout(make_shape(CTA_M, CTA_N), LayoutRight{});
  auto smem_tensor = make_tensor(make_smem_ptr(smem_data), smem_layout);
  if (threadIdx.x == 0) {
    auto gmem_tensor_coord = tma_load.get_tma_tensor(shape(gmem_tensor));
    auto gmem_tensor_coord_cta = local_tile(
        gmem_tensor_coord,
        Tile<Int<CTA_M>, Int<CTA_N>>{},
        make_coord(blockIdx.x, blockIdx.y));
    initialize_barrier(tma_load_mbar, /* 到达计数 */ 1);
    set_barrier_transaction_bytes(tma_load_mbar, tma_transaction_bytes);
    auto tma_load_per_cta = tma_load.get_slice(0);
    copy(tma_load.with(tma_load_mbar),
         tma_load_per_cta.partition_S(gmem_tensor_coord_cta),
         tma_load_per_cta.partition_D(smem_tensor));
  }
  __syncthreads();
  wait_barrier(tma_load_mbar, /* 阶段 */ 0);
  // 执行到此行之后，TMA 加载已完成
}
```

首先，内核的 `tma_load` 参数必须在第 2 行标注为 `__grid_constant__ const`。如果有两个要从 GMEM 拷贝到 SMEM 的张量，每个张量都必须拥有自己的 `TiledCopy` 实例，且每个实例都必须是 `__grid_constant__ const`。这是将 `cuTensorMap` 从主机传到设备的要求，可参见[相关文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-data-copies-using-tensor-memory-access-tma)。

下一个重要之处是，TMA 拷贝只由一个线程负责发出。代码片段中，所有 TMA 相关变量和指令都位于从第 12 行开始的 `if` 块内，该块只由线程 0 执行。另一方面，第 30 行让 CTA 中所有线程等待 TMA 操作完成。

##### 坐标与算术元组

下面检视 TMA 加载逻辑。逻辑从第 13 行开始：该行创建 `gmem_tensor_coord` 对象，用于保存待拷贝 GMEM 张量的坐标。如果尝试以下代码：

```
if (cute::thread(0)) { cute::print(gmem_tensor_coord); }
```

则在 `M=N=1024` 时可看到以下输出：

```
ArithTuple(_0,_0) o (1024,1024):(_1@1,_1@0)
```

对熟悉 CuTe 分块拷贝工作方式的读者而言，第 15–18 行不难理解：GMEM 张量被分块为更小的分区，每个 CTA 根据 block 坐标对分块张量切片，以获得自己的 GMEM 视图。但请注意，这里被分区的是上述表示 `gmem_tensor` 坐标的 `ArithTuple`，而不是 `gmem_tensor` 本身。具体而言，`ArithTuple` 被分为形状为 `[CTA_M,CTA_N]` 的矩阵块，然后每个 CTA 取得自己的矩阵块。

如果如下使用 `print_tensor` 打印 `gmem_tensor_coord_cta`：

```
if (cute::block(7)) { cute::print_tensor(gmem_tensor_coord_cta); }
```

则当 `CTA_M == CTA_N == 16` 时，会看到：

```
ArithTuple(0,112) o (_16,_16):(_1@1,_1@0):
  (0,112)  (1,112)  (2,112)  (3,112)  (4,112)  (5,112)  (6,112)  (7,112)  (8,112)  (9,112)  (10,112)  (11,112)  (12,112)  (13,112)  (14,112)  (15,112)
  (0,113)  (1,113)  (2,113)  (3,113)  (4,113)  (5,113)  (6,113)  (7,113)  (8,113)  (9,113)  (10,113)  (11,113)  (12,113)  (13,113)  (14,113)  (15,113)
  // 更多行
  (0,127)  (1,127)  (2,127)  (3,127)  (4,127)  (5,127)  (6,127)  (7,127)  (8,127)  (9,127)  (10,127)  (11,127)  (12,127)  (13,127)  (14,127)  (15,127)
```

这些数字是 `gmem_tensor` 中的坐标，对应值将被拷贝到 CTA 7 的 `smem_tensor`。建议读者运行这段代码，并将 `cute::block(7)` 替换为其他索引，以理解各 CTA 分别从 `gmem_tensor` 的哪些坐标拷贝数据。

接下来，第 25–27 行发出的拷贝操作具有常规 `TiledCopy` 操作签名，只是源张量被替换为已分区的坐标。

##### 内存屏障

前面尚未解释第 20、22 和 30 行，这些行都涉及常驻于 SMEM 的 `uint64_t` 变量 `tma_load_mbar`。该变量是一个异步事务屏障，用于在 TMA 加载与内核中消费其 SMEM 加载结果的其余部分之间同步。NVIDIA Hopper 架构[技术博客](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)对这种屏障给出了高层描述。对当前内核而言，重要要点如下：

1. 第 20 行在共享内存中初始化 mbarrier 对象。CuTe 方法 `initialize_barrier` 封装 PTX 指令 `mbarrier.init.shared.b64`，该指令还接收一个到达计数参数。在本语境中，由于只有一个线程启动 TMA 加载，应将到达计数设为 1。此外，mbarrier 的起始阶段始终设为 0。
2. 第 22 行使用 CuTe 方法 `set_barrier_transaction_bytes`，既对 mbarrier 对象执行 arrive-on 操作，又设置其预期事务计数。该方法封装 PTX 指令 `mbarrier.arrive.expect_tx.shared::cta.b64`。事务计数设为 TMA 加载传输的字节数，该数值在第 4 行计算。

3. 第 25–27 行的拷贝指令会分派到所需的 `cp.async.bulk.tensor` 变体，并始终使用所提供 mbarrier 对象的 `mbarrier::complete_tx::bytes` 作为完成机制。

4. 第 30 行对 mbarrier 对象执行等待操作。请注意，所有线程都会等待 mbarrier，而只有线程 0 到达 mbarrier。在 `wait_barrier` 之前必须调用 `__syncthreads()`，以解决线程分歧。
此处，`wait_barrier` 封装 PTX 指令 `mbarrier.try_wait.parity.shared::cta.b64`。`try_wait` 限定符（与 `test_wait` 相对）表示该等待是阻塞指令。`parity` 限定符要求提供阶段位，表示线程休眠，直到 mbarrier 的该阶段位翻转。这是 mbarrier 初始化后第一次用于跟踪完成状态，因此阶段参数传入 0。如果再执行一次 TMA 加载，就必须翻转阶段以重用 mbarrier。
总体而言，当[软件流水化](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md#pipelining)方案连续执行多次 TMA 加载时，[CUTLASS Pipeline API](https://github.com/NVIDIA/cutlass/blob/main/media/docs/pipeline.md) 提供了更高层的 mbarrier 生命周期管理方式。

5. `wait_barrier` 之后，内存一致性模型提供以下保证：TMA 加载对 SMEM 的写入，对所有调用 mbarrier wait 的线程可见；在本示例内核中，即对 CTA 中所有线程可见。
##### TMA 剩余矩阵块与步长要求

上述示例假设 `m%CTA_M==0` 且 `n%CTA_N==0`。但就 TMA 加载而言，可以完全取消该假设。从 GMEM 向 SMEM 加载剩余矩阵块时，无需自行处理越界逻辑；TMA 拷贝单元必然会对内存拷贝进行[谓词判定](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/0y_predication.md)，以避免越界读取。这与上文 TMA 加载使用带 `ArithTuple` 的特殊“隐式”CuTe 张量是一致的。如果改用普通 CuTe 张量，对其切片可能产生指向 GMEM 越界位置的新 CuTe 张量，从而不可避免地导致错误。

但对 TMA 而言，必须牢记 GMEM 张量自身的一项重要步长要求：16 字节边界要求。正如可以预期的，TMA 不支持拷贝 GMEM 中任意步长的区域。必须假设待拷贝矩阵块（i）具有一个连续方向（步长为 1），且（ii）其他步长均为 16 字节的倍数。CUTLASS 代码库中已对该条件进行[断言检查](https://github.com/NVIDIA/cutlass/blob/7d49e6c7e2f8896c47f586706e67e1fb215529dc/include/cute/atom/copy_traits_sm90_tma.hpp#L846)。

例如，对形状为 `(m,n)`、步长为 `(n,1)` 的行主序 float GMEM 张量，该条件要求 `n%4==0`。如果不满足，可在调用内核前对输入张量填充，使其具有正确范围。

## TMA 存储

掌握 TMA 加载基础后，由于两种操作存在许多相似之处，学习 TMA 存储会容易得多。与 TMA 加载类似，TMA 存储也分两步实现：在主机端定义 TMA 拷贝描述符，然后在内核中发出 TMA 存储操作。

#### 示例任务与代码

为便于说明，考虑 TMA 加载示例的反向操作：从多个 CTA 的 SMEM 拷贝到已分区 GMEM 张量的相应矩阵块。不同之处在于，拷贝到 GMEM 之前，先使用简单数字模式填充 CTA 中的 SMEM 矩阵块，否则拷贝的将是未定义值。以下是可运行代码片段：

```
template <typename T, int CTA_M=32, int CTA_N=32>
void host_fn(T* data, int M, int N) {
  using namespace cute;
  // 创建 GMEM 张量
  auto gmem_layout = make_layout(make_shape(M, N), LayoutRight{});
  auto gmem_tensor = make_tensor(make_gmem_ptr(T), gmem_layout);
  // 创建 SMEM 布局
  auto smem_layout = make_layout(make_shape(CTA_M, CTA_N), LayoutRight{});
  // 创建 TMA 对象
  auto tma_store = make_tma_copy(SM90_TMA_STORE{}, gmem_tensor, smem_layout);
  // 调用内核
  tma_store_kernel<CTA_M, CTA_N>
                  <<<dim3{M / CTA_M, N / CTA_N, 1}, CTA_M>>>
                  (tma_store, gmem_tensor, smem_layout);
}
template <typename T, int CTA_M, int CTA_N, class TmaStore, class GmemTensor>
void tma_store_kernel(__grid_constant__ const TmaStore tma_store, GmemTensor gmem_tensor) {
  using namespace cute;
  __shared__ T smem_data[CTA_M * CTA_N];
  auto smem_layout = make_layout(make_shape(CTA_M, CTA_N), LayoutRight{});
  auto smem_tensor = make_tensor(make_smem_ptr(T), smem_layout);
  // 填充 smem_data 的各行
  for (int j = 0; j < CTA_N; ++j) {
    smem_data(threadIdx.x, j) = threadIdx.x;
  }
  __syncthreads();
  tma_store_fence();
  if (threadIdx.x == 0) {
    auto gmem_tensor_coord = tma_store.get_tma_tensor(shape(gmem_tensor));
    auto gmem_tensor_coord_cta = local_tile(
      gmem_tensor_coord,
      Tile<Int<CTA_M>, Int<CTA_N>>{},
      make_coord(blockIdx.x, blockIdx.y));
    auto tma_store_per_cta = tma_store.get_slice(0);
    copy(tma_store,
         tma_store_per_cta.partition_S(smem_tensor),
         tma_store_per_cta.partition_D(gmem_tensor_coord_per_cta));
    // tma_store_arrive();
  }
  // tma_store_wait<0>();
}
```

除了改为调用 `tma_store_kernel` 外，主机代码与 TMA 加载的主机代码几乎完全相同。请注意，每个 CTA 被配置为拥有 `CTA_M` 个线程。本示例让每个 CTA 在 SMEM 中保存一个 `[CTA_M,CTA_N]` 矩阵块，并在第 29–32 行中由线程 `i` 用值 `i` 填充第 `i` 行。

内核代码第 39–49 行的 `if` 块与 `tma_load_kernel` 中的 `if` 块相似。特别是，只有线程 `0` 发出 TMA 存储操作。所有张量分块逻辑在概念上都相同，但拷贝方向反转：对 TMA 存储，`tma_store_per_cta.partition_S` 方法应用于 `smem_tensor`，`tma_store_per_cta.partition_D` 方法应用于 GMEM 张量坐标。与 TMA 加载类似，这些坐标也使用 `ArithTuple` 表示。

##### 内存栅栏

TMA 加载与存储代码最重要的差异是，TMA 存储不再使用 mbarrier 对象。这是因为 TMA 存储使用另一种机制强制内存一致性：内存栅栏。

内存栅栏的目的，是在执行线程于栅栏之前和之后请求的内存访问之间建立有保证的顺序。本示例需要确保第 29–32 行对 SMEM 的所有写入，对线程 0 执行的 TMA 存储可见。为此，第 35 行使用 CuTe 方法 `tma_store_fence()`，该方法封装 PTX 指令 `fence.proxy.async.shared::cta`。

该指令包含两个用于描述栅栏效果的重要限定符：[scope](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#scope) 和 [proxykind](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#proxies)。scope 表示参与栅栏所强制顺序的线程集合。在本例中，限定符 `cta` 把 scope 定义为 CTA 中所有线程；就内存一致性模型而言，这是可能的最小 scope。proxykind 表示除通用代理外，还有哪种代理参与栅栏强制的顺序。本例选择 `async.shared`，因为对每个 CTA 而言，TMA 存储在异步代理中执行。如果用不涉及异步代理的其他内存栅栏基础操作（例如 `__threadfence_block()`）替换异步栅栏，就会破坏内核正确行为所需的保证，实际上导致竞态。

##### TMA 存储的 arrive 与 wait

第 49 和 51 行分别有 `tma_store_arrive()` 和 `tma_store_wait<Count>()`。前者提交 TMA 存储操作（技术上作为一个 `cp.async.bulk-group`）；后者等待，直到已提交的 TMA 存储操作中最多只有 `Count` 个仍在等待完成（例如，若要求全部完成，将 `Count` 设为 0）。如果内核中还有其他工作需要等待 TMA 存储完成，这些操作就很有用；例如，写出完成后要重用已释放的 SMEM，就需要该模式。但当前内核在 TMA 存储完成后直接退出，因此无需 TMA 存储 arrive/wait 模式，相应代码行已被注释。

## 深入理解 TMA 操作

|  | TMA 加载 | TMA 存储 |
|---|---|---|
| 方向 | GMEM -> SMEM | SMEM -> GMEM |
| 同步方法 | 内存屏障 | 代理栅栏 |
| 同步时机 | 操作之后 | 操作之前 |

TMA 操作概要。

到目前为止，我们已学会如何调用 TMA 加载和 TMA 存储。上表对比了这两种操作。要调用任意一种操作，都需要在主机代码中通过 `cute::make_tma_copy` 创建类似 `TiledCopy` 的对象，再将该对象传入内核函数，并在内核中通过 `cute::copy` 实际调用操作。本节将深入分析在内核函数中调用这些 `TiledCopy` 对象时真正发生的事，并由此讨论两种扩展：TMA 存储归约和 TMA 加载多播。

#### TMA 加载与存储的 PTX 指令

PTX（[Parallel Thread Execution](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)，并行线程执行）是 NVIDIA GPU 的低级中间语言。对当前讨论而言，PTX 的相关部分是一组指令，可通过由 `asm volatile` 关键字包装的代码块插入 CUDA 代码。特别是，当我们按前文方式调用 `cute::copy(tma_load, ...)` 或 `cute::copy(tma_store, ...)` 时，会调用特定 PTX 指令来执行这些操作。研究 PTX 可以帮助我们更好地理解 TMA 加载和 TMA 存储。

先从 TMA 加载开始。回顾一下，在主机代码中创建 `tma_load` 对象时，必须提供 GMEM 张量（包含待拷贝源数据）和 SMEM 布局（描述数据在每个 CTA 内的样子）。CuTe 使用该张量和布局，确定内核调用 `cute::copy(tma_load, ...)` 时应执行的[底层 PTX 指令](https://github.com/NVIDIA/cutlass/blob/637b15906358191cb4238af419d408a65819d7ec/include/cute/arch/copy_sm90_tma.hpp#L100-L106)。PTX 指令根据 GMEM 张量的 rank 选择；此处 rank 表示张量维数，不是线性代数中的矩阵秩/零度。本示例的 GMEM 张量 rank 为 2，因此执行以下 PTX 指令：

```
asm volatile (
  "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
  " [%0], [%1, {%3, %4}], [%2];"
  :
  : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
    "r"(crd0), "r"(crd1)
  : "memory");
```

从该 PTX 指令中可以看到许多熟悉概念。例如，`gmem_int_desc` 表示保存在 TMA 描述符中的坐标，`mbarrier::complete_tx::bytes` 和 `smem_int_mbar` 表示内存屏障。`tensor.2d` 表示正在拷贝 rank-2 张量，即二维矩阵。

事实上，不仅 TMA 加载，所有 TMA 操作都是对某些 `cp.async.bulk` 指令的封装。[NVIDIA PTX 文档用了完整一节](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk)讨论 `cp.async.bulk` 指令，尤其是它们的语法和操作数。如果要更彻底地研究 TMA 操作，建议阅读该节及其引用资料；它们覆盖的范围远超本文目标。这里只讨论通过这些 `cp.async.bulk` 指令暴露的两种 TMA 扩展。

#### TMA 存储归约

回顾一下，TMA 存储将数据从多个 CTA 的 SMEM 拷贝到 GMEM 张量中的相应矩阵块。可将 TMA 存储理解为以下 Python 伪代码展示的赋值操作：

```
for cta_idx in range(number_of_ctas):
  gmem_dst[cta_idx] = smem_src[cta_idx]
```

如果希望改为执行以下操作呢？

```
for cta_idx in range(number_of_ctas):
  gmem_dst[cta_idx] += smem_src[cta_idx]
  # 或者：
  gmem_dst[cta_idx] = max(gmem_dst[cta_idx], smem_src[cta_idx])
  # 或者：
  gmem_dst[cta_idx] = min(gmem_dst[cta_idx], smem_src[cta_idx])
```

这些操作——即求和归约、最大值归约和最小值归约——在张量程序中都很常见。特别是，求和归约是 Split-K GEMM 中不可避免的子例程，最大值和最小值归约则经常用于 attention。尽管这些操作看起来简单，在 CUDA 内核中实现它们并不直接。在阅读下一段前，可以先想一想：要达成这些目标，需要在 GMEM 和 SMEM 之间执行多少轮数据移动？

将 CTA SMEM 中的值“累加”到 GMEM 张量某个矩阵块的常规归约实现，包含一次 GMEM 读取、一个处理块和一次 GMEM 写入。首先将 GMEM 中的原始值加载到 CTA SMEM 或寄存器，然后执行归约，最后将结果写回。该过程很慢。

只需对 TMA 存储 `TiledCopy` 对象的构造函数稍作修改，就能把这个三步过程压缩为一条 PTX 指令：使用 [`cp.reduce.async.bulk`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-reduce-async-bulk) 而不是 `cp.async.bulk`。具体来说，只需对主机代码做以下一行变更：

```
// 原始实现：创建 TMA 存储对象
auto tma_store = make_tma_copy(SM90_TMA_STORE{}, gmem_tensor, smem_layout);
// 改为创建 TMA 求和归约对象
auto tma_reduce_sum = make_tma_copy(SM90_TMA_REDUCE_ADD{}, gmem_tensor, smem_layout);
```

随后改用 `tma_reduce_sum`，它在底层调用 `cp.reduce.async.bulk` 而不是 `cp.async.bulk`。

顺带一提，PTX 指令 `cp.reduce.async.bulk` 从 CUDA 12.0 发布起就已可用，但直到 [CUTLASS 3.5](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-reduce-async-bulk) 才通过 CUTLASS 和 CuTe 暴露。我们希望未来版本会暴露其他归约操作；即使没有，也可以比较容易地修改 TMA reduce-add 的 CuTe 代码，使其执行最大值和最小值归约，以及 `cp.reduce.async.bulk` 提供的其他按位归约：`and`、`or`、`xor`、`inc` 和 `dec`。

#### TMA 加载多播

上一节看到，通过研究 PTX 指令可以发现 TMA 归约操作；在某些应用中，它可以代替 TMA 存储。本节将研究 TMA 加载的多播扩展。

为了帮助理解，先查看 [`cp.async.bulk.tensor`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor) 的完整语法：

```
// 全局内存 -> shared::cluster：
cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism
{.multicast}{.level::cache_hint}
  [dstMem],
  [tensorMap, tensorCoords],
  [mbar]
  {, im2colOffsets}
  {, ctaMask}
  {, cache-policy}
.dst =                  { .shared::cluster }
.src =                  { .global }
.dim =                  { .1d, .2d, .3d, .4d, .5d }
.completion_mechanism = { .mbarrier::complete_tx::bytes }
.load_mode =            { .tile, .im2col }
.level::cache_hint =    { .L2::cache_hint }
.multicast =            { .multicast::cluster  }
```

同样，无需完全理解 PTX 指令语法，也能看到许多熟悉概念，例如 `.dim`、作为 `src` 的 `.global`，以及作为 `completion_mechanism` 的 `.mbarrier`。本节聚焦 `multicast` 操作数。

多播是指将 GMEM 张量中的一个矩阵块拷贝到多个 CTA 的多个 SMEM 位置。GEMM 内核（即矩阵乘法）经常出现这种情况：输入矩阵的一个列矩阵块会被多个行矩阵块所需，反之亦然。这种情况下，普通 TMA 加载仍然能正常工作——只需把同一 TMA 描述符提供给需要该数据的多个 CTA——但 `.multicast` 操作数允许我们保证 L2 缓存命中。

下面将上述 TMA 加载示例扩展为多播版本。首先需要将内核的 cluster 维度定义为非平凡形状，因为一组 CTA 协同参与 TMA 加载多播的前提是它们属于同一个[线程块集群](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-block-clusters)。为保持简单，只如下修改网格维度：

```
// 旧网格维度和隐式平凡 cluster 维度
dim3 grid_dims = dim3{M / CTA_M, N / CTA_N, 1};
dim3 cluster_dums = dim3{1, 1, 1};
// 新网格维度和 cluster 维度
dim3 grid_dims = dim3{M / CTA_M, N / CTA_N, 2};
dim3 cluster_dums = dim3{1, 1, 2};
```

请注意，使用 cluster 时，cluster 维度必须整除网格维度，否则内核无法启动。在新内核中，对同一 cluster 内的每对 CTA，都把同一 GMEM 矩阵块加载到两个 CTA 各自的 SMEM。当且仅当两个 CTA 的 `blockIdx.x` 和 `blockIdx.y` 都相同时，才会发生该情况。

首先，在主机代码中对 TMA 加载 `TiledCopy` 对象的定义做以下修改：

```
// 原始实现：创建 TMA 加载对象
auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, gmem_tensor, smem_layout);
// 新实现：为给定 cluster 大小创建 TMA 加载多播对象
auto tma_load = make_tma_copy(SM90_TMA_LOAD_MULTICAST{},
      gmem_tensor, smem_layout, cute::_2{});
```

最后一个参数（cluster 大小）写为 `_2{}`，使用 [CuTe 为此提供的整数类型](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/01_layout.md#integers)，将其作为编译期常量传入。在实际代码中，更符合惯用风格的方式是预先定义 `ClusterShape` 类型（本例为 `Shape<_1,_1,_2>`），然后对该参数写 `size<2>ClusterShape{}`。

随后如下修改内核代码：

```
template <typename T, int CTA_M, int CTA_N, class ClusterShape,
          class TmaLoad, class GmemTensor>
void tma_load_kernel(__grid_constant__ const TmaLoad tma_load,
                     GmemTensor gmem_tensor) {
  using namespace cute;
  uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
  constexpr uint32_t cluster_size = size<2>(ClusterShape{}));
  constexpr uint16_t tma_mcast_mask = (uint16_t(1) << cluster_size) - 1;
  constexpr int tma_transaction_bytes = CTA_M * CTA_N * sizeof(T);
  __shared__ T smem_data[CTA_M * CTA_N];
  __shared__ uint64_t tma_load_mbar;
  auto smem_layout = make_layout(make_shape(CTA_M, CTA_N), LayoutRight{});
  auto smem_tensor = make_tensor(make_smem_ptr(T), smem_layout);
  auto gmem_tensor_coord = tma_load.get_tma_tensor(shape(gmem_tensor));
  auto gmem_tensor_coord_cta = local_tile(
        gmem_tensor_coord,
        Tile<Int<CTA_M>, Int<CTA_N>>{},
        make_coord(blockIdx.x, blockIdx.y));
  if (threadIdx.x == 0) {
    initialize_barrier(tma_load_mbar, /* 到达计数 */ 1);
  }
  __syncthreads();
  cute::cluster_sync();
  cutlass::arch::fence_barrier_init();
  if (threadIdx.x == 0) {
    set_barrier_transaction_bytes(tma_load_mbar, tma_transaction_bytes);
    auto tma_load_per_cta = tma_load.get_slice(block_rank_in_cluster);
    copy(tma_load.with(tma_load_mbar, tma_mcast_mask),
         tma_load_per_cta.partition_S(gmem_tensor_coord_per_cta),
         tma_load_per_cta.partition_D(smem_tensor));
  }
  __syncthreads();
  wait_barrier(tma_load_mbar, /* 阶段 */ 0);
  // 执行到此行后，TMA 加载已完成
  cute::cluster_sync();
}
```

代码已突出标出相关变更。首先，现在需要跟踪 CTA 在其 cluster 内的内部索引，该索引通过 CuTe 方法 `block_rank_in_cluster()` 获取。该方法返回特殊寄存器 [`%cluster_ctarank`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-cluster-ctarank) 的值；在本示例中取值为 0 或 1。为简洁起见，下文将其称为 `ctaid`。代码有以下三处需要解释的修改：

1. 额外的 cluster 同步基础操作。
2. 在多播操作中使用 `uint16` 位掩码。
3. 使用 `ctaid` 确定 `TiledCopy` 对象的切片，该切片用于划分 GMEM 和 SMEM 张量。

对于（1），使用 CuTe 方法 `cluster_sync()`，该方法依次执行 cluster 屏障的 arrive 和 wait。代码在两处插入该方法：第 7–8 行将 `cluster_sync()` 与栅栏一起使用，确保 mbarrier 初始化对整个 cluster 可见；第 41 行再次使用 `cluster_sync()`，确保 cluster 中的一个 CTA 不会在另一个 CTA 仍等待多播加载完成时提前退出。一般情况下，内核会对加载到 SMEM 的数据执行计算，最后一个 `cluster_sync()` 会出现在内核代码的最末尾。

对于（2），向 `copy` 操作传入 `uint16` 位掩码，用于指定哪些 CTA 参与 TMA 多播加载。掩码中置 1 的位表示活跃 CTA；一个 cluster 最多可包含 16 个 CTA（[最大非可移植大小](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#thread-block-clusters)），每个位的位置对应 `ctaid`。因此，本示例将 `tma_mcast_mask` 设为 `0b11`，表示 cluster 中两个 CTA 都参与。

最后，对于（3），`ctaid` 用于指定给定 CTA 发起 TMA 多播加载时，对 GMEM 切片所使用的偏移。为了清晰解释，考虑以下示例：从 GMEM 将一个 16x16 整数矩阵块加载到某个 cluster 内两个 CTA 的 SMEM。该矩阵块按行主序以 0–255 递增初始化。假设我们错误地向两个 CTA 的 `tma_load.get_slice` 都传入 0，加载完成后，两个 CTA 的 SMEM 中都会得到以下内容：

```
  0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
 16   17   18   19   20   21   22   23   24   25   26   27   28   29   30   31
 32   33   34   35   36   37   38   39   40   41   42   43   44   45   46   47
 48   49   50   51   52   53   54   55   56   57   58   59   60   61   62   63
 64   65   66   67   68   69   70   71   72   73   74   75   76   77   78   79
 80   81   82   83   84   85   86   87   88   89   90   91   92   93   94   95
 96   97   98   99  100  101  102  103  104  105  106  107  108  109  110  111
112  113  114  115  116  117  118  119  120  121  122  123  124  125  126  127
  0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
  0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
  0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
  0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
  0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
  0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
  0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
  0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
```

相比之下，如果向两个 CTA 都传入 1，则两个 CTA 的 SMEM 中都会得到：

```
  0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
  0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
  0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
  0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
  0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
  0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
  0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
  0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
128  129  130  131  132  133  134  135  136  137  138  139  140  141  142  143
144  145  146  147  148  149  150  151  152  153  154  155  156  157  158  159
160  161  162  163  164  165  166  167  168  169  170  171  172  173  174  175
176  177  178  179  180  181  182  183  184  185  186  187  188  189  190  191
192  193  194  195  196  197  198  199  200  201  202  203  204  205  206  207
208  209  210  211  212  213  214  215  216  217  218  219  220  221  222  223
224  225  226  227  228  229  230  231  232  233  234  235  236  237  238  239
240  241  242  243  244  245  246  247  248  249  250  251  252  253  254  255
```

最后，无论是由 `ctaid` 1 传入 0、`ctaid` 0 传入 1，还是由 `ctaid` 0 传入 0、`ctaid` 1 传入 1，都会正确地将完整矩阵块加载到两个 CTA 的 SMEM。这些打印结果表明，cluster 中每个 CTA 发出的多播操作会将 GMEM 数据的一半加载到两个 CTA 各自的 SMEM，而 `TiledCopy` 切片决定对应的那一半。这与 PTX 文档对 `cp.async.bulk.tensor` 多播的描述一致：

源数据被多播到每个目标 CTA 共享内存中与 `dstMem` 相同的 CTA-relative offset。

从 `TiledCopy` 对象的角度看，它通常拥有一个将“线程-值”元组映射到矩阵块逻辑坐标的 `TiledLayout_TV` 布局。在切片时，CuTe 把 `ctaid` 视为线程索引。例如，在 16x16 示例中打印 `TiledCopy` 会得到：

```
TiledCopy
  Tiler_MN:       (_16,_16)
  TiledLayout_TV: (_2,((_16,_16))):(_8,((_16,_1)))
Copy_Atom
  ThrID:        _1:_0
  ValLayoutSrc: (_1,_256):(_0,_1)
  ValLayoutDst: (_1,_256):(_0,_1)
  ValLayoutRef: (_1,_256):(_0,_1)
  ValueType:    32b
```

该布局包含两个“线程”，对应 cluster 中的两个 CTA。对 `ctaid` 1，偏移位置由 `(16,16)` 矩阵块中的逻辑坐标 `(8,0)` 给出。

## 结论

本文通过几个简化示例，介绍了如何使用 CUTLASS 库提供的方法，在 CUDA 内核中利用 TMA 加载、存储、存储归约和加载多播，在 GMEM 与 SMEM 之间执行内存拷贝。

我们首先概述 TMA，并介绍用户如何在 GPU 内核中调用这些操作。随后深入底层 PTX 指令，以建立对 TMA 的更深理解。希望本文能够帮助想要理解 TMA、复习相关知识，或调试现有 TMA 项目的读者。

本文省略了一些重要主题，例如 TMA 支持的 swizzle 模式，以及 TMA 以交织格式将数据从 GMEM 拷贝到 SMEM、并置换连续维之外步长的能力。当 TMA 与 Hopper 架构同样新增的 Warpgroup 矩阵乘累加（WGMMA）指令结合使用，以便将张量数据加载成与 WGMMA 兼容的内存格式时，这些主题很重要。后续讨论基于 Hopper 的 GEMM 时，我们将解释这些要点。

最后，本文所讨论内核的完整示例可在 [Colfax Research GitHub 仓库](https://github.com/ColfaxResearch/cfx-article-src/tree/master/tma)中找到。

1. 这篇博客太棒了！
一个小问题：查看内核启动时，
TMA 加载启动
<<>>
和 TMA 存储启动
<<>>
为什么你们把 block_dimension（num_threads_per_cta）放在 grid_dimension 之前？我以为标准启动方式是 <<>>？

  1. 看起来格式化把内容删掉了……抱歉。
TMA 加载启动：1, dim3{M / CTA_M, N / CTA_N, 1}
TMA 存储启动：CTA_M, dim3{M / CTA_M, N / CTA_N, 1}
标准顺序是：grid_dim, block_dim
但这里是：block_dim, grid_dim
  2. 你说得对，谢谢指正——现在终于修复了（虽然已经非常晚）。
2. 感谢这篇精彩的文章！关于“上述示例假设 `m%CTA_M==0` 且 `n%CTA_N==0`。但就 TMA 加载而言，可以完全取消该假设”，TMA 存储呢？

  1. TMA 存储也同样成立。它会进行谓词判定，因此不会发生越界写入。

    1. 谢谢！
