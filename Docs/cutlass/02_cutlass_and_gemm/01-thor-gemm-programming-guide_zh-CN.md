**CUTLASS 3.x (1)：Thor GEMM 编程入口指南**

本文面向已了解 CUDA 线程组织和基本 C++ 模板语法的读者，介绍 CUTLASS 如何分解矩阵乘加、构造 Kernel，以及如何调用它。前五部分建立 Dense FP16 Tensor Core GEMM 的公共用法，第六部分讨论 Block-Scaled、Grouped GEMM、MoE 和 Attention，第七部分给出其他路径与后续阅读。

# 1. GEMM 问题与三条变化轴

## 从一项矩阵乘加出发

本文采用线性组合形式的 GEMM，先定义它的矩阵与标量：

- **乘法输入：** A 为 $M\times K$ 矩阵，B 为 $K\times N$ 矩阵，二者沿 K 维相乘并归约。
- **源矩阵与输出：** C 和 D 都为 $M\times N$ 矩阵。C 提供要叠加的值，D 保存最终结果。
- **标量系数：** $\alpha$ 缩放矩阵乘积，$\beta$ 缩放 C。

把这些量组合起来，得到：

$$D=\alpha AB+\beta C$$

M、N 决定输出矩阵的大小。固定一个输出位置 $(m,n)$，计算分为两步：先将 A 的第 m 行与 B 的第 n 列相乘，并沿长度为 K 的归约维求和：

$$\operatorname{acc}_{m,n}=\sum_{k=0}^{K-1}a_{m,k}b_{k,n}$$

再计算 $d_{m,n}=\alpha\operatorname{acc}_{m,n}+\beta c_{m,n}$。$\operatorname{acc}_{m,n}$ 是累加结果；所有输出位置的累加结果在数学上组成 $\operatorname{Acc}=AB$。程序可以逐个输出 Tile 生成和使用这些结果，无须先在全局内存中保存一张完整的 Acc 矩阵。

CUTLASS 把沿 K 的乘加组织称为 Mainloop，把消费累加结果、执行后处理并写回 D 的部分称为 Epilogue。后处理也可以包含 Bias、Activation 或输出类型转换，因此更一般地写为：

$$D=\operatorname{Epilogue}(\operatorname{Acc},C,\text{其他参数})$$

Dense GEMM 使用 A/B 中全部元素完成上述乘法。本篇以线性组合 Epilogue 为基础，讨论实际场景对这项计算的改变。

## 问题集合：一次处理多少组矩阵

Single GEMM 处理一项矩阵乘加。若一次调用包含多项独立问题，还需要说明它们的尺寸是否相同、每项问题的数据如何定位。

- **Batched GEMM** 的输入是 L 组尺寸相同的矩阵，输出是 L 组 D。各组共享 M/N/K，问题尺寸统一写为 $(M,N,K,L)$，相邻批次的数据可以通过固定的批次 Stride 定位。第 l 组计算 $D_l=\alpha A_lB_l+\beta C_l$，L 只表示批次数，不参与 K 维归约；本文的基本调用取 $L=1$。
- **Grouped GEMM** 的输入是多组分别描述的矩阵问题。第 g 组拥有自己的 $(M_g,N_g,K_g)$、矩阵地址和 Stride，计算 $D_g=\alpha_gA_gB_g+\beta_gC_g$。各组的尺寸可以不同，因此输入描述和输出范围都要按组确定。
- **MoE 的 Expert 计算** 是这种不均衡问题集合的一个来源。路由阶段为 Expert e 选出 Token 索引 $I_e$，形成 $X_e=X[I_e,:]$，再计算该 Expert 的矩阵乘法 $Y_e=X_eW_e$。不同 Expert 接收的 Token 数 $T_e=|I_e|$ 不同，因而产生不同尺寸的 GEMM。Expert 计算后，外层算法按路由关系恢复 Token 顺序并合并结果。

这一变化轴决定一次调用包含哪些矩阵问题。每组内部仍然具有自己的输出空间和 K 维归约。

## 数值表示：存储值怎样参与乘法

矩阵尺寸相同，并不意味着存储内容的解释相同。低精度、压缩和复数表示都会改变 Mainloop 需要读取的信息。

- **Mixed-input GEMM** 的 A/B 使用不同类型或位宽，需要把输入转换为所选乘法操作可以消费的形式。
- **Block-Scaled GEMM** 将低精度量化值（Payload）与缩放因子（Scale）分开存储。Payload 是原始元素经过缩放、舍入并转换为低精度格式后保存的数据本体，不包含 Scale。缩放因子的名称取决于它覆盖的元素范围：

  - **Tensorwise scaling（按张量缩放）**：整个张量中的元素共享一个 Tensor Scale。
  - **Blockwise scaling（按块缩放）**：张量分成多个局部块，每个块内的元素共享一个 Block Scale，不同块可以使用不同的值。

  Block-Scaled 描述按块缩放的表示，还可以额外带有 Tensor Scale。本文采用的 NVFP4 同时使用这两级缩放；只有整张量 Scale 的表示则属于 Tensorwise scaling。恢复后的乘数可以写为：

  $$a_{m,k}\approx S_A[\phi_A(m,k)]\widehat a_{m,k},\qquad{}
  b_{k,n}\approx S_B[\phi_B(k,n)]\widehat b_{k,n}$$

  其中 $\widehat A/\widehat B$ 是低精度量化值，$\phi_A/\phi_B$ 给出各元素使用的 Scale。这里的 $S_A/S_B$ 表示重建乘数所需的完整缩放；具体格式还可以把它分成张量级和块级因子。后文的 SFA/SFB 是保存块级 Scale 的张量，称为 Scale Tensor；这不等于整张输入只有一个 Tensor Scale。第六部分以 NVFP4 说明这些量的格式、生成步骤和使用位置。
- **Sparse GEMM** 的逻辑输入仍是一张矩阵，物理存储则用压缩值和位置 Metadata 共同描述它。Sparse MMA 可以直接消费这些信息，无须先在内存中展开完整的 Dense 输入。
- **Complex GEMM** 的元素包含实部与虚部，需要按复数乘法关系组合各分量的乘积。其存储方式与其他数值路径的实现入口见第七部分。

数值表示决定元素类型、附加数据和部分积的计算方式。Single、Batched、Grouped 都可以采用这些表示，例如一组 Expert GEMM 可以同时采用 Block Scaling。

## 阶段依赖：多项矩阵计算怎样连接

前面的各组 GEMM 可以独立计算。复合算法还会让后一个阶段消费前一个阶段产生的中间结果。Attention 是典型例子：

$$S=QK^T/\sqrt d+\operatorname{Mask},\qquad{}
P=\operatorname{Softmax}(S),\qquad O=PV$$

第二次矩阵乘法需要第一阶段分数经过 Softmax 后的结果。因此完整算法还要描述 Mask、行归约，以及中间结果在两个乘法之间怎样传递。Distributed GEMM 则把同一项计算分到多个设备，在局部矩阵计算之外加入通信和结果合并。

问题集合、数值表示和阶段依赖共同确定要计算的内容。CUTLASS 用不同层次的组件组织这些计算，下面从一项基本 GEMM 说明。

# 2. CUTLASS 五层模型

CUTLASS 接收矩阵尺寸、地址和标量，计算结果写入 D。要完成这次调用，需要先把完整问题分给参与计算的 CTA，再组织局部数据的搬运与乘加，最后落实到基本架构操作。[CUTLASS 的五层 GEMM 模型](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/media/docs/cpp/gemm_api_3x.md)按作用范围描述这项分工：

![图 1. CUTLASS GEMM 的五层作用范围：Atom、Tiled、Collective、Kernel 与 Device](Imgaes/thor-gemm-programming-guide/cutlass-gemm-hierarchy.webp)

下面按 Device 到 Atom 的方向阅读，即从一次调用逐步进入局部计算。这些层次描述组件之间的包含与组合关系，并不是五个依次执行的计算阶段。

- **Device 层：面向一次具体 GEMM 调用的有状态主机接口。** 它接收本次问题尺寸、矩阵地址、Stride 和标量，将参数转换为设备执行需要的内部形式，并保存初始化后的调用状态。相同的 Kernel 类型可以用于多次尺寸和地址不同的调用，只要参数满足该类型的要求。

  - [`cutlass::gemm::device::GemmUniversalAdapter<>`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/device/gemm_universal_adapter.h)：把 Kernel 包装成主机端 GEMM 句柄，提供 `can_implement`、`get_workspace_size`、`initialize` 和 `run` 等接口，分别用于检查参数、查询辅助存储需求、初始化和启动。矩阵与 Workspace 的分配及生命周期仍由调用者负责。

- **Kernel 层：面向完整 GEMM 问题空间的设备端内核。** 它组合 Mainloop 与 Epilogue，并通过 Tile Scheduler 确定各 CTA 或 CTA Cluster 当前承担的工作。Kernel 将工作坐标交给 Collective，连接局部计算与输出；当前工作完成后是否继续领取工作，也由这一层的实现决定。

  - [`cutlass::gemm::kernel::GemmUniversal<>`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/kernel/gemm_universal_decl.h#L50-L57)：接收 ProblemShape 类型、Mainloop Collective、Epilogue Collective 和可选的 Tile Scheduler 类型，组合成完整 Kernel。这里的 ProblemShape 类型规定问题如何描述，具体尺寸由 Device 层在调用时传入。

- **Collective 层：协作线程共同完成局部计算的组件。** 它把 TiledMMA、TiledCopy、共享内存布局、流水级和同步机制组织在一起，规定哪些参与者负责加载或计算、数据何时可读，以及缓冲何时可复用。Mainloop 与 Epilogue 是这一层的两类计算组件，前者产生累加结果，后者消费结果并写回输出。

  - [`cutlass::gemm::collective::CollectiveMma<>`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/collective/collective_mma_decl.hpp)：Mainloop Collective 的接口，沿分配到的 K 范围组织 A/B Tile 的加载、同步和 MMA 累加，形成当前输出区域的 Accumulator。
  - [`cutlass::epilogue::collective::CollectiveEpilogue<>`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/epilogue/collective/collective_epilogue.hpp)：本文所用 Epilogue 实现的接口，读取累加结果，执行线性组合、可选融合操作和输出类型转换，再写回 D。

  使用 Builder 构造 Kernel 时，调用者通常不直接填写上述实现的全部模板参数，而是从两个编译期入口取得匹配的组件类型：

  - [`cutlass::gemm::collective::CollectiveBuilder<>`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/collective/collective_builder_decl.hpp#L77-L96)：接收架构、输入类型、布局、Alignment、Tile、Cluster、Stage 和 Mainloop Schedule 等配置，通过 `::CollectiveOp` 给出具体 Mainloop Collective 类型。
  - [`cutlass::epilogue::collective::CollectiveBuilder<>`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/epilogue/collective/collective_builder.hpp#L52-L73)：接收累加与后处理计算类型、C/D 类型和布局、分块、Epilogue Schedule 及融合操作等配置，通过 `::CollectiveOp` 给出具体 Epilogue Collective 类型。

  Builder 负责在编译期选择并生成类型，运行时执行计算的是生成的 Collective。Builder 因而是这一层的构造入口，不是额外的执行层。相应实现已经包含匹配的 Tiled 和 Atom，调用者无须逐层手写所有组件。

- **Tiled MMA/Copy 层：基本操作与参与者—数据布局组合成的空间计算或搬运对象。** 它在 Atom 的操作约束下描述局部数据怎样分区，以及各参与者对应哪些元素。TiledMMA 与 TiledCopy 分别描述乘加和搬运的空间分工，二者由 Collective 放入加载、计算和同步的执行顺序。

  - [`cute::TiledMMA<>`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cute/atom/mma_atom.hpp#L199-L240)：由 MMA Atom、Atom 的排列和 Tile 的置换等配置组成，建立参与者与 A/B 操作数、累加结果之间的映射，供局部矩阵乘法的分区与展开使用。
  - [`cute::TiledCopy<>`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cute/atom/copy_atom.hpp#L176-L223)：由 Copy Atom 和参与者—数据布局组成，规定源张量与目标张量如何分区，以及每个参与者负责搬运哪些元素。

- **Atom 层：基本架构操作及其操作数约束的封装。** 架构操作提供计算或搬运能力，Traits 描述操作需要的 Shape、数据类型和参与者—数据布局，Atom 将它们包装为可供 Tiled 层组合的接口。它只描述基本操作，不承担完整问题的分配或整个 K 维流水线。

  - [`cute::MMA_Atom<>`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cute/atom/mma_atom.hpp)：依据 MMA 操作及其 [`MMA_Traits`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cute/atom/mma_traits.hpp)，提供基本乘加的 `call`、Fragment 构造和操作数布局等接口。
  - [`cute::Copy_Atom<>`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cute/atom/copy_atom.hpp)：依据搬运操作及其 [`Copy_Traits`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cute/atom/copy_traits.hpp)，提供基本搬运的调用与源、目标布局接口。TMA、cp.async 等操作各有相应约束，不能仅因使用统一接口就任意互换。

  这里的 **Fragment** 是为特定操作组织的局部操作数或结果表示，其类型与组织方式需要符合所用 Atom 和 Traits 的要求。它不是统一的存储空间名称，也不总是线程寄存器中的数值数组；不同操作可以使用不同的表示。[Fragment 构造接口](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cute/atom/mma_atom.hpp#L123-L179)据此生成或沿用相应的局部对象，具体构造留到第三篇。

上述顺序用于从一次调用向内理解计算。实际构造类型时，先由两个 Builder 取得 Collective，再组合 `GemmUniversal`，最后用 `GemmUniversalAdapter` 包装主机接口。第四部分列出配置对应的模板参数，第五部分给出完整构造与调用过程。Traits、Fragment 和 Layout 的内部构造留到第三篇；下面先沿一个输出 Tile 说明这些层次怎样共同完成 GEMM。

# 3. 一个 Work Tile 怎样完成 GEMM

一次具体调用先给出完整问题的尺寸和数据，再由 Kernel 分配输出 Tile。Device 接收本次调用的运行时 `ProblemShape=(M,N,K,L)`、A/B/C/D 地址与 Stride、Epilogue 参数和调度参数。其中 M、N、K 定义单个矩阵乘法，标准 Single/Strided-Batched ProblemShape 使用 L 表示批次维，本文后续的 Single GEMM 基线取 $L=1$。

与运行时 ProblemShape 不同，`CollectiveBuilder` 的 `TileShape_MNK=(T_M,T_N,T_K)` 是编译期参数，表示 M×N×K Collective Tile。$T_M$ 和 $T_N$ 定义一次局部矩阵乘法覆盖的输出行、列范围，$T_K$ 定义 Collective Mainloop 一次 `k_tile` 迭代消费的 K 维宽度。`TileShape_MNK` 回答“这个 Kernel 类型按多大的局部 Shape 计算”，并不描述某次运行实际领取了哪一个 Tile。

`ClusterShape_MNK` 描述一个 CTA Cluster 在相应方向上包含多少个 CTA，分量以 CTA 个数为单位，而不是矩阵元素数。Cluster 内的 CTA 可以利用所选实现支持的共享数据搬运与同步机制。它与 Tile Shape 分别描述 CTA 的组织和局部数据的尺寸；一个 Cluster 可以包含多组局部计算，不能仅凭 Cluster 中的 CTA 总数判断一次 MMA 由几个 CTA 协同完成。

Kernel Tile Scheduler 结合 ProblemShape、CTA Tile 和 Cluster Shape 取得当前工作，用 `WorkTileInfo` 保存工作坐标，再将其转换为 CTA 坐标。默认 Full-K 路径为该工作保留完整的 K 迭代范围。

CTA 的工作坐标与完整 Collective Tile 的范围需要区分。1SM 路径由一个 CTA 完成该 Collective Tile；2SM 路径由一对 CTA 协同完成。源码用 [`CtaShape_MNK = shape_div(TileShape, AtomThrShapeMNK)`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp#L105-L122)表示每个 CTA 对应的部分，Scheduler 再使用这一级尺寸组织工作。因此，在 2SM 情形下，单个 CTA 的工作坐标不能直接当成整个 $T_M\times T_N$ 区域。

下面对完整的 Collective 输出区域推导数值计算。设第 $(p,q)$ 个区域覆盖行坐标集合 $I_p$ 和列坐标集合 $J_q$，完整区域满足 $|I_p|=T_M$、$|J_q|=T_N$，边界处只取有效坐标。对应的输出为：

$$D[I_p,J_q],\qquad D[I_p,J_q]:|I_p|\times|J_q|$$

设 Mainloop 第 $r$ 次迭代消费的 K Tile 为 $K_r$；除尾部 K Tile 外，通常有 $|K_r|=T_K$。这一轮对当前输出区域产生的局部贡献为：

$$P_{p,q}^{(r)}=A[I_p,K_r]B[K_r,J_q],\qquad P_{p,q}^{(r)}:|I_p|\times|J_q|$$

![图 2. CUTLASS GEMM 的 M/N 输出分块与 K Tile 迭代](Imgaes/thor-gemm-programming-guide/cutlass-threadblock-gemm.png)

*图 2. M/N 输出分块与 K Tile 迭代。图中的 $\mathrm{Block}_{m,n}$ 表示一个输出区域，A/B 的局部块沿 K 迭代并为它形成部分积。图片使用经典 Thread Block 术语；本文 2SM 路径的完整 Collective Tile 由一对 CTA 协同计算，每个 CTA 的工作坐标按前述 CTA Tile 解释。图片来源：[NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/media/images/cutlass-threadblock-gemm.png)。*

局部贡献直接累加到当前输出 Tile 的 Accumulator。记本次归约需要的 K Tile 数为 $R=\lceil K/T_K\rceil$，Mainloop 完成全部迭代后得到：

$$\operatorname{Acc}_{p,q}=\sum_{r=0}^{R-1}P_{p,q}^{(r)}$$

Mainloop 得到当前区域的 $\operatorname{Acc}_{p,q}$ 后，Epilogue 按参与 CTA 的分工读取累加结果、对应的 C Tile 以及标量或融合参数，生成对应的 D Tile：

$$D[I_p,J_q]=\operatorname{Epilogue}\!\left(\operatorname{Acc}_{p,q},C[I_p,J_q],\text{其他参数}\right)$$

当前的 $I_p\times J_q\times K_r$ 是 Mainloop 一次迭代处理的 M×N×K Collective Tile。`TiledMMA` 在这个 Tile 内建立参与者—数据映射，并沿自身的 M、N、K 模式静态展开，最后调用底层 MMA Atom。Scheduler 先定位 CTA 的工作，参与计算的 CTA 再共同完成当前 Collective Tile 的归约和输出；K 范围拆分的变体放在第七部分。

# 4. Scenario 怎样映射到模板参数

## 计算引擎决定基本乘加的形式

前一章说明了输出 Tile 和 K 维归约怎样组织。基本乘加可以由 CUDA Core 的线程级 FMA 完成，也可以由 Tensor Core 的矩阵 MMA 完成。

两类路径可以分别从基本操作、参与者和累加存储来理解：

- **SIMT：线程级乘加。** 输出元素或输出微块分给不同线程，各线程读取操作数，在自己的寄存器中沿 K 维反复更新累加值。CUTLASS 用 [`OpClassSimt`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/arch/mma.h#L115-L117)标记这类计算，基本更新为：

  $$\operatorname{acc}\leftarrow\operatorname{fma}(a,b,\operatorname{acc})$$

- **Tensor Core：矩阵级乘加。** MMA 按规定的矩阵 Shape 接收操作数并更新一组累加结果，参与计算的线程或 CTA 要按该操作要求组织数据。普通 Tensor Core 路径使用 [`OpClassTensorOp`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/arch/mma.h#L121-L122)。本文的 Dense Blackwell SS 路径通过 TMA 将 A/B Tile 从全局内存搬入共享内存，再由 TCGen05 MMA 读取共享内存描述符，累加结果保存在 Tensor Memory（TMEM）中。

![图 3. SIMT 与 Tensor Core 的空间分块和数据通路对比](Imgaes/thor-gemm-programming-guide/simt-tensor-core-tiling.jpg)

两类实现表达相同的矩阵乘加关系，输入格式、参与范围、累加存储和浮点运算顺序可以不同。它们都能通过 Collective、Kernel 和 Device 接口组合，但相同的高层接口不意味着使用相同的指令和数据通路。具体 FP32 SIMT 类型放在第七部分对照。

## Scenario 到 CUTLASS 模板参数的映射

本文用 Scenario 整理五组配置选择：问题拓扑 `Topology`、计算引擎 `ComputeEngine`、数值表示 `Numerics`、融合操作 `Fusion` 和执行组织 `Scheduling`。下表给出这些选择在 CUTLASS 中对应的模板参数。

`Scenario = (Topology, ComputeEngine, Numerics, Fusion, Scheduling)`

下表中的名称直接取自固定源码的 [Mainloop `CollectiveBuilder`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/collective/collective_builder_decl.hpp#L77-L96)、[Epilogue `CollectiveBuilder`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/epilogue/collective/collective_builder.hpp#L52-L73) 和 [`GemmUniversal`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/kernel/gemm_universal_decl.h#L50-L57)：

| Scenario 维度 | CUTLASS 层级与模板 | 对应的模板参数名 |
|-|-|-|
| `Topology` | Kernel：`cutlass::gemm::kernel::GemmUniversal` | `ProblemShapeOrThreadblockMma_`；在 CUTLASS 3.x 语义中即 `ProblemShape_` |
| `ComputeEngine` | Mainloop：`cutlass::gemm::collective::CollectiveBuilder` | `ArchTag`、`OpClass` |
| `ComputeEngine` | Epilogue：`cutlass::epilogue::collective::CollectiveBuilder` | `ArchTag`、`OpClass` |
| `Numerics` | Mainloop：`cutlass::gemm::collective::CollectiveBuilder` | `ElementA`、`GmemLayoutA`、`AlignmentA`、`ElementB`、`GmemLayoutB`、`AlignmentB`、`ElementAccumulator` |
| `Numerics` | Epilogue：`cutlass::epilogue::collective::CollectiveBuilder` | `ElementAccumulator`、`ElementCompute`、`ElementC`、`GmemLayoutTagC`、`AlignmentC`、`ElementD`、`GmemLayoutTagD`、`AlignmentD` |
| `Fusion` | Epilogue：`cutlass::epilogue::collective::CollectiveBuilder` | `FusionOpOrCallbacks` |
| `Scheduling` | Mainloop：`cutlass::gemm::collective::CollectiveBuilder` | `TileShape_MNK`、`ClusterShape_MNK`、`StageCountType`、`KernelScheduleType` |
| `Scheduling` | Epilogue：`cutlass::epilogue::collective::CollectiveBuilder` | `TileShape_MNK`、`ClusterShape_MNK`、`EpilogueTileType`、`EpilogueScheduleType` |
| `Scheduling` | Kernel：`cutlass::gemm::kernel::GemmUniversal` | `TileScheduler_` |

`CollectiveMainloopOrEpilogue_` 和 `CollectiveEpilogueOrThreadblockSwizzle_` 是 `GemmUniversal` 接收的两个已生成 Collective 类型，不是新的 Scenario 维度；三个模板中的 `Enable` 用于偏特化匹配，也不作为用户定义 Scenario 的参数。运行时 `Arguments` 只为已经确定的这些类型补充具体值，不属于这张编译期映射表。

这里的 Schedule 是选择执行组织方式的类型标签。Mainloop 的 `KernelScheduleType` 选择加载、乘加和协作方式，名称中的 Kernel 不表示它负责全局工作分配；`EpilogueScheduleType` 选择累加结果的读取、后处理与写回方式。Kernel 层的 `TileScheduler_` 才决定领取哪一项工作，以及相应的输出坐标或 K 范围。选定了当前工作之后，两个 Collective 仍按各自的 Schedule 完成局部计算。

第5部分用一组具体参数构造并调用 Dense Kernel。第6部分说明其他场景需要修改哪些类型和运行时输入。

# 5. Dense FP16 Tensor Core 完整基线

本章以一次 $D=\alpha AB+\beta C$ 调用说明五层模型的用法。计算组件在编译期确定，问题尺寸和矩阵数据在运行时提供。完整程序见 [dense_baseline.cu](exemples/dense_baseline.cu)，文件中的分段注释与本章小节一一对应；下面的代码是同一程序的分步讲解，不是各自独立的程序。类型声明位于函数外，从 Shape/Stride 开始的运行时代码位于 `main()` 中。`check` 的定义为便于讲解放在参考比较之后展示，完整文件中则位于 `main()` 之前。

## 固定输入类型与局部计算尺寸

本文按 CUTLASS 固定版本 [`8f50b052e1099fb982392a622caab69b97b63128`](https://github.com/NVIDIA/cutlass/tree/8f50b052e1099fb982392a622caab69b97b63128)组织类型和接口。Thor 的二进制目标为 `compute_110a/sm_110a`，C++ Builder 使用 [`cutlass::arch::Sm100`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/arch/arch.h#L100-L102)选择相应 Blackwell 实现族。

A/B 使用 FP16，累加和输出使用 FP32，四个矩阵都按 RowMajor 存储。局部计算 Tile 取 `(256,128,64)`，Cluster 取 `(2,2,1)` 用于讲解。

```cpp
// C++ 使用 Sm100 配方；Thor 二进制仍编译为 sm_110a。
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = float;
using ElementD = float;
using ElementAccumulator = float;  // K 维乘加的累加类型。
using ElementCompute = float;      // Epilogue 的计算类型。
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
// Alignment 以元素为单位，此处四个矩阵均对应 16 字节。
constexpr int AlignmentA = 8, AlignmentB = 8, AlignmentC = 4, AlignmentD = 4;
// Collective 的局部 (M,N,K)，不是完整问题尺寸。
using MmaTileShape = cute::Shape<
    cute::_256,  // M 维的局部 Tile 长度。
    cute::_128,  // N 维的局部 Tile 长度。
    cute::_64  // K 维的局部 Tile 长度。
>;
using ClusterShape = cute::Shape<
    cute::_2,  // M 方向的 CTA 个数。
    cute::_2,  // N 方向的 CTA 个数。
    cute::_1  // K 方向的 CTA 个数。
>;
```


`ElementA/B/C/D` 指定矩阵在内存中的元素类型。`ElementAccumulator` 指定 K 维乘加所形成的累加值类型，`ElementCompute` 指定 Epilogue 执行线性组合时使用的计算类型。本例后二者均为 FP32，但它们作用于不同阶段，不能从输出类型推定累加类型。

`LayoutA/B/C/D` 是传给 Builder 的布局标签。这里的 `RowMajor` 表示矩阵按行存储，它尚未给出本次矩阵的行长度或行间距；这些具体值要在运行时通过 Shape 与 Stride 补齐。

`AlignmentA/B` 声明输入访问所能满足的对齐粒度，以元素为单位，8 个 FP16 对应 16 字节；`AlignmentC/D` 的 4 个 FP32 也对应 16 字节。Builder 据此选择搬运实现，声明这个值不会自动对齐或填充矩阵。实际基址、连续维长度与行或批次的字节步长仍要满足所选实现的约束，后文在构造 Stride 时说明。

`MmaTileShape=(256,128,64)` 表示完整 Collective 输出区域为 256 行、128 列，每次 K Tile 迭代消费 64 个归约位置。`ClusterShape=(2,2,1)` 则组织四个 CTA；每个 CTA 对应的数据范围仍按第三部分的 1SM／2SM 规则确定。运行时 M/N/K 可以变化，这两个静态 Shape 则属于已经生成的 Kernel 类型。

## 先构造 Epilogue，再确定 Mainloop 的存储预算

Epilogue 的线性组合操作读取 FP32 Accumulator 和 C，计算 alpha/beta 组合并生成 FP32 D。Epilogue Builder 同时确定后处理的分块、搬运和共享内存需求：

```cpp
// 线性组合：D = alpha * Acc + beta * C。
using EpilogueOperation = cutlass::epilogue::fusion::LinearCombination<
    ElementD,  // ElementOutput_：输出 D 的类型。
    ElementCompute,  // ElementCompute_：线性组合的计算类型。
    ElementC,  // ElementSource_：源矩阵 C 的类型。
    float,  // ElementScalar_：alpha/beta 的类型。
    cutlass::FloatRoundStyle::round_to_nearest  // RoundStyle_：输出转换的舍入规则。
>;
// 先确定 Epilogue，后面据此预留共享内存。
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,  // ArchTag：架构实现族。
    OperatorClass,  // OpClass：计算引擎类别。
    MmaTileShape,  // TileShape_MNK：与 Mainloop 匹配的 Collective Tile。
    ClusterShape,  // ClusterShape_MNK：与 Mainloop 匹配的 CTA Cluster。
    cutlass::epilogue::collective::EpilogueTileAuto,  // EpilogueTileType：Epilogue 子分块；Auto 在编译期选择。
    ElementAccumulator,  // ElementAccumulator：Mainloop 提供的累加类型。
    ElementCompute,  // ElementCompute：后处理使用的计算类型。
    ElementC,  // ElementC：源矩阵 C 的存储类型。
    LayoutC,  // GmemLayoutTagC：C 的全局内存布局标签。
    AlignmentC,  // AlignmentC：C 的访问对齐，以元素数计。
    ElementD,  // ElementD：输出矩阵 D 的存储类型。
    LayoutD,  // GmemLayoutTagD：D 的全局内存布局标签。
    AlignmentD,  // AlignmentD：D 的访问对齐，以元素数计。
    cutlass::epilogue::collective::EpilogueScheduleAuto,  // EpilogueScheduleType：结果读取、后处理与写回方式。
    EpilogueOperation  // FusionOpOrCallbacks：融合操作或回调类型。
>::CollectiveOp;
```

Mainloop 沿 K 反复读取 A/B Tile。若每一轮都等数据搬运结束后才开始计算，再等计算结束后才加载下一轮，搬运与乘加就只能串行进行。流水线为不同迭代保留若干组可复用的缓冲，使后续数据的加载能够与当前数据的计算重叠。

在本节的 Dense TMA Mainloop 中，一个 Stage 对应一组 A/B Tile 缓冲及其同步状态。加载方等待缓冲可用后写入数据，计算方等待数据就绪后使用它，确认消费完成后才允许复用。`StageCountType` 控制这组流水线的缓冲级数；它不等于整项问题的 K Tile 数，有限个 Stage 会在 K 维迭代中循环使用。更多 Stage 会占用更多共享内存，也不保证执行一定更快。

Mainloop 与 Epilogue 共用 CTA 的共享内存预算。因此先获得 `CollectiveEpilogue::SharedStorage` 的大小，再用 `StageCountAutoCarveout` 为 Mainloop 推导可用的流水级数：

```cpp
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,  // ArchTag：架构实现族。
    OperatorClass,  // OpClass：计算引擎类别。
    ElementA,  // ElementA：A 的输入存储类型。
    LayoutA,  // GmemLayoutA：A 的全局内存布局标签。
    AlignmentA,  // AlignmentA：A 的访问对齐，以元素数计。
    ElementB,  // ElementB：B 的输入存储类型。
    LayoutB,  // GmemLayoutB：B 的全局内存布局标签。
    AlignmentB,  // AlignmentB：B 的访问对齐，以元素数计。
    ElementAccumulator,  // ElementAccumulator：K 维归约的累加类型。
    MmaTileShape,  // TileShape_MNK：完整 Collective 的局部 M/N/K 尺寸。
    ClusterShape,  // ClusterShape_MNK：CTA Cluster 的形状，以 CTA 个数计。
    // StageCountType：Mainloop 流水级数或自动推导策略。
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))  // carveout_bytes：为 Epilogue 预留的共享内存字节数。
    >,
    cutlass::gemm::collective::KernelScheduleAuto  // KernelScheduleType：Mainloop 的加载、乘加与协作方式。
>::CollectiveOp;  // 编译期选择实现。
```


`StageCountAutoCarveout` 的参数以字节为单位，表示需要从 Mainloop 的共享内存预算中预留的空间，并非直接填写 Stage 数。所选 Builder 结合每级 A/B 缓冲和同步存储的大小，扣除相应预留后计算可容纳的级数，见 [Dense Builder 的 Stage 推导](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl#L77-L102)。

Epilogue 的存储需求决定了这两个类型的声明顺序。运行时仍然由 Mainloop 产生累加结果，再由 Epilogue 消费。`KernelScheduleAuto` 与 `EpilogueScheduleAuto` 根据类型、Tile 和架构选择实现，自动 Stage 则计算存储预算。这些选择都在编译期完成，Auto 不表示运行时测量性能后再选择最快配置。

两个 Collective 确定后，将它们组合成完整 Kernel，再包装为主机调用接口：

```cpp
// 四个 int 为运行时 M/N/K/L 留出位置。
// TileScheduler_ 未显式指定，使用该 Kernel 的默认选择。
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<
        int,  // M 维的运行时整数类型。
        int,  // N 维的运行时整数类型。
        int,  // K 维的运行时整数类型。
        int  // L 维的运行时整数类型。
    >,  // ProblemShapeOrThreadblockMma_：3.x 的问题描述类型。
    CollectiveMainloop,  // CollectiveMainloopOrEpilogue_：3.x 的 Mainloop 类型。
    CollectiveEpilogue  // CollectiveEpilogueOrThreadblockSwizzle_：3.x 的 Epilogue 类型。
>;
// 将设备 Kernel 包装为主机调用句柄。
using GemmHandle = cutlass::gemm::device::GemmUniversalAdapter<
    GemmKernel  // GemmKernel_：被包装的设备端 Kernel 类型。
>;
```


`cute::Shape<int,int,int,int>` 为运行时 M/N/K/L 留出位置。这里生成了可调用的 Kernel 类型，调用时还要填入问题的具体值。[Example 71](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/examples/71_blackwell_gemm_with_collective_builder/71_blackwell_gemm_with_collective_builder.cu#L155-L250)提供相同的类型组合结构，但其默认 B/C/D 为 ColumnMajor，C/D 为 FP16。本文独立固定了全 RowMajor 和 FP32 C/D；对照示例时，应区分 API 组合方式与实例的具体配置。

## 用 Shape 与 Stride 描述本次矩阵

Shape 给出各坐标维度的长度，Stride 给出某一坐标增加一时，线性元素偏移增加多少。对于本节的普通稠密矩阵，将各坐标与对应 Stride 相乘后相加，就得到相对基址的元素偏移，再由元素类型确定字节地址。

CuTe 的 `Layout` 对象把 Shape 与 Stride 组合为坐标到线性索引的映射；它可以描述矩阵存储，也可以描述线程与数据的对应关系。前面的 `cutlass::layout::RowMajor` 则是 Builder 的布局标签，用于选择相应的 Stride 类型和实现。标签确定存储方向，运行时 Stride 补充具体间距，二者不能互相替代。相关定义见 [CuTe Layout](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/media/docs/cpp/cute/01_layout.md)。

取一项 `M=256,N=256,K=128,L=1` 的问题。Stride 类型从 Kernel 取得，具体步长再由问题尺寸生成：

```cpp
// 从当前 Kernel 取得匹配的 Stride 类型。
using StrideA = typename GemmKernel::StrideA;
using StrideB = typename GemmKernel::StrideB;
using StrideC = typename GemmKernel::StrideC;
using StrideD = typename GemmKernel::StrideD;

int M = 256, N = 256, K = 128, L = 1;
auto problem_shape = cute::make_shape(
    M,  // 输出行数。
    N,  // 输出列数。
    K,  // 归约长度。
    L   // 批次数，此处为单问题。
);
// 按紧密存储生成步长；带行尾填充时应改用实际步长。
auto stride_A = cutlass::make_cute_packed_stride(
    StrideA{},  // 当前 Kernel 要求的 A 步长类型。
    cute::make_shape(M, K, L)  // A 的 (行数, 归约长度, 批次数)。
);
// B 的坐标顺序是 (n,k,l)，数学形状仍为 K×N。
auto stride_B = cutlass::make_cute_packed_stride(
    StrideB{},  // 当前 Kernel 要求的 B 步长类型。
    cute::make_shape(N, K, L)  // B 的 (输出列数, 归约长度, 批次数)。
);
auto stride_C = cutlass::make_cute_packed_stride(
    StrideC{},  // 当前 Kernel 要求的 C 步长类型。
    cute::make_shape(M, N, L)  // C 的 (行数, 列数, 批次数)。
);
auto stride_D = cutlass::make_cute_packed_stride(
    StrideD{},  // 当前 Kernel 要求的 D 步长类型。
    cute::make_shape(M, N, L)  // D 的 (行数, 列数, 批次数)。
);
```


A 使用 `(m,k,l)` 坐标，B 使用 `(n,k,l)` 坐标，C/D 使用 `(m,n,l)` 坐标。对当前全 RowMajor、L=1 的实例：

- A 的 Stride 为 `(K,1,0)`，元素偏移是 `m*K+k`。
- B 的 Stride 为 `(1,N,0)`，元素偏移是 `n+k*N`。
- C/D 的 Stride 为 `(N,1,0)`，元素偏移是 `m*N+n`。

因此 B 在数学上仍是 $K\times N$ 的 RowMajor 矩阵。构造 Stride 时使用 `(N,K,L)`，只是 CUTLASS 为 B 选择的坐标顺序；对应的物理地址仍为 `k*N+n`。`L=1` 时批次 Stride 为零；同构多批次的步长则由 [`make_cute_packed_stride`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/tools/util/include/cutlass/util/packed_stride.hpp)按各矩阵尺寸生成。

换成自己的矩阵时，需要分别确定有效尺寸、实际存储和访问条件：

- **有效尺寸决定边界。** M/N/K 描述真正参与运算的元素范围，Tile 描述局部计算覆盖的范围。M/N 不整除输出 Tile 时，末尾 Tile 只包含部分有效坐标，所选实现通过边界加载和写回处理这些区域。局部 Tile 较大，并不要求问题尺寸随之取整。K 维尾部是否可用，还取决于输入访问要求。
- **实际存储决定 Stride。** `make_cute_packed_stride` 对应紧密排列。若 RowMajor 矩阵相邻两行的起点相差 ld 个元素，就应把实际 ld 填入该矩阵的行步长；例如 A 的 Stride 为 `(ldA,1,0)`，B 为 `(1,ldB,0)`。行尾填充不会改变逻辑 M/N/K，Buffer 容量、初始化索引和参考读取也要与实际步长一致。
- **访问粒度限制合法组合。** Tile 的边界处理不能替代对齐要求。本节紧密排列的 FP16 A/B 使用 16 字节 TMA 访问条件，连续维 K 和 N 分别需要满足 8 个元素的对齐。C/D 的 FP32 访问也要满足各自的对齐要求。换用其他类型或路径时，应先检查该实现对问题尺寸、基址和字节步长的要求，再由 `can_implement` 检查其覆盖的运行时条件。单纯增加分配空间或改大行步长，并不保证原来的问题尺寸已经合法。

选择 Kernel 时，先保留所需的数学问题，按实际存储生成 Stride，再判断参数是否适用。若显式扩展计算尺寸，输入也要随之调整。例如，扩展 K 会增加归约项，这些新增项需要产生零贡献。

## 准备矩阵数据并保留参考输入

本例用确定的有限值填充 A/B/C，并取非零 beta，以便同时检查乘积和 C 分支。A/B 在写入 Host 数组时就转换为 FP16，后面的参考计算读取同一份转换后的输入。

当前矩阵紧密排列，Buffer 大小分别是 MK、KN 和 MN。`DeviceAllocation` 的构造参数以元素为单位，复制操作把 Host 数组送入这些设备 Buffer：

```cpp
check(cudaSetDevice(0));  // 参数 0 为设备编号；在分配设备内存前选择。
std::vector<ElementA> h_A(size_t(M)*K);
std::vector<ElementB> h_B(size_t(K)*N);
std::vector<ElementC> h_C(size_t(M)*N);
std::vector<ElementD> h_D(size_t(M)*N);
// A/B 写入时即转为 FP16，参考计算也读取这些实际存储值。
for (int m=0; m<M; ++m)
  for (int k=0; k<K; ++k)
    h_A[size_t(m)*K+k] = ElementA(float((m+2*k)%7-3)/4);
for (int k=0; k<K; ++k)
  for (int n=0; n<N; ++n)
    h_B[size_t(k)*N+n] = ElementB(float((3*k+n)%5-2)/4);
for (int m=0; m<M; ++m)
  for (int n=0; n<N; ++n)
    h_C[size_t(m)*N+n] = float((m+n)%3-1)/4;

// 分配大小以元素为单位，D 由本次 GEMM 写入。
cutlass::DeviceAllocation<ElementA> d_A(h_A.size());  // ElementA：存储类型；size()：元素数。
cutlass::DeviceAllocation<ElementB> d_B(h_B.size());  // ElementB：存储类型；size()：元素数。
cutlass::DeviceAllocation<ElementC> d_C(h_C.size());  // ElementC：存储类型；size()：元素数。
cutlass::DeviceAllocation<ElementD> d_D(h_D.size());  // ElementD：存储类型；size()：元素数。
d_A.copy_from_host(h_A.data());  // 源 Host 地址；默认复制整个 d_A。
d_B.copy_from_host(h_B.data());  // 源 Host 地址；默认复制整个 d_B。
d_C.copy_from_host(h_C.data());  // 源 Host 地址；默认复制整个 d_C。
```


Host 数组保留本次输入，设备数组交给 Kernel 使用。本次 GEMM 写入 D 的所有有效输出位置，因此 D 分配后即可用于输出。

## 将参数交给 Adapter 并等待输出完成

一次调用的 `Arguments` 按问题、Mainloop 和 Epilogue 组织。Mainloop 接收 A/B 地址及 Stride；Epilogue 接收 alpha/beta、C/D 地址及 Stride。硬件信息描述当前选择的设备：

```cpp
// 记录已选择的设备，供 Kernel 配置使用。
cutlass::KernelHardwareInfo hardware_info{};
check(cudaGetDevice(&hardware_info.device_id));  // 输出参数：当前设备编号的保存地址。
hardware_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
    hardware_info.device_id  // 要查询 SM 数量的设备编号。
);
float alpha = 1.0f, beta = 0.5f;
typename GemmHandle::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,  // mode：普通 GEMM 模式。
    problem_shape,  // problem_shape：本次 M/N/K/L。
    { // mainloop：输入矩阵。
      d_A.get(),  // A 的设备地址。
      stride_A,   // A 的 (m,k,l) 步长。
      d_B.get(),  // B 的设备地址。
      stride_B    // B 的 (n,k,l) 步长。
    },
    { // epilogue：线性组合与输出。
      {
        alpha,  // Acc 的缩放系数。
        beta    // C 的缩放系数。
      },
      d_C.get(),  // 源矩阵 C 的设备地址。
      stride_C,   // C 的 (m,n,l) 步长。
      d_D.get(),  // 输出矩阵 D 的设备地址。
      stride_D    // D 的 (m,n,l) 步长。
    },
    hardware_info  // hw_info：设备编号与 SM 数量。
};
```


`cudaSetDevice` 在分配数据之前选择设备，`hardware_info` 记录该设备的信息供 Kernel 配置使用。Kernel 类型和本次参数确定后，Adapter 检查兼容性、准备辅助存储并提交执行：

```cpp
GemmHandle gemm;
check(gemm.can_implement(
    arguments  // 要检查的本次调用参数。
));
size_t workspace_bytes = GemmHandle::get_workspace_size(
    arguments  // 按本次问题和调度参数计算辅助存储需求。
);
cutlass::DeviceAllocation<uint8_t> workspace(
    workspace_bytes  // 分配元素数；uint8_t 的元素数等于字节数。
);
cudaStream_t stream = nullptr;  // 使用默认 stream。
check(gemm.initialize(
    arguments,        // args：本次问题、矩阵和后处理参数。
    workspace.get(),  // workspace：辅助存储的设备地址。
    stream            // stream：初始化所用执行流。
));
// 使用 initialize 保存的参数提交执行，不表示结果已经就绪。
check(gemm.run(
    stream  // Kernel 的执行流。
));
check(cudaStreamSynchronize(
    stream  // 等待此流上的工作完成。
));
d_D.copy_to_host(
    h_D.data()  // 目标 Host 地址；默认复制整个 d_D。
);
```


`can_implement` 检查本次参数是否满足该 Kernel 的要求；`get_workspace_size` 返回所需辅助存储字节数；`initialize` 将参数转换为设备执行需要的内部形式。Workspace 与矩阵 Buffer 都由调用者保留到执行结束。

这里的 **Workspace** 是调用者在设备全局内存中分配的辅助缓冲，其用途和容量取决于所选 Kernel、本次问题及调度参数；不需要这类空间时，查询结果也可以为零。它与前面讨论的 Mainloop/Epilogue 共享内存不同：`SharedStorage` 描述 Kernel 内的共享内存需求，参与 Stage 推导和启动配置，不能由 Workspace 代替。增加 Workspace 的分配量不会增加 Mainloop 的流水级数。[Adapter 的初始化实现](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/device/gemm_universal_adapter.h#L313-L347)分别处理 Workspace 初始化与共享内存需求。

本例使用默认 stream。`run` 提交 Kernel 后，`cudaStreamSynchronize` 等待该 stream 上的工作完成，随后将 D 复制回 Host。参考比较随后读取本次调用的输出。[Adapter 的初始化与启动接口](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/device/gemm_universal_adapter.h#L313-L335)也支持显式 stream。异步输入复制需要与这次执行建立依赖。

## 用相同输入计算参考结果

对每个 $(m,n)$，CPU 读取保留的 FP16 A/B 值，将它们提升为 double 后完成 K 维归约，再计算相同的 alpha/beta 组合。FP32 输出与参考值按照绝对误差和相对误差共同判断：

```cpp
double max_error = 0;
bool passed = true;
for (int m=0; m<M; ++m) {
  for (int n=0; n<N; ++n) {
    // 将同一份 FP16 输入提升为 double，计算独立参考值。
    double accum = 0;
    for (int k=0; k<K; ++k)
      accum += double(float(h_A[size_t(m)*K+k])) * double(float(h_B[size_t(k)*N+n]));
    double reference = double(alpha)*accum + double(beta)*double(h_C[size_t(m)*N+n]);
    double actual = h_D[size_t(m)*N+n];
    double error = std::abs(actual-reference);
    max_error = std::fmax(max_error,error);
    // 同时检查有限值、绝对误差与相对误差。
    passed &= std::isfinite(actual) && error <= 1e-5 + 1e-5*std::abs(reference);
  }
}

std::cout << (passed ? "PASS" : "FAIL")
          << " max_abs_error=" << max_error << '\n';
return passed ? 0 : 1;
```


`passed` 表示当前所有输出位置都满足给定误差条件，`max_error` 保存最大绝对误差。这里的容差用于本例的输入范围；更换类型、K 长度或量化方式后，需要相应确定允许误差。参考读取的是已经存入 A/B 的值，所以这次比较集中检查 GEMM 计算，不混入生成原始输入时的类型转换差异。

前面的 `check` 使用下面两个重载，将失败状态转换为异常。完整文件把它们定义在 `main()` 之前，再由 `main()` 统一捕获并报告错误：

```cpp
// 统一将 CUDA 与 CUTLASS 的失败状态转换为异常。
static void check(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
static void check(cutlass::Status result) {
  if (result != cutlass::Status::kSuccess)
    throw std::runtime_error(cutlassGetStatusString(result));
}
```


## 将基线用于另一项矩阵问题

修改实例时，先判断改变的是类型还是运行时参数。若 Element、矩阵 Layout Tag、Tile、Cluster、Schedule 和 Fusion 类型保持不变，只更换 M/N/K/L、矩阵地址、可变 Stride 或 alpha/beta，就仍然使用同一个 `GemmKernel` 类型。调用者重新准备相应 Buffer 和 `Arguments`，检查新参数，并按新的 Workspace 需求完成初始化。`run` 使用的是已经初始化的内部参数，只修改 Host 上的 `arguments` 后直接再次调用 `run`，不会自动更新先前保存的状态。

若更换 Element、RowMajor/ColumnMajor、Alignment、Tile、静态 Cluster、Stage、Schedule 或 Fusion 类型，就要重新实例化受影响的 Builder，并组合新的 Kernel 和 Adapter。此时 Stride 类型、共享内存需求和参数结构也应从新生成的类型取得。例如，改变 C/D 类型或后处理可能改变 Epilogue 的存储需求。使用 `StageCountAutoCarveout` 的 Mainloop 要根据新的存储需求重新计算流水级数。

上述修改方式适用于本节的静态 Tile/Cluster 配方。有些实现会在类型中预留动态维度，其具体值在运行时填写；第6部分的 Grouped 示例就使用动态 Cluster。运行时可以修改这些预留值，其余模板选择仍由类型确定。

上述代码依次构造 Dense 类型，描述问题、准备输入，并完成初始化、执行和输出比较。下面给出完整文件的组织与编译方式；第6部分再讨论其他场景对这些配置的修改。

## 完整示例与编译运行

[dense_baseline.cu](exemples/dense_baseline.cu)汇总了本章全部代码。文件先放置头文件和两节编译期类型构造，再定义 `check`；`main()` 按 Shape/Stride、数据准备、Adapter 调用和参考比较的顺序执行。末尾的迁移说明以注释保留，不增加另一轮计算。正文中的逐参数注释也保留在完整文件中。

示例使用以下头文件，其中 `iostream` 用于输出验证结果与错误信息：

```cpp
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
// CuTe 类型与 CUTLASS 的组件构造接口。
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
// 示例使用的设备内存管理与紧密步长工具。
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
```

在仓库根目录执行下面的命令。将 `CUTLASS_ROOT` 指向正文固定版本的 CUTLASS 源码，使用 CUDA 13.x 及其支持的宿主编译环境：

```bash
CUTLASS_ROOT=/home/jianyeshi/Note/GPUexpe/cutlass
nvcc -std=c++17 --expt-relaxed-constexpr \
  -gencode arch=compute_110a,code=sm_110a \
  -I"$CUTLASS_ROOT/include" \
  -I"$CUTLASS_ROOT/tools/util/include" \
  Docs/cutlass/02_cutlass_and_gemm/exemples/dense_baseline.cu \
  -o /tmp/dense_baseline

# 在支持该目标的 NVIDIA Thor 设备上执行。
/tmp/dense_baseline
```

程序输出 `PASS` 或 `FAIL` 以及最大绝对误差，数值验证通过时返回 0，不通过时返回 1；CUDA 或 CUTLASS 操作抛出异常时，完整文件输出 `ERROR` 和原因并返回 2。编译通过只说明能够构造程序和生成目标代码，数值正确性仍需在目标设备上运行验证。性能计时和问题尺寸的选择留待掌握基本过程后再讨论。

# 6. 相对 Dense 的四条场景路径

第5部分给出了 Dense 调用的完整过程。本章讨论数值表示、问题集合和阶段依赖变化时，数据与组件需要怎样调整。前三节沿用 Dense 的 Adapter 调用顺序；Attention 则先复用两次 GEMM，再进入专用融合 Kernel。各节的类型或参数代码为根据固定示例整理的节选。对应的完整 `.cu` 程序放在 [exemples](exemples/README.md)，各文件按本章小节组织注释，编译命令和验证范围见该目录说明。

回到第四部分的配置维度，四条路径分别从不同位置改变 Dense 基线：

- **Block-Scaled 改变操作数表示。** `Numerics` 增加 Payload 与 Scale 的类型和布局，硬件缩放路径还通过 `OpClassBlockScaledTensorOp` 选择相应计算引擎。
- **Grouped GEMM 改变问题集合与工作分配。** `Topology` 为每组保留独立 Shape，逐组地址与 Stride 交给匹配的 Mainloop/Epilogue，`Scheduling` 再组织各组产生的工作。
- **MoE 让路由结果决定问题集合。** Expert 的 Token Count 决定各项 GEMM 的尺寸；问题描述、地址组织和调度需要与这些尺寸相符，Expert 内部仍可选择不同的 `Numerics`。
- **Attention 改变阶段之间的依赖。** 两次 GEMM 之间需要 Mask、Softmax 和中间结果传递。融合实现由专用 Kernel 维护跨阶段状态，不能只用普通 GEMM 的 `Fusion` 参数表达完整算法。

它们不是互斥分类。例如，Grouped 或 MoE 的每项 GEMM 都可以使用 Block Scaling。下面分别展开这些变化，不把可组合的选择当成新的独立数学问题。

## Block-Scaled：从 Scale Block 到 SFA/SFB 与 GEMM 调用

NVFP4 用一组窄精度值和两级缩放因子表示原始张量。先说明各部分存放什么，再推导如何由原始值生成它们，最后把这些量接到 CUTLASS 的 Mainloop 和 Epilogue。量化关系可参阅[《NVFP4 GEMM requant量化方案精度分析及测试》](https://xiaopeng.feishu.cn/wiki/QoZMwTrpciWu7VkJCWvc0vyEnke)的格式与量化部分；下面统一采用“量化时除以 Scale、重建时乘以 Scale”的记号。

在进入具体格式之前，需要区分输入准备、数据视图与矩阵计算：

- **量化生成数据。** 调用者通过量化函数或 Kernel，从原始浮点输入生成低精度量化值和对应的 Scale。可以先在 CPU 上量化，再把结果复制到 GPU；也可以让 GPU 上的量化 Kernel 读取原始数据后生成结果。因此，量化应在这些输入被本节 GEMM 使用之前完成，而不一定发生在所有数据搬运之前。
- **CuTe view 描述数据访问。** 普通 `make_tensor(ptr, layout)` 将已有数据的访问方式与坐标布局组合起来。Engine 包装指针或迭代器，Layout 将坐标映射为偏移；创建或传递这个 view 不会自动求 Scale、执行缩放或生成低精度值。量化需要由实际的数值运算与转换完成，相关对象定义见 [CuTe Tensor 与 Engine](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/media/docs/cpp/cute/03_tensor.md)。
- **GEMM 消费量化结果。** 本节的硬件 Block-Scaled Mainloop 接收已经准备好的 Payload 与 Block Scale，并将它们交给支持块缩放的 MMA。调用者不需要先把 Block Scale 乘回 Payload、还原成完整浮点矩阵再提交。NVFP4 的 Tensor Scale 则按后文推导并入 Epilogue 的乘积系数。

完整示例见 [nvfp4_block_scaled.cu](exemples/nvfp4_block_scaled.cu)。它沿用下文的类型配置，取 `(M,N,K)=(256,1024,256)`，从原始浮点输入开始执行 CPU 量化，再构造 packed Payload、SFA/SFB 并调用 GEMM；量化误差与 GEMM 计算误差分开检查。

### NVFP4 由哪些数据组成

对本节采用的 Dense NVFP4 表示，先固定以下四项：

- **低精度量化值（Payload）：** 每个量化值采用 E2M1，占 4 bit，最大有限幅值为 6。它保存经过缩放后的近似数值，不包含下面两级 Scale。
- **Block Scale：** 每个量化块共享一个非负 E4M3 缩放因子。在本文的 CUTLASS 类型中使用 UE4M3，其最大有限值为 448。
- **Block Size：** 本节 Dense NVFP4 路径取 $V=16$，即每 16 个元素共享一个 Block Scale。
- **Tensor Scale：** 整个张量共享一个 FP32 缩放因子，记为 g，用于调整张量的整体范围。

E2M1 中的 E、M 分别表示指数位数和小数位数，再加一个符号位，共用 4 bit 编码一个数值。它是浮点格式，不是四位整数；指数与小数编码决定了可表示的离散值，最大有限幅值为 6 并不意味着区间内任意数值都能精确保存。UE4M3 是这里的非负 Scale 格式，仍按一个字节保存，Scale 自身也有转换和舍入误差，见[数值格式定义](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/exmy_base.h#L865-L883)。

这种两级结构见 [NVIDIA 对 NVFP4 的说明](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)。设原始值为 $x_i$，量化 Payload 为 $\widehat{x}_i$，元素 i 所属的块为 $b(i)$，该块的 Scale 为 $s_{b(i)}$。它们共同给出重建值：

$$x_i^{\mathrm{rec}}=g\,s_{b(i)}\,\widehat{x}_i\approx x_i$$

Payload 与 Block Scale 在不计 g 时能表示的最大幅值为：

$$6\times448=2688$$

这个数说明 Tensor Scale 的作用：原始张量的范围先由 g 调整，再由各块的 Scale 把局部数值变换到 E2M1 的范围。下面以非零、有限输入说明按最大幅值选择 Scale 的一种量化过程。具体实现还要处理 Scale 的舍入、下溢与零值。

### 从原始张量得到 Tensor Scale、Block Scale 和 Payload（量化后保存的低精度数值本身）

令 $\operatorname{amax}(\mathbf{x})=\max_i|x_i|$，$\mathcal I_b$ 表示第 b 个块的 16 个元素。量化可以按四步推导。

1. **求 Tensor Scale。** 用原始张量的最大幅值确定整体缩放：

   $$g=\frac{\operatorname{amax}(\mathbf{x})}{2688}$$

   g 用 FP32 保存。该式给出把张量最大幅值缩放到 2688 的目标，计算时仍使用实际保存的 g。

2. **按 Tensor Scale 归一化。** 对每个原始元素计算：

   $$x_i^{(1)}=\frac{x_i}{g}$$

   在理想算术下，归一化后的最大幅值为 2688。这里的 $\mathbf{x}^{(1)}$ 用来说明数值关系，量化实现可以在读取元素时直接完成除法，无须另存整个中间张量。

3. **求每个 Block Scale。** 对归一化后的每个块求最大绝对值，再用 E4M3 保存候选 Scale：

   $$s_b=Q_{\mathrm{UE4M3}}\!\left(
   \frac{\max_{i\in\mathcal I_b}|x_i^{(1)}|}{6}
   \right)$$

   $Q_{\mathrm{UE4M3}}$ 表示实际采用的 Scale 转换和舍入规则。后续生成 Payload 时要使用转换后的 $s_b$，这样计算端乘回的 Scale 才与量化端一致。

4. **生成 E2M1 Payload。** 每个元素再除以所属块的 Scale，并转换为 E2M1：

   $$\widehat{x}_i=Q_{\mathrm{E2M1}}\!\left(
   \frac{x_i^{(1)}}{s_{b(i)}}
   \right)$$

   $Q_{\mathrm{E2M1}}$ 按选定规则执行舍入与饱和。保存的结果是量化值 $\widehat{x}_i$、逐块的 $s_b$ 和张量级的 g。

例如，归一化后的一个块最大幅值为 12，则候选 Block Scale 为 2，且 2 可以由 UE4M3 精确表示。块内数值 3.1 除以 2 得到 1.55，按最近值舍入为 E2M1 的 1.5。局部重建值为 $2\times1.5=3$，再乘 g 才回到原始张量的尺度。两级 Scale 调整了表示范围，E2M1 舍入仍会产生误差。

全零张量可以约定 g=1、各块 Scale=1、Payload 全为 0；非零张量中的全零块也可使用 Scale=1、Payload 全为 0。非零张量的 g 或非零块的 Scale 若在转换时下溢为零，则需要量化器规定的非零下限或专门处理，避免继续除以零。Scale 舍入还可能使个别缩放值超过 E2M1 的范围，因此即使整体范围已经归一化，Payload 转换仍需明确饱和规则。

### 在 GEMM 中分别放置两级 Scale

A 的每一行、B 的每一列沿 K 划分为长度为 $V=16$ 的块。取 $L=1$、K 为 V 的整数倍，用 b 表示沿 K 的块编号、u 表示块内坐标：

$$k=bV+u,\qquad 0\leq u<V$$

A 的张量级因子记为 $g_A$，逐块 Scale 记为 $s^A_{m,b}$；B 对应为 $g_B$ 和 $s^B_{n,b}$。两个操作数的完整重建关系为：

$$a_{m,bV+u}^{\mathrm{rec}}
=g_A\,s^A_{m,b}\,\widehat a_{m,bV+u},\qquad{}
b_{bV+u,n}^{\mathrm{rec}}
=g_B\,s^B_{n,b}\,\widehat b_{bV+u,n}$$

SFA/SFB 保存这里的逐块 Scale。A 第 m 行中 $k=16,\ldots,31$ 的元素共享 $s^A_{m,1}$，下一行的同一 K 区间则有自己的 Scale。独立的逻辑 Scale 数量分别为 $M\times(K/V)$ 和 $N\times(K/V)$；$g_A/g_B$ 各为一个张量级标量。

回到第3部分的 Collective 输出区域 $(p,q)$。设 $\mathcal B_r$ 是第 r 个 K Tile 内有效的量化块编号集合，硬件 Block-Scaled Mainloop 对一个输出位置产生的局部贡献为：

$$\Delta\operatorname{acc}_{m,n}^{(r)}
=\sum_{b\in\mathcal B_r}s^A_{m,b}s^B_{n,b}
  \sum_{u=0}^{V-1}\widehat a_{m,bV+u}\widehat b_{bV+u,n}$$

块编号 b 改变时，$s^A_{m,b}s^B_{n,b}$ 也可以改变，因此 Block Scale 需要在 K 维归约中参与计算。Tensor Scale 则对整张 A/B 固定，可以提出归约：

$$D_{m,n}^{\mathrm{ref}}
=\alpha_{\mathrm{original}}\,g_Ag_B
  \sum_r\Delta\operatorname{acc}_{m,n}^{(r)}
+\beta C_{m,n}$$

所以 Mainloop 读取量化 A/B 与 SFA/SFB，Epilogue 使用 `alpha_effective = alpha_original * g_A * g_B`，beta 保持原值。若参考计算已经把 $g_A/g_B$ 乘回 A/B，则应继续使用 `alpha_original`，避免重复计入张量级缩放。

这里也能看出 Scale Block 与 K Tile 的区别：V 决定一次共享缩放覆盖多少元素，$T_K$ 决定 Mainloop 一轮处理的归约宽度。以 $T_K=256$、$V=16$ 为例，一个完整 K Tile 在 A 的每一行和 B 的每一列各对应 16 个 Block Scale。

### Scale Layout 由 Kernel 配方和 ProblemShape 共同确定

量化后，逻辑 SFA 的 $(m,b)$ 已有确定的 Scale 值，还需要找到它在设备 Buffer 中的位置。CUTLASS 为这个位置提供与所选 MMA 和搬运路径匹配的交错 Layout，量化器可以先生成逻辑 Scale 数组再重排，也可以按该 Layout 直接写入最终 Buffer。

固定版本的 [Example 72a](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm.cu#L95-L155)沿用第5部分的 `Sm100` 配方、FP32 累加与 Epilogue 计算，以及单问题的 Kernel/Adapter 组合关系；具体输入输出则改为 NVFP4 A/B 和 BF16 C/D。A 为 RowMajor，B 为 ColumnMajor，C/D 为 RowMajor。A/B 的 Alignment 为 32 个 FP4 元素，C/D 为 8 个 BF16 元素，都对应 16 字节。

该示例重新固定 `MmaTileShape=(256,256,256)`、静态 `ClusterShape=(2,4,1)`，Mainloop/Epilogue Schedule 都取 Auto，Epilogue 仍执行线性组合。下面的节选使用这组配置中的 `AlignmentA/B`、`ElementAccumulator`、Tile 和 Cluster；`CollectiveEpilogue` 也先按新的 C/D 类型生成。`StagePolicy` 表示使用该 Epilogue 的 `SharedStorage` 大小构造的 `StageCountAutoCarveout`，`MainloopSchedule` 表示 `KernelScheduleAuto`：

```cpp
// NVFP4 包装类型同时声明 Payload 与 Scale 的格式。
using ElementA = cutlass::nv_float4_t<
    cutlass::float_e2m1_t  // F4Type：4-bit Payload 类型；此处为 E2M1。
>;
using ElementB = cutlass::nv_float4_t<
    cutlass::float_e2m1_t  // F4Type：4-bit Payload 类型；此处为 E2M1。
>;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;  // 硬件块缩放路径。
using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,  // ArchTag：架构实现族。
    OperatorClass,  // OpClass：计算引擎类别。
    ElementA,  // ElementA：A 的 NVFP4 表示类型，含 Payload 与 Scale 格式。
    LayoutATag,  // GmemLayoutA：A 的全局内存布局标签。
    AlignmentA,  // AlignmentA：A 的访问对齐，以元素数计。
    ElementB,  // ElementB：B 的 NVFP4 表示类型，含 Payload 与 Scale 格式。
    LayoutBTag,  // GmemLayoutB：B 的全局内存布局标签。
    AlignmentB,  // AlignmentB：B 的访问对齐，以元素数计。
    ElementAccumulator,  // ElementAccumulator：K 维归约的累加类型。
    MmaTileShape,  // TileShape_MNK：完整 Collective 的局部 M/N/K 尺寸。
    ClusterShape,  // ClusterShape_MNK：CTA Cluster 的形状，以 CTA 个数计。
    StagePolicy,  // StageCountType：Mainloop 流水级数或自动推导策略。
    MainloopSchedule  // KernelScheduleType：Mainloop 的加载、乘加与协作方式。
>::CollectiveOp;

// Scale 配方与 Layout 必须取自实际生成的 Mainloop。
using ScaleConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;
using LayoutSFA   = typename CollectiveMainloop::LayoutSFA;
using LayoutSFB   = typename CollectiveMainloop::LayoutSFB;
constexpr int V   = ScaleConfig::SFVecSize;  // 当前 NVFP4 路径为 16
```

[`nv_float4_t`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/float_subbyte.h#L505-L513)把 `DataType` 设为 `float_e2m1_t`、`ScaleFactorType` 设为 `float_ue4m3_t`，供 Builder 识别数据和 Scale 格式。这个包装类型声明格式，并不把一个 Payload 和一个 Scale 存成同一个运行时元素。Payload 与 Scale 在运行时分别分配，生成的 `ScaleConfig::SFVecSize` 给出当前 Kernel 使用的量化块长度 V。

[Scale Layout 构造函数](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/detail/sm100_blockscaled_layout.hpp#L66-L124)把编译期 Scale 配方与本次 `(M,N,K,L)` 结合起来，生成覆盖完整问题的物理 Layout：

```cpp
auto problem = cute::make_shape(
    M,  // 输出行数，也是 A 的行数。
    N,  // 输出列数，也是 B 的列数。
    K,  // 归约长度。
    1   // 批次数 L。
);

// 用本次问题尺寸展开 Kernel 规定的 Scale Layout。
auto layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(
    problem  // 本次 M/N/K/L，用于展开 A 的 Scale 布局。
);
auto layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(
    problem  // 本次 M/N/K/L，用于展开 B 的 Scale 布局。
);

// filter_zeros 的参数是原 Layout；size 的参数是去除广播模式后的 Layout。
auto count_SFA = cute::size(cute::filter_zeros(layout_SFA));
auto count_SFB = cute::size(cute::filter_zeros(layout_SFB));
```

Layout 使用元素坐标定位 Scale：`layout_SFA(m,k,l)` 接收原矩阵的 k，逻辑 Scale 数组的第二个坐标则是量化块编号 b。因此把 $s^A_{m,b}$ 写入物理 Buffer 时，应使用该块的起点 $k=bV$：

```cpp
using ScaleA = typename ElementA::ScaleFactorType;
using ScaleB = typename ElementB::ScaleFactorType;

// host_SFA/host_SFB 已分别按 count_SFA/count_SFB 分配并初始化。
// logical_SFA/logical_SFB 是量化时实际使用的 Scale 值，按行连续保存。
// Layout 接收元素坐标 k，所以用块起点 b*V 定位 Scale。
for (int m = 0; m < M; ++m)
  for (int b = 0; b < K / V; ++b)
    host_SFA[layout_SFA(
        m,      // A 的行号。
        b * V,  // 当前量化块的起点 k。
        0       // 批次坐标 l。
    )] =
        ScaleA(logical_SFA[m * (K / V) + b]);

for (int n = 0; n < N; ++n)
  for (int b = 0; b < K / V; ++b)
    host_SFB[layout_SFB(
        n,      // B 的列号。
        b * V,  // 当前量化块的起点 k。
        0       // 批次坐标 l。
    )] =
        ScaleB(logical_SFB[n * (K / V) + b]);
```

以 $V=16$ 为例，`layout_SFA(m,16,0)` 到 `layout_SFA(m,31,0)` 指向同一个 Scale；块内 K 模式的 Stride 为 0，表达了 16 个量化值共享一次缩放。`filter_zeros` 在计算存储量时去掉这种广播重复，但保留物理布局要求的填充。分配容量因此应跟随 `count_SFA/count_SFB`，不能直接用逻辑 Scale 数量代替。

B 在数学中写作 $(k,n)$，而这里的 SFB Layout 按 $(n,k,l)$ 接收坐标：固定 n 后沿 K 选择量化块。A/B Payload 本身也要使用对应的矩阵 Layout 和 FP4 打包存储。

4 bit 是一个逻辑 FP4 元素的编码宽度，不等于一个 C++ 标量对象可以只占半个字节。`float_e2m1_t` 标量以字节容器保存编码，普通 `std::vector<float_e2m1_t>` 不会自动把两个元素压入一个字节。供当前 GEMM 使用的紧密打包 Payload 则在一个字节中保存两个 FP4 编码，需要通过支持子字节的容器、引用或迭代器访问，使逻辑元素索引落到正确的 4-bit 位置。固定示例用 [HostTensor 分别管理 Payload 与 Scale 存储](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm.cu#L178-L186)，也可对照 [Array 的子字节特化](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/array_subbyte.h#L46-L78)理解逻辑元素数与实际存储量的区别。

物理 Buffer 中的填充位置应初始化；K 不满足当前配置的对齐或尾部要求时，还需要按该 Kernel 支持的方式处理，生成一个 Scale Layout 并不会自动补齐输入。

四个 Buffer 中，A/B 保存量化 Payload，SFA/SFB 保存生成这些 Payload 时使用的 Scale，数值和物理位置均已确定。[Example 72a 的初始化代码](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm.cu#L330-L366)演示了 Buffer 的分配和复制。它分别随机填充 Payload 与 Scale 来验证 GEMM，原始浮点值的量化仍需由调用者完成。

### 用四组输入构造 Arguments

下面的 `Gemm` 表示按上述配置生成 `GemmKernel` 后得到的 `GemmUniversalAdapter` 类型。A/B Payload 与 SFA/SFB 复制到 Device 后，Mainloop 接收四组地址及布局：A/B 使用各自的 Stride，SFA/SFB 使用生成的完整 Layout。Epilogue 继续读取 Accumulator 和 C，并按 $\alpha\operatorname{Acc}+\beta C$ 生成 D。下面的 `alpha` 取前面推导的 `alpha_effective`，将张量级因子一并用于输出；SFA/SFB 只保存逐块 Scale。对应的运行时参数为：

```cpp
typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,  // mode：普通 GEMM 模式。
    {
      M,  // problem_shape：输出行数。
      N,  // 输出列数。
      K,  // 归约长度。
      1   // 批次数 L。
    },
    { // mainloop：两组 Payload，加两组 Block Scale。
      ptr_A,       // A Payload 的设备地址。
      stride_A,    // A Payload 的 (m,k,l) 步长。
      ptr_B,       // B Payload 的设备地址。
      stride_B,    // B Payload 的 (n,k,l) 步长。
      ptr_SFA,     // A 的逐块 Scale 设备地址。
      layout_SFA,  // A 的元素坐标到 Scale 物理位置的映射。
      ptr_SFB,     // B 的逐块 Scale 设备地址。
      layout_SFB   // B 的元素坐标到 Scale 物理位置的映射。
    },
    { // epilogue：Tensor Scale 通过 alpha 应用。
      {
        alpha,  // alpha_effective，已包含 g_A*g_B。
        beta    // C 的系数，不并入 g_A*g_B。
      },
      ptr_C,     // 源矩阵 C 的设备地址。
      stride_C,  // C 的 (m,n,l) 步长。
      ptr_D,     // 输出矩阵 D 的设备地址。
      stride_D   // D 的 (m,n,l) 步长。
    }
};
```

构造完参数后，复用第5部分的初始化、执行和同步顺序。A/B、SFA/SFB、C/D 与 Workspace 都应保留到 Kernel 使用结束。

Payload 类型、Scale 类型、V、Tile 与 Schedule 的组合首先由编译期 Builder 和具体 Collective 的类型约束确定。`can_implement(arguments)` 随后检查本次运行参数与已生成 Kernel 的兼容性；[当前 Block-Scaled Mainloop 的检查](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp#L592-L629)包括问题尺寸的 TMA 对齐及 SFA/SFB Layout 与配方的一致性。它不会读取每个 Scale 的数值来判断量化是否正确，数值关系要在执行后与参考结果比较。

### 反量化与误差比较

量化后先由 $\widehat{x}_i$、$s_{b(i)}$ 和 g 重建 $x_i^{\mathrm{rec}}$，再与原始 $x_i$ 比较，就能衡量这次数值表示产生的误差。设张量共有 $n_x$ 个元素，均方误差定义为：

$$\operatorname{MSE}
=\frac{1}{n_x}\sum_{i=0}^{n_x-1}
  \left(x_i-x_i^{\mathrm{rec}}\right)^2$$

MSE 对所有元素的平方误差求平均；$\|\mathbf{x}-\mathbf{x}^{\mathrm{rec}}\|_2$ 则是误差向量的 L2 范数，两者不是同一指标。使用 MSE 比较不同量化方案时，应采用同一份输入，并说明缩放规则和舍入方式。

对完整 GEMM，需要继续区分两次比较：

- **先检查量化输入的 GEMM 计算。** 保留实际写入的 Payload 和 Scale，用它们重建 A/B，并以高精度完成矩阵乘加。若重建已经包含 $g_A/g_B$，参考计算使用 `alpha_original`；若只重建局部 Scale 与 Payload 的乘积，则使用 `alpha_effective`。再执行相同的 beta 和输出类型转换，与 Kernel 的 D 按约定容差比较。参考读取逻辑 Scale 坐标，可以帮助发现物理 Layout 重排中的索引错误。
- **再评估量化对原始 GEMM 的影响。** 用未量化的高精度 A/B 计算另一份结果，与量化输入的参考结果比较。这样衡量的是输入表示和舍入对整个矩阵乘法的影响，不会把输入量化误差直接当成 Kernel 计算错误。

本节仍采用 Example 72a 的 BF16 输出。如果下一层还需要 NVFP4，就要对 D 再执行一次量化，并为 D 确定自己的 Tensor Scale 和 Block Scale。按上面的动态最大幅值规则计算 Tensor Scale，需要读取整个输出的幅值范围；当输出跨越多个 Tile 时，各 Epilogue 只掌握自己的局部结果。因此，输出 requant 能否融合还取决于输出 Scale 的来源及是否需要跨 Tile 归约，这与已知输入 Scale 的 Block-Scaled Mainloop 是两个不同的问题。

## Grouped GEMM：从多组矩阵到 Work Tile 分配

Grouped GEMM 一次处理 G 项独立矩阵乘法。各组的 Shape、矩阵地址和 Stride 可以不同，数值类型与局部计算方式由同一个 Kernel 类型确定。先计算各组产生多少个输出 Tile，再看这些工作如何分配。

完整示例见 [grouped_gemm.cu](exemples/grouped_gemm.cu)，使用下面三组问题、1SM 类型与动态 Cluster，包含描述数组的分配、复制、一次调用和逐组参考比较。

### 每组问题产生自己的输出 Tile

取输出 Tile 大小 $(T_M,T_N)=(128,256)$，考虑三组矩阵：

- Group 0 的 Shape 为 $(128,512,128)$，输出空间需要 $1\times2=2$ 个 Tile。
- Group 1 的 Shape 为 $(256,256,128)$，输出空间需要 $2\times1=2$ 个 Tile。
- Group 2 的 Shape 为 $(64,768,256)$，输出空间需要 $1\times3=3$ 个 Tile，其中 M 方向只有 64 行有效数据。

默认完整 K 归约下，第 g 组的输出 Tile 数为：

$$W_g=\left\lceil\frac{M_g}{T_M}\right\rceil
      \left\lceil\frac{N_g}{T_N}\right\rceil$$

三组一共有七个输出 Tile。Scheduler 取得一项工作后，需要确定它属于哪个 Group，以及该组内的 M/N Tile 坐标。Kernel 随后用同一组的 Shape、地址和 Stride 完成加载、K 维归约和输出写回。K 长度可以因组而异，因此相同的输出 Tile 数也未必意味着相同计算量。

Dense 使用一份 Shape 和地址描述。Grouped 有多份描述，因此每个工作项都要先确定自己属于哪一项问题。

### 用 Shape、地址和 Stride 三类数组描述问题

第 g 项问题需要以下信息：

```text
problem_shapes[g] = (M_g, N_g, K_g)
ptr_A[g], ptr_B[g], ptr_C[g], ptr_D[g]
stride_A[g], stride_B[g], stride_C[g], stride_D[g]
```

Shape 给出有效的矩阵范围，指针给出本组矩阵的起点，Stride 给出从坐标到地址的步长。三类数组中相同的 g 对应同一项 GEMM。

各组矩阵可以独立分配，也可以放进几个大 Buffer，再由每组 Offset 得到指针。合并分配不会改变每组 Shape 的独立性。Host 构造好这些描述后，将 Shape、Pointer 和 Stride 数组复制到 Device，供 Kernel 按 Group 索引读取。[Example 75 的分配与初始化](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/examples/75_blackwell_grouped_gemm/75_blackwell_grouped_gemm.cu#L457-L566)展示了这两层数据：矩阵本身，以及指向矩阵的描述数组。

### 为 Mainloop 与 Epilogue 同时选择逐组地址

[Example 75](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/examples/75_blackwell_grouped_gemm/75_blackwell_grouped_gemm.cu#L95-L168)的 1SM 配置沿用 `Sm100`、`OpClassTensorOp`、FP32 累加和 Epilogue 计算，但把 A/B 改为 `cutlass::float_e4m3_t`，C/D 改为 `cutlass::half_t`。A 为 RowMajor，B/C/D 为 ColumnMajor；AlignmentA/B 为 16，AlignmentC/D 为 8。其 Tile 为 `(128,256,128)`，对应本节前面推演的输出 Tile 大小。

该示例的 Cluster 类型为 `cute::Shape<int32_t,int32_t,cute::_1>`，M/N 两个 Cluster 维度留到运行时指定。“1SM”描述 MMA 的协作范围，Cluster 仍可以包含多个 CTA。下面节选 ProblemShape 与 Mainloop/Epilogue Schedule 的选择；构造两个 Builder 时，应使用刚才列出的数据类型、Layout、Alignment、Tile 和动态 Cluster 类型：

```cpp
// 每组独立保存 M/N/K，不再共用一份固定 Shape。
using ProblemShape = cutlass::gemm::GroupProblemShape<
    cute::Shape<
        int,  // M 维的运行时整数类型。
        int,  // N 维的运行时整数类型。
        int  // K 维的运行时整数类型。
    >  // ProblemShape_：单组问题的 M/N/K 类型。
>;
// Mainloop 与 Epilogue 同时选择 1SM Pointer-array 路径。
using MainloopSchedule =
    cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100;
using EpilogueSchedule =
    cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
```

[`GroupProblemShape`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/group_array_problem_shape.hpp#L54-L82)为每组保存独立 Shape。Mainloop Builder 的 Layout 参数使用 `LayoutA*`、`LayoutB*`，Epilogue Builder 使用 `LayoutC*`、`LayoutD*`，并分别传入上述 Schedule。这里的星号是模板类型表达的一部分，不是在运行时对 Layout 对象取地址；它让 Builder 选择通过指针数组取得各组地址和 Stride 的实现。

Epilogue 继续使用 `EpilogueTileAuto` 与 FP32 计算的线性组合，输出类型随 C/D 改为 FP16；Mainloop 的 Stage 仍通过该 Epilogue 的 `SharedStorage` 推导。下面把这两处 Pointer-array 选择写入 Builder，其他别名使用本节刚才确定的配置：

```cpp
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,  // ArchTag：架构实现族。
    OperatorClass,  // OpClass：计算引擎类别。
    MmaTileShape,  // TileShape_MNK：与 Mainloop 匹配的 Collective Tile。
    ClusterShape,  // ClusterShape_MNK：与 Mainloop 匹配的 CTA Cluster。
    cutlass::epilogue::collective::EpilogueTileAuto,  // EpilogueTileType：Epilogue 子分块；Auto 在编译期选择。
    ElementAccumulator,  // ElementAccumulator：Mainloop 提供的累加类型。
    ElementCompute,  // ElementCompute：后处理使用的计算类型。
    ElementC,  // ElementC：源矩阵 C 的存储类型。
    LayoutC*,  // GmemLayoutTagC：C 的布局标签；* 选择逐组地址与步长。
    AlignmentC,  // AlignmentC：C 的访问对齐，以元素数计。
    ElementD,  // ElementD：输出矩阵 D 的存储类型。
    LayoutD*,  // GmemLayoutTagD：D 的布局标签；* 选择逐组地址与步长。
    AlignmentD,  // AlignmentD：D 的访问对齐，以元素数计。
    EpilogueSchedule,  // EpilogueScheduleType：结果读取、后处理与写回方式。
    // FusionOpOrCallbacks：融合操作或回调类型。
    cutlass::epilogue::fusion::LinearCombination<
        ElementD,  // ElementOutput_：输出 D 的类型。
        ElementCompute,  // ElementCompute_：线性组合的计算类型。
        ElementC  // ElementSource_：源矩阵 C 的类型。
    >
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,  // ArchTag：架构实现族。
    OperatorClass,  // OpClass：计算引擎类别。
    ElementA,  // ElementA：A 的输入存储类型。
    LayoutA*,  // GmemLayoutA：A 的布局标签；* 选择逐组地址与步长。
    AlignmentA,  // AlignmentA：A 的访问对齐，以元素数计。
    ElementB,  // ElementB：B 的输入存储类型。
    LayoutB*,  // GmemLayoutB：B 的布局标签；* 选择逐组地址与步长。
    AlignmentB,  // AlignmentB：B 的访问对齐，以元素数计。
    ElementAccumulator,  // ElementAccumulator：K 维归约的累加类型。
    MmaTileShape,  // TileShape_MNK：完整 Collective 的局部 M/N/K 尺寸。
    ClusterShape,  // ClusterShape_MNK：CTA Cluster 的形状，以 CTA 个数计。
    // StageCountType：Mainloop 流水级数或自动推导策略。
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))  // carveout_bytes：为 Epilogue 预留的共享内存字节数。
    >,
    MainloopSchedule  // KernelScheduleType：Mainloop 的加载、乘加与协作方式。
>::CollectiveOp;
```

Mainloop 读取本组 A/B，Epilogue 读取和写回同组 C/D，二者都需要匹配的逐组寻址方式。两个 Collective 确定后，再组合 Kernel：

```cpp
// 组合 Grouped 问题类型和已经匹配的两个 Collective。
// TileScheduler_ 未显式指定，使用该 Kernel 的默认选择。
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,  // ProblemShapeOrThreadblockMma_：3.x 的问题描述类型。
    CollectiveMainloop,  // CollectiveMainloopOrEpilogue_：3.x 的 Mainloop 类型。
    CollectiveEpilogue  // CollectiveEpilogueOrThreadblockSwizzle_：3.x 的 Epilogue 类型。
>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<
    GemmKernel  // GemmKernel_：被包装的设备端 Kernel 类型。
>;
```

各组 Stride 数组的元素类型取自 `GemmKernel::InternalStrideA/B/C/D`，它们描述单组矩阵。取得类型后，再按每组尺寸生成步长。Kernel 对外接收的是 Stride 数组指针，不能将该指针类型用作数组元素。

### 将问题数组交给一次调用

以下参数节选假定 Device 上的各个描述数组已经准备好，`d_*` 表示相应数组的设备地址，`h_problem_shapes` 是保留在 Host 上的 Shape 数组。`fusion_args` 按当前 FP16 输出的线性组合准备，`scheduler_args` 按该 Kernel 的 Scheduler 参数类型准备。

由于 Cluster 为动态形状，`hardware_info` 除设备信息外还要填写 `cluster_shape` 和 `cluster_shape_fallback`。固定示例的默认值分别为 `(4,2,1)` 与 `(2,1,1)`，并在[运行参数构造处](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/examples/75_blackwell_grouped_gemm/75_blackwell_grouped_gemm.cu#L570-L584)传入。其他取值也需要满足所选实现和目标设备的 Cluster 要求。这里需要显式提供 Grouped 的运行时 Cluster 值，第5部分的静态 `(2,2,1)` Cluster 不会自动为它赋值。

```cpp
typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGrouped,  // mode：Grouped 模式。
    { // problem_shape：每组独立的问题描述。
      G,                            // num_groups：问题组数。
      d_problem_shapes,             // problem_shapes：Device Shape 数组。
      h_problem_shapes.data()       // host_problem_shapes：可选 Host Shape 数组。
    },
    { // mainloop：各数组相同的 g 对应同一组。
      d_ptr_A,     // 每组 A 地址组成的设备数组。
      d_stride_A,  // 每组 A 步长组成的设备数组。
      d_ptr_B,     // 每组 B 地址组成的设备数组。
      d_stride_B   // 每组 B 步长组成的设备数组。
    },
    { // epilogue：同一组的融合参数与 C/D。
      fusion_args, // 当前 Fusion 接口的 alpha/beta 等参数。
      d_ptr_C,     // 每组 C 地址组成的设备数组。
      d_stride_C,  // 每组 C 步长组成的设备数组。
      d_ptr_D,     // 每组 D 地址组成的设备数组。
      d_stride_D   // 每组 D 步长组成的设备数组。
    },
    hardware_info,  // hw_info：设备信息及首选/回退 Cluster 形状。
    scheduler_args  // scheduler：当前 Tile Scheduler 的运行时参数。
};
```

`GroupProblemShape` 保存 Group 数和 Device Shape 数组；可选的 Host Shape 数组供主机侧参数准备读取，若提供它，就应与 Device Shape 数组保持相同顺序和数值。Mainloop 和 Epilogue 分别接收自己使用的指针与 Stride 数组。最初可以给所有 Group 使用相同 alpha/beta；需要逐组标量时，再使用所选 Fusion 接口提供的标量数组。

初始化、执行和同步仍沿第5部分的顺序。Shape、指针与 Stride 数组，以及它们所描述的矩阵 Buffer，都应保留到相应执行结束。

### 区分问题集合与地址数组

Pointer-array 表示按数组取得地址，Grouped 表示每项问题有独立 Shape。这两个选择可以分别出现：[`ArrayProblemShape`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/group_array_problem_shape.hpp#L133-L172)允许同一 M/N/K 的 L 个批次使用指针数组，`GroupProblemShape` 则为 G 项问题分别保存 M/N/K。

因此，验证时应按问题描述选取边界。每组参考计算读取自己的 $(M_g,N_g,K_g)$ 和输入数据，只比较本组有效的 $M_g\times N_g$ 输出范围。若使用大 Buffer，还需要确认各组起点和步长不会使输出写入相邻组。矩阵是否合并分配、地址是否来自指针数组，都不能替代每组真实 Shape。

## MoE：从 Token 路由得到 Expert GEMM

MoE 的 Expert GEMM 位于路由与输出合并之间。完整过程包括：

1. **路由。** Router 为 Token 选择 Expert，并产生相应的 routing weight。
2. **聚集。** 按 Expert 归属组织 Token，保留原 Token 索引，并生成 `tokens_per_expert` 与各 Expert 的输入 Buffer。
3. **Expert GEMM。** 使用每个 Expert 的权重，计算它收到的 Token。
4. **恢复与合并。** 根据路由记录恢复 Token 顺序；同一 Token 的多路输出按约定的 routing weight 合并。

CUTLASS 在本节承担第三步，问题描述与 Scheduler 接收的是已经准备好的 Expert 数据。选择 MoE GEMM 类型不会自动执行 Router、Token 聚集或输出合并。

令 $I_e$ 是 Expert e 的 Token 索引，$T_e=|I_e|$；聚集后的激活矩阵与该 Expert 的权重相乘：

$$X_e\in\mathbb R^{T_e\times K},\qquad{}
W_e\in\mathbb R^{K\times M},\qquad{}
Y_e=X_eW_e\in\mathbb R^{T_e\times M}$$

K 是输入特征数，M 是输出特征数，$T_e$ 随路由结果变化。一次 MoE 调用由这些不同 Token 数的 Expert 计算组成，外层算法还负责输入聚集和最终结果合并。

完整示例见 [moe_expert_gemm.cu](exemples/moe_expert_gemm.cu)。它取三个 Expert、Token Count 为 `8、17、32`，按下文的最大槽位接口计算，并用固定 Top-1 路由展示输入聚集和输出恢复。路由关系由示例预先给定，不包含 Router 网络、Top-k 加权合并或完整 Expert FFN。

### 从矩阵方向理解 Token Count 为什么位于 N

固定版本 [Example 92 的普通 MoE Grouped 路径](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/examples/92_blackwell_moe_gemm/92_blackwell_moe_gemm_grouped.cu#L171-L231)把权重放在 A、激活放在 B。将上面的乘法转置，可得：

$$D_e=Y_e^T=W_e^TX_e^T=A_eB_e$$

其中 $A_e$ 为 $M\times K$，$B_e$ 为 $K\times T_e$，输出 $D_e$ 为 $M\times T_e$。此式对应 alpha=1、beta=0；需要线性组合时，再加入相应标量与 C。该实例的 GEMM Shape 因而是 $(M,T_e,K)$：Token Count 对应 N，M/K 则由模型的特征维度确定。

本节按该示例的 1SM、静态 Cluster `(1,1,1)` 配置解释后面的参数。它沿用 `Sm100`、`OpClassTensorOp`、FP32 累加与 Epilogue 计算，A/B 使用 `cutlass::float_e4m3_t`，C/D 使用 `cutlass::half_t`；A 为 RowMajor，B/C/D 为 ColumnMajor。AlignmentA/B 为 16，AlignmentC/D 为 8，Tile 为 `(128,16,128)`。

权重 A 由 TMA 加载，随 Token 数变化的激活 B 由 cp.async 加载，对应 `KernelMixedTmaCpAsyncWarpSpecialized1SmSm100`。Epilogue 使用 `EpilogueTileAuto`、`EpilogueScheduleAuto` 和线性组合，Mainloop Stage 仍由 Epilogue 存储占用推导。较小的 Tile N 用于覆盖少量 Token 的输出宽度。这组类型与 `MoEProblemShape` 一起生成后文的 `Gemm`。迁移时应一并调整这些类型和第5部分 Kernel 的问题尺寸。

### 用最大尺寸和 Token Count 描述每个 Expert

这类问题的 M/K 固定，只有 N 随 Expert 变化。[`MoEProblemShape`](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/group_array_problem_shape.hpp#L84-L131)因此用最大尺寸和 Token Count 数组描述各项问题。运行时准备代码节选如下：

```cpp
using ProblemShape = cutlass::gemm::MoEProblemShape<
    cute::Shape<
        int,  // M 维的运行时整数类型。
        int,  // N 维的运行时整数类型。
        int  // K 维的运行时整数类型。
    >  // ProblemShape_：每个 Expert 的 M/N/K 类型。
>;

// 路由已完成；这里只上传每个 Expert 的实际 Token 数。
cutlass::DeviceAllocation<int32_t> d_tokens_per_expert(E);  // E 个 Token Count。
d_tokens_per_expert.copy_from_host(
    h_tokens_per_expert.data()  // 源 Host 地址，按 Expert 顺序保存 Token Count。
);

// 设备端第 e 组使用 (max_m, tokens_per_expert[e], max_k)。
ProblemShape problem{
    max_m,  // max_m：固定的输出特征数 M。
    max_n,  // max_n：每个 Expert 槽位的 Token 容量上限。
    max_k,  // max_k：固定的输入特征数 K。
    E,      // num_groups：Expert 数量。
    d_tokens_per_expert.get()  // tokens_per_expert：设备上的实际 Token Count 数组。
};
```

设备端取得第 e 项问题时返回 `(max_m,tokens_per_expert[e],max_k)`。`max_n` 是当前分配可容纳的 Token 数上限，实际 $T_e$ 必须落在分配范围内。

普通 Example 92 为各 Expert 预留等大的矩阵槽位。A 的第 e 组起点相对基址偏移 `e*max_m*max_k` 个元素，B 偏移 `e*max_n*max_k`，C/D 偏移 `e*max_m*max_n`。这些固定槽位用于定位数据，$T_e$ 再决定槽位中参与计算的有效 N 范围。

相应的 `stride_C/stride_D` 从当前 Kernel 的类型取得，并按最大 `(max_m,max_n,E)` 构造。C/D 为 ColumnMajor，每组内沿 m 的步长为 1、沿 n 的步长为 `max_m`；Expert 之间的步长为最大槽位容量 `max_m*max_n`。因此存储描述使用 E 和最大尺寸。Dense 基线的 `L=1` 不适用于这里，每组实际 $T_e$ 也不改变预留的槽位间距。

所以，这个实例的 [Mainloop 参数](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/examples/92_blackwell_moe_gemm/92_blackwell_moe_gemm_grouped.cu#L404-L430)只需 A/B 基址：

```cpp
typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGrouped,  // mode：多 Expert 的 Grouped 模式。
    problem,  // problem_shape：最大尺寸、Expert 数量与实际 Token Count。
    { // mainloop：通过 Expert 索引和最大槽位寻址，不是指针数组。
      base_A,  // 所有 Expert 权重槽位的设备基址。
      base_B   // 所有 Expert 激活槽位的设备基址。
    },
    { // epilogue：C/D 的跨 Expert 步长按最大槽位构造。
      {
        alpha,  // Expert 矩阵乘积的缩放系数。
        beta    // 源矩阵 C 的缩放系数。
      },
      base_C,    // 所有 Expert 源矩阵槽位的设备基址。
      stride_C,  // C 的组内与跨 Expert 步长。
      base_D,    // 所有 Expert 输出槽位的设备基址。
      stride_D   // D 的组内与跨 Expert 步长。
    },
    hardware_info  // hw_info：当前设备信息；本例 Cluster 已静态固定。
};
```

Adapter 的调用顺序沿用第5部分，数值 Mainloop 和参数结构取自 Example 92。通用 Grouped 可以从指针数组找到各组数据；当前 MoE 实例从基址和最大槽位找到数据。选定当前 Group 后，两者都使用该组的真实 Shape 计算。

### 从 Expert 输出恢复 Token 输出

Scheduler 根据每个 $T_e$ 形成输出 Tile 集合。一个工作项选定 Expert e 后，读取该 Expert 的权重与激活，完成当前 Tile 的 K 维归约，再写回对应的 $D_e$。不同 Expert 的 Token 数会改变有效 Tile 数和尾部大小，工作分配策略围绕这种不均衡进行组织。

GEMM 返回的仍是按 Expert 组织的结果。外层算法需要按照路由记录恢复 Token 顺序；Top-k 路由还需要将同一个 Token 的多路 Expert 输出按 routing weight 合并。聚集索引、权重应用位置和合并规则应由完整 MoE 算子统一确定。

### 分别验证 Expert 计算与路由结果

这两类结果需要分别检查：

- **Expert GEMM 的结果。** 对每个 e 使用实际 $T_e$、聚集后的 Token 和本 Expert 的权重计算参考值，只比较对应的有效输出区域。
- **完整 MoE 的结果。** 检查聚集后的 Token 是否与路由索引一致，Top-k 路由是否按每条有效分支组织输入，routing weight 是否在约定位置应用，以及合并后的结果是否回到原 Token 顺序。逐 Expert 的矩阵乘法正确，不能代替这些检查。

若 Expert 采用 Block Scaling，还需要为它的量化输入准备相应 Scale Tensor，并按前一节的方法检查量化值与 Scale 的一致性；[Block-Scaled RC-Grouped MoE](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/examples/92_blackwell_moe_gemm/92_blackwell_moe_gemm_blockscaled_rcgrouped.cu)给出了这种组合的实现入口。

## Attention：从两个 GEMM 到在线 Softmax

Attention 的第二次矩阵乘法需要消费第一次乘法经过 Softmax 后的结果。先固定一个 Batch 和一个对应的 Query/KV Head，令：

$$Q\in\mathbb R^{S_q\times d},\qquad{}
K\in\mathbb R^{S_k\times d},\qquad{}
V\in\mathbb R^{S_k\times d_v}$$

输出为 $O\in\mathbb R^{S_q\times d_v}$。对 Query 行 i，有效 Key 位置 j 的分数、概率与输出是：

$$s_{i,j}=\frac{q_i^T k_j}{\sqrt d}+mask_{i,j},\qquad{}
p_{i,j}=\frac{\exp(s_{i,j}-m_i)}{\ell_i},\qquad{}
o_i=\sum_j p_{i,j}v_j$$

其中 $m_i=\max_j s_{i,j}$，$\ell_i=\sum_j\exp(s_{i,j}-m_i)$。下面假定每个 Query 行至少有一个有效 Key，先说明这一行如何完成计算。

本地 [attention_unfused.cu](exemples/attention_unfused.cu)实现下面的分离路径，取单 Batch、单 Head、等长 Q/K 序列和因果 Mask。它使用 FP16 Q/K/V，Softmax 后还将 P 显式转为 FP16，分数、累加与输出使用 FP32；参考计算计入相同的类型转换。这个教学实例与后文 BF16 Q、INT8 K/V 的专用融合示例是两套不同配置。

### 先把阶段依赖表示为两个独立 GEMM

最直接的实现先生成 $QK^T$，应用 $1/\sqrt d$ 和 Mask，再执行逐行 Softmax，最后计算 PV。两个 GEMM 的问题尺寸分别是：

- QKᵀ：$(M,N,K)=(S_q,S_k,d)$。
- PV：$(M,N,K)=(S_q,d_v,S_k)$。

第一项 GEMM 的输出经过缩放、Mask 和行归约后，成为第二项 GEMM 的输入。S/P 可显式保存在全局内存中，各阶段通过同一 stream 或事件依赖连接。这种实现可以分别观察分数、概率和最终输出，适合作为理解与核对融合算法的基础。

显式分数矩阵包含 $S_qS_k$ 个元素，随序列增长而增大。融合实现处理一个分数 Tile 后便消费它，省去完整 S/P 的存取。这要求 Softmax 能随 K/V Tile 逐步更新。

### 从第一个有效 Tile 建立状态

对当前 Query 行，保留三个量：已处理分数的最大值 $m_i$、相对这个最大值计算的指数和 $\ell_i$，以及未归一化的输出向量 $\mathbf u_i$。初始时：

$$m_i=-\infty,\qquad \ell_i=0,\qquad \mathbf u_i=\mathbf 0$$

设第一个含有效 Key 的 Tile 中，有效位置集合为 $J_1$。直接用这一 Tile 建立状态：

$$m_i^{(1)}=\max_{j\in J_1}s_{i,j},\qquad{}
\widetilde p_{i,j}^{(1)}=\exp(s_{i,j}-m_i^{(1)})$$

$$\ell_i^{(1)}=\sum_{j\in J_1}\widetilde p_{i,j}^{(1)},\qquad{}
\mathbf u_i^{(1)}=\sum_{j\in J_1}\widetilde p_{i,j}^{(1)}v_j$$

$\widetilde p$ 是尚未归一化的权重，已经可以和当前 V Tile 相乘。此时保存 $\ell_i$，就能把最终的除法延后到所有 K/V Tile 完成之后。

### 新的行最大值怎样修正旧结果

处理下一 Tile 时，新分数可能提高整行的最大值。设第 t 个 Tile 的有效位置集合为 $J_t$，则：

$$m_i^{(t)}=\max\left(m_i^{(t-1)},\max_{j\in J_t}s_{i,j}\right),\qquad{}
r_i^{(t)}=\exp\left(m_i^{(t-1)}-m_i^{(t)}\right)$$

旧的指数和与部分输出都相对于旧最大值计算，因此要同时乘以 $r_i^{(t)}$，再加入当前 Tile 的贡献：

$$\ell_i^{(t)}=r_i^{(t)}\ell_i^{(t-1)}
+\sum_{j\in J_t}\exp(s_{i,j}-m_i^{(t)})$$

$$\mathbf u_i^{(t)}=r_i^{(t)}\mathbf u_i^{(t-1)}
+\sum_{j\in J_t}\exp(s_{i,j}-m_i^{(t)})v_j$$

全部 Tile 完成后，输出 $\mathbf o_i=\mathbf u_i^{(T)}/\ell_i^{(T)}$。Correction 将已有的分母和输出贡献一起换到新的归一化基准下。

Mask 决定 $J_t$，所以必须在求最大值和指数之前生效。若某行在当前 Tile 中没有有效 Key，就保持该行状态不变；若整行都没有有效 Key，Softmax 的分母为零，算子需要另行约定输出行为，具体实现也要支持该约定。

### 把行状态放进融合 Kernel

对一个 Query Work Tile，Kernel 保留 Q 和行状态，依次读取 K/V Tile。QKᵀ MMA 生成分数，Softmax 阶段产生当前未归一化权重，PV MMA 将其与 V 相乘；Correction 使旧输出与新分数使用相同的归一化基准。最终阶段使用 Row Sum 完成除法并写回 O。

Pipeline 连接各阶段，传递它们共享的中间结果。普通 GEMM Epilogue 消费已经完成的 Accumulator；Attention 的行状态还会影响下一轮矩阵计算，需要由专用融合 Kernel 持续维护。

### 专用融合 Kernel 的接口与输入

固定版本的 [Mixed-Input FMHA](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/examples/python/CuTeDSL/cute/blackwell/kernel/attention/mixed_input_fmha/mixed_input_fmha_prefill_d256.py#L132-L190)使用 **Python CuTe DSL** 实现这组阶段。它沿用前面矩阵乘法与行归约的数学关系，具体配置来自专用 FMHA Kernel；第5部分的 Element、Tile、Cluster 和 Schedule 别名不直接用于这一接口。其运行时输入分为三组：

- **矩阵及量化数据：** `q_iter`、`k_iter`、`v_iter`、`o_iter` 提供 Q/K/V/O 地址；`scale_k_iter`、`scale_v_iter` 提供该 Mixed-input 实现所需的 K/V Scale。
- **问题尺寸：** `problem_shape` 按 `(batch, seqlen_q, seqlen_k, heads_q, heads_k, head_dim)` 描述完整调用。前面的数学推导固定一个 Batch 和一组对应 Head，这里再补上多 Batch、多 Head 的组织。
- **计算与执行参数：** `scale_softmax_log2`、`scale_output` 给出 Softmax 和输出系数，`window_size_left`、`window_size_right` 指定窗口边界，`stream` 指定执行流。

该示例还固定了以下数值与维度关系：

- 该 d256 示例采用 BF16 Q/O、INT8 K/V，QK 与 PV 都用 FP32 累加。K/V 的 Mixed-input 表示需要额外的 K/V Scale 指针和输入变换。
- 一个 `head_dim` 同时描述 Q/K 与 V/O 的特征宽度，当前实例取 $d_v=d$。
- `heads_q/heads_k` 决定多个 Query Head 怎样共享 KV Head，用于 GQA/MQA；两者需要满足相应整除与映射要求。

K/V 的 Scale 还需要确定共享范围。当前 d256 示例固定 `head_dim=256`，支持 `scale_granularity=128` 或 `256`；其 [Scale 的逻辑形状](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/examples/python/CuTeDSL/cute/blackwell/kernel/attention/mixed_input_fmha/mixed_input_fmha_prefill_d256.py#L1699-L1706)为 `(batch, heads_k, seqlen_k, head_dim / scale_granularity)`。因此，每个 Batch、KV Head、Token 的特征向量再按这一粒度分块，块内元素共享 Scale，而不是整张 K 或 V 只使用一个 Scale。

这些 Scale 在软件输入变换中应用。[K/V 反量化步骤](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/examples/python/CuTeDSL/cute/blackwell/kernel/attention/mixed_input_fmha/prefill_helpers.py#L234-L281)先读取 INT8 数据和对应 Scale，将 INT8 值转换为与 Q 相同的 BF16 操作数类型，再乘块级 Scale，得到后续 MMA 使用的操作数。这里的 BF16 操作数类型与 QK/PV 的 FP32 累加类型应分别理解；K/V Scale 也不是直接交给 NVFP4 硬件 Block-Scaled MMA 的 SFA/SFB。`scale_softmax_log2` 和 `scale_output` 则是另外的整体计算系数，分别作用于分数和最终输出。

示例内部的 [Load、MMA、Softmax 与 Correction Pipeline](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/examples/python/CuTeDSL/cute/blackwell/kernel/attention/mixed_input_fmha/mixed_input_fmha_prefill_d256.py#L507-L647)将这些数据连接起来。正文公式使用自然指数，代码使用 `exp2`，因此其 `scale_softmax_log2` 已包含 $\log_2 e$ 的换算。

### 按阶段与边界条件验证结果

核对融合结果时，参考计算应使用相同的输入值、反量化规则、Softmax 与输出系数、Mask、序列长度和 Head 映射，再比较 O。只检查输出是否有限，不能判断归一化或路由到 KV Head 的关系是否正确。

验证输入应覆盖短序列、非 Tile 整数倍的序列长度、因果或滑动窗口边界，以及幅值差异较大的分数。全 Mask 行需要先约定输出行为，并确认所选实现支持该约定，不能直接套用前面“至少有一个有效 Key”的推导。

若输出存在差异，先用分离实现比较 QKᵀ、Mask 后分数、概率及其行和，再比较 PV。需要进一步定位时，可以在调试实现中检查一个 Tile 的 Row Max、指数和与部分输出，判断差异来自输入表示、在线状态更新还是矩阵乘法。

# 7. 其他路径与后续阅读

## 其他数值表示从哪里改变 Mainloop

Block-Scaled 用独立 Scale 解释量化值。其他数值路径通过压缩信息、输入转换或多次基础 MMA，改变当前 K 段部分积的计算方式。

- **Mixed-input** 允许 A/B 使用不同类型或位宽。例如，量化权重可以比激活使用更窄的存储表示，但所选 MMA 仍有固定的操作数格式要求。因此 Mainloop 先加载存储值，再按需要解包、转换类型和调整布局，使它们成为 MMA 可以读取的操作数。若量化表示还包含 Scale 或零点参数，输入变换也要按对应规则使用这些数据。单纯转换类型并不自动完成反量化，入口见 [Mixed-input Builder](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/collective/builders/sm100_mixed_input_umma_builder.inl)。

  “零点”需要按具体接口的计算关系解释。设 q 是量化值，s 是 Scale；常见表示用整数零点 z 写成 $s(q-z)$。本文链接的 SM100 Mixed-input 实现则先转换输入类型、乘 Scale，再加上 `ptr_Z` 提供的值，忽略中间舍入时可写为 $qs+t$，其中 t 是加法项。因此从前一种表示准备参数时，应按 $t=-sz$ 换算，不能直接把 z 传作 t。[实际输入变换](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/detail/collective/mixed_input_utils.hpp#L1024-L1083)规定了这一顺序，实际数值还受转换与计算类型影响。这是该实现的参数约定，不是所有零点接口的统一定义；第六部分的 NVFP4 表示不包含这一平移项。

- **软件 Blockwise Scaling** 同样用量化值与 Scale 表示输入，但 Scale 参与运算的位置不同。固定一个输出位置，在两侧 Scale 都不变的 K 区间内，可以先用普通 Tensor MMA 求量化值的部分积，再乘对应的 Scale 乘积，加入完整累加结果。进入下一个缩放区间后，先前的部分积已经按自己的 Scale 合并，新区间则使用新的 Scale。因此不能先完成整个 K 归约，再统一乘一个块级因子。

  这里的块可以同时沿矩阵的两个维度划分。A 中一个 Scale 覆盖 `ScaleGranularityM × ScaleGranularityK` 个元素，B 中一个 Scale 覆盖 `ScaleGranularityN × ScaleGranularityK` 个元素；具体共享范围由 [Scale 配方与 Layout](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/examples/81_blackwell_gemm_blockwise/README.md)确定，不必等于 NVFP4 的 16 元素量化块。

  官方示例中的 **Groupwise** 同样讨论矩阵内部的缩放分组，而 **Grouped GEMM** 讨论一次调用中的独立矩阵问题，二者可以组合。例如，Example 81 的 [Groupwise 配置](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/examples/81_blackwell_gemm_blockwise/81_blackwell_gemm_groupwise.cu#L103-L106)取三个粒度为 `(1,128,128)`，使 A 的每一行沿 K 每 128 个元素共享一个 Scale，B 则按 N 与 K 方向的 128×128 块共享。这说明的是该实例的分组方式，不是所有 Groupwise 实现都必须采用的固定尺寸。

  Blockwise 说明 Scale 按块共享，“软件”则说明 Collective 显式组织部分积的缩放与合并。它不表示只有整张量级的 Tensor Scale，也不表示 GEMM 自动从原始浮点输入求 Scale 并完成量化；调用者仍需准备量化值与块级 Scale。矩阵乘法本身仍由 Tensor Core 执行，该路径使用 `OpClassTensorOp`；第六部分的硬件 Block-Scaled 路径则让 MMA 直接消费量化值与 Scale。两者的 Scale 粒度、Layout 和参数结构应分别从所选实现取得，不能仅因都使用块缩放而直接互换，入口见 [Blockwise Builder](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/collective/builders/sm100_blockwise_umma_builder.inl)。
- **Structured Sparse** 使用压缩值和 Metadata 描述逻辑稀疏矩阵，Sparse MMA 直接消费这些信息。输入同时带块缩放时，还要提供 Scale。对应入口分别为 [`OpClassSparseTensorOp` 的 Sparse Builder](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/collective/builders/sm100_sparse_umma_builder.inl)与 [`OpClassBlockScaledSparseTensorOp` 的 Sparse Block-Scaled Builder](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/collective/builders/sm100_blockscaled_sparse_umma_builder.inl)。本篇只定义这些输入关系，不展开 Sparse Kernel 构造。
- **Complex GEMM** 的输入包含实部与虚部，记为 $A=A_r+iA_i$、$B=B_r+iB_i$。将复数乘法拆为实数乘法并组合符号，可得：

$$\operatorname{Re}(AB)=A_rB_r-A_iB_i,\qquad{}
  \operatorname{Im}(AB)=A_rB_i+A_iB_r$$

  同一组数学关系可以由两种存储组织承载：

  - [Planar Complex](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/collective/builders/sm100_planar_complex_umma_builder.inl)将实部、虚部分别保存在独立的矩阵 Plane 中，Mainloop 从各 Plane 取得分量，再组合多次实数 MMA 的结果。
  - [Interleaved Complex TF32](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/collective/builders/sm100_interleaved_complex_umma_builder.inl)让每个复数元素的实部与虚部相邻交错保存。Mainloop 从交错存储中拆取、转换和重排分量，再由多次 TF32 MMA 形成同一复数结果。

  Planar 与 Interleaved 的区别在于数据怎样保存并供应给 MMA，逻辑 M/N/K 和复数乘法关系保持不变。
- **Fast-FP32 / 9xBF16** 的输入仍按 FP32 保存，Mainloop 则用多次 BF16 MMA 近似完成乘法。只把 FP32 转成一个 BF16 值，会舍去一部分有效信息；这条路径在取得第一个 BF16 分量后，用原值减去该分量，将残差缩放后继续转换，从而生成能补充前一分量的后续分量。

  固定实现为每侧输入生成三个 BF16 分量。完整展开时，A 的每个分量都与 B 的每个分量相乘，形成九组分量乘积，这就是 9xBF16 的含义。残差在分解时经过缩放，部分积合并时也要补偿相应尺度，才能近似恢复原来的 FP32 乘积。入口见 [9xBF16 Builder](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/collective/builders/sm100_9xBF16_umma_builder.inl)，分量生成过程见 [Mainloop 的输入变换](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/collective/sm100_mma_warpspecialized_emulated.hpp#L640-L683)。

  分量保留方式、参与计算的乘积项和累加顺序共同影响误差与计算量。它不等同于逐项执行 FP32 FMA，名称中的 Fast 也不保证对任意问题都更快；精度与性能仍需针对实际输入和目标设备检验。

## 改变工作分配或设备范围

第3部分以一个工作项完成当前输出 Tile 的全部 K 归约。Persistent 调度让 CTA/Cluster 完成当前工作后继续领取下一项工作；Stream-K 等方法进一步分配 K 范围，多个部分结果随后通过 Fixup 合并。这些方法调整工作分配与结果合并方式，Mainloop 仍计算自己取得的 K 范围。

Distributed GEMM 将问题分到多个设备，局部 GEMM 之间通过 All-Gather、Reduce-Scatter 或 All-Reduce 等通信连接。单设备内的类型构造仍可以沿本文进行，跨设备分片、通信和合并则由外层算法组织，见 [Distributed GEMM Example 82](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/examples/82_blackwell_distributed_gemm/82_blackwell_distributed_gemm.cu)。

这些实现的可用组合受固定版本和目标架构限制。类型与源码说明构造方法，数值正确性和性能需要在目标设备上检验。

## SIMT FP32 的类型对照

[SM100 FP32 SIMT 单元测试](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/test/unit/gemm/device/sm100_gemm_f32_f32_f32_simt_align1.cu#L54-L113)使用下面这组类型。它与第5部分的 Dense FP16 Tensor Core 基线共享 Builder→Collective→Kernel→Adapter 骨架，但 Mainloop 使用 CUDA Core FMA、显式三级 cp.async 流水和 `KernelMultistage`。

```cpp
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassSimt;  // CUDA Core FMA 路径。

using ElementA = float;
using ElementB = float;
using ElementC = float;
using ElementD = float;
using ElementAccumulator = float;
using ElementCompute = float;

using LayoutA = cutlass::layout::ColumnMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::ColumnMajor;
using LayoutD = LayoutC;

static constexpr int AlignmentA = 1;
static constexpr int AlignmentB = 1;
static constexpr int AlignmentC = 1;
static constexpr int AlignmentD = 1;

// 当前固定 SIMT 实现要求 Tile 的 K 维为 16。
using MmaTileShape = cute::Shape<
    cute::_128,  // M 维的局部 Tile 长度。
    cute::_128,  // N 维的局部 Tile 长度。
    cute::_16  // K 维的局部 Tile 长度。
>;
using ClusterShape = cute::Shape<
    cute::_1,  // M 方向的 CTA 个数。
    cute::_1,  // N 方向的 CTA 个数。
    cute::_1  // K 方向的 CTA 个数。
>;

using MainloopSchedule = cutlass::gemm::KernelMultistage;
using EpilogueSchedule = cutlass::epilogue::EpilogueSimtVectorized;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,  // ArchTag：架构实现族。
    OperatorClass,  // OpClass：计算引擎类别。
    ElementA,  // ElementA：A 的输入存储类型。
    LayoutA,  // GmemLayoutA：A 的全局内存布局标签。
    AlignmentA,  // AlignmentA：A 的访问对齐，以元素数计。
    ElementB,  // ElementB：B 的输入存储类型。
    LayoutB,  // GmemLayoutB：B 的全局内存布局标签。
    AlignmentB,  // AlignmentB：B 的访问对齐，以元素数计。
    ElementAccumulator,  // ElementAccumulator：K 维归约的累加类型。
    MmaTileShape,  // TileShape_MNK：完整 Collective 的局部 M/N/K 尺寸。
    ClusterShape,  // ClusterShape_MNK：CTA Cluster 的形状，以 CTA 个数计。
    // StageCountType：Mainloop 流水级数或自动推导策略。
    cutlass::gemm::collective::StageCount<
        3  // num_stages：显式流水级数；当前 SIMT 路径为 cp.async 流水。
    >,
    MainloopSchedule  // KernelScheduleType：Mainloop 的加载、乘加与协作方式。
>::CollectiveOp;

// SIMT 的后处理仍通过 Epilogue Builder 构造。
// 未显式传入 FusionOpOrCallbacks，采用默认线性组合。
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,  // ArchTag：架构实现族。
    OperatorClass,  // OpClass：计算引擎类别。
    MmaTileShape,  // TileShape_MNK：与 Mainloop 匹配的 Collective Tile。
    ClusterShape,  // ClusterShape_MNK：与 Mainloop 匹配的 CTA Cluster。
    cutlass::epilogue::collective::EpilogueTileAuto,  // EpilogueTileType：Epilogue 子分块；Auto 在编译期选择。
    ElementAccumulator,  // ElementAccumulator：Mainloop 提供的累加类型。
    ElementCompute,  // ElementCompute：后处理使用的计算类型。
    ElementC,  // ElementC：源矩阵 C 的存储类型。
    LayoutC,  // GmemLayoutTagC：C 的全局内存布局标签。
    AlignmentC,  // AlignmentC：C 的访问对齐，以元素数计。
    ElementD,  // ElementD：输出矩阵 D 的存储类型。
    LayoutD,  // GmemLayoutTagD：D 的全局内存布局标签。
    AlignmentD,  // AlignmentD：D 的访问对齐，以元素数计。
    EpilogueSchedule  // EpilogueScheduleType：结果读取、后处理与写回方式。
>::CollectiveOp;

// TileScheduler_ 未显式指定，使用该 Kernel 的默认选择。
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<
        int,  // M 维的运行时整数类型。
        int,  // N 维的运行时整数类型。
        int,  // K 维的运行时整数类型。
        int  // L 维的运行时整数类型。
    >,  // ProblemShapeOrThreadblockMma_：3.x 的问题描述类型。
    CollectiveMainloop,  // CollectiveMainloopOrEpilogue_：3.x 的 Mainloop 类型。
    CollectiveEpilogue  // CollectiveEpilogueOrThreadblockSwizzle_：3.x 的 Epilogue 类型。
>;

using GemmHandle = cutlass::gemm::device::GemmUniversalAdapter<
    GemmKernel  // GemmKernel_：被包装的设备端 Kernel 类型。
>;
```

这条固定源码路径只支持 FP32 A/B/Accumulator，且 `MmaTileShape` 的 K 维必须为 16。Direct GEMM 使用 `KernelMultistage`；Pointer-array 变体改用 `KernelPtrArrayMultistage` 和匹配的 Pointer-array Epilogue。


## 继续进入 Collective 与 CuTe 的内部实现

本文从矩阵问题出发，构造了 Kernel 类型，并给出填入运行时参数后的一次调用过程。Block-Scaled 改变 Mainloop 解释操作数的方式，Grouped/MoE 改变问题集合与工作分配，融合 Attention 则引入两次矩阵乘法之间的持续状态。处理新的场景时，可据此判断需要修改哪些类型和参数。

后续阅读可以按当前希望解决的问题选择：

- **已经会构造类型，想理解 Kernel 怎样执行。** 阅读[第二篇](https://xiaopeng.feishu.cn/wiki/Wod4wss3rirbjXkUcyjcUGJEn3e)，沿已经构造完成的 Kernel 看 Mainloop、Epilogue、Warp Role、Pipeline 和 Tile Scheduler 怎样共同执行一个 Work Tile。
- **已经理解执行过程，想继续构造或修改 CuTe 对象。** 阅读[第三篇](https://xiaopeng.feishu.cn/wiki/QJLZwqhRCiZYeBk5eL1cyU5kntg)，展开 Layout、Tensor、Atom、`TiledMMA` 与 `TiledCopy` 的参与者—数据对应关系，并追踪 TMA、共享内存描述符、TCGen05 与 TMEM 之间的数据传递。
