# layout

tensor_a = 原始 A 张量视图，用 GEMM 语义 A[m,k,l] 描述 global memory 中的 A

tensor descriptor a = 给 TMA/底层 copy 指令使用的 global tensor 寻址描述符；
  在这个例子里为了把连续的 K 维放前面，所以 descriptor 坐标写成 (k,m,l)

mA_mkl = 在 descriptor 之上重新包装出来的 GEMM 语义视图
mA_mkl[m,k,l] -> descriptor_a[k,m,l]
为了将算法逻辑统一封装起来，避免后续 GEMM 代码直接面对各种物理 layout。

gA_mkl = global memory 上的 A[m,k,l]，并且已经按照 CTA tile / K tile 分块

tAgA_mkl = 给 A 的 TMA/load 逻辑使用的 global A 分块视图。
gA = global A
tA = A 这个 operand 对应的 load/TMA partition 视图

TMA = Tensor Memory Accelerator，用 descriptor + coord 从 global tensor 中选 tile，异步搬到 shared memory

tensor descriptor = TMA 用来理解 A 在 global memory 中怎么寻址的元数据

load() = 根据 cta_coord 和 ktileiter，从 tAgA 中选出当前 K step 的 tile，然后发起 TMA 搬到 shared memory

cta_coord：当前 CTA 负责 C/D 的哪一个输出 tile

k_tile / ktileiter：当前 mainloop 走到第几个 K 分块

UTCCP = 从 shared memory 异步拷贝数据到 tensor memory，也就是 SMEM -> TMEM，是 SASS/微架构层面的说法

tcgen05.cp
= shared memory -> tensor memory ，是 PTX 层面的 Tensor Core data movement 指令。

tcgen05.mma = Blackwell Tensor Core MMA 指令族；
  根据具体变体，从 SMEM/TMEM 等来源读取 operand 或 scale factor，执行矩阵乘累加。

tensor_sfa(物理)：SFA 在 global memory 里的真实物理排列。

mSFA_mkl
= 把物理 packed descriptor 重新包装成 GEMM 语义下的 SFA[m,k,l] 视图
mSFA_mkl ≈ SFA[m, k_scale, l]

sSFA = TMA 搬到 shared memory 后的 SFA 布局
