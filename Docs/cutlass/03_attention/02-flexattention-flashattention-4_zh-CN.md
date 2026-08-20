# FlexAttention + FlashAttention-4：快速且灵活

![](Imgaes/flexattention-flashattention-4/1.jpg)

#### 摘要：

在 Hopper 和 Blackwell GPU 上，FlexAttention 现在拥有 FlashAttention-4 后端。

我们在 PyTorch 中增加了自动生成 CuTe DSL 分数/掩码修改函数的支持，并可针对自定义 attention 变体即时编译并实例化 FlashAttention-4。

在计算受限工作负载上，这比现有 Triton 实现获得了 1.2–3.2 倍的性能提升。

### FlexAttention 回顾

FlexAttention 是一个 PyTorch API，只需几行 Python 就能实现自定义 attention 变体，无需编写 CUDA。用户编写一个修改 attention 分数的 `score_mod` 或 `mask_mod` 函数，其余工作由编译器处理：ALiBi、滑动窗口、文档掩码、soft-capping 及其组合都通过同一接口工作。

在底层，它是在普通 FlashAttention 之上增加的两项扩展：

1. 对 softmax 前分数进行逐点修改，并支持从全局内存任意加载。
2. 在前向和反向过程中进行块稀疏迭代，使用一种简单数据结构在运行时编码依赖数据的稀疏性。

仅此而已。当然，细节决定成败；但正如我们在[最初的 FlexAttention 文章](https://pytorch.org/blog/flexattention/)和 [FlexAttention 推理文章](https://pytorch.org/blog/flexattention-for-inference/)中所展示的，这两项扩展覆盖了大量常用 attention 变体。

在这个版本中，FlexAttention 新增了 FlashAttention-4（FA4）后端。用法如下：

```python
import torch
from functools import partial

from torch.nn.attention.flex_attention import flex_attention

flex_flash = torch.compile(
    partial(flex_attention, kernel_options={"BACKEND": "FLASH"}), dynamic=False
)

def local_boost(score, b_idx, h_idx, q_idx, kv_idx):
    return torch.where(torch.abs(q_idx - kv_idx) <= 8, score * 2, score)

B, H, S, D = 2, 8, 2048, 128
q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
out = flex_flash(q, k, v, score_mod=local_boost)
```

将 `BACKEND` 设置为 `"FLASH"` 即可使用 FA4 后端。需要较新的 [PyTorch nightly](https://pytorch.org/get-started/locally/) 版本和较新的 [FlashAttention checkout](https://github.com/dao-AILab/flash-attention/tree/main/flash_attn/cute/README.md)；请查阅安装文档确认版本兼容性。这部分代码仍在积极开发，稳定过程中可能出现破坏性变更。

用 FlexAttention 普及 attention 研究

FlexAttention 最初的设计目标及其命名含义，就是为 AI 研究人员构建原型和试验新 attention 变体提供灵活性。实践证明它确实做到了：[已有数十篇论文引用 FlexAttention](https://scholar.google.com/scholar?oi=bibs&hl=en&cites=6169255249999382801)，超过一千个代码仓库采用了它：

![](Imgaes/flexattention-flashattention-4/2.png)

虽然 Flex 成功赋能了研究人员，但用户反复反馈，他们最终会遇到难以突破的性能上限。在最初发布博客时，我们在 Hopper GPU 上与 FlashAttention-3（FA3）比较，性能约为后者的 80%。

如果今天重新测量，尽管两个实现都有改进，FlexAttention 的吞吐量却大约只有 FlashAttention-3 的 60%。

![](Imgaes/flexattention-flashattention-4/3.png)

一种常见模式由此出现：研究人员使用 Flex 试验，找到可行方案，但当性能变得关键时便撞上高墙。此时，必须由专家把它移植到更低层的实现。FlashAttention-3 一直通过增加新参数扩展功能，但每种新参数或新模式都需要进行低层重写。我们只是把负担从研究人员转移给了机器学习工程师。

在 Hopper 上，完全优化版本之间的性能差距或许还值得用来换取灵活性，但在更新的 Blackwell GPU 上，情况已经不同。

下面比较 Blackwell GB200 GPU（功耗 1000 W）上现有 Triton 版 FlexAttention（所有自动调优选项均调到最大）与 PyTorch SDPA 提供的 cuDNN attention 等高度优化实现：

![](Imgaes/flexattention-flashattention-4/4.png)

曾经很小的差距已经扩大成鸿沟。

## Blackwell：更大的 Tensor Core，更大的问题

在 Blackwell 上，高性能 attention 需要深度流水化、warp 特化的内核。我们基于 Triton 的实现无法表达这些技术。建议阅读[《Reverse Engineering FlashAttention-4》](https://modal.com/blog/reverse-engineer-flash-attention-4)中对 FlashAttention 内核的精彩解释；该文详细介绍了实现如何利用 Blackwell 的新硬件能力，以及 softmax 计算方式的更新。和以往一样，关键是让 Tensor Core 保持忙碌；由于它们变得更快，这需要大量使用深度异步流水线。

Blackwell 引入了张量内存（TMEM），这是靠近 Tensor Core、由程序员管理的暂存区，用于保存中间结果。更重要的是，数据移动和矩阵乘法现在都完全异步。一个 warp 可以启动矩阵乘法或加载，然后立即继续执行其他工作。

Warp 特化把工作拆分成多个阶段：一些 warp 处理 softmax 等需要寄存器的同步工作，另一些 warp 则通过发出加载和矩阵乘法、协调同步来组织异步流水线。由于组织流水线的 warp 寄存器压力较低，可以让更多操作同时处于执行中，从而隐藏延迟。

Tensor Core 变得更大、更快，但负责指数等运算的特殊函数单元（SFU）没有跟上。对前向 attention 而言，这改变了瓶颈：softmax 的 `exp()` 现在与矩阵乘法一样昂贵。为了让 GPU 始终满载，需要在两个矩阵块之间进行 ping-pong，把一个矩阵块的矩阵乘法与另一个矩阵块的指数运算重叠。下面的时间线展示了这些阶段如何交替以隐藏延迟。

![](Imgaes/flexattention-flashattention-4/blackwell_pingpong_pipeline.svg)

反向过程更加棘手。TMEM 不足以同时保存全部累加器，因此当共享内存、寄存器和张量内存都承受巨大压力时，内核必须精心设计流水线，以重叠计算和数据移动。

这种低层编排是通用编译器难以自动发现的。正如 [Gluon 介绍文章](https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/01-intro.py#L18-L26)所说：“Triton 编译器能够为大量内核生成高效代码，但经过手工调优的低层代码仍可能胜过它。当这种情况发生时，由于所有细节都被隐藏，用户几乎无法显著提升性能。”对作为元 attention 实现的 FlexAttention 来说，这一问题更加困难：当模式由用户定义时，很难在编译器中为特定模式硬编码优化。因此，我们开始研究更低层的实现，寻找提升 Blackwell 性能的最佳方法。

## 以 FlashAttention-4 为基础

面向新 Blackwell 硬件的 attention 实现经历了大量变化。cuDNN 很早便增加了高性能 attention 支持，但 Hopper 上现有的 SOTA 实现 FA3 无法在 Blackwell 上工作。SM100 上已不再存在 WGMMA：它被 TCGEN05 Tensor Core 指令取代，而且 Tensor Core 操作需要不同的内存空间——张量内存。

Tri Dao 等人开始开发 [FlashAttention-4](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute)，这是能够充分利用新硬件的更新版实现。

从 FA3 到 FA4 的一项重大变化是 [CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html)。它是 NVIDIA CUTLASS 团队最近发布的 Python DSL，用于借助 CUTLASS 抽象编写高性能 CUDA 内核。SOTA attention 实现大量使用 `cute.Layouts` 等 CUTLASS 抽象，但任何尝试安装过 FlashAttention 的人都知道，漫长的编译时间会带来多大痛苦。因此，尽管过去有人提出用 CUTLASS C++ 重写 Flex，但 FlexAttention 的动态性质和编译开销削弱了这一方案的吸引力。CuTe DSL 允许用 Python 编写过去需要 CUTLASS C++ 才能实现的内容，使 JIT 风格的工作流对 FlexAttention 更加实用。

在早期与 Tri Dao 讨论这条路线后，我们决定合作完成这一同时服务于 FlexAttention 和 FlashAttention-4 的实现。

我们没有构建独立实现，而是合作直接扩展 FA4，共享同一套异步流水线基础设施，并在 FlexAttention 需要注入分数修改和稀疏性的地方增加扩展点。

这意味着需要同时为前向和反向过程增加分数修改支持——把 score mod 内联到 FlashAttention 实现中——并支持 FlexAttention 使用的块稀疏元数据。

这项工作大致分为两部分：修改 FA4 以生成 FlexAttention 模板实例，以及更新 Inductor，使其能够从 PyTorch 表示生成所需的 CuTe DSL 代码。

## Inductor → CuTe DSL：胶水层

那么，需要为 FlexAttention 生成什么？逐点修改和任意加载。幸运的是，Inductor 并非第一次完成这类工作，已经存在用于此类扩展的机制。例如，下面查看一个可用于实现 ALiBi 的分数修改；有趣的是，它正是 FlexAttention 项目的启发性示例。

粗略来说，`torch.compile` 接收用户代码，并通过多个 IR 对其进行变换。这些变换会产生层次越来越低的表示。在 FX IR 中，仍然可以看到熟悉的 PyTorch 操作符，以及变量在使用后立即被设为 None。AOTAutograd pass 会自动生成反向过程：由于 `(X + A)` 对 X 的导数等于 1，链式法则会让梯度直接传递。

值得注意的是，在到达最终生成待运行内核代码的 Inductor 之前，这个栈中的任何部分都不需要“知道”CuTe DSL 代码是什么。

可依次查看下面的标签页，了解 ALiBi 如何从用户代码逐步演变成最终的 CuTe DSL 内核。

原始代码　FX IR　AOTAutograd　CuTe DSL

```python
def alibi_mod(score, b, h, q_idx, kv_idx):
    scale = torch.exp2(-((h + 1) * 8.0 / H))
    bias = (kv_idx - q_idx) * scale
    return score + bias
```

用户代码：实现 ALiBi 的分数修改，也是 flex-attention 项目的启发性示例。

Inductor 把逐点 IR 降级为调用 `V.ops.<op>` 的 define-by-run 函数，然后换入一个 handler，针对目标后端重新解释这些调用。在实践中，这体现为 `ops_wrapper(...)` 和 `OpsWrapper`：它们允许在不修改 IR 本身的情况下，把一元和二元原语映射到一种新语言。对于 CuTe DSL，我们插入一个 CuTe DSL handler，将这些操作重写为 TensorSSA 表达式，使算术在由寄存器（RMEM）支持的 cute 张量上执行，并可对表达式进行公共子表达式消除（CSE）。

我们还为“任意加载”增加了专用加载路径。如果用户编写的 score/mask mod 依赖某个全局张量，我们会实体化一个 RMEM fragment，并在可能为间接索引的位置发出加载。这样便能把 Inductor 的索引表达式连接到 CuTe DSL 的 TensorSSA。

## 让 FlashAttention-4 支持 FlexAttention

我们为 FA4 增加了两项相互正交的扩展，使其能够充当 FlexAttention 后端：

1. 前向和反向过程中的分数修改
2. 前向和反向过程中的块稀疏迭代

两项扩展都使用 CuTe DSL 实现，因此可以内联到使 FA4 获得高性能的同一条异步流水线中。

![](Imgaes/flexattention-flashattention-4/5.svg)

可以把 FlashAttention 理解为每个 SM 处理一条 KV 矩阵块队列。Flex 化增加了两个 hook：块稀疏性控制哪些矩阵块进入队列，包括跳过空块和标记部分块；score/mask mod 则在 softmax warp 中作为逐点操作应用。

明确这种分工后，下面说明前向和反向 hook 如何嵌入其中。

### 分数修改

CuTe DSL 使该项目可行的一项能力是：不仅可以向实现传递数量可变的内核参数，并将其降级为特定实例，还可以直接传递用户 callable。回到 FlexAttention 的核心需求，我们必须能够在 FlashAttention 算法的精确位置注入用户修改。我们的工作以现有 FA4 实现为基础，而它在编写时已经考虑了 score mod 的接入。

在前向过程中，我们把 `S` 矩阵块从 TMEM 读回寄存器，以便应用修改、计算逐行最大值与总和，并为第二个矩阵乘法生成 `P` 矩阵块。我们定义了一个与 FlexAttention `score_mod` 签名对应的 CuTe DSL 接口；它不在内核中传递 `N` 个可变捕获值，而是传入一个 `aux_tensors` 列表，表示修改函数使用的所有全局内存区域。在内核内部，我们把寄存器 fragment 重新解释为 TensorSSA view，可选择进行向量化，并在这些矩阵块上内联用户 callable。

计算最大值、总和并形成 `P` 矩阵块本来就需要把 `S` 放入寄存器，因此我们在数据驻留 RMEM 时应用 score/mask mod，而不增加独立阶段。这样可以保持原有流水线结构，以及 TCGEN 工作与 SFU 工作之间的重叠。对 `aux_tensors` 的额外读取会在需要时直接发出，并与现有的 `S` 消耗阶段一同调度。

反向过程使用相同的接口形式，并生成一个 `score_mod_bwd` callable，但数据存活期有所不同。在标准 FA4 中，`S` 和 `dS` 矩阵块从不需要同时存活，因此不同阶段可以共享 TMEM。加入 score mod 后，反向路径取决于用户导数所需的数据。

如果梯度只依赖 `P` 或传入梯度，我们会保留默认调度，并继续避免 `S` 与 `dS` 在 TMEM 中重叠。如果导数依赖 softmax 前分数，则把所需的 `S` fragment 与 `P` 或 `dS` 一起保存在寄存器中，并在其贡献被消耗后立即丢弃。TMEM 仍然为主要累加器保留，代价是这些特定修改会带来更高的寄存器压力。

### 块稀疏迭代（前向与反向）

FlexAttention 对 FlashAttention 要求的第二项修改是块稀疏迭代。我们扩展 FA4 内核，使其接受 block-mask 元数据，也就是需要访问的行/列矩阵块，并用这些元数据驱动矩阵块调度器，使内核只访问掩码中存在的 `(m, n)` 矩阵块。我们还让块稀疏路径支持 GQA 打包和广播的 head 维度。

前述双矩阵块 ping-pong 带来的一个结果是：Blackwell 上的最小稀疏块大小为 256×128，高于 Triton 路径的 128×128。由于每个 CTA 为保持流水线填满而处理两个 M 矩阵块（`q_stage=2`），调度器能够跳过的最小工作单元是 256 行，因此 block-mask 的粒度必须与之匹配。

反向过程遍历同一个 block-mask，只为前向过程中存在的矩阵块计算梯度。反向内核已经沿行使用子矩阵块迭代，因此 256 行约束能够自然融入。

### 我们的贡献

这些扩展需要在整个 FA4 技术栈中进行上游更新：

- 前向和反向过程中的 score-mod hook，包括 SM90/SM100 正确性修复和 GQA 边界情况
- 面向 Blackwell 和 Hopper 的块稀疏前向与反向路径，以及为广播 mask mod 提供的 pack-GQA 支持
- 清理 score/mask-mod 路径中连续布局和扩展张量的接口
- 升级 CuTe DSL 并启用 TVM-FFI，以降低 CPU 分派开销

完成这些工作后，下面来看性能表现。

## 结果

### SDPA 支持的模式

对于 dense（noop）和因果掩码等标准 attention 模式，可以把 FlexAttention 的新 Flash 后端与现有 Triton 实现和 cuDNN 进行比较。

![](Imgaes/flexattention-flashattention-4/gb200_triton_flash_cudnn_d128_hkv16-1-scaled.png)

在 GB200 上，Flash 后端的前向过程比 Triton 快 1.6–3.2 倍，反向过程快 1.85–2.3 倍。对于反向过程，Flash 在部分情况下达到甚至超过 cuDNN；前向过程与 cuDNN 的差距更大，尤其是因果 attention。

可以看到，在前向过程中，Noop 与 cuDNN 非常接近，而 Causal 落后更多。这一差距突显出，与 FA4 内置 Causal 路径相比，块稀疏迭代引入了多少开销。

#### 因果模式为何落后，以及如何缩小差距

经过调查，问题之一是工作调度：阅读 [FA3 代码](https://github.com/Dao-AILab/flash-attention/tree/main/hopper)可以看到其中使用了[最长处理时间优先（LPT）调度](https://en.wikipedia.org/wiki/Longest-processing-time-first_scheduling)；FA4 为内置因果模式实现了这种调度，但 FlexAttention 没有使用。如果手动指定 LPT 调度，性能如下：

![](Imgaes/flexattention-flashattention-4/gb200_causal_flex_lpt_fa4_d128_hkv16-scaled.png)

手动指定 LPT 调度后，较短序列的前向过程最高可加速 1.6 倍；随着序列变长，提升逐渐降至约 1.1 倍。反向过程差异很小，因为其调度开销的摊销方式不同。性能仍未完全追平，但差距已经缩小。

LPT 调度在这里有效，是因为已知具体稀疏模式为因果模式，而且该调度对此情况最优。一般而言，我们无法提前知道模式：块稀疏性可能依赖数据，不同行可能拥有不同数量的活跃 KV 块。

可以依赖 CUDA 以负载均衡方式启动各个输出矩阵块，但这样会失去持久化调度带来的收益：无法把 MMA 和加载与 epilogue 重叠，也无法避免重复执行 prologue。这正是 [Cluster Launch Control（CLC）](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_cluster_launch_control.html#dynamic-scheduler-with-cluster-launch-control)所解决的问题。CLC 是支持动态工作调度的 Blackwell 特性：它不在启动时把矩阵块静态划分给各个 SM，而是允许工作者动态查询新矩阵块。当某个 SM 提前完成时——例如其所在行需要处理的块较少——它会立即接手下一个可用矩阵块，而不是空闲等待。CuTe DSL 4.4 增加了基于 CLC 的持久化调度支持，使 FlexAttention 无需用户指定调度，就能透明受益于块稀疏模式下更好的工作分配。

### FlexAttention 支持的模式

FlexAttention 真正面向的是 SDPA 不支持的模式：ALiBi、文档掩码、滑动窗口和任意用户自定义分数修改。

![](Imgaes/flexattention-flashattention-4/gb200_triton_flash_d128_hkv16-scaled.png)

对于 B200 上这些仅由 Flex 支持的模式：

- ALiBi：前向加速 1.2–2.1 倍，反向加速 1.9–2.9 倍
- 文档掩码：较长序列的前向最高加速 2.7 倍，反向最高加速 3 倍
- 滑动窗口：前向加速 1.4–2.1 倍，反向加速 1.8–2.2 倍

### Hopper（H200）结果

在 Hopper GPU 上，Flash 在所有序列长度下都始终更快。

![](Imgaes/flexattention-flashattention-4/h200_triton_flash_d128_hkv16-scaled.png)

对于 H200 上这些仅由 Flex 支持的模式：

- ALiBi：前向加速 1.30–1.54 倍，反向加速 1.36–1.65 倍
- 文档掩码：前向加速 1.41–1.89 倍，反向加速 1.48–2.01 倍
- 滑动窗口：前向加速 1.45–1.65 倍，反向加速 1.35–1.52 倍

即使序列较短（2K）也能获得提升，而且加速比会随序列长度增加而增大。

## 正确性与基准测试方法

本文所有基准测试数据均由 [`attention-gym/benchmarks/flex_perf.py`](https://github.com/meta-pytorch/attention-gym/blob/main/benchmarks/flex_perf.py) 生成。

### 正确性

我们通过将输出与 FP32 参考结果比较来验证 Flash 后端：先把 Q/K/V 转换为 FP32，执行 attention，再转换回原类型。上游测试套件会持续执行这些检查：

- PyTorch Inductor：在 [`test/inductor/test_flex_flash.py`](https://github.com/pytorch/pytorch/blob/main/test/inductor/test_flex_flash.py) 中覆盖大量 `score_mod`/`mask_mod` 模式，包括捕获的缓冲区和 view，并比较 Flash 与 Triton。
- FlashAttention（CuTe）：在许多（`seqlen_q`，`seqlen_k`）组合上对 `mask_mod` + 块稀疏执行压力测试，并在 [`tests/cute/test_mask_mod.py`](https://github.com/Dao-AILab/flash-attention/blob/main/tests/cute/test_mask_mod.py) 中使用 `flex_attention` 参考实现验证前向与反向结果。

除单元测试外，我们还在真实训练环境中验证了 Flash 后端：使用 [torchtitan](https://github.com/pytorch/torchtitan)，在 64 张 H100 GPU 上以序列长度 8192 训练 Llama 3 70B。两次运行经过 1000 个训练步骤后，最终损失都收敛到约 3.7：

![](Imgaes/flexattention-flashattention-4/flash_vs_triton_loss-scaled.png)

### 局限性

块大小约束：对于分页 attention，例如 vLLM 集成，通常会让内核块与页面大小对齐。目前，FA4 路径围绕 Hopper 上的 128×128 块和 Blackwell 上的 256×128 块进行调优；后者是因为 `q_stage=2`，因此修改块大小的灵活性有限。随着 FA4 提供更稳健的较小 `tile_m`/`tile_n` 选项，我们计划启用该特性。

动态标量：完全支持动态张量形状，并在运行时解析。但 `score_mod` 或 `mask_mod` 捕获的标量会固化进编译后的内核。如果 `soft_cap` 值在不同调用之间发生变化，每个不同的值都会触发重新编译：

```py
def tanh_softcap(score, b, h, q_idx, kv_idx):
    return soft_cap * tanh(score / soft_cap)
```

需要梯度的捕获缓冲区的反向过程：Flash 后端目前不支持这种情况。例如，可学习的 bias 张量：

```py
bias = torch.randn(seq_q, seq_kv, device='cuda', requires_grad=True)
def bias_func(score, b, h, q_idx, kv_idx):
    return score + bias[q_idx, kv_idx]
```

Triton 后端支持捕获缓冲区的梯度；遇到这些情况时应使用 Triton 后端。

使用块稀疏时的确定性反向过程：启用块稀疏后，Flash 后端的反向过程尚不具备确定性；仅使用 score mod 的工作负载具有确定性。我们正在积极修复该问题。

性能局限：

- 前向过程中沿 KV 维度的加载可能使流水线停顿，尤其是指针追逐模式，例如带有逐 token 元数据的文档掩码；此时，辅助张量加载难以与计算重叠。
- 在当前分块方式下，如果反向过程中的 score mod 需要 softmax 前分数，几乎总会发生寄存器溢出。例如，`score**2` 的梯度为 `2 * score * grad_score`，这要求 softmax 前分数在反向过程中保持存活。TMEM 已被主要 attention 累加器完全占用，而当前块大小很少能在 SMEM 中为 `S` 矩阵块留出空间，因此它会一直保存在寄存器中并发生大量溢出，造成明显减速。

## 后续工作

CuTe DSL 与 FA4 的集成正在缩小研究与生产之间的差距，这令人振奋。

具体到 Flash 后端，我们正在支持 score mod 捕获的动态标量，使其无需重新编译，例如在不同调用之间改变 `soft_cap` 值。在可预见的未来，捕获缓冲区的梯度仍将依赖 Triton 后端。我们还在探索动态持久化调度，以自动改善块稀疏模式下的工作分配。

虽然本文讨论的是 FA4 实现，但 Triton 实现仍支持范围广得多的硬件；我们计划继续改进两个后端。

## 致谢

这是一项跨仓库合作。

FlashAttention-4 内核工作——CuTe DSL 实现、调度，以及 score/mask mod 和块稀疏所需的扩展点——位于上游 [`Dao-AILab/flash-attention`](https://github.com/Dao-AILab/flash-attention)；编译器与集成工作——FlexAttention API 行为、Inductor 降级和 CuTe DSL 代码生成——位于上游 [`pytorch/pytorch`](https://github.com/pytorch/pytorch)。

感谢两个仓库的维护者、审阅者和贡献者，也感谢 NVIDIA CUTLASS/CuTe DSL 团队构建了让 JIT 风格工作流成为可行方案的抽象。

- FlashAttention / FA4（内核 + 扩展点）：Tri Dao、Ted Zadouri、Reuben Stern、Markus Hoehnerbach、Jay Shah
- PyTorch / Inductor（降级 + 代码生成 + 集成）：Markus Hoehnerbach
- CuTe DSL / CUTLASS：Fung Xie

## 延伸阅读与链接

- [Attention Gym](https://github.com/meta-pytorch/attention-gym)：FlexAttention 模式的示例脚本和基准测试
- [Colfax Research 指南](https://research.colfax-intl.com/a-users-guide-to-flexattention-in-flash-attention-cute-dsl/)：包含 `score_mod`/`mask_mod` 示例的实践指南
- [Reverse Engineering FlashAttention-4](https://modal.com/blog/reverse-engineer-flash-attention-4)：深入分析 FA4 内核架构
- [CuTe DSL 文档](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html)：NVIDIA 面向 CUDA 内核的 Python DSL
- [FlashAttention-4](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute)：使用 CuTe DSL 的 FA4 实现
