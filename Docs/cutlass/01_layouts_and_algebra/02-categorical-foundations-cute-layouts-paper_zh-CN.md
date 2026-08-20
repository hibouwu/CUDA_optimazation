# CuTe Layout 的范畴论基础

Jack Carlisle Jay Shah Reuben Stern Paul VanKoughnett 

Colfax Research 

research@colfax-intl.com 

2026 年 1 月

## 摘要

NVIDIA CUTLASS 库提供了一套稳健且富有表达力的方法，用于描述和操作 GPU 上的多维张量数据。这些方法在概念上以 CuTe layout 这一抽象概念及其丰富的 layout algebra 为基础，其中包括 composition、logical product 和 logical division 等操作。本文聚焦一类自然出现的 tractable layout，提出一个用于理解这种 layout algebra 的范畴论框架。为此，我们定义了两个 category——**Tuple** 和 **Nest**——其 morphism 会产生 layout。我们在这些 category 的 morphism 上定义一组操作，并证明它们与相应 layout 操作相容。此外，我们还完整刻画了由该构造产生的 layout。最后，我们提供这些范畴论构造的 Python 实现，以及证明其行为与 CUTLASS 一致的测试。该实现位于 Git 仓库：https://github.com/ColfaxResearch/layout-categories。

## 目录

1 引言 3
1.1 主要结果概要 6
1.2 文章结构 8
1.3 相关工作 9
1.4 实现 11
1.5 记号 17
2 Layout 及其代数 18
2.1 Flat layout 18
2.1.1 Tuple 18
2.1.2 基本定义 19
2.1.3 基本操作 27
2.1.4 Flat coalesce 36
2.1.5 Compact flat layout 41
2.1.6 Complement 44
2.1.7 其他操作 54
2.1.8 Tractable flat layout 56
2.2 Nested tuple 58
2.2.1 Profile 58
2.2.2 基本定义 60
2.2.3 Substitution 63
2.2.4 Refinement 64
2.3 Layout 67
2.3.1 基本定义 67
2.3.2 基本操作 70
2.3.3 Coalesce 72
2.3.4 Relative coalesce 74
2.3.5 Compact layout 77
2.3.6 Complement 78
2.3.7 Composition 80
2.3.8 Logical division 81
2.3.9 Logical product 84
2.3.10 Tractable layout 85

3 Layout 的 category 87
3.1 Category **Tuple** 87
3.1.1 基本定义 87
3.1.2 从 tuple morphism 到 flat layout 91
3.1.3 示例 99
3.1.4 Tuple morphism 的 realization 103
3.1.5 Tuple morphism 上的操作 106
3.2 Category **Nest** 124
3.2.1 基本定义 124
3.2.2 从 nested tuple morphism 到 layout 125
3.2.3 示例 128
3.2.4 Nested tuple morphism 的 realization 130
3.2.5 Refinement 131
3.2.6 Nested tuple morphism 上的操作 139
4 计算 147
4.1 Tractable layout 的 composition 147
4.1.1 Mutual refinement 148
4.1.2 从 mutual refinement 到可复合 morphism 152
4.1.3 Composition 算法 153
4.1.4 示例 154
4.1.5 更一般的 composition 160
4.1.6 Composition 的 admissibility 161
4.2 Logical division 与 logical product 163
4.2.1 Logical division 示例 163
4.2.2 Logical product 示例 164
A Category 入门 166
A.1 什么是 category？166
A.2 什么是 functor？169

## 第 1 章

## 引言

在现代计算中，尤其是 GPU 编程中，性能在很大程度上取决于多维数据如何在内存中存储和访问。我们关心的大多数数据——例如图像、视频以及机器学习中的张量——本质上都是多维的，但计算机内存从根本上说是一维的。这意味着，当需要加载、存储或以其他方式操作数据时，必须把多维逻辑坐标映射到一维物理坐标。这种映射称为 layout，是正确、高效读写内存的基础。此外，在 GPU 的 SIMT 执行模型中，layout 还用于描述和操作线程在数据上的划分方式。这对于优化内存访问模式以及正确调用面向 Tensor Core 等专用硬件的指令十分重要。

作为启发性示例，假设要在内存中存储一个 4×8 矩阵

$$
A = \left[ \begin{array}{c c c c c c c c} 1 2. 4 7 & 8 7. 2 1 & 3 4. 0 8 & 5 6. 9 3 & 4 5. 6 5 & 9. 1 7 & 7 3. 0 2 & 2 1. 3 9 \\ 6 4. 8 8 & 3 0. 4 1 & 1. 7 2 & 8 8. 0 4 & 9 2. 5 5 & 1 7. 0 6 & 5 0. 9 1 & 6 8. 7 7 \\ 3. 3 3 & 7 7. 1 9 & 6 1. 5 8 & 2 9. 4 6 & 1 5. 8 2 & 8 0. 7 5 & 4 4. 6 2 & 3 9. 2 8 \\ 9 1. 4 0 & 2 6. 1 2 & 6. 9 7 & 5 3. 0 3 & 5 8. 6 6 & 3 3. 7 9 & 1 1. 2 0 & 7 0. 5 5 \end{array} \right]
$$

为此，需要为 A 的每个条目指定内存地址。我们先为 A 的第 `(0,0)` 个条目选择某个地址，再为 A 的其他每个条目指定偏移。一种常见选择是 row-major layout：

<table><tr><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td></tr><tr><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td><td>14</td><td>15</td></tr><tr><td>16</td><td>17</td><td>18</td><td>19</td><td>20</td><td>21</td><td>22</td><td>23</td></tr><tr><td>24</td><td>25</td><td>26</td><td>27</td><td>28</td><td>29</td><td>30</td><td>31</td></tr></table>

记号 $L ^ { \mathsf { r o w } } = ( 4 , 8 ) : ( 8 , 1 )$ 表示矩阵第 `(i,j)` 个条目的偏移为

$$
(i, j) \cdot (8, 1) = 8 i + j.
$$

另一种常见选择是 column-major layout：

<table><tr><td>0</td><td>4</td><td>8</td><td>12</td><td>16</td><td>20</td><td>24</td><td>28</td></tr><tr><td>1</td><td>5</td><td>9</td><td>13</td><td>17</td><td>21</td><td>25</td><td>29</td></tr><tr><td>2</td><td>6</td><td>10</td><td>14</td><td>18</td><td>22</td><td>26</td><td>30</td></tr><tr><td>3</td><td>7</td><td>11</td><td>15</td><td>19</td><td>23</td><td>27</td><td>31</td></tr></table>

同样，记号 $L ^ { \mathsf { c o l } } = ( 4 , 8 ) : ( 1 , 4 )$ 表示矩阵第 `(i,j)` 个条目的偏移为

$$
(i, j) \cdot (1, 4) = i + 4 j.
$$

这些 layout 非常有用，但无法满足所有用途。例如，在高性能计算中，通常按以下方式计算矩阵乘积 AB：

1. 把操作数矩阵 A 和 B 划分成矩阵块；

2. 计算各个矩阵块之间的矩阵乘积；

3. 合并这些部分结果，得到完整结果 AB。

例如，可以把 4×8 矩阵 A 划分成 2×2 矩阵块，如下所示。

$$
A = \left[ \begin{array}{c c c c c} \left[ \begin{array}{c c} 1 2. 4 7 & 8 7. 2 1 \\ 6 4. 8 8 & 3 0. 4 1 \end{array} \right] & \left[ \begin{array}{c c} 3 4. 0 8 & 5 6. 9 3 \\ 1. 7 2 & 8 8. 0 4 \end{array} \right] & \left[ \begin{array}{c c} 4 5. 6 5 & 9. 1 7 \\ 9 2. 5 5 & 1 7. 0 6 \end{array} \right] & \left[ \begin{array}{c c} 7 3. 0 2 & 2 1. 3 9 \\ 5 0. 9 1 & 6 8. 7 7 \end{array} \right] \\ \left[ \begin{array}{c c} 3. 3 3 & 7 7. 1 9 \\ 9 1. 4 0 & 2 6. 1 2 \end{array} \right] & \left[ \begin{array}{c c} 6 1. 5 8 & 2 9. 4 6 \\ 6. 9 7 & 5 3. 0 3 \end{array} \right] & \left[ \begin{array}{c c} 1 5. 8 2 & 8 0. 7 5 \\ 5 8. 6 6 & 3 3. 7 9 \end{array} \right] & \left[ \begin{array}{c c} 4 4. 6 2 & 3 9. 2 8 \\ 1 1. 2 0 & 7 0. 5 5 \end{array} \right] \end{array} \right]
$$

现在假设要切出 A 的单个矩阵块，并假定 A 在内存中采用 column-major 格式。可以手动计算偏移：对第 `(i,j)` 个矩阵块，索引到其左上角条目的偏移为 $2i+8j$。另一方面，为了更好地组织该计算，可以使用矩阵块的 interleaved layout：

<table><tr><td>0</td><td>2</td><td>8</td><td>10</td><td>16</td><td>18</td><td>24</td><td>26</td></tr><tr><td>1</td><td>3</td><td>9</td><td>11</td><td>17</td><td>19</td><td>25</td><td>27</td></tr><tr><td>4</td><td>6</td><td>12</td><td>14</td><td>20</td><td>22</td><td>28</td><td>30</td></tr><tr><td>5</td><td>7</td><td>13</td><td>15</td><td>21</td><td>23</td><td>29</td><td>31</td></tr></table>

其中，各列由 A 的矩阵块给出，各行由矩阵块形状内部的坐标给出。这里使用 colexicographic 顺序，线性枚举矩阵块以及矩阵块内坐标，因此 layout $L ^ { \mathrm { t i l e d } }$ 的顶层 shape 为 `(4,8)`。

不过，$L ^ { \mathrm { t i l e d } }$ 所示的交织模式意味着：不存在任何 stride a、b，使它能够表示为 layout `(4,8):(a,b)`。相反，可以分解 shape `(4,8)` 的 mode，并定义

$$
L ^ {\text { tiled }} = ((2, 2), (2, 4)): ((1, 4), (2, 8)).
$$

之前的偏移计算 $2i+8j$，现在可以通过在坐标 `(0,(i,j))` 上求值 $L ^ { \mathrm { t i l e d } }$ 得到，而矩阵块 layout 本身由第一个 mode 给出。因此，给 A 赋予 layout $L ^ { \mathrm { t i l e d } }$ 形成 $A ^ { \mathrm { t i l e d } }$ 后，可以用以下 slice 得到 A 的第 `(i,j)` 个矩阵块：

$$
A _ {i, j} = A ^ {\text { tiled }} (\_, (i, j)).
$$

CUTLASS 中发展出的一项关键思想是：可以通过某些基本操作，从更简单的 layout 系统推导 $L ^ { \mathrm { t i l e d } }$ 这类有用但更复杂的辅助 layout。对 $L ^ { \mathrm { t i l e d } }$，相应操作称为 logical division。如果把矩阵块 layout 写成

$$
T = (2, 2): (1, 4) = \quad \begin{array}{c c} \hline 0 & 4 \\ \hline 1 & 5 \\ \hline \end{array}
$$

那么 $L ^ { \mathrm { t i l e d } }$ 就是以下 logical division：

$$
L ^ {\text { tiled }} = L ^ {\text { col }} \oslash T
$$

如下图所示。

<table><tr><td>0</td><td>4</td><td>8</td><td>12</td><td>16</td><td>20</td><td>24</td><td>28</td></tr><tr><td>1</td><td>5</td><td>9</td><td>13</td><td>17</td><td>21</td><td>25</td><td>29</td></tr><tr><td>2</td><td>6</td><td>10</td><td>14</td><td>18</td><td>22</td><td>26</td><td>30</td></tr><tr><td>3</td><td>7</td><td>11</td><td>15</td><td>19</td><td>23</td><td>27</td><td>31</td></tr></table>

$$
T = \begin{array}{c c} \hline 0 & 4 \\ \hline 1 & 5 \\ \hline \end{array}
$$

<table><tr><td>0</td><td>2</td><td>8</td><td>10</td><td>16</td><td>18</td><td>24</td><td>26</td></tr><tr><td>1</td><td>3</td><td>9</td><td>11</td><td>17</td><td>19</td><td>25</td><td>27</td></tr><tr><td>4</td><td>6</td><td>12</td><td>14</td><td>20</td><td>22</td><td>28</td><td>30</td></tr><tr><td>5</td><td>7</td><td>13</td><td>15</td><td>21</td><td>23</td><td>29</td><td>31</td></tr></table>

除 logical division 外，其他基本 layout 操作还包括 logical product、complement，以及最重要的 composition。这些 layout 操作是 CUTLASS 的支柱；深入理解其行为，有助于编写正确且高性能的代码。不过，这些操作的定义与构造相当微妙。例如，只有当 layout A 和 B 满足某些整除约束时，其 composition $B\circ A$ 才定义良好；CUTLASS 会在底层检查这些约束。具体而言，两个 layout 何时可复合，以及如何解释其 composition，并不总是显而易见。

## 1.1 主要结果概要

本文的主要思想是：把关注范围限制在 tractable layout 上，就能建立一套直观而强大的 layout 数学框架。Tractable layout 的条目满足一个简单的整除条件，参见定义 2.3.10.1。它几乎包括实践中遇到的所有 layout，例如：

• 随处可见的 row-major 和 column-major layout；

• 把数据存储在连续内存地址中的 compact layout；

• 广播多份数据副本的 projection；

• 支持带 padding 的加载和存储的 dilation。

如果 L 是 tractable layout，就可以用图表示 L。例如，layout $L ^ { \mathsf { r o w } }$、$L ^ { \mathsf { c o l } }$ 和 $L ^ { \mathrm { t i l e d } }$ 分别由下图表示。

$$
(4, 8): (8, 1)
$$

$$
\begin{array}{c} 8 \\ 4 \end{array} \xrightarrow {} \begin{array}{c} 4 \\ 8 \end{array}
$$

$$
(4, 8): (1, 4)
$$

![image](Imgaes/categorical-foundations-cute-layouts-paper/ef1061948d42a8da3554daa91f31edbb26803006f646c269d01cf2b4f21b6012.jpg)


$$
((2, 2), (2, 4)): ((1, 4), (2, 8))
$$

![image](Imgaes/categorical-foundations-cute-layouts-paper/94c89b562134bbb5f9c15a94235619ef1cfd373c18434aa3c3d7aa5d880d40c8.jpg)


这些图可以解释为某个 category 中的 morphism。因此，我们能够借助范畴论的力量描述 layout 及其操作。<sup>1</sup>

更精确地说，我们定义 category **Nest**：其 object 是由正整数组成的 nested tuple，其 morphism $f:S\to T$ 对应上述这类图；细节参见定义 3.1.1.13 和定义 3.2.1.1。如果 L 是 non-degenerate tractable layout，参见定义 2.3.1.24，那么存在一个本质上唯一、编码 L 的 Nest-morphism f，如下面的对应定理所示。

定理 A（参见 3.2.2.15）：存在一一对应关系

$$
\left\{ \begin{array}{l} N o n - d e g e n e r a t e \\ t r a c t a b l e l a y o u t s \end{array} \right\} \longleftrightarrow \left\{ \begin{array}{l} N o n - d e g e n e r a t e \\ \mathbf {N e s t - m o r p h i s m s} \\ o f s t a n d a r d f o r m \end{array} \right\}
$$

Composition、logical division 和 logical product 等 layout 操作，都可以在 category **Nest** 中自然解释。如果

$$
S \xrightarrow {f} T \xrightarrow {g} U
$$

是 Nest-morphism，则可以形成 composite

$$
S \xrightarrow {g \circ f} U
$$

只需把相应的图拼接在一起。例如：

![image](Imgaes/categorical-foundations-cute-layouts-paper/611e02cef2bdb0dead4b446450fcb03fb9f377e97e8d8a1fa9359fddf9a7c157.jpg)


我们证明，**Nest** 中的 composition 与 layout composition 相容。

定理 B（参见 3.2.6.21）：如果 f 和 g 是 non-degenerate 且可复合的 Nest-morphism，则

$$
L _ {g \circ f} = L _ {g} \circ L _ {f}.
$$

可以通过合并相邻箭头，对 Nest-morphism f 执行 coalesce。例如：

![image](Imgaes/categorical-foundations-cute-layouts-paper/cf51ba67b40ca70ba72e4a8bfbee68e6e9ce98aec9066a216fb4532a3f4cd645.jpg)


我们证明，该操作与 layout coalesce 相容。

定理 C（参见 3.2.6.13）：如果 f 是 Nest-morphism，则

$$
L _ {\text { coal } (f)} = \text { coal } (L _ {f}).
$$

Nest-morphism f 的 complement 是对未被 f 命中的条目所作的 inclusion。例如：

![image](Imgaes/categorical-foundations-cute-layouts-paper/78e93526fb7edc239f56f9507cbd3da522ef53e91b1518c5c331e3574ad40433.jpg)


我们证明，**Nest** 中的 complement 与 layout complement 相容。

定理 D（参见 3.2.6.20）：如果 $f:S\to T$ 是单射 Nest-morphism，且 $N={\mathsf{size}}(T)$，则

$$
\operatorname{coal} \left(L _ {f ^ {c}}\right) = \operatorname{comp} \left(L _ {f}, N\right).
$$

我们定义 Nest-morphism 的整除性，以及 logical division 操作

$$
f, g \mapsto f \oslash g
$$

其定义条件是 g 整除 f。例如：

![image](Imgaes/categorical-foundations-cute-layouts-paper/dae1c082f224dcc785d039b35bad77519606082d3fc53366b6270ad10e6895a7.jpg)


我们证明，**Nest** 中的 logical division 与 layout 的 logical division 相容。

定理 E（参见 3.2.6.26）：如果 f 和 g 是 non-degenerate Nest-morphism，且 g 整除 f，则

$$
\operatorname{coal} \left(L _ {f \oslash g}\right) = \operatorname{coal} \left(L _ {f} \oslash L _ {g}\right).
$$

我们定义 Nest-morphism 的 product admissibility，以及 logical product 操作

$$
f, g \mapsto f \otimes g
$$

其定义条件是 f 和 g 为 product admissible。例如：

![image](Imgaes/categorical-foundations-cute-layouts-paper/628e4334067a399c610f27223441883f009552d00be6191aa3613aa7eeb2d0bb.jpg)


我们证明，**Nest** 中的 logical product 与 layout 的 logical product 相容。

定理 F（参见 3.2.6.31）：如果 f 和 g 是 non-degenerate Nest-morphism，且 f 和 g 为 product admissible，则

$$
L _ {f \otimes g} = L _ {f} \otimes L _ {g}.
$$

第 4 章说明如何使用新框架计算 composition、logical division 和 logical product 等重要 layout 操作。特别地，我们提出一个算法（算法 4.1.3），用于计算 tractable layout A 与 B 的 composition $B\circ A$。省略细节后，算法的基本思想是：如果要计算 $B\circ A$，可以用适当选择的 Nest-morphism f 和 g 表示 A 与 B，复合这些 morphism 得到 $g\circ f$，再取其编码的 layout，从而得到

$$
B \circ A = L _ {g \circ f}.
$$

我们使用大量示例说明该算法。

## 1.2 文章结构

本文结构如下。

第 1.4 节介绍 layout 的 cute 实现细节。我们以模块 tract 的形式提供 category **Nest** 的 Python 实现，并说明 tract 与 cute 的相容性。Python 实现位于 Git 仓库：https://github.com/ColfaxResearch/layout-categories。

第 2 章是 layout 及其代数的综合参考。该章严格定义 layout 及其支持的操作，并建立这些操作的基本性质。章节中包含大量示例，对工程实践者可能很有帮助。

第 3 章提出一种处理 tractable layout 的新数学框架。特别地，我们把 layout 及其代数与 category 和 operad 理论联系起来。该章内容本身具有独立的数学价值，也有实践价值，因为它为可视化 layout 和计算各种 layout 操作提供了新框架。

第 4 章使用第 3 章建立的框架，给出计算 tractable layout A 与 B 的 composite 的算法，并通过大量示例说明该 composition 算法。

## 1.3 相关工作

虽然本文工作本质上属于理论研究，但其动机来自 GPU 编程中的实际应用，尤其是 CUTLASS。需要强调的是，本文建立的理论与实现无关：它不依赖实践中用于实现 layout 的具体编程语言或运行时系统。不过，处理具体实现时仍会出现某些实践问题。例如，CUTLASS 区分编译期常量（静态变量）和运行时值（动态变量），这些信息使编译器能够在代码生成期间进行优化。此类特定于实现的细节虽然对性能很重要，但不属于本文数学框架的范围。进一步讨论可参阅 CuTe 文档 [5]。

我们为 layout 建立的数学框架与计算机科学和数学的多个领域相联系。下面简要回顾 GPU 编程及相邻领域的相关工作，为本文贡献提供更广阔的背景。

• CUTLASS 的应用。CUTLASS 的先进应用包括 FlashAttention [7, 21]、EVT [4] 和 SonicMoE [10]。希望深入了解 CUTLASS 和 CuTe 实践用法的读者，可阅读 NVIDIA [3, 24, 25] 与 Colfax Research [18, 20, 17, 19] 提供的综合教程系列；这些教程介绍如何使用这些库进行 GPU 编程。

• Data layout 优化。Data layout 优化技术通过仔细设计张量在内存中的存储方式，改善 cache locality 和内存访问模式 [30, 8, 15, 11, 22]。内存带宽经常成为 GPU 的瓶颈，因此选择高效的内存存储和访问模式对 GPU 性能至关重要。

• 现代 layout 系统。CuTe [5, 6, 16] 和 Triton Linear Layout [14, 32] 等 layout 系统，已经成为管理张量计算中内存存储和访问的行业标准。Triton linear layout 以 $\mathbb{F}_2$ 线性代数为基础，并从 F-linear operator 的复合继承 compositional structure。它们还天然兼容 layout swizzle，而 swizzle 通常无法表示成 CuTe layout。另一方面，这类 layout 的表达能力不如 CuTe layout，因为其 size 和 cosize 必须等于 2 的幂，无法表达乘以非 2 次幂整数等变换。最近有研究表明，这两个 layout 系统都可以用整数集合关系表达 [1]。这为共同处理 CuTe layout、Triton linear layout 以及具有非矩形 shape 等更一般的 layout 提供了基础。

• 多面体编译。多面体模型 [28, 29, 26] 提供了一套数学框架，用于分析和变换具有 affine 边界和数组访问的循环嵌套。该模型的主要抽象，是把迭代空间表示成某个多面体中的整数点集合。这种形式允许执行复杂的循环变换，在保持程序语义的同时优化 locality 和并行性。Pluto [2]、Polly [9] 和 Tensor Comprehensions [27] 等工具利用多面体技术自动生成优化代码。

• 张量 contraction/decomposition。Tensor contraction [23, 31, 12] 把矩阵乘法推广到更高秩张量，在机器学习和科学计算中无处不在。高效实现 tensor contraction，依赖对 contraction 顺序和中间张量 layout 作出最优选择。

## 1.4 实现

本节说明如何在 NVIDIA CuTe DSL 中处理 layout，并用 cute 表示该实现。Git 仓库 https://github.com/ColfaxResearch/layout-categories 以 Python 模块 tract 的形式提供了本文范畴论框架的实现。下面展示 cute 与 tract 的相容性。

1. 构造 tuple 与 nested tuple：在 Python 中按如下方式构造 tuple 和 nested tuple。

```txt
1 S = (2,2,2)
2 T = ((2,2),(5,5))
3 U = ((2,2),4,(9,(3,3))) 
```

注意，如果要构造长度为 1 的 tuple，必须在条目后包含逗号。例如，

```python
1 S = (10,)
2 T = (10) 
```

返回

```txt
1 S = (10,)
2 T = 10 
```

2. 构造 layout 与 morphism：在 cute 中按如下方式构造 layout

$$
L = S: D
$$

：

```txt
L = cute.make_layout(shape=S, stride=D) 
```

例如，

```python
A = cute.make_layout(shape=((4,4),4), stride=((16,1),4))
B = cute.make_layout(shape=(8,64), stride=(64,1))
C = cute.make_layout(shape=100, stride=2) 
```

返回

```matlab
A = ((4,4),4):(16,1),4)
B = (8,64):(64,1)
C = 100:2 
```

在 tract 中按如下方式构造 nested tuple morphism

$$
S \xrightarrow [ \alpha ]{f} T
$$

：

```python
f = tract.make_morphism(domain=S, codomain=T, map_=alpha) 
```

例如，

```txt
f = tract.make_morphism(domain=(4,4), codomain=(4,2,4), map_=(1,3))
g = tract.make_morphism(domain=(2,2,2,2), codomain=(2,2,2,2), map_=(1,0,4,2))
h = tract.make_morphism(domain=(16,(4,4),(4,4)), codomain=(16,4,4), map_=(1,2,0,3,0)) 
```

返回

```txt
f = (4,4)--(1,3)-->(4,2,4)
g = (2,2,2,2)--(1,0,4,2)-->(2,2,2,2)
h = (16,(4,4),(4,4))--(1,2,0,3,0)-->(16,4,4) 
```

注意，在 tract 中指定映射时，使用符号 0 而不是 ∗。

3. 在 tractable layout 与 morphism 之间转换：如果 L 是 layout，可以使用以下调用检查 L 是否 tractable：

```javascript
tract.is_tractable(L) 
```

例如，

```txt
A = cute.make_layout(shape=(2,2,2), stride=(1,2,4))
B = cute.make_layout(shape=(2,2,2), stride=(1,7,4))
A_is_tractable = tract.is_tractable(A)
B_is_tractable = tract.is_tractable(B) 
```

返回

```txt
A = (2,2,2):(1,2,4)
B = (2,2,2):(1,7,4)
A_is_tractable = True
B_is_tractable = False 
```

如果 L 是 tractable layout，可以使用以下调用构造 standard representation $f_L$：

```txt
tract.compute_morphism(L) 
```

例如，

```python
L = cute.make_layout(shape=(2,2,2), stride=(1,2,4))
f_L = tract.compute_morphism(L) 
```

返回

```txt
L = (2, 2, 2) : (1, 2, 4)
f_L = (2, 2, 2) -- (1, 2, 3) --> (2, 2, 2) 
```

如果 f 是 nested tuple morphism，可以使用以下调用构造 f 编码的 layout $L_f$：

```txt
tract.compute_layout(f) 
```

例如，

```python
f = tract.make_morphism(domain=((5,5),8), codomain=(5,8,5), map_=(1,3,2))
L_f = tract.compute_layout(f) 
```

## 返回

```txt
f = ((5,5),8)--(1,3,2)-->(5,8,5)
L_f = ((5,5),8):(1,40),5) 
```

4. Composition：当该操作定义良好时，它从一对 layout A 和 B 产生 layout `B ◦ A`。精确定义参见定义 2.3.7.1。在 cute 中可通过以下调用计算 composition `B ◦ A`：

```javascript
cute.composition(B,A) 
```

例如，运行

```python
A = cute.make_layout(shape=((4,4),4), stride=((16,1),4))
B = cute.make_layout(shape=(8,64), stride=(64,1))
B_o_A = cute.composition(B,A) 
```

返回

```txt
A = ((4, 4), 4): ((16, 1), 4)
B = (8, 64): (64, 1)
B_o_A = ((4, 4), (2, 2)): ((2, 64), (256, 1)) 
```

如果 f 和 g 是可复合的 nested tuple morphism，可以在 tract 中使用以下调用计算 composition `g ◦ f`：

```javascript
tract.compose(f,g) 
```

例如，

```python
f = tract.make_morphism(domain=((2,2),(2,2)), codomain=((2,2,2),(2,2,2)), map_=(3,2,6,5))
g = tract.make_morphism(domain=((2,2,2),(2,2,2)), codomain=(2,2,2,2), map_=(1,0,2,0,3,4))
g_o_f = tract.compose(f,g) 
```

返回

```txt
f = ((2,2),(2,2))--(3,2,6,5)-->(2,2,2),(2,2,2))
g = ((2,2,2),(2,2,2))--(1,0,2,0,3,4)-->(2,2,2,2)
g_o_f = ((2,2),(2,2))--(2,0,4,3)-->(2,2,2,2) 
```

5. Coalesce：该操作从 layout A 产生 layout `coal(A)`。细节参见定义 2.3.3.1。在 cute 中可通过以下调用计算 `coal(A)`：

```txt
cute.coalesce(A) 
```

例如，

```txt
A = cute.make_layout(shape = ((2,2), (2,2), (5,5)), stride = ((1,2), (16,32), (64,640)))
coal_A = cute.coalesce(A) 
```

返回

```txt
A = ((2,2), (2,2), (5,5)): ((1,2), (16,32), (64,640))
coal_A = (4,20,5): (1,16,640) 
```

还有 relative coalesce 操作 `A ↦ coal(A,S)`，它额外接收一个 nested tuple S 作为输入，A 的 shape 是 S 的 refinement。细节参见定义 2.3.4.7。在 cute 中可通过以下调用计算 `coal(A,S)`：

```txt
A = cute.make_layout(shape = ((2,2),(3,3),(5,5)), stride = ((1,2),(4,12),(36,180)))
S = ((2,2),9,25)
coal_A_over_S = cute.coalesce(A,target_profile=S) 
```

返回

```python
A = ((2,2), (3,3), (5,5)): ((1,2), (4,12), (36,180))
S = ((2,2), 9, 25)
coal_A_over_S = ((2,2), 9, 25): ((1,2), 4, 36) 
```

如果 f 是 nested tuple morphism，可以形成 `coal(f)`。细节参见定义 3.2.6.11。在 tract 中可通过以下调用计算 `coal(f)`：

```txt
tract.coalesce(f) 
```

例如，

```txt
f = tract.make_morphism(domain=(2,2,10,10), codomain = (2,2,2,10,10), map_=(1,2,4,5))
coal_f = tract.coalesce(f) 
```

返回

```txt
f = (2, 2, 10, 10) -- (1, 2, 4, 5) -- > (2, 2, 2, 10, 10)
coal_f = (4, 100) -- (1, 3) -- > (4, 2, 100) 
```

6. Complement：当该操作定义良好时，它从 layout A 和正整数 N 产生 layout `comp(A,N)`。细节参见定义 2.3.6.5。在 cute 中可通过以下调用计算 `comp(A,N)`：

```javascript
cute.complement(A,N) 
```

例如，

```txt
A = cute.make_layout(shape = ((2,2),(2,2)), stride = ((8,2),(64,256)))
comp_A = cute.complement(A,4096) 
```

返回

```matlab
A = ((2,2),(2,2)):((8,2),(64,256))
comp_A = (2,2,4,2,8):(1,4,16,128,512) 
```

如果 f 是 nested tuple morphism，就可以形成 f 的 complement f<sup>c</sup>。细节参见定义 3.2.6.17。在 tract 中可通过以下调用计算 f<sup>c</sup>：

```javascript
tract.complement(f) 
```

例如，

```python
f = tract.make_morphism(domain=(2,2), codomain=(2,5,2,5), map_=(1,3))
comp_f = tract.complement(f) 
```

返回

```txt
f = (2,2)--(1,3)-->(2,5,2,5)
comp_A = (5,5)--(2,4)-->(2,5,2,5) 
```

7. Logical Division：当该操作定义良好时，它从一对 layout A 和 B 产生 layout `A ⊘ B`。细节参见定义 2.3.8.1。在 cute 中可通过以下调用计算 `A ⊘ B`：

```txt
cute.logical_divide(A,B) 
```

例如，

```python
A = cute.make_layout((64,32), stride = (32,1))
B = cute.make_layout((4,4), stride = (1,64))
quotient = cute.logical_divide(A,B) 
```

返回

```txt
A = (64,32):(32,1)
B = (4,4):(1,64)
quotient = ((4,4),(16,8)):(32,1),(128,4)) 
```

如果 f 和 g 是 nested tuple morphism，且 g 整除 f，就可以形成 logical division `f ⊘ g`。细节参见定义 3.2.6.23。在 tract 中可通过以下调用计算 `f ⊘ g`：

```javascript
tract.logical_divide(f,g) 
```

例如，

```txt
f = tract.make_morphism(domain=(4,8,4,8), codomain=(4,8,4,8), map_=(1,2,3,4))
g = tract.make_morphism(domain=(4,4), codomain=(4,8,4,8), map_=(1,3))
quotient = tract.logical_divide(f,g) 
```

返回

```txt
f = (4,8,4,8)--(1,2,3,4)-->(4,8,4,8)
g = (4,4)--(1,3)-->(4,8,4,8)
quotient = ((4,4),(8,8))--(1,3,2,4)-->(4,8,4,8) 
```

8. Logical Product：当该操作定义良好时，它从一对 layout A 和 B 产生 layout `A ⊗ B`。细节参见定义 2.3.9.1。在 cute 中可通过以下调用计算 `A ⊗ B`：

```javascript
cute.logical_product(A,B) 
```

例如，运行

```python
A = cute.make_layout((3,10,10), stride = (200,1,20))
B = cute.make_layout((2,2), stride = (1,2))
product = cute.logical_product(A,B) 
```

返回

```txt
A = (3,10,10):(200,1,20)
B = (2,2):(1,2)
product = ((3,10,10),(2,2)):((200,1,20),(10,600)) 
```

如果 f 和 g 是 nested tuple morphism，且 f 和 g 为 product admissible，就可以形成 logical product $f\otimes g$。细节参见定义 3.2.6.28。在 tract 中可通过以下调用计算 `f ⊗ g`：

```javascript
tract.logical_product(f,g) 
```

例如，

```txt
f = tract.make_morphism(domain=(2,2), codomain=(2,2,5,5), map_=(1,2))
g = tract.make_morphism(domain=(5,5), codomain=(5,5), map_=(2,1))
product = tract.logical_product(f,g) 
```

返回

```txt
f = (2,2)--(1,2)-->(2,2,5,5)
g = (5,5)--(2,1)-->(5,5)
product = ((2,2),(5,5))--(1,2,4,3)-->(2,2,5,5) 
```

## 1.5 记号

$\mathbb{Z}=\{\dots,-1,0,1,2,\dots\}$。$\mathbb{N}=\{0,1,2,\dots\}$。$\mathbb{Z}_{>0}=\{1,2,\dots\}$。$\mathbb{F}_2=\{0,1\}$，即阶为 2 的有限域。$[0,n)=\{0,\dots,n-1\}$，且 $[0,0)=\varnothing$。$\langle n\rangle=\{1,2,\dots,n\}$，且 $\langle0\rangle=\varnothing$。$\langle n\rangle_*=\{*,1,2,\dots,n\}$。$\delta_i^m=(0,\dots,1,\dots,0)$，即长度为 m、第 i 个条目为 1、其余条目均为 0 的 tuple。$\Sigma_n$ 是 $\langle n\rangle$ 上的对称群。对 tuple $X=(x_1,\dots,x_m)$ 和置换 $\sigma\in\Sigma_m$，$X^\sigma=(x_{\sigma(1)},\dots,x_{\sigma(m)})$。$X\star Y$ 是 X 与 Y 的 flat concatenation。$X^\flat$ 是 nested tuple X 的 flattening。$\text{prof}(X)$ 是 nested tuple X 的 profile。$(X_1,\dots,X_k)$ 是 $X_1,\dots,X_k$ 的嵌套 concatenation。$(X_1,\dots,X_k)_Q$ 是对 profile Q 作 $X_1,\dots,X_k$ 的 Q-substitution。$\text{Tuple}(V)$ 是条目位于集合 V 中的 tuple 集合。$\text{Nest}(V)$ 是条目位于集合 V 中的 nested tuple 集合。$\text{Profile}$ 是 profile 集合。$\text{FlatLayout}$ 是 flat layout 集合。$\text{Layout}$ 是 layout 集合。$B\circ A$ 是 A 与 B 的 composition。$A\oslash B$ 是 A 除以 B 的 logical division。$A\otimes B$ 是 A 与 B 的 logical product。$\textbf{Set}$ 是集合的 category。$\textbf{FinSet}$ 是有限集合的 category。$\textbf{Fin}$ 是由所有 $n\geq0$ 的 $\langle n\rangle$ 张成的 $\textbf{FinSet}$ full subcategory。$\textbf{FinSet}_*$ 是带基点有限集合的 category。$\textbf{Fin}_*$ 是由所有 $n\geq0$ 的 $\langle n\rangle$ 张成的 $\textbf{FinSet}_*$ full subcategory。$\textbf{Tuple}$ 是 tuple 与 tuple morphism 的 category。$\textbf{Nest}$ 是 nested tuple 与 nested tuple morphism 的 category。$\textbf{Ref}$ 是 nested tuple 与 refinement 的 category。$\textbf{Cat}$ 是小 category 与 functor 的 category。

## 第 2 章

# Layout 及其代数

本章旨在提供一套全面且具有数学基础的 layout 理论。第 2.1 节首先建立 flat layout 理论；第 2.2 节介绍 nested tuple 的必要背景；由此，第 2.3 节能够在完全一般的情况下讨论 layout。

## 2.1 Flat layout

本节考察 flat layout。它是 layout 的一个重要子类，其中 shape 和 stride 都是 tuple，而不是更一般的 nested tuple。为了形式化讨论，首先固定与 tuple 相关的记号。

## 2.1.1 Tuple

定义 2.1.1.1。如果 V 是集合，则条目位于 V 中的 tuple 是有限有序列表

$$
X = (x _ {1}, \ldots , x _ {m})
$$

其中对每个 $1\leq i\leq m$，都有 $x_i\in V$。这种 tuple $X=(x_1,\dots,x_m)$ 的长度为

$$
\operatorname{len} (X) = m.
$$

用 Tuple(V) 表示条目位于 V 中的所有 tuple 组成的集合。我们尤其关注 $V=\mathbb{Z}$ 的情况，此时把 $X\in\mathsf{Tuple}(\mathbb{Z})$ 称为整数 tuple。如果 X 是整数 tuple，则 X 的 size 是乘积

$$
\operatorname{size} (X) = x _ {1} \dots x _ {m}.
$$

示例 2.1.1.2。下面给出一些 tuple 及其 length 和 size：

$$
\begin{array}{l l} X = (3, 1 2 8, 1 2 8), & \text { len } (X) = 3, \quad \text { size } (X) = 4 9 1 5 2 \\ X = (5 1 2), & \text { len } (X) = 1, \quad \text { size } (X) = 5 1 2 \\ X = (), & \text { len } (X) = 0, \quad \text { size } (X) = 1 \end{array}
$$

定义 2.1.1.3。如果 $X=(x_1,\dots,x_m)$ 和 $Y=(y_1,\dots,y_n)$ 是 tuple，则记

$$
X \star Y = (x _ {1}, \ldots , x _ {m}, y _ {1}, \ldots , y _ {n})
$$

为 X 与 Y 的 concatenation。

示例 2.1.1.4。如果 $X=(64,32)$ 且 $Y=(8,8,8)$，则

$$
X \star Y = (6 4, 3 2, 8, 8, 8).
$$

注记 2.1.1.5。如果 V 是集合，则集合

$$
\operatorname{Tuple} (V) = \coprod_ {m \geq 0} V ^ {\times m}
$$

由所有条目位于 V 中的 tuple 组成，它是 V 上的自由结合 monoid。其 monoidal product 是 concatenation，单位元是空 tuple `()`。

定义 2.1.1.6。如果 X 和 $X'$ 是 tuple，且存在 tuple $X''$ 满足

$$
X ^ {\prime} \star X ^ {\prime \prime} = X.
$$

就称 $X'$ 整除 X。示例 2.1.1.7。如果 $X'=(81,9)$，$X=(81,9,64,8)$，则 $X'$ 整除 X，因为 tuple $X''=(64,8)$ 满足

$$
X ^ {\prime} \star X ^ {\prime \prime} = X.
$$

定义 2.1.1.8。如果 $X=(x_1,\dots,x_m)$ 是 tuple，$\sigma\in\Sigma_m$ 是置换，则记

$$
X ^ {\sigma} = (x _ {\sigma (1)}, \ldots , x _ {\sigma (m)})
$$

为 σ 对 X 的置换。这规定了 $\Sigma_m$ 在 $\mathbb{Z}^{\times m}$ 上的右作用。

示例 2.1.1.9。如果 $X=(8,16,32,64)$，$\sigma=(12)(34)$，则

$$
X ^ {\sigma} = (1 6, 8, 6 4, 3 2).
$$

记号 2.1.1.10。如果 n 是正整数，记

$$
[ 0, n) = \{0, 1, \dots , n - 1 \},
$$

如果 $S=(s_1,\ldots,s_m)$ 是正整数 tuple，则记

$$
[ 0, S) = [ 0, s _ {1}) \times \dots \times [ 0, s _ {m})
$$

表示满足 $0\leq x_i<s_i$ 的 tuple $(x_1,\ldots,x_m)$ 的集合。

示例 2.1.1.11。如果 $S=(3,2)$，则

$$
[ 0, S) = \{(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1) \}
$$

## 2.1.2 基本定义

固定记号后，现在可以定义 flat layout。

定义 2.1.2.1。flat layout 是一对

$$
\begin{array}{l} L = S: D \\ \qquad = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m}) \end{array}
$$

它由一个正整数 tuple

$$
\begin{array}{c} \text {shape} (L) = S \\ = (s _ {1}, \ldots , s _ {m}) \end{array}
$$

以及一个非负整数 tuple 组成；前者称为 L 的 shape，后者为

$$
\begin{array}{c} \text {stride} (L) = D \\ = (d _ {1}, \ldots , d _ {m}) \end{array}
$$

称为 L 的 stride。

注记 2.1.2.2。如果 L 是 flat layout，则根据定义，shape(L) 与 stride(L) 长度相同。注记 2.1.2.3。flat layout 是定义 2.3.1.1 中更一般 layout 的一个示例，因此有时也把 flat layout L 简称为 layout。

示例 2.1.2.4。下面是一些 flat layout 示例：

$$
\begin{array}{l} L _ {1} = (2, 2, 2): (1, 2, 4), \\ L _ {2} = (1 2 8): (5), \\ L _ {3} = (1 6, 1 2, 5 1 2, 5 1 2): (0, 0, 1, 5 1 2), \\ L _ {4} = (6, 1, 1 2, 2, 2): (2, 0, 1 2, 1 4 4, 1), \\ L _ {5} = (): (). \end{array}
$$

示例 2.1.2.5。可以把 layout $L=(8):(5)$ 描绘为

<table><tr><td>0</td><td>5</td><td>10</td><td>15</td><td>20</td><td>25</td><td>30</td><td>35</td></tr></table>

并把 layout $L=(3,5):(2,10)$ 描绘为

<table><tr><td>0</td><td>10</td><td>20</td><td>30</td><td>40</td></tr><tr><td>2</td><td>12</td><td>22</td><td>32</td><td>42</td></tr><tr><td>4</td><td>14</td><td>24</td><td>34</td><td>44</td></tr></table>

注记 2.1.2.17 将精确定义这些图表示相应 layout 的含义。

flat layout 最重要的示例或许是 column-major 和 row-major layout，下面对其作出定义。

定义 2.1.2.6。假设

$$
L = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m})
$$

是 flat layout。如果

$$
d _ {i} = s _ {1} \cdot \cdot \cdot s _ {i - 1}
$$

对每个 $1\leq i\leq m$ 都成立，就称 L 为 column-major。如果

$$
d _ {i} = s _ {i + 1} \dots s _ {m}.
$$

对每个 $1\leq i\leq m$ 都成立，就称 L 为 row-major。

示例 2.1.2.7。layout

$$
L = (3, 4): (1, 3) = \begin{array}{c c c c} \hline 0 & 3 & 6 & 9 \\ \hline 1 & 4 & 7 & 1 0 \\ \hline 2 & 5 & 8 & 1 1 \\ \hline \end{array}
$$

是 column-major，而 layout

$$
L = (3, 4): (4, 1) = \quad \begin{array}{c c c c} \hline 0 & 1 & 2 & 3 \\ \hline 4 & 5 & 6 & 7 \\ \hline 8 & 9 & 1 0 & 1 1 \\ \hline \end{array}
$$

是 row-major。这些图清楚说明了术语的由来：如果 L 是 rank 2 的 column-major layout，则 L 的列连续；如果 L 是 rank 2 的 row-major layout，则 L 的行连续。

示例 2.1.2.8。以下 layout

$$
\begin{array}{l} L _ {1} = (2, 2, 2, 2, 2): (1, 2, 4, 8, 1 6) \\ L _ {2} = (3, 1 2 8, 1 2 8): (1, 3, 3 8 4) \\ L _ {3} = (6 4): (1) \end{array}
$$

是 column-major，而以下 layout

$$
\begin{array}{l} L _ {4} = (2, 2, 2, 2, 2): (1 6, 8, 4, 2, 1) \\ L _ {5} = (3, 1 2 8, 1 2 8): (1 6 3 8 4, 1 2 8, 1) \\ L _ {6} = (6 4): (1) \end{array}
$$

是 row-major。

看过一些示例后，下面定义 flat layout 的若干重要属性。

定义 2.1.2.9。假设 $L=(s_1,\ldots,s_m):(d_1,\ldots,d_m)$ 是 flat layout。

• L 的 rank 为

$$
\operatorname{rank} (L) = m.
$$

• L 的 size 为

$$
\operatorname{size} (L) = \prod_ {i = 1} ^ {m} s _ {i}.
$$

• L 的 cosize 为

$$
\operatorname{cosize} (L) = 1 + \sum_ {i = 1} ^ {m} \left(s _ {i} - 1\right) \cdot d _ {i}.
$$

• 对任意 $1\leq i\leq\mathsf{rank}(L)$，L 的第 i 个 mode 是一对

$$
\operatorname{mode} _ {i} (L) = s _ {i}: d _ {i}.
$$

示例 2.1.2.10。layout

$$
L = (6 4, 3 2): (1, 1 2 8)
$$

满足 $\mathsf{rank}(L)=2$、$\mathsf{size}(L)=2048$、`cosize(L)=4032`。L 的 mode 为

$$
\begin{array}{l} \text {mode} _ {1} (L) = 6 4: 1 \\ \text {mode} _ {2} (L) = 3 2: 1 2 8. \end{array}
$$

示例 2.1.2.11。layout

$$
L = (3, 8, 8, 8): (1, 3, 2 4, 1 9 2).
$$

满足 $\mathsf{rank}(L)=4$、$\mathsf{size}(L)=1536$、$\mathsf{cosize}(L)=1536$。layout L 有四个 mode，例如 $\operatorname{mode}_3(L)=8:24$。

示例 2.1.2.12。layout

$$
L = (2, 2, 2, 2, 2): (1 6 0, 8 0, 4 0, 2 0, 1 0).
$$

满足 $\mathsf{rank}(L)=5$、$\mathsf{size}(L)=32$、$\mathsf{cosize}(L)=311$。layout L 有 5 个 mode，例如 $\operatorname{mode}_5(L)=2:10$。

如果 L 是 flat layout，则 L 编码一个 coordinate function $\varphi_L$。L 的 coordinate function 是从多维到一维的变换，通过与 `stride(L)` 取点积得到。回忆一下，如果 $S=(s_1,\ldots,s_m)$ 是正整数 tuple，则

$$
[ 0, S) = [ 0, s _ {1}) \times \dots \times [ 0, s _ {m})
$$

是所有满足 $0\leq x_i<s_i$ 的 tuple $(x_1,\ldots,x_m)$ 组成的集合。特别地，如果 $S=()` 是空 tuple，则 `[0,S)={()}`。

构造 2.1.2.13（Coordinate function）。如果

$$
L = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m})
$$

是 flat layout，则 L 的 coordinate function 是函数

$$
[ 0, \text { shape } (L)) \xrightarrow {\varphi_ {L}} \mathbb {Z}
$$

其定义为

$$
\begin{array}{c} \varphi_ {L} (x _ {1}, \ldots , x _ {m}) = (x _ {1}, \ldots , x _ {m}) \cdot (d _ {1}, \ldots , d _ {m}) \\ = x _ {1} d _ {1} + \dots + x _ {m} d _ {m}. \end{array}
$$

coordinate function $\varphi_L$ 通过 inclusion $[0,\mathsf{cosize}(L))\subset\mathbb{Z}$ 分解，记

$$
[ 0, \operatorname{shape} (L)) \xrightarrow {\varphi_ {L} ^ {\operatorname{cosize} (L)}} [ 0, \operatorname{cosize} (L)) \subset \mathbb {Z}
$$

表示分解后的映射。更一般地，对任意 $N\geq\mathsf{cosize}(L)$，用 $\varphi_L^N$ 表示 $\varphi_L$ 通过 $[0,N)\subset\mathbb{Z}$ 的分解。稍微滥用术语，也把这类映射 $\varphi_L^N$ 称为 L 的 coordinate function。

示例 2.1.2.14。如果 $L=(2,3):(1,5)$，则 coordinate function

$$
\varphi_ {L}: [ 0, 2) \times [ 0, 3) \to \mathbb {Z}
$$

定义为

$$
\begin{array}{l} \varphi_ {L} (0, 0) = (0, 0) \cdot (1, 5) = 0, \\ \varphi_ {L} (1, 0) = (1, 0) \cdot (1, 5) = 1, \\ \varphi_ {L} (0, 1) = (0, 1) \cdot (1, 5) = 5, \\ \varphi_ {L} (1, 1) = (1, 1) \cdot (1, 5) = 6, \\ \varphi_ {L} (0, 2) = (0, 2) \cdot (1, 5) = 1 0, \\ \varphi_ {L} (1, 2) = (1, 2) \cdot (1, 5) = 1 1. \end{array}
$$

示例 2.1.2.15。如果 $L=(2,2):(64,2)$，则 coordinate function

$$
\varphi_ {L}: [ 0, 2) \times [ 0, 2) \to \mathbb {Z}
$$

定义为

$$
\begin{array}{l} \varphi_ {L} (0, 0) = (0, 0) \cdot (6 4, 2) = 0, \\ \varphi_ {L} (1, 0) = (1, 0) \cdot (6 4, 2) = 6 4, \\ \varphi_ {L} (0, 1) = (0, 1) \cdot (6 4, 2) = 2, \\ \varphi_ {L} (1, 1) = (1, 1) \cdot (6 4, 2) = 6 6. \end{array}
$$

示例 2.1.2.16。如果 $E=():()` 是空 layout，则 E 的 coordinate function 是映射

$$
\varphi_ {E}: \left\{\left(\right) \right\} \to \mathbb {Z}
$$

其定义为

$$
\varphi (()) = 0.
$$

注记 2.1.2.17。现在可以精确说明下图

<table><tr><td>0</td><td>10</td><td>20</td><td>30</td><td>40</td></tr><tr><td>2</td><td>12</td><td>22</td><td>32</td><td>42</td></tr><tr><td>4</td><td>14</td><td>24</td><td>34</td><td>44</td></tr></table>

如何描绘 layout $L=(3,5):(2,10)`：grid 的第 `(i,j)` 个单元格标有 L 的 coordinate function 值

$$
\varphi_ {L} (i, j) = (i, j) \cdot (2, 1 0) = 2 i + 1 0 j
$$

。

在实践中，flat layout L 最重要的不变量是其 layout function $\Phi_L$。它通过在 coordinate function

$$
\varphi_ {L}: [ 0, S) \to \mathbb {Z}
$$

之前复合 colexicographic isomorphism

$$
\operatorname{colex} _ {S}: [ 0, S) \to [ 0, \operatorname{size} (S)).
$$

的逆得到。定义 2.1.2.18。假设 $S=(s_1,\ldots,s_m)$ 是正整数 tuple，并回忆

$$
[ 0, S) = [ 0, s _ {1}) \times \dots \times [ 0, s _ {m}).
$$

colexicographic isomorphism 是映射

$$
[ 0, S) \xrightarrow {\operatorname{colex} _ {S}} [ 0, \operatorname{size} (S))
$$

$$
(x _ {1}, \ldots , x _ {m}) \longmapsto \sum_ {i = 1} ^ {m} s _ {1} \dots s _ {i - 1} x _ {i}.
$$

当上下文中的 tuple S 明确时，有时简写 `colex = colex`<sub>S</sub>。逆 colexicographic isomorphism 是映射

$$
[ 0, \operatorname{size} (S)) \xrightarrow {\operatorname{colex} _ {S} ^ {- 1}} [ 0, S)
$$

其定义为

$$
\operatorname{colex} _ {S} ^ {- 1} (x) = \left(x _ {1}, \dots , x _ {m}\right)
$$

其中

$$
x _ {i} = \left\lfloor \frac {x}{s _ {1} \cdots s _ {i - 1}} \right\rfloor \mod s _ {i}.
$$

注意，如果 $S=()` 是空 tuple，则

$$
\operatorname{colex} _ {()}: \{\left(\right) \} \rightarrow \{0 \}
$$

以及

$$
\operatorname{colex} _ {()} ^ {- 1}: \{0 \} \to \{\left(\right) \}
$$

是 canonical isomorphism。

构造 2.1.2.19（Layout function）。如果

$$
L = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m}),
$$

是 flat layout，则 L 的 layout function 是 composite

![image](Imgaes/categorical-foundations-cute-layouts-paper/d1b5cd15b4928b3bb46d139c2c9d1fe8c748623fa688167daf81041f3c677cff.jpg)


显式地，$\Phi_L$ 定义为

$$
\Phi_ {L} (x) = x _ {1} d _ {1} + \dots + x _ {m} d _ {m}
$$

其中

$$
x _ {i} = \left\lfloor \frac {x}{s _ {1} \cdots s _ {i - 1}} \right\rfloor \mod s _ {i}.
$$

layout function $\Phi_L$ 通过 inclusion $[0,\mathsf{cosize}(L))\subset\mathbb{Z}$ 分解，记

$$
[ 0, \operatorname{size} (L)) \xrightarrow {\Phi_ {L} ^ {\operatorname{cosize} (L)}} [ 0, \operatorname{cosize} (L)) \subset \mathbb {Z}
$$

表示分解后的映射。更一般地，对任意 $N\geq\mathsf{cosize}(L)$，用 $\Phi_L^N$ 表示 $\Phi_L$ 通过 $[0,N)\subset\mathbb{Z}$ 的分解。稍微滥用术语，也把这类映射 $\Phi_L^N$ 称为 L 的 layout function。

示例 2.1.2.20。如果 $L=(2,3):(1,5)$，则 layout function

$$
\Phi_ {L}: [ 0, 6) \to \mathbb {Z}
$$

定义为

$$
\begin{array}{l} \Phi_ {L} (0) = (0, 0) \cdot (1, 5) = 0, \\ \Phi_ {L} (1) = (1, 0) \cdot (1, 5) = 1, \\ \Phi_ {L} (2) = (0, 1) \cdot (1, 5) = 5, \\ \Phi_ {L} (3) = (1, 1) \cdot (1, 5) = 6, \\ \Phi_ {L} (4) = (0, 2) \cdot (1, 5) = 1 0, \\ \Phi_ {L} (5) = (1, 2) \cdot (1, 5) = 1 1. \end{array}
$$

示例 2.1.2.21。如果 `L=(2,2):(64,2)`，则 layout function

$$
\Phi_ {L}: [ 0, 4) \to \mathbb {Z}
$$

定义为

$$
\begin{array}{l} \Phi_ {L} (0) = (0, 0) \cdot (6 4, 2) = 0, \\ \Phi_ {L} (1) = (1, 0) \cdot (6 4, 2) = 6 4, \\ \Phi_ {L} (2) = (0, 1) \cdot (6 4, 2) = 2, \\ \Phi_ {L} (3) = (1, 1) \cdot (6 4, 2) = 6 6. \end{array}
$$

示例 2.1.2.22。如果 $L=(4,2,2):(3,3,100)$，则例如 L 的 layout function 满足

$$
\begin{array}{l} \Phi_ {L} (7) = (3, 1, 0) \cdot (3, 3, 1 0 0) = 1 2, \\ \Phi_ {L} (9) = (1, 0, 1) \cdot (3, 3, 1 0 0) = 1 0 3. \end{array}
$$

示例 2.1.2.23。如果 $E=():()` 是空 layout，则

$$
\Phi_ {E}: \{0 \} \to \mathbb {Z}
$$

定义为

$$
\Phi_ {E} (0) = 0.
$$

示例 2.1.2.24。如果 L 是任意 flat layout，则 L 的 layout function $\Phi_L$ 满足

$$
\Phi_ {L} (0) = 0.
$$

注记 2.1.2.25。如果 $S=(s_1,\ldots,s_m)$ 是正整数 tuple，则 colexicographic isomorphism

$$
[ 0, S) \xrightarrow {\operatorname{colex} _ {S}} [ 0, \operatorname{size} (S))
$$

等于 column-major layout

$$
L = (s _ {1}, s _ {2}, \ldots , s _ {m}): (1, s _ {1}, \ldots , s _ {1} \dots s _ {m - 1}).
$$

的 coordinate function $\varphi_L^{\mathsf{cosize}(L)}$。这意味着，如果 flat layout L 是 column-major，则

$$
\begin{array}{l} \Phi_ {L} ^ {\text {cosize} (L)} = \varphi_ {L} ^ {\text {cosize} (L)} \circ \text {colex} _ {\text {shape} (L)} ^ {- 1} \\ \qquad = \varphi_ {L} ^ {\text {cosize} (L)} \circ \left(\varphi_ {L} ^ {\text {cosize} (L)}\right) ^ {- 1} \\ \qquad = \mathsf {i d} _ {[ 0, \text {size} (L))} \end{array}
$$

是 $[0,\mathsf{size}(L))$ 上的 identity map。

注记 2.1.2.26。存在不同 layout $A\neq B$，但 $\Phi_A=\Phi_B$。例如，layout

$$
\begin{array}{l} A = (7, 7): (1, 7) \\ B = (4 9): (1) \end{array}
$$

并不相等，但 $\Phi_A=\Phi_B$。后文将精确刻画两个 flat layout A 和 B 何时具有相同 layout function，参见命题 2.1.4.18。

在继续讨论 layout 操作之前，需要定义 non-degeneracy 概念。

定义 2.1.2.27。假设

$$
L = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m})
$$

是 flat layout。如果对任意 $1\leq i\leq m$ 都有

$$
s _ {i} = 1 \quad \Rightarrow \quad d _ {i} = 0.
$$

就称 L 为 non-degenerate。示例 2.1.2.28。以下 layout

$$
\begin{array}{l} L _ {1} = (4, 1): (1, 0) \\ L _ {2} = (8, 1, 8, 1): (2, 0, 1 6, 0) \end{array}
$$

是 non-degenerate，而以下 layout

$$
\begin{array}{l} L _ {3} = (4, 1): (1, 4) \\ L _ {4} = (8, 1, 8, 1): (2, 1 6, 1 6, 2 5 6) \end{array}
$$

是 degenerate。

观察 2.1.2.29。假设 layout L 为 non-degenerate 并不会真正损失一般性。更精确地说，如果

$$
\begin{array}{c} L = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m}) \\ L ^ {\prime} = (s _ {1}, \ldots , s _ {m}): (d _ {1} ^ {\prime}, \ldots , d _ {m} ^ {\prime}) \end{array}
$$

是具有相同 shape 的 flat layout，并且每当 $s_i>1$ 时都有 $d_i=d_i'$，则 $\varphi_L=\varphi_{L'}$ 且 $\Phi_L=\Phi_{L'}$。特别地，每当 $s_i=1$ 时，都可以自由地令 $d_i=0$，而不会改变 L 的 coordinate function 或 layout function。

## 2.1.3 基本操作

建立 flat layout 的基本词汇后，下面转向其支持的操作。本节定义一些基本操作，后续构造 coalesce、complement 和 composition 等更复杂操作时会用到它们。

## 2.1.3.1 Restriction

如果 L 是 flat layout，把它限制到 L 的一部分 mode 往往很有用。回忆一下，对非负整数 m，记

$$
\langle m \rangle = \{1, \dots , m \}.
$$

定义 2.1.3.1。假设

$$
L = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m})
$$

是 flat layout，并假设

$$
I = \left\{i _ {1} <   \dots <   i _ {k} \right\} \subset \langle m \rangle
$$

是一个子集。把 L 在 I 上的 restriction 定义为 flat layout

$$
L \mid_ {I} = (s _ {i _ {1}}, \dots , s _ {i _ {k}}): (d _ {i _ {1}}, \dots , d _ {i _ {k}}).
$$

示例 2.1.3.2。如果

<table><tr><td>0</td><td>5</td><td>10</td><td>15</td><td>20</td><td>25</td></tr><tr><td>10</td><td>15</td><td>20</td><td>25</td><td>30</td><td>35</td></tr><tr><td>20</td><td>25</td><td>30</td><td>35</td><td>40</td><td>45</td></tr></table>

且 $I=\{2\}$，则

<table><tr><td>0</td><td>5</td><td>10</td><td>15</td><td>20</td><td>25</td></tr></table>

示例 2.1.3.3。如果

$$
L = (3, 8, 8, 8): (1, 3, 2 4, 1 9 2)
$$

且 $I=\{1,2,3\}$，则

$$
L \mid_ {I} = (3, 8, 8): (1, 3, 2 4).
$$

示例 2.1.3.4。如果

$$
L = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m})
$$

是 flat layout，且 $I=\langle m\rangle$，则

$$
L \mid_ {I} = L.
$$

示例 2.1.3.5。如果

$$
L = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m})
$$

是 flat layout，且 $I=\emptyset$ 是空集，则

$$
L \mid_ {I} = (): ()
$$

是空 layout。

## 2.1.3.2 Squeeze

如果 L 是 flat layout，则操作 $L\mapsto\mathsf{squeeze}(L)$ 会移除 L 中所有满足 $s_i=1$ 的 mode $s_i:d_i$。构造 2.1.3.6。假设

$$
L = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m})
$$

是 flat layout，并令

$$
I = \{i \in \langle m \rangle \mid s _ {i} > 1 \}
$$

为对应 shape 条目不等于 1 的索引集合。定义

$$
\operatorname{squeeze} (L) = L \mid_ {I}.
$$

示例 2.1.3.7。如果

$$
L = (6 4, 6 4, 1): (1, 6 4, 0),
$$

则

$$
\operatorname{squeeze} (L) = (6 4, 6 4): (1, 6 4).
$$

示例 2.1.3.8。如果

$$
L = (6 4, 6 4, 1, 3 2, 1): (2 0 4 8, 3 2, 0, 1, 0)
$$

则

$$
\operatorname{squeeze} (L) = (6 4, 6 4, 3 2): (2 0 4 8, 3 2, 1).
$$

示例 2.1.3.9。如果 L 是 flat layout，则

$$
\operatorname{squeeze} (L) = L
$$

当且仅当 `shape(L)` 不包含等于 1 的条目。

示例 2.1.3.10。如果 L 是 flat layout，则

$$
\text { squeeze } (L) = (): ()
$$

是空 layout，当且仅当 `shape(L)` 的所有条目都等于 1。

该构造的一项基本性质是，$L\mapsto\mathsf{squeeze}(L)$ 不会改变 L 的 layout function。

引理 2.1.3.11。如果 L 是 flat layout，则

1. si $z \mathsf { e } ( \mathsf { s q u e e z e } ( L ) ) = \mathsf { s i z e } ( L ) ,$ 

2. $\mathsf{cosize}(\mathsf{squeeze}(L))=\mathsf{cosize}(L)$；

3. $\Phi _ { \mathsf { s q u e e z e } ( L ) } = \Phi _ { L }$ 

证明。令

$$
I = \left\{i _ {1} <   \dots <   i _ {k} \right\} \subset \langle m \rangle
$$

表示满足 $s_{i_j}>1$ 的索引集合，因此

$$
\operatorname{squeeze} (L) = \left(s _ {i _ {1}}, \dots , s _ {i _ {k}}\right): \left(d _ {i _ {1}}, \dots , d _ {i _ {k}}\right).
$$

对第一条结论，计算得

$$
\operatorname{size} (\operatorname{squeeze} (L)) = \prod_ {j = 1} ^ {k} s _ {i _ {j}} = \left(\prod_ {j = 1} ^ {k} s _ {i _ {j}}\right) \cdot \left(\prod_ {\langle m \rangle \backslash I} 1\right) = \prod_ {i = 1} ^ {m} s _ {i} = \operatorname{size} (L).
$$

对第二条结论，计算得

$$
\begin{array}{c} \text {cosize} (\text {squeeze} (L)) = 1 + \sum_ {j = 1} ^ {k} (s _ {i _ {j}} - 1) \cdot d _ {i _ {j}} = 1 + \sum_ {j = 1} ^ {k} (s _ {i _ {j}} - 1) \cdot d _ {i _ {j}} + \left(\sum_ {\langle m \rangle \setminus I} 0\right) \\ = 1 + \sum_ {i = 1} ^ {m} (s _ {i} - 1) \cdot d _ {i} \\ = \text {cosize} (L). \end{array}
$$

对第三条结论，只需证明：从 flat layout 中移除形如 $1:d_i$ 的 mode 不会改变 layout function。假设 $L=(s_1,\ldots,s_m):(d_1,\ldots,d_m)$，并假设某个 $s_i=1$。令

$$
L ^ {\prime} = (s _ {1} ^ {\prime}, \ldots , s _ {m - 1} ^ {\prime}): (d _ {1} ^ {\prime}, \ldots , d _ {m - 1} ^ {\prime})
$$

表示从 L 中移除第 i 个 mode 后所得 flat layout，因此

$$
s _ {j} ^ {\prime} = \left\{ \begin{array}{l l} s _ {j} & j <   i \\ s _ {j + 1} & i \leq j <   m, \end{array} \right. \quad \text { and } \quad d _ {j} ^ {\prime} = \left\{ \begin{array}{l l} d _ {j} & j <   i \\ d _ {j + 1} & i \leq j <   m. \end{array} \right.
$$

L 的 layout function 定义为

$$
\Phi_ {L} (x) = x _ {1} d _ {1} + \dots + x _ {m} d _ {m}
$$

其中 $x_j=\left\lfloor\frac{x}{s_1\cdots s_{j-1}}\right\rfloor\bmod s_j$；$L'$ 的 layout function 定义为

$$
\Phi_ {L ^ {\prime}} (x) = x _ {1} ^ {\prime} d _ {1} ^ {\prime} + \dots + x _ {m - 1} ^ {\prime} d _ {m - 1} ^ {\prime}
$$

其中 $x_j'=\left\lfloor\frac{x}{s_1'\cdots s_{j-1}'}\right\rfloor\bmod s_j'$。可以观察到

$$
x _ {j} ^ {\prime} = \left\{ \begin{array}{l l} x _ {j} & j <   i \\ x _ {j + 1} & i \leq j <   m, \end{array} \right.
$$

又因为 $x_i\in[0,1)$ 必然为 0，所以

$$
\begin{array}{r l} & {\Phi_ {L} (x) = x _ {1} d _ {1} + \dots + x _ {m} d _ {m}} \\ & {\qquad = x _ {1} d _ {1} + \dots + x _ {i - 1} d _ {i - 1} + x _ {i + 1} d _ {i + 1} + \dots + x _ {m} d _ {m}} \\ & {\qquad = x _ {1} ^ {\prime} d _ {1} ^ {\prime} + \dots + x _ {m - 1} ^ {\prime} d _ {m - 1} ^ {\prime}} \\ & {\qquad = \Phi_ {L ^ {\prime}} (x).} \end{array}
$$

## 2.1.3.3 过滤零 stride

如果 L 是 flat layout，则操作 $L\mapsto\mathsf{filter}(L)$ 会移除所有满足 $d_i=0$ 的 mode $s_i:d_i$。定义 2.1.3.12。假设

$$
L = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m})
$$

是 flat layout，并令

$$
I = \{i \in \langle m \rangle \mid d _ {i} > 0 \}
$$

为对应 stride 条目不等于 0 的索引集合。定义

$$
\operatorname{filter} (L) = L \mid_ {I}.
$$

示例 2.1.3.13。如果

$$
L = (6 4, 8, 8, 1 2 8): (8, 1, 0, 5 1 2)
$$

则

$$
\operatorname{filter} (L) = (6 4, 8, 1 2 8): (8, 1, 5 1 2).
$$

示例 2.1.3.14。如果

$$
L = (3, 2): (1 2, 0) = \quad \begin{array}{c c} \hline 0 & 0 \\ \hline 1 2 & 1 2 \\ \hline 2 4 & 2 4 \\ \hline \end{array}
$$

则

$$
\operatorname{filter} (L) = (3): (1 2) = \quad \begin{array}{c} \framebox {0} \\ \framebox {1 2} \\ \framebox {2 4} \end{array}
$$

示例 2.1.3.15。如果

$$
L = (3, 8, 8, 8): (1 6, 0, 0, 0)
$$

则

$$
\operatorname{filter} (L) = (3): (1 6).
$$

示例 2.1.3.16。如果 L 是 flat layout，则

$$
\operatorname{filter} (L) = L
$$

当且仅当 `stride(L)` 的所有条目都非零。

示例 2.1.3.17。如果 L 是 flat layout，则

$$
\operatorname{filter} (L) = (): ()
$$

是空 layout，当且仅当 `stride(L)` 的所有条目都等于 0。

## 2.1.3.4 Permute

回忆一下，如果 $X=(x_1,\dots,x_m)$ 是长度为 m 的 tuple，且 $\sigma\in\Sigma_m$ 是置换，则记

$$
X ^ {\sigma} = (x _ {\sigma (1)}, \ldots , x _ {\sigma (m)}).
$$

为 σ 对 X 的置换。

定义 2.1.3.18。如果 $L=(s_1,\ldots,s_m):(d_1,\ldots,d_m)$ 是 rank m 的 flat layout，且 $\sigma\in\Sigma_m$ 是置换，则定义

$$
\begin{array}{l} L ^ {\sigma} = \mathsf {s h a p e} (L) ^ {\sigma}: \mathsf {s t r i d e} (L) ^ {\sigma} \\ \qquad = (s _ {\sigma (1)}, \ldots , s _ {\sigma (m)}): (d _ {\sigma (1)}, \ldots , d _ {\sigma (m)}). \end{array}
$$

示例 2.1.3.19。如果

$$
L = (4, 2): (1 2, 2) =
$$

<table><tr><td>0</td><td>2</td></tr><tr><td>12</td><td>14</td></tr><tr><td>24</td><td>26</td></tr><tr><td>36</td><td>38</td></tr></table>

且 $\sigma=(12)\in\Sigma_2$ 是 transposition，则

$$
L ^ {\sigma} = (2, 4): (2, 1 2) =
$$

<table><tr><td>0</td><td>12</td><td>24</td><td>36</td></tr><tr><td>2</td><td>14</td><td>26</td><td>38</td></tr></table>

是转置后的 layout。

示例 2.1.3.20。如果

$$
L = (1 5, 1 2, 1 0): (2 4 0, 1, 2 4)
$$

且 $\sigma=(12)\in\Sigma_3$，则

$$
L ^ {\sigma} = (1 2, 1 5, 1 0): (1, 2 4 0, 2 4).
$$

示例 2.1.3.21。如果

$$
L = (2, 2, 2, 2, 2): (1, 2, 4, 8, 1 6)
$$

且 $\sigma=(15)(324)\in\Sigma_5$，则

$$
L ^ {\sigma} = (2, 2, 2, 2, 2): (1 6, 8, 2, 4, 1).
$$

示例 2.1.3.22。如果

$$
L = (s, \dots , s): (d, \dots , d)
$$

是所有 mode 都相等的 flat layout，则对任意 $\sigma\in\Sigma_m$，都有

$$
L ^ {\sigma} = L.
$$

## 2.1.3.5 Sort

如果 L 是 flat layout，通常把 L 置换成其 mode 按以下意义递增的形式会很有用。

定义 2.1.3.23。在整数对 $s:d$ 上定义线性序

$$
s: d \preceq s ^ {\prime}: d ^ {\prime} \quad \Leftrightarrow \quad \begin{array}{c} d <   d ^ {\prime}, \text {or} \\ d = d ^ {\prime} \text {and} s \leq s ^ {\prime}. \end{array}
$$

示例 2.1.3.24。有

$$
5: 8 \preceq 4: 1 2 \preceq 5: 1 2.
$$

定义 2.1.3.25。假设 L 是 flat layout。如果对任意 $1\leq i<\mathsf{rank}(L)$ 都有

$$
\operatorname{mode} _ {i} (L) \preceq \operatorname{mode} _ {i + 1} (L).
$$

就称 L 已排序。示例 2.1.3.26。以下 layout

$$
\begin{array}{l} L _ {1} = (1 2 8, 6 4, 2, 2): (1, 1 2 8, 8 1 9 2, 1 6 3 8 4) \\ L _ {2} = (2, 2, 2): (1, 1, 1) \end{array}
$$

已排序，而以下 layout

$$
\begin{array}{l} L _ {3} = (2, 4, 8, 1 6): (6 4, 1, 2, 4) \\ L _ {4} = (5, 3 2, 1 6): (1, 5, 5) \end{array}
$$

未排序。

示例 2.1.3.27。空 layout $E=():()` 已排序。

示例 2.1.3.28。如果

$$
L = (s _ {1}, \dots , s _ {m}): (0, \dots , 0)
$$

是 `stride(L)` 所有条目都等于 0 的 flat layout，则 L 已排序，当且仅当

$$
s _ {1} \leq s _ {2} \leq \dots \leq s _ {m}.
$$

flat layout L 是否已排序，与其 layout function $\Phi_L$ 的行为密切相关，如下一个引理所述。

引理 2.1.3.29。假设 L 是 flat layout。如果 $\Phi_L$ 非递减，则 L 已排序。

证明。证明其逆否命题。假设 L 未排序。我们将证明：在 $\Phi_L$ 的 domain 中存在某些 $x\leq y$，满足 $\Phi_L(x)>\Phi_L(y)$。如果存在某个 $1\leq i<m$ 使 $d_i>d_{i+1}$，则可以令

$$
x = \prod_ {j <   i} s _ {j}, \quad \text { and } \quad y = \prod_ {j <   i + 1} s _ {j},
$$

此时 $x<y$，但

$$
\begin{array}{r l} \Phi_ {L} (x) & = (0, \ldots , 1, 0, \ldots , 0) \cdot (d _ {1}, \ldots , d _ {i}, d _ {i + 1}, \ldots , d _ {m}) \\ & = d _ {i} \\ & > d _ {i + 1} \\ & = (0, \ldots , 0, 1, \ldots , 0) \cdot (d _ {1}, \ldots , d _ {m}) \\ & = \Phi_ {L} (y). \end{array}
$$

另一方面，如果存在某个 $1\leq i<m$ 使 $d_i=d_{i+1}$ 且 $s_i>s_{i+1}$，可以令

$$
x = \left(s _ {i} - 1\right) \left(\prod_ {j <   i} s _ {j}\right), \quad \text { and } \quad y = \left(s _ {i + 1} - 1\right) \left(\prod_ {j <   i + 1} s _ {j}\right),
$$

此时 $x<y$，但

$$
\begin{array}{l} \Phi_ {L} (x) = (0, \ldots , s _ {i} - 1, 0, \ldots , 0) \cdot (d _ {1}, \ldots , d _ {i}, d _ {i + 1}, \ldots , d _ {m}) \\ \qquad = (s _ {i} - 1) d _ {i} \\ \qquad > (s _ {i + 1} - 1) d _ {i} \\ \qquad = (s _ {i + 1} - 1) d _ {i + 1} \\ \qquad = (0, \ldots , 0, s _ {i + 1} - 1, \ldots , 0) \cdot (d _ {1}, \ldots , d _ {m}) \\ \qquad = \Phi_ {L} (y). \end{array}
$$

因此，$\Phi_L$ 不是非递减的。

注记 2.1.3.30。前述引理的逆命题不成立。例如，flat layout

$$
L = (3, 5, 7): (1, 1, 1)
$$

已排序，但

$$
\Phi_ {L} (7) = (0, 2, 0) \cdot (1, 1, 1) = 2
$$

严格大于

$$
\Phi_ {L} (1 6) = (0, 0, 1) \cdot (1, 1, 1) = 1.
$$

如果 L 是 flat layout，可以置换 L 的 mode，得到已排序 layout $\mathsf{sort}(L)$。

构造 2.1.3.31。假设

$$
L = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m})
$$

是 flat layout。在 $\langle m\rangle$ 上定义线性序 ⪯：当以下条件成立时，$i\preceq j$：

1. $\operatorname{mode}_i(L)\preceq\operatorname{mode}_j(L)$；

2. 如果 $\operatorname{mode}_i(L)=\operatorname{mode}_j(L)$，则 $i\leq j$。

令 $\sigma\in\Sigma_m$ 为与 $\langle m\rangle$ 上线性序 ⪯ 关联的置换。把 `sort(L)` 定义为 σ 对 L 的置换：

$$
\operatorname{sort} (L) = L ^ {\sigma}.
$$

示例 2.1.3.32。如果

$$
L = (2, 4, 8, 1 6): (6 4, 1, 2, 4)
$$

则

$$
\operatorname{sort} (L) = (4, 8, 1 6, 2): (1, 2, 4, 6 4).
$$

示例 2.1.3.33。如果

$$
L = (5, 3 2, 1 6): (1, 5, 5)
$$

则

$$
\operatorname{sort} (L) = (5, 1 6, 3 2): (1, 5, 5).
$$

示例 2.1.3.34。如果 L 已排序，则 $\mathsf{sort}(L)=L$。特别地，这意味着 $\mathsf{sort}(-)$ 是幂等操作：

$$
\operatorname{sort} (\operatorname{sort} (L)) = \operatorname{sort} (L).
$$

观察 2.1.3.35。如果 L 是 flat layout，通常有 $\Phi_{\mathsf{sort}(L)}\neq\Phi_L$。不过，layout function $\Phi_L$ 与 $\Phi_{\mathsf{sort}(L)}$ 始终具有相同的像。为说明这一点，写成

$$
\begin{array}{c} L = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m}), \text {and} \\ \mathsf {s o r t} (L) = (s _ {\sigma (1)}, \ldots , s _ {\sigma (m)}): (d _ {\sigma (1)}, \ldots , d _ {\sigma (m)}) \end{array}
$$

其中 $\sigma\in\Sigma_m$ 是某个置换。如果整数 n 位于 $\Phi_L$ 的像中，则存在 tuple $(x_1,\ldots,x_m)\in\prod_{i=1}^m[0,s_i)$，使得

$$
x _ {1} d _ {1} + \dots + x _ {m} d _ {m} = n
$$

此时 tuple $(x_{\sigma(1)},\ldots,x_{\sigma(m)})\in\prod_{i=1}^m[0,s_{\sigma(i)})$ 满足

$$
x _ {\sigma (1)} d _ {\sigma (1)} + \dots + x _ {\sigma (m)} d _ {\sigma (m)} = n.
$$

这证明 $\mathsf{Image}(\Phi_{\mathsf{sort}(L)})\subseteq\mathsf{Image}(\Phi_L)$；反向 inclusion 可类似证明。

## 2.1.3.6 Concatenate

回忆一下，如果 $X=(x_1,\dots,x_m)$ 和 $Y=(y_1,\dots,y_n)$ 是 tuple，则 X 与 Y 的 concatenation 是 tuple

$$
X \star Y = (x _ {1}, \dots , x _ {m}, y _ {1}, \dots , y _ {n}).
$$

该定义自然扩展到 flat layout 的 concatenation。

定义 2.1.3.36。假设

$$
\begin{array}{c} {L _ {1} = S _ {1}: D _ {1}} \\ {L _ {2} = S _ {2}: D _ {2}} \end{array}
$$

是 flat layout，则 $L_1$ 与 $L_2$ 的 concatenation 是 flat layout

$$
L _ {1} \star L _ {2} = S _ {1} \star S _ {2}: D _ {1} \star D _ {2}.
$$

flat layout 的 concatenation 满足结合律。因此更一般地，如果 $L_1,\ldots,L_k$ 是 flat layout，就可以形成 concatenation

$$
L _ {1} \star \dots \star L _ {k}.
$$

示例 2.1.3.37。如果 $L_1=(7,2):(2,1)$，$L_2=(3,3,3):(0,10,30)$，则

$$
L _ {1} \star L _ {2} = (7, 2, 3, 3, 3): (2, 1, 0, 1 0, 3 0).
$$

示例 2.1.3.38。如果 $E=():()` 是空 layout，则对任意 flat layout L，都有

$$
L \star E = L = E \star L.
$$

观察 2.1.3.39。假设

$$
L = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m})
$$

是 flat layout。如果记

$$
L _ {i} = (s _ {i}): (d _ {i}),
$$

则可以把 L 写成 concatenation

$$
L = L _ {1} \star \dots \star L _ {m}.
$$

如果 $L_1,\ldots,L_k$ 是 flat layout，则 concatenation $L_1\star\cdots\star L_k$ 的 layout function 由 $L_1,\ldots,L_k$ 的 layout function 按如下方式决定。

命题 2.1.3.40。假设 $L_1,\ldots,L_k$ 分别是 shape 为 $S_1,\ldots,S_k$、size 为 $N_1,\ldots,N_k$ 的 flat layout。则 $L_1\star\cdots\star L_k$ 的 coordinate function

$$
\left[ 0, S _ {1} \star \dots \star S _ {k}\right) \xrightarrow {\varphi_ {L _ {1} \star \cdots \star L _ {k}}} \mathbb {Z}
$$

等于 composite

$$
[ 0, S _ {1} \star \dots \star S _ {k}) \xrightarrow {\cong} [ 0, S _ {1}) \times \dots \times [ 0, S _ {k}) \xrightarrow {\varphi_ {L _ {1}} + \cdots + \varphi_ {L _ {k}}} \mathbb {Z},
$$

$$
X _ {1} \star \dots \star X _ {k} \longleftrightarrow (X _ {1}, \dots , X _ {k})
$$

而 $L_1\star\cdots\star L_k$ 的 layout function

$$
\left[ 0, N _ {1} \dots N _ {k}\right) \xrightarrow {\Phi_ {L _ {1} \star \cdots \star L _ {k}}} \mathbb {Z}
$$

等于 composite

$$
[ 0, N _ {1} \dots N _ {k}) \xrightarrow {\mathsf {c o l e x} _ {(N _ {1} , \ldots , N _ {k})} ^ {- 1}} [ 0, N _ {1}) \times \dots \times [ 0, N _ {k}) \xrightarrow {\Phi_ {L _ {1}} + \cdots + \Phi_ {L _ {k}}} \mathbb {Z}.
$$

证明。对每个 $1\leq i\leq k$，记 $L_i=S_i:D_i$。第一条结论成立，因为如果

$$
X \in [ 0, S _ {1} \star \dots \star S _ {k})
$$

在 canonical isomorphism

$$
X _ {1} \star \dots \star X _ {k} \in [ 0, S _ {1}) \times \dots \times [ 0, S _ {k})
$$

下对应于

$$
\begin{array}{r l} & {\varphi_ {L _ {1} \star \dots \star L _ {k}} (X) = X \cdot (D _ {1} \star \dots \star D _ {k})} \\ & {\qquad = (X _ {1} \star \dots \star X _ {k}) \cdot (D _ {1} \star \dots \star D _ {k})} \\ & {\qquad = (X _ {1} \cdot D _ {1}) + \dots + (X _ {k} \cdot D _ {k})} \\ & {\qquad = \varphi_ {L _ {1}} (X _ {1}) + \dots + \varphi_ {L _ {k}} (X _ {k}).} \end{array}
$$

对于第二条结论，证明下图交换：

$$
\begin{array}{c} [ 0, N _ {1}) \times \dots \times [ 0, N _ {1}) \xrightarrow {\operatorname{colex} _ {S _ {1}} ^ {- 1} \times \cdots \times \operatorname{colex} _ {S _ {k}} ^ {- 1}} [ 0, S _ {1}) \times \dots \times [ 0, S _ {1}) \\ \operatorname{colex} _ {(N _ {1}, \ldots , N _ {k})} ^ {- 1} \Bigg | \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \cong \Bigg | \qquad \qquad \qquad \qquad \qquad \varphi_ {L _ {1}} + \dots + \varphi_ {L _ {k}} \\ [ 0, N _ {1} \dots N _ {k}) \xrightarrow {\operatorname{colex} _ {S _ {1} * \cdots * S _ {k}} ^ {- 1}} [ 0, S _ {1} * \dots * S _ {k}) \xrightarrow {\varphi_ {L _ {1} * \cdots * L _ {k}}} \mathbb {Z} \end{array}
$$

左侧方块交换，因为 colexicographic isomorphism 满足结合律；右侧三角形由第一条结论可知交换。□

可以按如下方式描述 concatenated layout 的重要属性。

命题 2.1.3.41。假设 $L_1,\ldots,L_k$ 是 flat layout，则

1. $L_1\star\cdots\star L_k$ 的 rank 为

$$
\operatorname{rank} \left(L _ {1} \star \dots \star L _ {k}\right) = \sum_ {i = 1} ^ {k} \operatorname{rank} \left(L _ {i}\right),
$$

2. $L_1\star\cdots\star L_k$ 的 size 为

$$
\operatorname{size} \left(L _ {1} \star \dots \star L _ {k}\right) = \prod_ {i = 1} ^ {k} \operatorname{size} \left(L _ {i}\right),
$$

3. $L_1\star\cdots\star L_k$ 的 cosize 为

$$
\operatorname{cosize} \left(L _ {1} \star \dots \star L _ {k}\right) = 1 - k + \sum_ {i = 1} ^ {k} \operatorname{cosize} \left(L _ {i}\right).
$$

证明。对每个 $1\leq i\leq k$，记 $L_i=S_i:D_i$。对第 1 条，计算得

$$
\operatorname{rank} \left(L _ {1} \star \dots \star L _ {k}\right) = \operatorname{len} \left(S _ {1} \star \dots \star S _ {k}\right) = \sum_ {i = 1} ^ {k} \operatorname{len} \left(S _ {i}\right) = \sum_ {i = 1} ^ {k} \operatorname{rank} \left(L _ {i}\right).
$$

对第 2 条，计算得

$$
\operatorname{size} \left(L _ {1} \star \dots \star L _ {k}\right) = \operatorname{size} \left(S _ {1} \star \dots \star S _ {k}\right) = \prod_ {i = 1} ^ {k} \operatorname{size} \left(S _ {i}\right) = \prod_ {i = 1} ^ {k} \operatorname{size} \left(L _ {i}\right).
$$

对第 3 条，计算得

$$
\begin{array}{l} \text {cosize} (L _ {1} \star \dots \star L _ {k}) = 1 + \max (\Phi_ {L _ {1} \star \dots \star L _ {k}}) \\ \qquad = 1 + \sum_ {i = 1} ^ {k} \max (\Phi_ {L _ {i}}) \\ \qquad = 1 - k + (1 + \max (\Phi_ {L _ {1}})) + \dots + (1 + \max (\Phi_ {L _ {1}})) \\ \qquad = 1 - k + \text {cosize} (L _ {1}) + \dots + \text {cosize} (L _ {k}). \end{array}
$$

其中使用了命题 2.1.3.40 对 $\Phi_{L_1\star\cdots\star L_k}$ 的刻画。

## 2.1.4 Flat coalesce

前文已经看到，flat layout L 的 layout function $\Phi_L$ 是重要不变量。很多情况下，我们只关心 layout function $\Phi_L$，可以自由使用任何 layout function 为 $\Phi_L$ 的 layout。flat coalesce 操作

$$
L \mapsto \operatorname{coal} ^ {\flat} (L)
$$

提供 layout function 为 $\Phi_L$ 的最简单 flat layout，参见命题 2.1.4.19。

首先定义 coalesced flat layout 的概念。

定义 2.1.4.1。假设 $L=(s_1,\ldots,s_m):(d_1,\ldots,d_m)$ 是 flat layout。如果

1. 对任意 $1\leq i\leq m$，都有 $s_i\neq1$；

2. 对任意 $1\leq i<m$，都有 $s_id_i\neq d_{i+1}$，

就称 L 为 coalesced。示例 2.1.4.2。flat layout

$$
L = (3, 5, 2): (7, 2 1, 4)
$$

不是 coalesced，因为 $3\cdot7=21$。

示例 2.1.4.3。flat layout

$$
L = (2, 7, 6): (1, 3, 1 0)
$$

是 coalesced。

示例 2.1.4.4。空 layout $E=():()` 是 coalesced。

示例 2.1.4.5。如果 $L=(s):(d)$ 且 $s\neq1$，则 L 是 coalesced。

示例 2.1.4.6。如果 $L=(s_1,\ldots,s_m):(d_1,\ldots,d_m)$ 是满足 `rank(L)>1` 的 column-major layout，则 L 不是 coalesced，因为对任意 $1\leq i<m$，都有

$$
s _ {i} d _ {i} = s _ {i} (s _ {1} \dots s _ {i - 1}) = s _ {1} \dots s _ {i} = d _ {i + 1}.
$$

示例 2.1.4.7。如果 $L=(s_1,\ldots,s_m):(d_1,\ldots,d_m)$ 是 row-major layout，且对所有 $1\leq i\leq m$ 都有 $s_i>1$，则 L 是 coalesced：如果 $1\leq i<m$，则

$$
s _ {i} d _ {i} = s _ {i} s _ {i + 1} \dots s _ {m} > s _ {i + 2} \dots s _ {m} = d _ {i + 1}.
$$

示例 2.1.4.8。形如

$$
L = (s _ {1}, \ldots , s _ {m}): (0, \ldots , 0)
$$

的 flat layout 是 coalesced，当且仅当 $m\leq1$。

如果 L 是 flat layout，可以移除满足 $s_i=1$ 的 mode，并合并满足 $s_id_i=d_{i+1}$ 的 mode，从而得到与 L 具有相同 layout function 的 coalesced layout $\mathsf{coal}^\flat(L)$。更精确地，作如下构造。

构造 2.1.4.9。假设 L 是 flat layout，并写成

$$
\operatorname{squeeze} (L) = \left(s _ {1}, \dots , s _ {m}\right): \left(d _ {1}, \dots , d _ {m}\right).
$$

令 ∼ 为 $\langle m\rangle$ 上由以下关系生成的等价关系：如果

$$
s _ {i} d _ {i} = d _ {i + 1}.
$$

则令 $i\sim i+1$。商集 $\langle m\rangle/\sim$ 按“当 $i\leq i'$ 时，$[i]\leq[i']$”排序，因此可把 $\langle m\rangle/\sim$ 与 $\langle\bar m\rangle$ 等同，其中 $\bar m$ 是 $\langle m\rangle/\sim$ 的大小。如果 $i\in\langle\bar m\rangle$ 对应等价类

$$
I = \{i ^ {\prime}, i ^ {\prime} + 1, \dots , i ^ {\prime} + k \} \in \langle m \rangle / \sim ,
$$

则把整数 $\bar s_i$ 和 $\bar d_i$ 定义为

$$
\bar {s} _ {i} = s _ {i ^ {\prime}} s _ {i ^ {\prime} + 1} \dots s _ {i ^ {\prime} + k}
$$

and 

$$
\bar {d} _ {i} = d _ {i ^ {\prime}},
$$

并定义

$$
\mathsf {c o a l} ^ {\flat} (L) = (\bar {s} _ {1}, \dots , \bar {s} _ {\bar {m}}): (\bar {d} _ {1}, \dots , \bar {d} _ {\bar {m}}).
$$

观察 2.1.4.10。考察定义可知，也可以等价地把 $\mathsf{coal}^\flat(L)$ 定义为从

$$
\operatorname{squeeze} (L) = \left(s _ {1}, \dots , s _ {m}\right): \left(d _ {1}, \dots , d _ {m}\right)
$$

反复执行操作

$$
s _ {i}, s _ {i + 1}: d _ {i}, s _ {i} d _ {i} \quad \rightsquigarrow \quad s _ {i} s _ {i + 1}: d _ {i}
$$

直到结果成为 coalesced 后所得的 flat layout。

示例 2.1.4.11。如果 $L=(2,2,2,2,2):(8,16,1024,2048,4096)$，则

$$
\operatorname{coal} ^ {\flat} (L) = (4, 8): (8, 1 0 2 4).
$$

示例 2.1.4.12。如果 $L=(3,4,1,5):(1,8,3,32)$，则

$$
\operatorname{coal} ^ {\flat} (L) = (3, 2 0): (1, 8).
$$

示例 2.1.4.13。如果 $L=(s_1,\ldots,s_m):(d_1,\ldots,d_m)$ 是 column-major，且并非所有 $s_i$ 都等于 1，则

$$
\operatorname{coal} ^ {\flat} (L) = \left(s _ {1} \dots s _ {m}\right): (1).
$$

示例 2.1.4.14。如果 L 是 row-major，则

$$
\operatorname{coal} ^ {\flat} (L) = \operatorname{squeeze} (L).
$$

下面证明操作 $L\mapsto\mathsf{coal}^\flat(L)$ 的结果是 coalesced layout。

引理 2.1.4.15。如果 L 是 flat layout，则 $\mathsf{coal}^\flat(L)$ 是 coalesced。

证明。沿用构造 2.1.4.9 的记号，令

$$
\operatorname{squeeze} (L) = \left(s _ {1}, \dots , s _ {m}\right): \left(d _ {1}, \dots , d _ {m}\right)
$$

and let 

$$
\mathsf {c o a l} ^ {\flat} (L) = (\bar {s} _ {1}, \dots , \bar {s} _ {\bar {m}}): (\bar {d} _ {1}, \dots , \bar {d} _ {\bar {m}}).
$$

要证明 $\mathsf{coal}^\flat(L)$ 是 coalesced。假设 $1\leq i\leq\bar m$。则 i 对应一个非空等价类 $I\in\langle m\rangle/\sim$，且

$$
\bar {s} _ {i} = \prod_ {i ^ {\prime} \in I} s _ {i ^ {\prime}}
$$

是整数 $s_{i'}>1$ 的乘积，所以 $\bar s_i>1$。

假设 $1\leq i<\bar m$。我们声称 $\bar s_i\bar d_i\neq\bar d_{i+1}$。假设 i 对应等价类

$$
\{i ^ {\prime}, i ^ {\prime} + 1, \dots , i ^ {\prime} + k \} \in \langle m \rangle / \sim ,
$$

并假设 i+1 对应等价类

$$
\left\{i ^ {\prime} + k + 1, i ^ {\prime} + k + 2, \dots , i ^ {\prime} + k + \ell \right\} \in \langle m \rangle / \sim .
$$

对 $0\leq t<k$ 使用等式 $s_{i'+t}d_{i'+t}=d_{i'+t+1}$，可写成

$$
\begin{array}{r l} & {\bar {s} _ {i} \bar {d} _ {i} = \bar {d} _ {i} \bar {s} _ {i} = d _ {i ^ {\prime}} s _ {i ^ {\prime}} s _ {i ^ {\prime} + 1} \dots s _ {i ^ {\prime} + k}} \\ & {\qquad = d _ {i ^ {\prime} + 1} s _ {i ^ {\prime} + 1} \dots s _ {i ^ {\prime} + k}} \\ & {\qquad \vdots} \\ & {\qquad = d _ {i ^ {\prime} + k} s _ {i ^ {\prime} + k}} \\ & {\qquad = s _ {i ^ {\prime} + k} d _ {i ^ {\prime} + k}} \end{array}
$$

由于 $i'+k$ 与 $i'+k+1$ 不属于同一等价类，因此

$$
\bar {s} _ {i} \bar {d} _ {i} = s _ {i ^ {\prime} + k} d _ {i ^ {\prime} + k} \neq d _ {i ^ {\prime} + k + 1} = \bar {d} _ {i + 1}.
$$

示例 2.1.4.16。如果 L 是 coalesced，则 $\mathsf{coal}^\flat(L)=L$。特别地，这意味着 $\mathsf{coal}^\flat(-)$ 是幂等操作：

$$
\operatorname{coal} ^ {\flat} \left(\operatorname{coal} ^ {\flat} (L)\right) = \operatorname{coal} ^ {\flat} (L).
$$

接下来证明，对 flat layout 执行 coalesce 不会改变 layout function。

引理 2.1.4.17。如果 L 是 flat layout，则 $\Phi_{\mathsf{coal}^\flat(L)}=\Phi_L$。

证明。根据观察 2.1.4.10，只需证明：把 $s_i,s_{i+1}:d_i,s_id_i$ 的一个实例替换为 $s_is_{i+1}:d_i$ 不会改变 layout function。假设

$$
L = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m})
$$

是 flat layout，并存在某个 $1\leq i<m$ 满足 $d_{i+1}=s_id_i$。令

$$
L ^ {\prime} = (s _ {1} ^ {\prime}, \ldots , s _ {m - 1} ^ {\prime}): (d _ {1} ^ {\prime}, \ldots , d _ {m - 1} ^ {\prime})
$$

表示把 L 的第 i 个和第 i+1 个 mode 合并后所得 flat layout。更精确地，

$$
s _ {j} ^ {\prime} = \left\{ \begin{array}{l l} s _ {j} & j <   i \\ s _ {i} s _ {i + 1} & j = i \\ s _ {j + 1} & i <   j <   m, \end{array} \right. \quad \text { and } \quad d _ {j} ^ {\prime} = \left\{ \begin{array}{l l} d _ {j} & j \leq i \\ d _ {j + 1} & i <   j <   m. \end{array} \right.
$$

L 的 layout function 定义为

$$
\Phi_ {L} (x) = x _ {1} d _ {1} + \dots + x _ {m} d _ {m}
$$

其中 $x_j=\left\lfloor\frac{x}{s_1\cdots s_{j-1}}\right\rfloor\bmod s_j$；L<sup>′</sup> 的 layout function 定义为

$$
\Phi_ {L ^ {\prime}} (x) = x _ {1} ^ {\prime} d _ {1} ^ {\prime} + \dots + x _ {m - 1} ^ {\prime} d _ {m - 1} ^ {\prime}
$$

其中 $x_j'=\left\lfloor\frac{x}{s_1'\cdots s_{j-1}'}\right\rfloor\bmod s_j'$。可以观察到

$$
x _ {j} ^ {\prime} = \left\{ \begin{array}{l l} x _ {j} & j <   i \\ x _ {i} + x _ {i + 1} s _ {i} & j = i \\ x _ {j + 1} & i <   j <   m, \end{array} \right.
$$

因此

$$
\begin{array}{r l} \Phi_ {L} (x) & = x _ {1} d _ {1} + \dots + x _ {m} d _ {m} \\ & = x _ {1} d _ {1} + \dots + x _ {i} d _ {i} + x _ {i + 1} s _ {i} d _ {i} + \dots + x _ {m} d _ {m} \\ & = x _ {1} d _ {1} + \dots + (x _ {i} + x _ {i + 1} s _ {i}) d _ {i} + \dots + x _ {m} d _ {m} \\ & = x _ {1} ^ {\prime} d _ {1} ^ {\prime} + \dots + x _ {m - 1} ^ {\prime} d _ {m - 1} ^ {\prime} \\ & = \Phi_ {L ^ {\prime}} (x). \end{array}
$$

可以使用 coalesce 操作刻画两个 flat layout 何时具有相同 layout function。命题 2.1.4.18。假设 A 和 B 是 flat layout，则

$$
\Phi_ {A} = \Phi_ {B} \quad \Leftrightarrow \quad \operatorname{coal} ^ {\flat} (A) = \operatorname{coal} ^ {\flat} (B).
$$

证明。如果 $\mathsf{coal}^\flat(A)=\mathsf{coal}^\flat(B)$，则根据引理 2.1.4.17，

$$
\Phi_ {A} = \Phi_ {\mathsf {c o a l} ^ {\flat} (A)} = \Phi_ {\mathsf {c o a l} ^ {\flat} (B)} = \Phi_ {B}.
$$

反之，假设 $\mathsf{coal}^\flat(A)\neq\mathsf{coal}^\flat(B)$。我们将证明 $\Phi_A\neq\Phi_B$。写成

$$
\begin{array}{l} \mathsf {c o a l} ^ {\flat} (A) = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m}), \\ \mathsf {c o a l} ^ {\flat} (B) = (t _ {1}, \ldots , t _ {n}): (e _ {1}, \ldots , e _ {m}). \end{array}
$$

如果 m、n 中一个非零而另一个为 0，则显然 $\Phi_A\neq\Phi_B$，因此可以假设 $m,n\geq1$。令 i 为满足 $(s_i,d_i)\neq(t_i,e_i)$ 的最小整数。特别地，对任意 $j<i$ 都有 $s_1\cdots s_j=t_1\cdots t_j$。需要考虑两种情况：

• 情况 1：假设 $d_i\neq e_i$。令 $N=s_1\cdots s_{i-1}=t_1\cdots t_{i-1}$。则

$$
\Phi_ {\mathsf {c o a l} ^ {\flat} (A)} (N) = d _ {i} \neq e _ {i} = \Phi_ {\mathsf {c o a l} ^ {\flat} (B)} (N)
$$

所以 $\Phi_{\mathsf{coal}^\flat(A)}\neq\Phi_{\mathsf{coal}^\flat(B)}$，进而 $\Phi_A\neq\Phi_B$。

• 情况 2：假设 $d_i=e_i$，因此 $s_i\neq t_i$。不失一般性，假设 $s_i<t_i$。令 $N=s_1\cdots s_i=(t_1\cdots t_{i-1})s_i$。则

$$
\Phi_ {\mathsf {c o a l} ^ {\flat} (A)} (N) = d _ {i + 1}
$$

而

$$
\begin{array}{r} \Phi_ {\mathsf {c o a l} ^ {\flat} (B)} (N) = s _ {i} e _ {i} \\ = s _ {i} d _ {i}, \end{array}
$$

由于 $\mathsf{coal}^\flat(A)$ 是 coalesced，有 $d_{i+1}\neq s_id_i$。因此 $\Phi_{\mathsf{coal}^\flat(A)}\neq\Phi_{\mathsf{coal}^\flat(B)}$，进而 $\Phi_A\neq\Phi_B$。

前一命题给出 $\mathsf{coal}^\flat(L)$ 的以下抽象刻画。

命题 2.1.4.19。如果 L 是 flat layout，则 $\mathsf{coal}^\flat(L)$ 是 layout function 为 $\Phi_L$ 且 rank 最小的唯一 flat layout。

证明。假设 $L'$ 是满足 $\Phi_{L'}=\Phi_L$ 的 flat layout。根据命题 2.1.4.18，

$$
\operatorname{coal} ^ {\flat} (L) = \operatorname{coal} ^ {\flat} \left(L ^ {\prime}\right),
$$

and it follows that 

$$
\operatorname{rank} \left(\operatorname{coal} ^ {\flat} (L)\right) = \operatorname{rank} \left(\operatorname{coal} ^ {\flat} \left(L ^ {\prime}\right)\right) \leq \operatorname{rank} \left(L ^ {\prime}\right),
$$

其中等号成立，当且仅当

$$
L ^ {\prime} = \operatorname{coal} ^ {\flat} (L ^ {\prime}) = \operatorname{coal} ^ {\flat} (L).
$$

## 2.1.5 Compact flat layout

在讨论 layout complement 之前，必须定义一类称为 compact flat layout 的重要 layout。它们是 layout function 为双射的 flat layout。用描绘 layout 的标准 grid 图来说，如果每个整数 $0\leq i<\mathsf{size}(L)$ 恰好出现一次，则 flat layout L 是 compact 的。例如，layout

<table><tr><td>0</td><td>3</td><td>6</td><td>9</td><td>12</td><td>15</td></tr><tr><td>1</td><td>4</td><td>7</td><td>10</td><td>13</td><td>16</td></tr><tr><td>2</td><td>5</td><td>8</td><td>11</td><td>14</td><td>17</td></tr></table>

是 compact 的，而 layout

<table><tr><td>0</td><td>6</td><td>12</td><td>18</td><td>24</td><td>30</td></tr><tr><td>2</td><td>8</td><td>14</td><td>20</td><td>26</td><td>32</td></tr><tr><td>4</td><td>10</td><td>16</td><td>22</td><td>28</td><td>34</td></tr></table>

以及

<table><tr><td>0</td><td>2</td><td>4</td><td>6</td><td>8</td><td>10</td></tr><tr><td>1</td><td>3</td><td>5</td><td>7</td><td>9</td><td>11</td></tr><tr><td>2</td><td>4</td><td>6</td><td>8</td><td>10</td><td>12</td></tr></table>


不是 compact 的。更精确地，有以下定义。


定义 2.1.5.1。假设 L 是 flat layout。如果

$$
[ 0, \operatorname{size} (L)) \xrightarrow {\Phi_ {L} ^ {\operatorname{cosize} (L)}} [ 0, \operatorname{cosize} (L))
$$

是 isomorphism，就称 L 为 compact。

示例 2.1.5.2。flat layout

$$
L = (2, 2, 2, 2): (1, 2, 4, 8)
$$

是 compact 的。更一般地，如果 L 是 column-major，则 L 是 compact 的。

示例 2.1.5.3。flat layout

$$
L = (3, 6 4, 3 2): (2 0 4 8, 3 2, 1)
$$

是 compact 的。更一般地，如果 L 是 row-major，则 L 是 compact 的。

示例 2.1.5.4。空 layout

$$
E = (): ()
$$

是 compact 的。

示例 2.1.5.5。假设

$$
L = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m})
$$

是 flat layout。如果 L 存在某个满足 $s_i>1$ 且 $d_i=0$ 的 mode，则 L 不是 compact 的。

可以如下显式刻画 compact layout。

命题 2.1.5.6。假设 L 是 flat layout，并写成

$$
\operatorname{squeeze} (L) = \left(s _ {1}, \dots , s _ {m}\right): \left(d _ {1}, \dots , d _ {m}\right).
$$

则 L 是 compact 的，当且仅当存在置换 $\sigma\in\Sigma_m$，使

$$
d _ {\sigma (i)} = s _ {\sigma (1)} \dots s _ {\sigma (i - 1)}
$$

对所有 $1\leq i\leq m$ 都成立。换言之，L 是 compact 的，当且仅当存在置换 $\sigma\in\Sigma_m$，使 $\mathsf{squeeze}(L)^\sigma$ 为 column-major。

证明。假设 L 是 flat layout，并写成

$$
\operatorname{squeeze} (L) = \left(s _ {1}, \dots , s _ {m}\right): \left(d _ {1}, \dots , d _ {m}\right).
$$

首先假设 L 是 compact 的，因此存在置换 $\sigma\in\Sigma_m$，使每个 $1\leq i\leq m$ 都满足 $d_{\sigma(i)}=s_{\sigma(1)}\cdots s_{\sigma(i-1)}$。如果写成 $S^\sigma=(s_{\sigma(1)},\ldots,s_{\sigma(m)})$，则可把 $\Phi_L^{\mathsf{cosize}(L)}$ 写成 composite

$$
[ 0, \operatorname{size} (L)) \xrightarrow {\operatorname{colex} _ {S} ^ {- 1}} [ 0, S) \xrightarrow {\cong} [ 0, S ^ {\sigma}) \xrightarrow {\operatorname{colex} _ {S ^ {\sigma}}} [ 0, \operatorname{cosize} (L))
$$

$$
(x _ {1}, \dots , x _ {m}) \longmapsto (x _ {\sigma (1)}, \dots , x _ {\sigma (m)})
$$

由于其中每个映射都是 isomorphism，composite $\Phi_L^{\mathsf{cosize}(L)}$ 也是 isomorphism。

反之，假设 $\Phi_L^{\mathsf{cosize}(L)}$ 是 isomorphism。首先注意，stride $d_1,\ldots,d_m$ 必须两两不同。假设 $d_i=d_j$，令 $\delta_i^m$ 和 $\delta_j^m$ 分别表示第 i 个和第 j 个条目为 1、其余条目均为 0 的 tuple。这些 tuple 满足

$$
\delta_ {i} ^ {m} \cdot (d _ {1}, \ldots , d _ {m}) = d _ {i} = d _ {j} = \delta_ {j} ^ {m} \cdot (d _ {1}, \ldots , d _ {m}),
$$

由于 $\Phi_L^{\mathsf{cosize}(L)}$ 是单射，必须有 $i=j$。既然 stride $d_1,\ldots,d_m$ 两两不同，令 $\sigma\in\Sigma_m$ 为满足下式的置换：

$$
d _ {\sigma (1)} <   d _ {\sigma (2)} <   \dots <   d _ {\sigma (m)}.
$$

下面对 $i\geq1$ 使用归纳法证明 $d_{\sigma(i)}=s_{\sigma(1)}\cdots s_{\sigma(i-1)}$。对基本情况 $i=1$，注意 1 位于 $\Phi_L^{\mathsf{cosize}(L)}$ 的像中，而 $\Phi_L^{\mathsf{cosize}(L)}$ 的最小非零值是 $d_{\sigma(1)}$，所以 $d_{\sigma(1)}=1$。假设 $i>1$，且已经对所有 $j<i$ 证明该结论。考察 stride $d_{\sigma(i)}$。不存在形如 $(x_1,\ldots,x_{i-1},0,\ldots,0)^\sigma$ 的 tuple，使

$$
(x _ {1}, \dots , x _ {i - 1}, 0, \dots , 0) ^ {\sigma} \cdot (d _ {1}, \dots , d _ {m}) = s _ {\sigma (1)} \dots s _ {\sigma (i - 1)},
$$

因为这类表达式可能取得的最大值为

$$
\sum_ {j = 1} ^ {i - 1} (s _ {\sigma (j)} - 1) (s _ {\sigma (1)} \dots s _ {\sigma (j - 1)}) = s _ {\sigma (1)} \dots s _ {\sigma (i - 1)} - 1.
$$

由于 $\Phi_L^{\mathsf{cosize}(L)}$ 是满射，且 $d_{\sigma(i)}<d_{\sigma(i+1)}<\cdots<d_{\sigma(m)}$，所以 $\Phi_L^{\mathsf{cosize}(L)}$ 的下一个最大值是 $d_{\sigma(i)}$，必须有 $d_{\sigma(i)}=s_{\sigma(1)}\cdots s_{\sigma(i-1)}$，结论得证。□

本节最后给出 flat layout L 为 compact 的一组等价条件。

## 命题 2.1.5.7。假设 L 是 flat layout，则以下条件等价。

1. L 是 compact 的。

2. $\mathsf{coal}^\flat(L)$ 是 compact 的。

3. `squeeze(L)` 是 compact 的。

4. `sort(L)` 是 compact 的。

证明。条件 1、2、3 的等价性来自

$$
\Phi_ {L} = \Phi_ {\text { coal } ^ {\flat} (L)} = \Phi_ {\text { squeeze } (L)}.
$$

只需继续证明：L 为 compact 当且仅当 $\mathsf{sort}(L)$ 为 compact。使用事实

$$
\operatorname{squeeze} (\operatorname{sort} (L)) = \operatorname{sort} (\operatorname{squeeze} (L)),
$$

we have 

$$
\begin{array}{l l} \text {sort} (L) \text {is compact.} & \Leftrightarrow \quad \text {squeeze} (\text {sort} (L)) \text {is compact.} \\ & \Leftrightarrow \quad \text {sort} (\text {squeeze} (L)) \text {is compact.} \end{array}
$$

对某个置换 $\tau\in\Sigma_m$，有 $\mathsf{sort}(\mathsf{squeeze}(L))=\mathsf{squeeze}(L)^\tau$。因此，存在置换 σ 使 `squeeze(L)`<sup>σ</sup> 为 column-major，当且仅当存在置换 $\sigma'\in\Sigma_m$ 使 `sort(squeeze(L))` 为 column-major；具体为 $\sigma'=\tau^{-1}\sigma$。所以

`sort(squeeze(L))` 是 compact 的 ⇔ `squeeze(L)` 是 compact 的。

⇔ L 是 compact 的。

## 2.1.6 Complement

本节定义互为 complement 的 flat layout。回忆定义 2.1.5.1：如果 layout function

$$
\Phi_ {L} ^ {\text { cosize } (L)}: [ 0, \text { size } (L)) \to [ 0, \text { cosize } (L))
$$

是 isomorphism，则 flat layout L 是 compact 的。

定义 2.1.6.1。假设 A 和 B 是 flat layout。如果 concatenated layout $A\star B$ 是 compact 的，就称 B 是 A 的 complement，并记作 $A\perp B$。

示例 2.1.6.2。如果 $A=(3):(5)$，$B=(5):(1)$，则 $A\perp B$，因为

$$
A \star B = (3, 5): (5, 1)
$$

是 compact 的。

<table><tr><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td></tr></table>

<table><tr><td>0</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td></tr><tr><td>5</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td></tr><tr><td>10</td><td>10</td><td>11</td><td>12</td><td>13</td><td>14</td></tr></table>

示例 2.1.6.3。如果 $A=(4,2,10):(1400,2,20)$，$B=(2,5,7,2):(1,4,200,5600)$，则 $A\perp B$，因为

$$
A \star B = (4, 2, 1 0, 2, 5, 7, 2): (1 4 0 0, 2, 2 0, 1, 4, 2 0 0, 5 6 0 0)
$$

是 compact 的。

示例 2.1.6.4。如果 A 是 flat layout，$E=():()` 是空 layout，则 $A\perp E$ 当且仅当 A 是 compact 的，因为

$$
A \star E = A.
$$

示例 2.1.6.5。如果 A 和 B 是 flat layout，则

$$
A \perp B \quad \Leftrightarrow \quad B \perp A.
$$

示例 2.1.6.6。如果 A 是 flat layout，则 $A\perp A$ 当且仅当 `size(A)=1`。

观察 2.1.6.7。A 存在 complement 的必要条件是 $\Phi_A$ 为单射。不过，确实存在 $\Phi_A$ 为单射但 A 不存在 complement 的 flat layout A。例如，考虑 layout

$$
A = (2, 2): (1, 3).
$$

A 的 layout function 是单射，因为

$$
\Phi_ {A} (0) = 0, \Phi_ {A} (1) = 1, \Phi_ {A} (2) = 3, \text { and } \Phi_ {A} (3) = 4,
$$

但 A 不存在 complement。假设

$$
B = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m})
$$

是任意其他 flat layout。如果不存在 tuple

$$
\left(x _ {1}, x _ {2}, y _ {1}, \dots , y _ {m}\right) \in [ 0, 2) \times [ 0, 2) \times [ 0, s _ {1}) \times \dots \times [ 0, s _ {m})
$$

满足 $\varphi_{A\star B}(x_1,x_2,y_1,\dots,y_m)=2$，则 $A\star B$ 不是 compact 的。反之，假设存在这样的 tuple $(x_1,x_2,y_1,\dots,y_m)$。则 $\varphi_B(y_1,\dots,y_m)\in\{0,1,2\}$。

• 情况 1：如果 $\varphi_B(y_1,\ldots,y_m)=0$，则

$$
\varphi_ {A \star B} (0, 0, 0, \dots , 0) = 0 = \varphi_ {A \star B} (0, 0, y _ {1}, \dots , y _ {m}).
$$

• 情况 2：如果 $\varphi_B(y_1,\dots,y_m)=1$，则

$$
\varphi_ {A \star B} (1, 0, 0, \dots , 0) = 1 = \varphi_ {A \star B} (0, 0, y _ {1}, \dots , y _ {m}).
$$

• 情况 3：如果 $\varphi_B(y_1,\dots,y_m)=2$，则

$$
\varphi_ {A \star B} (0, 1, 0, \dots , 0) = 3 = \varphi_ {A \star B} (1, 0, y _ {1}, \dots , y _ {m}).
$$

无论哪种情况，都可推出 $\varphi_{A\star B}$ 不是单射，因此 $\Phi_{A\star B}$ 也不是单射。这意味着 $A\star B$ 不是 compact 的，所以 B 不是 A 的 complement。

观察 2.1.6.8。Complement 并不唯一。例如，如果

$$
A = (8, 8): (2, 3 2),
$$

则以下每个 layout

$$
\begin{array}{l} B _ {1} = (2, 2): (1, 1 6) \\ B _ {2} = (2, 2): (1 6, 1) \\ B _ {3} = (5, 2, 2, 1): (2 5 6, 1, 1 6, 0) \end{array}
$$

都是 A 的 complement。因此，存在一个可能为空的集合

complements<sup>♭</sup>(A) = {flat layout $B \mid B$ 是 $A$ 的 complement}。

它由所有与 A 互为 complement 的 layout 组成。

给出 B 为 A 的 complement 的一组等价条件会很有用，参见命题 2.1.6.10。为此，需要下面的技术引理，它描述 concatenation 与 `squeeze(−)`、`sort(−)` 和 $\mathsf{coal}^\flat(-)$ 操作之间的相互作用。

引理 2.1.6.9。假设 A 和 B 是 flat layout，则

1. squeeze $( A \star B ) = { \mathsf { s q u e e z e } } ( A ) \star { \mathsf { s q u e e z e } } ( B )$ *4 

2. sor $\therefore ( A \star B ) = { \mathsf { s o r t } } ( L \star { \mathsf { s o r t } } ( B ) )$ , and 

3. coa $\mathsf { I } ^ { \flat } ( A \star B ) = \mathsf { c o a l } ^ { \flat } ( A \star \mathsf { c o a l } ^ { \flat } ( B ) )$ 

证明。写成

$$
\begin{array}{l} A = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m}) \\ B = (t _ {1}, \ldots , t _ {n}): (e _ {1}, \ldots , e _ {n}). \end{array}
$$

令 $\{i_1<\dots<i_{m'}\}\subset\langle m\rangle$ 表示满足 $s_{i_k}\neq1$ 的索引，$\{j_1,\dots,j_{n'}\}\subset\langle n\rangle$ 表示满足 $t_{j_\ell}\neq1$ 的索引，则

$$
\begin{array}{c} \text {squeeze} (A \star B) = (s _ {i _ {1}}, \ldots , s _ {i _ {m ^ {\prime}}}, t _ {j _ {1}}, \ldots , t _ {j _ {n ^ {\prime}}}): (d _ {i _ {1}}, \ldots , d _ {i _ {m ^ {\prime}}}, e _ {j _ {1}}, \ldots , e _ {j _ {n ^ {\prime}}}) \\ = \text {squeeze} (A) \star \text {squeeze} (B). \end{array}
$$

这证明了第 1 条。对于第 2 条，注意对任意 flat layout L 和任意置换 $\sigma\in\Sigma_{\mathsf{len}(L)}$，都有 $\mathsf{sort}(L)=\mathsf{sort}(L^\sigma)$。结论来自以下观察：

$$
A \star \operatorname{sort} (B) = (A \star B) ^ {\sigma}
$$

其中 σ 是形如 $\sigma=\mathsf{id}\times\sigma'\in\Sigma_m\times\Sigma_n\subset\Sigma_{m+n}$ 的 block permutation。对于第 3 条，只需证明 $A\star B$ 与 $A\star\mathsf{coal}^\flat(B)$ 具有相同 layout function；这可由命题 2.1.3.40 得出。

命题 2.1.6.10。假设 A 和 B 是 flat layout，则以下条件等价。

1. $A \perp B .$ 

2. $B \perp A .$ 

3. $A \perp \mathsf { s q u e e z e } ( B )$ 

4. $A \perp \mathsf { c o a l } ^ { \flat } ( B )$ 

5. $A \perp \mathsf { s o r t } ( B ) .$ 

证明。使用命题 2.1.5.7 和引理 2.1.6.9 证明这些条件等价。首先注意，$\mathsf{sort}(A\star B)=\mathsf{sort}(B\star A)$，这说明条件 1 与 2 等价。其次，根据引理 2.1.6.9，如果 op(−) 是 `squeeze(−)`、`sort(−)` 或 $\mathsf{coal}^\flat(-)$ 中任一操作，则

$$
\mathsf {o p} (A \star B) = \mathsf {o p} (A \star \mathsf {o p} (B)),
$$

因此

$$
\begin{array}{l l l} A \perp B & \Leftrightarrow & A \star B \text {is compact.} \\ & \Leftrightarrow & \mathsf {o p} (A \star B) \text {is compact.} \\ & \Leftrightarrow & \mathsf {o p} (A \star \mathsf {o p} (B)) \text {is compact.} \\ & \Leftrightarrow & \mathsf {o p} (B) \text {is a complement of} A. \end{array}
$$

下面刻画 flat layout 何时存在 complement。为此，作如下定义。

定义 2.1.6.11。假设 A 是 flat layout，并写成

$$
\operatorname{sort} (\text { squeeze } (A)) = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m}).
$$

如果对任意 $1\leq i<m$，整数 $s_id_i$ 都整除 $d_{i+1}$，就称 A 是 complementable 的。

示例 2.1.6.12。flat layout

$$
A _ {1} = (4, 1, 1, 4, 4): (6 4, 0, 0, 1, 8)
$$

是 complementable 的，而 flat layout

$$
A _ {2} = (4, 4, 4): (6 4, 1, 1)
$$

不是 complementable 的。

示例 2.1.6.13。flat layout

$$
A _ {1} = (1 0, 2): (4, 8 0)
$$

是 complementable 的，而 flat layout

$$
A _ {2} = (1 0, 2): (8 0, 4)
$$

不是 complementable 的。

示例 2.1.6.14。如果 A 是 compact 的，则根据命题 2.1.5.6，A 是 complementable 的。

示例 2.1.6.15。假设

$$
A = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m})
$$

是 flat layout。如果存在任意 $1\leq i\leq m$ 满足 $s_i\neq1$ 且 $d_i=0$，则 A 不是 complementable 的。

如果 A 是 complementable 的，可以按如下方式构造 A 的 complement。

构造 2.1.6.16。假设 A 是 flat layout，并写成

$$
\operatorname{sort} (\text { squeeze } (A)) = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m}).
$$

如果 A 是 complementable 的，则定义 flat layout $\mathsf{comp}^\flat(A)$ 为

$$
\operatorname{comp} ^ {\flat} (A) = \operatorname{coal} ^ {\flat} (C)
$$

其中

$$
C = \left(d _ {1}, \frac {d _ {2}}{s _ {1} d _ {1}}, \frac {d _ {3}}{s _ {2} d _ {2}}, \dots , \frac {d _ {m}}{s _ {m - 1} d _ {m - 1}}\right): \left(1, s _ {1} d _ {1}, s _ {2} d _ {2}, \dots , s _ {m - 1} d _ {m - 1}\right).
$$

示例 2.1.6.17。如果 $A=(8,8):(1,8)$，则

$$
\mathsf {c o m p} ^ {\flat} (A) = (): ()
$$

是空 layout。更一般地，如果 A 是 compact 的，则 $\mathsf{comp}^\flat(A)=():()` 是空 layout。

示例 2.1.6.18。如果 $A=(2,2):(2,8)$，则

$$
\mathsf {c o m p} ^ {\flat} (A) = (2, 2): (1, 4).
$$

示例 2.1.6.19。如果 $A=(3,3,8):(16,96,1)$，则

$$
\operatorname{comp} ^ {\flat} (A) = (2, 2): (8, 4 8).
$$

下面证明 $\mathsf{comp}^\flat(A)$ 确实是 A 的 complement。

引理 2.1.6.20。假设 A 是 flat layout。如果 A 是 complementable 的，则

$$
A \perp \operatorname{comp} ^ {\flat} (A).
$$

证明。写成

$$
\operatorname{sort} (\text { squeeze } (A)) = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m}),
$$

于是 $\mathsf{comp}^\flat(A)=\mathsf{coal}^\flat(C)$，其中

$$
C = \left(d _ {1}, \frac {d _ {2}}{s _ {1} d _ {1}}, \frac {d _ {3}}{s _ {2} d _ {2}}, \dots , \frac {d _ {m}}{s _ {m - 1} d _ {m - 1}}\right): \left(1, s _ {1} d _ {1}, s _ {2} d _ {2}, \dots , s _ {m - 1} d _ {m - 1}\right).
$$

根据命题 2.1.6.10，只需证明 C 是 `sort(squeeze(A))` 的 complement。确实如此，因为 concatenation

$$
\operatorname{sort} (\text { squeeze } (A)) \star C
$$

等于

$$
\left(s _ {1}, \dots , s _ {m}, d _ {1}, \frac {d _ {2}}{s _ {1} d _ {1}}, \dots , \frac {d _ {m}}{s _ {m - 1} d _ {m - 1}}\right): (d _ {1}, \dots , d _ {m}, 1, s _ {1} d _ {1}, \dots , s _ {m - 1} d _ {m - 1}),
$$

其排序结果等于

$$
\left(d _ {1}, s _ {1}, \frac {d _ {2}}{s _ {1} d _ {1}}, \dots , \frac {d _ {m}}{s _ {m - 1} d _ {m - 1}}, s _ {m}\right): (1, d _ {1}, s _ {1} d _ {1}, \dots , s _ {m - 1} d _ {m - 1}, d _ {m})
$$

这是 column-major。

前文已经证明，如果 A 是 complementable 的，则 A 存在 complement。下面证明其逆命题也成立。

命题 2.1.6.21。假设 A 是 flat layout。存在 A 的 complement B，当且仅当 A 是 complementable 的。

证明。如果 A 是 complementable 的，则根据引理 2.1.6.20，layout $B=\mathsf{comp}^\flat(A)$ 是 A 的 complement。反之，假设存在 A 的 complement B，并考虑 flat layout

$$
\begin{array}{l} L = \text { sort } \big (\text { squeeze } (A) \star \text { squeeze } (B) \big) \\ = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {n}). \end{array}
$$

由于 $\Phi_L(0)=0$ 且 $\Phi_L$ 是单射，知道 $d_1\neq0$。下面证明 $d_i=s_1\cdots s_{i-1}$，也就是 L 为 column-major。由于

$$
\Phi_ {L} ^ {\text { cosize } (L)}: [ 0, \text { size } (L)) \to [ 0, \text { cosize } (L))
$$

是双射，知道 1 位于 $\Phi_L$ 的像中，因此 $d_1=1$。假设 $1<i\leq m$，并已对所有 $j<i$ 证明 $d_j=s_1\cdots s_{j-1}$。考察 stride $d_i$。不存在 $(x_1,\ldots,x_{i-1},0,\ldots,0)$ 使 $(x_1,\ldots,x_{i-1},0,\ldots,0)\cdot(d_1,\ldots,d_m)=s_1\cdots s_{i-1}$，因为这类表达式可能取得的最大值是

$$
\sum_ {j = 1} ^ {i - 1} (s _ {j} - 1) (s _ {1} \dots s _ {j - 1}) = s _ {1} \dots s _ {i} - 1.
$$

由于 $\Phi_L$ 是满射，且 $d_i\leq d_{i+1}\leq\cdots\leq d_m$，所以 $\Phi_L$ 的下一个最大值是 $d_i$，必须有 $d_i=s_1\cdots s_{i-1}$，结论得证。

回到主要目标，考虑 layout

$$
\operatorname{sort} (\operatorname{squeeze} (A)) = \left(s _ {1} ^ {\prime}, \dots , s _ {m ^ {\prime}} ^ {\prime}\right): \left(d _ {1} ^ {\prime}, \dots , d _ {m ^ {\prime}} ^ {\prime}\right).
$$

则存在 $j_1<\cdots<j_{m'}$，使每个 $1\leq i\leq m'$ 都有 $s_i'=s_{j_i}$ 且 $d_i'=d_{j_i}$。如果 $1\leq i<m'$，则

$$
s _ {i} ^ {\prime} d _ {i} ^ {\prime} = s _ {j _ {i}} d _ {j _ {i}} = s _ {j _ {i}} s _ {1} \cdot \cdot \cdot s _ {j _ {i} - 1}
$$

整除

$$
d _ {i + 1} ^ {\prime} = s _ {1} \dots s _ {j _ {i + 1} - 1},
$$

因此 A 是 complementable 的。

下一个目标是对 flat layout A 的 complement $\mathsf{comp}^\flat(A)$ 给出抽象刻画。为此，需要以下引理。

引理 2.1.6.22。假设 A 是 flat layout。如果 A 是 complementable 且已排序，则 layout function

$$
\Phi_ {A}: [ 0, \operatorname{size} (A)) \to \mathbb {Z}
$$

递增。

证明。写成

$$
A = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m}).
$$

如果 $1\leq k\leq m$，我们声称

$$
d _ {1} (s _ {1} - 1) + d _ {2} (s _ {2} - 1) + \dots + d _ {k - 1} (s _ {k - 1} - 1) \leq d _ {k}.
$$

当 $k=1$ 时，该结论平凡成立；对 k 归纳，有

$$
\begin{array}{r l} d _ {1} (s _ {1} - 1) + \dots + d _ {k - 2} (s _ {k - 2} - 1) + d _ {k - 1} (s _ {k - 1} - 1) & \leq d _ {k - 1} + d _ {k - 1} (s _ {k} - 1) \\ & = d _ {k - 1} s _ {k - 1} \\ & \leq d _ {k}. \end{array}
$$

现在假设 $x,y\in[0,\mathsf{size}(A))$ 且 $x\leq y$。在 colexicographic isomorphism 下，这些整数对应 tuple

$$
(x _ {1}, \dots , x _ {m}), (y _ {1}, \dots , y _ {m}) \in [ 0, s _ {1}) \times \dots \times [ 0, s _ {m})
$$

由于 $x\leq y$，存在最大的 $1\leq k\leq m$，使 $x_k<y_k$，且对所有 $k<\ell\leq m$ 都有 $x_\ell=y_\ell$。于是可计算

$$
\begin{array}{l} \Phi_ {A} (x) = d _ {1} x _ {1} + \dots + d _ {k - 1} x _ {k - 1} + d _ {k} x _ {k} + d _ {k + 1} x _ {k + 1} + \dots + d _ {m} x _ {m} \\ \qquad = d _ {1} x _ {1} + \dots + d _ {k - 1} x _ {k - 1} + d _ {k} x _ {k} + d _ {k + 1} y _ {k + 1} + \dots + d _ {m} y _ {m} \\ \qquad \leq d _ {1} (s _ {1} - 1) + \dots + d _ {k - 1} (s _ {k - 1} - 1) + d _ {k} x _ {k} + d _ {k + 1} y _ {k + 1} + \dots + d _ {m} y _ {m} \\ \qquad \leq d _ {k} + d _ {k} x _ {k} + d _ {k + 1} y _ {k + 1} + \dots + d _ {m} y _ {m} \\ \qquad = d _ {k} (x _ {k} + 1) + d _ {k + 1} y _ {k + 1} + \dots + d _ {m} y _ {m} \\ \qquad \leq d _ {k} y _ {k} + d _ {k + 1} y _ {k + 1} + \dots + d _ {m} y _ {m} \\ \qquad \leq d _ {1} y _ {1} + \ldots d _ {m} y _ {m} \\ \qquad = \Phi_ {A} (y). \end{array}
$$

命题 2.1.6.23。假设 A 和 B 是 flat layout。如果

1. $A \perp B ,$ 

2. $\mathsf { s i z e } ( B ) = \mathsf { s i z e } ( \mathsf { c o m p } ^ { \flat } ( A ) ) .$ 4 

3. B 是 coalesced；

4. B 已排序，

则 $B=\mathsf{comp}^\flat(A)$。

证明。条件 1 和 2 意味着 $\Phi_B$ 与 $\Phi_{\mathsf{comp}^\flat(A)}$ 具有相同的像。由于 B 和 $\mathsf{comp}^\flat(A)$ 已排序，根据引理 2.1.6.22，$\Phi_B$ 和 $\Phi_{\mathsf{comp}^\flat(A)}$ 都递增。结合这两个事实可得 $\Phi_B=\Phi_{\mathsf{comp}^\flat(A)}$。命题 2.1.4.18 和条件 3 于是给出

$$
B = \operatorname{coal} ^ {\flat} (B) = \operatorname{coal} ^ {\flat} (\operatorname{comp} ^ {\flat} (A)) = \operatorname{comp} ^ {\flat} (A).
$$

定义 2.1.6.24。假设 A 和 B 是 flat layout，N 是正整数。如果 B 是 A 的 complement，并且

$$
\operatorname{size} (A) \cdot \operatorname{size} (B) = N.
$$

就称 B 是 A 的 N-complement。定义 2.1.6.25。假设 A 是 flat layout，并写成

$$
\operatorname{sort} (\text { squeeze } (A)) = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m}).
$$

如果以下条件成立，就称 A 是 N-complementable 的：

1. 对所有 $1\leq i<m$，整数 $s_id_i$ 整除 $d_{i+1}$；

2. 整数 $s_md_m$ 整除 N。

观察 2.1.6.26。如果 A 是 complementable 的，且 $s_m:d_m$ 是 layout `sort(squeeze(A))` 的最后一个 mode，则 A 为 N-complementable，当且仅当 N 是 $s_md_m$ 的正整数倍。

观察 2.1.6.27。N-complement 并不唯一。例如，如果 $A=(2,2):(1,50)$ 且 $N=100$，则 layout $B_1=(25):(2)$ 和 $B_2=(5,5):(2,10)$ 都是 A 的 N-complement。更一般地，如果 B 是 A 的 N-complement，则 $\mathsf{coal}^\flat(B)$ 也是 A 的 N-complement。

注记 2.1.6.28。假设 A 是 flat layout，$B_1$ 和 $B_2$ 是 A 的 N-complement。layout function $\Phi_{B_1}$ 和 $\Phi_{B_2}$ 不一定相等，但必然具有相同的像。例如，如果 $A=(4):(63)$ 且 $N=252$，则 $B_1=(7,9):(1,7)$ 和 $B_2=(9,7):(7,1)$ 都是 A 的 N-complement，而且 $\Phi_{B_1}\neq\Phi_{B_2}$，因为

$$
\Phi_ {B _ {1}} (1) = 1 \neq 7 = \Phi_ {B _ {2}} (1).
$$

更一般地，如果 B 是 A 的 N-complement，则 `sort(B)` 也是 A 的 N-complement。

构造 2.1.6.29。假设 A 是 flat layout，N 是正整数，并且 A 是 N-complementable 的。如果写成

$$
\operatorname{sort} (\text { squeeze } (A)) = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m}),
$$

则定义 flat layout $\mathsf{comp}^\flat(A,N)$ 为

$$
\operatorname{comp} ^ {\flat} (A, N) = \operatorname{coal} ^ {\flat} (C)
$$

其中

$$
C = \left(d _ {1}, \frac {d _ {2}}{s _ {1} d _ {1}}, \frac {d _ {3}}{s _ {2} d _ {2}}, \dots , \frac {N}{s _ {m} d _ {m}}\right): \left(1, s _ {1} d _ {1}, s _ {2} d _ {2}, \dots , s _ {m} d _ {m}\right).
$$

示例 2.1.6.30。如果 $A=(3,10):(80,4)$ 且 $N=2400$，则

$$
\mathsf {c o m p} ^ {\flat} (A, N) = (4, 2, 1 0): (1, 4 0, 2 4 0).
$$

引理 2.1.6.31。假设 A 是 flat layout，N 是正整数，并且 A 是 N-complementable 的。则 comp<sup>♭</sup>(A,N) 是 A 的 N-complement。

证明。写成

$$
\operatorname{sort} (\text { squeeze } (A)) = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m}),
$$

于是 $\mathsf{comp}^\flat(A,N)=\mathsf{coal}^\flat(C)$，其中

$$
C = \left(d _ {1}, \frac {d _ {2}}{s _ {1} d _ {1}}, \frac {d _ {3}}{s _ {2} d _ {2}}, \dots , \frac {N}{s _ {m} d _ {m}}\right): (1, s _ {1} d _ {1}, s _ {2} d _ {2}, \dots , s _ {m} d _ {m}).
$$

首先计算

$$
\begin{array}{l} \operatorname{size} (A) \cdot \operatorname{size} (B) = \left(\prod_ {i = 1} ^ {m} s _ {i}\right) \cdot \left(d _ {1} \cdot \left(\prod_ {i = 2} ^ {m} \frac {d _ {i}}{s _ {i - 1} d _ {i - 1}}\right) \cdot \frac {N}{s _ {m} d _ {m}}\right) \\ = \frac {\left(\prod_ {i = 1} ^ {m} s _ {i}\right) \left(\prod_ {i = 1} ^ {m} d _ {i}\right)}{\left(\prod_ {i = 1} ^ {m} s _ {i} d _ {i}\right)} \cdot N \\ = N. \end{array}
$$

需要检查 $A\star B$ 是 compact 的。等价地，需要检查 $\Phi_{A\star B}^N$ 是 isomorphism。根据引理 2.1.5.6，只需证明

$$
\text { squeeze } (A) \star \text { squeeze } (B)
$$

是 compact 的。确实如此，因为该 layout 等于

$$
\left(s _ {1}, \dots , s _ {m}, d _ {1}, \frac {d _ {2}}{s _ {1} d _ {1}}, \dots , \frac {N}{s _ {m} d _ {m}}\right): (d _ {1}, \dots , d _ {m}, 1, s _ {1} d _ {1}, \dots , s _ {m} d _ {m})
$$

因此其排序结果

$$
\operatorname{sort} (\text { squeeze } (A) \star \text { squeeze } (B))
$$

等于

$$
\left(d _ {1}, s _ {1}, \frac {d _ {2}}{s _ {1} d _ {1}}, \ldots , \frac {d _ {m}}{s _ {m - 1} d _ {m - 1}}, s _ {m}, \frac {N}{s _ {m} d _ {m}}\right): (1, d _ {1}, s _ {1} d _ {1}, \ldots , s _ {m - 1} d _ {m - 1}, d _ {m}, s _ {m} d _ {m})
$$

这是 column-major。

命题 2.1.6.32。假设 A 是 flat layout，N 是正整数。存在 A 的 N-complement B，当且仅当 A 是 N-complementable 的。

证明。如果 A 是 N-complementable 的，则根据引理 2.1.6.31，layout $B=\mathsf{comp}^\flat(A,N)$ 是 A 的 N-complement。

另一方面，假设存在 A 的 N-complement B。考虑 flat layout

$$
\begin{array}{l} L := \text { sort } \big (\text { squeeze } (A) \star \text { squeeze } (B) \big) \\ = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {n}). \end{array}
$$

由于 $\Phi_L(0)=0$ 且 $\Phi_L$ 是单射，知道 $d_1\neq0$。下面证明 $d_i=s_1\cdots s_{i-1}$，也就是 L 为 column-major。由于

$$
\Phi_ {L} ^ {N}: [ 0, N) \to [ 0, N)
$$

是双射，知道 1 位于 $\Phi_L$ 的像中，因此 $d_1=1$。假设 $1<i\leq m$，并已对所有 $j<i$ 证明 $d_j=s_1\cdots s_{j-1}$。考察 stride $d_i$。不存在 $(x_1,\ldots,x_{i-1},0,\ldots,0)$ 使 $(x_1,\ldots,x_{i-1},0,\ldots,0)\cdot(d_1,\ldots,d_m)=s_1\cdots s_{i-1}$，因为这类表达式可能取得的最大值是

$$
\sum_ {j = 1} ^ {i - 1} (s _ {j} - 1) \left(s _ {1} \dots s _ {j - 1}\right) = s _ {1} \dots s _ {i} - 1.
$$

由于 $\Phi_L$ 是满射，且 $d_i\leq d_{i+1}\leq\cdots\leq d_m$，所以 $\Phi_L$ 的下一个最大值是 $d_i$，必须有 $d_i=s_1\cdots s_{i-1}$，结论得证。

回到主要目标，考虑 layout

$$
\operatorname{sort} (\text { squeeze } (A)) = \left(s _ {1} ^ {\prime}, \dots , s _ {m ^ {\prime}} ^ {\prime}\right): \left(d _ {1} ^ {\prime}, \dots , d _ {m ^ {\prime}} ^ {\prime}\right).
$$

则存在 $j_1<\cdots<j_{m'}$，使每个 $1\leq i\leq m'$ 都有 $s_i'=s_{j_i}$ 且 $d_i'=d_{j_i}$。如果 $1\leq i<m'$，则

$$
s _ {i} ^ {\prime} d _ {i} ^ {\prime} = s _ {j _ {i}} d _ {j _ {i}} = s _ {j _ {i}} s _ {1} \dots s _ {j _ {i} - 1}
$$

整除

$$
d _ {i + 1} ^ {\prime} = s _ {1} \dots s _ {j _ {i + 1} - 1}.
$$

如果 $i=m'$，则

$$
s _ {m ^ {\prime}} ^ {\prime} d _ {m ^ {\prime}} ^ {\prime} = s _ {j _ {m ^ {\prime}}} d _ {j _ {m ^ {\prime}}} = s _ {j _ {m ^ {\prime}}} s _ {1} \dots s _ {j _ {m ^ {\prime}} - 1}
$$

整除

$$
N = s _ {1} \cdot \cdot \cdot s _ {m}.
$$

因此 A 是 N-complementable 的。

命题 2.1.6.33。假设 N 是正整数，A 是 N-complementable flat layout。如果 B 是满足以下条件的 flat layout：

1. B 是 A 的 N-complement；

2. B 是 coalesced；

3. B 已排序，

则 $B=\mathsf{comp}^\flat(A,N)$。

证明。条件 1 和 2 意味着 $\Phi_B$ 与 $\Phi_{\mathsf{comp}^\flat(A,N)}$ 具有相同的像。由于 B 和 $\mathsf{comp}^\flat(A,N)$ 已排序，根据引理 2.1.6.22，二者的 layout function 都递增。结合这两个事实可得 $\Phi_B=\Phi_{\mathsf{comp}^\flat(A,N)}$。命题 2.1.4.18 和条件 3 于是给出

$$
B = \operatorname{coal} ^ {\flat} (B) = \operatorname{coal} ^ {\flat} (\operatorname{comp} ^ {\flat} (A, N)) = \operatorname{comp} ^ {\flat} (A, N).
$$

引理 2.1.6.34。假设 A 是 flat layout。如果 $N_1\leq N_2$ 是正整数，并且 A 同时是 N<sub>1</sub>-complementable 和 N<sub>2</sub>-complementable 的，则

$$
\Phi_ {\mathsf {c o m p} ^ {\flat} (A, N _ {2})} \mid_ {[ 0, N _ {1})} = \Phi_ {\mathsf {c o m p} ^ {\flat} (A, N _ {1})}.
$$

证明。写成

$$
\begin{array}{c} \text {sort(squeeze(A)) = (s_{1} ,\ldots,s_{m}):(d_{1} ,\ldots,d_{m}) ,} \\ C = \left(d _ {1}, \frac {d _ {2}}{s _ {1} d _ {1}}, \frac {d _ {3}}{s _ {2} d _ {2}}, \ldots , \frac {d _ {m}}{s _ {m - 1} d _ {m - 1}}\right): \left(1, s _ {1} d _ {1}, s _ {2} d _ {2}, \ldots , s _ {m - 1} d _ {m - 1}\right) \end{array}
$$

再写成

$$
\begin{array}{l} E _ {1} = \left(\frac {N _ {1}}{s _ {m} d _ {m}}\right): (s _ {m} d _ {m}), \\ E _ {2} = \left(\frac {N _ {2}}{s _ {m} d _ {m}}\right): (s _ {m} d _ {m}), \\ C _ {1} = C \star E _ {1}, \\ C _ {2} = C \star E _ {2}, \end{array}
$$

因此

$$
\begin{array}{c} \mathsf {c o m p} ^ {\flat} (A) = \mathsf {c o a l} ^ {\flat} (C) \\ \mathsf {c o m p} ^ {\flat} (A, N _ {1}) = \mathsf {c o a l} ^ {\flat} (C _ {1}) \\ \mathsf {c o m p} ^ {\flat} (A, N _ {2}) = \mathsf {c o a l} ^ {\flat} (C _ {2}). \end{array}
$$

于是得到交换图

$$
\begin{array}{c} [ 0, \text {size} (C _ {1})) \xrightarrow {\text {colex} _ {(\text {size} (C) , N _ {1})} ^ {- 1}} [ 0, \text {size} (C)) \times [ 0, N _ {1}) \xrightarrow {\Phi_ {C} \times s _ {m} d _ {m}} \mathbb {Z} \times \mathbb {Z} \xrightarrow {+} \mathbb {Z} \\ \Big \downarrow \subseteq \\ [ 0, \text {size} (C _ {2})) \xrightarrow {\text {colex} _ {(\text {size} (C) , N _ {2})} ^ {- 1}} [ 0, \text {size} (C)) \times [ 0, N _ {2}) \xrightarrow {\Phi_ {C} \times s _ {m} d _ {m}} \mathbb {Z} \times \mathbb {Z} \xrightarrow {+} \mathbb {Z} \end{array}
$$

根据命题 2.1.3.40，上行 composite 是 $C_1=C\star E_1$ 的 layout function，下行 composite 是 $C_2=C\star E_2$ 的 layout function。这说明 $\Phi_{C_2}$ 在 $[0,\mathsf{size}(C_1))$ 上的 restriction 是 $\Phi_{C_1}$；再结合以下事实即可得到结论：

$$
\begin{array}{l} \Phi_ {\mathsf {c o m p} ^ {\flat} (A, N _ {1})} = \Phi_ {C _ {1}} \\ \Phi_ {\mathsf {c o m p} ^ {\flat} (A, N _ {2})} = \Phi_ {C _ {2}}. \end{array}
$$

## 2.1.7 其他操作

本节定义 flat layout 上的另外几种操作，即 composition、flat division 和 flat product。它们是嵌套 layout 上更自然操作的 flattening 版本。虽然并不经常直接使用这些操作，但为完整起见仍将其收入本文。

## 2.1.7.1 Composition

如果 A 和 B 是 flat layout，则 composite $B\circ A$ 是一个 flat layout，其 layout function 是 A 与 B 的 layout function 的 composite。更精确地，有以下定义。

定义 2.1.7.1。假设 A 和 B 是 flat layout。如果以下条件成立，就称 flat layout C 是 A 与 B 的 composition，并记作 $C=B\circ A$：

1. C 是 non-degenerate；

2. sh $\mathsf { i a p e } ( A ) = { \mathsf { s h a p e } } ( R )$ 

3. $\Phi _ { R } = \Phi _ { B } \circ \Phi _ { A } ^ { \mathsf { s i z e } ( B ) }$ 

注记 2.1.7.2。定义中的条件 2 保证 $\Phi_R$ 和 $\Phi_A$ 具有相同 domain，条件 3 蕴含 $\mathsf{cosize}(A)\leq\mathsf{size}(B)$。

示例 2.1.7.3。如果 $A=(2,3):(5,6)$，$B=(80):(10)$，则

$$
B \circ A = (2, 3): (5 0, 6 0).
$$

更一般地，如果

$$
A = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m})
$$

是 non-degenerate flat layout，并且

$$
B = (t): (e)
$$

是满足 $t\geq\mathsf{cosize}(A)$ 的 rank 1 flat layout，则 A 与 B 可复合，而且

$$
B \circ A = (s _ {1}, \ldots , s _ {m}): (t d _ {1}, \ldots , t d _ {m}).
$$

示例 2.1.7.4。如果 $A=(128,128):(0,0)$，$B=(64,32):(1,64)$，则

$$
B \circ A = (1 2 8, 1 2 8): (0, 0).
$$

更一般地，如果 A 是每个 stride 条目都为零的 flat layout，B 是任意 flat layout，则 A 与 B 可复合，并且 $B\circ A=A$。

示例 2.1.7.5。如果 $A=(64,32):(2,256)$，$B=(2048,2048):(1,2048)$，则

$$
B \circ A = (6 4, 3 2): (2, 2 5 6).
$$

更一般地，如果 A 是任意 flat layout，B 是满足 `cosize(A)≤size(B)` 的 column-major flat layout，则 $B\circ A=A$。

示例 2.1.7.6。如果 $A=(4):(2)$，$B=(2,2,6):(12,6,1)$，则不存在满足 $R=B\circ A$ 的 flat layout R。

注记 2.1.7.7。如果 $B'$ 与 B 具有相同 layout function，则 $B\circ A=B'\circ A$。

注记 2.1.7.8。flat layout 是更一般 layout 概念的特殊情况，参见定义 2.3.1.1。事实表明，在某些情况下——例如示例 2.1.7.6——不存在满足 $C=B\circ A$ 的 flat layout C，但存在满足该等式的嵌套 layout C，参见示例 2.3.7.6。因此，在完全一般地定义 layout 之后再进一步讨论和分析 composition。

## 2.1.7.2 Flat division

如果 A 和 B 是 flat layout，则 A 除以 B 的 flat division，是更自然的 layout logical division 的 flattening 版本。细节参见第 2.3.8 节。

定义 2.1.7.9。假设 A 和 B 是 flat layout，B 是 size(A)-complementable 的，并令

$$
B ^ {c} = \operatorname{comp} ^ {\flat} (B, \text { size } (A)).
$$

把 A 除以 B 的 flat division 定义为 flat layout

$$
A \oslash^ {\flat} B = A \circ (B \star B ^ {c}).
$$

示例 2.1.7.10。如果 `A=(2,2,2,2):(1,4,2,8)`，$B=(2,2):(4,2)$，则

$$
A \oslash^ {\flat} B = (2, 2, 2, 2): (4, 2, 1, 8).
$$

示例 2.1.7.11。如果 `A=(3,5,9,6):(54,0,6,1)`，$B=(6,3):(135,1)$，则

$$
A \oslash^ {\flat} B = (6, 3, 5, 9): (1, 5 4, 0, 6).
$$

示例 2.1.7.12。如果 A 是任意 flat layout，$B=():()` 是空 layout，则

$$
A \oslash^ {\flat} B = A.
$$

## 2.1.7.3 Flat product

如果 A 和 B 是 flat layout，则 A 与 B 的 flat product $A\otimes^\flat B$，是更自然的 layout logical product 的 flattening 版本。细节参见第 2.3.9 节。

定义 2.1.7.13。假设 A 和 B 是 flat layout，A 是 `size(A)·cosize(B)`-complementable 的，并令

$$
A ^ {c} = \operatorname{comp} ^ {\flat} (A, \text { size } (A) \cdot \text { cosize } (B)).
$$

把 A 与 B 的 flat product 定义为

$$
A \otimes^ {\flat} B = A \star (A ^ {c} \circ B).
$$

示例 2.1.7.14。如果 $A=(2,2,2):(1,2,4)$，$B=(2,2,2):(1,2,4)$，则

$$
A \otimes^ {\flat} B = (2, 2, 2, 2, 2, 2): (1, 2, 4, 8, 1 6, 3 2).
$$

示例 2.1.7.15。如果 $A=(2,2,2):(1,2,4)$，$B=(3,5):(5,1)$，则

$$
A \otimes^ {\flat} B = (2, 2, 2, 3, 5): (1, 2, 4, 4 0, 8).
$$

示例 2.1.7.16。如果 A 是任意 flat layout，$B=():()` 是空 layout，则

$$
A \otimes^ {\flat} B = A.
$$

## 2.1.8 Tractable flat layout

本节定义一类行为特别良好的 flat layout，称为 tractable flat layout。它包括最重要的 layout 示例，例如 row-major、column-major、compact 和 complementable layout。后文将看到，tractable flat layout 恰好是由某个 category **Tuple** 产生的 layout。

定义 2.1.8.1。假设 L 是 flat layout，并写成

$$
\operatorname{sort} (L) = \left(s _ {1}, \dots , s _ {m}\right): \left(d _ {1}, \dots , d _ {m}\right).
$$

如果对每个 $1\leq i<m$，以下条件之一成立，就称 L 是 tractable 的：

1. $d_i=0$；或

2. $s_id_i$ 整除 $d_{i+1}$。

示例 2.1.8.2。flat layout

$$
L = (1 2): (1 7)
$$

是 tractable 的。更一般地，任意 rank 1 flat layout 都是 tractable 的。

示例 2.1.8.3。flat layout

$$
L = (2, 4, 3 2): (1, 2, 8)
$$

是 tractable 的。更一般地，任意 column-major layout

$$
L = (s _ {1}, \dots , s _ {m}): (1, s _ {1}, \dots , s _ {1} \dots s _ {m - 1})
$$

都是 tractable 的。

示例 2.1.8.4。flat layout

$$
L = (2, 4, 3 2): (1 2 8, 3 2, 1)
$$

是 tractable 的。更一般地，任意 row-major layout

$$
L = (s _ {1}, \ldots , s _ {m}): (s _ {2} \dots s _ {m}, \ldots , s _ {m}, 1)
$$

都是 tractable 的。

示例 2.1.8.5。flat layout

$$
L = (3, 3, 1, 3, 3, 1, 3): (8 1, 1, 0, 9, 3, 0, 2 7)
$$

是 tractable 的。更一般地，任意 compact flat layout 都是 tractable 的。

示例 2.1.8.6。flat layout

$$
L = (3, 7, 7): (0, 1 5, 0)
$$

是 tractable 的。更一般地，恰好有一个非零 stride 的任意 flat layout 都是 tractable 的。

示例 2.1.8.7。flat layout

$$
L = (2, 2, 2, 2): (1, 2 0 4 8, 1 6, 6 4)
$$

是 tractable 的。更一般地，任意 complementable flat layout 都是 tractable 的。

示例 2.1.8.8。假设 L 是 flat layout。如果 L 是 tractable 的，且 $I\subset\langle m\rangle$ 是任意子集，则 restriction $L|_I$ 是 tractable 的。特别地，如果 L 是 tractable 的，则 `squeeze(L)` 和 `filter(L)` 都是 tractable 的。

示例 2.1.8.9。flat layout

$$
L = (4, 8): (3, 3)
$$

不是 tractable 的。特别地，这说明 tractable flat layout $L_1$ 和 $L_2$ 的 concatenation $L_1\star L_2$ 不一定 tractable。

观察 2.1.8.10。如果 L 是 tractable flat layout，并且 `stride(L)` 没有条目等于 0，则 L 是 complementable 的。特别地，如果 L 是 tractable 的，则 `filter(L)` 是 complementable 的。

本节最后列出 flat layout L 为 tractable 的一组等价条件。

命题 2.1.8.11。假设 L 是 flat layout，则以下条件等价。

1. L 是 tractable 的。

2. `sort(L)` 是 tractable 的。

3. `filter(L)` 是 tractable 的。

4. `filter(L)` 是 complementable 的。

证明。假设 L 是 flat layout。

$(1\Leftrightarrow2)$：由以下事实可得：

$$
\operatorname{sort} (\operatorname{sort} (L)) = \operatorname{sort} (L).
$$

$(1\Leftrightarrow3)$：由以下事实可得：

$$
\operatorname{sort} (\text { filter } (L)) = \text { filter } (\operatorname{sort} (L)).
$$

$(3\Leftrightarrow4)$：如果

$$
L = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m})
$$

是每个 stride 条目 $d_i$ 都非零的 flat layout，则 tractability 的定义与 complementability 的定义相同。

## 2.2 Nested tuple

本节介绍 nested tuple，它是为了完全一般地定义 layout 而对 tuple 作出的推广。

## 2.2.1 Profile

nested tuple S 由其 flattening 和 profile 决定；前者是普通 tuple，后者描述 S 的加括号模式。Profile 的精确定义如下。

定义 2.2.1.1。profile P 是以下两种形式之一：

1. $P=*$；或

2. 对某个 $r\geq0$，P 是由 profile $P_1,\ldots,P_r$ 组成的 tuple $P=(P_1,\ldots,P_r)$。

用 Profile 表示 profile 集合。

示例 2.2.1.2。下面是一些 profile 示例。

$$
\begin{array}{l} P _ {1} = (*, *) \\ P _ {2} = (*, (*, *)) \\ P _ {3} = ((*, *), (*, *)) \\ P _ {4} = ((*, *, *), (*, ()) \\ P _ {5} = () \\ P _ {6} = * \end{array}
$$

下面定义 profile 的一些重要属性。

定义 2.2.1.3。假设 P 是 profile。

• P 的 rank 为

$$
\operatorname{rank} (P) = \left\{ \begin{array}{l l} 1 & P = * \\ r & P = (P _ {1}, \ldots , P _ {r}) \text {   is   a   tuple   of   profiles. } \end{array} \right..
$$

• P 的 length 为

$$
\mathsf {l e n} (P) = \left\{ \begin{array}{l l} 1 & P = * \\ \sum_ {i = 1} ^ {r} \mathsf {l e n} (P _ {i}) & P = (P _ {1}, \ldots , P _ {r}) \text {   is   a   tuple   of   profiles. } \end{array} \right.
$$

• P 的 depth 为

$$
\mathsf {d e p t h} (P) = \left\{ \begin{array}{l l} 0 & P = * \\ 1 + \max _ {1 \leq i \leq r} (\mathsf {d e p t h} (P _ {i})) & P = (P _ {1}, \ldots , P _ {r}) \text {   is   a   tuple   of   profiles. } \end{array} \right.
$$

示例 2.2.1.4。下面给出一些 profile 及其 rank、length 和 depth：

$$
P = * \quad \operatorname{rank} (P) = 1, \quad \operatorname{len} (P) = 1, \quad \operatorname{depth} (P) = 0
$$

$$
P = (*, *, *) \quad \operatorname{rank} (P) = 3, \quad \operatorname{len} (P) = 3, \quad \operatorname{depth} (P) = 1
$$

$$
P = (((*, *), *, *), *, *), \quad \operatorname{rank} (P) = 3, \quad \operatorname{len} (P) = 6, \quad \operatorname{depth} (P) = 3
$$

$$
P = ((((), ()), (*, (*, *))), \quad \operatorname{rank} (P) = 2, \quad \operatorname{len} (P) = 3, \quad \operatorname{depth} (P) = 3
$$

定义 2.2.1.5。假设 P 是满足 `rank(P)=r` 的 profile。如果 $1\leq i\leq r$，则 P 的第 i 个 mode 为

$$
\mathsf {m o d e} _ {i} (P) = \left\{ \begin{array}{l l} P & \mathsf {d e p t h} (P) = 0 (\text { hence } i = r = 1), \\ P _ {i} & P = (P _ {1}, \ldots , P _ {r}) \text { has   depth } \geq 1. \end{array} \right.
$$

示例 2.2.1.6。如果 $P=((*,*),(()),((*,(*,*))))$，则 P 的 mode 为

$$
\begin{array}{l} \text {mode} _ {1} (P) = (*, *) \\ \text {mode} _ {2} (P) = (()) \\ \text {mode} _ {3} (P) = (*, (*, *)) \end{array}
$$

以下记号会很有用。

记号 2.2.1.7。假设 P 是 depth 大于 0 的 profile。对任意 $1\leq j\leq\mathsf{rank}(P)$，记

$$
\begin{array}{l} \mathsf {l e n} _ {j} (X) = \mathsf {l e n} (\mathsf {m o d e} _ {j} (P)), \\ \mathsf {l e n} _ {<   j} (P) = \sum_ {i = 1} ^ {j - 1} \mathsf {l e n} _ {i} (X), \\ \mathsf {l e n} _ {\leq j} (X) = \mathsf {l e n} _ {<   j} (P) + \mathsf {l e n} _ {j} (P) \end{array}
$$

profile 支持的最重要操作是 substitution：如果 Q 是 length 为 m 的 profile，且 $P_1,\ldots,P_m$ 是 profile，则把 Q 的第 i 个条目替换为 profile $P_i$，可以得到新 profile $(P_1,\ldots,P_m)_Q$。更精确地，有以下定义。

定义 2.2.1.8。假设 Q 是 length 为 m 的 profile，$P_1,\ldots,P_m$ 是 profile。则 $P_1,\ldots,P_m$ 的 Q-substitution 是 profile

$$
(P _ {1}, \dots , P _ {m}) _ {Q}
$$

其定义如下。记 `depth(Q)=d`，`rank(Q)=r`。

• 如果 $d=0$，则 $m=1$，并定义

$$
(P _ {1}) _ {Q} = P _ {1}.
$$

• 接下来假设 $d>0$，并且已对所有 depth 小于 d 的 profile $Q'$ 定义 Q′-substitution。可以写成

$$
Q = (Q _ {1}, \dots , Q _ {r})
$$

其中每个 mode $Q_i=\mathsf{mode}_i(Q)$ 的 depth 都小于 d。对每个 $1\leq i\leq r$，令

$$
\ell_ {i} = \operatorname{len} (P _ {1}) + \dots + \operatorname{len} (P _ {i - 1}),
$$

则定义

$$
(P _ {1}, \dots , P _ {r}) _ {Q} = ((P _ {1}, \dots , P _ {\ell_ {2}}) _ {Q _ {1}}, \dots , (P _ {\ell_ {r} + 1}, \dots , P _ {\ell_ {r + 1}}) _ {Q _ {r}}).
$$

示例 2.2.1.9。如果 $Q=(*,*)$，$P_1=(*,*)$，$P_2=(*,*,*)$，则

$$
(P _ {1}, P _ {2}) _ {Q} = ((*, *), (*, *, *))
$$

更一般地，如果 $Q=(*,\ldots,*)$ 是满足 `depth(Q)=1` 且 `len(Q)=rank(Q)=r` 的 profile，则

$$
(P _ {1}, \dots , P _ {r}) _ {Q} = (P _ {1}, \dots , P _ {r})
$$

就是普通 concatenation。

旁注 2.2.1.10。Q-substitution 有一种 operad 解释。profile 集合 Profile 具有非对称 operad 结构：集合

$$
\operatorname{Profile} (n) = \{P \in \operatorname{Profile} | \operatorname{len} (P) = n \}
$$

构成 Profile 的 n 元操作集合；如果 $n=m_1+\cdots+m_r$，则 structure map

$$
\operatorname{Profile} \left(m _ {1}\right) \times \dots \times \operatorname{Profile} \left(m _ {r}\right) \times \operatorname{Profile} (n) \longrightarrow \operatorname{Profile} \left(m _ {1} + \dots + m _ {r}\right)
$$

$$
(P _ {1}, \dots , P _ {r}), Q \longmapsto (P _ {1}, \dots , P _ {r}) _ {Q}
$$

由 Q-substitution 给出。还可以在该非对称 operad 上形成 cofree symmetric operad，这相当于为 n 元操作集合赋予平凡的对称群作用。

## 2.2.2 基本定义

定义 profile 及其基本性质后，现在可以定义 nested tuple。

定义 2.2.2.1。如果 V 是集合，则条目位于 V 中的 nested tuple X 是一对 $(X^\flat,P)$，包括：

1. 条目位于 V 中的 tuple $X^\flat=(x_1,\dots,x_m)$，称为 X 的 flattening；

2. length 为 m 的 profile $\mathsf{prof}(X)=P$，称为 X 的 profile。

用 Nest(V) 表示条目位于集合 V 中的所有 nested tuple 组成的集合。

示例 2.2.2.2。下面给出一些 nested tuple 及其 flattening 和 profile。

$$
X = (2, (2, 2))
$$

$$
X = 2 5
$$

$$
X ^ {\flat} = (2, 2, 2)
$$

$$
X ^ {\flat} = (2 5)
$$

$$
\operatorname{prof} (X) = (*, (*, *))
$$

$$
\operatorname{prof} (X) = *
$$

$$
X = (((2, 2, 2), 8), 6 4)
$$

$$
X ^ {\flat} = (2, 2, 2, 8, 2 6)
$$

$$
\operatorname{prof} (X) = (((*, *, *), *), *)
$$

$$
X = ((), (3 2, (\)), (4, 8))
$$

$$
X ^ {\flat} = (3 2, 4, 8)
$$

$$
\operatorname{prof} (X) = \left(\left(\right), \left(*, \left(\right)\right), \left(*, *\right)\right)
$$

记号 2.2.2.3。有时写成

$$
X = (x _ {1}, \ldots , x _ {m}) _ {P}
$$

表示满足 $X^\flat=(x_1,\dots,x_m)$ 且 `prof(X)=P` 的 nested tuple。

观察 2.2.2.4。如果 V 是任意集合，则根据定义有 pullback square

$$
\begin{array}{c} \text {Nest} (V) \xrightarrow {\text {prof} (-)} \text {Profile} \\ (-) ^ {b} \Big \downarrow \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \text {len} (-) \\ \text {Tuple} (V) \xrightarrow {\text {len} (-)} \mathbb {N}. \end{array}
$$

注记 2.2.2.5。根据 profile 的递归定义，也可以等价地把条目位于 V 中的 nested tuple 定义为以下两种形式之一：

1. V 的一个元素；或

2. 由条目位于 V 中的 nested tuple 组成的 tuple。

下面定义 nested tuple 的一些重要属性。nested tuple X 的每个这类属性，都继承自其 flattening $X^\flat$ 或 profile `prof(X)`。

定义 2.2.2.6。假设 X 是条目位于 V 中的 nested tuple。

• X 的 rank 为

$$
\operatorname{rank} (X) = \operatorname{rank} (P)
$$

• X 的 length 为

$$
\operatorname{len} (X) = \operatorname{len} (P) = \operatorname{len} (X ^ {\flat})
$$

• X 的 depth 为

$$
\operatorname{depth} (X) = \operatorname{depth} (P)
$$

• 如果 $V=\mathbb{Z}$，则 X 的 size 为

$$
\operatorname{size} (X) = \operatorname{size} (X ^ {\flat}).
$$

示例 2.2.2.7。下面给出一些整数 nested tuple 及其 rank、length、depth 和 size：

$$
\begin{array}{l l} X = 2 7 & \text {rank} (X) = 1, \quad \text {len} (X) = 1, \quad \text {depth} (X) = 0, \quad \text {size} (X) = 2 7 \\ X = (2, 1 0, 5) & \text {rank} (X) = 3, \quad \text {len} (X) = 3, \quad \text {depth} (X) = 1, \quad \text {size} (X) = 1 0 0 \\ X = (((3, 4), 2, 2), 8, 9), & \text {rank} (X) = 3, \quad \text {len} (X) = 6, \quad \text {depth} (X) = 3, \quad \text {size} (X) = 3 0 9 6 \\ X = ((((),()), (2, (5, 5))), & \text {rank} (X) = 2, \quad \text {len} (X) = 3, \quad \text {depth} (X) = 3, \quad \text {size} (X) = 5 0 \end{array}
$$

示例 2.2.2.8。depth 为 0 的整数 nested tuple 就是一个整数。

示例 2.2.2.9。depth 为 1 的整数 nested tuple 就是整数 tuple。如果 X 是这类 nested tuple，则 $\mathsf{rank}(X)=\mathsf{len}(X)$。

定义 2.2.2.10。假设 $X=(x_1,\dots,x_m)_P$ 是满足 `rank(X)=r` 的 nested tuple。如果 $1\leq i\leq r$，则 X 的第 i 个 mode 是 nested tuple

$$
\operatorname{mode} _ {i} (X) = \left(x _ {\text { len } _ {<   i} (P) + 1}, \dots , x _ {\text { len } _ {\leq i} (P)}\right) _ {\operatorname{mode} _ {i} (P)}.
$$

示例 2.2.2.11。如果

$$
X = ((3), 4, ((1 0, 1 0), 1 2)),
$$

则 X 的 mode 为

$$
\begin{array}{l} \text { mode } _ {1} (X) = (3) \\ \text { mode } _ {2} (X) = 4 \\ \text { mode } _ {3} (X) = ((1 0, 1 0), 1 2) \end{array}
$$

示例 2.2.2.12。如果 `X=(32,5,6,64)`，则 X 的 mode 为

$$
\begin{array}{l} \mathrm{mode} _ {1} (X) = 3 2 \\ \mathrm{mode} _ {2} (X) = 5 \\ \mathrm{mode} _ {3} (X) = 6 \\ \mathrm{mode} _ {4} (X) = 6 4 \end{array}
$$

引入以下记号会比较方便。

记号 2.2.2.13。假设 X 是满足 `depth(X)>0` 的整数 nested tuple。对任意 `1≤j≤rank(X)`，记

$$
\begin{array}{c} \mathsf {l e n} _ {j} (X) = \mathsf {l e n} (\mathsf {m o d e} _ {j} (X)), \\ \mathsf {l e n} _ {<   j} (X) = \sum_ {i = 1} ^ {j - 1} \mathsf {l e n} _ {i} (X), \\ \mathsf {l e n} _ {\leq j} (X) = \mathsf {l e n} _ {<   j} (X) + \mathsf {l e n} _ {j} (X) \end{array}
$$

类似地，记

$$
\begin{array}{c} \text {size} _ {j} (X) = \text {size} (\text {mode} _ {j} (X)), \\ \text {size} _ {<   j} (X) = \prod_ {i = 1} ^ {j - 1} \text {size} _ {j} (X), \text {and} \\ \text {size} _ {\leq j} (X) = \text {size} _ {<   j} (X) \cdot \text {size} _ {j} (X). \end{array}
$$

定义 2.2.2.14。如果 $X=(x_1,\ldots,x_m)_P$ 是 nested tuple，且 $1\leq i\leq m$，则 X 的第 i 个 entry 为

$$
\operatorname{entry} _ {i} (X) = \operatorname{entry} _ {i} \left(X ^ {\flat}\right) = x _ {i}.
$$

示例 2.2.2.15。如果

$$
X = ((3), 4, ((1 0, 1 0), 1 2)),
$$

则 X 的 entry 为

$$
\begin{array}{l} \text {entry} _ {1} (X) = 3 \\ \text {entry} _ {2} (X) = 4 \\ \text {entry} _ {3} (X) = 1 0 \\ \text {entry} _ {4} (X) = 1 0 \\ \text {entry} _ {5} (X) = 1 2. \end{array}
$$

示例 2.2.2.16。如果 `X=(32,5,6,64)`，则 X 的 entry 为

$$
\begin{array}{l} \text {entry} _ {1} (X) = 3 2 \\ \text {entry} _ {2} (X) = 5 \\ \text {entry} _ {3} (X) = 6 \\ \text {entry} _ {4} (X) = 4. \end{array}
$$

示例 2.2.2.17。如果 X 是 depth 1 的 nested tuple，则对所有 $1\leq i\leq\mathsf{rank}(X)=\mathsf{len}(X)$，都有 $\mathsf{mode}_i(X)=\mathsf{entry}_i(X)$。

观察 2.2.2.18。如果 X 是整数 nested tuple，则 X 的 entry 是整数，而 X 的 mode 本身是整数 nested tuple。

最后，引入 nested tuple 的 congruence 概念，用来表示 nested tuple 何时具有相同 profile。

定义 2.2.2.19。如果 $X_1$ 和 $X_2$ 是 nested tuple，并且

$$
\operatorname{prof} (X _ {1}) = \operatorname{prof} (X _ {2}).
$$

就称 $X_1$ 和 $X_2$ congruent。示例 2.2.2.20。下面给出一些 nested tuple $X_1$、$X_2$，以及它们是否 congruent：

$$
\begin{array}{l l l} X _ {1} = 2 7 & X _ {2} = 1 0 0 & \text {congruent} \\ X _ {1} = (2, 2) & X _ {2} = (8, 6 4) & \text {congruent} \\ X _ {1} = ((4, 8), (4, 8)) & X _ {2} = ((1, 1), (5, 1 0)) & \text {congruent} \\ X _ {1} = ((6 4, (8, 8)), (2 5, (5, 5))) & X _ {2} = ((2, (3, 5)), (7, (1 1, 1 3))) & \text {congruent} \\ X _ {1} = 2 7 & X _ {2} = (1 0 0) & \text {not congruent} \\ X _ {1} = (2, 2) & X _ {2} = (8, 6 4, 1 2 8) & \text {not congruent} \\ X _ {1} = ((4, 8), (4, 8)) & X _ {2} = (((1, 1), (5, 1 0))) & \text {not congruent} \end{array}
$$

## 2.2.3 Substitution

回忆一下，如果 Q 是 length 为 r 的 profile，$P_1,\ldots,P_r$ 是 profile，则前文定义了 profile

$$
(P _ {1}, \ldots , P _ {r}) _ {Q}
$$

称为 $P_1,\ldots,P_r$ 的 Q-substitution。把 Q 的第 i 个条目替换为 profile $P_i$，即可得到该 profile。可以按如下方式把它扩展成 nested tuple 上的操作。

定义 2.2.3.1。假设 $X_1,\ldots,X_m$ 是 profile 分别为 $P_1,\ldots,P_m$ 的 nested tuple，Q 是 length 为 m 的 profile。定义 Q-substitution

$$
(X _ {1}, \ldots , X _ {m}) _ {Q}
$$

为具有以下 flattening

$$
(X _ {1}, \ldots , X _ {m}) _ {Q} ^ {\flat} = X _ {1} ^ {\flat} \star \dots \star X _ {m} ^ {\flat}
$$

和 profile

$$
(P _ {1}, \ldots , P _ {m}) _ {Q}.
$$

的 nested tuple。更一般地，如果 $X_1,\ldots,X_m$ 是 nested tuple，Y 是 length 为 m 的 nested tuple，则定义

$$
(X _ {1}, \dots , X _ {m}) _ {Y} = (X _ {1}, \dots , X _ {m}) _ {\operatorname{prof} (Y)}.
$$

示例 2.2.3.2。如果 $(X_1,X_2,X_3)=(64,16,4)$，$Q=(*,(*,*))$，则

$$
(X _ {1}, X _ {2}, X _ {3}) _ {Q} = (6 4, (3 2, 4))
$$

示例 2.2.3.3。如果 $(X_1,X_2,X_3,X_4)=((2,2),(3,3),(5,5),(7,7))$，$Q=((* ,*),(*,*))$，则

$$
(X _ {1}, X _ {2}, X _ {3}, X _ {4}) _ {Q} = (((2, 2), (3, 3)), ((5, 5), (7, 7))).
$$

示例 2.2.3.4。如果 `X=(12)` 且 $Q=*$，则

$$
(X) _ {Q} = 1 2.
$$

示例 2.2.3.5。如果 $X_1=2$、$X_2=2$、$X_3=(5,5)$，且 $Q=(*,*,*)$，则

$$
(X _ {1}, X _ {2}, X _ {3}) _ {Q} = (2, 2, (5, 5)) = (X _ {1}, X _ {2}, X _ {3}).
$$

更一般地，如果 $X_1,\ldots,X_m$ 是任意 nested tuple，且 $P=(*,\ldots,*)$，则

$$
(X _ {1}, \ldots , X _ {m}) _ {Q} = (X _ {1}, \ldots , X _ {k})
$$

是 $X_1,\ldots,X_m$ 的 concatenation。

旁注 2.2.3.6。nested tuple 的 substitution 有一种 operad 解释。整数 nested tuple 的集合 Nest(ℤ) 是 operad Profile 上的 algebra，其 structure map 由 Q-substitution 给出：

$$
\operatorname{Nest} (\mathbb {Z}) \times \dots \times \operatorname{Nest} (\mathbb {Z}) \times \operatorname{Profile} (n) \longrightarrow \operatorname{Nest} (\mathbb {Z})
$$

$$
(X _ {1}, \dots , X _ {m}), Q \longmapsto (X _ {1}, \dots , X _ {m}) _ {Q}.
$$

## 2.2.4 Refinement

本节介绍 nested tuple 上一种称为 refinement 的重要关系。直观而言，如果 $X'$ 和 X 是整数 nested tuple，而且可以把 X 的每个 entry 替换成同样 size 的某个 nested tuple，从而得到 $X'$，就称 $X'$ refine X。更精确地，有以下定义。

定义 2.2.4.1。如果 $X'$ 和 X 是 nested tuple，则在以下任一条件成立时，称 $X'$ refine X：

1. $X=\mathsf{size}(X')$；或

2. (a) $\mathsf{depth}(X'),\mathsf{depth}(X)>0$；

(b) $\mathsf{rank}(X')=\mathsf{rank}(X)$；

(c) 对每个 $1\leq i\leq\mathsf{rank}(X)$，mode<sub>i</sub>(X<sup>′</sup>) refine mode<sub>i</sub>(X)。

记号 2.2.4.2。记

$$
X ^ {\prime} \twoheadrightarrow X
$$

表示 $X'$ refine X。

示例 2.2.4.3。下面是一些 nested tuple refinement 示例。

$$
\begin{array}{c} (2, (2, 2)) \twoheadrightarrow 8 \\ ((2, 2), (3, 3), (5, 5)) \twoheadrightarrow (4, 9, 2 5) \\ (6 4) \twoheadrightarrow 6 4 \\ (8, ((2, 2, 2), ((1, 4), (2, 2)))) \twoheadrightarrow (8, (8, 8)) \end{array}
$$

观察 2.2.4.4。nested tuple 的 refinement 具有自反性、传递性和反对称性，因此 refinement 在正整数 nested tuple 集合上规定了一个偏序。

如果 $X'$ refine X，可以把 $X'$ 理解为：用 size 为 $x_i$ 的某个 nested tuple $X_i'$ 替换 X 的每个 entry $x_i$ 后得到的结果。称 nested tuple $X_i'$ 为 $X'$ relative to X 的第 i 个 mode。更精确地，有以下定义。

构造 2.2.4.5。假设 X 是 length 为 m 的整数 nested tuple，$X'$ refine X。对任意 $1\leq i\leq m$，定义 nested tuple

$$
X _ {i} ^ {\prime} = \mathsf {m o d e} _ {i} (X ^ {\prime}, X),
$$

并称其为 $X'$ relative to X 的第 i 个 mode，公式为

$$
\mathsf {m o d e} _ {i} (X ^ {\prime}, X) = \left\{ \begin{array}{l l} X ^ {\prime} & \mathsf {d e p t h} (X) = 0 (\text {hence} i = \ell = 1) \\ \mathsf {m o d e} _ {i - N} (\mathsf {m o d e} _ {j} (X ^ {\prime}), \mathsf {m o d e} _ {j} (X)) & j \text {is the largest integer such that} \\ & N := \mathsf {l e n} _ {<   j} (X) <   i. \end{array} \right.
$$

示例 2.2.4.6。如果 $X=((4,9),(25,36))$，且 $X'=(((2,2),(3,3)),(25,(6,(2,3))))$，则 $X'$ refine X，并且 $X'$ relative to X 的 mode 为

$$
\begin{array}{l} \text {mode} _ {1} (X ^ {\prime}, X) = (2, 2) \\ \text {mode} _ {2} (X ^ {\prime}, X) = (3, 3) \\ \text {mode} _ {3} (X ^ {\prime}, X) = 2 5 \\ \text {mode} _ {4} (X ^ {\prime}, X) = (6, (2, 3)). \end{array}
$$

示例 2.2.4.7。如果 X 是任意 nested tuple，则 X refine X，而且对任意 $1\leq i\leq\mathsf{len}(X)$，有

$$
\operatorname{mode} _ {i} (X, X) = \operatorname{entry} _ {i} (X).
$$

示例 2.2.4.8。如果 $X=X^\flat$ 是 tuple，且 $X'$ refine X，则对任意 $1\leq i\leq\mathsf{len}(X)$，有

$$
\operatorname{mode} _ {i} \left(X ^ {\prime}, X\right) = \operatorname{mode} _ {i} \left(X ^ {\prime}\right).
$$

示例 2.2.4.9。如果 $X'$ 是满足 $\mathsf{size}(X')=N$ 的 nested tuple，则 $X'$ refine N，而且 $X'$ relative to N 的唯一 mode 是

$$
\operatorname{mode} _ {1} \left(X ^ {\prime}, N\right) = X ^ {\prime}.
$$

记号 2.2.4.10。如果 $X'\twoheadrightarrow X$ 是 refinement，且 $1\leq i\leq\mathsf{len}(X)$，则记

$$
\begin{array}{c} \operatorname{len} _ {i} (X ^ {\prime}, X) = \operatorname{len} (\operatorname{mode} _ {i} (X ^ {\prime}, X)) \\ \operatorname{len} _ {<   i} (X ^ {\prime}, X) = \sum_ {j <   i} \operatorname{len} _ {j} (X ^ {\prime}, X) \\ \operatorname{len} _ {\leq i} (X ^ {\prime}, X) = \sum_ {j \leq i} \operatorname{len} _ {j} (X ^ {\prime}, X) \end{array}
$$

定义 2.2.4.11。假设 $X'$ refine X，并记 $X_i'=\mathsf{mode}_i(X',X)$。则 $X'$ relative to X 的 flattening 是 nested tuple

$$
\operatorname{flat} \left(X ^ {\prime}, X\right) = \left(X _ {1} ^ {\prime}, \dots , X _ {m} ^ {\prime}\right).
$$

示例 2.2.4.12。如果 $X'=(((2,2),(3,3)),((5,5),(7,7)))$，且 $X=((4,9),(25,49))$，则

$$
\operatorname{flat} \left(X ^ {\prime}, X\right) = ((2, 2), (3, 3), (5, 5), (7, 7)).
$$

示例 2.2.4.13。如果 X 是任意 nested tuple，则 X relative to X 的 flattening 为

$$
\operatorname{flat} (X, X) = X ^ {\flat}.
$$

示例 2.2.4.14。如果 $X=X^\flat$ 是 tuple，且 $X'$ refine X，则 $X'$ relative to X 的 flattening 为

$$
\operatorname{flat} \left(X ^ {\prime}, X\right) = X ^ {\prime}.
$$

示例 2.2.4.15。如果 $X'$ 是满足 `size(X′)=N` 的 nested tuple，则 $X'$ refine N，而且 $X'$ relative to N 的 flattening 为

$$
\operatorname{flat} \left(X ^ {\prime}, N\right) = (N).
$$

观察 2.2.4.16。如果 $X'$ refine X，则 $\mathsf{flat}(X',X)$ refine $X^\flat$。

## 2.3 Layout

建立 nested tuple 的必要背景后，下面转向 layout。Layout 是 flat layout 的推广，其中 shape 和 stride 可以是 nested tuple，而不只是 flat tuple。

## 2.3.1 基本定义

定义 2.3.1.1。layout 是一对

$$
L = S: D
$$

它由正整数 nested tuple

$$
\operatorname{shape} (L) = S
$$

与非负整数 nested tuple

$$
\operatorname{stride} (L) = D
$$

组成；前者称为 L 的 shape，后者称为 L 的 stride，并要求 S 与 D congruent。

定义 2.3.1.2。如果 $L=S:D$ 是 layout，则 L 的 rank、length、depth、size 和 profile 分别定义为 S 的 rank、length、depth、size 和 profile。

示例 2.3.1.3。layout $L=(3,(3,2)):(3,(1,10))$ 可以描绘如下。

<table><tr><td>0</td><td>1</td><td>2</td><td>10</td><td>11</td><td>12</td></tr><tr><td>3</td><td>4</td><td>5</td><td>13</td><td>14</td><td>15</td></tr><tr><td>6</td><td>7</td><td>8</td><td>16</td><td>17</td><td>18</td></tr></table>

示例 2.3.1.4。layout `L=((2,2),(2,2)):((1,4),(2,8))` 可以描绘如下。

<table><tr><td>0</td><td>2</td><td>8</td><td>10</td></tr><tr><td>1</td><td>3</td><td>9</td><td>11</td></tr><tr><td>4</td><td>6</td><td>12</td><td>14</td></tr><tr><td>5</td><td>7</td><td>13</td><td>15</td></tr></table>

示例 2.3.1.5。layout

$$
L = 1 0: 4
$$

满足 `rank(L)=1`、`len(L)=1`、`depth(L)=0`、`size(L)=10`、`prof(L)=∗`。

示例 2.3.1.6。layout

$$
L = (7, (2, 1 0, 4), (3, 7)): (1, (7, 1 4, 1 4 0), (5 6 0, 1 6 8 0))
$$

$$
\text { has   } \operatorname{rank} (L) = 3, \text {   len } (L) = 6, \text {   depth } (L) = 2, \text {   size } (L) = 1 1 7 6 0, \text {   and   } \operatorname{prof} (L) = (*, (*, *, *), (*, *))
$$

示例 2.3.1.7。layout

$$
L = ((2, 2, 2, (2, 2))): ((1, 0, 8, (0, 1 6)))
$$

$$
\text { has   } \operatorname{rank} (L) = 1, \operatorname{len} (L) = 5, \operatorname{depth} (L) = 3, \operatorname{size} (L) = 3 2, \text {   and   } \operatorname{prof} (L) = ((*, *, *, (*, *))).
$$

示例 2.3.1.8。pair

$$
S: D = (2, (2, 2)): (1, 2, 4)
$$

不是 layout，因为 S 与 D 不 congruent。

定义 2.3.1.9。如果 $L=S:D$ 是 layout，则对任意 $1\leq i\leq\mathsf{rank}(L)$，把 L 的第 i 个 mode 定义为 layout

$$
\operatorname{mode} _ {i} (L) = \operatorname{mode} _ {i} (S): \operatorname{mode} _ {i} (D),
$$

对任意 $1\leq i\leq\mathsf{len}(L)$，把 L 的第 i 个 entry 定义为 layout

$$
\operatorname{entry} _ {i} (L) = \operatorname{entry} _ {i} (S): \operatorname{entry} _ {i} (D).
$$

示例 2.3.1.10。如果 $L=((2,2),9):((3,6),12)$，则 L 的 mode 为

$$
\begin{array}{l} \text { mode } _ {1} (L) = (2, 2): (3, 6) \\ \text { mode } _ {2} (L) = 9: 1 2 \end{array}
$$

而 L 的 entry 为

$$
\begin{array}{l} \text {entry} _ {1} (L) = 2: 3 \\ \text {entry} _ {2} (L) = 2: 6 \\ \text {entry} _ {3} (L) = 9: 1 2. \end{array}
$$

注记 2.3.1.11。如果 L 是 layout，则 L 的 mode 也是 layout，L 的 entry 是 depth 0 的 layout。

注记 2.3.1.12。flat layout L 恰好是 depth 1 的 layout。另一方面，如果 L 是 layout，可以按如下方式得到 flat layout $L^\flat$。

定义 2.3.1.13。如果 $L=S:D$ 是 layout，把 L 的 flattening 定义为 flat layout

$$
L ^ {\flat} = S ^ {\flat}: D ^ {\flat}.
$$

示例 2.3.1.14。$L=10:4$ 的 flattening 是 $L^\flat=(10):(4)$。

示例 2.3.1.15。layout

$$
L = \left((2, 2, 2, (2, 2))\right): \left((1, 0, 8, (0, 1 6))\right)
$$

的 flattening 是

$$
L ^ {\flat} = (2, 2, 2, 2, 2): (1, 0, 8, 0, 1 6).
$$

注记 2.3.1.16。如果 L 是 layout，则 $\mathsf{len}(L)=\mathsf{rank}(L^\flat)$，并且对任意 $1\leq i\leq\mathsf{len}(L)$，有

$$
\operatorname{entry} _ {i} (L) = \operatorname{mode} _ {i} \left(L ^ {\flat}\right).
$$

可以使用上述 flattening 构造，把许多概念从 flat layout 扩展到嵌套 layout。例如：

构造 2.3.1.17（Layout function）。如果 L 是嵌套 layout，把 L 的 layout function $\Phi_L$ 定义为

$$
\Phi_ {L} = \Phi_ {L ^ {\flat}},
$$

其中 $\Phi_{L^\flat}$ 是构造 2.1.2.19 的 layout function。类似地，如果 N 满足 $\mathsf{Image}(\Phi_L)\subset[0,N)$，则定义

$$
\Phi_ {L} ^ {N} = \Phi_ {L ^ {\flat}} ^ {N}
$$

为 $\Phi_L$ 通过 inclusion $[0,N)\subset\mathbb{Z}$ 的分解。

示例 2.3.1.18。如果 $L=((2,2),2):((3,0),10)$，则 L 的 layout function

$$
\Phi_ {L}: [ 0, 8) \to \mathbb {Z}
$$

定义为

$$
\Phi_ {L} \begin{array}{c c c c c c c c c} & 0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 \\ & \Big \downarrow & \Big \downarrow & \Big \downarrow & \Big \downarrow & \Big \downarrow & \Big \downarrow & \Big \downarrow & \Big \downarrow \\ & 0 & 3 & 0 & 3 & 1 0 & 1 3 & 1 0 & 1 3 \end{array}
$$

给定 layout L，可以得到 flat layout $L^\flat$ 和 profile $P=\mathsf{prof}(L)$。反之，如果给定 flat layout L 和一个与 L length 相同的 profile P，则可按如下方式构造 flattening 为 L、profile 为 P 的 layout。

构造 2.3.1.19。如果 L 是 flat layout，P 是满足 `len(P)=len(L)` 的 profile，则可以定义

$$
L = L _ {P}
$$

为具有 shape

$$
\operatorname{shape} (L) = \operatorname{shape} (L) _ {P}
$$

和 stride

$$
\operatorname{stride} (L) = \operatorname{stride} (L) _ {P}
$$

的 layout，其中 $(-)_P$ 是定义 2.2.1.8 的 P-substitution 操作。

示例 2.3.1.20。如果 $L=(8,8,8):(1,64,8)$，$P=(*,(*,*))$，则

$$
L _ {P} = (8, (8, 8)), (1, (6 4, 8)).
$$

示例 2.3.1.21。如果 $L=(128):(2)$，$P=*$，则

$$
L _ {P} = 1 2 8: 2.
$$

命题 2.3.1.22。如果 L<sup>′</sup> 是 flat layout，P 是满足 `len(L′)=len(P)` 的 profile，则存在唯一 layout L，其 flattening 为 $L^\flat=L'$，profile 为 `prof(L)=P`；该 layout 就是 $L=L_P'$。

证明。由 nested tuple 的定义可得，因为 nested tuple 由其 flattening 和 profile 唯一确定。□

观察 2.3.1.23。前一命题说明存在 pullback square

$$
\begin{array}{c} \text {Layout} \xrightarrow {\text {prof(-)}} \text {Profile} \\ (-) ^ {b} \Biggl \downarrow \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \text {len(-)} \\ \text {FlatLayout} \xrightarrow {\text {len(-)}} \mathbb {N} \end{array}
$$

可以按如下方式把 non-degeneracy 概念扩展到嵌套情况。

定义 2.3.1.24。假设 L 是 layout。如果对所有 $1\leq i\leq\mathsf{len}(L)$，以下条件都成立，就称 L 是 non-degenerate：

$$
\operatorname{entry} _ {i} (\operatorname{shape} (L)) \quad \Rightarrow \quad \operatorname{entry} _ {i} (\operatorname{stride} (L))
$$

示例 2.3.1.25。以下 layout

$$
\begin{array}{l} L _ {1} = ((2, 2), 1): ((1, 2), 0) \\ L _ {2} = ((8, 8), (1, 1 6)): ((2, 3 2), (0, 1 2 8)) \end{array}
$$

是 non-degenerate，而以下 layout

$$
\begin{array}{l} L _ {3} = ((2, 2), 1): ((1, 2), 4) \\ L _ {4} = ((8, 8), (1, 1 6)): ((2, 3 2), (1 0 2 4, 1 2 8)) \end{array}
$$

是 degenerate。

## 2.3.2 基本操作

建立 layout 的基本词汇后，下面转向其支持的操作。本节定义一些基本操作，后续构造 coalesce、complement、composition、logical division 和 logical product 等更复杂操作时会用到它们。

## 2.3.2.1 Flattening

如果 L 是 layout，可以通过 flatten L 的 shape 和 stride 得到 flat layout $L^\flat$。

定义 2.3.2.1。如果 $L=S:D$ 是 layout，把 L 的 flattening 定义为 flat layout

$$
L ^ {\flat} = S ^ {\flat}: D ^ {\flat}.
$$

示例 2.3.2.2。layout

$$
L = ((2, 2, 2, (2, 2))): ((1, 0, 8, (0, 1 6)))
$$

的 flattening 是

$$
L ^ {\flat} = (2, 2, 2, 2, 2): (1, 0, 8, 0, 1 6).
$$

示例 2.3.2.3。$L=10:4$ 的 flattening 是 $L^\flat=(10):(4)$。

示例 2.3.2.4。假设 L 是 layout。则 `depth(L)=1` 当且仅当 $L=L^\flat$。

## 2.3.2.2 Concatenate

可以通过 concatenate 各自的 shape 和 stride 来 concatenate layout。

定义 2.3.2.5。如果 $L=S:D$ 和 $L'=S':D'$ 是 layout，则 L 与 L′ 的 concatenation 是 layout `(L,L′)`：

$$
(L, L ^ {\prime}) = (S, S ^ {\prime}): (D, D ^ {\prime}).
$$

更一般地，如果 $L_1,\ldots,L_k$ 是任意有限 layout 集合，且 $L_i=S_i:D_i$，则 $L_1,\ldots,L_k$ 的 concatenation 是 layout

$$
(L _ {1}, \dots , L _ {k}) = (S _ {1}, \dots , S _ {k}): (D _ {1}, \dots , D _ {k}).
$$

注记 2.3.2.6。nested tuple 的 concatenation——因而 layout 的 concatenation——不满足结合律。例如，取 $L_1=3:4$、$L_2=2:2$、$L_3=5:1$，则

$$
\left(L _ {1}, \left(L _ {2}, L _ {3}\right)\right) = (3, (2, 5)): (4, (2, 1)) \neq ((3, 2), 5): ((4, 2), 1) = \left(\left(L _ {1}, L _ {2}\right), L _ {3}\right).
$$

此外，这两个 layout 都不等于“三重”concatenation `(L1,L2,L3)=(3,2,5):(4,2,1)`。不过，它们具有相同 flattening，因此也具有相同 layout function。

示例 2.3.2.7。如果 $L=(3,7,2):(1,3,6)$，$L'=(2,(2,(4,3))):(5,3,(2,2))$，则

$$
(L, L ^ {\prime}) = ((3, 7, 2), (2, (2, (4, 3)))): ((1, 3, 6), (5, (3, (2, 2))))
$$

注记 2.3.2.8。Concatenation 会增加 layout 的 depth。更精确地，

$$
\operatorname{depth} (L, L ^ {\prime}) = 1 + \max (\operatorname{depth} (L), \operatorname{depth} (L ^ {\prime})).
$$

注记 2.3.2.9。当 L 和 L′ 是 flat layout 时，定义 2.3.2.5 的 concatenation 与定义 2.1.3.36 的 flat layout concatenation 并不相同。二者满足关系

$$
L \star L ^ {\prime} = (L, L ^ {\prime}) ^ {\flat}.
$$

注记 2.3.2.10。如果 L 是满足 `depth(L)>0` 且 `rank(L)=r` 的任意 layout，则可写成

$$
L = (\operatorname{mode} _ {1} (L), \dots , \operatorname{mode} _ {r} (L))
$$

即其各 mode 的 concatenation。

示例 2.3.2.11。如果

$$
L = ((5, (7, 7)), 2, (4, 5)): ((1, (3 5, 5)), 0, (1, 8))
$$

则 $L=(L_1,L_2,L_3)$，其中

$$
\begin{array}{l} L _ {1} = \big (5, (7, 7) \big): \big (1, (3 5, 5) \big), \\ L _ {2} = 2: 0, \text {and} \\ L _ {3} = (4, 5): (1, 8). \end{array}
$$

## 2.3.2.3 Substitution

回忆一下，如果 $X_1,\ldots,X_k$ 是 nested tuple，P 是满足 `len(P)=k` 的 profile，则可以形成 P-substitution

$$
(X _ {1}, \ldots , X _ {k}) _ {P}
$$

它通过把 P 的第 i 个条目替换为 nested tuple $X_i$ 得到。可以按如下方式把该构造从 nested tuple 扩展到 layout。

定义 2.3.2.12。假设 $L=S:D$ 是 layout，P 是满足 `len(P)=rank(L)` 的 profile。定义

$$
L _ {P} = S _ {P}: D _ {P}
$$

其中 $S_P$ 和 $D_P$ 是 S 与 D 的各 mode 的 P-substitution。

示例 2.3.2.13。如果 $P=(*,(*,*))$，$L=(8,8,8):(1,8,64)$，则

$$
L _ {P} = (8, (8, 8)): (1, (8, 6 4)).
$$

示例 2.3.2.14。如果 $P=(*,(*,*))$，并且

$$
L = ((2, 2), (3, 3), (5, 5)): ((2, 1), (1 2, 4), (1 8 0, 3 6)),
$$

则

$$
L _ {P} = ((2, 2), ((3, 3), (5, 5))): ((2, 1), ((1 2, 4), (1 8 0, 3 6))).
$$

示例 2.3.2.15。如果 $L=(16):(1)$，$P=*$，则

$$
L _ {P} = 1 6: 1.
$$

## 2.3.3 Coalesce

回忆一下，如果 L 是 flat layout，则 $\mathsf{coal}^\flat(L)$ 是 layout function 为 $\Phi_L$ 且 rank 最小的唯一 flat layout。在任意嵌套 layout 的情形下，可以作类似构造。首先定义 coalesced layout。

定义 2.3.3.1。假设 L 是 layout。如果以下条件之一成立，就称 L 为 coalesced：

1. $L = 1 : 0 ,$ 

2. `depth(L)=0` 且 `shape(L)>1`；或

3. `depth(L)=1`、`rank(L)>1`，并且 L 在定义 2.1.4.1 的意义下是 coalesced。

示例 2.3.3.2。layout

$$
L = (2, (2, 2)): (1, (1 6, 5 1 2))
$$

不是 coalesced，因为 `depth(L)>1`。

示例 2.3.3.3。layout

$$
L = (6 4): (2)
$$

不是 coalesced，而 layout

$$
L ^ {\prime} = 6 4: 2
$$

是 coalesced。

示例 2.3.3.4。layout

$$
L = 1: 8
$$

不是 coalesced，而 layout

$$
L ^ {\prime} = 1: 0
$$

是 coalesced。

示例 2.3.3.5。空 layout

$$
E = (): ()
$$

不是 coalesced。

观察 2.3.3.6。回忆一下，如果

$$
\operatorname{entry} _ {i} (\operatorname{shape} (L)) = 1 \quad \Rightarrow \quad \operatorname{entry} _ {i} (\operatorname{stride} (L)) = 0.
$$

则 layout L 是 non-degenerate。如果 L 是 coalesced，则 L 是 non-degenerate。

如果 L 是任意 layout，可以按如下方式得到 coalesced layout `coal(L)`。

构造 2.3.3.7。假设 L 是 layout，并写成

$$
\operatorname{coal} ^ {\flat} \left(L ^ {\flat}\right) = \left(s _ {1}, \dots , s _ {m}\right): \left(d _ {1}, \dots , d _ {m}\right).
$$

1. 如果 $m>1$，定义

$$
\operatorname{coal} (L) = \operatorname{coal} ^ {\flat} \left(L ^ {\flat}\right)
$$

2. 如果 $m=1$，定义

$$
\operatorname{coal} (L) = s _ {1}: d _ {1}
$$

3. 如果 $m=0$，定义

$$
\operatorname{coal} (L) = 1: 0.
$$

示例 2.3.3.8。如果 $E=():()` 是空 layout，则

$$
\operatorname{coal} (E) = 1: 0.
$$

示例 2.3.3.9。如果 $L=(1,1):(2,4)$，则

$$
\operatorname{coal} (L) = 1: 0.
$$

示例 2.3.3.10。如果 $L=(512):(4)$，则

$$
\operatorname{coal} (L) = 5 1 2: 4.
$$

示例 2.3.3.11。如果 $L=(2,2,2):(1,2,4)$，则

$$
\operatorname{coal} (L) = 8: 1.
$$

示例 2.3.3.12。如果 $L=((2,2,2),(5,5)):((1,2,4),(10,50))$，则

$$
\operatorname{coal} (L) = (8, 2 5): (1, 1 0).
$$

注记 2.3.3.13。如果 L 是 layout，则 `coal(L)` 的 depth 为 0 或 1。

命题 2.3.3.14。如果 A 和 B 是 layout，则

$$
\Phi_ {A} = \Phi_ {B} \quad \Leftrightarrow \quad \operatorname{coal} (A) = \operatorname{coal} (B).
$$

证明。使用命题 2.1.4.18，有

$$
\begin{array}{l l l} \Phi_ {A} = \Phi_ {B} & \Leftrightarrow & \Phi_ {A ^ {\flat}} = \Phi_ {B ^ {\flat}} \\ & \Leftrightarrow & \operatorname{coal} ^ {\flat} (A ^ {\flat}) = \operatorname{coal} ^ {\flat} (B ^ {\flat}) \\ & \Leftrightarrow & \operatorname{coal} (A) = \operatorname{coal} (B). \end{array}
$$

定义 2.3.3.15。如果 L 是 layout，把 L 的 complexity 定义为整数

$$
\text { complexity } (L) = \text { len } (L) + \text { depth } (L).
$$

命题 2.3.3.16。如果 L 是 layout 且 `size(L)>1`，则 `coal(L)` 是 layout function 为 $\Phi_L$ 且 complexity 最小的唯一 layout。

证明。假设 L′ 是与 L 具有相同 layout function 的 layout，并假设 $\mathsf{coal}(L')\neq1:0$。则

$$
\operatorname{len} \left(L ^ {\prime}\right) \geq \operatorname{len} (\operatorname{coal} \left(L ^ {\prime}\right)) = \operatorname{len} (\operatorname{coal} (L)).
$$

需要考虑两种情况。

• 情况 1：假设 `len(L′)>1`。则 `depth(L′)≥1≥depth(coal(L))`。结合这些不等式可知

$$
\text { complexity } (L ^ {\prime}) \geq \text { complexity } (\text { coal } (L)),
$$

其中等号成立，当且仅当 $L'=\mathsf{coal}(L')=\mathsf{coal}(L)$。

• 情况 2：假设 `len(L′)=1`。则对某些整数 $s>1$ 和 $d\geq0$，有 $L'=(s):(d)$ 或 $L'=s:d$。两种情况下都有 $\mathsf{coal}(L')=s:d$，并且

$$
\text { complexity } (L ^ {\prime}) \geq \text { complexity } (\text { coal } (L)),
$$

其中等号成立，当且仅当 $L'=s:d=\mathsf{coal}(L)$。

注记 2.3.3.17。之所以需要排除 `size(L)=1`，唯一原因是：如果 `size(L)=1`，则 `1:0` 和空 layout `():()` 是两个不同 layout，二者都具有最小 complexity，且与 L 具有相同 layout function，即平凡 layout function $0\mapsto0$。

## 2.3.4 Relative coalesce

coalesce 有一个重要变体称为 relative coalesce，记作 $\mathsf{coal}(L,\bar S)$。该操作额外接收 nested tuple $\bar S$ 作为输入，并要求 `shape(L)` refine $\bar S$。relative coalesce 在确保所得 shape 仍 refine $\bar S$ 的同时，尽可能简化 layout L。

定义 2.3.4.1。假设 $L=S:D$ 是 layout，$\bar S$ 是 length 为 m、被 S refine 的 nested tuple。回忆一下，对任意 $1\leq i\leq m$，可以考察 S relative to $\bar S$ 的第 i 个 mode，记作

$$
\operatorname{mode} _ {i} (S, \bar {S}).
$$

由于 S 与 D congruent，存在 nested tuple

$$
\mathsf {m o d e} _ {i} (D, \bar {S})
$$

与 $\mathsf{mode}_i(S,\bar S)$ 对应，并把 L relative to $\bar S$ 的第 i 个 mode 定义为 layout

$$
\operatorname{mode} _ {i} (L, \bar {S}) = \operatorname{mode} _ {i} (S, \bar {S}): \operatorname{mode} _ {i} (D, \bar {S}).
$$

示例 2.3.4.2。如果 $\bar S=(4,(9,25))$，且

$$
L = ((2, 2), ((3, 3), (5, (1, 5)))): ((1, 2), ((6, 1 8), (9 0, (0, 4 5 0))))
$$

则

$$
\begin{array}{l} \text {mode} _ {1} (L, \bar {S}) = (2, 2): (1, 2) \\ \text {mode} _ {2} (L, \bar {S}) = (3, 3): (6, 1 8) \\ \text {mode} _ {3} (L, \bar {S}) = (5, (1, 5)): (9 0, (0, 4 5 0)). \end{array}
$$

观察 2.3.4.3。假设 $L=S:D$ 是 layout，$\bar S$ 是 length 为 m、profile 为 P 且被 S refine 的 nested tuple。对任意 $1\leq i\leq m$，如果记

$$
L _ {i} = \operatorname{mode} _ {i} (L, \bar {S}),
$$

则

$$
L = (L _ {1}, \dots , L _ {m}) _ {P}
$$

是其 relative mode 的 P-substitution。

定义 2.3.4.4。假设 $L=S:D$ 是 layout，$\bar S$ 是 length 为 m、profile 为 P 且被 S refine 的 nested tuple。如果每个 relative mode

$$
\mathsf {m o d e} _ {i} (L, \bar {S})
$$

都是 coalesced，就称 L 在 $\bar S$ 上 coalesced。

观察 2.3.4.5。在定义 2.3.4.4 的设定下，如果 L 在 $\bar S$ 上 coalesced，则 L 是 non-degenerate。

示例 2.3.4.6。如果 L 是 layout，则 L 在 `shape(L)` 上 coalesced，当且仅当 L 是 non-degenerate，即

$$
\operatorname{entry} _ {i} (\operatorname{shape} (L)) = 1 \quad \Rightarrow \quad \operatorname{entry} _ {i} (\operatorname{stride} (L)) = 0.
$$

定义 2.3.4.7（Relative coalesce）。假设 $L=S:D$ 是 layout，$\bar S$ 是 length 为 m、profile 为 P 且被 S refine 的 nested tuple。定义

$$
\operatorname{coal} (L, \bar {S}) = (\operatorname{coal} (L _ {1}), \dots , \operatorname{coal} (L _ {m})) _ {P}.
$$

注记 2.3.4.8。在定义 2.3.4.7 的设定下，$\mathsf{coal}(L,\bar S)$ 的 shape refine $\bar S$。

引理 2.3.4.9。如果 $L=S:D$ 是 layout，且 S refine $\bar S$，则

$$
\Phi_ {\mathrm{coal} (L, \bar {S})} = \Phi_ {L}.
$$

证明。与上文相同，令

$$
L _ {i} = \operatorname{mode} _ {i} (L, S)
$$

表示 L relative to $\bar S$ 的第 i 个 mode，并令 $\bar L_i=\mathsf{coal}(L_i)$。则

$$
\begin{array}{r l} \Phi_ {\mathsf {c o a l} (L, \bar {S})} & = \Phi_ {(\bar {L} _ {1}, \dots , \bar {L} _ {m}) _ {\bar {S}}} \\ & = \Phi_ {(\bar {L} _ {1}, \dots , \bar {L} _ {m})} \\ & = \Phi_ {\mathsf {c o a l} ((\bar {L} _ {1}, \dots , \bar {L} _ {m}))} \\ & = \Phi_ {\mathsf {c o a l} ((L _ {1}, \dots , L _ {m}))} \\ & = \Phi_ {(L _ {1}, \dots , L _ {m})} \\ & = \Phi_ {(L _ {1}, \dots , L _ {m}) _ {\bar {S}}} \\ & = \Phi_ {L}. \end{array}
$$

命题 2.3.4.10。假设 A 和 B 是 layout，$\bar S$ 是 length 为 m 的 nested tuple，并且 `shape(A)` 与 `shape(B)` 都 refine $\bar S$。则

$$
\Phi_ {A} = \Phi_ {B} \quad \Leftrightarrow \quad \mathsf {c o a l} (A, \bar {S}) = \mathsf {c o a l} (B, \bar {S})
$$

证明。如果 $\mathsf{coal}(A,\bar S)=\mathsf{coal}(B,\bar S)$，则使用引理 2.3.4.9，有

$$
\Phi_ {A} = \Phi_ {\mathsf {c o a l} (A, \bar {S})} = \Phi_ {\mathsf {c o a l} (B, \bar {S})} = \Phi_ {B}.
$$

反之，假设 $\Phi_A=\Phi_B$。我们将证明 $\mathsf{coal}(A,\bar S)=\mathsf{coal}(B,\bar S)$。令 $P=\mathsf{prof}(\bar S)$，并对任意 $1\leq i\leq m$，令

$$
\begin{array}{l} A _ {i} = \mathsf {m o d e} _ {i} (A, \bar {S}) \\ B _ {i} = \mathsf {m o d e} _ {i} (B, \bar {S}). \end{array}
$$

Since 

$$
\operatorname{coal} (A, \bar {S}) = (\operatorname{coal} (A _ {1}), \dots , \operatorname{coal} (A _ {m})) _ {P}
$$

and 

$$
\operatorname{coal} (B, \bar {S}) = (\operatorname{coal} (B _ {1}), \dots , \operatorname{coal} (B _ {m})) _ {P}
$$

只需对所有 $1\leq i\leq m$ 证明 $\mathsf{coal}(A_i)=\mathsf{coal}(B_i)$。根据 colexicographic isomorphism 的结合律，可以把 A 的 layout function $\Phi_A$ 写成

$$
[ 0, \operatorname{size} (A)) \xrightarrow {\operatorname{colex} ^ {- 1}} \prod_ {j = 1} ^ {m} [ 0, \operatorname{size} (A _ {j})) \xrightarrow {\prod \Phi_ {A _ {j}}} \prod_ {j = 1} ^ {m} \mathbb {Z} \xrightarrow {+} \mathbb {Z}
$$

并把 B 的 layout function $\Phi_B$ 写成

$$
[ 0, \operatorname{size} (B)) \xrightarrow {\operatorname{colex} ^ {- 1}} \prod_ {j = 1} ^ {m} [ 0, \operatorname{size} (B _ {j})) \xrightarrow {\prod \Phi_ {B _ {j}}} \prod_ {j = 1} ^ {m} \mathbb {Z} \xrightarrow {+} \mathbb {Z}
$$

固定 $1\leq i\leq m$，考虑子集

$$
[ 0, \operatorname{size} (A _ {i})) \subset \prod_ {j = 1} ^ {m} [ 0, \operatorname{size} (A _ {j}))
$$

及其像

$$
\operatorname{colex} ([ 0, \operatorname{size} (A _ {i}))) \subset [ 0, \operatorname{size} (A)).
$$

由于对所有 $1\leq j\leq m$ 都有 $\mathsf{size}(A_j)=\mathsf{size}(B_j)$，它与以下像相同：

$$
\operatorname{colex} ([ 0, \operatorname{size} (B _ {j}))) \subset [ 0, \operatorname{size} (B)) = [ 0, \operatorname{size} (B)).
$$

$\Phi_A$ 在该子集上的 restriction 是 $\Phi_{A_i}$，$\Phi_B$ 在该子集上的 restriction 是 $\Phi_{B_i}$，所以 $\Phi_{A_i}=\Phi_{B_i}$。根据命题 2.3.3.14，有 $\mathsf{coal}(A_i)=\mathsf{coal}(B_i)$。因此

$$
\operatorname{coal} (A, \bar {S}) = \operatorname{coal} (B, \bar {S}),
$$

这正是所需结论。

## 2.3.5 Compact layout

可以轻易把 compact layout 概念扩展到嵌套情况。仍然用描绘 layout 的标准 grid 图来说，如果每个整数 $0\leq i<\mathsf{size}(L)$ 恰好出现一次，则 layout L 是 compact 的。更精确地，有以下定义。

定义 2.3.5.1。假设 L 是 layout。如果 layout function

$$
\Phi_ {L} ^ {\text { cosize } (L)}: [ 0, \text { size } (L)) \to [ 0, \text { cosize } (L))
$$

是 isomorphism，就称 L 为 compact。

示例 2.3.5.2。layout

$$
A = ((2, 2), (2, 2)): ((1, 4), (2, 8)) =
$$

<table><tr><td>0</td><td>2</td><td>8</td><td>10</td></tr><tr><td>1</td><td>3</td><td>9</td><td>11</td></tr><tr><td>4</td><td>6</td><td>12</td><td>14</td></tr><tr><td>5</td><td>7</td><td>13</td><td>15</td></tr></table>

是 compact 的，而 layout

$$
B = ((2, 2), (2, 2)): ((1, 4), (2, 3 2)) =
$$

<table><tr><td>0</td><td>2</td><td>32</td><td>34</td></tr><tr><td>1</td><td>3</td><td>33</td><td>35</td></tr><tr><td>4</td><td>6</td><td>36</td><td>38</td></tr><tr><td>5</td><td>7</td><td>37</td><td>39</td></tr></table>

以及

$$
C = ((2, 2), (2, 2)): ((1, 4), (2, 0)) = \begin{array}{c c c c} \hline 0 & 2 & 0 & 2 \\ \hline 1 & 3 & 1 & 3 \\ \hline 4 & 6 & 4 & 6 \\ \hline 5 & 7 & 5 & 7 \\ \hline \end{array}
$$

不是 compact 的。

示例 2.3.5.3。以下 layout 是 compact 的：

$$
\begin{array}{l} L _ {1} = (2, (2, 2)): (8, (1, 4)) \\ L _ {2} = ((8, 1), (8, 3 2)): ((2, 0), (1 6, 1 2 8)) \\ L _ {3} = 6 4: 1 \end{array}
$$

示例 2.3.5.4。layout

$$
L = (2, (2, 2)): (4, (8, 1 6))
$$

不是 compact 的，因为整数 $1\in[0,29)=[0,\mathsf{cosize}(L))$ 不在 $\Phi_L$ 的像中。更一般地，如果 $\mathsf{size}(L)\neq\mathsf{cosize}(L)$，则 L 不是 compact 的。

本节最后列出 layout L 为 compact 的一些等价条件。

命题 2.3.5.5。假设 L 是 layout，则以下条件等价。

1. L 是 compact 的。

2. $L^\flat$ 是 compact 的。

3. `coal(L)` 是 compact 的。

证明。这些条件的等价性来自

$$
\Phi_ {L} = \Phi_ {L ^ {\flat}} = \Phi_ {\text { coal } (L)}.
$$

## 2.3.6 Complement

可以按如下方式轻易把 complement 概念扩展到嵌套情况。

定义 2.3.6.1。假设 A 和 B 是 layout。如果 concatenated layout `(A,B)` 是 compact 的，就称 B 是 A 的 complement，并记作 $A\perp B$。

引理 2.3.6.2。假设 A 和 B 是 layout，则

$$
A \perp B \quad \Leftrightarrow \quad A ^ {\flat} \perp B ^ {\flat}.
$$

证明。由观察 $(A,B)^\flat=A^\flat\star B^\flat$ 可得。

定义 2.3.6.3。假设 A 是 layout。如果 $A^\flat$ 是 complementable 的，就称 A 是 complementable 的。

引理 2.3.6.4。假设 A 是 layout。存在 A 的 complement B，当且仅当 A 是 complementable 的。

证明。如果 A 是 complementable 的，则 $A^\flat$ 是 complementable 的，因此存在 flat layout B，使 flat concatenation $A^\flat\star B$ 是 compact 的。于是 concatenation `(A,B)` 也是 compact 的，所以 A 存在 complement。反之，假设存在 layout B，使 `(A,B)` 是 compact 的。则 $B^\flat$ 是 $A^\flat$ 的 complement；根据命题 2.1.6.21，$A^\flat$ 是 complementable 的，所以根据定义，A 也是 complementable 的。□

定义 2.3.6.5。假设 A 是 layout。如果 A 是 complementable 的，则定义

$$
\operatorname{comp} (A) = \operatorname{coal} (\operatorname{comp} ^ {\flat} (A ^ {\flat})),
$$

如构造 2.1.6.16 所示。如果 N 是正整数，且 A 是 N-complementable 的，则定义

$$
\operatorname{comp} (A, N) = \operatorname{coal} (\operatorname{comp} ^ {\flat} (A ^ {\flat}, N))
$$

如构造 2.1.6.29 所示。

注记 2.3.6.6。假设 A 是 complementable layout。几乎总有 $\mathsf{comp}(A)=\mathsf{comp}^\flat(A^\flat)$。更精确地，如果 $\mathsf{comp}^\flat(A^\flat)$ 的 length 大于 1，则

$$
\operatorname{comp} (A) = \operatorname{comp} ^ {\flat} (A ^ {\flat}),
$$

如果 $\mathsf{comp}^\flat(A^\flat)=(s):(d)$ 的 length 为 1，则

$$
\operatorname{comp} (A) = s: d,
$$

如果 $\mathsf{comp}^\flat(A^\flat)=():()`，则

$$
\operatorname{comp} (A) = 1: 0.
$$

定义 2.3.6.7。假设 A 是 layout，N 是正整数。如果 $A\perp B$，并且

$$
\operatorname{size} (A) \cdot \operatorname{size} (B) = N.
$$

就称 layout B 是 A 的 N-complement。定义 2.3.6.8。假设 A 是 layout，N 是正整数。如果 flat layout $A^\flat$ 在定义 2.1.6.24 的意义下是 N-complementable 的，就称 A 是 N-complementable 的。

命题 2.3.6.9。假设 A 是 layout。存在 A 的 N-complement，当且仅当 A 是 N-complementable 的。

证明。如果 B 是 A 的 N-complement，则 $B^\flat$ 是 $A^\flat$ 的 N-complement；根据命题 2.1.6.32，$A^\flat$ 是 N-complementable 的，因此 A 也是。反之，如果 A 是 N-complementable 的，则 $\mathsf{comp}(A,N)$ 是 A 的 N-complement。□

示例 2.3.6.10。如果 `A=((4,2),(2,2)):((3,24),(192,96))` 且 `N=768`，则

$$
\operatorname{comp} (A, N) = (3, 2, 2, 2): (1, 1 2, 4 8, 3 8 4).
$$

示例 2.3.6.11。如果 $A=((16,4),64):((1,16),64)$ 且 `N=4096`，则

$$
\begin{array}{c} \mathsf {c o m p} (A, N) = \mathsf {c o a l} (\left(\left(\right): \left(\right)\right) \\ = 1: 0. \end{array}
$$

示例 2.3.6.12。如果 $A=((16,4),64):((1,16),64)$ 且 `N=8192`，则

$$
\begin{array}{c} \mathsf {c o m p} (A, N) = \mathsf {c o a l} ((2): (4 0 9 6)) \\ = 2: 4 0 9 6. \end{array}
$$

示例 2.3.6.13。如果 $A=((16,4),64):((8,1),128)$ 且 `N=16384`，则

$$
\begin{array}{c} \mathsf {c o m p} (A, N) = \mathsf {c o a l} ((2, 2): (4, 8 1 9 2)) \\ = (2, 2): (4, 8 1 9 2). \end{array}
$$

## 2.3.7 Composition

本节讨论 layout 上最重要的操作，即 composition。如果 A 和 B 是 layout，则二者的 composition 是 layout $B\circ A$，其 layout function 是 A 与 B 的 layout function 的 composite。更精确地，有以下定义。

定义 2.3.7.1（Layout 的 composition）。假设 A 和 B 是 layout。A 与 B 的 composite 是唯一满足以下性质的 layout $B\circ A$：

1. `shape(B ◦ A)` refine `shape(A)`；

2. $B\circ A$ 在 `shape(A)` 上 coalesced；

3. $\Phi _ { B \circ A } = \Phi _ { B } \circ \Phi _ { A } ^ { \mathsf { s i z e } ( B ) }$ 

注记 2.3.7.2。为了使 $B\circ A$ 存在，必须有

$$
\operatorname{Image} \left(\Phi_ {A}\right) \subseteq [ 0, \text { size } (B)).
$$

注记 2.3.7.3。layout composition 的定义中隐含一项断言：最多只有一个 layout 满足这三个条件。命题 2.3.4.10 证明了这一点。可以把 A 与 B 的 weak composite 定义为满足条件 1 和 3、但不一定满足条件 2 的 layout C；此时

$$
B \circ A = \operatorname{coal} (C, \text { shape } (A))
$$

后文将看到，尝试计算 layout composition 时，可以先计算 A 与 B 的任意 weak composite C，再在 `shape(A)` 上执行 coalesce，形成实际 composite $B\circ A$。注记 2.3.7.4。根据观察 2.3.4.5，composition 定义中的条件 2 蕴含 $B\circ A$ 是 non-degenerate。

示例 2.3.7.5。如果 $A=(3,5):(10,2)$，$B=(100):(7)$，则

$$
B \circ A = (3, 5): (7 0, 1 4).
$$

示例 2.3.7.6。如果 $A=(4):(2)$，$B=(2,2,6):(12,6,1)$，则 A 与 B 的 composition 为

$$
B \circ A = ((2, 2)): ((6, 1)).
$$

注记 2.3.7.7。示例 2.3.7.6 说明，flat layout A 与 B 的 composition 不一定是 flat 的。

示例 2.3.7.8。如果 $A=((2,4),8):((4,8),8)$，$B=(4,4,4,4):(2,4,8,16)$，则

$$
B \circ A = ((2, (2, 2)), (2, 4)): ((4, (8, 8)), (8, 8)).
$$

<table><tr><td>0</td><td>4</td><td>8</td><td>12</td><td>16</td><td>20</td><td>24</td><td>28</td></tr><tr><td>1</td><td>5</td><td>9</td><td>13</td><td>17</td><td>21</td><td>25</td><td>29</td></tr><tr><td>2</td><td>6</td><td>10</td><td>14</td><td>18</td><td>22</td><td>26</td><td>30</td></tr><tr><td>3</td><td>7</td><td>11</td><td>15</td><td>19</td><td>23</td><td>27</td><td>31</td></tr></table>

示例 2.3.7.9。如果 `A=((3,(2,2)),24):((3,(9,18)),72)`，`B=(9,8,3,8):(24,3,1,384)`，则

$$
B \circ A = ((3, (2, 2)), (3, 8)): ((7 2, (3, 6)), (1, 3 8 4))
$$

下面建立一些用于计算 layout composition 的有用性质。

命题 2.3.7.10。假设 A 是 layout，B 和 $\tilde B$ 是满足以下条件的 layout：

$\mathsf { s i z e } ( B ) \leq \mathsf { s i z e } ( \tilde { B } )$ , and 

$\Phi _ { \tilde { B } } \ | _ { \mathsf { s i z e } ( B ) } = \Phi _ { B }$ 

如果 A 与 B 可复合，则

$$
B \circ A = \tilde {B} \circ A.
$$

证明。假设 A 与 B 可复合。则 `cosize(A)≤size(B)`，由以下等式可知，B ◦ A 也是 A 与 $\tilde B$ 的 composite：

$$
\begin{array}{r l} & {\Phi_ {\tilde {B}} \circ \Phi_ {A} ^ {\mathrm{size} (\tilde {B})} = (\Phi_ {\tilde {B}}) | _ {\mathrm{size} (B)} \circ \Phi_ {A} ^ {\mathrm{size} (B)}} \\ & {\qquad = \Phi_ {B} \circ \Phi_ {A} ^ {\mathrm{size} (B)}.} \end{array}
$$

推论 2.3.7.11。如果 A 和 B 是 layout，则 A 与 B 可复合，当且仅当 A 与 `coal(B)` 可复合，并且

$$
B \circ A = \operatorname{coal} (B) \circ A.
$$

建立 layout composition 的基本性质后，下面转向 composition 最重要的两个实例：logical division 与 logical product。

## 2.3.8 Logical division

本节定义 layout 的 logical division。作为启发性示例，考虑 layout

出于各种用途，可能需要对 layout A 分块。例如，下面是使用不同 layout B 对 A 进行的分块。

<table><tr><td>0</td><td>4</td><td>8</td><td>12</td><td>16</td><td>20</td><td>24</td><td>28</td></tr><tr><td>1</td><td>5</td><td>9</td><td>13</td><td>17</td><td>21</td><td>25</td><td>29</td></tr><tr><td>2</td><td>6</td><td>10</td><td>14</td><td>18</td><td>22</td><td>26</td><td>30</td></tr><tr><td>3</td><td>7</td><td>11</td><td>15</td><td>19</td><td>23</td><td>27</td><td>31</td></tr></table>

$$
B = \begin{array}{c c} \hline 0 & 4 \\ \hline 1 & 5 \\ \hline \end{array}
$$

<table><tr><td>0</td><td>4</td><td>8</td><td>12</td><td>16</td><td>20</td><td>24</td><td>28</td></tr><tr><td>1</td><td>5</td><td>9</td><td>13</td><td>17</td><td>21</td><td>25</td><td>29</td></tr><tr><td>2</td><td>6</td><td>10</td><td>14</td><td>18</td><td>22</td><td>26</td><td>30</td></tr><tr><td>3</td><td>7</td><td>11</td><td>15</td><td>19</td><td>23</td><td>27</td><td>31</td></tr></table>

<table><tr><td>0</td><td>4</td><td>8</td><td>12</td></tr><tr><td>2</td><td>6</td><td>10</td><td>14</td></tr></table>

<table><tr><td>0</td><td>4</td><td>8</td><td>12</td><td>16</td><td>20</td><td>24</td><td>28</td></tr><tr><td>1</td><td>5</td><td>9</td><td>13</td><td>17</td><td>21</td><td>25</td><td>29</td></tr><tr><td>2</td><td>6</td><td>10</td><td>14</td><td>18</td><td>22</td><td>26</td><td>30</td></tr><tr><td>3</td><td>7</td><td>11</td><td>15</td><td>19</td><td>23</td><td>27</td><td>31</td></tr></table>

<table><tr><td>0</td><td>4</td><td>16</td><td>20</td></tr><tr><td>2</td><td>6</td><td>18</td><td>22</td></tr></table>

处理这类 tiled layout 时，希望使用形如“（矩阵块内坐标，矩阵块）”的坐标索引 layout：矩阵块指定当前处理哪个 tile，矩阵块内坐标指定该 tile 内的坐标。例如，如果 A 和 B 的 rank 都为 2，希望把 `((i,j),(k,ℓ))` 写成 A 的第 `(k,ℓ)` 个矩阵块中第 `(i,j)` 个条目的索引。logical division $A\oslash B$ 正是提供这种能力的 layout。

定义 2.3.8.1。假设 A 和 B 是 layout，并假设

$$
B ^ {c} = \operatorname{comp} (B, \text { size } (A))
$$

是 B 相对于 `size(A)` 的 complement。则 A 除以 B 的 logical division 是 layout

$$
\begin{array}{c} A \oslash B = A \circ (B, B ^ {c}) \\ = (A \circ B, A \circ B ^ {c}). \end{array}
$$

示例 2.3.8.2。如果 $A=(4,8):(1,4)$，$B=(2,2):(1,4)$，则

$$
A \oslash B = ((2, 2), (2, 4)): ((1, 4), (2, 8)),
$$

如下图所示。

<table><tr><td>0</td><td>4</td><td>8</td><td>12</td><td>16</td><td>20</td><td>24</td><td>28</td></tr><tr><td>1</td><td>5</td><td>9</td><td>13</td><td>17</td><td>21</td><td>25</td><td>29</td></tr><tr><td>2</td><td>6</td><td>10</td><td>14</td><td>18</td><td>22</td><td>26</td><td>30</td></tr><tr><td>3</td><td>7</td><td>11</td><td>15</td><td>19</td><td>23</td><td>27</td><td>31</td></tr></table>

$$
B = \begin{array}{c c} \hline 0 & 4 \\ \hline 1 & 5 \\ \hline \end{array}
$$

<table><tr><td>0</td><td>2</td><td>8</td><td>10</td><td>16</td><td>18</td><td>24</td><td>26</td></tr><tr><td>1</td><td>3</td><td>9</td><td>11</td><td>17</td><td>19</td><td>25</td><td>27</td></tr><tr><td>4</td><td>6</td><td>12</td><td>14</td><td>20</td><td>22</td><td>28</td><td>30</td></tr><tr><td>5</td><td>7</td><td>13</td><td>15</td><td>21</td><td>23</td><td>29</td><td>31</td></tr></table>

注记 2.3.8.3。$A\oslash B$ 中每个条目的颜色表示它所属的矩阵块，透明度表示它代表该矩阵块中的哪个条目。因此，$A\oslash B$ 的每列颜色相同，每行透明度相同。

示例 2.3.8.4。如果 $A=(4,8):(1,4)$，$B=(2,2):(4,1)$，则

$$
A \oslash B = ((2, 2), (2, 4)): ((4, 1), (2, 8)),
$$

如下图所示。

<table><tr><td>0</td><td>4</td><td>8</td><td>12</td><td>16</td><td>20</td><td>24</td><td>28</td></tr><tr><td>1</td><td>5</td><td>9</td><td>13</td><td>17</td><td>21</td><td>25</td><td>29</td></tr><tr><td>2</td><td>6</td><td>10</td><td>14</td><td>18</td><td>22</td><td>26</td><td>30</td></tr><tr><td>3</td><td>7</td><td>11</td><td>15</td><td>19</td><td>23</td><td>27</td><td>31</td></tr></table>

$$
B = \begin{array}{c c} \hline 0 & 1 \\ \hline 4 & 5 \\ \hline \end{array}
$$

<table><tr><td>0</td><td>2</td><td>8</td><td>10</td><td>16</td><td>18</td><td>24</td><td>26</td></tr><tr><td>4</td><td>6</td><td>12</td><td>14</td><td>20</td><td>22</td><td>28</td><td>30</td></tr><tr><td>1</td><td>3</td><td>9</td><td>11</td><td>17</td><td>19</td><td>25</td><td>27</td></tr><tr><td>5</td><td>7</td><td>13</td><td>15</td><td>21</td><td>23</td><td>29</td><td>31</td></tr></table>

注记 2.3.8.5。注意前两个示例之间的差异。两个示例对 A 的分块完全相同，但每个矩阵块的 layout 不同。第一个示例中的矩阵块采用 column-major layout，第二个示例则采用 row-major layout，因此执行 logical division 时会得到不同 layout。

示例 2.3.8.6。如果 $A=(4,8):(1,4)$，$B=(2,4):(2,4)$，则

$$
A \oslash B = ((2, 4), (2, 2)): ((2, 4), (1, 1 6)).
$$

示例 2.3.8.7。如果 $A=(4,6):(1,40)$，$B=6:4$，则

$$
A \oslash B = (6, 4): (4 0, 1).
$$

示例 2.3.8.8。如果 `A=(4,6,2,4,2,5):(36,1,18,0,0,144)`，$B=(4,10):(1,192)$，则

$$
A \oslash B = (((4, (2, 5)), (6, 2, 4)): ((3 6, (0, 1 4 4)), (1, 1 8, 0))
$$

示例 2.3.8.9。如果 `A=(8,(4,4))`，`B=(2,(8,16))`，则

$$
A \oslash B = ((2, 2), (2, (4, 4))): ((4, 8), (2, (8, 1 6))).
$$

## 2.3.9 Logical product

本节定义 layout 的 logical product。

定义 2.3.9.1。假设 A 和 B 是 layout，并假设

$$
A ^ {c} = \operatorname{comp} (A, \text { size } (A) \cdot \text { cosize } (B))
$$

是 A 相对于 `size(A)·cosize(B)` 的 complement。则 A 与 B 的 logical product 是 layout

$$
A \otimes B = (A, A ^ {c} \circ B).
$$

观察 2.3.9.2。根据命题 2.3.7.10 和命题 2.3.7.11，如果令

$$
\widetilde {A} ^ {c} = \operatorname{comp} (A, N)
$$

其中 N 是任意满足 $N\geq\mathsf{size}(A)\cdot\mathsf{cosize}(B)$ 的有效值，则

$$
A ^ {c} \circ B = \tilde {A} ^ {c} \circ B.
$$

这意味着计算 $A\otimes B$ 时，可以把 $A^c$ 取为 A 的任意足够大的已排序 complement。

示例 2.3.9.3。如果 $A=(2,2):(5,10)$ 和 $B=(3,5):(5,1)$ 是以下 layout：

$$
A = \begin{array}{c c} \hline 0 & 1 0 \\ \hline 5 & 1 5 \\ \hline \end{array}
$$

<table><tr><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td></tr><tr><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td></tr><tr><td>10</td><td>11</td><td>12</td><td>13</td><td>14</td></tr></table>

则 $A\otimes B$ 是 layout

$$
A \otimes B = ((2, 2), (3, 5)): ((5, 1 0), (2 0, 1))
$$

如下图所示。

<table><tr><td>0</td><td>20</td><td>40</td><td>1</td><td>21</td><td>41</td><td>2</td><td>22</td><td>42</td><td>3</td><td>23</td><td>43</td><td>4</td><td>24</td><td>44</td></tr><tr><td>5</td><td>25</td><td>45</td><td>6</td><td>26</td><td>46</td><td>7</td><td>27</td><td>47</td><td>8</td><td>28</td><td>48</td><td>9</td><td>29</td><td>49</td></tr><tr><td>10</td><td>30</td><td>50</td><td>11</td><td>31</td><td>51</td><td>12</td><td>32</td><td>52</td><td>13</td><td>33</td><td>53</td><td>14</td><td>34</td><td>54</td></tr><tr><td>15</td><td>35</td><td>55</td><td>16</td><td>36</td><td>56</td><td>17</td><td>37</td><td>57</td><td>18</td><td>38</td><td>58</td><td>19</td><td>39</td><td>59</td></tr></table>

示例 2.3.9.4。如果 $A=(3,3):(6,1)$，$B=(10,12):(24,2)$，则

$$
A \otimes B = ((3, 3), (1 0, 1 2)): ((6, 1), (2 1 6, 1 8)).
$$

示例 2.3.9.5。如果 `A=(2,10):(1680,4)`，`B=(4,9):(2,56)`，则

$$
A \otimes B = ((2, 1 0), ((2, 2), (3, 3))): ((1 6 8 0, 4), ((2, 4 0), (5 6 0, 3 3 6 0))).
$$

示例 2.3.9.6。如果 `A=(4,(2,2)):(9,(1,3))`，`B=((2,4),8):((1,4),2)`，则

$$
A \otimes B = ((4, (2, 2)), ((2, 4), 8)): ((9, (1, 3)), ((3 6, 1 4 4), 7 2)).
$$

## 2.3.10 Tractable layout

本节定义一类行为特别良好的 layout，称为 tractable layout。后文将看到，tractable layout 恰好是由某个 category **Nest** 产生的 layout。

定义 2.3.10.1。如果 flat layout $L^\flat$ 在定义 2.1.8.1 的意义下是 tractable 的，就称 layout L 是 tractable 的。显式地，如果 flat layout

$$
\operatorname{sort} \left(L ^ {\flat}\right) = \left(s _ {1}, \dots , s _ {m}\right): \left(d _ {1}, \dots , d _ {m}\right)
$$

对每个 $1\leq i<m$ 都满足以下条件之一，则 L 是 tractable 的：

1. $d_i=0$；或

2. $s_id_i$ 整除 $d_{i+1}$。

示例 2.3.10.2。layout

$$
L = (((1 2))): ((1 7))
$$

是 tractable 的。更一般地，任意 length 1 layout L 都是 tractable 的。

示例 2.3.10.3。layout

$$
L = ((2, 4), 3 2): ((1, 2), 8)
$$

是 tractable 的。更一般地，任意 column-major layout 都是 tractable 的。

示例 2.3.10.4。layout

$$
L = (2, (4, 3 2)): (1 2 8, (3 2, 1))
$$

是 tractable 的。更一般地，任意 row-major layout L 都是 tractable 的。

示例 2.3.10.5。layout

$$
L = ((3, 3), (1, 3), (3, 1, 3)): ((8 1, 1), (0, 8), (3, 0, 2 7))
$$

是 tractable 的。更一般地，任意 compact layout 都是 tractable 的。

示例 2.3.10.6。layout

$$
L = ((3, 7, 7)): ((0, 1 5, 0))
$$

是 tractable 的。更一般地，恰好有一个非零 stride entry 的任意 layout 都是 tractable 的。

示例 2.3.10.7。layout

$$
L = (2, (2, (2, 2))): (1, (2 0 4 8, (1 6, 6 4)))
$$

是 tractable 的。更一般地，任意 complementable layout 都是 tractable 的。

示例 2.3.10.8。layout

$$
L = ((8, 8), (5, 5)): ((8, 1), (1 0, 2))
$$

不是 tractable 的。特别地，这说明 tractable layout $L_1$ 与 $L_2$ 的 concatenation `(L1,L2)` 不一定 tractable。

## 第 3 章

# Layout 的 category

全面探索 layout algebra 后，现在转向本文的数学核心：把 layout realization 为适当定义的 category 中的 morphism。在此过程中，我们建立一套 layout 图的图形演算，使 layout 操作的计算更加直接。

## 3.1 Category Tuple

本节定义 category **Tuple**，其 object 是正整数 tuple，其 morphism 称为 tuple morphism。每个 tuple morphism $f:S\to T$ 编码一个 flat layout $L_f$。tuple morphism 的 composition 与 layout composition 相容：如果 f 和 g 是可复合的 tuple morphism，则

$$
L _ {g \circ f} = L _ {g} \circ L _ {f}.
$$

我们定义 realization functor（定理 3.1.4.4）

$$
| \cdot |: \text { Tuple } \to \text { FinSet }
$$

它通过公式

$$
| f | = \Phi_ {L _ {f}} ^ {\text { size } (T)}.
$$

恢复 $L_f$ 的 layout function。我们建立一套“tuple morphism algebra”，其中包括 sort（第 3.1.5.3 节）、coalesce（第 3.1.5.4 节）、complement（第 3.1.5.6 节）、concatenate（第 3.1.5.5 节）、flat division（第 3.1.5.7 节）和 flat product（第 3.1.5.8 节）等操作；这些操作与 flat layout 上的对应操作相容。

## 3.1.1 基本定义

定义 3.1.1.1。用 Fin<sub>∗</sub> 表示以下 category，其 object 是带基点有限集合

$$
\langle m \rangle_ {*} = \{*, 1, 2, \ldots , m \}
$$

其中 $m\geq0$；其 morphism $\alpha:\langle m\rangle_*\to\langle n\rangle_*$ 是满足 $\alpha(*)=*$ 的函数。称这些 morphism 为 pointed map，或简称 map。

旁注 3.1.1.2。Fin<sub>∗</sub> 是带基点有限集合 category FinSet<sub>∗</sub> 的 skeleton。

记号 3.1.1.3。如果 pointed map $\alpha:\langle m\rangle_*\to\langle n\rangle_*$ 的 codomain 已明确，有时写成

$$
\alpha = (\alpha (1), \dots , \alpha (m))
$$

即一个 length 为 m、条目位于 $\langle n\rangle_*$ 中的 tuple。

示例 3.1.1.4。Fin<sub>∗</sub> 中存在 morphism $\alpha:\langle4\rangle_*\to\langle6\rangle_*$，定义为

$$
\alpha = (2, 1, *, 6),
$$

可用下图可视化。

![image](Imgaes/categorical-foundations-cute-layouts-paper/1d932e5197de984f50f44364a441c51e8c01176ee6a20812587959cf1a948276.jpg)


注意，对应条目 3 的圆点没有箭头，反映它被映射到 ∗。

示例 3.1.1.5。Fin<sub>∗</sub> 中存在 morphism $\beta:\langle5\rangle_*\to\langle3\rangle_*$，定义为

$$
\beta = (*, 1, 2, 3, *) ,
$$

可用下图可视化。

![image](Imgaes/categorical-foundations-cute-layouts-paper/3d8222e56590b7b24aedccd4c39e86cbdf4bfaac12964a9f9932f0bb2c12467e.jpg)


示例 3.1.1.6。对任意 $m\geq0$，Fin<sub>∗</sub> 中存在唯一形如 $\pi:\langle m\rangle_*\to\langle0\rangle_*$ 的 morphism，即

$$
\pi = (*, \dots , *).
$$

示例 3.1.1.7。对任意 $n\geq0$，Fin<sub>∗</sub> 中存在唯一形如 $\delta:\langle0\rangle_*\to\langle m\rangle_*$ 的 morphism，即

$$
\delta = ().
$$

旁注 3.1.1.8。category Fin<sub>∗</sub> 是交换 operad 的 operator category，因此有时写成

$$
\mathbf {F i n} _ {*} = \mathbf {C o m m} ^ {\otimes}.
$$

我们尤其关注 $\mathsf{Fin}_*$ 中的 tractable morphism，定义如下。

定义 3.1.1.9。如果对任意 $j\in\langle n\rangle\subset\langle n\rangle_*$，原像 $\alpha^{-1}(j)$ 为空或仅含一个元素，就称 pointed map $\alpha:\langle m\rangle_*\to\langle n\rangle_*$ 是 tractable 的。

示例 3.1.1.10。以下 map

![image](Imgaes/categorical-foundations-cute-layouts-paper/9398fc32796fb475da78edf997e9c71a347748e7b38219e6a1b23134782a8f27.jpg)


是 tractable 的，而以下 map

![image](Imgaes/categorical-foundations-cute-layouts-paper/aced812813ea3f1c753a16585b6b4601e2167dc9cbf0dcade1b6fd4688edc022.jpg)


不是 tractable 的。

注记 3.1.1.11。如果把 Fin<sub>∗</sub> 中的 morphism $\alpha:\langle m\rangle_*\to\langle n\rangle_*$ 表示为 tuple，即

$$
\alpha = (\alpha (1), \dots , \alpha (m))
$$

则 α 是 tractable 的，当且仅当 α 中没有任何正整数出现超过一次。旁注 3.1.1.12。由 tractable pointed map 构成的 wide subcategory

$$
\mathbf {E} _ {0} ^ {\otimes} \subset \mathbf {C o m m} ^ {\otimes} = \mathbf {F i n} _ {*}
$$

是 $\mathsf{E}_0$ operad 的 operator category。

定义 3.1.1.13。用 **Tuple** 表示以下 category，其 object 是正整数 tuple

$$
S = (s _ {1}, \ldots , s _ {m})
$$

，其中 morphism

$$
f: (s _ {1}, \dots , s _ {m}) \to (t _ {1}, \dots , t _ {n})
$$

由 tractable pointed map $\alpha:\langle m\rangle_*\to\langle n\rangle_*$ 指定，并满足性质

如果 $1\leq i\leq m$ 且 $\alpha(i)\neq*$，则 $s_i=t_{\alpha(i)}$。

称这种 morphism f 位于 α 之上，并称 f 为 tuple morphism。

记号 3.1.1.14。如果 $f:(s_1,\ldots,s_m)\to(t_1,\ldots,t_n)$ 是位于 α 之上的 tuple morphism，有时把 f 描绘为

$$
(s _ {1}, \ldots , s _ {m}) \xrightarrow [ \alpha ]{f} (t _ {1}, \ldots , t _ {n}).
$$

本文建立的 layout 图形演算，以 **Tuple** 中 morphism 的自然可视化为基础，如下例所示。

示例 3.1.1.15。tuple morphism

$$
(3, 1 2 8, 1 2 8) \xrightarrow [ (1 , 3 , 5) ]{f} (3, 2, 1 2 8, 2, 1 2 8)
$$

可用下图可视化。

![image](Imgaes/categorical-foundations-cute-layouts-paper/88914cb5525127454469911d9442cff03c6cb2459b991fb5f4596d5f5be2c551.jpg)


示例 3.1.1.16。tuple morphism

$$
(3, 1 2 8, 1 2 8) \xrightarrow [ (* , 2 , 1) ]{g} (1 2 8, 1 2 8)
$$

可用下图可视化。

$$
\begin{array}{c} 1 2 8 \\ 1 2 8 \\ 3 \end{array} \xrightarrow {} \begin{array}{c} 1 2 8 \\ 1 2 8 \\ 1 2 8 \end{array}
$$

示例 3.1.1.17。tuple morphism

$$
(1 6, 1 6, 1 6, 1, 3 2) \xrightarrow [ (* , * , 1 , * , 2) ]{h} (1 6, 3 2, 1, 1)
$$

可用下图可视化。

$$
\begin{array}{c} 3 2 \\ 1 \\ 1 6 \\ 1 6 \\ 1 6 \end{array} \begin{array}{c} 1 \\ 1 \\ 3 2 \\ 1 6 \end{array} h
$$

观察 3.1.1.18。可以把 category **Tuple** 与一些著名 operad 联系起来。用 $\mathbb{Z}_{>0}^{\mathrm{div}}$ 表示正整数在整除关系下构成的偏序集，并把它视为以整数乘法为 product 的对称 monoidal category。用 $(\mathbb{Z}_{>0}^{\mathsf{div}})^\otimes$ 表示 $\mathbb{Z}_{>0}^{\mathrm{div}}$ 的 operator category。则存在显然的 functor

$$
\mathbf {T u p l e} \rightarrow (\mathbb {Z} _ {> 0} ^ {\mathrm{div}}) ^ {\otimes},
$$

and 

$$
\mathbf {T u p l e} \to \mathbf {E} _ {0} ^ {\otimes},
$$

使下图

$$
\begin{array}{c} \text { Tuple } \longrightarrow (\mathbb {Z} _ {> 0} ^ {\mathrm{div}}) ^ {\otimes} \\ \Big \downarrow \qquad \qquad \qquad \qquad \qquad \qquad \Big \downarrow \\ E _ {0} ^ {\otimes} \longrightarrow \text { Comm } ^ {\otimes} \end{array}
$$

交换。这把 **Tuple** 展示为 pullback operad

$$
\mathbf {T u p l e} \subset \mathbf {E} _ {0} ^ {\otimes} \times_ {\mathbf {C o m m} ^ {\otimes}} (\mathbb {Z} _ {> 0} ^ {\text { div }}) ^ {\otimes}
$$

中由以下 morphism 构成的 wide subcategory：

$$
(s _ {1}, \ldots , s _ {m}) \xrightarrow [ \alpha ]{f} (t _ {1}, \ldots , t _ {n})
$$

它们满足

$$
\alpha (i) \neq 1 \quad \Rightarrow \quad s _ {i} = t _ {\alpha (i)}.
$$

## 3.1.2 从 tuple morphism 到 flat layout

使用 category **Tuple** 的动机在于，每个 tuple morphism f 都编码一个 flat layout $L_f$；而每个 tractable layout L 都会产生一个 tuple morphism $f_L$。定理 3.1.2.10 将证明，这两个构造在某种意义下互逆，而且 tractable layout 恰好是由 tuple morphism 编码的 layout。

构造 3.1.2.1。假设

$$
(s _ {1}, \ldots , s _ {m}) \xrightarrow [ \alpha ]{f} (t _ {1}, \ldots , t _ {n})
$$

是 tuple morphism。把 $L_f$ 定义为以下 flat layout：其 shape

$$
\operatorname{shape} \left(L _ {f}\right) = \left(s _ {1}, \dots , s _ {m}\right)
$$

是 f 的 domain，其 stride

$$
\operatorname{stride} \left(L _ {f}\right) = \left(d _ {1}, \dots , d _ {m}\right)
$$

由公式定义

$$
d _ {i} = \left\{ \begin{array}{l l} 0 & \alpha (i) = * \\ \prod_ {j <   \alpha (i)} t _ {j} & \alpha (i) \neq *. \end{array} \right.
$$

称 $L_f$ 为 f 编码的 layout，或与 f 关联的 layout。

示例 3.1.2.2。示例 3.1.1.15 的 tuple morphism

![image](Imgaes/categorical-foundations-cute-layouts-paper/edd5f63b063a9d495f4bec1bf4bd779a308d682eca01695c6559e4cbab6f0e20.jpg)


编码 layout

$$
L _ {f} = (3, 1 2 8, 1 2 8): (1, 6, 1 5 3 6).
$$

注意，通过定理 3.1.2.1 的公式计算 stride，相当于沿某个特定 shape 条目的箭头到达其目标条目，再把目标条目下方的所有条目相乘；空乘积取 1。

示例 3.1.2.3。示例 3.1.1.16 的 tuple morphism

$$
(3, 1 2 8, 1 2 8) \xrightarrow [ (* , 2 , 1) ]{g} (1 2 8, 1 2 8)
$$

编码 layout

$$
L _ {g} = (3, 1 2 8, 1 2 8): (0, 1 2 8, 1).
$$

示例 3.1.2.4。示例 3.1.1.17 的 tuple morphism

$$
(1 6, 1 6, 1 6, 1, 3 2) \xrightarrow [ (* , * , 1 , * , 2) ]{h} (1 6, 3 2, 1, 1)
$$

编码 layout

$$
L _ {h} = (1 6, 1 6, 1 6, 1, 3 2): (0, 0, 1, 0, 1 6).
$$

前文说明了如何计算 tuple morphism f 编码的 flat layout $L_f$。另一方面，如果 L 是 tractable 的，也可以沿反方向构造编码 L 的 tuple morphism f。回忆定义 2.1.8.1：如果

$$
\operatorname{sort} (L) = \left(s _ {1}, \dots , s _ {m}\right): \left(d _ {1}, \dots , d _ {m}\right)
$$

满足以下性质，则 flat layout L 是 tractable 的：

$$
\text {   If   } 1 \leq i <   m, \text {   then   } d _ {i} = 0, \text {   or   } s _ {i} d _ {i} \text {   divides   } d _ {i + 1}.
$$

构造 3.1.2.5。假设 $L=(s_1,\ldots,s_m):(d_1,\ldots,d_m)$ 是 tractable 的，并令

$$
\operatorname{sort} (L) = \left(s _ {1} ^ {\prime}, \dots , s _ {m} ^ {\prime}\right): \left(d _ {1} ^ {\prime}, \dots , d _ {m} ^ {\prime}\right),
$$

于是存在某个置换 $\sigma\in\Sigma_m$，使 `sort(L)=L^σ`。换言之，对每个 $1\leq i\leq m$，有 $s_i'=s_{\sigma(i)}$、$d_i'=d_{\sigma(i)}$。如果每个 $d_i'$ 都非零，令 $k=0$；否则令 k 为满足 $d_k'=0$ 的最大整数。令 $\ell=2(m-k)$，并令

$$
(t _ {1} ^ {\prime}, \ldots , t _ {\ell} ^ {\prime}) = \left(d _ {k + 1} ^ {\prime}, s _ {k + 1} ^ {\prime}, \frac {d _ {k + 2} ^ {\prime}}{s _ {k + 1} ^ {\prime} d _ {k + 1} ^ {\prime}}, s _ {k + 2} ^ {\prime}, \frac {d _ {k + 3} ^ {\prime}}{s _ {k + 2} ^ {\prime} d _ {k + 2} ^ {\prime}}, \ldots , \frac {d _ {m} ^ {\prime}}{s _ {m - 1} ^ {\prime} d _ {m - 1} ^ {\prime}}, s _ {m} ^ {\prime}\right).
$$

定义

$$
f _ {L} ^ {\prime}: (s _ {1}, \ldots , s _ {m}) \to (t _ {1} ^ {\prime}, \ldots , t _ {\ell} ^ {\prime})
$$

为位于 map $\alpha:\langle m\rangle_*\to\langle\ell\rangle_*$ 之上的 tuple morphism，其中

$$
\alpha^ {\prime} (i) = \left\{ \begin{array}{l l} * & \sigma^ {- 1} (i) \leq k \\ 2 (\sigma^ {- 1} (i) - k) & k + 1 \leq \sigma^ {- 1} (i) \leq m. \end{array} \right.
$$

令 $J=\{j_1<\cdots<j_n\}\subset\langle\ell\rangle$ 表示满足“$j_i$ 为偶数或 $t_{j_i}\neq1$”的索引集合。令

$$
(t _ {1}, \dots , t _ {n}) = (t _ {j _ {1}} ^ {\prime}, \dots , t _ {j _ {n}} ^ {\prime}),
$$

并令 $\iota:\langle n\rangle_*\to\langle\ell\rangle_*$ 为 inclusion map $i\mapsto j_i$。根据构造，map $\alpha'$ 分解为 $\alpha'=\iota\circ\alpha$；把 L 的 standard representation 定义为 tuple morphism

$$
(s _ {1}, \ldots , s _ {m}) \xrightarrow [ \alpha ]{f _ {L}} (t _ {1}, \ldots , t _ {n}).
$$

$$
L = (2, 2): (3, 3 0),
$$

则 L 是 tractable 的，其 standard representation 是 tuple morphism

![image](Imgaes/categorical-foundations-cute-layouts-paper/eeb21ca167f0010e78bd086fdb3527cf25c9f4ead3d053ddd8ec4f1ab2ffbde6.jpg)


非正式地说，通过定理 3.1.2.5 计算 $f_L$ 相当于：

• 把 codomain 初始化为 `()`；

• 按递增顺序遍历 L 的非零 stride；

• 如果 $d_j$ 是当前 stride，$d_i$ 是上一个访问的 stride，则追加

– 当 s<sub>i</sub>d<sub>i</sub> = d<sub>j</sub> 时追加 `(s_j)`；或

$$
\left(\frac {d _ {j}}{s _ {i} d _ {i}}, s _ {j}\right) \text {   if   } s _ {i} d _ {i} <   d _ {j},
$$

以及

• 映射 $s_j\mapsto s_j$。

示例 3.1.2.7。如果

$$
L = (1 2 8, 1 2 8): (1 2 8, 1),
$$

则 L 是 tractable 的，其 standard representation 是 tuple morphism

$$
\begin{array}{c} 1 2 8 \\ 1 2 8 \end{array} \xrightarrow {} \begin{array}{c} 1 2 8 \\ 1 2 8 \end{array} f _ {L}
$$

示例 3.1.2.8。如果

$$
L = (2, 2, 2, 2): (2 4, 0, 3, 4 8 0),
$$

则 L 是 tractable 的，其 standard representation 是 tuple morphism

![image](Imgaes/categorical-foundations-cute-layouts-paper/314302fe5e5cc602aec203f347381b3cb9c1947434ce71e1695cd4190814a4bb.jpg)


下面证明定理 3.1.2.5 的 tuple morphism $f_L$ 确实编码 layout L。

引理 3.1.2.9。假设 L 是 tractable flat layout，$f=f_L$ 是 L 的 standard representation。则 f 编码的 layout 为

$$
L _ {f} = L.
$$

证明。假设 $L=(s_1,\ldots,s_m):(d_1,\ldots,d_m)$ 是 tractable 的，并令

$$
(s _ {1}, \ldots , s _ {m}) \xrightarrow [ \alpha ]{f} (t _ {1}, \ldots , t _ {n})
$$

为 L 的 standard representation。显然

$$
\operatorname{shape} \left(L _ {f}\right) = \left(s _ {1}, \dots , s _ {m}\right) = \operatorname{shape} (L).
$$

需要检查 `stride(L_f)=stride(L)`。换言之，需要检查对任意 $1\leq i\leq m$，都有

$$
d _ {i} = \left\{ \begin{array}{l l} 0 & \alpha (i) = * \\ \prod_ {j <   \alpha (i)} t _ {j} & \alpha (i) \neq *. \end{array} \right.
$$

沿用定理 3.1.2.5 的记号。如果 $\alpha(i)=*$，则 $\alpha'(i)=*$，因此 $\sigma^{-1}(i)\leq k$。这意味着

$$
d _ {i} = d _ {\sigma^ {- 1} (i)} ^ {\prime} = 0.
$$

否则假设 $\alpha(i)\neq*$。则 $\alpha'(i)\neq*$，因此 $k+1\leq\sigma^{-1}(i)\leq m$。计算得

$$
\begin{array}{c}\prod_{j <   \alpha (i)}t_{j} = \prod_{\substack{j^{\prime} <   \alpha^{\prime}(i)\\ t_{j^{\prime}}^{\prime}\neq 1}}t_{j^{\prime}}^{\prime} = \prod_{j^{\prime} <   \alpha^{\prime}(i)}t_{j^{\prime}}^{\prime} = \prod_{j^{\prime} <   2(\sigma^{-1}(i) - k)}t_{j^{\prime}}^{\prime}\\ = d_{\sigma^{-1}(i)}^{\prime}\\ = d_{i}. \end{array}
$$

我们已经证明，如果 L 是 tractable flat layout，则存在编码 L 的 tuple morphism f。下面证明逆命题，由此得出 tractable flat layout 恰好是由 tuple morphism 编码的 layout。

命题 3.1.2.10。假设 L 是 flat layout。存在编码 L 的 tuple morphism f，当且仅当 L 是 tractable 的。

证明。首先，假设 L 是 flat layout，$f:(s_1,\ldots,s_m)\to(t_1,\ldots,t_n)$ 是满足 $L_f=L$ 的 tuple morphism。要证明 $L_f$ 是 tractable 的。令

$$
\operatorname{sort} (L) = \left(s _ {1} ^ {\prime}, \dots , s _ {m} ^ {\prime}\right): \left(d _ {1}, \dots , d _ {m}\right)
$$

为 L 的排序，并假设 $1\leq i<m$。下面证明 $d_i=0$，或 $s_i'd_i$ 整除 $d_{i+1}$。如果 $d_i=0$，结论已得。否则假设 $d_i\neq0$，则

$$
d _ {i} = \prod_ {j <   k} t _ {j}
$$

其中某个 $1\leq k\leq n$ 满足 $s_i'=t_k$。由于 $d_{i+1}\geq d_i$，知道 $d_{i+1}\neq0$，因此 $d_{i+1}$ 具有形式

$$
d _ {i + 1} = \prod_ {j <   \ell} t _ {j}
$$

其中某个 $1\leq\ell\leq n$。需要考虑两种情况：

• 情况 1：如果 $\ell>k$，则

$$
d _ {i + 1} = \prod_ {j <   \ell} t _ {j} = \left(\prod_ {j \leq k} t _ {j}\right) \left(\prod_ {k <   j <   \ell} t _ {j}\right) = s _ {i} ^ {\prime} d _ {i} \left(\prod_ {k <   j <   \ell} t _ {j}\right),
$$

所以 $s_i'd_i$ 整除 $d_{i+1}$。

• 情况 2：如果 $\ell\leq k$，由于

$$
\prod_ {j <   \ell} t _ {j} = d _ {i + 1} \geq d _ {i} = \prod_ {j <   k} t _ {j},
$$

必须有

$$
t _ {\ell} = \dots = t _ {k - 1} = 1,
$$

and 

$$
d _ {i + 1} = d _ {i}.
$$

特别地，有 $s_{i+1}'=t_\ell=1$。但由于 $\mathsf{sort}(L_f)$ 已排序且 $d_{i+1}=d_i$，有 $s_i'\leq s_{i+1}'=1$，所以 $s_i'=1$。因此

$$
s _ {i} ^ {\prime} d _ {i} = d _ {i + 1},
$$

特别地，$s_i'd_i$ 整除 $d_{i+1}$。

因此 L 是 tractable 的。

接下来假设 L 是 tractable 的。可以取 $f=f_L$ 为 L 的 standard representation，参见构造 3.1.2.5；此时根据引理 3.1.2.9，有 $L=L_f$。□

注记 3.1.2.11。需要注意，许多不同的 tuple morphism 会产生同一个 layout。例如，下图中的每个 tuple morphism

![image](Imgaes/categorical-foundations-cute-layouts-paper/a92bb77975979d7562e9ea3f6d5357c37198cf733f0c8ca2ca97437f19394b94.jpg)


都编码 layout

$$
L _ {f} = L _ {g} = L _ {h} = (4, 4, 4): (1 4, 5 6, 5 6 0 0).
$$

其中 f 最简单：f 的像上方没有多余条目，不像 g；未被 f 命中的条目已经合并，不像 h。为了精确描述 f 在这些 morphism 中的简洁性，引入 standard form 概念。

定义 3.1.2.12。假设

$$
(s _ {1}, \ldots , s _ {m}) \xrightarrow [ \alpha ]{f} (t _ {1}, \ldots , t _ {n})
$$

是 tuple morphism。如果以下条件成立，就称 f 具有 standard form：

1. 如果 $n>1$，则 $n\in\mathsf{Image}(\alpha)$；

2. 如果 $1\leq j<n$，则

$$
j \notin \operatorname{Image} (\alpha) \quad \Rightarrow \quad \begin{array}{c} t _ {j} \neq 1, \text {   and   } \\ j + 1 \in \operatorname{Image} (\alpha) \end{array}
$$

示例 3.1.2.13。注记 3.1.2.11 中的 tuple morphism f 具有 standard form，而 g 和 h 没有。

示例 3.1.2.14。以下 tuple morphism

![image](Imgaes/categorical-foundations-cute-layouts-paper/b2f29e6a84fc1aff2c1e4b314ec81e745dafe56b153845100c0af8bdccedce4f.jpg)


具有 standard form，而以下 tuple morphism

![image](Imgaes/categorical-foundations-cute-layouts-paper/26f697f8ef845f294e88f8d0c7256cac88290eeedadfc5a2f8686ba22151e045.jpg)


没有。

示例 3.1.2.15。如果 L 是 tractable layout，则根据构造，L 的 standard representation $f_L$ 具有 standard form。

如果把范围限制在具有 standard form 的 tuple morphism 上，则它们与 tractable layout 几乎一一对应。不过，需要排除一种有问题的情况，如下例所示。

示例 3.1.2.16。考虑下图中的 tuple morphism f 和 g。

![image](Imgaes/categorical-foundations-cute-layouts-paper/38bcd5b34ee35a5babce05ae699b69605e029da991131507e2699d00b815af14.jpg)


f 和 g 都具有 standard form，并且

$$
L _ {f} = (8, 1, 1): (1, 8, 8) = L _ {g}.
$$

该示例说明，形如 $s_i=1$ 且 $\alpha(i)\neq*$ 的条目可能导致具有 standard form 的表示 tuple morphism 不唯一。在 layout 一侧，这对应 shape 条目 $s_i=1$ 但 stride $d_i\neq0$。为了排除这类病态示例，引入 non-degeneracy 概念。

定义 3.1.2.17。假设

$$
(s _ {1}, \ldots , s _ {m}) \xrightarrow [ \alpha ]{f} (t _ {1}, \ldots , t _ {n})
$$

是 tuple morphism，并且

$$
L = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m})
$$

是 flat layout。

1. 如果下式成立，就称 f 是 non-degenerate：

$$
s _ {i} = 1 \quad \Rightarrow \quad \alpha (i) = *.
$$

2. 如果

$$
s _ {i} = 1 \quad \Rightarrow \quad d _ {i} = 0.
$$

就称 L 是 non-degenerate。观察 3.1.2.18。如果 f 是 non-degenerate tuple morphism，则 f 编码的 layout $L_f$ 是 non-degenerate。反之，如果 L 是 non-degenerate flat layout，则 L 的 standard representation $f_L$ 是 non-degenerate。

观察 3.1.2.19。把范围限制到 non-degenerate flat layout 不会真正损失一般性。如果 L 是任意 flat layout，则 `filter(L)` 是与 L 具有相同 coordinate function 和 layout function 的 non-degenerate flat layout。

具有 standard form 的 non-degenerate tuple morphism 的基本性质是：它们由所编码的 layout 刻画。精确表述如下。

引理 3.1.2.20。假设 f 和 g 是具有 standard form 的 non-degenerate tuple morphism。如果 $L_f=L_g$，则 $f=g$。

证明。假设

$$
(s _ {1}, \ldots , s _ {m}) \xrightarrow [ \alpha ]{f} (t _ {1}, \ldots , t _ {n})
$$

and 

$$
\left(s _ {1}, \ldots , s _ {m}\right) \xrightarrow [ \beta ]{g} \left(u _ {1}, \ldots , u _ {p}\right)
$$

是具有 standard form 的 non-degenerate tuple morphism，并满足

$$
L _ {f} = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m}) = L _ {g}.
$$

要证明 $f=g$。首先证明 $(t_1,\ldots,t_n)=(u_1,\ldots,u_p)$。令

$$
\begin{array}{l} X = \left\{t _ {1} \dots t _ {j} \mid 1 \leq j \leq n \right\} \\ Y = \left\{u _ {1} \dots u _ {k} \mid 1 \leq k \leq p \right\} \end{array}
$$

分别表示 $(t_1,\ldots,t_n)$ 和 $(u_1,\ldots,u_p)$ 的前缀乘积集合。我们声称 $X=Y$，因为二者都等于

$$
Z = \left\{d _ {i}, s _ {i} d _ {i} \mid 1 \leq i \leq m \text {   and   } d _ {i} \neq 0 \right\}.
$$

下面证明 $X=Z$。假设 $1\leq j\leq n$。如果存在某个 $i\in\langle m\rangle$ 满足 $\alpha(i)=j$，则 $t_1\cdots t_j=s_id_i$。另一方面，如果 j 不在 α 的像中，由于 f 具有 standard form，存在某个 $i\in\langle m\rangle$ 使 $\alpha(i)=j+1$；此时 $t_1\cdots t_j=d_i$。这证明 $X\subseteq Z$。

反之，如果 $1\leq i\leq m$ 且 $d_i\neq0$，则 $d_i=t_1\cdots t_{\alpha(i)-1}$，$s_id_i=t_1\cdots t_{\alpha(i)}$，这证明 $Z\subseteq X$。因此 $X=Z$。相同论证证明 $Y=Z$。

由于 f 和 g 是具有 standard form 的 non-degenerate morphism，知道每个 $t_j$ 和 $u_k$ 都大于 1，因此

$$
\begin{array}{c} t _ {1} <   t _ {1} t _ {2} <   \dots <   t _ {1} \dots t _ {n}, \\ u _ {1} <   u _ {1} u _ {2} <   \dots <   u _ {1} \dots u _ {p}, \end{array}
$$

又因为 $X=Y$，所以 $n=p$，并且对每个 $1\leq j\leq n$，有 $t_1\cdots t_j=u_1\cdots u_j$。因此 $(t_1,\ldots,t_n)=(u_1,\ldots,u_p)$。

接下来需要证明 $\alpha=\beta$。反设存在某个 $i\in\langle m\rangle$ 使 $\alpha(i)\neq\beta(i)$。需要考虑两种情况。

• 如果 $\alpha(i)=*\neq\beta(i)$，则

$$
0 = d _ {i} = t _ {1} \dots t _ {\beta (i) - 1},
$$

矛盾。$\alpha(i)\neq*=\beta(i)$ 的情况类似。

• 如果 $\alpha(i)\neq*\neq\beta(i)$，不失一般性，假设 $\alpha(i)<\beta(i)$；此时

$$
d _ {i} = t _ {1} \dots t _ {\alpha (i) - 1} <   t _ {1} \dots t _ {\beta (i) - 1} = d _ {i},
$$

矛盾。

因此 $\alpha=\beta$，所以 $f=g$。

现在可以证明对应定理，它把具有 standard form 的 non-degenerate tuple morphism 与 non-degenerate tractable flat layout 等同起来。

定理 3.1.2.21。构造 3.1.2.1 和 3.1.2.5 中的映射

![image](Imgaes/categorical-foundations-cute-layouts-paper/cb27ef7256faf20a6511162738e53c87a9371bb307ef00382e58e652d5c915b3.jpg)


![image](Imgaes/categorical-foundations-cute-layouts-paper/8363bd889d8770d8444df2fc42acbf1f5c828f27072b413e1fb882f89e9809eb.jpg)


![image](Imgaes/categorical-foundations-cute-layouts-paper/2d694e90302cff836f2269cc4c344e49b99229762880ce5d6254ca4f694f4452.jpg)


在具有 standard form 的 non-degenerate tuple morphism 与 non-degenerate tractable flat layout 之间建立一一对应关系。

证明。要证明构造 $f\mapsto L_f$ 与 $L\mapsto f_L$ 在限制到上述形式的 tuple morphism 和 layout 后互逆。如果 L 是 non-degenerate tractable flat layout，根据引理 3.1.2.9，有 $L_{f_L}=L$。接下来，假设 f 是具有 standard form 的 non-degenerate tuple morphism，且 $L=L_f$ 是 f 编码的 layout。f 和 $f_{L_f}$ 都是具有 standard form 的 non-degenerate tuple morphism，而且二者编码的 layout 相等，所以根据引理 3.1.2.20，有 $f=f_{L_f}$。□

## 3.1.3 示例

本节介绍 tuple morphism 的一些重要 family，并描述它们产生的 flat layout。

示例 3.1.3.1（Identity morphism）。如果对某个 tuple S 有 $f=\operatorname{id}_S$，就称 tuple morphism f 为 identity morphism。如果 $f=\operatorname{id}_S$ 是 identity morphism，则 $L_f$ 是 shape 为 S 的 column-major layout。例如，下面给出 identity morphism f 及其关联 layout $L_f$：

$$
\begin{array}{c c c}4 \longmapsto 4\\4 \longmapsto 4\\2 \longmapsto 2\\2 \longmapsto 2\\2 \longmapsto 2\\f\end{array}\qquad \rightsquigarrow \qquad L _ {f} = (2, 2, 2, 4, 4): (1, 2, 4, 8, 3 2)
$$

示例 3.1.3.2（Isomorphism）。如果存在 tuple morphism $g:T\to S$，使 $g\circ f=\mathsf{id}_S$ 且 $f\circ g=\mathsf{id}_T$，则 tuple morphism $f:S\to T$ 是 isomorphism。如果 f 是 isomorphism，则其关联 layout $L_f$ 是 compact 的。例如，下面给出 isomorphism f 及其关联 layout $L_f$：

$$
\begin{array}{c c}4&2\\4&4\\2&4\\2&2\\2&2\\f\end{array}\qquad \rightsquigarrow \qquad L _ {f} = (2, 2, 2, 4, 4): (2, 1, 6 4, 4, 1 6)
$$

观察 3.1.3.3。注意，如果 tuple morphism

$$
(s _ {1}, \ldots , s _ {m}) \xrightarrow [ \alpha ]{f} (t _ {1}, \ldots , t _ {n})
$$

是 isomorphism，则 $\alpha:\langle m\rangle_*\to\langle m\rangle_*$ 是双射，因此 $\alpha|_{\langle m\rangle}\in\Sigma_m$ 是置换。反之，如果 $\sigma\in\Sigma_m$ 是置换，$(s_1,\ldots,s_m)$ 是正整数 tuple，则可以构造 isomorphism

$$
\left(s _ {1}, \ldots , s _ {m}\right) \xrightarrow [ \sigma_ {*} ]{f} \left(s _ {\sigma (1)}, \ldots , s _ {\sigma (m)}\right).
$$

因此，domain 为 $(s_1,\ldots,s_m)$ 的 tuple isomorphism f 与 $\Sigma_m$ 中的置换之间存在一一对应关系。

示例 3.1.3.4（Projection）。假设 $S=(s_1,\ldots,s_m)$ 是 shape，并假设

$$
\left\{i _ {1} <   \dots <   i _ {r} \right\} \subset \langle m \rangle
$$

是某个子集。令

$$
\left(s _ {1}, \ldots , s _ {m}\right) \xrightarrow [ \alpha ]{p} \left(s _ {i _ {1}}, \ldots , s _ {i _ {r}}\right)
$$

为位于 map α 之上的 tuple morphism，其中

$$
\alpha (x) = \left\{ \begin{array}{l l} j & x = i _ {j} \\ * & \text { else. } \end{array} \right.
$$

称 p 为 $(s_1,\ldots,s_m)$ 到 $(s_{i_1},\ldots,s_{i_r})$ 的 projection。p 编码的 layout 为

$$
L _ {p} = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m}),
$$

其中

$$
d _ {i} = \left\{ \begin{array}{l l} s _ {i _ {1}} \dots s _ {i _ {j - 1}} & i = i _ {j} \text {   for   some   } 1 \leq j \leq r \\ 0 & \text { otherwise. } \end{array} \right.
$$

例如，下面给出 `(64,64,3,8)` 到 `(64,3)` 的 projection p，以及其关联 layout。

$$
\begin{array}{c c c}8&\\3&\\6 4&\longrightarrow&3\\6 4&\longmapsto&6 4\\&p\end{array}\qquad \rightsquigarrow \qquad L _ {p} = (6 4, 6 4, 3, 8): (1, 0, 6 4, 0)
$$

示例 3.1.3.5（Dilation）。假设 $S=(s_1,\ldots,s_m)$ 是 shape，$c_1,\ldots,c_m$ 是正整数。tuple morphism

$$
\left(s _ {1}, \ldots , s _ {m}\right) \xrightarrow [ (* , 2 , * , 4 , \ldots , * , 2 m) ]{f} \left(c _ {1}, s _ {1}, \ldots , c _ {m}, s _ {m}\right)
$$

称为 $(s_1,\ldots,s_m)$ 按 $(c_1,\ldots,c_m)$ 的 dilation。该 morphism 关联的 layout 为 $L_f=(s_1,\ldots,s_m):(d_1,\ldots,d_m)$，其中

$$
d _ {i} = \prod_ {j <   i} c _ {j} s _ {j}.
$$

例如，下面给出 `(512,512)` 按 `(2,4)` 的 dilation f，以及其关联 layout。

$$
\begin{array}{c c c}&5 1 2\\&\nearrow&4\\5 1 2&5 1 2\\5 1 2&\nearrow&2\\&f\end{array}\qquad \rightsquigarrow \qquad L _ {f} = (5 1 2, 5 1 2): (2, 4 0 9 6)
$$

示例 3.1.3.6（Expansion）。假设 $S=(s_1,\ldots,s_m)$ 是正整数 tuple，并假设 $1\leq m'\leq m$，使 $S'=(s_1,\ldots,s_{m'})$ 整除 S。则 tuple morphism

$$
\left(s _ {1}, \ldots , s _ {m ^ {\prime}}\right) \xrightarrow [ (1 , 2 , \ldots , m ^ {\prime}) ]{e} \left(s _ {1}, \ldots , s _ {m ^ {\prime}}, \ldots , s _ {m}\right)
$$

称为 $S'$ 到 S 的 expansion。e 编码的 layout 是 shape 为 $(s_1,\ldots,s_{m'})$ 的 column-major layout。例如，下面给出 $S'=(4,4)$ 到 $S=(4,4,8,8)$ 的 expansion。

$$
\begin{array}{c c c}8&\\8&\\4 \longmapsto 4&\rightsquigarrow&L _ {e} = (4, 4): (1, 4)\\4 \longmapsto 4&\\e&\end{array}
$$

Expansion 的一项重要性质是：如果 $f:S\to T$ 是任意 tuple morphism，$e:T\to T'$ 是 expansion，则

$$
L _ {e \circ f} = L _ {f}.
$$

换言之，在 f 之后复合 expansion 不会改变 f 编码的 layout。

示例 3.1.3.7（Restriction）。假设

$$
(s _ {1}, \dots , s _ {m}) \xrightarrow [ \alpha ]{f} (t _ {1}, \dots , t _ {n})
$$

是 tuple morphism，并假设

$$
I = \left\{i _ {1} <   \dots <   i _ {r} \right\} \subset \langle m \rangle
$$

是索引子集。则 tuple morphism

$$
(s _ {i _ {1}}, \ldots , s _ {i _ {r}}) \xrightarrow [ \alpha \circ \iota ]{f | _ {I}} (t _ {1}, \ldots , t _ {n})
$$

称为 f 在 I 上的 restriction。如果 f 编码的 layout 为

$$
L _ {f} = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m}),
$$

则 $f|_I$ 编码的 layout 为

$$
L _ {f | _ {I}} = (s _ {i _ {1}}, \dots , s _ {i _ {r}}): (d _ {i _ {1}}, \dots , d _ {i _ {r}}).
$$

例如，下面给出 tuple morphism f 在 $I=\{2,4\}$ 上的 restriction $f|_I$。

$$
\begin{array}{c c}4&\\8&\rightarrow 4\\1 6&\rightarrow 1 6\\2&\rightarrow 8\\f&\end{array}\qquad \rightsquigarrow \qquad L _ {f} = (2, 1 6, 8, 4): (0, 8, 1, 1 2 8)
$$

$$
\begin{array}{c c}4&\rightarrow 4\\1 6&\rightarrow 1 6\\f | _ {I}&\sim\end{array}\qquad \qquad L _ {f | _ {I}} = (1 6, 4): (8, 1 2 8)
$$

示例 3.1.3.8（Entry inclusion）。前述构造有以下重要特殊情况。如果 $f:(s_1,\ldots,s_m)\to(t_1,\ldots,t_n)$ 是 tuple morphism，且 $1\leq i\leq m$，则 f 的第 i 个 entry $f_i$ 为

$$
(s _ {i}) \xrightarrow [ (i) ]{f _ {i}} (t _ {1}, \ldots , t _ {n})
$$

如果 f 编码的 layout 为

$$
L _ {f} = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m}),
$$

则 $f_i$ 编码的 layout 为

$$
L _ {f _ {i}} = (s _ {i}): (d _ {i}).
$$

例如，下面给出 tuple morphism f 及其第四个 entry $f_4$。

![image](Imgaes/categorical-foundations-cute-layouts-paper/89d60f9cfafe6b3e3578f8f76806cc3e3f8aa42ba5c7b4bf967498eab4542bbc.jpg)


$$
\begin{array}{c c c}&4\\4&1 6\\&8\end{array}\quad \rightsquigarrow \quad L _ {f _ {4}} = (4): (1 2 8)
$$

注记 3.1.3.9。给定 $\langle n\rangle_*\in\mathsf{Fin}_*$，对每个 $i\in\langle n\rangle$，存在 morphism $\varphi_i:\langle1\rangle_*\to\langle n\rangle_*$，将 $*\mapsto*$、$1\mapsto i$。对位于 $\alpha:\langle m\rangle_*\to\langle n\rangle_*$ 之上的 tuple morphism $f:(s_1,\ldots,s_m)\to(t_1,\ldots,t_n)$，第 i 个 entry 位于 composite $\alpha\circ\varphi_i:\langle1\rangle_*\to\langle n\rangle_*$ 之上。

示例 3.1.3.10（Factorization）。假设

$$
(s _ {1}, \ldots , s _ {m}) \xrightarrow [ \alpha ]{f} (t _ {1}, \ldots , t _ {n})
$$

是 tuple morphism，并假设

$$
J = \left\{j _ {1} <   \dots <   j _ {\ell} \right\} \subset \langle n \rangle
$$

是满足 $\mathsf{Image}(\alpha)\subseteq J\cup\{*\}$ 的子集。如果把 map $k\mapsto j_k$ 记为 $\iota:\langle\ell\rangle_*\to\langle n\rangle_*$，则 α 通过唯一 map $\bar\alpha:\langle m\rangle_*\to\langle\ell\rangle_*$ 分解为 $\alpha=\iota\circ\bar\alpha$；把 f through J 的 factorization 定义为 tuple morphism

$$
(s _ {1}, \ldots , s _ {m}) \xrightarrow [ \bar {\alpha} ]{f | ^ {J}} (t _ {j _ {1}}, \ldots , t _ {j _ {\ell}}).
$$

如果 f 编码的 layout 为

$$
L _ {f} = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m}),
$$

则 $f|^J$ 编码的 layout 为

$$
L _ {f | ^ {J}} = (s _ {1}, \ldots , s _ {m}): (d _ {1} ^ {\prime}, \ldots , d _ {m} ^ {\prime}),
$$

其中

$$
d _ {i} ^ {\prime} = \frac {d _ {i}}{\left(\prod_ {k <   \alpha (i) \text { and } k \notin J} t _ {j}\right)}.
$$

例如，下面给出 tuple morphism f through $J=\{2,4,5\}$ 的 factorization $f|^J$。

![image](Imgaes/categorical-foundations-cute-layouts-paper/10c7cbca3c76f69e661144fa79637d1745ef824ea6766e4c358705d881ecbdb5.jpg)


$$
\begin{array}{c c}1 0&\\8 \xrightarrow {} 8&\rightsquigarrow\\8 \xrightarrow {} 8&\\f | ^ {J}&\end{array}\qquad L _ {f | ^ {J}} = (8, 8): (8, 1)
$$

注记 3.1.3.11。Factorization 有一种范畴论解释。沿用示例 3.1.3.10 的记号，可以观察到存在位于 ι 之上的 tuple morphism $i:(t_{j_1},\dots,t_{j_\ell})\to(t_1,\dots,t_n)$，而 $f|^J$ 是 f 沿 i 的 pullback：

$$
\begin{array}{c} (s _ {1}, \ldots , s _ {m}) \xrightarrow {f | ^ {J}} (t _ {j _ {1}}, \ldots , t _ {j _ {\ell}}) \\ \mathrm{id} \Biggl \downarrow \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \Biggl \downarrow i \\ (s _ {1}, \ldots , s _ {m}) \xrightarrow [ f ]{} (t _ {1}, \ldots , t _ {n}) \end{array}
$$

## 3.1.4 Tuple morphism 的 realization

前文已经看到，tuple morphism $f:S\to T$ 编码一个 flat layout $L_f$。本节构造 realization functor

$$
| \cdot |: \text { Tuple } \to \text { FinSet. }
$$

使这种编码显式化。realization functor $|\cdot|$ 把 tuple morphism f 映射到 $L_f$ 的 layout function $|f|$。为了构造 realization functor $|\cdot|$，首先构造辅助 functor

$$
F: \mathbf {T u p l e} \rightarrow \mathbf {F i n S e t}
$$

供后续构造使用。

构造 3.1.4.1。定义 functor

$$
F: \text { Tuple } \to \text { FinSet }
$$

如下。

• 对 **Tuple** 中的 object $S=(s_1,\ldots,s_m)$，定义

$$
F S = [ 0, S) = \prod_ {i = 1} ^ {m} [ 0, s _ {i}).
$$

• 对 **Tuple** 中位于 α 之上的 morphism $f:(s_1,\ldots,s_m)\to(t_1,\ldots,t_n)$，把 $Ff$ 定义为 map

$$
[ 0, S) \xrightarrow {F f} [ 0, T)
$$

其定义为

$$
(F f) (x _ {1}, \ldots , x _ {m}) = (y _ {1}, \ldots , y _ {n})
$$

其中

$$
y _ {j} = \left\{ \begin{array}{l l} x _ {i} & \text { there   exists } 1 \leq i \leq m \text { with } \alpha (i) = j, \\ 0 & \text { else }. \end{array} \right.
$$

容易验证 $F(g\circ f)=Fg\circ Ff$ 且 $F\mathsf{id}_S=\mathsf{id}_{FS}$，因此 F 确实是 functor。

示例 3.1.4.2。假设 $f:(4,4)\to(4,4,4)$ 是位于 $\alpha=(1,3)$ 之上的 tuple morphism。则

$$
F f: [ 0, (4, 4)) \to [ 0, (4, 4, 4))
$$

定义为

$$
(F f) (x _ {1}, x _ {2}) = (x _ {1}, 0, x _ {2}).
$$

示例 3.1.4.3。假设 $g:(3,256,256,512)\to(3,256,256)$ 是位于 $\beta=(*,3,2,*)$ 之上的 tuple morphism。则

$$
F g: [ 0, (3, 2 5 6, 2 5 6, 5 1 2)) \to [ 0, (3, 2 5 6, 2 5 6))
$$

定义为

$$
(F g) (x _ {1}, x _ {2}, x _ {3}, x _ {4}) = (0, x _ {3}, x _ {2}).
$$

构造 3.1.4.4。定义 functor

$$
| \cdot |: \text { Tuple } \to \text { FinSet }
$$

如下。

• 对 **Tuple** 中的 object $S=(s_1,\ldots,s_m)$，定义

$$
| S | = [ 0, \operatorname{size} (S)) = \{0, 1, \dots , \operatorname{size} (S) - 1 \}.
$$

• 对 tuple morphism $f:S\to T$，定义

$$
| f | = \operatorname{colex} _ {T} \circ F f \circ \operatorname{colex} _ {S} ^ {- 1}
$$

回忆定理 2.1.2.18。

如果 $f:S\to T$ 和 $g:T\to U$ 是可复合 tuple morphism，则

$$
\begin{array}{r l} & {| g \circ f | = \mathsf {c o l e x} _ {U} \circ F (g \circ f) \circ \mathsf {c o l e x} _ {S} ^ {- 1}} \\ & {\qquad = \mathsf {c o l e x} _ {U} \circ F g \circ F f \circ \mathsf {c o l e x} _ {S} ^ {- 1}} \\ & {\qquad = \mathsf {c o l e x} _ {U} \circ F g \circ \mathsf {c o l e x} _ {T} ^ {- 1} \circ \mathsf {c o l e x} _ {T} \circ F f \circ \mathsf {c o l e x} _ {S} ^ {- 1}} \\ & {\qquad = | g | \circ | f |} \end{array}
$$

如果 $f=\operatorname{id}_S$ 是 identity morphism，则

$$
\begin{array}{r l} | \mathsf {i d} _ {S} | & = \mathsf {c o l e x} _ {S} \circ F \mathsf {i d} _ {S} \circ \mathsf {c o l e x} _ {S} ^ {- 1} \\ & = \mathsf {c o l e x} _ {S} \circ \mathsf {i d} _ {S} \circ \mathsf {c o l e x} _ {S} ^ {- 1} \\ & = \mathsf {c o l e x} _ {S} \circ \mathsf {c o l e x} _ {S} ^ {- 1} \\ & = \mathsf {i d} _ {| S |}, \end{array}
$$

所以 $|\cdot|$ 确实规定了 functor。接下来观察到，对 **Tuple** 中的 morphism f，map $|f|$ 是 $L_f$ 的 layout function。由此可以轻易推出，**Tuple** 中 morphism 的 composition 与 flat layout 的 composition 相容，参见推论 3.1.4.6。

引理 3.1.4.5。如果 $f:S\to T$ 是 tuple morphism，则 f 的 realization $|f|$ 是 $L_f$ 的 layout function。

$$
| f | = \Phi_ {L _ {f}} ^ {\mathrm{size} (T)}
$$

证明。令 $S=(s_1,\ldots,s_m)$、$T=(t_1,\ldots,t_n)$，并令

$$
L _ {f} = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m})
$$

表示与 f 关联的 layout，其 stride $d_i$ 由公式定义

$$
d _ {i} = \left\{ \begin{array}{l l} 0 & \alpha (i) = * \\ \prod_ {j <   \alpha (i)} t _ {j} & \text {else.} \end{array} \right.
$$

通过在之前复合 colex<sub>S</sub>：$\prod_{i=1}^m[0,s_i)\to[0,\mathsf{size}(S))$，只需证明对任意 $(x_1,\ldots,x_m)\in\prod_{i=1}^m[0,s_i)$，有

$$
\left(\operatorname{colex} _ {T} \circ F f\right) \left(x _ {1}, \dots , x _ {m}\right) = \left(x _ {1}, \dots , x _ {m}\right) \cdot \left(d _ {1}, \dots , d _ {m}\right).
$$

对一般输入 $(x_1,\ldots,x_m)\in\prod_{i=1}^m[0,s_i)$，有

$$
(F f) (x _ {1}, \dots , x _ {m}) = (y _ {1}, \dots , y _ {n})
$$

其中，当 $\alpha(i)=j$ 时 $y_j=x_i$，否则 $y_j=0$。因此

$$
\begin{array}{l} (\mathsf {c o l e x} _ {T} \circ F f) (x _ {1}, \ldots , x _ {m}) = (y _ {1}, \dots , y _ {n}) \cdot (1, t _ {1}, \ldots , t _ {1} \dots t _ {n - 1}) \\ \qquad = \sum_ {j = 1} ^ {n} y _ {j} \cdot t _ {1} \dots t _ {j - 1} \\ \qquad = \sum_ {i = 1} ^ {m} x _ {i} d _ {i} \\ \qquad = (x _ {1}, \ldots , x _ {m}) \cdot (d _ {1}, \ldots , d _ {m}), \end{array}
$$

这正是所需结论。

推论 3.1.4.6。如果 f 和 g 是 non-degenerate 且可复合的 tuple morphism，则

$$
L _ {g \circ f} = L _ {g} \circ L _ {f}
$$

证明。假设 **Tuple** 中的 morphism $f:S\to T$ 和 $g:T\to U$ 分别位于 α 和 β 之上。写成 $S=(s_1,\ldots,s_m)$、$T=(t_1,\ldots,t_n)$。需要检查：

1. `shape(L_{g∘f})` refine `shape(L_f)`：成立，因为 $L_f$ 与 $L_{g\circ f}$ 的 shape 都等于 S。

2. $L_{g\circ f}$ 在 `shape(L_f)` 上 coalesced：成立，因为 tuple morphism $g\circ f$ 是 non-degenerate，因此 layout $L_{g\circ f}$ 也是。

3. $\Phi_{L_{g\circ f}}=\Phi_{L_g}\circ\Phi_{L_f}^{\mathsf{size}(L_g)}$：使用引理 3.1.4.5，有

$$
\begin{array}{c} \Phi_ {L _ {g} \circ f} ^ {\text {size} (U)} = | g \circ f | \\ = | g | \circ | f | \\ = \Phi_ {L _ {g}} ^ {\text {size} (U)} \circ \Phi_ {L _ {f}} ^ {\text {size} (T)}. \end{array}
$$

再在之后复合 inclusion $[0,\mathsf{size}(U))\subset\mathbb{Z}$，并注意 $\mathsf{size}(T)=\mathsf{size}(L_g)$，即可得到结论。

## 3.1.5 Tuple morphism 上的操作

下一个目标是建立“tuple morphism algebra”，其中包括 coalesce、complement、composition、flat division 和 flat product 等操作。我们将证明，每个操作都与 flat layout 上的相应操作相容。

## 3.1.5.1 Sum

tuple morphism f 与 g 的 sum $f\oplus g$，通过 concatenate f 和 g 的 domain 与 codomain 得到。为了精确定义该操作，先在 $\mathsf{Fin}_*$ 的 morphism 上定义相应操作。

定义 3.1.5.1。假设 $\alpha:\langle m\rangle_*\to\langle n\rangle_*$ 和 $\beta:\langle p\rangle_*\to\langle q\rangle_*$ 是 $\mathsf{Fin}_*$ 中的 morphism。把 α 与 β 的 sum 定义为 morphism

$$
\alpha \oplus \beta : \langle m + p \rangle_ {*} \rightarrow \langle n + q \rangle_ {*}
$$

其定义为

$$
(\alpha \oplus \beta) (x) = \left\{ \begin{array}{l l} \alpha (x) & 1 \leq x \leq m \\ n + \beta (x - m) & m + 1 \leq x \leq m + p \\ * & x = *. \end{array} \right.
$$

该操作满足结合律，因此对 $\mathsf{Fin}_*$ 中任意有限 morphism 集合 $\alpha_1,\ldots,\alpha_k$，可以考虑 sum $\alpha_1\oplus\cdots\oplus\alpha_k$。

注记 3.1.5.2。如果 α 和 β 是 tractable pointed map，则 $\alpha\oplus\beta$ 是 tractable 的。

现在可以定义 **Tuple** 中 morphism 的 sum。

定义 3.1.5.3。假设 tuple morphism $f:S\to T$ 和 $g:U\to V$ 分别位于 α 和 β 之上。把 f 与 g 的 sum 定义为 tuple morphism

$$
f \oplus g: S \star U \to T \star V
$$

它位于 $\alpha\oplus\beta$ 之上。该操作满足结合律，因此对 **Tuple** 中任意有限 morphism 集合 $f_1,\ldots,f_k$，可以考虑 sum $f_1\oplus\cdots\oplus f_k$。

示例 3.1.5.4。下面是 tuple morphism f 与 g 的 sum $f\oplus g$ 的示例。

$$
\begin{array}{c c c} 3 2 \longmapsto 3 2 & 4 & 4 \\ 1 6 \longmapsto 1 6 & 4 \xrightarrow {} 2 & 4 \\ f & g & f \oplus g \end{array}
$$

示例 3.1.5.5。下面是 tuple morphism f 与 g 的 sum $f\oplus g$ 的另一个示例。

$$
\begin{array}{c c c} f & g & f \oplus g \\ \hline \end{array}
$$

注记 3.1.5.6。tuple morphism 的 sum 有一种范畴论解释：如果 $f:S\to T$ 和 $g:U\to V$ 是 tuple morphism，则

$$
f \oplus g: S \star U \to T \star V
$$

是 arrow category Ar(Tuple) 中 f 与 g 的 coproduct。

## 3.1.5.2 Squeeze

经常需要从 tuple 中移除所有整数 1 的实例。squeeze functor 可以完成这一操作。

定义 3.1.5.7。定义 functor

$$
\text { Tuple } \xrightarrow {\text { squeeze } (-)} \text { Tuple }
$$

如下。如果 $S=(s_1,\ldots,s_m)$ 是 **Tuple** 中的 object，定义

$$
\operatorname{squeeze} (S) = \left(s _ {i _ {1}}, \dots , s _ {i _ {k}}\right)
$$

其中 $\{i_1<\dots<i_k\}\subset\langle m\rangle$ 是满足 $s_{i_j}\neq1$ 的索引。如果 $f:(s_1,\ldots,s_m)\to(t_1,\ldots,t_n)$ 是 tuple morphism，定义

$$
\operatorname{squeeze} (f): \operatorname{squeeze} (S) \to \operatorname{squeeze} (T)
$$

为 tuple morphism

$$
\operatorname{squeeze} (f) = \left(f \mid_ {I}\right) | ^ {J}
$$

其中 $f|_I$ 是 f 在

$$
I = \{i \in \langle m \rangle \mid s _ {i} \neq 1 \}
$$

上的 restriction，如定义 3.1.3.7 所示；$(f|_I)|^J$ 是 $f|_I$ through

$$
J = \{j \in \langle n \rangle \mid t _ {j} \neq 1 \},
$$

的 factorization，如定义 3.1.3.10 所示。

示例 3.1.5.8。下面给出 morphism f 和对应的 morphism `squeeze(f)`。

![image](Imgaes/categorical-foundations-cute-layouts-paper/821c58baf4aff53588c7863f04be451fcc5c07f2bcca6dd32e17fd5502257b49.jpg)


示例 3.1.5.9。如果 $f:(s_1,\ldots,s_m)\to(t_1,\ldots,t_n)$ 是 tuple morphism，则

$$
f = \operatorname{squeeze} (f) \quad \Leftrightarrow \quad \text {   no   } s _ {i}, t _ {j} \text {   is   equal   to   1.   }
$$

命题 3.1.5.10。如果 f 是 tuple morphism，则

$$
L _ {\text { squeeze } (f)} = \text { squeeze } (L _ {f}).
$$

证明。假设 $f:(s_1,\ldots,s_m)\to(t_1,\ldots,t_n)$ 是 tuple morphism，并令

$$
L _ {f} = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m})
$$

为与 f 关联的 flat layout。令 $I=\{i_1<\cdots<i_{m'}\}\subset\langle m\rangle$ 表示满足 $s_{i_k}\neq1$ 的索引子集。则

$$
\begin{array}{l} L _ {f | _ {I}} = (s _ {i _ {1}}, \ldots , s _ {i _ {k}}): (d _ {i _ {1}}, \ldots , d _ {i _ {k}}) \\ = \text { squeeze } (L _ {f}). \end{array}
$$

令 $J=\{j_1<\cdots<j_{n'}\}\subset\langle n\rangle$ 表示满足 $t_{j_k}\neq1$ 的索引子集，因此 `squeeze(f)=(f|_I)|^J`。令 β 表示 `squeeze(f)` 所在的 map。则

$$
L _ {\text { squeeze } (f)} = L _ {(f | _ {I}) | ^ {J}} = (s _ {i _ {1}}, \ldots , s _ {i _ {k}}): (d _ {i _ {1}} ^ {\prime}, \ldots , d _ {i _ {k}} ^ {\prime})
$$

其中

$$
\begin{array}{c} d _ {i _ {k}} ^ {\prime} = \frac {d _ {i _ {k}}}{\left(\prod_ {\ell <   \beta (k) \text { and } \ell \notin J} t _ {\ell}\right)} \\ = d _ {i _ {k}} \end{array}
$$

因为对任意 $\ell\notin J$ 都有 $t_\ell=1$。因此

$$
\begin{array}{c} L _ {\text { squeeze} (f)} = (s _ {i _ {1}}, \ldots , s _ {i _ {k}}): (d _ {i _ {1}}, \ldots , d _ {i _ {k}}) \\ = \text { squeeze } (L _ {f}). \end{array}
$$

观察 3.1.5.11。如果 f 是 tuple morphism，则

$$
\operatorname{squeeze} (\operatorname{squeeze} (f)) = \operatorname{squeeze} (f),
$$

so 

$$
\text { Tuple } \xrightarrow {\text { squeeze } (-)} \text { Tuple }
$$

是幂等 functor。

## 3.1.5.3 Sort

sort 操作 $f\mapsto\mathsf{sort}(f)$ 置换 f 的 domain，使所得 morphism 按以下意义排序。

定义 3.1.5.12。称 tuple morphism

$$
(s _ {1}, \ldots , s _ {m}) \xrightarrow [ \alpha ]{f} (t _ {1}, \ldots , t _ {n})
$$

已排序，如果对任意 $1\leq i,j\leq m$，以下条件都成立：

1. 如果 $\alpha(i)=*\neq\alpha(j)$，则 $i<j$；

2. 如果 $\alpha(i)=*=\alpha(j)$，则

$$
i \leq j \quad \Rightarrow \quad s _ {i} \leq s _ {j}.
$$

3. 如果 $\alpha(i)\neq*\neq\alpha(j)$，则

$$
i \leq j \quad \Rightarrow \quad \alpha (i) \leq \alpha (j).
$$

示例 3.1.5.13。下图中的 morphism $f_1$、$f_2$、$f_3$

$$
\begin{array}{c c c} 1 2 8 & \xrightarrow {} & 1 2 8 \\ 5 1 2 & \xrightarrow {} & 5 1 2 \\ 3 & \xrightarrow {} & f _ {1} \end{array} \quad \begin{array}{c c c} 4 & \xrightarrow {} & 4 \\ 1 & \xrightarrow {} & 1 \\ 1 & \xrightarrow {} & 8 \\ 1 & \xrightarrow {} & 6 4 \end{array} \quad \begin{array}{c c c} 6 0 & \xrightarrow {} & 6 0 \\ 2 0 & \xrightarrow {} & 2 \\ 3 2 & \xrightarrow {} & 2 0 \\ 8 & \xrightarrow {} & 4 \end{array}
$$

已排序，而下图中的 morphism $g_1$、$g_2$、$g_3$

![image](Imgaes/categorical-foundations-cute-layouts-paper/894632f74a4d4d369335e4a5be37bf17744424d745a127d47d936bfc74b4c31f.jpg)


未排序。morphism $g_1$、$g_2$、$g_3$ 分别违反条件 3、1、2。

命题 3.1.5.14。如果 f 是已排序 tuple morphism，则 flat layout $L_f$ 已排序。

证明。假设

$$
(s _ {1}, \ldots , s _ {m}) \xrightarrow [ \alpha ]{f} (t _ {1}, \ldots , t _ {n})
$$

已排序，并考虑 layout

$$
L _ {f} = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m}).
$$

假设 $1\leq i<m$。要证明 $d_i<d_{i+1}$，或 $d_i=d_{i+1}$ 且 $s_i\leq s_{i+1}$。需要考虑两种情况。

• 情况 1：假设 $\alpha(i)=*$，所以 $d_i=0$。如果 $\alpha(i+1)=*$，则 $d_{i+1}=0$；由于 f 已排序，有 $s_i\leq s_{i+1}$。如果 $\alpha(i+1)\neq*$，则 $d_{i+1}\geq1>0=d_i$。

• 情况 2：假设 $\alpha(i)\neq*$，此时 $\alpha(i+1)\neq*$ 且 $\alpha(i)<\alpha(i+1)$。则

$$
d _ {i} = \prod_ {j <   \alpha (i)} t _ {j} \leq \prod_ {j <   \alpha (i + 1)} = d _ {i + 1},
$$

其中等号只在 $s_i=t_{\alpha(i)}=1$ 时成立，这蕴含 $s_i\leq s_{i+1}$。

因此 $L_f$ 已排序。

下面定义 **Tuple** 上的 $\mathsf{sort}(-)$ 操作。如果 f 是 tuple morphism，则 $\mathsf{sort}(f)$ 通过在 f 之前复合适当置换 g 得到。

构造 3.1.5.15。假设

$$
(s _ {1}, \ldots , s _ {m}) \xrightarrow [ \alpha ]{f} (t _ {1}, \ldots , t _ {n})
$$

是 tuple morphism。按如下方式定义置换 $\sigma\in\Sigma_m$。令

$$
\begin{array}{l} P = \{i \in \langle m \rangle | \alpha (i) = * \}, \\ Q = \{i \in \langle m \rangle | \alpha (i) \neq * \}, \end{array}
$$

因此 $\langle m\rangle$ 是 P 与 Q 的不交并。如果以下条件之一成立，定义 P 上的线性序 $i_1\preceq_P i_2$：

1. $s_{i_1}<s_{i_2}$；或

2. $s_{i_1}=s_{i_2}$ 且 $i_1\leq i_2$。

如果 $\alpha(j_1)\leq\alpha(j_2)$，定义 Q 上的线性序 $j_1\preceq_Q j_2$。如果以下条件之一成立，定义 $\langle m\rangle$ 上的线性序 $i_1\preceq i_2$：

1. $i_1\in P$ 且 $i_2\in Q$；

2. $i_1,i_2\in P$ 且 $i_1\preceq_P i_2$；或

3. $i_1,i_2\in Q$ 且 $i_1\preceq_Q i_2$。

令 σ 为与 $\langle m\rangle$ 上线性序 ⪯ 关联的置换，$\sigma^{-1}$ 为其逆。map $\sigma_*^{-1}:\langle m\rangle_*\to\langle m\rangle_*$ 由 tuple morphism

$$
g: \big (s _ {\sigma^ {- 1} (1)}, \ldots , s _ {\sigma^ {- 1} (m)} \big) \to \big (s _ {1}, \ldots , s _ {m} \big),
$$

覆盖，并把 $\mathsf{sort}(f)$ 定义为 composite

$$
\operatorname{sort} (f) = f \circ g.
$$

示例 3.1.5.16。示例 3.1.5.13 中 morphism $g_1$、$g_2$、$g_3$ 的排序结果如下

。

![image](Imgaes/categorical-foundations-cute-layouts-paper/2fca18bb84010c2fd808045c17da471790bedec76f4c35ec31f3e85c36f1a5a0.jpg)



引理 3.1.5.17。假设 $f:S\to T$ 是 tuple morphism。则 f 已排序，当且仅当 $\mathsf{sort}(f)=f$。证明。sort(−) 的构造保证对任意 tuple morphism f，`sort(f)` 已排序。特别地，如果 $f=\mathsf{sort}(f)$，则 f 已排序。反之，如果 f 已排序，则构造 3.1.5.15 中的置换 $\sigma\in\Sigma_m$ 是 identity permutation，所以 $g=\mathsf{id}_S$，于是


$$
\operatorname{sort} (f) = f \circ \mathrm{id} _ {S} = f.
$$

命题 3.1.5.18。如果 f 是 tuple morphism，则

$$
L _ {\text { sort } (f)} = \text { sort } (L _ {f}).
$$

证明。沿用构造 3.1.5.15 的记号，有 `sort(f)=f∘g`，其中

$$
g: \left(s _ {\sigma^ {- 1} (1)}, \ldots , s _ {\sigma^ {- 1} (m)}\right) \to \left(s _ {1}, \ldots , s _ {m}\right)
$$

位于 $\sigma_*^{-1}:\langle m\rangle_*\to\langle m\rangle_*$ 之上。如果 $L_f=(s_1,\dots,s_m):(d_1,\dots,d_m)$，则

$$
\begin{array}{c} L _ {\mathsf {s o r t} (f)} = (s _ {1} ^ {\prime}, \ldots , s _ {m} ^ {\prime}): (d _ {1} ^ {\prime}, \ldots , d _ {m} ^ {\prime}) \\ = (s _ {\sigma^ {- 1} (1)}, \ldots , s _ {\sigma^ {- 1} (m)}): (d _ {\sigma^ {- 1} (1)}, \ldots , d _ {\sigma^ {- 1} (m)}). \end{array}
$$

由于 $L_{\mathsf{sort}(f)}$ 的 mode 是 $L_f$ 的 mode 的一个置换，只需证明 $L_{\mathsf{sort}(f)}$ 已排序。假设 $1\leq i<m$。先假设 $\sigma^{-1}(i)\in P$，因此 $d_i'=d_{\sigma^{-1}(i)}=0$。如果 $\sigma^{-1}(i+1)\in P$，则 $d_{i+1}'=0$；根据 σ 的构造，有 $s_i'=s_{\sigma^{-1}(i)}\leq s_{\sigma^{-1}(i+1)}=s_{i+1}'$。如果 $\sigma^{-1}(i+1)\in Q$，则 $d_{i+1}'>0=d_i'$。接下来假设 $\sigma^{-1}(i)\in Q$。根据 σ 的构造，$\sigma^{-1}(i+1)\in Q$ 且 $\alpha(\sigma^{-1}(i))<\alpha(\sigma^{-1}(i+1))$，并有

$$
\begin{array}{r l} & d _ {i} ^ {\prime} = d _ {\sigma^ {- 1} (i)} = \prod_ {j <   \alpha (\sigma^ {- 1} (i))} t _ {j} \\ & \qquad \leq \prod_ {j <   \alpha (\sigma^ {- 1} (i + 1))} t _ {j} \\ & \qquad = d _ {\sigma^ {- 1} (i + 1)} \\ & \qquad = d _ {i + 1} ^ {\prime}, \end{array}
$$

其中等号成立，当且仅当 $t_{\alpha(\sigma^{-1}(i))}=\cdots=t_{\alpha(\sigma^{-1}(i+1))-1}=1$。特别地，$s_i'=s_{\sigma^{-1}(i)}=t_{\alpha(\sigma^{-1}(i))}=1$，因此 $s_i'\leq s_{i+1}'$。所以 $L_{\mathsf{sort}(f)}$ 已排序，进而 $L_{\mathsf{sort}(f)}=\mathsf{sort}(L_f)$。□ 注记 3.1.5.19。操作 $\mathsf{sort}(-)$ 不是 functorial 的。例如，考虑 tuple morphism $(2,3)\xrightarrow{f}(3,2)$ 和 $(10,25)\xrightarrow{g}(25,10)$。

f 与 g 可复合，并满足 $g\circ f=\mathsf{id}_{(25,10)}$，但已排序 morphism $(10,25)\xrightarrow{\mathsf{sort}(f)}(10,25)$ 和 $(25,10)\xrightarrow{\mathsf{sort}(g)}(25,10)$

不可复合。

## 3.1.5.4 Coalesce

首先引入 coalesced tuple morphism 的概念。

定义 3.1.5.20。假设 $f:S\to T$ 是位于 α 之上的 tuple morphism。如果以下条件成立，就称 f 为 coalesced：

1. $S=\mathsf{squeeze}(S)$；

2. 对任意 $1\leq i<\mathsf{len}(S)$，以下条件中恰有一个成立：

(a) $\alpha ( i ) = * \neq \alpha ( i + 1 )$ 

(b) $\alpha ( i ) \neq * = \alpha ( i + 1 )$ 2 

(c) $\alpha ( i ) > \alpha ( i + 1 ) , \mathrm { o r }$ 

(d) $\alpha ( i ) < \alpha ( i + 1 )$ , and there exists $\alpha ( i ) < j < \alpha ( i + 1 )$ with $t _ { j } > 1$ 

示例 3.1.5.21。如果存在某个 $1\leq i<\mathsf{len}(S)$ 满足 $\alpha(i+1)=\alpha(i)+1$，则 f 不是 coalesced。

注记 3.1.5.22。如果 $f:S\to T$ 是满足 $f=\mathsf{squeeze}(f)$ 的 tuple morphism，则 f 为 coalesced，当且仅当对任意 $1\leq i<\mathsf{len}(S)$，以下条件之一成立：

1. $\alpha ( i ) = * \neq \alpha ( i + 1 ) .$ 

2. $\alpha ( i ) \neq * = \alpha ( i + 1 )$ 

3. $\alpha ( i ) > \alpha ( i + 1 ) , \mathrm { o r }$ 

4. $\alpha ( i + 1 ) \neq \alpha ( i ) + 1 .$ 

示例 3.1.5.23。以下 morphism

![image](Imgaes/categorical-foundations-cute-layouts-paper/1a1fe11033b7a1889b2604ab64a658ecbbfb108168c09eaa54240a839c5a05b5.jpg)


是 coalesced，而以下 morphism

![image](Imgaes/categorical-foundations-cute-layouts-paper/2efb74c510834ce37156da0d1be47639531b68afa2fe22e190513eadc6e15f8b.jpg)


不是 coalesced。

命题 3.1.5.24。假设 f 是 tuple morphism。则 f 是 coalesced，当且仅当 $L_f$ 是 coalesced。

证明。假设 $f:(s_1,\ldots,s_m)\to(t_1,\ldots,t_n)$ 是 tuple morphism，并令

$$
L _ {f} = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m})
$$

为 f 编码的 layout。

先假设 f 是 coalesced。则 `shape(L_f)=domain(f)` 中没有条目等于 1。假设 $1\leq i<m$。要证明 $s_id_i\neq d_{i+1}$。如果 $d_i=0$，则 $\alpha(i)=*$，并有

$$
\begin{array}{r l} s _ {i} d _ {i} = d _ {i + 1} & \Leftrightarrow \quad d _ {i + 1} = 0 \\ & \Leftrightarrow \quad \alpha (i + 1) = * \end{array}
$$

但根据 f 为 coalesced 的假设，有 $\alpha(i+1)\neq*$，所以 $s_id_i\neq d_{i+1}$。如果 $d_i\neq0$，则 $\alpha(i)\neq*$。如果 $\alpha(i+1)=*$，则 $d_{i+1}=0$，所以 $s_id_i\neq d_{i+1}$。如果 $\alpha(i+1)<\alpha(i)$，则 $d_i\geq d_{i+1}$；由于 $s_i\neq1$，有 $s_id_i>d_{i+1}$。最后，如果 $\alpha(i)<\alpha(i+1)$，则

$$
\begin{array}{c} s _ {i} d _ {i} = s _ {i} \cdot \left(\prod_ {j <   \alpha (i)} t _ {j}\right) = \prod_ {j \leq \alpha (i)} t _ {j} \\ <   \prod_ {j <   \alpha (i + 1)} t _ {j} \\ = d _ {i + 1}. \end{array}
$$

因此 $L_f$ 是 coalesced。

接下来假设 layout $L_f$ 是 coalesced。则 `domain(f)=shape(L_f)` 中没有条目等于 1。假设 $1\leq i<m$。如果 $\alpha(i)=*$，则 $d_i=0$；由于 $L_f$ 是 coalesced，必有 $d_{i+1}\neq s_id_i=0$，所以 $\alpha(i+1)\neq*$。假设 $\alpha(i)\neq*$ 且 $\alpha(i)<\alpha(i+1)$。由于 $L_f$ 是 coalesced，有 $s_id_i\neq d_{i+1}$。但写成

$$
s _ {i} d _ {i} = \prod_ {j \leq \alpha (i)} t _ {j},
$$

and 

$$
d _ {i + 1} = \prod_ {j <   \alpha (i + 1)} t _ {j},
$$

可知 $\prod_{\alpha(i)<j<\alpha(i+1)}t_j\neq1$。特别地，存在某个 $\alpha(i)<j<\alpha(i+1)$ 满足 $t_j>1$。所以 f 是 coalesced。□

下面定义 tuple morphism 上的 coal(−) 操作。

构造 3.1.5.25。假设 f 是 tuple morphism。按如下方式定义 morphism `coal(f)`：

1. 首先令 $g=\mathsf{squeeze}(f)$，并用 $\beta:\langle m\rangle_*\to\langle n\rangle_*$ 表示 g 所在的 map。

2. 接下来在 $\langle m\rangle$ 上定义等价关系 ∼：如果以下条件之一成立，则 $i\sim i'$：

(a) $\beta ( i ^ { \prime \prime } ) = * \mathrm { f o r } i \le i ^ { \prime \prime } \le i ^ { \prime } ,$ or 

(b) $\beta ( i ^ { \prime \prime } ) = \beta ( i ) + ( i ^ { \prime \prime } - i ) \mathrm { f o r } i \le i ^ { \prime \prime } \le i ^ { \prime } .$ 

商集 $\langle m\rangle/\sim$ 按“当 $i_1\leq i_2$ 时，$[i_1]\leq[i_2]$”排序，因此可把它与 $\langle\bar m\rangle$ 等同，其中 $\bar m$ 是 $\langle m\rangle/\sim$ 的大小。

3. 接下来在 $\langle n\rangle$ 上定义等价关系 ∼：如果存在 $i\in\langle m\rangle$，使

$$
\beta (i + (j ^ {\prime \prime} - j)) = \beta (i) + (j ^ {\prime \prime} - j)
$$

对所有 $j\leq j''\leq j'$ 成立，则 $j\sim j'$。商集 $\langle n\rangle/\sim$ 按“当 $j_1\leq j_2$ 时，$[j_1]\leq[j_2]$”排序，因此可把它与 $\langle\bar n\rangle$ 等同，其中 $\bar n$ 是 $\langle n\rangle/\sim$ 的大小。

4. 接着观察到 map $\beta:\langle m\rangle_*\to\langle n\rangle_*$ 下降为 map

$$
\bar {\beta}: \langle \bar {m} \rangle_ {*} \to \langle \bar {n} \rangle_ {*}
$$

其定义为 $\bar\beta([i])=[\beta(i)]$。

5. `coal(f)` 的 domain $\bar S=(\bar s_1,\dots,\bar s_{\bar m})$ 通过以下方式定义：令

$$
\bar {s} _ {i} = \prod_ {i ^ {\prime} \in I} s _ {i ^ {\prime}}
$$

其中 $i\in\langle\bar m\rangle$ 对应等价类 $I\in\langle m\rangle/\sim$。`coal(f)` 的 codomain $\bar T=(\bar t_1,\dots,\bar t_{\bar n})$ 通过以下方式定义：令

$$
\bar {t} _ {j} = \prod_ {j ^ {\prime} \in J} t _ {j ^ {\prime}}
$$

其中 $j\in\langle\bar n\rangle$ 对应等价类 $J\in\langle n\rangle/\sim$。随后定义

$$
\operatorname{coal} (f): \bar {S} \to \bar {T}
$$

为位于 $\bar\beta$ 之上的 tuple morphism。

示例 3.1.5.26。下面给出 tuple morphism f 与 coalesced morphism `coal(f)` 的示例。

![image](Imgaes/categorical-foundations-cute-layouts-paper/d8b19fdd4e539c59e482168f12f48ca24e61269e6a41636aa90f580bb6ec80a4.jpg)


示例 3.1.5.27。可以按如下方式 coalesce 示例 3.1.5.8 中的 morphism f。

![image](Imgaes/categorical-foundations-cute-layouts-paper/d2a71ed8b2bdc41bf93b7c1828b5503cca6f7404673d54f2513733831c096f6f.jpg)



命题 3.1.5.28。如果 f 是 tuple morphism，则


1. `coal(f)` 是 coalesced；

2. $L _ { \mathsf { c o a l } ( f ) } = { \mathsf { c o a l } } ( L _ { f } ) .$ 

证明。首先证明 `coal(f)` 是 coalesced。由构造可立即得出：应用 squeeze 会消除所有等于 1 的 mode，而构造中取商会合并所有满足 $\alpha(i+1)=\alpha(i)+1$ 的相邻 mode。

接下来证明 $L_{\mathsf{coal}(f)}=\mathsf{coal}(L_f)$。根据命题 2.1.4.18 和命题 3.1.5.24，只需证明 $\Phi_{\mathsf{coal}(f)}=\Phi_f$。对 f 应用 squeeze(−) 显然不会影响关联 layout function，因此需要证明构造中取商不会改变关联 layout 的 layout function。该商可以分步形成，每一步合并满足 $\alpha(i)=*=\alpha(i+1)$ 或 $\alpha(i+1)=\alpha(i)+1$ 的相邻 mode。二者分别对应把相邻 mode $s_i,s_{i+1}:0,0$ 替换成 $s_is_{i+1}:0$，以及把 $s_i,s_{i+1}:d_i,s_id_i$ 替换成 $s_is_{i+1}:d_i$。这两类操作都不会改变 layout function，因此 $\Phi_{L_{\mathsf{coal}(f)}}=\Phi_{\mathsf{coal}(L_f)}$，结论得证。

## 3.1.5.5 Concatenate

下面定义 tuple morphism 上的 concatenation 操作。该操作可应用于满足下述“不相交”条件的 tuple morphism。

定义 3.1.5.29。假设 $\alpha:\langle m\rangle_*\to\langle n\rangle_*$ 和 $\beta:\langle p\rangle_*\to\langle n\rangle_*$ 是 Fin<sub>∗</sub> 中具有相同 codomain 的 morphism。如果

$$
\operatorname{Image} (\alpha) \cap \operatorname{Image} (\beta) = \{* \}.
$$

就称 α 与 β 的像不相交。构造 3.1.5.30。如果 $\alpha:\langle m\rangle_*\to\langle n\rangle_*$ 和 $\beta:\langle p\rangle_*\to\langle n\rangle_*$ 的像不相交，则存在定义良好的 morphism

$$
\alpha \star \beta : \langle m + p \rangle_ {*} \rightarrow \langle n \rangle_ {*}
$$

其定义为

$$
(\alpha \star \beta) (i) = \left\{ \begin{array}{l l} * & i = * \\ \alpha (i) & 1 \leq i \leq m \\ \beta (i - m) & m + 1 \leq i \leq m + p. \end{array} \right.
$$

该操作满足结合律，因此对 Fin<sub>∗</sub> 中任意像两两不相交的 morphism 集合 $\alpha_1,\ldots,\alpha_k$，可以考虑 $\alpha_1\star\cdots\star\alpha_k$。

注记 3.1.5.31。如果 α 和 β 是 tractable pointed map，且二者像不相交，则 $\alpha\star\beta$ 是 tractable 的。

定义 3.1.5.32。假设

$$
f: S \to T
$$

and 

$$
g: U \to T
$$

是分别位于 α 和 β 之上的 tuple morphism。如果 morphism α 与 β 的像不相交，就称 f 与 g 的像不相交。

示例 3.1.5.33。考虑下图中的 tuple morphism f、g、h。

![image](Imgaes/categorical-foundations-cute-layouts-paper/bbc295a80b3292b94b3c5dfb231f40bcb1036c4fa27780a91a0d5bd608d7769f.jpg)


f 与 g 的像不相交，而 h 与 g 的像不相交条件不成立。

构造 3.1.5.34。假设

$$
f: S \to T, \text { and } g: U \to T
$$

是分别位于 α 和 β 之上的 tuple morphism，并且 f 与 g 的像不相交。把 f 与 g 的 concatenation 定义为 morphism

$$
f \star g: S \star U \to T
$$

它位于 $\alpha\star\beta$ 之上。该操作满足结合律，因此对任意像两两不相交的有限 morphism 集合 $f_i$，可以考虑 $f_1\star\cdots\star f_k$。

示例 3.1.5.35。如果 f 与 g 是示例 3.1.5.33 中 **Tuple** 的 morphism，则其 concatenation 如下图所示。

![image](Imgaes/categorical-foundations-cute-layouts-paper/23b5376f57658573396c48d14914ecdb3afd5e61c8de10a9ae66e3fb56356181.jpg)



f ⋆ g


示例 3.1.5.36。假设 $f:(s_1,\ldots,s_m)\to(t_1,\ldots,t_n)$ 是 tuple morphism，并对任意 $1\leq i\leq m$ 令

$$
f _ {i}: (s _ {i}) \to (t _ {1}, \dots , t _ {n})
$$

表示示例 3.1.3.8 中 f 的第 i 个 entry。则可以写成

$$
f = f _ {1} \star \dots \star f _ {m}
$$

即其 entry 的 concatenation。

引理 3.1.5.37。假设 tuple morphism $f_1:S_1\to T$ 与 $f_2:S_2\to T$ 的像不相交。如果 $g:T\to U$ 是任意 tuple morphism，则

$$
g \circ (f _ {1} \star f _ {2}) = (g \circ f _ {1}) \star (g \circ f _ {2}).
$$

证明。假设 $f_1$、$f_2$、g 分别位于 $\alpha_1:\langle m_1\rangle_*\to\langle n\rangle_*$、$\alpha_2:\langle m_2\rangle_*\to\langle n\rangle_*$、$\beta:\langle n\rangle_*\to\langle p\rangle_*$ 之上。所讨论的两个 map 具有相同 domain 和 codomain，因此只需证明

$$
\beta \circ (\alpha_ {1} \star \alpha_ {2}) = (\beta \circ \alpha_ {1}) \star (\beta \circ \alpha_ {2}).
$$

计算得

$$
\begin{array}{l} (\beta \circ (\alpha_ {1} \star \alpha_ {2})) (i) = \beta ((\alpha_ {1} \star \alpha_ {2}) (i)) \\ = \left\{ \begin{array}{l l} \beta (*) & i = * \\ \beta (\alpha_ {1} (i)) & 1 \leq i \leq m _ {1} \\ \beta (\alpha_ {2} (i - m _ {1})) & m _ {1} + 1 \leq i \leq m _ {1} + m _ {2} \end{array} \right. \\ = \left\{ \begin{array}{l l} * & i = * \\ (\beta \circ \alpha_ {1}) (i) & 1 \leq i \leq m _ {1} \\ (\beta \circ \alpha_ {2}) (i - m _ {1}) & m _ {1} + 1 \leq i \leq m _ {1} + m _ {2} \end{array} \right. \\ = ((\beta \circ \alpha_ {1}) \star (\beta \circ \alpha_ {2})) (i). \end{array}
$$

命题 3.1.5.38。假设 $f_1,\ldots,f_k$ 是 **Tuple** 中具有相同 codomain 且像两两不相交的 morphism。则 layout $L_{f_1},\ldots,L_{f_k}$ 满足

$$
L _ {f _ {1} \star \dots \star f _ {k}} = L _ {f _ {1}} \star \dots \star L _ {f _ {k}}.
$$

证明。先对 $k=2$ 证明。假设

$$
f = (s _ {1}, \ldots , s _ {m}) \rightarrow (t _ {1}, \ldots , t _ {n}), \text {and} g: (u _ {1}, \ldots , u _ {p}) \rightarrow (t _ {1}, \ldots , t _ {n})
$$

的像不相交，并写成

$$
L _ {f} = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m}), \text {   and   } L _ {g} = (u _ {1}, \dots , u _ {p}): (d _ {1} ^ {\prime}, \dots , d _ {p} ^ {\prime}).
$$

则 layout $L_{f\star g}$ 为

$$
L _ {f \star g} = (s _ {1}, \ldots , s _ {m}, u _ {1}, \ldots , u _ {p}): (e _ {1}, \ldots , e _ {m + m ^ {\prime}})
$$

其中

$$
\begin{array}{l} e _ {i} = \prod_ {j <   (\alpha \star \beta) (i)} t _ {j} \\ = \left\{ \begin{array}{l l} \prod_ {j <   \alpha (i)} t _ {j} & 1 \leq i \leq m \\ \prod_ {j <   \beta (i - m)} t _ {j} & m + 1 \leq i \leq m + m ^ {\prime}. \end{array} \right. \\ = \left\{ \begin{array}{l l} d _ {i} & 1 \leq i \leq m \\ d _ {i - m} ^ {\prime} & m + 1 \leq i \leq m + m ^ {\prime}. \end{array} \right. \end{array}
$$

这完成了 $k=2$ 时的证明。一般情况由 tuple morphism concatenation 与 flat layout concatenation 的结合律得到。□

## 3.1.5.6 Complement

首先定义互为 complement 的 tuple morphism。

定义 3.1.5.39。假设 $f:S\to T$ 和 $g:U\to T$ 是 tuple morphism。如果以下条件成立，就称 g 是 f 的 complement：

1. f 与 g 的像不相交；

2. concatenation

$$
f \star g: S \star U \xrightarrow {\cong} T
$$

是 isomorphism。

示例 3.1.5.40。如果 f 与 g 是下图中的 morphism，

$$
\begin{array}{c}1 6\\\rightarrow 3 2\\3 2 \xrightarrow {} 3 2\\3 2 \xrightarrow {} 1 0\end{array}f \quad \text {   g   } \quad\begin{array}{c}1 6\\\rightarrow 3 2\\1 0 \xrightarrow {} 3 2\\1 6\end{array}
$$

则 g 是 f 的 complement。

示例 3.1.5.41。如果 f 是下图中的 morphism，

$$
\begin{array}{c} 2 5 6 \\ 1 2 8 \\ 1 2 8 \end{array} \xrightarrow {} \begin{array}{c} 1 2 8 \\ 2 5 6 \end{array}
$$

则 f 不存在 complement。

下面证明，互为 complement 的 tuple morphism 会产生互为 complement 的 flat layout。

命题 3.1.5.42。如果 $f:S\to T$ 是 tuple morphism，且 g 是 f 的 complement，则 $L_g$ 是 $L_f$ 的 size(T)-complement。

证明。记 $S=\mathsf{domain}(f)$、$U=\mathsf{domain}(g)$、$T=\mathsf{codomain}(f)=\mathsf{codomain}(g)$。首先注意

$$
\begin{array}{r l} \mathsf {s i z e} (L _ {f}) \cdot \mathsf {s i z e} (L _ {g}) & = \mathsf {s i z e} (L _ {f} \star L _ {g}) \\ & = \mathsf {s i z e} (L _ {f \star g}) \\ & = \mathsf {s i z e} (S \star U) \\ & = \mathsf {s i z e} (T). \end{array}
$$

其次，$f\star g$ 是 isomorphism，因此

$$
| f \star g | = \Phi_ {L _ {f \star g}} ^ {\mathrm{size} (T)}
$$

也是 isomorphism；这里使用了引理 3.1.4.5 对 $\Phi_{L_{f\star g}}^{\mathsf{size}(T)}$ 的刻画。

命题 3.1.5.43。如果 f 是单射 tuple morphism，则

$$
\operatorname{coal} ^ {\flat} \left(L _ {f ^ {c}}\right) = \operatorname{comp} ^ {\flat} \left(L _ {f}, \operatorname{size} (T)\right).
$$

证明。根据命题 3.1.5.42，$L_{f^c}$ 是 $L_f$ 的 size(T)-complement。由于 $f^c$ 已排序，$L_{f^c}$ 也已排序；根据命题 2.1.6.33，

$$
\operatorname{coal} ^ {\flat} \left(L _ {f ^ {c}}\right) = \operatorname{comp} ^ {\flat} \left(L _ {f}, \operatorname{size} (T)\right),
$$

因为这两个 layout 都是 $L_f$ 的同 size、flat、已排序、coalesced complement。□

命题 3.1.5.44。如果 $f:(s_1,\ldots,s_m)\to(t_1,\ldots,t_n)$ 是具有 standard form 的单射 tuple morphism，则

$$
L _ {f ^ {c}} = \mathsf {c o m p} ^ {\flat} (L _ {f}).
$$

证明。写成

$$
L _ {f} = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m})
$$

表示 f 编码的 layout。根据命题 3.1.5.42，$L_{f^c}$ 是 $L_f$ 的 size(T)-complement，其中

$$
\begin{array}{c} \text {size} (T) = t _ {1} \dots t _ {n} = (t _ {1} \dots t _ {n - 1}) t _ {n} \\ = d _ {m} s _ {m}. \end{array}
$$

根据构造，$f^c$ 已排序，因此 $L_{f^c}$ 也已排序。此外，因为 f 具有 standard form，所以 $f^c$ 是 coalesced。根据命题 2.1.6.23，

$$
L _ {f ^ {c}} = \mathsf {c o m p} ^ {\flat} (L _ {f}).
$$

定义 3.1.5.45。假设 f 是位于 $\alpha:\langle m\rangle_*\to\langle n\rangle_*$ 之上的 tuple morphism。如果 α 是单射，就称 f 是 complementable 的。

构造 3.1.5.46。假设 $f:(s_1,\ldots,s_m)\to(t_1,\ldots,t_n)$ 是 complementable tuple morphism。令 $j_1<\cdots<j_{n-m}$ 表示 $\langle n\rangle$ 中不在 α 像内的索引集合。把 f 的 complement 定义为 tuple morphism

$$
f ^ {c}: (t _ {j _ {1}}, \dots , t _ {j _ {k}}) \to (t _ {1}, \dots , t _ {n})
$$

它位于 map $\mathsf{complement}(\alpha):\langle n-m\rangle_*\to\langle n\rangle_*$ 之上，该 map 定义为 $k\mapsto j_k$。由构造可知，$f^c$ 在定义 3.1.5.39 的意义下是 f 的 complement。

示例 3.1.5.47。下面给出 morphism f 及其 complement $f^c$ 的示例。

![image](Imgaes/categorical-foundations-cute-layouts-paper/7013fe7d3b12b3129acd1049b70e42107baefe0668239d91c17cef334011b740.jpg)


命题 3.1.5.48。如果 f 是 tuple morphism，g 是 f 的 complement，则

$$
\operatorname{sort} (g) = f ^ {c}.
$$

证明。假设 f 位于 $\alpha:\langle m\rangle_*\to\langle n\rangle_*$ 之上，`sort(g)` 位于 $\beta:\langle n-m\rangle_*\to\langle n\rangle_*$ 之上，$f^c$ 位于 $\alpha^c:\langle n-m\rangle_*\to\langle n\rangle_*$ 之上。则 β 和 $\alpha^c$ 是具有相同像的递增 map，即

$$
\operatorname{Image} (\beta) = \langle n \rangle \setminus \operatorname{Image} (\alpha) = \operatorname{Image} \left(\alpha^ {c}\right),
$$

因此 $\beta=\alpha^c$，进而 `sort(g)=f^c`。

命题 3.1.5.49。假设 f 是 tuple morphism。f 存在 complement，当且仅当 f 在定义 3.1.5.45 的意义下是 complementable 的。

证明。如果 f 位于非单射 map α 之上，则对任意与 f 的像不相交的 morphism $f^*$，morphism $f\star f^*$ 位于非单射 map 之上，因此不是 isomorphism。反之，如果 f 位于单射 map 之上，则构造 3.1.5.46 的 morphism $f^c$ 是 f 的 complement。□

命题 3.1.5.50。如果 f 是 complementable tuple morphism，则

$$
\operatorname{sort} (f) = \left(f ^ {c}\right) ^ {c}.
$$

证明。两个 map 都递增、单射且具有相同的像，所以相等。

## 3.1.5.7 Flat division

本节定义 tuple morphism 上的 division 操作。

定义 3.1.5.51。如果 f 和 g 是 tuple morphism，并且 g 与 f 可复合，就称 g 整除 f。换言之，

$$
\operatorname{codomain} (g) = \operatorname{domain} (f).
$$

定义 3.1.5.52。假设 $g:S\to T$ 和 $f:T\to U$ 是 tuple morphism。f 除以 g 的 flat division 是 tuple morphism

$$
f \oslash^ {\flat} g = f \circ (g \star g ^ {c}).
$$

示例 3.1.5.53。下面给出 tuple morphism f、g 及其 flat quotient $f\oslash^\flat g$ 的示例。

$$
\begin{array}{c c c} 1 2 8 \longmapsto 1 2 8 & 1 2 8 \longmapsto 1 2 8 \\ g & 2 & f \\ \hline \end{array} \quad \begin{array}{c c c} 1 2 8 \longmapsto 1 2 8 \\ 2 \longmapsto 2 \\ \hline \end{array} \quad \begin{array}{c c c} 2 & 1 2 8 \\ 1 2 8 \longmapsto 1 2 8 \\ \hline \end{array} \quad \begin{array}{c c c} f \otimes^ {b} g \\ \hline \end{array}
$$

示例 3.1.5.54。下面给出 tuple morphism f、g 及其 flat quotient $f\oslash^\flat g$ 的示例。

![image](Imgaes/categorical-foundations-cute-layouts-paper/6039b08bd0e80764f025e6b91c9d166f2ca197703d5d0d6105688dbb6788b82b.jpg)


示例 3.1.5.55。下面给出 tuple morphism f、g 及其 flat quotient $f\oslash^\flat g$ 的示例。

![image](Imgaes/categorical-foundations-cute-layouts-paper/83b48eae2c5cdc04bde54607429487a128e20ab497181a086a7a3408e46ab53e.jpg)


命题 3.1.5.56。如果 f 和 g 是 non-degenerate 且可复合的 tuple morphism，则

$$
\operatorname{coal} ^ {\flat} \left(L _ {f \oslash^ {\flat} g}\right) = \operatorname{coal} ^ {\flat} \left(L _ {f} \oslash^ {\flat} L _ {g}\right)
$$

证明。根据命题 3.2.6.20，有

$$
\operatorname{coal} ^ {\flat} \left(L _ {g ^ {c}}\right) = \operatorname{comp} ^ {\flat} \left(L _ {g}, \operatorname{size} \left(L _ {f}\right)\right),
$$

计算得

$$
\begin{array}{l} \text {coal} ^ {\flat} (L _ {f} \oslash^ {\flat} L _ {g}) = \text {coal} ^ {\flat} (L _ {f} \circ (L _ {g} \star \text {comp} (L _ {g}, \text {size} (L _ {f})))) \\ \qquad = \text {coal} (L _ {f} \circ (L _ {g} \star L _ {g ^ {c}})) \\ \qquad = \text {coal} (L _ {f} \circ L _ {g \star g ^ {c}}) \\ \qquad = \text {coal} (L _ {f \circ (g \star g ^ {c})}) \\ \qquad = \text {coal} (L _ {f \oslash^ {\flat} g}). \end{array}
$$

## 3.1.5.8 Flat product

本节定义 tuple morphism 上的 product 操作。

定义 3.1.5.57。假设 f 和 g 是 tuple morphism。如果 `codomain(g)=domain(f^c)`，就称 f 与 g 为 product admissible。如果 f 和 g 为 product admissible，则把二者的 flat product 定义为

$$
f \otimes^ {\flat} g = f \star (f ^ {c} \circ g).
$$

示例 3.1.5.58。如果 f 和 g 是下图中的 tuple morphism，

$$
\begin{array}{c c} & 1 6 \\ & 1 6 \\ 1 6 \longmapsto 1 6 & 8 \longmapsto 8 \\ 1 6 \longmapsto 1 6 & 8 \longmapsto 8 \\ g & f \end{array}
$$

则 f 与 g 为 product admissible，而且 $f\otimes^\flat g$ 是下图中的 tuple morphism。

$$
\begin{array}{c} 1 6 \longmapsto 1 6 \\ 1 6 \longmapsto 1 6 \\ 8 \longmapsto 8 \\ 8 \longmapsto 8 \\ f \otimes^ {b} g \end{array}
$$

示例 3.1.5.59。如果 f 和 g 是下图中的 tuple morphism，

$$
\begin{array}{c} \text {   g   } \\ \text {   f   } \end{array}
$$

则 f 与 g 为 product admissible，而且 $f\otimes^\flat g$ 是下图中的 tuple morphism。

![image](Imgaes/categorical-foundations-cute-layouts-paper/c4ec3b33c0519c4ad77a7a7ba38e561c8c59789c5a50525e3762e59b51ea4772.jpg)


引理 3.1.5.60。如果 f 与 g 为 product admissible，且 g 是单射，则 $f\otimes^\flat g$ 是单射，并且

$$
(f \otimes^ {\flat} g) ^ {c} = f ^ {c} \circ g ^ {c}.
$$

证明。tuple morphism $(f\otimes^\flat g)^c$ 与 $f^c\circ g^c$ 都是单射、递增且具有相同 codomain，因此只需证明它们具有相同的像。$(f\otimes^\flat g)^c=(f\star(f^c\circ g))^c$ 的像由既不在 f 的像中、也不在 $f^c\circ g$ 的像中的条目组成。$f^c$ 的像由不在 f 的像中的条目组成，因此 composition $f^c\circ g^c$ 的像也由既不在 f 的像中、也不在 $f^c\circ g$ 的像中的条目组成。□

命题 3.1.5.61。假设 f 与 g 为 product admissible，g 与 h 也为 product admissible。则

1. $f\otimes^\flat g$ 与 h 为 product admissible；

2. f 与 $g\otimes^\flat h$ 为 product admissible；

3. $( f \otimes ^ { \flat } g ) \otimes ^ { \flat } h = f \otimes ^ { \flat } ( g \otimes ^ { \flat } h )$ 

证明。使用引理 3.1.5.37 和引理 3.1.5.60，计算得

$$
\begin{array}{l} f \otimes^ {\flat} (g \otimes^ {\flat} h) = f \star (f ^ {c} \circ (g \otimes^ {\flat} h)) \\ \qquad = f \star (f ^ {c} \circ (g \star (g ^ {c} \circ h))) \\ \qquad = f \star ((f ^ {c} \circ g) \star (f ^ {c} \circ (g ^ {c} \circ h))) \\ \qquad = f \star ((f ^ {c} \circ g) \star ((f ^ {c} \circ g ^ {c}) \circ h)) \\ \qquad = f \star (f ^ {c} \circ g) \star ((f \otimes^ {\flat} g) ^ {c} \circ h) \\ \qquad = (f \otimes^ {\flat} g) \star ((f \otimes^ {\flat} g) ^ {c} \circ h) \\ \qquad = (f \otimes^ {\flat} g) \otimes^ {\flat} h. \end{array}
$$

命题 3.1.5.62。假设 f 和 g 是 non-degenerate tuple morphism，并且 f 与 g 为 product admissible。则

$$
L _ {f \otimes^ {\flat} g} = L _ {f} \otimes^ {\flat} L _ {g}.
$$

证明。假设 $f:S\to T$ 和 $g:U\to V$ 为 product admissible，并令

$$
L _ {f} ^ {*} = \operatorname{comp} ^ {\flat} (L _ {f}, \operatorname{size} (L _ {f}) \cdot \operatorname{cosize} (L _ {g})).
$$

由于 f 是单射，而且 g 的 codomain 是 $f^c$ 的 domain，因此

$$
\operatorname{size} \left(L _ {f}\right) \cdot \operatorname{cosize} \left(L _ {g}\right) \leq \operatorname{size} (S) \cdot \operatorname{size} (V) = \operatorname{size} (T).
$$

使用该事实以及

$$
\Phi_ {\mathrm{comp} (L _ {f}, \mathrm{size} (T))} = \Phi_ {L _ {f ^ {c}}},
$$

可得

$$
\begin{array}{c} L _ {f} ^ {*} \circ L _ {g} = \mathsf {c o m p} (L _ {f}, \mathsf {s i z e} (T)) \circ L _ {g} \\ = L _ {f c} \circ L _ {g}. \end{array}
$$

$$
\begin{array}{r l} L _ {f} \otimes^ {\flat} L _ {g} & = L _ {f} \star (L _ {f} ^ {*} \circ L _ {g}) \\ & = L _ {f} \star (L _ {f ^ {c}} \circ L _ {g}) \\ & = L _ {f} \star L _ {f ^ {c} \circ g} \\ & = L _ {f \star (f ^ {c} \circ g)} \\ & = L _ {f \otimes^ {\flat} g} \end{array}
$$

使用该事实，计算得

## 3.2 Category Nest

上一节介绍了 category **Tuple**，其 morphism 编码 flat tractable layout。本节介绍 category **Nest**，其 morphism 编码具有任意嵌套的 tractable layout。

## 3.2.1 基本定义

回忆一下，对 nested tuple S，用 $S^\flat$ 表示 S 的 flattening。例如，如果 $S=(64,(8,8))$，则 $S^\flat=(64,8,8)$。

定义 3.2.1.1。用 **Nest** 表示以下 category，其 object 是正整数 nested tuple，其中 morphism

$$
f: S \to T
$$

由 tuple morphism 指定

$$
f ^ {\flat}: S ^ {\flat} \to T ^ {\flat}.
$$

换言之，

$$
\operatorname{Hom} _ {\mathbf {N e s t}} (S, T) = \operatorname{Hom} _ {\mathbf {T u p l e}} (S ^ {\flat}, T ^ {\flat}).
$$

显式地，**Nest** 中的 morphism $f:S\to T$ 由 tractable pointed map $\alpha:\langle\mathsf{len}(S)\rangle_*\to\langle\mathsf{len}(T)\rangle_*$ 指定，并满足性质：

如果 $1\leq i\leq\mathsf{len}(S)$ 且 $\alpha(i)\neq*$，则 $\mathsf{entry}_i(S)=\mathsf{entry}_{\alpha(i)}(T)$。

称这种 morphism f 位于 α 之上，并称 f 为 nested tuple morphism。

记号 3.2.1.2。如果 $f:S\to T$ 是位于 α 之上的 nested tuple morphism，把 f 描绘为

$$
S \xrightarrow [ \alpha ]{f} T
$$

示例 3.2.1.3。下面是一些 nested tuple morphism 示例。

$$
(6 4, (8, 8)) \xrightarrow [ (1 , 2 , 3) ]{f} (6 4, 8, 8)
$$

$$
((2, 2), 2) \xrightarrow [ (* , 5 , 2) ]{g} (1 0, 2, 2, (3, 2, 3))
$$

$$
6 4 \xrightarrow [ (2) ]{h} ((6 4, 6 4), 5 1 2).
$$

观察 3.2.1.4。如果 X 是集合，用 $X^{\mathrm{ind}}$ 表示 X 上的 indiscrete category。该 category 的 object 是 X 的元素，任意两个 object 之间存在唯一的 isomorphism。根据 **Nest** 的定义，有 pullback square

$$
\begin{array}{c} \text {Nest} \xrightarrow {\text {prof(-)}} \text {Profile} ^ {\text {ind}} \\ (-) ^ {b} \Biggl \downarrow \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \text {len(-)} \\ \text {Tuple} \xrightarrow {\text {len(-)}} \mathbb {N} ^ {\text {ind}} \end{array}
$$

可以把它视为 pullback square 2.2.2.4 的 categorification。

示例 3.2.1.5。假设 S 是 length 为 m 的 nested tuple。如果 $1\leq i\leq m$，则存在 nested tuple morphism

$$
\operatorname{entry} _ {i} (S) \to S
$$

它位于 map $\langle1\rangle_*\to\langle m\rangle_*$ 之上，该 map 定义为 $1\mapsto i$。例如，如果 $S=(64,(8,8))$ 且 $i=1$，则存在 nested tuple morphism

$$
6 4 \xrightarrow [ (1) ]{} (6 4, (8, 8)).
$$

示例 3.2.1.6。假设 S 是 rank r 的 nested tuple。如果 $1\leq i\leq r$，则存在 canonical nested tuple morphism

$$
\operatorname{mode} _ {i} (S) \to S
$$

它位于 map $\langle\mathsf{len}_i(S)\rangle_*\to\langle\mathsf{len}(S)\rangle_*$ 之上，该 map 定义为 $j\mapsto j+\mathsf{len}_{<i}(S)$。例如，如果 $S=(64,(8,8))$，则存在 nested tuple morphism

$$
(8, 8) \xrightarrow [ (2 , 3) ]{} (6 4, (8, 8)).
$$

观察 3.2.1.7。存在联系 category **Nest** 与 **Tuple** 的 functor。首先有 inclusion functor

$$
\text { Tuple } \xrightarrow {\subset} \text { Nest }
$$

它把 tuple morphism $f:S\to T$ 视为 nested tuple morphism。其次有 flattening functor

$$
\text { Nest } \xrightarrow {(-) ^ {b}} \text { Tuple }
$$

它把 nested tuple morphism $f:S\to T$ 映射到底层 tuple morphism $f^\flat:S^\flat\to T^\flat$。composite

$$
\text { Tuple } \xrightarrow {\subset} \text { Nest } \xrightarrow {(-) ^ {b}} \text { Tuple }
$$

是 **Tuple** 上的 identity functor，所以 **Tuple** 是 **Nest** 的 retractive subcategory。此外，这些 functor 构成 category 的 adjoint equivalence。

注记 3.2.1.8。也可以考虑某个 morphism 编码 tractable layout、但不等价于 **Tuple** 的 category C。作者考虑过若干这类示例，但留待未来研究。

## 3.2.2 从 nested tuple morphism 到 layout

category **Nest** 的关键特性是：如果 $f:S\to T$ 是 nested tuple morphism，则 f 编码一个 layout $L_f$。该 layout 通过给 flat layout $L_{f^\flat}$ 赋予 S 的嵌套 profile 得到。更精确地，有以下构造。

构造 3.2.2.1。假设

$$
f: S \to T
$$

是 nested tuple morphism，并假设 $P=\mathsf{prof}(S)$。把 $L_f$ 定义为 layout

$$
L _ {f} = (L _ {f ^ {\flat}}) _ {P}
$$

其中 $(-)_P$ 是定义 2.3.1.19 的 P-substitution 操作。称 $L_f$ 为 f 编码的 layout。

构造 3.2.2.2。假设

$$
(s _ {1}, \ldots , s _ {m}) _ {P} \xrightarrow [ \alpha ]{f} (t _ {1}, \ldots , t _ {n}) _ {Q}
$$

是 nested tuple morphism。把 $L_f$ 定义为以下 layout：其 shape

$$
\mathsf {s h a p e} (L _ {f}) = (s _ {1}, \dots , s _ {m}) _ {P}
$$

是 f 的 domain，其 stride

$$
\mathsf {s t r i d e} (L _ {f}) = (d _ {1}, \dots , d _ {m}) _ {P}
$$

的 entry 由公式定义

$$
d _ {i} = \left\{ \begin{array}{l l} 0 & \alpha (i) = * \\ \prod_ {j <   \alpha (i)} t _ {j} & \alpha (i) \neq *. \end{array} \right.
$$

称 $L_f$ 为 f 编码的 layout。

示例 3.2.2.3。以下 morphism 编码的 layout

$$
((8, 8), (4, 4)) \xrightarrow [ (1 , 4 , 3 , 2) ]{f} (8, 4, 4, 8)
$$

是

$$
L _ {f} = ((8, 8), (4, 4)): ((1, 1 2 8), (3 2, 8)).
$$

示例 3.2.2.4。以下 morphism 编码的 layout

$$
(1 2 8, (4, 4, 2)) \xrightarrow [ (3 , 1 , 2 , *) ]{g} ((4, 4), 1 2 8)
$$

是

$$
L _ {g} = (1 2 8, (4, 4, 2)): (1 6, (1, 4, 0)).
$$

观察 3.2.2.5。flattening functor

$$
\text { Nest } \xrightarrow {(-) ^ {b}} \text { Tuple }
$$

与 layout flattening 相容：如果 f 是 nested tuple morphism，则

$$
(L _ {f}) ^ {\flat} = L _ {f ^ {\flat}}.
$$

如果 L 是 tractable layout，可以按如下方式构造编码 L 的 nested tuple morphism。

构造 3.2.2.6。假设 L 是 tractable layout。把 L 的 standard representation 定义为 nested tuple morphism

$$
f _ {L}: S \to T
$$

其中 $(f_L)^\flat=f_{L^\flat}$ 是 $L^\flat$ 的 standard representation，$S=\mathsf{shape}(L)$ 是 L 的 shape，T 是 $f_{L^\flat}$ 的 codomain。

示例 3.2.2.7。如果

$$
L = (3 2, (2, 2)): (1 9 2, (2 4, 3))
$$

则 L 的 standard representation 为

$$
(3 2, (2, 2)) \xrightarrow [ (6 , 4 , 2) ]{f _ {L}} (3, 2, 4, 2, 4, 3 2).
$$

引理 3.2.2.8。如果 L 是 tractable layout，$f=f_L$ 是 L 的 standard representation，则

$$
L _ {f} = L.
$$

证明。有

$$
(L _ {f}) ^ {\flat} = L _ {f ^ {\flat}} = L ^ {\flat}
$$

and 

$$
\operatorname{shape} (L _ {f}) = \operatorname{shape} (L).
$$

命题 3.2.2.9。假设 L 是 layout。存在编码 L 的 nested tuple morphism f，当且仅当 L 是 tractable 的。

证明。先假设对某个 nested tuple morphism f 有 $L=L_f$。则 $(L_f)^\flat=L_{f^\flat}$；根据命题 3.1.2.10，$L^\flat$ 是 tractable 的，因此 L 也是。反之，如果 L 是 tractable 的，可以取 $f=f_L$ 为 L 的 standard representation；根据引理 3.2.2.8，有 $L_f=L$。□

为了在 tractable layout 与某些 nested tuple morphism 之间建立一一对应关系，引入 nested tuple morphism 的 standard form 概念。

定义 3.2.2.10。假设 $f:S\to T$ 是 nested tuple morphism。如果以下条件成立，就称 f 具有 standard form：

1. $f^\flat$ 在定义 3.1.2.12 的意义下具有 standard form；

2. T 是 flat 的。

示例 3.2.2.11。nested tuple morphism

$$
((2, 2), (3, 3)) \xrightarrow [ (4 , 6 , 2 , 3 ]{f} (1 0, 3, 3, 2, 1 0, 2)
$$

具有 standard form。

示例 3.2.2.12。nested tuple morphism

$$
((2, 2), (3, 3)) \xrightarrow [ (4 , 6 , 2 , 3 ]{f} ((1 0, 3, 3), (2, 1 0, 2))
$$

不具有 standard form，因为 g 的 codomain 不是 flat 的。

与 flat 情况相同，为了在具有 standard form 的 nested tuple morphism 与 tractable layout 之间获得一一对应关系，需要限制到 non-degenerate nested tuple morphism 和 non-degenerate layout。为此，作如下定义。

定义 3.2.2.13。假设

$$
S \xrightarrow [ \alpha ]{f} T
$$

是 nested tuple morphism，并假设

$$
L = S: D
$$

是 layout。

1. 如果下式成立，就称 f 是 non-degenerate：

$$
\operatorname{entry} _ {i} (S) = 1 \quad \Rightarrow \quad \alpha (i) = *.
$$

2. 如果

$$
\operatorname{entry} _ {i} (S) = 1 \quad \Rightarrow \quad \operatorname{entry} _ {i} (D) = 0.
$$

就称 L 是 non-degenerate。注记 3.2.2.14。如果 f 是 nested tuple morphism，则 f 是 non-degenerate 当且仅当 $f^\flat$ 是 non-degenerate。如果 L 是 layout，则 L 是 non-degenerate 当且仅当 $L^\flat$ 是 non-degenerate。

命题 3.2.2.15。构造 3.2.2.2 和 3.2.2.6 中的映射

![image](Imgaes/categorical-foundations-cute-layouts-paper/a17f588a85286fcaa472b07cbf359622cddc4aa1ca3895b34532dcc3def61e64.jpg)


$$
\left\{ \begin{array}{c} N o n - d e g e n e r a t e \\ n e s t e d t u p l e m o r p h i s m s \\ o f s t a n d a r d f o r m \end{array} \right\} \longleftrightarrow \left\{ \begin{array}{c} N o n - d e g e n e r a t e \\ t r a c t a b l e l a y o u t s \end{array} \right\}
$$

![image](Imgaes/categorical-foundations-cute-layouts-paper/3542866dadbdea60b846dca05941d9207df40ba143754fcb6995a1d8cefb5891.jpg)


在具有 standard form 的 nested tuple morphism 与 tractable layout 之间建立一一对应关系。

证明。命题 3.2.2.9 已经证明，如果 L 是 tractable layout，且 $f=f_L$ 是 L 的 standard representation，则 $L_f=L$。接下来假设 f 具有 standard form，并令 $L=L_f$ 为 f 编码的 layout。要证明 f 等于 L 的 standard representation $f_L$。根据命题 3.1.2.21，$f^\flat$ 等于 $L^\flat$ 的 standard representation $f_{L^\flat}$；又因为

$$
\operatorname{domain} (f) = \operatorname{shape} (L) = \operatorname{domain} \left(f _ {L}\right),
$$

and 

$$
\operatorname{codomain} (f) = \operatorname{codomain} \left(f ^ {\flat}\right) = \operatorname{codomain} \left(f _ {L ^ {\flat}}\right) = \operatorname{codomain} \left(f _ {L}\right),
$$

所以 $f=f_L$。

## 3.2.3 示例

本节列出 nested tuple morphism 的一些重要 family。

示例 3.2.3.1（Reparenthesization）。假设 $S_1$ 和 $S_2$ 是具有相同 flattening 的 nested tuple：

$$
S _ {1} ^ {\flat} = S _ {2} ^ {\flat}.
$$

则存在 reparenthesization isomorphism

$$
\mathsf {i d} _ {S _ {1}} ^ {S _ {2}}: S _ {1} \xrightarrow {\cong} S _ {2}
$$

它位于 identity 之上。这些 morphism 具有传递性，即

$$
\mathsf {i d} _ {S _ {2}} ^ {S _ {3}} \circ \mathsf {i d} _ {S _ {1}} ^ {S _ {2}} = \mathsf {i d} _ {S _ {1}} ^ {S _ {3}},
$$

并与 identity 相容，即

$$
\mathrm{id} _ {S} ^ {S} = \mathrm{id} _ {S}.
$$

如果 $f=\mathrm{id}_{S_1}^{S_2}$ 是 reparenthesization isomorphism，则 $L_f$ 是 shape 为 $S_1$ 的 column-major layout。

示例 3.2.3.2（Flattening）。作为前一示例的特殊情况，如果 S 是任意 nested tuple，则存在 flattening isomorphism

$$
\mathrm{id} _ {S} ^ {S ^ {\flat}}: S \xrightarrow {\cong} S ^ {\flat}
$$

以及 unflattening isomorphism

$$
\mathrm{id} _ {S ^ {\flat}} ^ {S}: S ^ {\flat} \xrightarrow {\cong} S
$$

观察 3.2.3.3。如果 $f:S\to T$ 是 nested tuple morphism，则 f 等于 composite

$$
S \xrightarrow {\mathrm{id} _ {S} ^ {S ^ {b}}} S ^ {b} \xrightarrow {f ^ {b}} T ^ {b} \xrightarrow {\mathrm{id} _ {T ^ {b}} ^ {T}} T.
$$

换言之，存在 canonical factorization

$$
f = \mathsf {i d} _ {T ^ {\flat}} ^ {T} \circ f ^ {\flat} \circ \mathsf {i d} _ {S} ^ {S ^ {\flat}}.
$$

示例 3.2.3.4（Entry）。假设

$$
S \xrightarrow [ \alpha ]{f} T
$$

是 nested tuple morphism。假设 $1\leq i\leq\mathsf{len}(S)$，并记 $j=\alpha(i)$。称 nested tuple morphism

$$
\operatorname{entry} _ {i} (S) \xrightarrow [ (j) ]{\operatorname{entry} _ {i} (f)} T
$$

为 f 的第 i 个 entry。$\mathsf{entry}_i(f)$ 编码的 layout 为

$$
L _ {\text { entry } _ {i} (f)} = \text { entry } _ {i} (L _ {f}).
$$

示例 3.2.3.5（Entry inclusion）。作为前一示例的特殊情况，如果 S 是 nested tuple，且 $1\leq i\leq\mathsf{len}(S)$，可以取 $f=\operatorname{id}_S$；此时

$$
\operatorname{entry} _ {i} \left(\operatorname{id} _ {S}\right): \operatorname{entry} _ {i} (S) \longrightarrow S
$$

是 S 的第 i 个 entry 的 inclusion。

示例 3.2.3.6（Mode）。假设

$$
S \xrightarrow [ \alpha ]{f} T
$$

是 nested tuple morphism。假设 $1\leq i\leq\mathsf{rank}(S)$，并记

$$
\begin{array}{c} N = \mathsf {l e n} _ {<   i} (S) \\ \ell = \mathsf {l e n} _ {i} (S). \end{array}
$$

称 nested tuple morphism

$$
\operatorname{mode} _ {i} (S) \xrightarrow [ (N + 1 , \dots , N + \ell) ]{\operatorname{mode} _ {i} (f)} T
$$

为 S 的第 i 个 mode。$\mathsf{mode}_i(f)$ 编码的 layout 为

$$
L _ {\text { mode } _ {i} (f)} = \text { mode } _ {i} (L _ {f}).
$$

示例 3.2.3.7（Mode inclusion）。作为前一示例的特殊情况，可以取 $f=\operatorname{id}_S$；此时

$$
\operatorname{mode} _ {i} (\operatorname{id} _ {S}): \operatorname{mode} _ {i} (S) \to S
$$

是 S 的第 i 个 mode 的 inclusion。有时把该 map 记为

$$
\operatorname{incl} _ {i} (S) = \operatorname{mode} _ {i} (\operatorname{id} _ {S}).
$$

## 3.2.4 Nested tuple morphism 的 realization

在 flat 情况下，我们构造了 realization functor

$$
\text { Tuple } \xrightarrow {| \cdot |} \text { FinSet }
$$

它把 tuple morphism f 映射到 $L_f$ 的 layout function。可以在之前复合 flattening functor `Nest → Tuple`，把它扩展为 realization functor

$$
\text { Nest } \xrightarrow {| \cdot |} \text { FinSet }
$$

。

定义 3.2.4.1。把 realization functor

$$
\text { Nest } \xrightarrow {| \cdot |} \text { FinSet }
$$

定义为 composite

$$
\mathbf {N e s t} \xrightarrow {(-) ^ {b}} \mathbf {T u p l e} \xrightarrow {| \cdot |} \mathbf {F i n S e t}
$$

引理 3.2.4.2。如果 $f:S\to T$ 是 nested tuple morphism，则 f 的 realization $|f|$ 是 $L_f$ 的 layout function。

$$
| f | = \Phi_ {L _ {f}} ^ {\mathrm{size} (T)}.
$$

证明。由 3.1.4.5 立即可得，因为

$$
| f | = | f ^ {\flat} | = \Phi_ {L _ {f ^ {\flat}}} ^ {\text { size } (T)} = \Phi_ {L _ {f}} ^ {\text { size } (T)}
$$

## 3.2.5 Refinement

本节从范畴论角度重新考察 nested tuple 的 refinement。回忆第 2.2.4 节：nested tuple $S'$ refine S，记作

$$
S ^ {\prime} \longrightarrow S
$$

如果可以把 S 的每个 entry 替换为相同 size 的某个 nested tuple，从而得到 $S'$。例如，

$$
(2, (2, 2)) \twoheadrightarrow 8,
$$

and 

$$
((2, 2), (3, 3), (5, 5)) \twoheadrightarrow (4, 9, 2 5).
$$

如果 `len(S)=m` 且 `prof(S)=P`，则可以把

$$
S ^ {\prime} = (S _ {1} ^ {\prime}, \ldots , S _ {m} ^ {\prime}) _ {P}
$$

写成 relative mode 的 P-substitution

$$
S _ {i} ^ {\prime} = \operatorname{mode} _ {i} (S ^ {\prime}, S).
$$

称普通 concatenation

$$
(S _ {1} ^ {\prime}, \ldots , S _ {m} ^ {\prime}) = \mathsf {f l a t} (S ^ {\prime}, S)
$$

为 $S'$ relative to S 的 flattening。

用 **Ref** 表示正整数 nested tuple 在 refinement 下构成的 poset category，因此 **Ref** 中的 morphism 是 refinement $S'\to S$。如果 S 是 nested tuple，令

$$
\mathbf {R e f} (S) = \{S ^ {\prime} \mid S ^ {\prime} \text {   refines   } S \}
$$

表示 refine S 的 nested tuple 构成的 poset。等价地，Ref(S) 是 slice category $\mathsf{Ref}_{/S}$。

构造 3.2.5.1（Relative mode inclusion）。假设 $S'\to S$ 是 refinement，并写成

$$
S _ {i} ^ {\prime} = \operatorname{mode} _ {i} (S ^ {\prime}, S)
$$

表示 $S'$ relative to S 的 mode。则 $S'$ 与 $(S_1',\ldots,S_m')$ 具有相同 flattening，因此存在 reparenthesization isomorphism

$$
\mathsf {i d} _ {(S _ {1} ^ {\prime}, \dots , S _ {m} ^ {\prime})} ^ {S ^ {\prime}}: (S _ {1} ^ {\prime}, \dots , S _ {m} ^ {\prime}) \xrightarrow {\cong} S ^ {\prime}
$$

并定义

$$
\operatorname{incl} _ {i} (S ^ {\prime}, S): S _ {i} ^ {\prime} \to S ^ {\prime}
$$

为 composite

$$
S _ {i} ^ {\prime} \xrightarrow {\operatorname{incl} _ {i} ((S _ {1} ^ {\prime} , \ldots , S _ {m} ^ {\prime}))} (S _ {1} ^ {\prime}, \ldots , S _ {m} ^ {\prime}) \xrightarrow {\operatorname{id} _ {(S _ {1} ^ {\prime} , \ldots , S _ {m} ^ {\prime})} ^ {S ^ {\prime}}} S ^ {\prime}
$$

即 $(S_1',\ldots,S_m')$ 的第 i 个 mode inclusion 与 reparenthesization isomorphism $(S_1',\ldots,S_m')\cong S'$ 的 composite。示例 3.2.5.2。如果 $S=(4,(9,25))$，$S'=((2,2),((3,3),25))$，则 $S'$ refine S，而且 $\mathsf{incl}_2(S',S)$ 是 nested tuple morphism

$$
(3, 3) \xrightarrow [ (3 , 4) ]{\operatorname{incl} _ {2} (S ^ {\prime} , S)} ((2, 2), ((3, 3), 2 5)).
$$

构造 3.2.5.3（Relative mode）。假设 $f':S'\to T'$ 是 nested tuple morphism，且 $S'$ refine S。把 $f'$ relative to S 的第 i 个 mode 定义为

$$
\operatorname{mode} _ {i} \left(f ^ {\prime}, S\right) = f ^ {\prime} \circ \operatorname{incl} _ {i} \left(S ^ {\prime}, S\right): S _ {i} ^ {\prime} \rightarrow T ^ {\prime}
$$

即 composite

$$
S _ {i} ^ {\prime} \xrightarrow {\operatorname{incl} _ {i} (S ^ {\prime} , S)} S ^ {\prime} \xrightarrow {f ^ {\prime}} T ^ {\prime}
$$

特别地，

$$
\operatorname{mode} _ {i} \left(\operatorname{id} _ {S ^ {\prime}}, S\right) = \operatorname{incl} _ {i} \left(S ^ {\prime}, S\right).
$$

示例 3.2.5.4。假设 $S=(4,(9,25))$，$S'=((2,2),((3,3),25))$，因此 $S'$ refine S。如果 $f'$ 是 nested tuple morphism

$$
((2, 2), ((3, 3), 2 5)) \xrightarrow [ (1 , 3 , 2 , * , 4) ]{f ^ {\prime}} (2, 3, 2, 2 5).
$$

则 $\mathsf{mode}_2(f',S)$ 是 nested tuple morphism

$$
(3, 3) \xrightarrow [ (2 , *) ]{\operatorname{mode} _ {2} \left(f ^ {\prime} , S\right)} (2, 3, 2, 2 5).
$$

构造 3.2.5.5（Pullback）。假设 $f:S\to T$ 是位于 α 之上的 nested tuple morphism，$T'\to T$ 是 refinement。令

$$
T _ {j} ^ {\prime} = \operatorname{mode} _ {j} (T ^ {\prime}, T)
$$

表示 $T'$ relative to T 的第 j 个 mode，并对任意 $1\leq i\leq\mathsf{len}(S)$，令

$$
S _ {i} ^ {\prime} = \left\{ \begin{array}{l l} \text {entry} _ {i} (S) & \alpha (i) = * \\ T _ {j} ^ {\prime} & \alpha (i) = j. \end{array} \right.
$$

把 $T'$ 沿 f 的 pullback 定义为 nested tuple

$$
S ^ {\prime} = f ^ {*} T ^ {\prime} = \operatorname{sub} (S, (S _ {1} ^ {\prime}, \dots , S _ {m} ^ {\prime})).
$$

对任意 $1\leq i\leq m$，令

$$
f _ {i} ^ {\prime}: S _ {i} ^ {\prime} \to T ^ {\prime}
$$

在 $\alpha(i)=*$ 时为 trivial map，在 $\alpha(i)=j$ 时为 inclusion

$$
\operatorname{incl} _ {j} (T ^ {\prime}, T): S _ {i} ^ {\prime} = T _ {j} ^ {\prime} \to T ^ {\prime}
$$

$f_1',\ldots,f_m'$ 的像不相交，因此形成 concatenation

$$
(f _ {1} ^ {\prime}, \ldots , f _ {m} ^ {\prime}): (S _ {1} ^ {\prime}, \ldots , S _ {m} ^ {\prime}) \to T ^ {\prime}.
$$

把 $f'=T^{\prime *}f$ 定义为 composite

$$
S ^ {\prime} \xrightarrow {\mathsf {i d} _ {S ^ {\prime}} ^ {(S _ {1} ^ {\prime} , \ldots , S _ {m} ^ {\prime})}} (S _ {1} ^ {\prime}, \ldots , S _ {m} ^ {\prime}) \xrightarrow {(f _ {1} ^ {\prime} , \ldots , f _ {m} ^ {\prime})} T ^ {\prime}.
$$

称 $f'$ 为 f along $T'$ 的 pullback，并把这类 pullback 描绘为方块

$$
\begin{array}{c c c} S ^ {\prime} & \xrightarrow {f ^ {\prime}} & T ^ {\prime} \\ \Big \downarrow & \searrow & \Big \downarrow \\ S & \xrightarrow {f} & T. \end{array}
$$

示例 3.2.5.6。假设 $f:(64,32)\to(4,64,4,32)$ 位于 $\alpha=(2,4)$ 之上。则有 pullback square

$$
\begin{array}{c} ((1 6, 4), (1 6, 2)) \xrightarrow {f ^ {\prime}} ((2, 2), (1 6, 4), (2, 2), (1 6, 2)) \\ \Big \downarrow \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \Big \downarrow \\ (6 4, 3 2) \xrightarrow {f} (4, 6 4, 4, 3 2) \end{array}
$$

其中 $f'$ 位于 $\alpha'=(3,4,7,8)$ 之上。

示例 3.2.5.7。假设 S 是具有以下 flattening 的 nested tuple：

$$
S ^ {\flat} = (s _ {1}, \dots , s _ {m}),
$$

并假设 $S'\to S$ 是具有 relative flattening

$$
(S _ {1} ^ {\prime}, \ldots , S _ {m} ^ {\prime}).
$$

的 refinement。则 $S'\to S$ 沿 unflattening isomorphism

$$
\mathsf {i d} _ {(s _ {1}, \dots , s _ {m})} ^ {S}: (s _ {1}, \dots , s _ {m}) \to S
$$

的 pullback 是 reparenthesization isomorphism

$$
\begin{array}{c} (S _ {1} ^ {\prime}, \ldots , S _ {m} ^ {\prime}) \xrightarrow {\mathsf {i d} _ {(S _ {1} ^ {\prime} , \ldots , S _ {m} ^ {\prime})} ^ {S ^ {\prime}}} S ^ {\prime} \\ \Big \downarrow \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \Big \downarrow \\ (s _ {1}, \ldots , s _ {m}) \xrightarrow {\mathsf {i d} _ {(s _ {1} , \ldots , s _ {m})} ^ {S}} S. \end{array}
$$

示例 3.2.5.8。假设 $S'\to S$ 是 refinement，并考虑第 i 个 entry inclusion

$$
s _ {i} \rightarrow S.
$$

则 $S'\to S$ 沿 $s_i\to S$ 的 pullback 是第 i 个 relative mode inclusion

$$
\begin{array}{c} S _ {i} ^ {\prime} \xrightarrow {\text {incl} _ {i} (S ^ {\prime} , S)} S ^ {\prime} \\ \Big \downarrow \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \end{array}
$$

观察 3.2.5.9。上述 pullback 构造规定了 contravariant functor

$$
\mathbf {N e s t} ^ {\mathrm{op}} \longrightarrow \mathbf {C a t}
$$

$$
\begin{array}{c} S \longmapsto \mathbf {R e f} (S) \\ f \Big \downarrow \qquad \qquad \qquad \qquad \qquad \qquad \qquad \uparrow_ {f ^ {*}} \\ T \longmapsto \mathbf {R e f} (T) \end{array}
$$

$$
\begin{array}{c} f ^ {*} T ^ {\prime} \leftarrow f ^ {*} T ^ {\prime \prime} \\ \Big \uparrow \\ T ^ {\prime} \leftarrow T ^ {\prime \prime} \end{array}
$$

pullback 的关键性质是，$f'$ 的 layout function 等于 f 的 layout function。

引理 3.2.5.10。假设

$$
\begin{array}{c c c} S ^ {\prime} & \xrightarrow {f ^ {\prime}} & T ^ {\prime} \\ \Big \downarrow & \searrow & \Big \downarrow \\ S & \xrightarrow {f} & T \end{array}
$$

是 pullback square，其中 f 位于 α 之上。令

$$
f _ {i} ^ {\prime}: S _ {i} ^ {\prime} \to T
$$

表示 f′ relative to S 的第 i 个 mode，并令

$$
(L _ {f}) ^ {\flat} = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m}).
$$

则对任意 $1\leq i\leq m$，有

$$
\operatorname{coal} (L _ {f _ {i} ^ {\prime}}) = s _ {i}: d _ {i}.
$$

证明。假设 $1\leq i\leq m$。如果 $\alpha(i)=*$，则 $f_i'$ 是 trivial map，所以

$$
L _ {f _ {i} ^ {\prime}} = s _ {i}: 0 = s _ {i}: d _ {i}.
$$

特别地，$\mathsf{coal}(L_{f_i'})=s_i:0=s_i:d_i$。接下来假设 $\alpha(i)=j\neq*$。根据 $f'$ 的构造，

$$
f _ {i} ^ {\prime} = \operatorname{incl} _ {j} (T ^ {\prime}, T): T _ {j} ^ {\prime} \to T ^ {\prime}.
$$

它位于 map $\alpha_i'$ 之上，该 map 定义为 $t\mapsto\mathsf{len}_{<j}(T',T)+t$。对每个 $1\leq t<\mathsf{len}(T_i')$，有 $\alpha_i'(t+1)=\alpha_i'(t)+1$，所以 $L_{f_i'}$ 是 size 为 $\mathsf{size}(T_j')=t_j=s_i$ 的 column-major layout。这意味着 $\mathsf{coal}(L_{f_i'})$ 是形如

$$
\operatorname{coal} (L _ {f _ {i} ^ {\prime}}) = s _ {i}: e
$$

的 depth 0 layout，其中某个整数 $e\geq0$。我们声称 $e=d_i$。如果记 $t_{j'}'=\mathsf{entry}_{j'}(T')$，则

$$
\begin{array}{l} e = \mathsf {e n t r y} _ {1} (\mathsf {s t r i d e} (L _ {f _ {i} ^ {\prime}})) = \prod_ {j ^ {\prime} <   \alpha_ {i} ^ {\prime} (1)} t _ {j ^ {\prime}} ^ {\prime} \\ \qquad = \prod_ {j ^ {\prime} \leq \mathsf {l e n} _ {<   j} (T ^ {\prime}, T)} t _ {j ^ {\prime}} ^ {\prime} \\ \qquad = \prod_ {j ^ {\prime} <   j} \mathsf {s i z e} (T _ {j ^ {\prime}} ^ {\prime}) \\ \qquad = \prod_ {j ^ {\prime} <   j} t _ {j ^ {\prime}} \\ \qquad = d _ {i}. \end{array}
$$

命题 3.2.5.11。如果

$$
\begin{array}{c c c} S ^ {\prime} & \xrightarrow {f ^ {\prime}} & T ^ {\prime} \\ \Big \downarrow & \searrow & \Big \downarrow \\ S & \xrightarrow {f} & T \end{array}
$$

是 pullback square，则 $\Phi_{L_f}=\Phi_{L_{f'}}$。

证明。先固定记号。令 $m=\mathsf{len}(S)$，并令

$$
\begin{array}{c} S ^ {\flat} = (s _ {1}, \ldots , s _ {m}), \\ S _ {i} ^ {\prime} = \mathsf {m o d e} _ {i} (S ^ {\prime}, S), \\ T _ {j} ^ {\prime} = \mathsf {m o d e} _ {j} (T ^ {\prime}, T), \\ (L _ {f}) ^ {\flat} = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m}). \end{array}
$$

考虑 reparenthesization isomorphism

$$
\mathsf {i d} _ {(S _ {1} ^ {\prime}, \dots , S _ {m} ^ {\prime})} ^ {S ^ {\prime}}: (S _ {1} ^ {\prime}, \dots , S _ {m} ^ {\prime}) \to S ^ {\prime}
$$

该 map 与 $f'$ 的 composite 是 concatenation $(f_1',\ldots,f_m')$，其中 $f_i'$ 在 $\alpha(i)=*$ 时为 trivial map，否则为 relative mode inclusion

$$
\operatorname{incl} _ {i} (T ^ {\prime}, T): S _ {i} ^ {\prime} = T _ {j} ^ {\prime} \to T ^ {\prime}
$$

。使用引理 3.2.5.10 以及 $L_{f'}=L_{(f_1',\dots,f_m')}$，计算得

$$
\begin{array}{l} \mathsf {c o a l} (L _ {f ^ {\prime}}) = \mathsf {c o a l} (L _ {(f _ {1} ^ {\prime}, \ldots , f _ {m} ^ {\prime})}) \\ \qquad = \mathsf {c o a l} ((L _ {f _ {1} ^ {\prime}}, \ldots , L _ {f _ {m} ^ {\prime}})) \\ \qquad = \mathsf {c o a l} ((\mathsf {c o a l} (L _ {f _ {1} ^ {\prime}}), \ldots , \mathsf {c o a l} (L _ {f _ {m} ^ {\prime}}))) \\ \qquad = \mathsf {c o a l} ((s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m})) \\ \qquad = \mathsf {c o a l} (L _ {f}). \end{array}
$$

根据命题 2.3.3.14，得到 $\Phi_{L_{f'}}=\Phi_{L_f}$。

构造 3.2.5.12（Pushforward）。假设 $f:S\to T$ 是位于 α 之上的 nested tuple morphism，$S'\to S$ 是 refinement。令

$$
S _ {i} ^ {\prime} = \operatorname{mode} _ {i} (S ^ {\prime}, S)
$$

表示 $S'$ relative to S 的第 i 个 mode，并对任意 $1\leq j\leq\mathsf{len}(T)$，令

$$
T _ {j} ^ {\prime} = \left\{ \begin{array}{l l} \mathsf {e n t r y} _ {j} (T) & j \notin \mathsf {I m a g e} (\alpha) \\ S _ {i} ^ {\prime} & \alpha (i) = j. \end{array} \right.
$$

把 $S'$ 沿 f 的 pushforward 定义为 nested tuple

$$
T ^ {\prime} = f _ {*} S ^ {\prime} = \operatorname{sub} (T, (T _ {1} ^ {\prime}, \dots , T _ {n} ^ {\prime})).
$$

对任意 $1\leq i\leq m$，令

$$
f _ {i} ^ {\prime}: S _ {i} ^ {\prime} \to T ^ {\prime}
$$

在 $\alpha(i)=*$ 时为 trivial map，在 $\alpha(i)=j$ 时为 relative mode inclusion

$$
\operatorname{incl} _ {j} \left(T ^ {\prime}, T\right): S _ {i} ^ {\prime} = T _ {j} ^ {\prime} \rightarrow T ^ {\prime}
$$

$f_1',\ldots,f_m'$ 的像不相交，因此可以形成 concatenation

$$
(f _ {1} ^ {\prime}, \ldots , f _ {m} ^ {\prime}): (S _ {1} ^ {\prime}, \ldots , S _ {m} ^ {\prime}) \to T ^ {\prime}.
$$

把 $f'=S_*'f$ 定义为 composite

$$
S ^ {\prime} \xrightarrow {\mathsf {i d} _ {S ^ {\prime}} ^ {(S _ {1} ^ {\prime} , \ldots , S _ {m} ^ {\prime})}} (S _ {1} ^ {\prime}, \ldots , S _ {m} ^ {\prime}) \xrightarrow {(f _ {1} ^ {\prime} , \ldots , f _ {m} ^ {\prime})} T ^ {\prime}.
$$

称 $f'$ 为 f along $S'$ 的 pushforward，并把这类 pushforward 描绘为

$$
\begin{array}{c c c} S ^ {\prime} & \xrightarrow {f ^ {\prime}} & T ^ {\prime} \\ \Big \downarrow & & \Big \downarrow \\ S & \xrightarrow {f} & T \end{array}
$$

示例 3.2.5.13。如果 $f:(64,32)\to(4,64,4,32)$ 位于 $\alpha=(2,4)$ 之上，则有 pushforward square

$$
\begin{array}{c} ((1 6, 4), (1 6, 2)) \xrightarrow {f ^ {\prime}} (4, (1 6, 4), 4, (1 6, 2)) \\ \Big \downarrow \\ (6 4, 3 2) \xrightarrow [ f ]{} (4, 6 4, 4, 3 2) \end{array}
$$

pushforward 的关键性质是，$f'$ 的 layout function 等于 f 的 layout function。

引理 3.2.5.14。假设

$$
\begin{array}{c} S ^ {\prime} \xrightarrow {f ^ {\prime}} T ^ {\prime} \\ \Big \downarrow \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \Big \downarrow \\ S \xrightarrow {f} T \end{array}
$$

是 pushforward square，其中 f 位于 α 之上。令

$$
f _ {i} ^ {\prime}: S _ {i} ^ {\prime} \to T
$$

表示 $f'$ relative to S 的第 i 个 mode，并令

$$
(L _ {f}) ^ {\flat} = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m}).
$$

则对任意 $1\leq i\leq m$，有

$$
\operatorname{coal} (L _ {f _ {i} ^ {\prime}}) = s _ {i}: d _ {i}.
$$

证明。与引理 3.2.5.10 的证明完全相同。

命题 3.2.5.15。如果

$$
\begin{array}{c} S ^ {\prime} \xrightarrow {f ^ {\prime}} T ^ {\prime} \\ \Big \downarrow \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \Big \downarrow \\ S \xrightarrow {f} T \end{array}
$$

是 pushforward square，则 $\Phi_{L_f}=\Phi_{L_{f'}}$。

证明。与命题 3.2.5.11 的证明完全相同。

观察 3.2.5.16。上述 pushforward 构造规定了 covariant functor

Nest Cat 

$$
\begin{array}{c c} S \longmapsto \mathbf {R e f} (S) & \qquad S ^ {\prime \prime} \to S ^ {\prime} \\ \Big \downarrow_ {f} \qquad \qquad \Big \downarrow_ {f _ {*}} \\ T \longmapsto \mathbf {R e f} (T) & \qquad f _ {*} S ^ {\prime \prime} \to f _ {*} S ^ {\prime} \end{array}
$$

观察 3.2.5.17。如果 $f:S\to T$ 是 nested tuple 的 isomorphism，则

$$
\operatorname{Ref} (T) \xrightarrow {f ^ {*}} \operatorname{Ref} (S)
$$

and 

$$
\mathbf {R e f} (S) \xrightarrow {f _ {*}} \mathbf {R e f} (T)
$$

是 category 的互逆 isomorphism。具体而言，

$$
(f ^ {- 1}) ^ {*} = f _ {*} \quad \text { and } \quad (f ^ {- 1}) _ {*} = f ^ {*}.
$$

观察 3.2.5.18。如果 $S_1$ 和 $S_2$ 是满足 $\mathsf{flat}(S_1)=\mathsf{flat}(S_2)$ 的 nested tuple，则存在 canonical nested tuple isomorphism $S_1\cong S_2$，因此存在 canonical category isomorphism

$$
\mathbf {R e f} (S _ {1}) \cong \mathbf {R e f} (S _ {2}).
$$

还需要定义一个概念，称为 mutual refinement。第 4 章在 layout composition 算法中使用该概念时，它的重要性会变得清楚。

定义 3.2.5.19。假设 T 和 U 是 nested tuple。`(T,U)` 的 mutual refinement 是形如下图的 diagram：

![image](Imgaes/categorical-foundations-cute-layouts-paper/cc1b31a6974bd0438b26bd4d4bd636df42891252b07223ac829b12291b50fff4.jpg)


显式地，它是一对 nested tuple $(T',U')$，满足：

1. $T'$ refine T；

2. $U'$ refine U；

3. $T'$ 整除 $U'$。

除 mutual refinement 的定义外，还需要以下事实。

引理 3.2.5.20。假设 T 和 U 是 nested tuple。`(T,U)` 的 mutual refinement 与 $(T^\flat,U^\flat)$ 的 mutual refinement 之间存在一一对应关系。

证明。如果 $(T',U')$ 是 `(T,U)` 的 mutual refinement，则沿 unflattening isomorphism $\mathsf{id}_{T^\flat}^T$ 和 $\mathsf{id}_{U^\flat}^U$ 作 pullback，会得到 mutual refinement

$$
\begin{array}{c c} (\mathsf {i d} _ {T ^ {\flat}} ^ {T}) ^ {*} T ^ {\prime} & \longrightarrow (\mathsf {i d} _ {U ^ {\flat}} ^ {U}) ^ {*} U ^ {\prime} \\ \Big \downarrow & \Big \downarrow \\ T ^ {\flat} & U ^ {\flat} \end{array}
$$

它属于 $(T^\flat,U^\flat)$。反之，如果 $((T^\flat)',(U^\flat)')$ 是 $(T^\flat,U^\flat)$ 的 mutual refinement，则沿 flattening isomorphism $\mathrm{id}_T^{T^\flat}$ 和 $\mathsf{id}_U^{U^\flat}$ 作 pullback，会得到 mutual refinement

$$
\begin{array}{c c} (\mathsf {i d} _ {T} ^ {T ^ {\flat}}) ^ {*} (T ^ {\flat}) ^ {\prime} \longmapsto (\mathsf {i d} _ {U} ^ {U ^ {\flat}}) ^ {*} (U ^ {\flat}) ^ {\prime} \\ \Big \downarrow & \Big \downarrow \\ T & U \end{array}
$$

它属于 `(T,U)`。

## 3.2.6 Nested tuple morphism 上的操作

下一个任务是建立“nested tuple morphism algebra”。既然已经建立 tuple morphism algebra，可以通过为各种操作的输出赋予适当 profile，把它扩展到嵌套情况。

## 3.2.6.1 Concatenate

下面定义 nested tuple morphism 上的 concatenation 操作，它与 layout concatenation 相容，即

$$
L _ {(f, g)} = (L _ {f}, L _ {g}).
$$

通过 concatenate f 与 g 的 domain 来 concatenate nested tuple morphism f 和 g。为了使定义良好，f 和 g 需要满足下述不相交条件。

定义 3.2.6.1。假设 f 和 g 是具有相同 codomain 的 nested tuple morphism。如果 $f^\flat$ 与 $g^\flat$ 在定义 3.1.5.32 的意义下像不相交，就称 f 与 g 的像不相交。

示例 3.2.6.2。如果

$$
f: (3, (5 1 2, 5 1 2)) \to (2, 5 1 2, 2, 5 1 2)
$$

位于 `(*,2,4)` 之上，并且

$$
g: (2, 2) \to (2, 5 1 2, 2, 5 1 2)
$$

g 位于 `(1,3)` 之上，则 f 与 g 的像不相交。

示例 3.2.6.3。如果

$$
f: (2, (3 2, 6 4)) \rightarrow (3 2, (2, 2, 2), 6 4)
$$

f 位于 $\alpha=(3,1,5)$ 之上，并且

$$
g: ((2, 2)) \to (3 2, (2, 2, 2), 6 4)
$$

g 位于 $\beta=(2,4)$ 之上，则 f 与 g 的像不相交。

构造 3.2.6.4。假设 nested tuple morphism $f:S\to T$ 和 $g:U\to T$ 分别位于 α 和 β 之上，并且 f 与 g 的像不相交。把 f 与 g 的 concatenation 定义为 nested tuple morphism

$$
(f, g): (S, U) \to T
$$

with 

$$
\operatorname{flat} ((f, g)) = f ^ {\flat} \star g ^ {\flat}.
$$

更一般地，如果对 $1\leq i\leq k$，$f_i:S_i\to T$ 是 nested tuple morphism，而且 $f_1,\ldots,f_k$ 的像两两不相交，则定义 concatenation

$$
(f _ {1}, \dots , f _ {k}): (S _ {1}, \dots , S _ {k}) \to T.
$$

为满足下式的 nested tuple morphism：

$$
(f _ {1}, \dots , f _ {k}) ^ {\flat} = f _ {1} ^ {\flat} \star \dots \star f _ {k} ^ {\flat}.
$$

示例 3.2.6.5。示例 3.2.6.2 中 morphism f 与 g 的 concatenation 是 nested tuple morphism

$$
(f, g): ((3, (5 1 2, 5 1 2)), (2, 2)) \rightarrow (2, 5 1 2, 2, 5 1 2)
$$

它位于 $\alpha\star\beta=(*,2,4,1,3)$ 之上。

示例 3.2.6.6。示例 3.2.6.3 中 morphism f 与 g 的 concatenation 是 nested tuple morphism

$$
(f, g): ((2, (3 2, 6 4)), ((2, 2))) \to (3 2, (2, 2, 2), 6 4)
$$

它位于 $\alpha\star\beta=(3,1,5,2,4)$ 之上。

示例 3.2.6.7。如果

$$
f: (2, 2) \to (2, 3, 5, 2, 3, 5)
$$

f 位于 $\alpha=(1,4)$ 之上，

$$
g: (3, 3) \to (2, 3, 5, 2, 3, 5)
$$

g 位于 $\beta=(2,5)$ 之上，

$$
h: (5, 5) \to (2, 3, 5, 2, 3, 5)
$$

h 位于 $\gamma=(3,6)$ 之上，则 f、g、h 的像两两不相交，而且 concatenation

$$
(f, g, h): ((2, 2), (3, 3), (5, 5)) \to (2, 3, 5, 2, 3, 5)
$$

位于 $\alpha\star\beta\star\gamma=(1,4,2,5,3,6)$ 之上。

示例 3.2.6.8。假设 $f:S\to T$ 是 nested tuple morphism，并假设

$$
S ^ {\flat} = (s _ {1}, \dots , s _ {m}).
$$

回忆示例 3.2.3.4：对任意 $1\leq i\leq m$，存在 nested tuple morphism

$$
f _ {i}: s _ {i} \to T.
$$

称为 f 的第 i 个 entry。这些 morphism 的像两两不相交，而且 concatenation

$$
(f _ {1}, \ldots , f _ {m}): S ^ {\flat} \to T
$$

是 composite

$$
(f _ {1}, \dots , f _ {m}) = f \circ \mathrm{id} _ {S ^ {\flat}} ^ {S}
$$

，如示例 3.2.3.2 所示。

示例 3.2.6.9。假设 $f:S\to T$ 是 nested tuple morphism，并假设

$$
S = (S _ {1}, \dots , S _ {r}).
$$

回忆示例 3.2.3.6：对任意 $1\leq i\leq r$，存在 nested tuple morphism

$$
f _ {i}: S _ {i} \to T.
$$

称为 f 的第 i 个 mode。这些 morphism 的像两两不相交，而且 concatenation

$$
(f _ {1}, \dots , f _ {r}): S \to T
$$

等于 f。换言之，每个 nested tuple morphism f 都可以写成其 mode 的 concatenation：

$$
f = (f _ {1}, \dots , f _ {r}).
$$

命题 3.2.6.10。如果 $f_1,\ldots,f_k$ 是具有相同 codomain 且像两两不相交的 nested tuple morphism，则

$$
L _ {(f _ {1}, \dots , f _ {k})} = (L _ {f _ {1}}, \dots , L _ {f _ {k}}).
$$

证明。根据构造，

$$
\begin{array}{c} \text {shape} ((L _ {f _ {1}}, \ldots , L _ {f _ {k}})) = (\text {shape} (L _ {f _ {1}}), \ldots , \text {shape} (L _ {f _ {k}})) \\ = \text {shape} (L _ {(f _ {1}, \ldots , f _ {k})}). \end{array}
$$

and using Proposition 3.1.5.38, we have 

$$
\begin{array}{r l} (L _ {f _ {1}}, \ldots , L _ {f _ {k}}) ^ {\flat} & = L _ {f _ {1}} ^ {\flat} \star \dots \star L _ {f _ {k}} ^ {\flat} \\ & = L _ {f _ {1} ^ {\flat}} \star \dots \star L _ {f _ {k} ^ {\flat}} \\ & = L _ {f _ {1} ^ {\flat} \star \dots \star f _ {k} ^ {\flat}} \\ & = L _ {(f _ {1}, \ldots , f _ {k}) ^ {\flat}} \\ & = (L _ {(f _ {1}, \ldots , f _ {k})}) ^ {\flat}. \end{array}
$$

## 3.2.6.2 Coalesce

如果 f 是 nested tuple morphism，或许可以把 `coal(f)` 定义为 $\mathsf{coal}^\flat(f^\flat)$。理论上这是合理定义，但为了与 cute 实现相容，我们对 `coal(f)` 的定义作一处小修改。

定义 3.2.6.11。假设 $f:S\to T$ 是 nested tuple morphism，并写成

$$
\operatorname{coal} ^ {\flat} \left(f ^ {\flat}\right): \left(s _ {1}, \dots , s _ {m}\right)\rightarrow \left(t _ {1}, \dots , t _ {n}\right).
$$

• 情况 1：如果 $m>1$，定义

$$
\operatorname{coal} (f) = \operatorname{coal} ^ {\flat} \left(f ^ {\flat}\right).
$$

• 情况 2：如果 $m=1$，把 `coal(f)` 定义为 composite

$$
s _ {1} \xrightarrow [ (1) ]{} (s _ {1}) \xrightarrow {\operatorname{coal} ^ {b} (f ^ {b})} (t _ {1}, \dots , t _ {n}).
$$

• 情况 3：如果 $m=0$，把 `coal(f)` 定义为 composite

$$
1 \xrightarrow [ (*) ]{} () \xrightarrow {\operatorname{coal} ^ {b} (f ^ {b})} (t _ {1}, \dots , t _ {n}).
$$

示例 3.2.6.12。如果

$$
f: ((2, 2), (3, 3), (5, 5)) \to (5, 5, 3, 3, 2, 2)
$$

f 位于 $\alpha=(5,6,3,4,1,2)$ 之上，则

$$
\operatorname{coal} (f): (4, 9, 2 5) \rightarrow (2 5, 9, 4)
$$

`coal(f)` 位于 $\alpha'=(3,2,1)$ 之上。

命题 3.2.6.13。如果 $f:S\to T$ 是 nested tuple morphism，则

$$
\operatorname{coal} \left(L _ {f}\right) = L _ {\operatorname{coal} (f)}.
$$

证明。再次写成

$$
(s _ {1}, \ldots , s _ {m}) \xrightarrow [ \alpha ]{\text {coal} ^ {\flat} (f ^ {\flat})} (t _ {1}, \ldots , t _ {n}).
$$

需要考虑三种情况。

• 情况 1：假设 $m>1$，则

$$
\begin{array}{l} L _ {\text { coal } (f)} = L _ {\text { coal } ^ {\flat} (f ^ {\flat})} \\ \quad = \text { coal } ^ {\flat} (L _ {f ^ {\flat}}) \\ \quad = \text { coal } ((L _ {f}) ^ {\flat}) \\ \quad = \text { coal } (L _ {f}). \end{array}
$$

• 情况 2：假设 $m=1$，则

$$
\begin{array}{l} L _ {\mathsf {c o a l} (f)} = s _ {1}: t _ {1} \ldots , t _ {\alpha (1) - 1} \\ \qquad = \mathsf {c o a l} ((s _ {1}): (t _ {1} \dots t _ {\alpha (1) - 1})) \\ \qquad = \mathsf {c o a l} (L _ {\mathsf {c o a l} ^ {\flat} (f ^ {\flat})}) \\ \qquad = \mathsf {c o a l} (\mathsf {c o a l} ^ {\flat} (L _ {f ^ {\flat}})) \\ \qquad = \mathsf {c o a l} ((L _ {f}) ^ {\flat}) \\ \qquad = \mathsf {c o a l} (L _ {f}). \end{array}
$$

• 情况 3：假设 $m=0$，则

$$
\begin{array}{l} L _ {\mathsf {c o a l} (f)} = 1: 0 \\ \qquad = \mathsf {c o a l} (\mathbf {\Omega}): (\mathbf {\Omega}) \\ \qquad = \mathsf {c o a l} (L _ {\mathsf {c o a l} ^ {\flat} (f ^ {\flat})}) \\ \qquad = \mathsf {c o a l} (\mathsf {c o a l} ^ {\flat} (L _ {f ^ {\flat}})) \\ \qquad = \mathsf {c o a l} ((L _ {f}) ^ {\flat}) \\ \qquad = \mathsf {c o a l} (L _ {f}). \end{array}
$$

## 3.2.6.3 Complement

本节定义互为 complement 的 nested tuple morphism。

定义 3.2.6.14。假设 $f:S\to T$ 和 $g:U\to T$ 是像不相交的 nested tuple morphism。如果

$$
(f, g): (S, U) \to T
$$

是 isomorphism，就称 g 是 f 的 complement。

注记 3.2.6.15。如果 $f:S\to T$ 和 $g:U\to T$ 是 nested tuple morphism，则 g 是 f 的 complement，当且仅当 $g^\flat$ 是 $f^\flat$ 的 complement，因为 $(f,g)^\flat=f^\flat\star g^\flat$。

命题 3.2.6.16。如果 $f:S\to T$ 是 nested tuple morphism，$g:U\to T$ 是 f 的 complement，则 $L_g$ 是 $L_f$ 的 size(T)-complement。

证明。观察 3.2.2.5 蕴含

$$
\begin{array}{l} (L _ {f}) ^ {\flat} = L _ {f ^ {\flat}}, \text {and} \\ (L _ {g}) ^ {\flat} = L _ {g ^ {\flat}} \end{array}
$$

而引理 2.3.6.2 允许把问题归约到 flat 情况，即命题 3.1.5.42。

构造 3.2.6.17。假设 $f:S\to T$ 是 nested tuple morphism。把 f 的 complement 定义为 composite

![image](Imgaes/categorical-foundations-cute-layouts-paper/913fbe69c4fc455d52ffa86e8273f962b26e3b2fbca8bd6f7aae9197e8fd03ce.jpg)


其中 $(f^\flat)^c$ 如构造 3.1.5.46 所定义，$\mathsf{id}_{T^\flat}^T:T^\flat\cong T$ 是 unflattening isomorphism。

示例 3.2.6.18。nested tuple morphism

$$
((2, 2), (5, 5)) \xrightarrow [ (1 , 4 , 2 , 5) ]{f} ((2, 5, 7), (2, 5, 7))
$$

的 complement 是

$$
(7, 7) \xrightarrow [ (3 , 6) ]{f ^ {c}} ((2, 5, 7), (2, 5, 7)).
$$

命题 3.2.6.19。假设 $f:S\to T$ 和 $g:U\to T$ 是 nested tuple morphism。如果 f 是单射，且 g 是 f 的 complement，则 $L_g$ 是 $L_f$ 的 size(T)-complement。

证明。由命题 3.1.5.42 和引理 2.3.6.2 可得，因为

$$
\begin{array}{l} (L _ {f}) ^ {\flat} = L _ {f ^ {\flat}} \\ (L _ {g}) ^ {\flat} = L _ {g ^ {\flat}}. \end{array}
$$

命题 3.2.6.20。如果 $f:S\to T$ 是单射 nested tuple morphism，则

$$
\operatorname{coal} \left(L _ {f ^ {c}}\right) = \operatorname{comp} \left(L _ {f}, \text { size } (T)\right).
$$

证明。由于 $f^c$ 通过在 $(f^\flat)^c$ 之后复合 reparenthesization isomorphism 得到，所以

$$
L _ {f ^ {c}} = L _ {(f ^ {\flat}) ^ {c}}
$$

因此根据前述 flat 结果，

$$
\operatorname{coal} ^ {\flat} \left(L _ {f ^ {c}}\right) = \operatorname{comp} ^ {\flat} \left(L _ {f}, \operatorname{size} (T)\right).
$$

对等式两侧应用 coal(−)，得到结论。

## 3.2.6.4 Composition

可以使用第 3.2.4 节的 realization functor，证明 nested tuple morphism 的 composition 与关联 layout 的 composition 相容。

定理 3.2.6.21。如果 f 和 g 是 non-degenerate 且可复合的 nested tuple morphism，则

$$
L _ {g \circ f} = L _ {g} \circ L _ {f}.
$$

证明。假设 $f:S\to T$ 和 $g:T\to U$ 是 non-degenerate nested tuple morphism。需要检查：

1. `shape(L_{g∘f})` refine `shape(L_f)`：成立，因为

$$
\operatorname{shape} \left(L _ {f}\right) = S = \operatorname{shape} \left(L _ {g \circ f}\right).
$$

2. $L_{g\circ f}$ 在 `shape(L_f)` 上 coalesced：成立，因为 nested tuple morphism $g\circ f$ 是 non-degenerate，因此 layout $L_{g\circ f}$ 也是。

3. Φ $ \cdot _ { L _ { g \circ f } } = \Phi _ { L _ { g } } \circ \Phi _ { L _ { f } } ^ { \mathsf { s i z e } ( L _ { g } ) }$ : Using Lemma 3.2.4.2, we have 

$$
\begin{array}{r l} \Phi_ {L _ {g \circ f}} ^ {\text {size} (U)} & = | g \circ f | \\ & = | g | \circ | f | \\ & = \Phi_ {L _ {g}} ^ {\text {size} (U)} \circ \Phi_ {L _ {f}} ^ {\text {size} (T)} \end{array}
$$

再在之后复合 inclusion $[0,\mathsf{size}(U))\subset\mathbb{Z}$，并注意 $\mathsf{size}(T)=\mathsf{size}(L_g)$，即可得到结论。

## 3.2.6.5 Logical division

下面介绍 nested tuple morphism 的 logical division。该构造通过在 flat division 中引入嵌套 profile 得到，没有额外相容性约束。

定义 3.2.6.22。假设 f 和 g 是 nested tuple morphism。如果 g 与 f 可复合，就称 g 整除 f。换言之，

$$
\operatorname{codomain} (g) = \operatorname{domain} (f).
$$

定义 3.2.6.23。假设 $g:S\to T$ 和 $f:T\to U$ 是 nested tuple morphism。把 f 除以 g 的 logical division 定义为 nested tuple morphism

$$
f \oslash g = f \circ (g, g ^ {c}).
$$

示例 3.2.6.24。以下 f

$$
((2, 2), 2) \xrightarrow [ (2 , 4 , *) ]{f} ((4, 2), (4, 2))
$$

除以

$$
(2, 2) \xrightarrow [ (1 , 3) ]{g} ((2, 2), 2)
$$

的 logical division 为

$$
((2, 2), 2) \xrightarrow [ (2 , * , 4) ]{f \oslash g} ((4, 2), (4, 2)).
$$

示例 3.2.6.25。以下 f

$$
(8, 8, 5 1 2, 5 1 2, 5 1 2) \xrightarrow [ (* , * , 1 , 2 , 3) ]{f} (5 1 2, 5 1 2, 5 1 2)
$$

除以

$$
(8, 5 1 2) \xrightarrow [ (1 , 5) ]{g} (8, 8, 5 1 2, 5 1 2, 5 1 2)
$$

的 logical division 为

$$
((8, 5 1 2), (8, 5 1 2, 5 1 2)) \xrightarrow [ (* , 1 , * , 2 , 3) ]{f \oslash g} ((4, 2), (4, 2)).
$$

命题 3.2.6.26。如果 $g:S\to T$ 和 $f:T\to U$ 是 non-degenerate nested tuple morphism，则

$$
\operatorname{coal} \left(L _ {f \oslash g}\right) = \operatorname{coal} \left(L _ {f} \oslash L _ {g}\right).
$$

证明。根据命题 3.2.6.20，有

$$
\operatorname{coal} \left(\operatorname{comp} \left(L _ {g}, \text { size } \left(L _ {f}\right)\right)\right) = \operatorname{coal} \left(L _ {g ^ {c}}\right)
$$

计算得

$$
\begin{array}{l} \text {coal} (L _ {f} \oslash L _ {g}) = \text {coal} (L _ {f} \circ (L _ {g}, \text {comp} (L _ {g}, \text {size} (L _ {f})))) \\ \qquad = \text {coal} (L _ {f} \circ (L _ {g}, L _ {g ^ {c}})) \\ \qquad = \text {coal} (L _ {f} \circ L _ {(g, g ^ {c})}) \\ \qquad = \text {coal} (L _ {f} \circ L _ {(g, g ^ {c})}) \\ \qquad = \text {coal} (L _ {f \circ (g, g ^ {c})}) \\ \qquad = \text {coal} (L _ {f \oslash g}). \end{array}
$$

命题 3.2.6.27。如果 f 和 g 是 nested tuple morphism，且 g 整除 f，则

$$
(f \oslash g) ^ {\flat} = f ^ {\flat} \oslash^ {\flat} g ^ {\flat}.
$$

证明。计算得

$$
\begin{array}{l} (f \oslash g) ^ {\flat} = (f \circ (g, g ^ {c})) ^ {\flat} \\ \qquad = f ^ {\flat} \circ (g, g ^ {c}) ^ {\flat} \\ \qquad = f ^ {\flat} \circ (g ^ {\flat} \star (g ^ {c}) ^ {\flat}) \\ \qquad = f ^ {\flat} \circ (g ^ {\flat} \star (g ^ {\flat}) ^ {c}) \\ \qquad = f ^ {\flat} \oslash^ {\flat} g ^ {\flat}. \end{array}
$$

## 3.2.6.6 Logical product

本节定义 nested tuple morphism 的 logical product。

定义 3.2.6.28。假设 f 和 g 是 nested tuple morphism。如果 `codomain(g)=domain(f^c)`，就称 f 与 g 为 product admissible。如果 f 与 g 为 product admissible，把二者的 logical product 定义为 nested tuple morphism

$$
f \otimes g = (f, f ^ {c} \circ g).
$$

示例 3.2.6.29。nested tuple morphism

$$
(8, 8) \xrightarrow [ (1 , 2) ]{f} (8, 8, 1 6, 1 6)
$$

and 

$$
(1 6, 1 6) \xrightarrow [ (1 , 2) ]{g} (1 6, 1 6)
$$

为 product admissible，其 logical product 为

$$
((8, 8), (1 6, 1 6)) \xrightarrow [ (1 , 2 , 3 , 4) ]{f \otimes g} (8, 8, 1 6, 1 6).
$$

示例 3.2.6.30。nested tuple morphism

$$
(1 2 8, 1 2 8) \xrightarrow [ (3 , 4) ]{f} (3 2, 3 2, 1 2 8, 1 2 8)
$$

and 

$$
(3 2) \xrightarrow [ (2) ]{g} (3 2, 3 2)
$$

为 product admissible，其 logical product 为

$$
((1 2 8, 1 2 8), (3 2)) \xrightarrow [ (3 , 4 , 2) ]{f \otimes g} (3 2, 3 2, 1 2 8, 1 2 8).
$$

命题 3.2.6.31。假设 f 和 g 是 non-degenerate nested tuple morphism，并且 f 与 g 为 product admissible。则

$$
L _ {f \otimes g} = L _ {f} \otimes L _ {g}.
$$

证明。假设 $f:S\to T$ 和 $g:U\to V$ 为 product admissible，并令

$$
L _ {f} ^ {*} = \operatorname{comp} (L _ {f}, \text { size } (L _ {f}) \cdot \text { cosize } (L _ {g}))
$$

由于 f 是单射，且 `codomain(g)=domain(f^c)`，所以

$$
\operatorname{size} \left(L _ {f}\right) \cdot \operatorname{cosize} \left(L _ {g}\right) \leq \operatorname{size} (S) \cdot \operatorname{size} (V) = \operatorname{size} (T).
$$

使用该事实以及

$$
\Phi_ {\mathrm{comp} (L _ {f}, \mathrm{size} (T))} = \Phi_ {L _ {f c}},
$$

可得

$$
\begin{array}{c} L _ {f} ^ {*} \circ L _ {g} = \mathsf {c o m p} (L _ {f}, \mathsf {s i z e} (T)) \circ L _ {g} \\ = L _ {f ^ {c}} \circ L _ {g}. \end{array}
$$

使用该事实，计算得

$$
\begin{array}{r l} L _ {f} \otimes L _ {g} & = (L _ {f}, L _ {f} ^ {*} \circ L _ {g}) \\ & = (L _ {f}, L _ {f ^ {c}} \circ L _ {g}) \\ & = (L _ {f}, L _ {f ^ {c} \circ g}) \\ & = L _ {(f, f ^ {c} \circ g)} \\ & = L _ {f \otimes g} \end{array}
$$

## 第 4 章

## 计算

category **Tuple** 和 **Nest** 为 tractable layout 的计算提供了强大框架。但实践中经常遇到这样的 tractable layout A 和 B：它们在 cute 中可复合，其 standard representation 在 **Tuple** 或 **Nest** 中却不可复合。本章说明如何借助 mutual refinement 概念，仍然使用 **Tuple** 和 **Nest** 计算 tractable layout 的 composition、logical division 和 logical product。第 4.1.1 节引入该概念，算法 4.1.1 给出 mutual refinement 的计算算法，并通过大量显式示例进行说明。

## 4.1 Tractable layout 的 composition

假设要计算 tractable layout

$$
\begin{array}{l} A = (6, 6): (6, 1), \\ B = (1 2, 3, 6): (1, 7 2, 1 2). \end{array}
$$

的 composition $B\circ A$。或许会尝试通过计算 A 与 B 的 standard representation f 和 g 的 composite 来计算 $B\circ A$：

$$
\begin{array}{c c} 6 & 6 \\ 6 & 6 \end{array} f \quad \begin{array}{c c} 6 & 3 \\ 3 & 6 \\ 1 2 & 1 2 \end{array}
$$

不过，这些 morphism 不可复合，因为 f 的 codomain `(6,6)` 不等于 g 的 domain `(12,3,6)`。这意味着不能直接使用 f 和 g 计算 composite $B\circ A$。但可以寻找 `(6,6)` 与 `(12,3,6)` 的 mutual refinement 来继续计算，如下图所示：

$$
\begin{array}{c} 6 \searrow \\ 3 \searrow 6 \\ 6 \swarrow 2 \searrow 3 \\ 6 \searrow 6 \searrow 1 2 \end{array}
$$

这是把 f 和 g 转换为可复合 morphism $f'$ 和 $g'$ 的工具：

![image](Imgaes/categorical-foundations-cute-layouts-paper/5c4ef193a4ea19eea993819b984cc2b799503da03b57d5990b2af3c18a4657af.jpg)


![image](Imgaes/categorical-foundations-cute-layouts-paper/d01caf72cc2adb90b6ea1bc6f69469c2ae5511a6112d5acc8ba96abe60b2e941.jpg)


morphism $f'$ 和 $g'$ 可复合，因此可以形成 composite

![image](Imgaes/categorical-foundations-cute-layouts-paper/9ff8b66369fd1e2be3045cca8238fb721e72bb3bdcb07eeed852067563ec5352.jpg)


计算所编码的 layout，得到

$$
B \circ A = L _ {g ^ {\prime} \circ f ^ {\prime}} = ((2, 3), 6): ((6, 7 2), 1).
$$

本节目标是把这一计算过程形式化为计算 tractable layout A 与 B 的 composite 的算法。如示例所示，计算中的非平凡步骤是：

1. 寻找某些 nested tuple 的 mutual refinement；

2. 使用 mutual refinement 把 f 和 g 转换成可复合 morphism $f'$ 和 $g'$。

下面两节专门解释这些步骤。

## 4.1.1 Mutual refinement

在使用第 3 章的范畴论框架精确定义 mutual refinement 前，先作非正式概览。考虑启发性示例中的 tuple `(6,6)` 和 `(12,3,6)`。前文断言下图

$$
\begin{array}{c} 6 \searrow \\ 3 \searrow 6 \\ 6 \swarrow 2 \searrow 3 \\ 6 \searrow 6 \searrow 1 2 \end{array}
$$

是 `(6,6)` 与 `(12,3,6)` 的 mutual refinement。可以更精确地描述它：图的左半部分表示 refinement `(6,(2,3)) ↠ (6,6)`，右半部分表示 refinement `((6,2),3,6) ↠ (12,3,6)`：

$$
\begin{array}{l} 6 \xrightarrow {\angle} 2 \\ 6 - 6 \end{array} \qquad \qquad \leftrightarrow \qquad \qquad (6, 6) \ll - (6, (2, 3))
$$

$$
\begin{array}{l} 6 \\ 3 \searrow 6 \\ 2 \searrow 3 \\ 6 \searrow 1 2 \end{array} \quad \leftrightarrow \quad ((6, 2), 3, 6) \longrightarrow (1 2, 3, 6)
$$

图的两半可以粘合，对应 nested tuple `(6,(2,3))` 整除 `((6,2),3,6)`，记作

$$
(6, (2, 3)) \succrightarrow ((6, 2), 3, 6).
$$

综合这些观察，可以把 mutual refinement 精确表示为

$$
\begin{array}{c c c} 6 & \\ 3 & \searrow & 6 \\ 6 \swarrow & 2 & \searrow & 3 \\ 6 & - & 6 & - & 1 2 \end{array} \qquad \leftrightarrow \qquad \begin{array}{c c c} (6, (2, 3)) \longmapsto ((6, 2), 3, 6) \\ \downarrow & & \downarrow \\ (6, 6) & & (1 2, 3, 6) \end{array}
$$

这里选择纵向描绘 refinement `(6,(2,3)) ↠ (6,6)` 和 `((6,2),3,6) ↠ (12,3,6)`。现在可以精确定义 mutual refinement。

定义 4.1.1.1。假设 T 和 U 是 nested tuple。`(T,U)` 的 mutual refinement 是形如下式的 diagram：

$$
\begin{array}{c c c} T ^ {\prime} & \longrightarrow & U ^ {\prime} \\ \Big \downarrow & & \Big \downarrow \\ T & & U \end{array}
$$

显式地，它是一对 nested tuple $(T',U')$，满足：

1. $T'$ refine T；

2. $U'$ refine U；

3. $T'$ 整除 U′。

示例 4.1.1.2。$T=(6,6)$ 与 $U=(2,6,3)$ 的一个 mutual refinement 为

$$
\begin{array}{c c} ((2, 3), (2, 3)) \longrightarrow & (2, (3, 2), 3) \\ \Big \downarrow & \Big \downarrow \\ (6, 6) & (2, 6, 3) \end{array}
$$

把该 mutual refinement 描绘如下。

$$
\begin{array}{c} 3 \\ 6 \text {   \text {   }   } 2 \text {   \text {   }   } 3 \\ 3 \text {   \text {   }   } 6 \\ 6 \text {   \text {   }   } 2 \text {   \text {   }   } 2 \end{array}
$$

示例 4.1.1.3。$T=(8,8,8)$ 与 $U=(2,8,8,8)$ 的一个 mutual refinement 为

$$
\begin{array}{c} ((2, 4), (2, 4), (2, 4)) \longrightarrow (2, (4, 2), (4, 2), (4, 2)) \\ \Big \downarrow \\ (8, 8, 8) \end{array}
$$

把该 mutual refinement 描绘如下。

$$
\begin{array}{c} 2 \\ 4 \text { —— } 8 \\ 8 \text { —— } 2 \\ 4 \text { —— } 8 \\ 8 \text { —— } 2 \\ 4 \text { —— } 8 \\ 8 \text { —— } 2 \end{array}
$$

示例 4.1.1.4。$T=(4,2,2,32)$ 与 $U=(32,32)$ 的一个 mutual refinement 为

$$
\begin{array}{c c} (4, 2, 2, (2, 1 6)) \xrightarrow {} ((4, 2, 2, 2), (1 6, 2)) \\ \Big \downarrow & \Big \downarrow \\ (4, 2, 2, 3 2) & (3 2, 3 2) \end{array}
$$

把该 mutual refinement 描绘如下。

$$
\begin{array}{c} 2 \\ 1 6 - 3 2 \\ 3 2 - 2 \\ 2 - 2 \\ 2 - 2 \\ 4 - 4 - 3 2 \end{array}
$$

示例 4.1.1.5。如果 $T=(8,8)$，$U=(3,8,8)$，则 T 与 U 不存在 mutual refinement。

示例 4.1.1.6。如果 T 和 U 是满足 `size(T)=2^k`、`size(U)=2^ℓ` 且 $k\leq\ell$ 的 tuple，则 T 与 U 存在 mutual refinement。更一般地，如果 T 和 U 的 size 都是某个固定整数的幂，且 `size(T)≤size(U)`，则 T 与 U 存在 mutual refinement。

观察 4.1.1.7。前述各例都考察 flat tuple T 与 U 的 mutual refinement。但 mutual refinement 的定义允许 T 与 U 为任意 nested tuple。把范围限制到 flat 情况并不损失一般性，因为 nested tuple pair `(T,U)` 的 mutual refinement 与其 flattening pair $(T^\flat,U^\flat)$ 的 mutual refinement 之间存在一一对应关系，参见引理 3.2.5.20。特别地，`(T,U)` 存在 mutual refinement，当且仅当 $(T^\flat,U^\flat)$ 存在 mutual refinement。

完成适当定义后，下面给出计算 `(T,U)` mutual refinement 的算法。

算法 4.1.1：Mutual refinement 算法

1 输入：nested tuple T 和 U。
2 输出：如果存在，则返回 `(T,U)` 的 mutual refinement `(T',U')`；否则返回 None。

3 X ← T; Y ← U
4 X', Y', X $_{mode}$ , Y $_{mode}$ ← ()
5 i ← 1; j ← 1
6 while i ≤ len(X) and j ≤ len(Y) do
7    if entryi(X) = entryj(Y) then
8    append entryi(X) to X $_{mode}$ ; append X $_{mode}$ to X'; X $_{mode}$ ← ()
9    append entryj(Y) to Y $_{mode}$ ;
10    append Y $_{mode}$ to Y';
11    Y $_{mode}$ ← ()
12    i ← i + 1;
13    j ← j + 1
14    else if entryi(X) divides entryj(Y) then
15    append entryi(X) to X $_{mode}$ ;
16    append X $_{mode}$ to X';
17    X $_{mode}$ ← ()
18    append entryi(X) to Y $_{mode}$ 19    entryj(Y) ← entryj(Y)/entryi(X);
20    i ← i + 1
21    else if entryj(Y) divides entryi(X) then
22    append entryj(Y) to X $_{mode}$ ;
23    append entryj(Y) to Y $_{mode}$ ;
24    append Y $_{mode}$ to Y';
25    Y $_{mode}$ ← ()
26    entryi(X) ← entryi(X)/entryj(Y);
27    j ← j + 1
28    else
29    return None
30 end if
31 end while
32 if Y $_{mode}$ ≠ () then
33    append entryj(Y) to Y $_{mode}$ ;
34    append Y $_{mode}$ to Y';
35    j ← j + 1
36 end if
37 while j < len(Y) do
38    append entryj(Y) to Y';
39    j ← j + 1
40 end while
41 T' ← (X') $_{prof(T)}$ ;
42 U' ← (Y') $_{prof(U)}$ 43 return (T',U') 

## 4.1.2 从 mutual refinement 到可复合 morphism

回忆一下，为了计算以下 layout 的 composition $B\circ A$

$$
\begin{array}{l} A = (6, 6): (6, 1) \text {and} \\ B = (1 2, 3, 6): (1, 7 2, 1 2), \end{array}
$$

我们构造了 tuple morphism

$$
\begin{array}{c c} 6 & 6 \\ 6 & 6 \end{array} f \quad \begin{array}{c c} 6 & 3 \\ 3 & 6 \\ 1 2 & 1 2 \end{array}
$$

以及一个 mutual refinement。

$$
\begin{array}{c} 6 \searrow \\ 3 \searrow 6 \\ 6 \swarrow 2 \searrow 3 \\ 6 \swarrow 6 \searrow 1 2 \end{array}
$$

计算的下一步，是使用 mutual refinement 把 f 和 g 转换成可复合 morphism $f'$ 和 $g'$。在给出该过程的正式范畴论定义之前，先用示例说明。

使用 f 和 mutual refinement 的左半部分构造 $f'$：

![image](Imgaes/categorical-foundations-cute-layouts-paper/72bb5a90c7e07be2f4a2f1d29bd010cc38c7bbd8daa6b355c866fd1179336d9e.jpg)


该构造执行以下替换：

$$
\begin{array}{c c c}6&\rightsquigarrow&6 - 6\\\searrow&6 - 6&\searrow\\6&\end{array}
$$

并执行替换

![image](Imgaes/categorical-foundations-cute-layouts-paper/34dadeeb02a970f2f730d0bc28f2341da19e333fb89add6927a94d357266f0a2.jpg)


更一般地，执行替换

![image](Imgaes/categorical-foundations-cute-layouts-paper/d0ae701daa1cd82349a5e2a64d1667902227253b4bcbfbfb6b3f86fa50f82283.jpg)


使用 g 和 mutual refinement 右半部分构造 g′ 的过程类似。

$$
\begin{array}{c c}6&\\3&\searrow\\2&\searrow\\6&\longrightarrow\\\hline\end{array}\begin{array}{c c}6&\\3&\swarrow\\2&\longrightarrow\\6&\longmapsto\\\hline\end{array}\begin{array}{c c}3&\\6&\\\hline\end{array}\quad \rightsquigarrow \quad\begin{array}{c c}6&\\3&\swarrow\\2&\longmapsto\\6&\longmapsto\\\hline\end{array}\begin{array}{c c}3&\\6&\\\hline\end{array}\quad g ^ {\prime}
$$

该构造执行以下替换：

$$
6 \longrightarrow 6 \longmapsto 6 \quad \rightsquigarrow \quad 6 \longmapsto 6
$$

$$
3 \longrightarrow 3 \longmapsto 3 \quad \rightsquigarrow \quad 3 \longmapsto 3
$$

$$
\begin{array}{c c c}2&&2 \longmapsto 2\\6 \xrightarrow {\quad} 1 2 \longmapsto 1 2&\rightsquigarrow&6 \longmapsto 6\end{array}
$$

更一般地，执行替换

![image](Imgaes/categorical-foundations-cute-layouts-paper/6427b8e7b6ed709583725f07d8691938d622561723cfbc6a7d5b0357edb81551.jpg)


给出过程的非正式描述后，下面作出精确定义。

构造 4.1.2.1。假设 $f:S\to T$ 和 $g:U\to V$ 是 nested tuple morphism，$(T',U')$ 是 `(T,U)` 的 mutual refinement。可以使用第 3.2.5 节的 pullback 和 pushforward 构造形成 diagram：

$$
\begin{array}{c c c c c} S ^ {\prime} & \xrightarrow {\tilde {f}} & T ^ {\prime} & \xrightarrow {i} & U ^ {\prime} \\ \Big \downarrow & \searrow & \Big \downarrow & & \Big \downarrow \\ S & \xrightarrow [ f ] & T & & U \end{array} \xrightarrow [ g ]{\quad} V
$$

如果令 $f'=i\circ\tilde f$、$g'=\tilde g$，则

$$
S ^ {\prime} \xrightarrow {f ^ {\prime}} U ^ {\prime} \xrightarrow {g ^ {\prime}} V ^ {\prime}
$$

是可复合 nested tuple morphism。

## 4.1.3 Composition 算法



算法 4.1.2：Tractable layout composition 算法



1 输入：tractable layout A 和 B。

算法 4.1.2（续）：Tractable layout composition 算法

2 输出：如果存在，则返回 A 与 B 的 weak composite C；否则返回 None。

3 取 A 与 `coal(B)` 各自的 standard representation

![image](Imgaes/categorical-foundations-cute-layouts-paper/38be5817b4cffe82fc1e5a8539ce54c67754b2a9016ece511a1f9214cd8b4669.jpg)


。4 使用算法 4.1.1 生成 `(T,U)` 的 mutual refinement

![image](Imgaes/categorical-foundations-cute-layouts-paper/23c6f7a2e3c3ffcc404c6670eabed1e2cb75b4267c91bdfe736c1345a68aca9b.jpg)


。如果 `(T,U)` 不存在 mutual refinement，则返回 None。5 使用构造 4.1.2.1 得到可复合 nested tuple morphism

$$
S ^ {\prime} \xrightarrow {f ^ {\prime}} U ^ {\prime} \xrightarrow {g ^ {\prime}} V ^ {\prime}
$$

6 复合 $f'$ 与 $g'$，并计算编码的 layout

$$
C = L _ {g ^ {\prime} \circ f ^ {\prime}}
$$

7 返回 C

定理 4.1.3.1。如果 A 和 B 是 tractable layout，则前述算法输出的 C 是 A 与 B 的 weak composite。因此，

$$
B \circ A = \operatorname{coal} (C, \operatorname{shape} (A)).
$$

证明。命题 3.2.5.15 告诉我们

$$
\Phi_ {L _ {g ^ {\prime}}} = \Phi_ {L _ {g}} = \Phi_ {\mathsf {c o a l} (B)} = \Phi_ {B},
$$

命题 3.2.5.11 和示例 3.1.3.6 告诉我们

$$
\Phi_ {L _ {f ^ {\prime}}} = \Phi_ {L _ {f}} = \Phi_ {A}.
$$

定理 3.2.6.21 于是蕴含

$$
\begin{array}{r} \Phi_ {C} = \Phi_ {L _ {g ^ {\prime} \circ f ^ {\prime}}} = \Phi_ {g ^ {\prime}} \circ \Phi_ {f ^ {\prime}} ^ {\mathrm{size} (U ^ {\prime})} \\ = \Phi_ {B} \circ \Phi_ {A} ^ {\mathrm{size} (B)}. \end{array}
$$

根据构造，$L_{f'}$ 的 shape $S'$ refine A 的 shape S，所以 C 是 A 与 B 的 weak composite。□

## 4.1.4 示例

本节说明如何使用算法 4.1.3 计算 tractable layout A 与 B 的 composition $B\circ A$。

示例 4.1.4.1。假设 $A=(4):(1)$，$B=(2,2):(2,1)$。

1. 取 A 与 coa $| ( B ) = B$ 

$$
\begin{array}{c} 4 \longmapsto 4 \\ f \end{array} \qquad \qquad \begin{array}{c} 2 \\ 2 \\ g \end{array}
$$

2. 应用算法 4.1.1 得到 mutual refinement 

$$
4 \leq \begin{array}{l} 2 - 2 \\ 2 - 2 \end{array}
$$

3. 形成 diagram 

![image](Imgaes/categorical-foundations-cute-layouts-paper/478d9766571f93615b10ac14083520b0240943330cba7f649e47e4a3b76d42aa.jpg)


4. 解析 diagram 

![image](Imgaes/categorical-foundations-cute-layouts-paper/f4636f6a12862b6eb86fde915fcc4e0d414090928b5791c23b69962cce7a0ce3.jpg)


5. 复合 $f^'$ 与 $g^'$，得到 

$$
4 \leq \begin{array}{c} 2 \\ 2 \end{array} \bigotimes_ {g ^ {\prime} \circ f ^ {\prime}} ^ {2}
$$

6. 计算关联 layout 

$$
L _ {g ^ {\prime} \circ f ^ {\prime}} = ((2, 2)): ((2, 1)).
$$

7. $L_{g^'\circ f^'}$ 在 (4) 上 coalesced，所以

$$
B \circ A = ((2, 2)): ((2, 1)).
$$

示例 4.1.4.2。假设 $A=(6,6):(6,1)$，$B=(12,3,6):(1,72,12)$。

1. 取 A 与 ${ \mathsf { c o a l } } ( B ) = B$ 

$$
\begin{array}{c} 6 \\ 6 \end{array} \xrightarrow {} \begin{array}{c} 6 \\ 6 \end{array} f
$$

$$
\begin{array}{c} 6 \xrightarrow {} 3 \\ 3 \xrightarrow {} 6 \\ 1 2 \longmapsto 1 2 \end{array} g
$$

2. 应用算法 4.1.1 得到 mutual refinement 

$$
\begin{array}{c} 6 \searrow \\ 3 \searrow 6 \\ 6 \swarrow 2 \searrow 3 \\ 6 \text { - - - } 6 \text { - - - } 1 2 \end{array}
$$

3. 形成 diagram 

![image](Imgaes/categorical-foundations-cute-layouts-paper/99b486f6093c99a488f4090be8890226bbda3df622ec2626f44d704090199c93.jpg)


4. 解析 diagram，得到 

![image](Imgaes/categorical-foundations-cute-layouts-paper/1d2ef844619fecd333562d183a93783fd95954dd9e15235177a1a9765b13a279.jpg)


5. 复合 $f^'$ 与 $g^'$，得到 

![image](Imgaes/categorical-foundations-cute-layouts-paper/0c00f7033a1449ca32fbab952a19b920e467c39d61449bbdf6650b804a913618.jpg)


6. 计算关联 layout 

$$
L _ {g ^ {\prime} \circ f ^ {\prime}} = ((2, 3), 6): ((6, 7 2), 1).
$$

7. $L_{g^'\circ f^'}$ 在 (6, 6) 上 coalesced，因此 

$$
B \circ A = ((2, 3), 6): ((6, 7 2), 1).
$$

示例 4.1.4.3。假设 $A=(8,8):(8,1)$，$B=(16,16):(16,1)$。

1. 取 A 与 coal(B) = B. 

$$
\begin{array}{c c} 8 & 8 \\ 8 & 8 \end{array} \quad f \quad g
$$

2. 应用算法 4.1.1 得到 mutual refinement 

$$
\begin{array}{c} 4 \\ 4 \searrow \\ 8 \text {   - - -   } 2 \\ 8 \text {   - - -   } 8 \end{array} \begin{array}{c} 1 6 \\ \text {   - - -   } 1 6 \end{array}
$$

3. 形成 diagram 

![image](Imgaes/categorical-foundations-cute-layouts-paper/c9e81d362a0eb52da1c86556a59d0fb75a9d1def74a0047229d3a181337f51e9.jpg)


4. 解析 diagram，得到 

![image](Imgaes/categorical-foundations-cute-layouts-paper/7e81ced34411756f8f8618c0b7ab47560190eaa478aa79c6a2982a9c57e5d983.jpg)


5. 复合 $f^'$ 与 $g^'$，得到 

![image](Imgaes/categorical-foundations-cute-layouts-paper/f64e5081c96a6af0ed5114c1c39cff2a79d5ab49aac928052c4fe820d41127e7.jpg)


6. 计算关联 layout 

$$
L _ {g ^ {\prime} \circ f ^ {\prime}} = ((2, 4), 8): ((1 2 8, 1), 1 6)
$$

7. $L_{g^'\circ f^'}$ 在 (8, 8) 上 coalesced，因此 

$$
B \circ A = ((2, 4), 8): ((1 2 8, 1), 1 6)
$$

示例 4.1.4.4。假设 $A=(16,16):(16,1)$，$B=(8,8,8):(64,8,1)$。

1. 取 A 与 coa $| ( B ) = B$ 

$$
\begin{array}{c c} 1 6 & 1 6 \\ 1 6 & \text {   f   } \end{array} \quad \begin{array}{c c} 8 & 8 \\ 8 & 8 \\ 8 & 8 \end{array} \quad g
$$

2. 应用算法 4.1.1 得到 mutual refinement 

$$
\begin{array}{c} 2 \\ 4 \bigwedge \\ 4 \bigwedge \\ 1 6 \bigwedge \\ 1 6 \bigwedge \\ 8 \bigwedge \end{array}
$$

3. 形成 diagram 

![image](Imgaes/categorical-foundations-cute-layouts-paper/e52ae499578c8522d5d0c2b63041ce71770284b331657062661766274d9adeb6.jpg)


4. 解析 diagram，得到 

![image](Imgaes/categorical-foundations-cute-layouts-paper/ec5303983a24b9b0047976708abeffe6351743f6b97284cc53687b13b82fa7b1.jpg)


5. 复合 $f^'$ 与 $g^'$，得到 

![image](Imgaes/categorical-foundations-cute-layouts-paper/db842ccdb277087b96b26390c12c9bd7a192e57005cfd4a02bcbf5bcb7074362.jpg)


6. 计算关联 layout 

$$
L _ {g ^ {\prime} \circ f ^ {\prime}} = ((4, 4), (8, 2)): ((1 6, 1), (6 4, 8)).
$$

7. $L_{g^'\circ f^'}$ 在 (16, 16) 上 coalesced，因此 

$$
B \circ A = ((4, 4), (8, 2)): ((1 6, 1), (6 4, 8)).
$$

示例 4.1.4.5。假设 $A=(6,6):(5,60)$，$B=(10,360):(2,60)$。

1. 取 A 与 ${ \mathsf { c o a l } } ( B ) = B$ 

![image](Imgaes/categorical-foundations-cute-layouts-paper/e710f9a6f7d1b9a1dc4d1a5e2def91babfa7df1bde9201ee7c416d7758425b00.jpg)


2. 应用算法 4.1.1 得到 mutual refinement 

![image](Imgaes/categorical-foundations-cute-layouts-paper/f5bce72e34cb284e93156a4287404f658358b8c590be9aad70b59a38dfbdd082.jpg)


3. 形成 diagram 

![image](Imgaes/categorical-foundations-cute-layouts-paper/f85d4163b58431cdda0599f0720b327482805416b0a0f55cbc0c773ee3eb3084.jpg)


4. 解析 diagram，得到 

![image](Imgaes/categorical-foundations-cute-layouts-paper/b2786a69cd901056bbdc5ae3b2b777e3792f5ff2d7c1bb3d7ad3b5ce9a555973.jpg)


5. 复合 $f^'$ 与 $g^'$，得到 

![image](Imgaes/categorical-foundations-cute-layouts-paper/904fd32192aa2609bd17b30dc0fd7df3f8df9275f0e3cd1b46f3263b4a281cb1.jpg)


6. 计算关联 layout 

$$
L _ {g ^ {\prime} \circ f ^ {\prime}} = ((2, 3), 6): ((1 0, 6 0), 3 6 0).
$$

7. layout $L_{g'\circ f'}$ 在 `(6,6)` 上 coalesced，所以

$$
B \circ A = ((2, 3), 6): ((1 0, 6 0), 3 6 0).
$$

## 4.1.5 更一般的 composition

本文建立的图形演算可以自然扩展，用于计算 tractable layout A 与任意 CuTe layout B 的 composition $B\circ A$。非正式地说，只需允许 tuple 的条目位于 $\mathbb{Q}_{>0}\supset\mathbb{Z}_{>0}$ 中。下面用一个计算示例说明该扩展。

考虑 layout $A=(4,4):(4,1)$ 和 $B=(8,8):(3,7)$。layout A 是 tractable 的，其 standard representation 是下图中的 tuple morphism f。

![image](Imgaes/categorical-foundations-cute-layouts-paper/44661e5bb33ec3c4fd85d70dd8ce442c618dcf95a29b0c239001998a894fc966.jpg)


layout B 不是 tractable 的，但仍可以用下图描绘 B：

![image](Imgaes/categorical-foundations-cute-layouts-paper/2f38e4e464e72f5b9a7ce9eb2ad211ccfd4907fd7c9bb1d4b390d0b14b428a0f.jpg)


该图不对应真正的 tuple morphism，因为“codomain tuple” $(3,8,\frac7{24},8)$ 含有非整数条目。不过，它仍通过通常的前缀乘积公式编码 layout B，并且仍可作为 composition 算法的有效输入。应用算法 4.1.1 得到 mutual refinement

![image](Imgaes/categorical-foundations-cute-layouts-paper/cded2d62505106d2381d302b78707e83d0add4ff5419f65c4a1c7bad6fc19a8e.jpg)


形成 diagram

![image](Imgaes/categorical-foundations-cute-layouts-paper/ea1da707dd7e1d5b2190229d334e5e230866a46ab8e957c29e7fbe2655bfa6cf.jpg)


解析该 diagram，得到

![image](Imgaes/categorical-foundations-cute-layouts-paper/70c51112cd6ee6414fa650d5e1354c98509dc7fb661b5019cb3f0a0e8d005fbe.jpg)


再复合 $f'$ 和 $g'$，得到 diagram

![image](Imgaes/categorical-foundations-cute-layouts-paper/022a308c1dc8246bb03b09558d81b99121ac6b20563ad27e99dabca8f8be98af.jpg)


所编码的 layout 为 `((2,2),4):((12,7),3)`，它在 `(4,4)` 上 coalesced，因此

$$
B \circ A = ((2, 2), 4): ((1 2, 7), 3).
$$

## 4.1.6 Composition 的 admissibility

[16] 引入了 composition admissibility 概念，它是 layout A 与 B 的 composition $B\circ A$ 存在的充分条件。下面回顾其定义。与 [16] 相同，把关注范围限制到没有 shape 条目等于 1 的 flat layout，并假设 composition 中第一个 layout 没有 stride 等于 0。

定义 4.1.6.1。假设

$$
A = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m})
$$

是没有 $s_i=1$、也没有 $d_i=0$ 的 flat layout。假设 B 是具有 shape

$$
\operatorname{shape} (B) = \left(u _ {1}, \dots , u _ {p}\right).
$$

的 flat layout。如果以下条件成立，就称 A 与 B 对 composition admissible：

1. 对每个 $1\leq i\leq m$，存在 $1\leq k\leq\ell\leq p$，使得

(a) $u _ { 1 } \cdots u _ { k - 1 }$ divides d<sub>i</sub>, $d _ { i }$ 

(b) $d _ { i }$ divides $u _ { 1 } \cdots u _ { k }$ (properly if $k < p )$ 7 

(c) $u _ { 1 } \cdots u _ { \ell - 1 }$ divides $s _ { i } d _ { i }$ 2 

(d) $s _ { i } d _ { i }$ divides $u _ { 1 } \cdot \cdot \cdot u _ { \ell }$ (properly if $\ell < p )$ 

2. 区间

$$
\left[ d _ {i}, d _ {i} \left(s _ {i} - 1\right) \right] \cap \left[ 1, s _ {1} \dots s _ {m - 1}\right)
$$

两两不相交。

注记 4.1.6.2。上述定义中的索引 k、ℓ 在 [16] 中称为“division index”。注记 4.1.6.3。[16] 使用 layout 的“extended layout function”，可视为把最后一个 shape 条目 $s_m$ 替换为 ∞ 后所得 layout 的 layout function。本文给出的定义，是处理普通 layout function 时的相应版本。

引理 4.1.6.4。假设 $T=(t_1,\dots,t_n)$ 和 $U=(u_1,\dots,u_p)$ 是正整数 tuple，并假设 `(T,U)` 存在 mutual refinement。则对 T 与 U 的任意前缀乘积 $t_1\cdots t_j$ 和 $u_1\cdots u_k$，以下条件之一成立：

1. $t_1\cdots t_j$ 大于 $u_1\cdots u_k$；或

2. $t_1\cdots t_j$ 整除 $u_1\cdots u_k$。

证明。选择 `(T,U)` 的某个 mutual refinement $(T',U')$，并把 $U'$ 的 flattening 写成 $(u_1',\ldots,u_{p'}')$。T 或 U 的任意前缀乘积也是 $U'$ 的前缀乘积；而固定正整数 tuple 的前缀乘积满足 $x\leq y\Rightarrow x\mid y$，结论得证。□

定理 4.1.6.5。假设 A 是没有 shape 条目等于 1、也没有 stride 条目等于 0 的 flat tractable layout。假设 B 是 flat tractable layout。令 $f:S\to T$ 和 $g:U\to V$ 分别表示 A 与 `coal(B)` 的 standard representation。如果 T 与 U 存在 mutual refinement，则 A 与 B 对 composition admissible。

证明。写成 $S=(s_1,\ldots,s_m)$、$T=(t_1,\ldots,t_n)$、$U=(u_1,\ldots,u_p)$，并用 $\alpha:\langle m\rangle_*\to\langle n\rangle_*$ 表示 f 所在的 map。需要检查定义 4.1.6.1 的条件成立。

1. 假设 $1\leq i\leq m$。对某个 $j=\alpha(i)$，有 $d_i=t_1\cdots t_{j-1}$。假设 `(T,U)` 有 mutual refinement $(T',U')$，并写成 $(T')^\flat=(t_1',\ldots,t_{n'}')$、$(U')^\flat=(u_1',\ldots,u_{p'}')$。

• (a) 与 (b)：由于 $T'$ refine T，存在某个 $1\leq a\leq n'$，使

$$
d _ {i} = t _ {1} \dots t _ {j - 1} = t _ {1} ^ {\prime} \dots t _ {a} ^ {\prime} = u _ {1} ^ {\prime} \dots u _ {a} ^ {\prime}.
$$

取满足 $u_1\cdots u_{k-1}\leq u_1'\cdots u_a'$ 的最大 $k\in\langle p\rangle$。

假设 $k<p$。可以观察到

$$
u _ {1} \cdot \cdot \cdot u _ {k - 1} \leq d _ {i} <   u _ {1} \cdot \cdot \cdot u _ {k}.
$$

其中第二个不等式由 k 的最大性成立。引理 4.1.6.4 蕴含 $u_1\cdots u_{k-1}$ 整除 $d_i$，且 $d_i$ 真整除 $u_1\cdots u_k$。

假设 $k=p$。可以观察到

$$
u _ {1} \dots u _ {k - 1} \leq d _ {i} = t _ {1} \dots t _ {j - 1} <   t _ {1} \dots t _ {n} \leq u _ {1} \dots u _ {p} = u _ {1} \dots u _ {k}.
$$

引理 4.1.6.4 蕴含 $u_1\cdots u_{k-1}$ 整除 $d_i$，且 $d_i$ 整除 $u_1\cdots u_k$；实际上是真整除，但这里不要求。

• (c) 与 (d)：同样，由于 $T'$ refine T，存在某个 $1\leq b\leq n'$，使

$$
s _ {i} d _ {i} = t _ {1} \dots t _ {j} = t _ {1} ^ {\prime} \dots t _ {b} ^ {\prime} = u _ {1} ^ {\prime} \dots u _ {b} ^ {\prime}.
$$

取满足 $u_1\cdots u_{\ell-1}\leq u_1'\cdots u_b'$ 的最大 $\ell\in\langle p\rangle$。

假设 $\ell<p$。可以观察到

$$
u _ {1} \dots u _ {\ell - 1} \leq s _ {i} d _ {i} <   u _ {1} \dots u _ {\ell}.
$$

其中第二个不等式由 ℓ 的最大性成立。引理 4.1.6.4 蕴含 $u_1\cdots u_{\ell-1}$ 整除 $s_id_i$，且 $s_id_i$ 真整除 $u_1\cdots u_\ell$。

– 假设 $\ell=p$。可以观察到

$$
u _ {1} \dots u _ {\ell - 1} \leq s _ {i} d _ {i} = t _ {1} \dots t _ {j} \leq t _ {1} \dots t _ {n} \leq u _ {1} \dots u _ {p} = u _ {1} \dots u _ {k}.
$$

引理 4.1.6.4 给出相应整除关系。

2. 对 $\langle m\rangle$ 中任意 $i\neq i'$，有 $d_i=t_1\cdots t_{j-1}$、$s_i=t_j$、$d_{i'}=t_1\cdots t_{j'-1}$、$s_{i'}=t_{j'}$，其中 $j=\alpha(i)$、$j'=\alpha(i')$。于是

$$
[ d _ {i}, d _ {i} (s _ {i} - 1) ] = [ t _ {1} \dots t _ {j - 1}, t _ {1} \dots t _ {j - 1} (t _ {j} - 1) ]
$$

and 

$$
[ d _ {i ^ {\prime}}, d _ {i ^ {\prime}} (s _ {i ^ {\prime}} - 1) ] = [ t _ {1} \dots t _ {j ^ {\prime} - 1}, t _ {1} \dots t _ {j ^ {\prime} - 1} (t _ {j ^ {\prime}} - 1) ]
$$

如果 $j'>j$，则

$$
t _ {1} \dots t _ {j - 1} (t _ {j} - 1) <   t _ {1} \dots t _ {j ^ {\prime} - 1}
$$

所以区间不重叠；另一顺序的情况类似。

## 4.2 Logical division 与 logical product

本节说明如何使用 composition 算法 4.1.3 计算 logical division 和 logical product。

## 4.2.1 Logical division 示例

回忆一下，如果 A 和 B 是 layout，则 logical division $A\oslash B$ 定义为

$$
A \oslash B = A \circ (B, B ^ {c})
$$

其中

$$
B ^ {c} = \operatorname{comp} (B, \text { size } (A)).
$$

示例 4.2.1.1。假设要计算 logical division $A\oslash B$，其中 $A=(8,8):(8,1)$，$B=(2,2):(1,4)$。可以写成 $A=L_g$、$B=L_h$、$B^c=L_{h^c}$，其中 h 和 $h^c$ 是下图中的 tuple morphism。

![image](Imgaes/categorical-foundations-cute-layouts-paper/29cd8e36011b3e73d2052609c8baedc358baee60c2a147728110fc2194dcc829.jpg)


因此 `(B,B^c)` 由下图中的 nested tuple morphism $f=(h,h^c)$ 编码。

![image](Imgaes/categorical-foundations-cute-layouts-paper/4fa71e21d0fd94aaf12e70f46b663049a8bf22892a989793e56835faa36d52b7.jpg)


随后与之前一样执行 composition 算法。使用算法 4.1.1 找到 mutual refinement

![image](Imgaes/categorical-foundations-cute-layouts-paper/021f589259d5c234ce128c749af632a0d867d109e87d4e94e2a3a30e07185274.jpg)


形成 diagram

![image](Imgaes/categorical-foundations-cute-layouts-paper/634d1f736f363ed0ca8b505fd103b0c4fa3b28b8a7c07163ff69bdb77237d445.jpg)


解析 diagram，得到

![image](Imgaes/categorical-foundations-cute-layouts-paper/dfd3d4e6c03fcb9209a49e8caf98ce8fb79bc3178d512a3d577edec9dcaccfff.jpg)


再复合 f 与 $g'$，得到

![image](Imgaes/categorical-foundations-cute-layouts-paper/92fc0604a9403d86eb6cbc9901718825a403ff5bc10e8833240af8538f0b1a8b.jpg)


该 nested tuple morphism 编码的 layout 为

$$
L _ {g ^ {\prime} \circ f} = ((2, 2), (2, 2)): ((8, 3 2), (1 6, 1))
$$

它在 `((2,2),(2,2))` 上 coalesced，因此

$$
A \oslash B = ((2, 2), (2, 2)): ((8, 3 2), (1 6, 1)).
$$

## 4.2.2 Logical product 示例

回忆一下，如果 A 和 B 是 layout，则 logical product `A ⊗ B` 定义为

$$
A \otimes B = (A, A ^ {c} \circ B)
$$

其中

$$
A ^ {c} = \operatorname{comp} (A, \text { size } (A) \cdot \text { cosine } (B)).
$$

特别地，如果要手工计算 $A\otimes B$，只需计算 $A^c\circ B$，再把结果与 A concatenate。

示例 4.2.2.1。假设要计算 logical product `A ⊗ B`，其中 $A=(2,2):(1,2)$，$B=(5,5):(5,1)$。则

$$
\begin{array}{r l} A ^ {c} & = \text { comp } (A, \text { size } (A) \cdot \text { cosize } (B)) \\ & = \text { comp } (A, 1 0 0) \\ & = (2 5): (4). \end{array}
$$

按照上一节的方式继续。

1. 取 B 与 $\mathsf{coal}(A^c)=A^c$ 的 standard representation。

![image](Imgaes/categorical-foundations-cute-layouts-paper/7a4ec91297944d91a02f45ede18a8ebf85b61e0b03a9e694e3b87b43387f482f.jpg)


2. 应用算法 4.1.1 得到 mutual refinement 

$$
\begin{array}{l} 5 - 5 \\ 5 - 5 \end{array} \searrow 2 5
$$

3. 形成 diagram 

![image](Imgaes/categorical-foundations-cute-layouts-paper/f37c36d6be520cc4080dc474921ae202fa7e7b8792fee6f9450248f6a31ed35f.jpg)


4. 解析 diagram，得到 

![image](Imgaes/categorical-foundations-cute-layouts-paper/5867b5a4106b1f562bf5b5868ef6ce46a629117dc825a51d2ba6a015073b44c9.jpg)


5. 复合 $f^'$ 与 $g^'$，得到 

![image](Imgaes/categorical-foundations-cute-layouts-paper/35bc8fa41c3f2d1377826a9de4897a9b71a9749f8238da972a17f217b8ac2673.jpg)


6. 计算编码的 layout

$$
L _ {g ^ {\prime} \circ f ^ {\prime}} = (5, 5): (2 0, 4).
$$

7. layout $L_{g'\circ f'}$ 在 `(5,5)` 上 coalesced，所以

$$
A ^ {c} \circ B = (5, 5): (2 0, 4).
$$

因此

$$
A \otimes B = ((2, 2), (5, 5)): ((1, 2), (2 0, 4)).
$$

## 附录 A

# Category 入门

全文自由使用 category 的语言；category 是抽象 morphism 及其 composition 概念的数学对象。本附录旨在简洁、友好地介绍 category 基础，尤其回答以下问题：

1. 什么是 category？

2. 什么是 functor？

能够自信地回答这些问题并想到相应示例的读者，将能理解本文最重要的概念和构造。希望学习 natural transformation、adjunction 和 (co)limit 等更高级范畴论概念的读者，可参阅 [13]。

## A.1 什么是 category？

先回答第一个问题。在给出定义前，考虑一个启发性示例。假设 X 和 Y 是集合。函数 $f:X\to Y$ 为每个元素 $x\in X$ 分配某个元素 $f(x)\in Y$。称 X 为 f 的 domain，Y 为 f 的 codomain。

示例 A.1.0.1。存在函数 $f:\mathbb{Z}\to\mathbb{Z}$，定义为

$$
f (x) = 2 x.
$$

示例 A.1.0.2。存在函数 $g:\mathbb{Z}\to\mathsf{Bool}$，其中 $\mathsf{Bool}=\{\mathbf{True},\mathbf{False}\}$，定义为

$$
g (x) = \left\{ \begin{array}{l l} \text { True } & x \text {   is   even }, \\ \text { False } & x \text {   is   odd }. \end{array} \right..
$$

如果 $f:X\to Y$ 和 $g:Y\to Z$ 是函数，则可以复合 f 与 g。二者的 composite 是函数 $g\circ f:X\to Z$，定义为

$$
(g \circ f) (x) = g (f (x)).
$$

示例 A.1.0.3。如果 f 和 g 是示例 A.1.0.1 与 A.1.0.2 中的函数，则 composite $g\circ f:\mathbb{Z}\to\mathsf{Bool}$ 定义为

$$
(g \circ f) (x) = \mathbf {T r u e}.
$$

函数 composition 满足两项基本性质。首先，composition 满足结合律：如果 f 与 g 可复合，g 与 h 可复合，则

$$
h \circ (g \circ f) = (h \circ g) \circ f.
$$

其次，每个集合 X 都有 identity function $\mathsf{id}_X:X\to X$，定义为

$$
\operatorname{id} _ {X} (x) = x.
$$

如果 $f:X\to Y$ 是任意函数，则在之前复合 $\mathsf{id}_X$ 或在之后复合 id<sub>Y</sub> 都不会改变函数 f：

$$
f \circ \mathrm{id} _ {X} = f = \mathrm{id} _ {Y} \circ f.
$$

在纯数学和应用数学中，经常会遇到某个 object 集合及其 object 之间的 morphism，它们具有与集合和函数相同的形式行为：morphism 可以按结合律复合，object 具有 identity morphism。集合之间的函数是原型示例，但 category 中的 object 不一定是集合，morphism 也不一定是函数。后文会看到许多这类示例。为了捕捉这种反复出现的结构，定义 category 概念。

## 定义 A.1.0.4。category C 包括

1. 一组 object：

$$
\mathsf {o b} (\mathbf {C}) = \{X, Y, Z, \dots \}.
$$

这些 object 可以是集合、tuple、数、向量空间、矩阵或其他数学结构，具体取决于 category C。

2. 这些 object 之间的一组 morphism：

$$
\operatorname{mor} (\mathbf {C}) = \{f, g, h, \dots \}.
$$

C 中每个 morphism $f:X\to Y$ 都有 domain X 和 codomain Y，它们是 C 中的 object。

3. composition 规则：如果 $f:X\to Y$ 和 $g:Y\to Z$ 是 C 中的 morphism，则存在 morphism

$$
g \circ f: X \to Z
$$

称为 f 与 g 的 composite。C 中 morphism 的 composition 满足结合律，即

$$
h \circ (g \circ f) = (h \circ g) \circ f,
$$

只要相应 composition 定义良好。

4. identity morphism：如果 X 是 C 中的 object，则存在 morphism

$$
\mathrm{id} _ {X}: X \to X
$$

称为 X 上的 identity morphism。如果 $f:X\to Y$ 是 C 中任意 morphism，则

$$
f \circ \mathrm{id} _ {X} = f = \mathrm{id} _ {Y} \circ f.
$$

下面查看 category 的一些重要示例，先从启发性示例开始。

示例 A.1.0.5。存在 category **Set**，其 object 是集合，morphism 是函数。morphism composition 由函数 composition 给出：

$$
(g \circ f) (x) = g (f (x))
$$

集合 X 上的 identity morphism 是 identity function

$$
\operatorname{id} _ {X} (x) = x.
$$

示例 A.1.0.6。存在 category **Vect**，其 object 是 $n\geq0$ 时的向量空间 $\mathbb{R}^n$，morphism 是矩阵。具体地，**Vect** 中的 morphism

$$
A: \mathbb {R} ^ {n} \to \mathbb {R} ^ {m}
$$

是一个 $m\times n$ 矩阵 A。**Vect** 中的 composition 由矩阵乘积给出：

$$
B \circ A = B A,
$$

$\mathbb{R}^n$ 上的 identity morphism 是 $n\times n$ 矩阵

$$
\mathsf {i d} _ {\mathbb {R} ^ {n}} = I _ {n} = \left[ \begin{array}{c c c c c} 1 & 0 & \dots & & 0 \\ 0 & 1 & & & \\ \vdots & & \ddots & & \vdots \\ & & & 1 & 0 \\ 0 & & \dots & 0 & 1 \end{array} \right].
$$

示例 A.1.0.7。存在 category **Div**，其 object 是整数 $a\geq1$；当 a 整除 b 时，存在唯一 morphism

$$
\mathsf {d i v} _ {a} ^ {b}: a \to b
$$

如果 a 整除 b 且 b 整除 c，则 a 整除 c，因此存在定义良好的 composition 规则

$$
\mathsf {d i v} _ {b} ^ {c} \circ \mathsf {d i v} _ {a} ^ {b} = \mathsf {d i v} _ {a} ^ {c},
$$

identity morphism

$$
\mathsf {i d} _ {a} = \mathsf {d i v} _ {a} ^ {a}
$$

存在，因为每个正整数 a 都整除自身。

除 category 定义外，还需要理解若干重要范畴论概念。例如，isomorphism 概念很重要，它推广了集合双射的概念。

定义 A.1.0.8。假设 C 是 category，$f:X\to Y$ 是 C 中的 morphism。如果 C 中存在 morphism $f^{-1}:Y\to X$，满足

1. $f ^ { - 1 } \circ f = \mathsf { i d } _ { X }$ , and 

2. $f \circ f ^ { - 1 } = \mathsf { i d } _ { Y } .$ 

就称 f 是 isomorphism。示例 A.1.0.9。在 category **Set** 中，isomorphism 就是双射：函数 $f:X\to Y$ 满足，对每个 $y\in Y$，存在唯一 $x\in X$ 使 $f(x)=y$。例如，函数 $f:\mathbb{Z}\to\mathbb{Z}$ 定义为

$$
f (x) = x + 1 0
$$

是双射，其逆 $f^{-1}:\mathbb{Z}\to\mathbb{Z}$ 定义为

$$
f ^ {- 1} (x) = x - 1 0.
$$

示例 A.1.0.10。在 category **Vect** 中，isomorphism 是可逆矩阵。例如，矩阵

$$
A = \left[ \begin{array}{c c} 3 & 2 \\ 1 & 1 \end{array} \right]
$$

可逆，其逆为

$$
A ^ {- 1} = \left[ \begin{array}{c c} 1 & - 2 \\ - 1 & 3 \end{array} \right]
$$

因为

$$
A ^ {- 1} A = \left[ \begin{array}{c c} 1 & 0 \\ 0 & 1 \end{array} \right] = A A ^ {- 1}
$$

示例 A.1.0.11。在 category **Div** 中，唯一的 isomorphism 是 identity morphism

$$
\mathsf {i d} _ {a} = \mathsf {d i v} _ {a} ^ {a}.
$$

这是因为，如果 a 整除 b 且 b 整除 a，则 $a=b$。

## A.2 什么是 functor？

接下来回答第二个问题。

定义 A.2.0.1。假设 C 和 D 是 category。functor $F:C\to D$ 包括：

1. 对 C 中每个 object X，D 中有一个 object FX；

2. 对 C 中每个 morphism $f:X\to Y$，D 中有一个 morphism

$$
F f: F X \to F Y
$$

，

并满足以下性质：

1. F 与 composition 相容：如果 f 和 g 是 C 中可复合的 morphism，则

$$
F (g \circ f) = F g \circ F f.
$$

2. F 与 identity 相容：如果 X 是 C 中的 object，则

$$
F \mathrm{id} _ {X} = \mathrm{id} _ {F X}.
$$

示例 A.2.0.2。存在 functor `F: Div → Set`，定义如下。对 object，F 定义为

$$
F a = [ 0, a ] = \{x \in \mathbb {R} \mid 0 \leq x \leq a \}.
$$

对 morphism，F 定义为

$$
F \mathsf {d i v} _ {a} ^ {b} (x) = \frac {b}{a} \cdot x.
$$

下面验证 F 是 functor。

1. F 与 composition 相容：如果 a 整除 b，且 b 整除 c，则

$$
(F \mathsf {d i v} _ {b} ^ {c} \circ F \mathsf {d i v} _ {a} ^ {b}) (x) = F \mathsf {d i v} _ {b} ^ {c} (F \mathsf {d i v} _ {a} ^ {b} (x)) = \frac {c}{b} \cdot (\frac {b}{a} \cdot x) = \frac {c}{a} \cdot x = F \mathsf {d i v} _ {a} ^ {c} (x).
$$

2. F 与 identity 相容：如果 $a\geq1$，则

$$
F \mathsf {i d} _ {a} (x) = F \mathsf {d i v} _ {a} ^ {a} (x) = \frac {a}{a} \cdot x = \mathsf {i d} _ {F a} (x).
$$

## 参考文献



[1] Somashekaracharya G. Bhaskaracharya et al. Modeling Layout Abstractions Using Integer Set Relations. 2025. arXiv: 2511.10374. url: https://arxiv.org/abs/2511.10374. 





[2] Uday Bondhugula et al. “A Practical Automatic Polyhedral Parallelizer and Locality Optimizer”. In: Proceedings of the 29th ACM SIGPLAN Conference on Programming Language Design and Implementation. PLDI ’08. ACM, 2008. doi: 10.1145/1375581.1375595. 





[3] Cris Cecka, Vijay Thakkar, and Tejash Shah. CUTLASS: Principled Abstractions for Handling Multidimensional Data Through Tensors and Spatial Microkernels. NVIDIA Technical Blog. 2025. url: https://developer.nvidia.com/blog/cutlass-principled-abstractions-forhandling-multidimensional-data-through-tensors-and-spatial-microkernels/. 





[4] Zhaodong Chen et al. “EVT: Accelerating Deep Learning Training with Epilogue Visitor Tree”. In: Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS), Volume 3. ACM, 2024, pp. 301–316. doi: 10. 1145/3620666.3651369. 





[5] NVIDIA Corporation. CuTe — NVIDIA CuTe Documentation. 2024. url: https://docs. nvidia.com/cutlass/media/docs/cpp/cute/index.html. 





[6] NVIDIA Corporation. CuTe DSL — NVIDIA CUTLASS Documentation. 2025. url: https: //docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl.html. 





[7] Tri Dao. “FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning”. In: International Conference on Learning Representations (ICLR). 2024. 





[8] Oliver J. Sharp David F. Bacon Susan L. Graham. “Compiler Transformations for High-Performance Computing”. In: ACM Computing Surveys 26.4 (Dec. 1994), pp. 345–420. doi: 10.1145/197405.197406. 





[9] Tobias Grosser, Armin Gr¨oßlinger, and Christian Lengauer. “Polly—Performing Polyhedral Optimizations on a Low-Level Intermediate Representation”. In: Parallel Processing Letters 22.4 (2012). 





[10] Wentao Guo et al. SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations. 2025. arXiv: 2512.14080 [cs.LG]. url: https://arxiv.org/abs/2512.14080. 





[11] Yong-Jae Ju and Henry Dietz. “Reduction of Cache Coherence Overhead by Compiler Data Layout and Loop Transformation”. In: Languages and Compilers for Parallel Computing. Ed. by Utpal Banerjee et al. Berlin, Heidelberg: Springer, 1992. doi: 10.1007/BFb0038675. 





[12] Tamara G. Kolda and Brett W. Bader. “Tensor Decompositions and Applications”. In: SIAM Review 51.3 (2009), pp. 455–500. doi: 10.1137/07070111X. url: https://doi.org/10.1137/ 07070111X. 





[13] Saunders Mac Lane. Categories for the Working Mathematician. 2nd ed. Vol. 5. Graduate Texts in Mathematics. Springer, 1998. isbn: 978-0-387-98403-0. 





[14] OpenAI. Linear Layouts in Triton. 2025. url: https://github.com/triton-lang/triton/ blob/main/include/triton/Tools/LinearLayout.h. 





[15] Easwaran Raman, Robert Hundt, and Sandya Mannarswamy. “Structure Layout Optimization for Multithreaded Programs”. In: Proceedings of the International Symposium on Code Generation and Optimization. CGO ’07. Washington, DC, USA: IEEE Computer Society, 2007, pp. 271–282. doi: 10.1109/CGO.2007.18. 





[16] Jay Shah. A Note on the Algebra of CuTe Layouts. Tech. rep. Colfax Research, Jan. 2024. url: https://research.colfax-intl.com/wp-content/uploads/2024/01/layout_algebra.pdf. 





[17] Jay Shah, Paul VanKoughnett, and Ryo Asai. CUTLASS Tutorial: GEMM with Thread Block Clusters on NVIDIA Blackwell GPUs. Colfax Research. 2025. url: https://research.colfaxintl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwellgpus/. 





[18] Jay Shah, Paul VanKoughnett, and Ryo Asai. CUTLASS Tutorial: Persistent Kernels and Stream-K. Colfax Research. 2024. url: https://research.colfax-intl.com/cutlass-tutorialpersistent-kernels-and-stream-k/. 





[19] Jay Shah, Paul VanKoughnett, and Ryo Asai. CUTLASS Tutorial: Sub-byte GEMM on NVIDIA Blackwell GPUs. Colfax Research. 2025. url: https://research.colfax-intl.com/cutlasstutorial-sub-byte-gemm-on-nvidia-blackwell-gpus/. 





[20] Jay Shah, Paul VanKoughnett, and Ryo Asai. CUTLASS Tutorial: Writing GEMM Kernels Using Tensor Memory For NVIDIA Blackwell GPUs. Colfax Research. 2025. url: https : //research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensormemory-for-nvidia-blackwell-gpus/. 





[21] Jay Shah et al. “FlashAttention-3: Fast and Accurate Attention with Asynchrony and Lowprecision”. In: Advances in Neural Information Processing Systems (NeurIPS). 2024. 





[22] Kamal Sharma et al. “Data Layout Optimization for Portable Performance”. In: European Conference on Parallel Processing. Euro-Par 2015. Berlin, Heidelberg: Springer, 2015. doi: 10.1007/978-3-662-48096-0\_20. 





[23] Yang Shi et al. “Tensor Contractions with Extended BLAS Kernels on CPU and GPU”. In: arXiv preprint (2016). eprint: arXiv:1606.05696. url: https://arxiv.org/abs/1606.05696. 





[24] Brandon Sun et al. Achieve CUTLASS C++ Performance with Python APIs Using CuTe DSL. NVIDIA Technical Blog. Nov. 2025. url: https://developer.nvidia.com/blog/achievecutlass-c-performance-with-python-apis-using-cute-dsl/. 





[25] Vijay Thakkar et al. CUTLASS 3.x: Orthogonal, Reusable, and Composable Abstractions for GEMM Kernel Design. NVIDIA Technical Blog. 2025. url: https://developer.nvidia.com/ blog / cutlass - 3 - x - orthogonal - reusable - and - composable - abstractions - for - gemm - kernel-design/. 





[26] Arun Thangamani, Vincent Loechner, and St´ephane Genaud. “A Survey of General-purpose Polyhedral Compilers”. In: ACM Transactions on Architecture and Code Optimization 21.4 (2024), pp. 1–26. doi: 10.1145/3674735. 





[27] Nicolas Vasilache et al. “Tensor Comprehensions: Framework-Agnostic High-Performance Machine Learning Abstractions”. In: arXiv preprint arXiv:1802.04730 (2018). Facebook AI Research Technical Report. 





[28] Sven Verdoolaege. “isl: An Integer Set Library for the Polyhedral Model”. In: Mathematical Software – ICMS 2010. Ed. by Komei Fukuda et al. Vol. 6327. Lecture Notes in Computer Science. Berlin, Heidelberg: Springer, 2010. doi: 10.1007/978-3-642-15582-6_49. 





[29] Sven Verdoolaege. Presburger Formulas and Polyhedral Compilation. Technical Report / Tutorial. Version v0.02-13-g53eb23d, August 21, 2021. Polly Labs and KU Leuven, 2021. 





[30] Zhiying Xu et al. “ALT: Breaking the Wall between Data Layout and Loop Optimizations for Deep Learning Compilation”. In: Proceedings of the Eighteenth European Conference on Computer Systems. EuroSys ’23. New York, NY, USA: Association for Computing Machinery, 2023. doi: 10.1145/3552326.3587440. 





[31] Tuowen Zhao et al. “Polyhedral Specification and Code Generation of Sparse Tensor Contraction with Co-Iteration”. In: ACM Transactions on Architecture and Code Optimization 20.1 (Dec. 2022). Article 16, pp. 1–26. doi: 10.1145/3566054. 





[32] Keren Zhou et al. “Linear Layouts: Robust Code Generation of Eficient Tensor Computation Using F2”. In: Proceedings of the 31st ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1. ASPLOS ’26. Pittsburgh, PA, USA: ACM, Mar. 2026, pp. 1–18. doi: 10.1145/3760250.3762221. 

