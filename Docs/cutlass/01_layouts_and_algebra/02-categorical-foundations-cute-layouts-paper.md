# Categorical Foundations for CuTe Layouts

Jack Carlisle Jay Shah Reuben Stern Paul VanKoughnett 

Colfax Research 

research@colfax-intl.com 

January 2026 

## Abstract

NVIDIA’s CUTLASS library provides a robust and expressive set of methods for describing and manipulating multi-dimensional tensor data on the GPU. These methods are conceptually grounded in the abstract notion of a CuTe layout and a rich algebra of such layouts, including operations such as composition, logical product, and logical division. In this paper, we present a categorical framework for understanding this layout algebra by focusing on a naturally occurring class of tractable layouts. To this end, we define two categories Tuple and Nest whose morphisms give rise to layouts. We define a suite of operations on morphisms in these categories and prove their compatibility with the corresponding layout operations. Moreover, we give a complete characterization of the layouts which arise from our construction. Finally, we provide a Python implementation of our categorical constructions, along with tests that demonstrate alignment with CUTLASS behavior. This implementation can be found at our git repository https://github.com/ColfaxResearch/layout-categories. 

## Contents

1 Introduction 3
1.1 Summary of main results 6
1.2 Organization 8
1.3 Related work 9
1.4 Implementation 11
1.5 Notation 17
2 Layouts and their algebra 18
2.1 Flat Layouts 18
2.1.1 Tuples 18
2.1.2 Basic definitions 19
2.1.3 Basic operations 27
2.1.4 Flat coalesce 36
2.1.5 Compact flat layouts 41
2.1.6 Complements 44
2.1.7 Further operations 54
2.1.8 Tractable flat layouts 56
2.2 Nested Tuples 58
2.2.1 Profiles 58
2.2.2 Basic definitions 60
2.2.3 Substitution 63
2.2.4 Refinement 64
2.3 Layouts 67
2.3.1 Basic definitions 67
2.3.2 Basic operations 70
2.3.3 Coalesce 72
2.3.4 Relative coalesce 74
2.3.5 Compact layouts 77
2.3.6 Complements 78
2.3.7 Composition 80
2.3.8 Logical division 81
2.3.9 Logical product 84
2.3.10 Tractable layouts 85 

3 Categories of layouts 87
3.1 The category Tuple 87
3.1.1 Basic definitions 87
3.1.2 From tuple morphisms to flat layouts 91
3.1.3 Examples 99
3.1.4 Realization of tuple morphisms 103
3.1.5 Operations on tuple morphisms 106
3.2 The category Nest 124
3.2.1 Basic definitions 124
3.2.2 From nested tuple morphisms to layouts 125
3.2.3 Examples 128
3.2.4 Realization of nested tuple morphisms 130
3.2.5 Refinements 131
3.2.6 Operations on nested tuple morphisms 139
4 Computations 147
4.1 Composition of tractable layouts 147
4.1.1 Mutual refinements 148
4.1.2 From mutual refinements to composable morphisms 152
4.1.3 The composition algorithm 153
4.1.4 Examples 154
4.1.5 More general compositions 160
4.1.6 Admissibility for composition 161
4.2 Logical division and logical product 163
4.2.1 Logical division examples 163
4.2.2 Logical product examples 164
A An introduction to categories 166
A.1 What is a category? 166
A.2 What is a functor? 169 

## Chapter 1

## Introduction

In modern computing, particularly in GPU programming, performance depends critically on how multi-dimensional data is stored and accessed in memory. While most data that we care about—such as images, videos, and tensors in machine learning—are inherently multi-dimensional, a computer’s memory is fundamentally one-dimensional. This means that when we want to load, store, or otherwise manipulate data, we need to map its multi-dimensional logical coordinates to one-dimensional physical coordinates. This mapping, known as a layout, is essential for reading from and writing to memory correctly and eficiently. Moreover, with respect to the GPU’s SIMT execution model, layouts are used to describe and manipulate partitionings of threads over data. This is important to ensure optimized memory access patterns and correct invocation of specialized hardware instructions such as those used to target tensor cores. 

As a motivating example, suppose we want to store the 4 × 8 matrix 

$$
A = \left[ \begin{array}{c c c c c c c c} 1 2. 4 7 & 8 7. 2 1 & 3 4. 0 8 & 5 6. 9 3 & 4 5. 6 5 & 9. 1 7 & 7 3. 0 2 & 2 1. 3 9 \\ 6 4. 8 8 & 3 0. 4 1 & 1. 7 2 & 8 8. 0 4 & 9 2. 5 5 & 1 7. 0 6 & 5 0. 9 1 & 6 8. 7 7 \\ 3. 3 3 & 7 7. 1 9 & 6 1. 5 8 & 2 9. 4 6 & 1 5. 8 2 & 8 0. 7 5 & 4 4. 6 2 & 3 9. 2 8 \\ 9 1. 4 0 & 2 6. 1 2 & 6. 9 7 & 5 3. 0 3 & 5 8. 6 6 & 3 3. 7 9 & 1 1. 2 0 & 7 0. 5 5 \end{array} \right]
$$

in memory. In order to do so, we need to specify a memory address for each entry of A. We do this by choosing some address for the (0, 0)th entry of A, and specifying an ofset for each other entry of A. One common choice is the row-major layout 

<table><tr><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td></tr><tr><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td><td>14</td><td>15</td></tr><tr><td>16</td><td>17</td><td>18</td><td>19</td><td>20</td><td>21</td><td>22</td><td>23</td></tr><tr><td>24</td><td>25</td><td>26</td><td>27</td><td>28</td><td>29</td><td>30</td><td>31</td></tr></table>

The notation $L ^ { \mathsf { r o w } } = ( 4 , 8 ) : ( 8 , 1 )$ indicates that the ofset of the (i, j)th entry of our matrix is 

$$
(i, j) \cdot (8, 1) = 8 i + j.
$$

Another common choice is the column-major layout 

<table><tr><td>0</td><td>4</td><td>8</td><td>12</td><td>16</td><td>20</td><td>24</td><td>28</td></tr><tr><td>1</td><td>5</td><td>9</td><td>13</td><td>17</td><td>21</td><td>25</td><td>29</td></tr><tr><td>2</td><td>6</td><td>10</td><td>14</td><td>18</td><td>22</td><td>26</td><td>30</td></tr><tr><td>3</td><td>7</td><td>11</td><td>15</td><td>19</td><td>23</td><td>27</td><td>31</td></tr></table>

Again, the notation $L ^ { \mathsf { c o l } } = ( 4 , 8 ) : ( 1 , 4 )$ indicates that the ofset of the $( i , j ) \mathrm { t h }$ entry of our matrix is 

$$
(i, j) \cdot (1, 4) = i + 4 j.
$$

These layouts are extremely useful, but do not sufice for all purposes. For example, in high-performance computing, one often computes matrix products AB by 

1. dividing the operand matrices A and B into tiles, 

2. computing matrix products of the various tiles, and 

3. combining these partial results to obtain the full result AB. 

For instance, we could divide our $4 \times 8$ matrix A into $2 \times 2$ tiles, as depicted below. 

$$
A = \left[ \begin{array}{c c c c c} \left[ \begin{array}{c c} 1 2. 4 7 & 8 7. 2 1 \\ 6 4. 8 8 & 3 0. 4 1 \end{array} \right] & \left[ \begin{array}{c c} 3 4. 0 8 & 5 6. 9 3 \\ 1. 7 2 & 8 8. 0 4 \end{array} \right] & \left[ \begin{array}{c c} 4 5. 6 5 & 9. 1 7 \\ 9 2. 5 5 & 1 7. 0 6 \end{array} \right] & \left[ \begin{array}{c c} 7 3. 0 2 & 2 1. 3 9 \\ 5 0. 9 1 & 6 8. 7 7 \end{array} \right] \\ \left[ \begin{array}{c c} 3. 3 3 & 7 7. 1 9 \\ 9 1. 4 0 & 2 6. 1 2 \end{array} \right] & \left[ \begin{array}{c c} 6 1. 5 8 & 2 9. 4 6 \\ 6. 9 7 & 5 3. 0 3 \end{array} \right] & \left[ \begin{array}{c c} 1 5. 8 2 & 8 0. 7 5 \\ 5 8. 6 6 & 3 3. 7 9 \end{array} \right] & \left[ \begin{array}{c c} 4 4. 6 2 & 3 9. 2 8 \\ 1 1. 2 0 & 7 0. 5 5 \end{array} \right] \end{array} \right]
$$

Suppose now that we wanted to slice out individual tiles of $A ,$ which we assume is laid out in columnmajor format in memory. To do this, one could manually compute ofsets as follows: for the $( i , j )$ th tile, the ofset to index into the top-left entry of the tile is given by $2 i + 8 j$ . On the other hand, to better organize this computation, we could use the interleaved layout of tiles 

<table><tr><td>0</td><td>2</td><td>8</td><td>10</td><td>16</td><td>18</td><td>24</td><td>26</td></tr><tr><td>1</td><td>3</td><td>9</td><td>11</td><td>17</td><td>19</td><td>25</td><td>27</td></tr><tr><td>4</td><td>6</td><td>12</td><td>14</td><td>20</td><td>22</td><td>28</td><td>30</td></tr><tr><td>5</td><td>7</td><td>13</td><td>15</td><td>21</td><td>23</td><td>29</td><td>31</td></tr></table>

where the columns are given by tiles of A and the rows are given by coordinates within the tile shape. Here, we use colexicographic ordering to linearly enumerate tiles and coordinates within tiles, hence the top-level shape (4, 8) of the layout $L ^ { \mathrm { t i l e d } }$ 

However, note that the interleaving pattern shown for $L ^ { \mathrm { t i l e d } }$ means that it can’t be expressed as a layout $( 4 , 8 ) : ( a , b )$ for any strides a, b. Instead, we can factor the modes of the shape $( 4 , 8 )$ and define 

$$
L ^ {\text { tiled }} = ((2, 2), (2, 4)): ((1, 4), (2, 8)).
$$

The prior ofset calculation $2 i + 8 j$ then appears through evaluating $L ^ { \mathrm { t i l e d } }$ on the coordinate $( 0 , ( i , j ) )$ , and the tile layout itself is given by the first mode. Thus, after endowing A with the layout $L ^ { \mathrm { t i l e d } }$ to form $A ^ { \mathrm { t i l e d } }$ , we can obtain the $( i , j ) \mathrm { t h }$ tile of A as the slice 

$$
A _ {i, j} = A ^ {\text { tiled }} (\_, (i, j)).
$$

A key idea developed in CUTLASS is that useful but more complex auxiliary layouts such as $L ^ { 1 }$ tiled may be systematically deduced from simpler layouts via certain fundamental operations. In the case of $L ^ { \mathrm { t i l e d } }$ , the operation in question is called logical division. If we write 

$$
T = (2, 2): (1, 4) = \quad \begin{array}{c c} \hline 0 & 4 \\ \hline 1 & 5 \\ \hline \end{array}
$$

for the tile layout, then $L ^ { \mathrm { t i l e d } }$ is the logical division 

$$
L ^ {\text { tiled }} = L ^ {\text { col }} \oslash T
$$

as depicted below. 

<table><tr><td>0</td><td>4</td><td>8</td><td>12</td><td>16</td><td>20</td><td>24</td><td>28</td></tr><tr><td>1</td><td>5</td><td>9</td><td>13</td><td>17</td><td>21</td><td>25</td><td>29</td></tr><tr><td>2</td><td>6</td><td>10</td><td>14</td><td>18</td><td>22</td><td>26</td><td>30</td></tr><tr><td>3</td><td>7</td><td>11</td><td>15</td><td>19</td><td>23</td><td>27</td><td>31</td></tr></table>

$$
T = \begin{array}{c c} \hline 0 & 4 \\ \hline 1 & 5 \\ \hline \end{array}
$$

<table><tr><td>0</td><td>2</td><td>8</td><td>10</td><td>16</td><td>18</td><td>24</td><td>26</td></tr><tr><td>1</td><td>3</td><td>9</td><td>11</td><td>17</td><td>19</td><td>25</td><td>27</td></tr><tr><td>4</td><td>6</td><td>12</td><td>14</td><td>20</td><td>22</td><td>28</td><td>30</td></tr><tr><td>5</td><td>7</td><td>13</td><td>15</td><td>21</td><td>23</td><td>29</td><td>31</td></tr></table>

In addition to logical division, other fundamental layout operations include logical products, complements, and most importantly, composition. These layout operations are the backbone of CUTLASS, and a deep understanding of their behavior is helpful for writing correct and highly performant code. However, the definitions and constructions of these operations are fairly subtle. For example, the composition $B \circ A$ of layouts A and B is well-defined only if A and B satisfy certain divisibility constraints, which CUTLASS checks under the hood. In particular, it is not always obvious when two layouts are composable, or how to interpret their composition. 

## 1.1 Summary of main results

The main idea of this work is that we can develop an intuitive and powerful mathematical framework for working with layouts by restricting our attention to tractable layouts, whose entries satisfy a simple divisibility condition (see Definition 2.3.10.1). Tractable layouts include almost all layouts one encounters in practice, such as 

• row-major and column-major layouts, which are ubiquitous, 

• compact layouts, which store data in consecutive memory addresses, 

• projections, which broadcast multiple copies of data, and 

• dilations, which enable padded loads and stores. 

If L is a tractable layout, then we can represent L with a diagram. For example, the layouts $L ^ { \mathsf { r o w } }$ $L ^ { \mathsf { c o l } }$ , and $L ^ { \mathrm { t i l e d } }$ are represented by the following diagrams. 

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


These diagrams may be interpreted as morphisms in a category. This allows us to leverage the power of category theory to describe layouts and their operations.<sup>1</sup> 

More precisely, we define a category Nest whose objects are nested tuples of positive integers, and whose morphisms $f : S  T$ correspond to diagrams such as those above (see Definition 3.1.1.13 and Definition 3.2.1.1 for details). If L is a non-degenerate tractable layout (see Definition 2.3.1.24), then there is an essentially unique Nest-morphism f which encodes $L ,$ as illustrated by the following correspondence theorem. 

Theorem A. (see 3.2.2.15) There is a one-to-one correspondence 

$$
\left\{ \begin{array}{l} N o n - d e g e n e r a t e \\ t r a c t a b l e l a y o u t s \end{array} \right\} \longleftrightarrow \left\{ \begin{array}{l} N o n - d e g e n e r a t e \\ \mathbf {N e s t - m o r p h i s m s} \\ o f s t a n d a r d f o r m \end{array} \right\}
$$

Layout operations such as composition, logical division, and logical products may be interpreted naturally in the category Nest. If 

$$
S \xrightarrow {f} T \xrightarrow {g} U
$$

are Nest-morphisms, then we may form the composite 

$$
S \xrightarrow {g \circ f} U
$$

by pasting the associated diagrams together. For example, 

![image](Imgaes/categorical-foundations-cute-layouts-paper/611e02cef2bdb0dead4b446450fcb03fb9f377e97e8d8a1fa9359fddf9a7c157.jpg)


We prove that composition in Nest is compatible with layout composition. 

Theorem B. (see 3.2.6.21) If f and g are non-degenerate composable Nest-morphisms, then 

$$
L _ {g \circ f} = L _ {g} \circ L _ {f}.
$$

We can coalesce a Nest-morphism f by collapsing adjacent arrows. For example, 

![image](Imgaes/categorical-foundations-cute-layouts-paper/cf51ba67b40ca70ba72e4a8bfbee68e6e9ce98aec9066a216fb4532a3f4cd645.jpg)


We prove that this operation is compatible with layout coalesce. 

Theorem C. (see 3.2.6.13) If f is a Nest-morphism, then 

$$
L _ {\text { coal } (f)} = \text { coal } (L _ {f}).
$$

The complement of a Nest-morphism f is the inclusion of the entries not hit by f. For example, 

![image](Imgaes/categorical-foundations-cute-layouts-paper/78e93526fb7edc239f56f9507cbd3da522ef53e91b1518c5c331e3574ad40433.jpg)


We prove that complements in Nest are compatible with layout complements. 

Theorem D. (see 3.2.6.20) $I f f : S \to T$ is an injective Nest-morphism and $N = { \mathsf { s i z e } } ( T )$ , then 

$$
\operatorname{coal} \left(L _ {f ^ {c}}\right) = \operatorname{comp} \left(L _ {f}, N\right).
$$

We define divisibility of Nest-morphisms, and a logical division operation 

$$
f, g \mapsto f \oslash g
$$

when g divides f. For example, 

![image](Imgaes/categorical-foundations-cute-layouts-paper/dae1c082f224dcc785d039b35bad77519606082d3fc53366b6270ad10e6895a7.jpg)


We prove that logical division in Nest is compatible with logical division of layouts. 

Theorem E. (see 3.2.6.26) If f and g are non-degenerate Nest-morphisms and $g$ divides $f ,$ then 

$$
\operatorname{coal} \left(L _ {f \oslash g}\right) = \operatorname{coal} \left(L _ {f} \oslash L _ {g}\right).
$$

We define product admissibility of Nest-morphisms, and a logical product operation 

$$
f, g \mapsto f \otimes g
$$

when f and g are product admissible. For example, 

![image](Imgaes/categorical-foundations-cute-layouts-paper/628e4334067a399c610f27223441883f009552d00be6191aa3613aa7eeb2d0bb.jpg)


We prove that the logical products in Nest are compatible with logical products of layouts. 

Theorem F. (see 3.2.6.31) If f and g are non-degenerate Nest-morphisms and $f$ and $g$ are product admissible, then 

$$
L _ {f \otimes g} = L _ {f} \otimes L _ {g}.
$$

In Chapter 4, we illustrate how our new framework may be used to compute important layout operations such as composition, logical division, and logical products. In particular, we present an algorithm (Algorithm 4.1.3) for computing the composition $B \circ A$ of tractable layouts A and B. Eliding details, the basic idea of our algorithm is that if we want to compute the composition $B \circ A$ , we can represent A and B by suitably chosen Nest-morphisms $f$ and $^ { g , }$ compose these morphisms to form $g \circ f .$ , then take the encoded layout to obtain 

$$
B \circ A = L _ {g \circ f}.
$$

We illustrate this algorithm with many examples. 

## 1.2 Organization

The current work is organized as follows. 

In section 1.4, we provide details regarding the cute implementation of layouts. We provide a Python implementation of the category Nest in the form of a module tract, and illustrate the compatibility of tract with cute. Our Python implementation may be found at our git repository https://github.com/ColfaxResearch/layout-categories. 

Chapter 2 serves as a comprehensive reference for layouts and their algebra. It provides rigorous definitions of layouts and the operations they support, and establishes the fundamental properties of these operations. This chapter is replete with examples, and may be of use to the working programmer. 

In Chapter 3, we present a new mathematical framework for working with tractable layouts. In particular, we connect layouts and their algebra to the theory of categories and operads. The content of this chapter is of independent mathematical interest. It is also of practical value, as it provides a new framework for visualizing layouts and computing their various operations. 

In Chapter 4, we provide an algorithm for computing the composite of tractable layouts A and B using the framework developed in Chapter 3. We illustrate the composition algorithm with many examples. 

## 1.3 Related work

While the current work is theoretical in nature, it is motivated by practical applications in GPU programming, most notably CUTLASS. We emphasize that the theory developed here is implementationagnostic: it is independent of the particular programming language or runtime system used to realize layouts in practice. Nevertheless, certain practical considerations arise when working with concrete implementations. For instance, CUTLASS distinguishes between compile-time constants (static variables) and runtime values (dynamic variables). This information enables compiler optimizations during code generation. Such implementation-specific details, while important for performance, lie outside the scope of our mathematical framework. Further discussion of this can be found in the CuTe documentation [5]. 

The mathematical framework we develop for layouts draws connections to several areas of computer science and mathematics. We briefly review relevant work on GPU programming and adjacent areas to provide a greater context for our contributions. 

• Applications of CUTLASS. State-of-the-art applications of CUTLASS include FlashAttention [7, 21], EVT [4], and SonicMoE [10]. For readers seeking a deeper understanding of CUTLASS and CuTe in practice, we recommend the comprehensive tutorial series from NVIDIA [3, 24, 25] and Colfax Research [18, 20, 17, 19] on GPU programming with these libraries. 

• Data layout optimization Data layout optimization techniques seek to improve cache locality and memory access patterns by carefully considering how tensors are stored in memory [30, 8, 15, 11], [22]. Choosing eficient memory storage and access patterns is crucial for GPU performance, where memory bandwidth is often a bottleneck. 

• Modern layout systems Layout systems such as CuTe [5, 6, 16] and Triton Linear Layouts [14, 32] have become industry standards for managing memory storage and access in tensor computations. Triton linear layouts are based on $\mathbb { F } _ { 2 } .$ -linear algebra, and inheret compositional structure from the composition of F -linear operators. These are also naturally compatible with layout swizzles, which can generally not be represented as a CuTe layout. On the other hand, these layouts are not as expressive as CuTe layouts since they are required to have size and cosize equal to a power of 2, and can not express transformations such as scaling by a non power-of-two integer. Recently, it was shown that both of these layout systems may be expressed in terms of integer set relations [1]. This provides a common ground for working with CuTe and Triton linear layouts, as well as more general layouts, such as those with non-rectangular shapes. 

• Polyhedral compilation The polyhedral model [28], [29], [26] provides a mathematical frame work for analyzing and transforming loop nests with afine bounds and array accesses. The primary abstraction of this model is the representation of an iteration space as the collection of integer points in some polyhedron. This formalism allows for complex loop transformations that preserve program semantics while optimizing for locality and parallelism. Tools such as Pluto [2], Polly [9], and Tensor Comprehensions [27] leverage polyhedral techniques to automatically generate optimized code. 

• Tensor contraction/decomposition Tensor contractions [23, 31, 12] generalize matrix multiplication to higher-rank tensors, and are ubiquitous in machine learning and scientific computing. The eficient implementation of tensor contractions relies on optimal choices of contraction order and intermediate tensor layouts. 

## 1.4 Implementation

In this section, we illustrate how to work with layouts in NVIDIA’s CuTe DSL, which we denote as cute. We provide an implementation of our categorical framework in the form of a Python module tract in our git repository https://github.com/ColfaxResearch/layout-categories. Here, we show the compatibility of cute and tract. 

1. Constructing tuples and nested tuples: We construct tuples and nested tuples in Python as follows. 

```txt
1 S = (2,2,2)
2 T = ((2,2),(5,5))
3 U = ((2,2),4,(9,(3,3))) 
```

Note that if we want to construct a tuple of length 1, we must include a comma following the tuple’s entry. For example, 

```python
1 S = (10,)
2 T = (10) 
```

returns 

```txt
1 S = (10,)
2 T = 10 
```

2. Constucting layouts and morphisms: We construct a layout 

$$
L = S: D
$$

in cute as follows. 

```txt
L = cute.make_layout(shape=S, stride=D) 
```

For example, 

```python
A = cute.make_layout(shape=((4,4),4), stride=((16,1),4))
B = cute.make_layout(shape=(8,64), stride=(64,1))
C = cute.make_layout(shape=100, stride=2) 
```

returns 

```matlab
A = ((4,4),4):(16,1),4)
B = (8,64):(64,1)
C = 100:2 
```

We construct a nested tuple morphism 

$$
S \xrightarrow [ \alpha ]{f} T
$$

in tract as follows. 

```python
f = tract.make_morphism(domain=S, codomain=T, map_=alpha) 
```

For example, 

```txt
f = tract.make_morphism(domain=(4,4), codomain=(4,2,4), map_=(1,3))
g = tract.make_morphism(domain=(2,2,2,2), codomain=(2,2,2,2), map_=(1,0,4,2))
h = tract.make_morphism(domain=(16,(4,4),(4,4)), codomain=(16,4,4), map_=(1,2,0,3,0)) 
```

returns 

```txt
f = (4,4)--(1,3)-->(4,2,4)
g = (2,2,2,2)--(1,0,4,2)-->(2,2,2,2)
h = (16,(4,4),(4,4))--(1,2,0,3,0)-->(16,4,4) 
```

Note that we use the symbol 0 rather than ∗ when specifying maps in tract. 

3. Translating between tractable layouts and morphisms: If L is a layout, we can check if L is tractable with 

```javascript
tract.is_tractable(L) 
```

For example, 

```txt
A = cute.make_layout(shape=(2,2,2), stride=(1,2,4))
B = cute.make_layout(shape=(2,2,2), stride=(1,7,4))
A_is_tractable = tract.is_tractable(A)
B_is_tractable = tract.is_tractable(B) 
```

returns 

```txt
A = (2,2,2):(1,2,4)
B = (2,2,2):(1,7,4)
A_is_tractable = True
B_is_tractable = False 
```

If L is a tractable layout, then we can construct the standard representation $f _ { L }$ with 

```txt
tract.compute_morphism(L) 
```

For example, 

```python
L = cute.make_layout(shape=(2,2,2), stride=(1,2,4))
f_L = tract.compute_morphism(L) 
```

returns 

```txt
L = (2, 2, 2) : (1, 2, 4)
f_L = (2, 2, 2) -- (1, 2, 3) --> (2, 2, 2) 
```

If f is a nested tuple morphism, we can construct the layout $L _ { f }$ encoded by f with 

```txt
tract.compute_layout(f) 
```

For example, 

```python
f = tract.make_morphism(domain=((5,5),8), codomain=(5,8,5), map_=(1,3,2))
L_f = tract.compute_layout(f) 
```

## returns

```txt
f = ((5,5),8)--(1,3,2)-->(5,8,5)
L_f = ((5,5),8):(1,40),5) 
```

4. Composition: When defined, this operation produces a layout B ◦ A from a pair of layouts A and B. See Definition 2.3.7.1 for a precise definition. We can compute the composition B ◦ A in cute with 

```javascript
cute.composition(B,A) 
```

For example, running 

```python
A = cute.make_layout(shape=((4,4),4), stride=((16,1),4))
B = cute.make_layout(shape=(8,64), stride=(64,1))
B_o_A = cute.composition(B,A) 
```

returns 

```txt
A = ((4, 4), 4): ((16, 1), 4)
B = (8, 64): (64, 1)
B_o_A = ((4, 4), (2, 2)): ((2, 64), (256, 1)) 
```

If f and g are composable nested tuple morphisms, we can compute the composition g ◦ f in tract with 

```javascript
tract.compose(f,g) 
```

For example, 

```python
f = tract.make_morphism(domain=((2,2),(2,2)), codomain=((2,2,2),(2,2,2)), map_=(3,2,6,5))
g = tract.make_morphism(domain=((2,2,2),(2,2,2)), codomain=(2,2,2,2), map_=(1,0,2,0,3,4))
g_o_f = tract.compose(f,g) 
```

returns 

```txt
f = ((2,2),(2,2))--(3,2,6,5)-->(2,2,2),(2,2,2))
g = ((2,2,2),(2,2,2))--(1,0,2,0,3,4)-->(2,2,2,2)
g_o_f = ((2,2),(2,2))--(2,0,4,3)-->(2,2,2,2) 
```

5. Coalesce: This operation produces a layout coal(A) from a layout A. See Definition 2.3.3.1 for details. We can compute coal(A) in cute with 

```txt
cute.coalesce(A) 
```

For example, 

```txt
A = cute.make_layout(shape = ((2,2), (2,2), (5,5)), stride = ((1,2), (16,32), (64,640)))
coal_A = cute.coalesce(A) 
```

returns 

```txt
A = ((2,2), (2,2), (5,5)): ((1,2), (16,32), (64,640))
coal_A = (4,20,5): (1,16,640) 
```

There is also a relative coalesce operation A 7→ coal(A, S), which receives as input an additional nested tuple S which is refined by the shape of A. See Definition 2.3.4.7 for details. We can compute coal(A, S) in cute with 

```txt
A = cute.make_layout(shape = ((2,2),(3,3),(5,5)), stride = ((1,2),(4,12),(36,180)))
S = ((2,2),9,25)
coal_A_over_S = cute.coalesce(A,target_profile=S) 
```

returns 

```python
A = ((2,2), (3,3), (5,5)): ((1,2), (4,12), (36,180))
S = ((2,2), 9, 25)
coal_A_over_S = ((2,2), 9, 25): ((1,2), 4, 36) 
```

If f is a nested tuple morphism, we may form coal(f). See Definition 3.2.6.11 for details. We compute coal(f) in tract with 

```txt
tract.coalesce(f) 
```

For example, 

```txt
f = tract.make_morphism(domain=(2,2,10,10), codomain = (2,2,2,10,10), map_=(1,2,4,5))
coal_f = tract.coalesce(f) 
```

returns 

```txt
f = (2, 2, 10, 10) -- (1, 2, 4, 5) -- > (2, 2, 2, 10, 10)
coal_f = (4, 100) -- (1, 3) -- > (4, 2, 100) 
```

6. Complement: When defined, this operation produces a layout comp(A, N) from a layout A and positive integer N. See Definition 2.3.6.5 for details. We can compute comp(A, N) in cute with 

```javascript
cute.complement(A,N) 
```

For example, 

```txt
A = cute.make_layout(shape = ((2,2),(2,2)), stride = ((8,2),(64,256)))
comp_A = cute.complement(A,4096) 
```

returns 

```matlab
A = ((2,2),(2,2)):((8,2),(64,256))
comp_A = (2,2,4,2,8):(1,4,16,128,512) 
```

If f is a nested tuple morphism, then we may form the complement f<sup>c</sup> of f. See Definition 3.2.6.17 for details. We compute f<sup>c</sup> in tract with 

```javascript
tract.complement(f) 
```

For example, 

```python
f = tract.make_morphism(domain=(2,2), codomain=(2,5,2,5), map_=(1,3))
comp_f = tract.complement(f) 
```

returns 

```txt
f = (2,2)--(1,3)-->(2,5,2,5)
comp_A = (5,5)--(2,4)-->(2,5,2,5) 
```

7. Logical Division: When defined, this operation produces a layout A ⊘ B from a pair of layouts A and B. See Definition 2.3.8.1 for details. We compute A ⊘ B in cute with 

```txt
cute.logical_divide(A,B) 
```

For example, 

```python
A = cute.make_layout((64,32), stride = (32,1))
B = cute.make_layout((4,4), stride = (1,64))
quotient = cute.logical_divide(A,B) 
```

returns 

```txt
A = (64,32):(32,1)
B = (4,4):(1,64)
quotient = ((4,4),(16,8)):(32,1),(128,4)) 
```

If f and g are nested tuple morphisms and g divides f, then we may form the logical division f ⊘ g. See Definition 3.2.6.23 for details. We compute f ⊘ g in tract with 

```javascript
tract.logical_divide(f,g) 
```

For example, 

```txt
f = tract.make_morphism(domain=(4,8,4,8), codomain=(4,8,4,8), map_=(1,2,3,4))
g = tract.make_morphism(domain=(4,4), codomain=(4,8,4,8), map_=(1,3))
quotient = tract.logical_divide(f,g) 
```

returns 

```txt
f = (4,8,4,8)--(1,2,3,4)-->(4,8,4,8)
g = (4,4)--(1,3)-->(4,8,4,8)
quotient = ((4,4),(8,8))--(1,3,2,4)-->(4,8,4,8) 
```

8. Logical Product: When defined, this operation produces a layout A ⊗ B from a pair of layouts A and B. See Definition 2.3.9.1 for details. We compute A ⊗ B in cute with 

```javascript
cute.logical_product(A,B) 
```

For example, running 

```python
A = cute.make_layout((3,10,10), stride = (200,1,20))
B = cute.make_layout((2,2), stride = (1,2))
product = cute.logical_product(A,B) 
```

returns 

```txt
A = (3,10,10):(200,1,20)
B = (2,2):(1,2)
product = ((3,10,10),(2,2)):((200,1,20),(10,600)) 
```

If f and g are nested tuple morphisms and f and g are product admissible, then we may form the logical product $f \otimes g .$ . See Definition 3.2.6.28 for details. We compute f ⊗ g in tract with 

```javascript
tract.logical_product(f,g) 
```

For example, 

```txt
f = tract.make_morphism(domain=(2,2), codomain=(2,2,5,5), map_=(1,2))
g = tract.make_morphism(domain=(5,5), codomain=(5,5), map_=(2,1))
product = tract.logical_product(f,g) 
```

returns 

```txt
f = (2,2)--(1,2)-->(2,2,5,5)
g = (5,5)--(2,1)-->(5,5)
product = ((2,2),(5,5))--(1,2,4,3)-->(2,2,5,5) 
```

## 1.5 Notation

$\mathbb{Z} = \{\dots, -1, 0, 1, 2, \dots\}$ $\mathbb{N} = \{0, 1, 2, \dots\}$ $\mathbb{Z}_{>0} = \{1, 2, \dots\}$ $\mathbb{F}_2 = \{0, 1\}$ , the finite field of order 2. $[0, n) = \{0, \dots, n-1\}$ , and $[0, 0) = \varnothing$ . $\langle n \rangle = \{1, 2, \dots, n\}$ , and $\langle 0 \rangle = \varnothing$ . $\langle n \rangle_* = \{*, 1, 2, \dots, n\}$ $\delta_i^m = (0, \dots, 1, \dots, 0)$ , the tuple of length $m$ with $i$ th entry 1 and all other entries 0. $\Sigma_n =$ the symmetric group on $\langle n \rangle$ . $X^\sigma = (x_{\sigma(1)}, \dots, x_{\sigma(m)})$ for a tuple $X = (x_1, \dots, x_m)$ and a permutation $\sigma \in \Sigma_m$ . $X \star Y =$ the flat concatenation of $X$ and $Y$ . $X^\flat =$ the flattening of a nested tuple $X$ . $\text{prof}(X) =$ the profile of a nested tuple $X$ . $(X_1, \dots, X_k) =$ the (nested) concatenation of $X_1, \dots, X_k$ . $(X_1, \dots, X_k)_Q =$ the $Q$ -substitution of $X_1, \dots, X_k$ for a profile $Q$ . $\text{Tuple}(V) =$ the set of tuples with entries in a set $V$ . $\text{Nest}(V) =$ the set of nested tuples with entries in a set $V$ . $\text{Profile} =$ the set of profiles. $\text{FlatLayout} =$ the set of flat layouts. $\text{Layout} =$ the set of layouts. $B \circ A =$ the composition of $A$ and $B$ . $A \oslash B =$ the logical division of $A$ by $B$ . $A \otimes B =$ the logical product of $A$ and $B$ . $\textbf{Set} =$ the category of sets. $\textbf{FinSet} =$ the category of finite sets. $\textbf{Fin} =$ the full subcategory of $\textbf{FinSet}$ spanned by $\langle n \rangle$ for $n \geq 0$ . $\textbf{FinSet}_* =$ the category of pointed finite sets. $\textbf{Fin}_* =$ the full subcategory of $\textbf{FinSet}_*$ spanned by $\langle n \rangle$ for $n \geq 0$ . $\textbf{Tuple} =$ the category of tuples and tuple morphisms. $\textbf{Nest} =$ the category of nested tuples and nested tuple morphisms. $\textbf{Ref} =$ the category of nested tuples and refinements. $\textbf{Cat} =$ the category of (small) categories and functors. 

## Chapter 2

# Layouts and their algebra

The goal of this chapter is to provide a comprehensive and mathematically grounded theory of layouts. We begin by developing a theory of flat layouts in section 2.1. We introduce the necessary background on nested tuples in section 2.2, so that we may cover layouts in full generality in section 2.3. 

## 2.1 Flat Layouts

In this section, we examine flat layouts, an important subclass of layouts in which both shape and stride are tuples, rather than more general nested tuples. To formalize our discussion, we begin by fixing notation related to tuples. 

## 2.1.1 Tuples

Definition 2.1.1.1. If V is a set, then a tuple with entries in V is a finite ordered list 

$$
X = (x _ {1}, \ldots , x _ {m})
$$

of elements $x _ { i } \in V$ for each $1 \leq i \leq m$ . The length of such a tuple $X = ( x _ { 1 } , \dots , x _ { m } )$ is 

$$
\operatorname{len} (X) = m.
$$

We write Tuple(V ) for the collection of all tuples with entries in V . We are especially interested in the case $V = \mathbb { Z } .$ in which case we refer to $X \in { \mathsf { T u p l e } } ( \mathbb { Z } )$ as a tuple of integers. If X is a tuple of integers, then the size of X is the product 

$$
\operatorname{size} (X) = x _ {1} \dots x _ {m}.
$$

Example 2.1.1.2. Here are some examples of tuples, together with their length and size: 

$$
\begin{array}{l l} X = (3, 1 2 8, 1 2 8), & \text { len } (X) = 3, \quad \text { size } (X) = 4 9 1 5 2 \\ X = (5 1 2), & \text { len } (X) = 1, \quad \text { size } (X) = 5 1 2 \\ X = (), & \text { len } (X) = 0, \quad \text { size } (X) = 1 \end{array}
$$

Definition 2.1.1.3. If $X = ( x _ { 1 } , \dots , x _ { m } )$ and $Y = \left( y _ { 1 } , \dots y _ { n } \right)$ are tuples, then we write 

$$
X \star Y = (x _ {1}, \ldots , x _ {m}, y _ {1}, \ldots , y _ {n})
$$

for the concatenation of X and Y. 

Example 2.1.1.4. If $X = ( 6 4 , 3 2 )$ and $Y = ( 8 , 8 , 8 )$ , then 

$$
X \star Y = (6 4, 3 2, 8, 8, 8).
$$

Remark 2.1.1.5. If V is a set, then the collection 

$$
\operatorname{Tuple} (V) = \coprod_ {m \geq 0} V ^ {\times m}
$$

of all tuples with entries in V is the free associative monoid on V. The monoidal product is concatenation, and the unit is the empty tuple (). 

Definition 2.1.1.6. If X and $X ^ { \prime }$ are tuples, we say $X ^ { \prime }$ divides X if there exists a tuple $X ^ { \prime \prime }$ with 

$$
X ^ {\prime} \star X ^ {\prime \prime} = X.
$$

Example 2.1.1.7. If $X ^ { \prime } = ( 8 1 , 9 )$ and $X = ( 8 1 , 9 , 6 4 , 8 )$ , then $X ^ { \prime }$ divides X, since the tuple $X ^ { \prime \prime } = ( 6 4 , 8 )$ satisfies 

$$
X ^ {\prime} \star X ^ {\prime \prime} = X.
$$

Definition 2.1.1.8. If $X = ( x _ { 1 } , \dots , x _ { m } )$ is a tuple and $\sigma \in \Sigma _ { m }$ is a permutation, then we write 

$$
X ^ {\sigma} = (x _ {\sigma (1)}, \ldots , x _ {\sigma (m)})
$$

for the permutation of X by σ. This specifies a right action of $\Sigma _ { m }$ on $\mathbb { Z } ^ { \times m }$ 

Example 2.1.1.9. If $X = ( 8 , 1 6 , 3 2 , 6 4 )$ and $\sigma = ( 1 2 ) ( 3 4 )$ , then 

$$
X ^ {\sigma} = (1 6, 8, 6 4, 3 2).
$$

Notation 2.1.1.10. If n is a positive integer, we write 

$$
[ 0, n) = \{0, 1, \dots , n - 1 \},
$$

and if $S = ( s _ { 1 } , \ldots , s _ { m } ) $ is a tuple of positive integers, we write 

$$
[ 0, S) = [ 0, s _ {1}) \times \dots \times [ 0, s _ {m})
$$

for the collection of tuples $( x _ { 1 } , \ldots , x _ { m } )$ with $0 \leq x _ { i } < s _ { i }$ 

Example 2.1.1.11. If $S = ( 3 , 2 )$ , then 

$$
[ 0, S) = \{(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1) \}
$$

## 2.1.2 Basic definitions

Having fixed notation, we are ready to define flat layouts. 

Definition 2.1.2.1. A flat layout is a pair 

$$
\begin{array}{l} L = S: D \\ \qquad = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m}) \end{array}
$$

consisting of a tuple of positive integers 

$$
\begin{array}{c} \text {shape} (L) = S \\ = (s _ {1}, \ldots , s _ {m}) \end{array}
$$

called the shape of L, and a tuple of non-negative integers 

$$
\begin{array}{c} \text {stride} (L) = D \\ = (d _ {1}, \ldots , d _ {m}) \end{array}
$$

called the stride of L. 

Remark 2.1.2.2. If L is a flat layout, then by definition, shape(L) and stride(L) have the same length. Remark 2.1.2.3. A flat layout is an example of the more general layout of Definition 2.3.1.1, so we sometimes refer to a flat layout L as a layout. 

Example 2.1.2.4. Here are some examples of flat layouts: 

$$
\begin{array}{l} L _ {1} = (2, 2, 2): (1, 2, 4), \\ L _ {2} = (1 2 8): (5), \\ L _ {3} = (1 6, 1 2, 5 1 2, 5 1 2): (0, 0, 1, 5 1 2), \\ L _ {4} = (6, 1, 1 2, 2, 2): (2, 0, 1 2, 1 4 4, 1), \\ L _ {5} = (): (). \end{array}
$$

Example 2.1.2.5. We can depict the layout $L = ( 8 ) : ( 5 )$ as 

<table><tr><td>0</td><td>5</td><td>10</td><td>15</td><td>20</td><td>25</td><td>30</td><td>35</td></tr></table>

and we can depict the layout $L = ( 3 , 5 ) : ( 2 , 1 0 )$ as 

<table><tr><td>0</td><td>10</td><td>20</td><td>30</td><td>40</td></tr><tr><td>2</td><td>12</td><td>22</td><td>32</td><td>42</td></tr><tr><td>4</td><td>14</td><td>24</td><td>34</td><td>44</td></tr></table>

We make precise the sense in which these pictures represent the associated layout in Remark 2.1.2.17. 

Perhaps the most important examples of flat layouts are the column-major and row-major layouts, which we define below. 

Definition 2.1.2.6. Suppose 

$$
L = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m})
$$

is a flat layout. We say L is column-major if 

$$
d _ {i} = s _ {1} \cdot \cdot \cdot s _ {i - 1}
$$

for each $1 \leq i \leq m$ . We say L is row-major if 

$$
d _ {i} = s _ {i + 1} \dots s _ {m}.
$$

for each $1 \leq i \leq m$ 

Example 2.1.2.7. The layout 

$$
L = (3, 4): (1, 3) = \begin{array}{c c c c} \hline 0 & 3 & 6 & 9 \\ \hline 1 & 4 & 7 & 1 0 \\ \hline 2 & 5 & 8 & 1 1 \\ \hline \end{array}
$$

is column-major, while the layout 

$$
L = (3, 4): (4, 1) = \quad \begin{array}{c c c c} \hline 0 & 1 & 2 & 3 \\ \hline 4 & 5 & 6 & 7 \\ \hline 8 & 9 & 1 0 & 1 1 \\ \hline \end{array}
$$

is row-major. These pictures make clear the reason for the terminology: If L is a column-major layout of rank 2, then the columns of L are contiguous, and if L is a row-major layout of rank 2, then the rows of L are contiguous. 

Example 2.1.2.8. The layouts 

$$
\begin{array}{l} L _ {1} = (2, 2, 2, 2, 2): (1, 2, 4, 8, 1 6) \\ L _ {2} = (3, 1 2 8, 1 2 8): (1, 3, 3 8 4) \\ L _ {3} = (6 4): (1) \end{array}
$$

are column-major, while the layouts 

$$
\begin{array}{l} L _ {4} = (2, 2, 2, 2, 2): (1 6, 8, 4, 2, 1) \\ L _ {5} = (3, 1 2 8, 1 2 8): (1 6 3 8 4, 1 2 8, 1) \\ L _ {6} = (6 4): (1) \end{array}
$$

are row-major. 

Now that we’ve seen a few examples, lets define some important attributes of flat layouts. 

Definition 2.1.2.9. Suppose $L = ( s _ { 1 } , \ldots , s _ { m } ) : ( d _ { 1 } , \ldots , d _ { m } )$ is a flat layout. 

• The rank of L is 

$$
\operatorname{rank} (L) = m.
$$

• The size of L is 

$$
\operatorname{size} (L) = \prod_ {i = 1} ^ {m} s _ {i}.
$$

• The cosize of L is 

$$
\operatorname{cosize} (L) = 1 + \sum_ {i = 1} ^ {m} \left(s _ {i} - 1\right) \cdot d _ {i}.
$$

• For any $1 \leq i \leq \mathsf { r a n k } ( L )$ , the ith mode of L is the pair 

$$
\operatorname{mode} _ {i} (L) = s _ {i}: d _ {i}.
$$

Example 2.1.2.10. The layout 

$$
L = (6 4, 3 2): (1, 1 2 8)
$$

has $\mathsf { r a n k } ( L ) = 2 , \mathsf { s i z e } ( L ) = 2 0 4 8 ,$ , and cosize(L) = 4032. The modes of L are 

$$
\begin{array}{l} \text {mode} _ {1} (L) = 6 4: 1 \\ \text {mode} _ {2} (L) = 3 2: 1 2 8. \end{array}
$$

Example 2.1.2.11. The layout 

$$
L = (3, 8, 8, 8): (1, 3, 2 4, 1 9 2).
$$

has $\mathsf { r a n k } ( L ) = 4 , \mathsf { s i z e } ( L ) = 1 5 3 6$ , and $\mathsf { c o s i z e } ( L ) = 1 5 3 6$ . The layout L has four modes, for example mode $ \operatorname { \cdot } 3 ( L ) = 8 : 2 4 .$ 

Example 2.1.2.12. The layout 

$$
L = (2, 2, 2, 2, 2): (1 6 0, 8 0, 4 0, 2 0, 1 0).
$$

has $\mathsf { r a n k } ( L ) = 5 , \mathsf { s i z e } ( L ) = 3 2$ , and $\mathsf { c o s i z e } ( L ) = 3 1 1$ . The layout L has 5 modes, for example mode ${ \mathfrak { s } } ( L ) = 2 : 1 0$ 

If L is a flat layout, then L encodes a coordinate function $\varphi _ { L }$ . The coordinate function of L is a multi-dimensional to one-dimensional transformation given by taking a dot product with stride(L). Recall that if $S = ( s _ { 1 } , \ldots , s _ { m } ) $ is a tuple of positive integers, then 

$$
[ 0, S) = [ 0, s _ {1}) \times \dots \times [ 0, s _ {m})
$$

is the set of all tuples $\left( x _ { 1 } , \ldots , x _ { m } \right)$ such that $0 \leq x _ { i } < s _ { i }$ . In particular, if $S = ( )$ is the empty tuple, then $[ 0 , S ) = \{ ( ) \}$ 

Construction 2.1.2.13 (Coordinate functions). If 

$$
L = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m})
$$

is a flat layout, then the coordinate function of L is the function 

$$
[ 0, \text { shape } (L)) \xrightarrow {\varphi_ {L}} \mathbb {Z}
$$

given by 

$$
\begin{array}{c} \varphi_ {L} (x _ {1}, \ldots , x _ {m}) = (x _ {1}, \ldots , x _ {m}) \cdot (d _ {1}, \ldots , d _ {m}) \\ = x _ {1} d _ {1} + \dots + x _ {m} d _ {m}. \end{array}
$$

The coordinate function $\varphi _ { L }$ factors through the inclusion $[ 0 , \mathsf { c o s i z e } ( L ) ) \subset \mathbb { Z }$ , and we write 

$$
[ 0, \operatorname{shape} (L)) \xrightarrow {\varphi_ {L} ^ {\operatorname{cosize} (L)}} [ 0, \operatorname{cosize} (L)) \subset \mathbb {Z}
$$

for the factored map. More generally, for any $N \geq { \cos } { \mathsf { i z e } } ( L )$ , we write $\varphi _ { L } ^ { N }$ for the factorization of $\varphi _ { L }$ through $[ 0 , N ) \subset \mathbb { Z } ,$ , and by a mild abuse of terminology, we refer to such a map $\varphi _ { L } ^ { N }$ as the coordinate function of L. 

Example 2.1.2.14. If $L = ( 2 , 3 ) : ( 1 , 5 )$ , then the coordinate function 

$$
\varphi_ {L}: [ 0, 2) \times [ 0, 3) \to \mathbb {Z}
$$

is given by 

$$
\begin{array}{l} \varphi_ {L} (0, 0) = (0, 0) \cdot (1, 5) = 0, \\ \varphi_ {L} (1, 0) = (1, 0) \cdot (1, 5) = 1, \\ \varphi_ {L} (0, 1) = (0, 1) \cdot (1, 5) = 5, \\ \varphi_ {L} (1, 1) = (1, 1) \cdot (1, 5) = 6, \\ \varphi_ {L} (0, 2) = (0, 2) \cdot (1, 5) = 1 0, \\ \varphi_ {L} (1, 2) = (1, 2) \cdot (1, 5) = 1 1. \end{array}
$$

Example 2.1.2.15. If $L = ( 2 , 2 ) : ( 6 4 , 2 )$ , then the coordinate function 

$$
\varphi_ {L}: [ 0, 2) \times [ 0, 2) \to \mathbb {Z}
$$

is given by 

$$
\begin{array}{l} \varphi_ {L} (0, 0) = (0, 0) \cdot (6 4, 2) = 0, \\ \varphi_ {L} (1, 0) = (1, 0) \cdot (6 4, 2) = 6 4, \\ \varphi_ {L} (0, 1) = (0, 1) \cdot (6 4, 2) = 2, \\ \varphi_ {L} (1, 1) = (1, 1) \cdot (6 4, 2) = 6 6. \end{array}
$$

Example 2.1.2.16. If $E = ( ) : ( )$ is the empty layout, then the coordinate function of E is the map 

$$
\varphi_ {E}: \left\{\left(\right) \right\} \to \mathbb {Z}
$$

given by 

$$
\varphi (()) = 0.
$$

Remark 2.1.2.17. We can now, for example, give a precise description of the sense in which the image 

<table><tr><td>0</td><td>10</td><td>20</td><td>30</td><td>40</td></tr><tr><td>2</td><td>12</td><td>22</td><td>32</td><td>42</td></tr><tr><td>4</td><td>14</td><td>24</td><td>34</td><td>44</td></tr></table>

depicts the layout $L = ( 3 , 5 ) : ( 2 , 1 0 )$ : The $( i , j ) \mathrm { t h }$ cell of the grid is labeled by the value 

$$
\varphi_ {L} (i, j) = (i, j) \cdot (2, 1 0) = 2 i + 1 0 j
$$

of the coordinate function of L. 

In practice, the most important invariant of a flat layout L is its layout function $\Phi _ { L }$ , which is obtained by precomposing the coordinate function 

$$
\varphi_ {L}: [ 0, S) \to \mathbb {Z}
$$

with the inverse of the colexicographic isomorphism 

$$
\operatorname{colex} _ {S}: [ 0, S) \to [ 0, \operatorname{size} (S)).
$$

Definition 2.1.2.18. Suppose $S = ( s _ { 1 } , \ldots , s _ { m } ) $ is a tuple of positive integers and recall that 

$$
[ 0, S) = [ 0, s _ {1}) \times \dots \times [ 0, s _ {m}).
$$

The colexicographic isorphism is the map 

$$
[ 0, S) \xrightarrow {\operatorname{colex} _ {S}} [ 0, \operatorname{size} (S))
$$

$$
(x _ {1}, \ldots , x _ {m}) \longmapsto \sum_ {i = 1} ^ {m} s _ {1} \dots s _ {i - 1} x _ {i}.
$$

We sometimes write colex = colex<sub>S</sub> when the tuple $S$ is clear from context. The inverse of the colexicographic isomorphism is the map 

$$
[ 0, \operatorname{size} (S)) \xrightarrow {\operatorname{colex} _ {S} ^ {- 1}} [ 0, S)
$$

given by 

$$
\operatorname{colex} _ {S} ^ {- 1} (x) = \left(x _ {1}, \dots , x _ {m}\right)
$$

where 

$$
x _ {i} = \left\lfloor \frac {x}{s _ {1} \cdots s _ {i - 1}} \right\rfloor \mod s _ {i}.
$$

Note that if $S = ( )$ is the empty tuple, then 

$$
\operatorname{colex} _ {()}: \{\left(\right) \} \rightarrow \{0 \}
$$

and 

$$
\operatorname{colex} _ {()} ^ {- 1}: \{0 \} \to \{\left(\right) \}
$$

are the canonical isomorphisms. 

Construction 2.1.2.19 (Layout functions). If 

$$
L = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m}),
$$

is a flat layout, then the layout function of L is the composite 

![image](Imgaes/categorical-foundations-cute-layouts-paper/d1b5cd15b4928b3bb46d139c2c9d1fe8c748623fa688167daf81041f3c677cff.jpg)


Explicitly, $\Phi _ { L }$ is given by 

$$
\Phi_ {L} (x) = x _ {1} d _ {1} + \dots + x _ {m} d _ {m}
$$

where 

$$
x _ {i} = \left\lfloor \frac {x}{s _ {1} \cdots s _ {i - 1}} \right\rfloor \mod s _ {i}.
$$

The layout function $\Phi _ { L }$ factors through the inclusion $[ 0 , \mathsf { c o s i z e } ( L ) ) \subset \mathbb { Z } .$ , and we write 

$$
[ 0, \operatorname{size} (L)) \xrightarrow {\Phi_ {L} ^ {\operatorname{cosize} (L)}} [ 0, \operatorname{cosize} (L)) \subset \mathbb {Z}
$$

for the factored map. More generally, for any $N \geq { \cos } { \mathsf { i z e } } ( L )$ , we write $\Phi _ { L } ^ { N }$ for the factorization of $\Phi _ { L }$ through $[ 0 , N ) \subset \mathbb { Z } .$ , and by a mild abuse of terminology, we refer to such a map $\varphi _ { L } ^ { N }$ as the layout function of L. 

Example 2.1.2.20. If $L = ( 2 , 3 ) : ( 1 , 5 )$ , then the layout function 

$$
\Phi_ {L}: [ 0, 6) \to \mathbb {Z}
$$

is given by 

$$
\begin{array}{l} \Phi_ {L} (0) = (0, 0) \cdot (1, 5) = 0, \\ \Phi_ {L} (1) = (1, 0) \cdot (1, 5) = 1, \\ \Phi_ {L} (2) = (0, 1) \cdot (1, 5) = 5, \\ \Phi_ {L} (3) = (1, 1) \cdot (1, 5) = 6, \\ \Phi_ {L} (4) = (0, 2) \cdot (1, 5) = 1 0, \\ \Phi_ {L} (5) = (1, 2) \cdot (1, 5) = 1 1. \end{array}
$$

Example 2.1.2.21. If $L = ( 2 , 2 )$ : (64, 2), then the layout function 

$$
\Phi_ {L}: [ 0, 4) \to \mathbb {Z}
$$

is given by 

$$
\begin{array}{l} \Phi_ {L} (0) = (0, 0) \cdot (6 4, 2) = 0, \\ \Phi_ {L} (1) = (1, 0) \cdot (6 4, 2) = 6 4, \\ \Phi_ {L} (2) = (0, 1) \cdot (6 4, 2) = 2, \\ \Phi_ {L} (3) = (1, 1) \cdot (6 4, 2) = 6 6. \end{array}
$$

Example 2.1.2.22. If $L = ( 4 , 2 , 2 ) : ( 3 , 3 , 1 0 0 )$ , then for example, the layout function of L satisfies 

$$
\begin{array}{l} \Phi_ {L} (7) = (3, 1, 0) \cdot (3, 3, 1 0 0) = 1 2, \\ \Phi_ {L} (9) = (1, 0, 1) \cdot (3, 3, 1 0 0) = 1 0 3. \end{array}
$$

Example 2.1.2.23. If $E = ( ) : ( )$ is the empty layout, then 

$$
\Phi_ {E}: \{0 \} \to \mathbb {Z}
$$

is given by 

$$
\Phi_ {E} (0) = 0.
$$

Example 2.1.2.24. If L is any flat layout, then the layout function $\Phi _ { L }$ of L satisfies 

$$
\Phi_ {L} (0) = 0.
$$

Remark 2.1.2.25. If $S = ( s _ { 1 } , \ldots , s _ { m } ) $ is a tuple of positive integers, then the colexicographic isomorphism 

$$
[ 0, S) \xrightarrow {\operatorname{colex} _ {S}} [ 0, \operatorname{size} (S))
$$

is equal to the coordinate function $\varphi _ { L } ^ { \mathsf { c o s i z e } ( L ) }$ of the column major layout 

$$
L = (s _ {1}, s _ {2}, \ldots , s _ {m}): (1, s _ {1}, \ldots , s _ {1} \dots s _ {m - 1}).
$$

This implies that if a flat layout L is column-major, then 

$$
\begin{array}{l} \Phi_ {L} ^ {\text {cosize} (L)} = \varphi_ {L} ^ {\text {cosize} (L)} \circ \text {colex} _ {\text {shape} (L)} ^ {- 1} \\ \qquad = \varphi_ {L} ^ {\text {cosize} (L)} \circ \left(\varphi_ {L} ^ {\text {cosize} (L)}\right) ^ {- 1} \\ \qquad = \mathsf {i d} _ {[ 0, \text {size} (L))} \end{array}
$$

is the identity map on $[ 0 , \mathsf { s i z e } ( L ) )$ 

Remark 2.1.2.26. There exist distinct layouts $A \neq B$ with $\Phi _ { A } = \Phi _ { B }$ . For example, the layouts 

$$
\begin{array}{l} A = (7, 7): (1, 7) \\ B = (4 9): (1) \end{array}
$$

are not equal, yet $\Phi _ { A } = \Phi _ { B }$ . Later, we will characterize precisely when two flat layouts A and B have the same layout function (see Proposition 2.1.4.18). 

Before moving on to our discussion of layout operations, we need to define the notion of nondegeneracy. 

Definition 2.1.2.27. Suppose 

$$
L = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m})
$$

is a flat layout. We say L is non-degenerate if for any $1 \leq i \leq m$ , we have 

$$
s _ {i} = 1 \quad \Rightarrow \quad d _ {i} = 0.
$$

Example 2.1.2.28. The layouts 

$$
\begin{array}{l} L _ {1} = (4, 1): (1, 0) \\ L _ {2} = (8, 1, 8, 1): (2, 0, 1 6, 0) \end{array}
$$

are non-degenerate, while the layouts 

$$
\begin{array}{l} L _ {3} = (4, 1): (1, 4) \\ L _ {4} = (8, 1, 8, 1): (2, 1 6, 1 6, 2 5 6) \end{array}
$$

are degenerate. 

Observation 2.1.2.29. There is no real loss of generality in assuming that a layout L is non-degenerate. More precisely, if 

$$
\begin{array}{c} L = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m}) \\ L ^ {\prime} = (s _ {1}, \ldots , s _ {m}): (d _ {1} ^ {\prime}, \ldots , d _ {m} ^ {\prime}) \end{array}
$$

are flat layouts with the same shape, and $d _ { i } = d _ { i } ^ { \prime }$ whenever $s _ { i } > 1$ , then $\varphi _ { L } = \varphi _ { L ^ { \prime } }$ , and $\Phi _ { L } = \Phi _ { L ^ { \prime } }$ . In particular, we are free to set $d _ { i } = 0$ whenever $s _ { i } = 1$ without altering the coordinate function or layout function of L. 

## 2.1.3 Basic operations

Having established the basic vocabulary for flat layouts, we turn to the operations they support. In this section, we define basic operations that will be needed to construct more sophisticated operations such as coalesce, complement, and composition. 

## 2.1.3.1 Restriction

If L is a flat layout, it is often useful to restrict to a subset of the modes of L. Recall that for a non-negative integer m, we write 

$$
\langle m \rangle = \{1, \dots , m \}.
$$

Definition 2.1.3.1. Suppose 

$$
L = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m})
$$

is a flat layout, and suppose 

$$
I = \left\{i _ {1} <   \dots <   i _ {k} \right\} \subset \langle m \rangle
$$

is a subset. We define the restriction of L to I to be the flat layout 

$$
L \mid_ {I} = (s _ {i _ {1}}, \dots , s _ {i _ {k}}): (d _ {i _ {1}}, \dots , d _ {i _ {k}}).
$$

Example 2.1.3.2. If 

<table><tr><td>0</td><td>5</td><td>10</td><td>15</td><td>20</td><td>25</td></tr><tr><td>10</td><td>15</td><td>20</td><td>25</td><td>30</td><td>35</td></tr><tr><td>20</td><td>25</td><td>30</td><td>35</td><td>40</td><td>45</td></tr></table>

and $I = \{ 2 \}$ , then 

<table><tr><td>0</td><td>5</td><td>10</td><td>15</td><td>20</td><td>25</td></tr></table>

Example 2.1.3.3. If 

$$
L = (3, 8, 8, 8): (1, 3, 2 4, 1 9 2)
$$

and $I = \{ 1 , 2 , 3 \}$ , then 

$$
L \mid_ {I} = (3, 8, 8): (1, 3, 2 4).
$$

Example 2.1.3.4. If 

$$
L = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m})
$$

is a flat layout and $I = \langle m \rangle$ , then 

$$
L \mid_ {I} = L.
$$

Example 2.1.3.5. If 

$$
L = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m})
$$

is a flat layout and $I = \emptyset$ is the empty set, then 

$$
L \mid_ {I} = (): ()
$$

is the empty layout. 

## 2.1.3.2 Squeeze

If L is a flat layout, then the operation $L \mapsto { \mathsf { s q u e e z e } } ( L )$ removes all modes $s _ { i } : d _ { i }$ of L where $s _ { i } = 1$ Construction 2.1.3.6. Suppose 

$$
L = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m})
$$

is a flat layout, and let 

$$
I = \{i \in \langle m \rangle \mid s _ {i} > 1 \}
$$

be the collection of indices whose corresponding shape entry is not equal to 1. We define 

$$
\operatorname{squeeze} (L) = L \mid_ {I}.
$$

Example 2.1.3.7. If 

$$
L = (6 4, 6 4, 1): (1, 6 4, 0),
$$

then 

$$
\operatorname{squeeze} (L) = (6 4, 6 4): (1, 6 4).
$$

Example 2.1.3.8. If 

$$
L = (6 4, 6 4, 1, 3 2, 1): (2 0 4 8, 3 2, 0, 1, 0)
$$

then 

$$
\operatorname{squeeze} (L) = (6 4, 6 4, 3 2): (2 0 4 8, 3 2, 1).
$$

Example 2.1.3.9. If L is a flat layout, then 

$$
\operatorname{squeeze} (L) = L
$$

if and only if shape(L) contains no entries equal to 1. 

Example 2.1.3.10. If L is a flat layout, then 

$$
\text { squeeze } (L) = (): ()
$$

is the empty layout if and only if all entries of shape(L) are equal to 1. 

An essential property of this construction is that $L \mapsto { \mathsf { s q u e e z e } } ( L )$ leaves the layout function of L unchanged. 

Lemma 2.1.3.11. If L is a flat layout, then 

1. si $z \mathsf { e } ( \mathsf { s q u e e z e } ( L ) ) = \mathsf { s i z e } ( L ) ,$ 

2. cosiz $\mathsf { e } ( { \mathsf { s q u e e z e } } ( L ) ) = { \mathsf { c o s i z e } } ( L )$ , and 

3. $\Phi _ { \mathsf { s q u e e z e } ( L ) } = \Phi _ { L }$ 

Proof. Let 

$$
I = \left\{i _ {1} <   \dots <   i _ {k} \right\} \subset \langle m \rangle
$$

denote the collection of indices with $s _ { i _ { i } } > 1$ , so that 

$$
\operatorname{squeeze} (L) = \left(s _ {i _ {1}}, \dots , s _ {i _ {k}}\right): \left(d _ {i _ {1}}, \dots , d _ {i _ {k}}\right).
$$

For the first assertion, we compute 

$$
\operatorname{size} (\operatorname{squeeze} (L)) = \prod_ {j = 1} ^ {k} s _ {i _ {j}} = \left(\prod_ {j = 1} ^ {k} s _ {i _ {j}}\right) \cdot \left(\prod_ {\langle m \rangle \backslash I} 1\right) = \prod_ {i = 1} ^ {m} s _ {i} = \operatorname{size} (L).
$$

For the second assertion, we compute 

$$
\begin{array}{c} \text {cosize} (\text {squeeze} (L)) = 1 + \sum_ {j = 1} ^ {k} (s _ {i _ {j}} - 1) \cdot d _ {i _ {j}} = 1 + \sum_ {j = 1} ^ {k} (s _ {i _ {j}} - 1) \cdot d _ {i _ {j}} + \left(\sum_ {\langle m \rangle \setminus I} 0\right) \\ = 1 + \sum_ {i = 1} ^ {m} (s _ {i} - 1) \cdot d _ {i} \\ = \text {cosize} (L). \end{array}
$$

For the third assertion, it sufices to show that removing a mode of the form $1 : d _ { i }$ from a flat layout leaves the layout function unchanged. Suppose $L = ( s _ { 1 } , \ldots , s _ { m } ) : ( d _ { 1 } , \ldots , d _ { m } )$ , and suppose that some $s _ { i } = 1$ . Let 

$$
L ^ {\prime} = (s _ {1} ^ {\prime}, \ldots , s _ {m - 1} ^ {\prime}): (d _ {1} ^ {\prime}, \ldots , d _ {m - 1} ^ {\prime})
$$

denote the flat layout obtained from $L$ by removing its ith mode, so that 

$$
s _ {j} ^ {\prime} = \left\{ \begin{array}{l l} s _ {j} & j <   i \\ s _ {j + 1} & i \leq j <   m, \end{array} \right. \quad \text { and } \quad d _ {j} ^ {\prime} = \left\{ \begin{array}{l l} d _ {j} & j <   i \\ d _ {j + 1} & i \leq j <   m. \end{array} \right.
$$

The layout function for L is given by 

$$
\Phi_ {L} (x) = x _ {1} d _ {1} + \dots + x _ {m} d _ {m}
$$

where $x _ { j } = \left\lfloor { \frac { x } { s _ { 1 } \cdot \cdot \cdot s _ { j - 1 } } } \right\rfloor$ mod $s _ { j }$ , and the layout function for $L ^ { \prime }$ is given by 

$$
\Phi_ {L ^ {\prime}} (x) = x _ {1} ^ {\prime} d _ {1} ^ {\prime} + \dots + x _ {m - 1} ^ {\prime} d _ {m - 1} ^ {\prime}
$$

where $x _ { j } ^ { \prime } = \left\lfloor { \frac { x } { s _ { 1 } ^ { \prime } \cdot \cdot \cdot s _ { j - 1 } ^ { \prime } } } \right\rfloor$ mod $s _ { j } ^ { \prime }$ . We observe that 

$$
x _ {j} ^ {\prime} = \left\{ \begin{array}{l l} x _ {j} & j <   i \\ x _ {j + 1} & i \leq j <   m, \end{array} \right.
$$

and since $x _ { i } \in [ 0 , 1 )$ is necessarily 0, we have 

$$
\begin{array}{r l} & {\Phi_ {L} (x) = x _ {1} d _ {1} + \dots + x _ {m} d _ {m}} \\ & {\qquad = x _ {1} d _ {1} + \dots + x _ {i - 1} d _ {i - 1} + x _ {i + 1} d _ {i + 1} + \dots + x _ {m} d _ {m}} \\ & {\qquad = x _ {1} ^ {\prime} d _ {1} ^ {\prime} + \dots + x _ {m - 1} ^ {\prime} d _ {m - 1} ^ {\prime}} \\ & {\qquad = \Phi_ {L ^ {\prime}} (x).} \end{array}
$$

## 2.1.3.3 Filter zeros

If L is a flat layout, then the operation $L \mapsto { \mathsf { f i l t e r } } ( L )$ removes all modes $s _ { i } : d _ { i }$ with $d _ { i } = 0$ Definition 2.1.3.12. Suppose 

$$
L = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m})
$$

is a flat layout, and let 

$$
I = \{i \in \langle m \rangle \mid d _ {i} > 0 \}
$$

be the collection of indices whose corresponding stride entry is not equal to 0. We define 

$$
\operatorname{filter} (L) = L \mid_ {I}.
$$

Example 2.1.3.13. If 

$$
L = (6 4, 8, 8, 1 2 8): (8, 1, 0, 5 1 2)
$$

then 

$$
\operatorname{filter} (L) = (6 4, 8, 1 2 8): (8, 1, 5 1 2).
$$

Example 2.1.3.14. If 

$$
L = (3, 2): (1 2, 0) = \quad \begin{array}{c c} \hline 0 & 0 \\ \hline 1 2 & 1 2 \\ \hline 2 4 & 2 4 \\ \hline \end{array}
$$

then 

$$
\operatorname{filter} (L) = (3): (1 2) = \quad \begin{array}{c} \framebox {0} \\ \framebox {1 2} \\ \framebox {2 4} \end{array}
$$

Example 2.1.3.15. If 

$$
L = (3, 8, 8, 8): (1 6, 0, 0, 0)
$$

then 

$$
\operatorname{filter} (L) = (3): (1 6).
$$

Example 2.1.3.16. If L is a flat layout, then 

$$
\operatorname{filter} (L) = L
$$

if and only if all entries of stride(L) are nonzero. 

Example 2.1.3.17. If L is a flat layout, then 

$$
\operatorname{filter} (L) = (): ()
$$

is the empty layout if and only if all entries of stride(L) are equal to 0. 

## 2.1.3.4 Permute

Recall that if $X = ( x _ { 1 } , \dots , x _ { m } )$ is a tuple of length m and $\sigma \in \Sigma _ { m }$ is a permutation, then we write 

$$
X ^ {\sigma} = (x _ {\sigma (1)}, \ldots , x _ {\sigma (m)}).
$$

for the permutation of X by σ. 

Definition 2.1.3.18. If $L = ( s _ { 1 } , \ldots , s _ { m } ) : ( d _ { 1 } , \ldots , d _ { m } )$ is a flat layout of rank m and $\sigma \in \Sigma _ { m }$ is a permutation, we define 

$$
\begin{array}{l} L ^ {\sigma} = \mathsf {s h a p e} (L) ^ {\sigma}: \mathsf {s t r i d e} (L) ^ {\sigma} \\ \qquad = (s _ {\sigma (1)}, \ldots , s _ {\sigma (m)}): (d _ {\sigma (1)}, \ldots , d _ {\sigma (m)}). \end{array}
$$

Example 2.1.3.19. If 

$$
L = (4, 2): (1 2, 2) =
$$

<table><tr><td>0</td><td>2</td></tr><tr><td>12</td><td>14</td></tr><tr><td>24</td><td>26</td></tr><tr><td>36</td><td>38</td></tr></table>

and $\sigma = ( 1 2 ) \in \Sigma _ { 2 }$ is the transposition, then 

$$
L ^ {\sigma} = (2, 4): (2, 1 2) =
$$

<table><tr><td>0</td><td>12</td><td>24</td><td>36</td></tr><tr><td>2</td><td>14</td><td>26</td><td>38</td></tr></table>

is the transposed layout. 

Example 2.1.3.20. If 

$$
L = (1 5, 1 2, 1 0): (2 4 0, 1, 2 4)
$$

and $\sigma = ( 1 2 ) \in \Sigma _ { 3 }$ , then 

$$
L ^ {\sigma} = (1 2, 1 5, 1 0): (1, 2 4 0, 2 4).
$$

Example 2.1.3.21. If 

$$
L = (2, 2, 2, 2, 2): (1, 2, 4, 8, 1 6)
$$

and $\sigma = ( 1 5 ) ( 3 2 4 ) \in \Sigma _ { 5 }$ , then 

$$
L ^ {\sigma} = (2, 2, 2, 2, 2): (1 6, 8, 2, 4, 1).
$$

Example 2.1.3.22. If 

$$
L = (s, \dots , s): (d, \dots , d)
$$

is a flat layout all of whose modes are equal, then for any $\sigma \in \Sigma _ { m }$ , we have 

$$
L ^ {\sigma} = L.
$$

## 2.1.3.5 Sort

If L is a flat layout, it is often useful to permute L so that its modes are increasing, in the following sense. 

Definition 2.1.3.23. We define a linear ordering on pairs $s : d$ of integers by 

$$
s: d \preceq s ^ {\prime}: d ^ {\prime} \quad \Leftrightarrow \quad \begin{array}{c} d <   d ^ {\prime}, \text {or} \\ d = d ^ {\prime} \text {and} s \leq s ^ {\prime}. \end{array}
$$

Example 2.1.3.24. We have 

$$
5: 8 \preceq 4: 1 2 \preceq 5: 1 2.
$$

Definition 2.1.3.25. Suppose L is a flat layout. We say L is sorted if for any $1 \leq i < \mathsf { r a n k } ( L )$ , we have 

$$
\operatorname{mode} _ {i} (L) \preceq \operatorname{mode} _ {i + 1} (L).
$$

Example 2.1.3.26. The layouts 

$$
\begin{array}{l} L _ {1} = (1 2 8, 6 4, 2, 2): (1, 1 2 8, 8 1 9 2, 1 6 3 8 4) \\ L _ {2} = (2, 2, 2): (1, 1, 1) \end{array}
$$

are sorted, while the layouts 

$$
\begin{array}{l} L _ {3} = (2, 4, 8, 1 6): (6 4, 1, 2, 4) \\ L _ {4} = (5, 3 2, 1 6): (1, 5, 5) \end{array}
$$

are not sorted. 

Example 2.1.3.27. The empty layout $E = ( ) : ( )$ is sorted. 

Example 2.1.3.28. If 

$$
L = (s _ {1}, \dots , s _ {m}): (0, \dots , 0)
$$

is a flat layout with all entries of stride(L) equal to 0, then L is sorted if and only if 

$$
s _ {1} \leq s _ {2} \leq \dots \leq s _ {m}.
$$

Whether or not a flat layout L is sorted is intimately related to the behavior of the layout function $\Phi _ { L }$ of L, as described in the following lemma. 

Lemma 2.1.3.29. Suppose L is a flat layout. ${ J f \Phi } _ { L }$ is non-decreasing, then L is sorted. 

Proof. We prove the contrapositive. Suppose that L is not sorted. We will show that there exists some $x \leq y$ in the domain of $\Phi _ { L }$ with $\Phi _ { L } ( x ) > \Phi _ { L } ( y )$ . If there exists some $1 \leq i <$ m such that $d _ { i } > d _ { i + 1 }$ ， then we can let 

$$
x = \prod_ {j <   i} s _ {j}, \quad \text { and } \quad y = \prod_ {j <   i + 1} s _ {j},
$$

in which case $x < y .$ , but 

$$
\begin{array}{r l} \Phi_ {L} (x) & = (0, \ldots , 1, 0, \ldots , 0) \cdot (d _ {1}, \ldots , d _ {i}, d _ {i + 1}, \ldots , d _ {m}) \\ & = d _ {i} \\ & > d _ {i + 1} \\ & = (0, \ldots , 0, 1, \ldots , 0) \cdot (d _ {1}, \ldots , d _ {m}) \\ & = \Phi_ {L} (y). \end{array}
$$

On the other hand, if there exists some $1 \leq i <$ m such that $d _ { i } = d _ { i + 1 }$ and $s _ { i } > s _ { i + 1 }$ , we can set 

$$
x = \left(s _ {i} - 1\right) \left(\prod_ {j <   i} s _ {j}\right), \quad \text { and } \quad y = \left(s _ {i + 1} - 1\right) \left(\prod_ {j <   i + 1} s _ {j}\right),
$$

in which case $x < y$ , but 

$$
\begin{array}{l} \Phi_ {L} (x) = (0, \ldots , s _ {i} - 1, 0, \ldots , 0) \cdot (d _ {1}, \ldots , d _ {i}, d _ {i + 1}, \ldots , d _ {m}) \\ \qquad = (s _ {i} - 1) d _ {i} \\ \qquad > (s _ {i + 1} - 1) d _ {i} \\ \qquad = (s _ {i + 1} - 1) d _ {i + 1} \\ \qquad = (0, \ldots , 0, s _ {i + 1} - 1, \ldots , 0) \cdot (d _ {1}, \ldots , d _ {m}) \\ \qquad = \Phi_ {L} (y). \end{array}
$$

We conclude that $\Phi _ { L }$ is not non-decreasing. 

Remark 2.1.3.30. The converse of the previous lemma is false. For example, the flat layout 

$$
L = (3, 5, 7): (1, 1, 1)
$$

is sorted, but 

$$
\Phi_ {L} (7) = (0, 2, 0) \cdot (1, 1, 1) = 2
$$

is strictly greater than 

$$
\Phi_ {L} (1 6) = (0, 0, 1) \cdot (1, 1, 1) = 1.
$$

If L is a flat layout, then we can permute the modes of L to obtain a sorted layout $\mathsf { s o r t } ( L )$ 

Construction 2.1.3.31. Suppose 

$$
L = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m})
$$

is a flat layout. Define a linear ordering ⪯ on ⟨m⟩ by $i \preceq j$ if 

1. mode $( L ) \preceq { m o d e } _ { j } ( L )$ , and 

2. if mode $: ( L ) = { \mathsf { m o d e } } _ { j } ( L )$ then $i \leq j$ 

Let $\sigma \in \Sigma _ { m }$ be the permutation associated to the linear ordering ⪯ of ⟨m⟩. We define sort(L) to be permutation of L by σ: 

$$
\operatorname{sort} (L) = L ^ {\sigma}.
$$

Example 2.1.3.32. If 

$$
L = (2, 4, 8, 1 6): (6 4, 1, 2, 4)
$$

then 

$$
\operatorname{sort} (L) = (4, 8, 1 6, 2): (1, 2, 4, 6 4).
$$

Example 2.1.3.33. If 

$$
L = (5, 3 2, 1 6): (1, 5, 5)
$$

then 

$$
\operatorname{sort} (L) = (5, 1 6, 3 2): (1, 5, 5).
$$

Example 2.1.3.34. If L is sorted, then $\mathsf { s o r t } ( L ) = L$ . In particular, this implies that $\mathsf { s o r t } ( - )$ is an idempotent operation: 

$$
\operatorname{sort} (\operatorname{sort} (L)) = \operatorname{sort} (L).
$$

Observation 2.1.3.35. If L is a flat layout, then typically $\Phi _ { \mathsf { s o r t } ( L ) } \neq \Phi _ { L }$ . However, the layout functions $\Phi _ { L }$ and $\Phi _ { \mathsf { s o r t } ( L ) }$ always have the same image. To see this, let’s write 

$$
\begin{array}{c} L = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m}), \text {and} \\ \mathsf {s o r t} (L) = (s _ {\sigma (1)}, \ldots , s _ {\sigma (m)}): (d _ {\sigma (1)}, \ldots , d _ {\sigma (m)}) \end{array}
$$

for some permutation $\sigma \in \Sigma _ { m }$ . If an integer n is in the image of $\Phi _ { L } .$ , then there exists a tuple $\begin{array} { r } { ( x _ { 1 } , \ldots , x _ { m } ) \in \prod _ { i = 1 } ^ { m } \left[ 0 , s _ { i } \right) } \end{array}$ such that 

$$
x _ {1} d _ {1} + \dots + x _ {m} d _ {m} = n
$$

in which case the tuple $\begin{array} { r } { \left( x _ { \sigma \left( 1 \right) } , \ldots , x _ { \sigma \left( m \right) } \right) \in \prod _ { i = 1 } ^ { m } \left[ 0 , s _ { \sigma \left( i \right) } \right) } \end{array}$ satisfies 

$$
x _ {\sigma (1)} d _ {\sigma (1)} + \dots + x _ {\sigma (m)} d _ {\sigma (m)} = n.
$$

This proves that $\mathsf { I m a g e } ( \Phi _ { \mathsf { s o r t } } ( L ) ) \subseteq \mathsf { I m a g e } ( \Phi _ { L } )$ , and the reverse inclusion is proved similarly. 

## 2.1.3.6 Concatenate

Recall that if $X = ( x _ { 1 } , \dots , x _ { m } )$ and $Y = \left( y _ { 1 } , \dots , y _ { n } \right)$ are tuples, then the concatenation of X and $Y$ is the tuple 

$$
X \star Y = (x _ {1}, \dots , x _ {m}, y _ {1}, \dots , y _ {n}).
$$

This definition extends naturally to the concatenation of flat layouts. 

Definition 2.1.3.36. Suppose 

$$
\begin{array}{c} {L _ {1} = S _ {1}: D _ {1}} \\ {L _ {2} = S _ {2}: D _ {2}} \end{array}
$$

are flat layouts. Then the concatenation of $L _ { 1 }$ and $L _ { 2 }$ is the flat layout 

$$
L _ {1} \star L _ {2} = S _ {1} \star S _ {2}: D _ {1} \star D _ {2}.
$$

Concatenation of flat layouts is associative, so more generally, if $L _ { 1 } , \ldots , L _ { k }$ are flat layouts, we may form the concatenation 

$$
L _ {1} \star \dots \star L _ {k}.
$$

Example 2.1.3.37. If $L _ { 1 } = ( 7 , 2 ) : ( 2 , 1 )$ and $L _ { 2 } = ( 3 , 3 , 3 ) : ( 0 , 1 0 , 3 0 )$ , then 

$$
L _ {1} \star L _ {2} = (7, 2, 3, 3, 3): (2, 1, 0, 1 0, 3 0).
$$

Example 2.1.3.38. If $E = ( ) : ( )$ is the empty layout, then for any flat layout L we have 

$$
L \star E = L = E \star L.
$$

Observation 2.1.3.39. Suppose 

$$
L = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m})
$$

is a flat layout. If we write 

$$
L _ {i} = (s _ {i}): (d _ {i}),
$$

then we can write L as the concatenation 

$$
L = L _ {1} \star \dots \star L _ {m}.
$$

If $L _ { 1 } , \ldots , L _ { k }$ are flat layouts, then the layout function of the concatenation $L _ { 1 } { \star } { \cdot } \cdot { \star } L _ { k }$ is determined by the layout functions of $L _ { 1 } , \ldots , L _ { k }$ as follows. 

Proposition 2.1.3.40. Suppose $L _ { 1 } , \ldots , L _ { k }$ are flat layouts of shape $S _ { 1 } , \ldots , S _ { k }$ , and size $N _ { 1 } , \ldots , N _ { k }$ respectively. Then the coordinate function 

$$
\left[ 0, S _ {1} \star \dots \star S _ {k}\right) \xrightarrow {\varphi_ {L _ {1} \star \cdots \star L _ {k}}} \mathbb {Z}
$$

of $L _ { 1 } \star \cdots \star L _ { k }$ is equal to the composite 

$$
[ 0, S _ {1} \star \dots \star S _ {k}) \xrightarrow {\cong} [ 0, S _ {1}) \times \dots \times [ 0, S _ {k}) \xrightarrow {\varphi_ {L _ {1}} + \cdots + \varphi_ {L _ {k}}} \mathbb {Z},
$$

$$
X _ {1} \star \dots \star X _ {k} \longleftrightarrow (X _ {1}, \dots , X _ {k})
$$

and the layout function 

$$
\left[ 0, N _ {1} \dots N _ {k}\right) \xrightarrow {\Phi_ {L _ {1} \star \cdots \star L _ {k}}} \mathbb {Z}
$$

of $L _ { 1 } \star \cdots \star L _ { k }$ is equal to the composite 

$$
[ 0, N _ {1} \dots N _ {k}) \xrightarrow {\mathsf {c o l e x} _ {(N _ {1} , \ldots , N _ {k})} ^ {- 1}} [ 0, N _ {1}) \times \dots \times [ 0, N _ {k}) \xrightarrow {\Phi_ {L _ {1}} + \cdots + \Phi_ {L _ {k}}} \mathbb {Z}.
$$

Proof. Let’s write $L _ { i } = S _ { i } : D _ { i }$ for each $1 \leq i \leq k$ . The first claim holds because if 

$$
X \in [ 0, S _ {1} \star \dots \star S _ {k})
$$

corresponds to 

$$
X _ {1} \star \dots \star X _ {k} \in [ 0, S _ {1}) \times \dots \times [ 0, S _ {k})
$$

under the canonical isomorphism $[ 0 , S _ { 1 } \star \cdot \cdot \cdot \star S _ { k } ) \cong [ 0 , S _ { 1 } ) \times \cdot \cdot \cdot \times [ 0 , S _ { k } )$ , then 

$$
\begin{array}{r l} & {\varphi_ {L _ {1} \star \dots \star L _ {k}} (X) = X \cdot (D _ {1} \star \dots \star D _ {k})} \\ & {\qquad = (X _ {1} \star \dots \star X _ {k}) \cdot (D _ {1} \star \dots \star D _ {k})} \\ & {\qquad = (X _ {1} \cdot D _ {1}) + \dots + (X _ {k} \cdot D _ {k})} \\ & {\qquad = \varphi_ {L _ {1}} (X _ {1}) + \dots + \varphi_ {L _ {k}} (X _ {k}).} \end{array}
$$

For the second claim, we argue that the diagram 

$$
\begin{array}{c} [ 0, N _ {1}) \times \dots \times [ 0, N _ {1}) \xrightarrow {\operatorname{colex} _ {S _ {1}} ^ {- 1} \times \cdots \times \operatorname{colex} _ {S _ {k}} ^ {- 1}} [ 0, S _ {1}) \times \dots \times [ 0, S _ {1}) \\ \operatorname{colex} _ {(N _ {1}, \ldots , N _ {k})} ^ {- 1} \Bigg | \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \cong \Bigg | \qquad \qquad \qquad \qquad \qquad \varphi_ {L _ {1}} + \dots + \varphi_ {L _ {k}} \\ [ 0, N _ {1} \dots N _ {k}) \xrightarrow {\operatorname{colex} _ {S _ {1} * \cdots * S _ {k}} ^ {- 1}} [ 0, S _ {1} * \dots * S _ {k}) \xrightarrow {\varphi_ {L _ {1} * \cdots * L _ {k}}} \mathbb {Z} \end{array}
$$

commutes. The left-hand square commutes since colexicographic isomorphisms are associative, and the right-hand triangle commutes by the first claim. □ 

We can describe the important attributes of a concatenated layout as follows. 

Proposition 2.1.3.41. Suppose $L _ { 1 } , \ldots , L _ { k }$ are flat layouts. Then 

1. the rank of $L _ { 1 } \star \cdots \star L _ { k }$ is 

$$
\operatorname{rank} \left(L _ {1} \star \dots \star L _ {k}\right) = \sum_ {i = 1} ^ {k} \operatorname{rank} \left(L _ {i}\right),
$$

2. the size of $L _ { 1 } \star \cdots \star L _ { k }$ is 

$$
\operatorname{size} \left(L _ {1} \star \dots \star L _ {k}\right) = \prod_ {i = 1} ^ {k} \operatorname{size} \left(L _ {i}\right),
$$

3. the cosize of $L _ { 1 } \star \cdots \star L _ { k }$ is 

$$
\operatorname{cosize} \left(L _ {1} \star \dots \star L _ {k}\right) = 1 - k + \sum_ {i = 1} ^ {k} \operatorname{cosize} \left(L _ {i}\right).
$$

Proof. Let’s write $L _ { i } = S _ { i } : D _ { i }$ for each $1 \leq i \leq k$ . For 1, we compute 

$$
\operatorname{rank} \left(L _ {1} \star \dots \star L _ {k}\right) = \operatorname{len} \left(S _ {1} \star \dots \star S _ {k}\right) = \sum_ {i = 1} ^ {k} \operatorname{len} \left(S _ {i}\right) = \sum_ {i = 1} ^ {k} \operatorname{rank} \left(L _ {i}\right).
$$

For 2, we compute 

$$
\operatorname{size} \left(L _ {1} \star \dots \star L _ {k}\right) = \operatorname{size} \left(S _ {1} \star \dots \star S _ {k}\right) = \prod_ {i = 1} ^ {k} \operatorname{size} \left(S _ {i}\right) = \prod_ {i = 1} ^ {k} \operatorname{size} \left(L _ {i}\right).
$$

For 3, we compute 

$$
\begin{array}{l} \text {cosize} (L _ {1} \star \dots \star L _ {k}) = 1 + \max (\Phi_ {L _ {1} \star \dots \star L _ {k}}) \\ \qquad = 1 + \sum_ {i = 1} ^ {k} \max (\Phi_ {L _ {i}}) \\ \qquad = 1 - k + (1 + \max (\Phi_ {L _ {1}})) + \dots + (1 + \max (\Phi_ {L _ {1}})) \\ \qquad = 1 - k + \text {cosize} (L _ {1}) + \dots + \text {cosize} (L _ {k}). \end{array}
$$

where we have used our identification of $\Phi _ { L _ { 1 } \star \cdots \star L _ { k } }$ from Proposition 2.1.3.40. 

## 2.1.4 Flat coalesce

We have seen that the layout function $\Phi _ { L }$ of a flat layout L is an important invariant. In many cases, we are only interested in the layout function $\Phi _ { L }$ , and are free to work with any layout whose layout function is $\Phi _ { L }$ . The flat coalesce operation 

$$
L \mapsto \operatorname{coal} ^ {\flat} (L)
$$

provides us with the simplest flat layout whose layout function is $\Phi _ { L }$ (see Proposition 2.1.4.19). 

We begin by defining the notion of a coalesced flat layout. 

Definition 2.1.4.1. Suppose $L = ( s _ { 1 } , \ldots , s _ { m } ) : ( d _ { 1 } , \ldots , d _ { m } )$ is a flat layout. We say L is coalesced if 

1. for any $1 \leq i \leq m$ , we have $s _ { i } \neq 1$ , and 

2. for any $1 \leq i < m$ , we have $s _ { i } d _ { i } \neq d _ { i + 1 }$ 

Example 2.1.4.2. The flat layout 

$$
L = (3, 5, 2): (7, 2 1, 4)
$$

is not coalesced because $3 \cdot 7 = 2 1$ 

Example 2.1.4.3. The flat layout 

$$
L = (2, 7, 6): (1, 3, 1 0)
$$

is coalesced. 

Example 2.1.4.4. The empty layout $E = ( ) : ( )$ is coalesced. 

Example 2.1.4.5. If $L = ( s ) : ( d )$ and $s \neq 1$ , then L is coalesced. 

Example 2.1.4.6. If $L = ( s _ { 1 } , \ldots , s _ { m } ) : ( d _ { 1 } , \ldots , d _ { m } )$ is a column-major layout with rank $( L ) > 1$ , then L is not coalesced, since for any $1 \leq i < m$ , we have 

$$
s _ {i} d _ {i} = s _ {i} (s _ {1} \dots s _ {i - 1}) = s _ {1} \dots s _ {i} = d _ {i + 1}.
$$

Example 2.1.4.7. If $L = ( s _ { 1 } , \ldots , s _ { m } ) : ( d _ { 1 } , \ldots , d _ { m } ) $ is a row-major layout with $s _ { i } ~ > ~ 1$ for all $1 \leq i \leq m$ , then L is coalesced: If $1 \leq i < m$ , then 

$$
s _ {i} d _ {i} = s _ {i} s _ {i + 1} \dots s _ {m} > s _ {i + 2} \dots s _ {m} = d _ {i + 1}.
$$

Example 2.1.4.8. A flat layout of the form 

$$
L = (s _ {1}, \ldots , s _ {m}): (0, \ldots , 0)
$$

is coalesced if and only if $m \leq 1$ 

If L is a flat layout, then we may obtain a coalesced layout ${ \mathsf { c o a l } } ^ { \flat } ( L )$ with the same layout function as L by removing modes with $s _ { i } = 1$ , and combining modes with $s _ { i } d _ { i } = d _ { i + 1 }$ . More precisely, we make the following construction. 

Construction 2.1.4.9. Suppose L is a flat layout, and write 

$$
\operatorname{squeeze} (L) = \left(s _ {1}, \dots , s _ {m}\right): \left(d _ {1}, \dots , d _ {m}\right).
$$

Let ∼ be the equivalence relation on ⟨m⟩ generated by $i \sim i + 1$ if 

$$
s _ {i} d _ {i} = d _ {i + 1}.
$$

The quotient $\langle m \rangle / \sim$ is ordered by $[ i ] \leq [ i ^ { \prime } ] \mathrm { i f } i \leq i ^ { \prime }$ , so we may identify $\langle m \rangle / \sim \mathrm { w i t h } \ \langle \bar { m } \rangle$ , where m¯ is the size of $\langle m \rangle / \sim . \mathrm { ~ I f ~ } i \in \langle \bar { m } \rangle$ corresponds to the equivalence class 

$$
I = \{i ^ {\prime}, i ^ {\prime} + 1, \dots , i ^ {\prime} + k \} \in \langle m \rangle / \sim ,
$$

then we define integers $\bar { s } _ { i }$ and ${ \bar { d } } _ { i }$ as 

$$
\bar {s} _ {i} = s _ {i ^ {\prime}} s _ {i ^ {\prime} + 1} \dots s _ {i ^ {\prime} + k}
$$

and 

$$
\bar {d} _ {i} = d _ {i ^ {\prime}},
$$

and define 

$$
\mathsf {c o a l} ^ {\flat} (L) = (\bar {s} _ {1}, \dots , \bar {s} _ {\bar {m}}): (\bar {d} _ {1}, \dots , \bar {d} _ {\bar {m}}).
$$

Observation 2.1.4.10. Examining the definition, we could equivalently define ${ \mathsf { c o a l } } ^ { \flat } ( L )$ to be the flat layout obtained from 

$$
\operatorname{squeeze} (L) = \left(s _ {1}, \dots , s _ {m}\right): \left(d _ {1}, \dots , d _ {m}\right)
$$

by iteratively performing the operation 

$$
s _ {i}, s _ {i + 1}: d _ {i}, s _ {i} d _ {i} \quad \rightsquigarrow \quad s _ {i} s _ {i + 1}: d _ {i}
$$

until the result is coalesced. 

Example 2.1.4.11. If $L = ( 2 , 2 , 2 , 2 , 2 ) : ( 8 , 1 6 , 1 0 2 4 , 2 0 4 8 , 4 0 9 6 ) .$ , then 

$$
\operatorname{coal} ^ {\flat} (L) = (4, 8): (8, 1 0 2 4).
$$

Example 2.1.4.12. If $L = ( 3 , 4 , 1 , 5 ) : ( 1 , 8 , 3 , 3 2 )$ , then 

$$
\operatorname{coal} ^ {\flat} (L) = (3, 2 0): (1, 8).
$$

Example 2.1.4.13. If $L = ( s _ { 1 } , \ldots , s _ { m } ) : ( d _ { 1 } , \ldots , d _ { m } )$ is column-major, and not all $s _ { i }$ are equal to 1, then 

$$
\operatorname{coal} ^ {\flat} (L) = \left(s _ {1} \dots s _ {m}\right): (1).
$$

Example 2.1.4.14. If L is row-major, then 

$$
\operatorname{coal} ^ {\flat} (L) = \operatorname{squeeze} (L).
$$

Let’s justify that the operation $L \mapsto { \mathsf { c o a l } } ^ { \flat } ( L )$ results in a coalesced layout. 

Lemma 2.1.4.15. $I f L$ is a flat layout, then ${ \mathsf { c o a l } } ^ { \flat } ( L )$ is coalesced. 

Proof. Borrowing the notation of Construction 2.1.4.9, let 

$$
\operatorname{squeeze} (L) = \left(s _ {1}, \dots , s _ {m}\right): \left(d _ {1}, \dots , d _ {m}\right)
$$

and let 

$$
\mathsf {c o a l} ^ {\flat} (L) = (\bar {s} _ {1}, \dots , \bar {s} _ {\bar {m}}): (\bar {d} _ {1}, \dots , \bar {d} _ {\bar {m}}).
$$

We want to show that ${ \mathsf { c o a l } } ^ { \flat } ( L )$ is coalesced. Suppose $1 \leq i \leq \bar { m }$ . Then i corresponds to a $\left( \mathrm { n o n - e m p t y } \right)$ equivalence class $I \in \langle m \rangle / \sim$ , and 

$$
\bar {s} _ {i} = \prod_ {i ^ {\prime} \in I} s _ {i ^ {\prime}}
$$

is a product of integers $s _ { i ^ { \prime } } > 1$ , so $\bar { s } _ { i } > 1$ 

Suppose $1 \leq i < \bar { m }$ . We claim that $\bar { s } _ { i } \bar { d } _ { i } \neq \bar { d } _ { i + 1 }$ . Suppose i corresponds to the equivalence class 

$$
\{i ^ {\prime}, i ^ {\prime} + 1, \dots , i ^ {\prime} + k \} \in \langle m \rangle / \sim ,
$$

and suppose i + 1 corresponds to the equivalence class 

$$
\left\{i ^ {\prime} + k + 1, i ^ {\prime} + k + 2, \dots , i ^ {\prime} + k + \ell \right\} \in \langle m \rangle / \sim .
$$

Then by using the equalities $s _ { i ^ { \prime } + t } d _ { i ^ { \prime } + t } = d _ { i ^ { \prime } + t + 1 }$ for $0 \leq t < k$ , we may write 

$$
\begin{array}{r l} & {\bar {s} _ {i} \bar {d} _ {i} = \bar {d} _ {i} \bar {s} _ {i} = d _ {i ^ {\prime}} s _ {i ^ {\prime}} s _ {i ^ {\prime} + 1} \dots s _ {i ^ {\prime} + k}} \\ & {\qquad = d _ {i ^ {\prime} + 1} s _ {i ^ {\prime} + 1} \dots s _ {i ^ {\prime} + k}} \\ & {\qquad \vdots} \\ & {\qquad = d _ {i ^ {\prime} + k} s _ {i ^ {\prime} + k}} \\ & {\qquad = s _ {i ^ {\prime} + k} d _ {i ^ {\prime} + k}} \end{array}
$$

and since $i ^ { \prime } + k$ and $i ^ { \prime } + k + 1$ do not lie in the same equivalence class, we have 

$$
\bar {s} _ {i} \bar {d} _ {i} = s _ {i ^ {\prime} + k} d _ {i ^ {\prime} + k} \neq d _ {i ^ {\prime} + k + 1} = \bar {d} _ {i + 1}.
$$

Example 2.1.4.16. If L is coalesced, then ${ \mathsf { c o a l } } ^ { \flat } ( L ) = L$ . In particular, this implies that $\mathsf { c o a l } ^ { \flat } ( - )$ is an idempotent operation: 

$$
\operatorname{coal} ^ {\flat} \left(\operatorname{coal} ^ {\flat} (L)\right) = \operatorname{coal} ^ {\flat} (L).
$$

Next, we argue that coalescing a flat layout leaves the layout function unchanged. 

Lemma 2.1.4.17. If L is a flat layout, then $\Phi _ { \mathsf { c o a l } ^ { \flat } ( L ) } = \Phi _ { L }$ 

Proof. By Observation 2.1.4.10, it sufices to show that replacing an instance of $s _ { i } , s _ { i + 1 } : d _ { i } , s _ { i } d _ { i }$ with $s _ { i } s _ { i + 1 } : d _ { i }$ leaves the layout function unchanged. Suppose 

$$
L = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m})
$$

is a flat layout, and there exists some $1 \leq i < m$ such that $d _ { i + 1 } = s _ { i } d _ { i }$ . Let 

$$
L ^ {\prime} = (s _ {1} ^ {\prime}, \ldots , s _ {m - 1} ^ {\prime}): (d _ {1} ^ {\prime}, \ldots , d _ {m - 1} ^ {\prime})
$$

denote the flat layout obtained from L by combining the ith and (i + 1)th modes of L. More precisely, we have 

$$
s _ {j} ^ {\prime} = \left\{ \begin{array}{l l} s _ {j} & j <   i \\ s _ {i} s _ {i + 1} & j = i \\ s _ {j + 1} & i <   j <   m, \end{array} \right. \quad \text { and } \quad d _ {j} ^ {\prime} = \left\{ \begin{array}{l l} d _ {j} & j \leq i \\ d _ {j + 1} & i <   j <   m. \end{array} \right.
$$

The layout function for L is given by 

$$
\Phi_ {L} (x) = x _ {1} d _ {1} + \dots + x _ {m} d _ {m}
$$

where $x _ { j } = \left\lfloor { \frac { x } { s _ { 1 } \cdot \cdot \cdot s _ { j - 1 } } } \right\rfloor$ mod $s _ { j } .$ , and the layout function for L<sup>′</sup> is given by 

$$
\Phi_ {L ^ {\prime}} (x) = x _ {1} ^ {\prime} d _ {1} ^ {\prime} + \dots + x _ {m - 1} ^ {\prime} d _ {m - 1} ^ {\prime}
$$

where $x _ { j } ^ { \prime } = \left\lfloor { \frac { x } { s _ { 1 } ^ { \prime } \cdot \cdot \cdot s _ { j - 1 } ^ { \prime } } } \right\rfloor$ mod $s _ { j } ^ { \prime } .$ . We observe that 

$$
x _ {j} ^ {\prime} = \left\{ \begin{array}{l l} x _ {j} & j <   i \\ x _ {i} + x _ {i + 1} s _ {i} & j = i \\ x _ {j + 1} & i <   j <   m, \end{array} \right.
$$

and so 

$$
\begin{array}{r l} \Phi_ {L} (x) & = x _ {1} d _ {1} + \dots + x _ {m} d _ {m} \\ & = x _ {1} d _ {1} + \dots + x _ {i} d _ {i} + x _ {i + 1} s _ {i} d _ {i} + \dots + x _ {m} d _ {m} \\ & = x _ {1} d _ {1} + \dots + (x _ {i} + x _ {i + 1} s _ {i}) d _ {i} + \dots + x _ {m} d _ {m} \\ & = x _ {1} ^ {\prime} d _ {1} ^ {\prime} + \dots + x _ {m - 1} ^ {\prime} d _ {m - 1} ^ {\prime} \\ & = \Phi_ {L ^ {\prime}} (x). \end{array}
$$

We can use the coalesce operation to characterize when two flat layouts have the same layout function. Proposition 2.1.4.18. Suppose A and B are flat layouts. Then 

$$
\Phi_ {A} = \Phi_ {B} \quad \Leftrightarrow \quad \operatorname{coal} ^ {\flat} (A) = \operatorname{coal} ^ {\flat} (B).
$$

Proof. If $\mathsf { c o a l } ^ { \flat } ( A ) = \mathsf { c o a l } ^ { \flat } ( B )$ , then by Lemma 2.1.4.17, we have 

$$
\Phi_ {A} = \Phi_ {\mathsf {c o a l} ^ {\flat} (A)} = \Phi_ {\mathsf {c o a l} ^ {\flat} (B)} = \Phi_ {B}.
$$

Inversely, suppose that coa ${ \mathsf { \Omega } } ^ { \flat } ( A ) \neq { \mathsf { c o a l } } ^ { \flat } ( B )$ . We will argue that $\Phi _ { A } \neq \Phi _ { B }$ . Let’s write 

$$
\begin{array}{l} \mathsf {c o a l} ^ {\flat} (A) = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m}), \\ \mathsf {c o a l} ^ {\flat} (B) = (t _ {1}, \ldots , t _ {n}): (e _ {1}, \ldots , e _ {m}). \end{array}
$$

If one of $m , n$ is nonzero and the other is 0, then clearly $\Phi _ { A } \neq \Phi _ { B }$ , so we may assume $m , n \geq 1$ . Let i denote the least integer such that $( s _ { i } , d _ { i } ) \neq ( t _ { i } , e _ { i } )$ . Then, in particular, we have $s _ { 1 } \cdot \cdot \cdot s _ { j } = t _ { 1 } \cdot \cdot \cdot t _ { j }$ for any $j < i .$ There are two cases to consider: 

• (Case 1): Suppose $d _ { i } \neq e _ { i } .$ . Let $N = s _ { 1 } \cdot \cdot \cdot s _ { i - 1 } = t _ { 1 } \cdot \cdot \cdot t _ { i - 1 }$ . Then 

$$
\Phi_ {\mathsf {c o a l} ^ {\flat} (A)} (N) = d _ {i} \neq e _ {i} = \Phi_ {\mathsf {c o a l} ^ {\flat} (B)} (N)
$$

so $\Phi _ { \mathsf { c o a l } ^ { \flat } ( A ) } \neq \Phi _ { \mathsf { c o a l } ^ { \flat } ( B ) }$ , and hence $\Phi _ { A } \neq \Phi _ { B }$ 

• (Case 2): Suppose $d _ { i } = e _ { i } .$ , so that $s _ { i } \neq t _ { i }$ . Without loss of generality we may assume $s _ { i } < t _ { i }$ Let $N = s _ { 1 } \cdot \cdot \cdot s _ { i } = ( t _ { 1 } \cdot \cdot \cdot t _ { i - 1 } ) s _ { i }$ . Then 

$$
\Phi_ {\mathsf {c o a l} ^ {\flat} (A)} (N) = d _ {i + 1}
$$

while 

$$
\begin{array}{r} \Phi_ {\mathsf {c o a l} ^ {\flat} (B)} (N) = s _ {i} e _ {i} \\ = s _ {i} d _ {i}, \end{array}
$$

and since coa $^ { \flat } ( A )$ is coalesced, we have $d _ { i + 1 } \neq s _ { i } d _ { i }$ . We deduce that $\Phi _ { \mathsf { c o a l } ^ { \flat } ( A ) } \neq \Phi _ { \mathsf { c o a l } ^ { \flat } ( B ) }$ , and hence $\Phi _ { A } \neq \Phi _ { B }$ 

The previous proposition afords us the following abstract characterization of ${ \mathsf { c o a l } } ^ { \flat } ( L )$ 

Proposition 2.1.4.19. If L is a flat layout, then ${ \mathsf { c o a l } } ^ { \flat } ( L )$ is the unique flat layout of minimal rank whose layout function is $\Phi _ { L }$ 

Proof. Suppose $L ^ { \prime }$ is a flat layout with $\Phi _ { L ^ { \prime } } = \Phi _ { L }$ . Then by Proposition 2.1.4.18, we have 

$$
\operatorname{coal} ^ {\flat} (L) = \operatorname{coal} ^ {\flat} \left(L ^ {\prime}\right),
$$

and it follows that 

$$
\operatorname{rank} \left(\operatorname{coal} ^ {\flat} (L)\right) = \operatorname{rank} \left(\operatorname{coal} ^ {\flat} \left(L ^ {\prime}\right)\right) \leq \operatorname{rank} \left(L ^ {\prime}\right),
$$

where equality holds if and only if 

$$
L ^ {\prime} = \operatorname{coal} ^ {\flat} (L ^ {\prime}) = \operatorname{coal} ^ {\flat} (L).
$$

## 2.1.5 Compact flat layouts

Before treating layout complements, we must define an important family of layouts called compact flat layouts. These are the flat layouts whose layout functions are bijective. In terms of the standard grid diagrams depicting layouts, a flat layout L is compact if each integer $0 \leq i < \mathsf { s i z e } ( L )$ appears exactly once. For instance, the layout 

<table><tr><td>0</td><td>3</td><td>6</td><td>9</td><td>12</td><td>15</td></tr><tr><td>1</td><td>4</td><td>7</td><td>10</td><td>13</td><td>16</td></tr><tr><td>2</td><td>5</td><td>8</td><td>11</td><td>14</td><td>17</td></tr></table>

is compact, while the layouts 

<table><tr><td>0</td><td>6</td><td>12</td><td>18</td><td>24</td><td>30</td></tr><tr><td>2</td><td>8</td><td>14</td><td>20</td><td>26</td><td>32</td></tr><tr><td>4</td><td>10</td><td>16</td><td>22</td><td>28</td><td>34</td></tr></table>

and 

<table><tr><td>0</td><td>2</td><td>4</td><td>6</td><td>8</td><td>10</td></tr><tr><td>1</td><td>3</td><td>5</td><td>7</td><td>9</td><td>11</td></tr><tr><td>2</td><td>4</td><td>6</td><td>8</td><td>10</td><td>12</td></tr></table>


are not compact. More precisely, we have the following definition. 


Definition 2.1.5.1. Suppose L is a flat layout. We say L is compact if 

$$
[ 0, \operatorname{size} (L)) \xrightarrow {\Phi_ {L} ^ {\operatorname{cosize} (L)}} [ 0, \operatorname{cosize} (L))
$$

is an isomorphism. 

Example 2.1.5.2. The flat layout 

$$
L = (2, 2, 2, 2): (1, 2, 4, 8)
$$

is compact. More generally, if L is column-major, then $L$ is compact. 

Example 2.1.5.3. The flat layout 

$$
L = (3, 6 4, 3 2): (2 0 4 8, 3 2, 1)
$$

is compact. More generally, if L is row-major, then L is compact. 

Example 2.1.5.4. The empty layout 

$$
E = (): ()
$$

is compact. 

Example 2.1.5.5. Suppose 

$$
L = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m})
$$

is a flat layout. If there is some mode of L with $s _ { i } > 1$ and $d _ { i } = 0$ , then L is not compact. 

We can give an explicit characterization of compact layouts as follows. 

Proposition 2.1.5.6. Suppose L is a flat layout, and write 

$$
\operatorname{squeeze} (L) = \left(s _ {1}, \dots , s _ {m}\right): \left(d _ {1}, \dots , d _ {m}\right).
$$

then L is compact if and only if there exists a permutation $\sigma \in \Sigma _ { m }$ such that 

$$
d _ {\sigma (i)} = s _ {\sigma (1)} \dots s _ {\sigma (i - 1)}
$$

for all $1 \leq i \leq m$ . In other words, L is compact if and only if there exists a permutation $\sigma \in \Sigma _ { m }$ such that squeeze $( L ) ^ { \sigma }$ is column-major. 

Proof. Suppose L is a flat layout, and write 

$$
\operatorname{squeeze} (L) = \left(s _ {1}, \dots , s _ {m}\right): \left(d _ {1}, \dots , d _ {m}\right).
$$

Suppose first that L is compact, so there exists a permutation $\sigma \in \Sigma _ { m }$ such that $d _ { \sigma ( i ) } = s _ { \sigma ( 1 ) } \cdot \cdot \cdot s _ { \sigma ( i - 1 ) }$ for each $1 \leq i \leq m$ . If we write $S ^ { \sigma } = ( s _ { \sigma ( 1 ) } , \ldots , s _ { \sigma ( m ) } )$ , then we can write $\Phi _ { L } ^ { \mathsf { c o s i z e } ( L ) }$ as the composite 

$$
[ 0, \operatorname{size} (L)) \xrightarrow {\operatorname{colex} _ {S} ^ {- 1}} [ 0, S) \xrightarrow {\cong} [ 0, S ^ {\sigma}) \xrightarrow {\operatorname{colex} _ {S ^ {\sigma}}} [ 0, \operatorname{cosize} (L))
$$

$$
(x _ {1}, \dots , x _ {m}) \longmapsto (x _ {\sigma (1)}, \dots , x _ {\sigma (m)})
$$

and since each of these maps is an isomorphism, so is the composite $\Phi _ { L } ^ { \mathsf { c o s i z e } ( L ) }$ 

Conversely, suppose that $\Phi _ { L } ^ { \mathsf { c o s i z e } ( L ) }$ is an isomorphism. First, we note that the strides $d _ { 1 } , \ldots , d _ { m }$ must be pairwise distinct: Suppose $d _ { i } = d _ { j }$ , and let $\delta _ { i } ^ { m }$ and $\delta _ { j } ^ { m }$ denote the tuples whose ith $\left( \mathrm { r e s p . ~ } j \mathrm { t h } \right)$ entry is 1, and all of whose other entries are 0. These tuples satisfy 

$$
\delta_ {i} ^ {m} \cdot (d _ {1}, \ldots , d _ {m}) = d _ {i} = d _ {j} = \delta_ {j} ^ {m} \cdot (d _ {1}, \ldots , d _ {m}),
$$

and since $\Phi _ { L } ^ { \mathsf { c o s i z e } ( L ) }$ is injective, we must have $i = j$ . Given that the strides $d _ { 1 } , \ldots , d _ { m }$ are pairwise distinct, let $\sigma \in \Sigma _ { m }$ be the permutation such that 

$$
d _ {\sigma (1)} <   d _ {\sigma (2)} <   \dots <   d _ {\sigma (m)}.
$$

We will argue by induction on $i \geq 1$ that $d _ { \sigma ( i ) } = s _ { \sigma ( 1 ) } \cdot \cdot \cdot s _ { \sigma ( i - 1 ) }$ . For the base case $i = 1$ , we note that 1 is in the image of $\Phi _ { L } ^ { \mathsf { c o s i z e } ( L ) }$ , and the smallest non-zero value of $\Phi _ { L } ^ { \mathsf { c o s i z e } ( L ) }$ is $d _ { \sigma ( 1 ) }$ , so it follows that $d _ { \sigma ( 1 ) } = 1$ . Suppose $i > 1$ , and that we have proved the claim for all $j < i$ . Consider the stride $d _ { \sigma ( i ) }$ We know that there is no tuple of the form $( x _ { 1 } , \ldots , x _ { i - 1 } , 0 , \ldots , 0 ) ^ { \sigma }$ such that 

$$
(x _ {1}, \dots , x _ {i - 1}, 0, \dots , 0) ^ {\sigma} \cdot (d _ {1}, \dots , d _ {m}) = s _ {\sigma (1)} \dots s _ {\sigma (i - 1)},
$$

since the largest possible value of such an expression is 

$$
\sum_ {j = 1} ^ {i - 1} (s _ {\sigma (j)} - 1) (s _ {\sigma (1)} \dots s _ {\sigma (j - 1)}) = s _ {\sigma (1)} \dots s _ {\sigma (i - 1)} - 1.
$$

Since $\Phi _ { L } ^ { \mathsf { c o s i z e } } ( L )$ is surjective, and $d _ { \sigma ( i ) } < d _ { \sigma ( i + 1 ) } < \cdot \cdot \cdot < d _ { \sigma ( m ) }$ , it follows that the next largest value of $\Phi _ { L } ^ { \mathsf { c o s i z e } ( L ) } \ \mathrm { i s } \ d _ { \sigma ( i ) }$ , so we must have $d _ { \sigma ( i ) } = s _ { \sigma ( 1 ) } \cdot \cdot \cdot s _ { \sigma ( i - 1 ) }$ , as claimed. □ 

We conclude this section by giving a family of equivalent conditions for a flat layout L to be compact. 

## Proposition 2.1.5.7. Suppose L is a flat layout. Then the following are equivalent.

1. L is compact. 

2. $\mathsf { c o a l } ^ { \flat } ( L )$ is compact. 

3. squeeze(L) is compact. 

4. sort(L) is compact. 

Proof. The equivalence of 1, 2, and 3, follows from the fact that 

$$
\Phi_ {L} = \Phi_ {\text { coal } ^ {\flat} (L)} = \Phi_ {\text { squeeze } (L)}.
$$

It remains to prove that L is compact if and only if ${ \mathsf { s o r t } } ( L )$ is compact. Using the fact that 

$$
\operatorname{squeeze} (\operatorname{sort} (L)) = \operatorname{sort} (\operatorname{squeeze} (L)),
$$

we have 

$$
\begin{array}{l l} \text {sort} (L) \text {is compact.} & \Leftrightarrow \quad \text {squeeze} (\text {sort} (L)) \text {is compact.} \\ & \Leftrightarrow \quad \text {sort} (\text {squeeze} (L)) \text {is compact.} \end{array}
$$

Now sor $\mathsf { \cdot } ( \mathsf { s q u e e z e } ( L ) ) = \mathsf { s q u e e z e } ( L ) ^ { \tau }$ for some permutation $\tau \in \Sigma _ { m }$ , so there exists a permutation σ such that squeeze(L)<sup>σ</sup> is column-major if and only if there exists a permutation $\sigma ^ { \prime } \in \Sigma _ { m }$ such that sort(squeeze(L)) is column-major, namely $\boldsymbol { \sigma } ^ { \prime } = \boldsymbol { \tau } ^ { - 1 } \boldsymbol { \sigma }$ . It follows that 

sort(squeeze(L)) is compact. ⇔ squeeze(L) is compact. 

⇔ L is compact. 

## 2.1.6 Complements

In this section, we define the notion of complementary flat layouts. Recall from Definition 2.1.5.1 that a flat layout L is compact if the layout function 

$$
\Phi_ {L} ^ {\text { cosize } (L)}: [ 0, \text { size } (L)) \to [ 0, \text { cosize } (L))
$$

is an isomorphism. 

Definition 2.1.6.1. Suppose A and B are flat layouts. We say B is a complement of A, and write $A \perp B ,$ , if the concatenated layout $A \star B$ is compact. 

Example 2.1.6.2. If $A = ( 3 ) : ( 5 )$ and $B = ( 5 ) : ( 1 )$ , then $A \perp B$ since 

$$
A \star B = (3, 5): (5, 1)
$$

is compact. 

<table><tr><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td></tr></table>

<table><tr><td>0</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td></tr><tr><td>5</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td></tr><tr><td>10</td><td>10</td><td>11</td><td>12</td><td>13</td><td>14</td></tr></table>

Example 2.1.6.3. If $A = ( 4 , 2 , 1 0 ) : ( 1 4 0 0 , 2 , 2 0 )$ and $B = ( 2 , 5 , 7 , 2 ) : ( 1 , 4 , 2 0 0 , 5 6 0 0 )$ , then $A \perp B$ since 

$$
A \star B = (4, 2, 1 0, 2, 5, 7, 2): (1 4 0 0, 2, 2 0, 1, 4, 2 0 0, 5 6 0 0)
$$

is compact. 

Example 2.1.6.4. If A is a flat layout and $E = ( ) : ( )$ is the empty layout, then $A \perp A$ if and only if A is compact, since 

$$
A \star E = A.
$$

Example 2.1.6.5. If A and B are flat layouts, then 

$$
A \perp B \quad \Leftrightarrow \quad B \perp A.
$$

Example 2.1.6.6. If A is a flat layout, then $A \perp A$ if and only if size $( A ) = 1$ 

Observation 2.1.6.7. In order for A to admit a complement, it is necessary that $\Phi _ { A }$ is injective. There do, however, exist flat layouts A such that $\Phi _ { A }$ is injective, and A does not admit a complement. For example, consider the layout 

$$
A = (2, 2): (1, 3).
$$

The layout function of A is injective since 

$$
\Phi_ {A} (0) = 0, \Phi_ {A} (1) = 1, \Phi_ {A} (2) = 3, \text { and } \Phi_ {A} (3) = 4,
$$

but A does not admit a complement: Suppose 

$$
B = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m})
$$

is any other flat layout. If there does not exist a tuple 

$$
\left(x _ {1}, x _ {2}, y _ {1}, \dots , y _ {m}\right) \in [ 0, 2) \times [ 0, 2) \times [ 0, s _ {1}) \times \dots \times [ 0, s _ {m})
$$

such that $\varphi _ { A \star B } ( x _ { 1 } , x _ { 2 } , y _ { 1 } , \dots , y _ { m } ) = 2$ , then $A \star B$ is not compact. Suppose otherwise that there is such a tuple $( x _ { 1 } , x _ { 2 } , y _ { 1 } , \dots , y _ { m } )$ . Then $\varphi _ { B } ( y _ { 1 } , \dots , y _ { m } ) \in \{ 0 , 1 , 2 \}$ 

• (Case 1): ${ \mathrm { I f ~ } } \varphi _ { B } ( y _ { 1 } , \ldots , y _ { m } ) = 0 .$ , then 

$$
\varphi_ {A \star B} (0, 0, 0, \dots , 0) = 0 = \varphi_ {A \star B} (0, 0, y _ {1}, \dots , y _ {m}).
$$

• (Case 2): If $\varphi _ { B } ( y _ { 1 } , \dots , y _ { m } ) = 1$ , then 

$$
\varphi_ {A \star B} (1, 0, 0, \dots , 0) = 1 = \varphi_ {A \star B} (0, 0, y _ {1}, \dots , y _ {m}).
$$

• (Case 3): If $\varphi _ { B } ( y _ { 1 } , \dots , y _ { m } ) = 2 ,$ then 

$$
\varphi_ {A \star B} (0, 1, 0, \dots , 0) = 3 = \varphi_ {A \star B} (1, 0, y _ {1}, \dots , y _ {m}).
$$

In any case, we deduce that $\varphi _ { A \star B }$ is not injective, hence neither is $\Phi _ { A \star B }$ . This implies that $A \star B$ is not compact, so B is not a complement of A. 

Observation 2.1.6.8. Complements are not unique. For example, if 

$$
A = (8, 8): (2, 3 2),
$$

then each of the layouts 

$$
\begin{array}{l} B _ {1} = (2, 2): (1, 1 6) \\ B _ {2} = (2, 2): (1 6, 1) \\ B _ {3} = (5, 2, 2, 1): (2 5 6, 1, 1 6, 0) \end{array}
$$

is a complement of A. Instead, there is a (possibly empty) set 

complements<sup>♭</sup>(A) = {flat layouts $B \mid B$ is a complement of $A \}$ . 

of layouts which are complementary to A. 

It will be useful to provide a family of equivalent conditions for B to be a complement of A (see Proposition 2.1.6.10). In order to do so, we need the following technical lemma, which describes the interplay between concatenation, and the operations squeeze(−), sort(−), and $\mathsf { c o a l } ^ { \flat } ( - )$ 

Lemma 2.1.6.9. Suppose A and B are flat layouts. Then 

1. squeeze $( A \star B ) = { \mathsf { s q u e e z e } } ( A ) \star { \mathsf { s q u e e z e } } ( B )$ *4 

2. sor $\therefore ( A \star B ) = { \mathsf { s o r t } } ( L \star { \mathsf { s o r t } } ( B ) )$ , and 

3. coa $\mathsf { I } ^ { \flat } ( A \star B ) = \mathsf { c o a l } ^ { \flat } ( A \star \mathsf { c o a l } ^ { \flat } ( B ) )$ 

Proof. Write 

$$
\begin{array}{l} A = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m}) \\ B = (t _ {1}, \ldots , t _ {n}): (e _ {1}, \ldots , e _ {n}). \end{array}
$$

If we let $\{ i _ { 1 } < \dots < i _ { m ^ { \prime } } \} \subset \langle m \rangle$ denote the indices with $s _ { i _ { k } } \neq 1$ , and $\{ j _ { 1 } , \dotsc , j _ { n ^ { \prime } } \} \subset \langle n \rangle$ denote the indices with $t _ { j \ell } \neq 1$ , then 

$$
\begin{array}{c} \text {squeeze} (A \star B) = (s _ {i _ {1}}, \ldots , s _ {i _ {m ^ {\prime}}}, t _ {j _ {1}}, \ldots , t _ {j _ {n ^ {\prime}}}): (d _ {i _ {1}}, \ldots , d _ {i _ {m ^ {\prime}}}, e _ {j _ {1}}, \ldots , e _ {j _ {n ^ {\prime}}}) \\ = \text {squeeze} (A) \star \text {squeeze} (B). \end{array}
$$

This proves 1. For 2, we note that for any flat layout L, and any permutation $\sigma \in \Sigma _ { \mathsf { l e n } ( L ) }$ , we have $\mathsf { s o r t } ( L ) = \mathsf { s o r t } ( L ^ { \sigma } )$ . The result follows from the observation that 

$$
A \star \operatorname{sort} (B) = (A \star B) ^ {\sigma}
$$

where σ is a block permutation of the form $\sigma = \mathsf { i d } \times \sigma ^ { \prime } \in \Sigma _ { m } \times \Sigma _ { n } \subset \Sigma _ { m + n }$ . For $3 . ,$ it sufices to show that $A \star B$ and $A \star \mathsf { c o a l } ^ { \flat } ( B )$ have the same layout function. This follows from Proposition 2.1.3.40. 

Proposition 2.1.6.10. Suppose A and B are flat layouts. Then the following are equivalent. 

1. $A \perp B .$ 

2. $B \perp A .$ 

3. $A \perp \mathsf { s q u e e z e } ( B )$ 

4. $A \perp \mathsf { c o a l } ^ { \flat } ( B )$ 

5. $A \perp \mathsf { s o r t } ( B ) .$ 

Proof. We use Proposition 2.1.5.7 and Lemma 2.1.6.9 to prove the equivalence of these conditions. First, we note that sor $\langle { A \star B } \rangle = { \mathsf { s o r t } } ( B \star A )$ , which implies the equivalence of 1 and 2. Next, we note that, by Lemma 2.1.6.9, if op(−) is any of the operations squeeze(−), sort(−), or ${ \mathsf { c o a l } } ^ { \flat } ( - )$ , then 

$$
\mathsf {o p} (A \star B) = \mathsf {o p} (A \star \mathsf {o p} (B)),
$$

and so 

$$
\begin{array}{l l l} A \perp B & \Leftrightarrow & A \star B \text {is compact.} \\ & \Leftrightarrow & \mathsf {o p} (A \star B) \text {is compact.} \\ & \Leftrightarrow & \mathsf {o p} (A \star \mathsf {o p} (B)) \text {is compact.} \\ & \Leftrightarrow & \mathsf {o p} (B) \text {is a complement of} A. \end{array}
$$

We would like to characterize when a flat layout admits a complement. To this end, we make the following definition. 

Definition 2.1.6.11. Suppose A is a flat layout, and write 

$$
\operatorname{sort} (\text { squeeze } (A)) = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m}).
$$

We say A is complementable if for any $1 \leq i < m$ , the integer $s _ { i } d _ { i }$ divides $d _ { i + 1 }$ 

Example 2.1.6.12. The flat layout 

$$
A _ {1} = (4, 1, 1, 4, 4): (6 4, 0, 0, 1, 8)
$$

is complementable, while the flat layout 

$$
A _ {2} = (4, 4, 4): (6 4, 1, 1)
$$

is not complementable. 

Example 2.1.6.13. The flat layout 

$$
A _ {1} = (1 0, 2): (4, 8 0)
$$

complementable, while the flat layout 

$$
A _ {2} = (1 0, 2): (8 0, 4)
$$

is not complementable. 

Example 2.1.6.14. If A is compact, then by Proposition 2.1.5.6, A is complementable. 

Example 2.1.6.15. Suppose 

$$
A = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m})
$$

is a flat layout. If there is any $1 \leq i \leq m$ such that $s _ { i } \neq 1$ and $d _ { i } = 0$ , then A is not complementable. 

If A is complementable, then we can construct a complement of A as follows. 

Construction 2.1.6.16. Suppose A is a flat layout, and write 

$$
\operatorname{sort} (\text { squeeze } (A)) = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m}).
$$

If A is complementable, then we define a flat layout $\mathsf { c o m p } ^ { \flat } ( A )$ as 

$$
\operatorname{comp} ^ {\flat} (A) = \operatorname{coal} ^ {\flat} (C)
$$

where 

$$
C = \left(d _ {1}, \frac {d _ {2}}{s _ {1} d _ {1}}, \frac {d _ {3}}{s _ {2} d _ {2}}, \dots , \frac {d _ {m}}{s _ {m - 1} d _ {m - 1}}\right): \left(1, s _ {1} d _ {1}, s _ {2} d _ {2}, \dots , s _ {m - 1} d _ {m - 1}\right).
$$

Example 2.1.6.17. If $A = ( 8 , 8 ) : ( 1 , 8 )$ , then 

$$
\mathsf {c o m p} ^ {\flat} (A) = (): ()
$$

is the empty layout. More generally, if A is compact, then $\mathsf { c o m p } ^ { \flat } ( A ) = ( ) : ( )$ is the empty layout. 

Example 2.1.6.18. If $A = ( 2 , 2 ) : ( 2 , 8 )$ , then 

$$
\mathsf {c o m p} ^ {\flat} (A) = (2, 2): (1, 4).
$$

Example 2.1.6.19. If $A = ( 3 , 3 , 8 ) : ( 1 6 , 9 6 , 1 )$ , then 

$$
\operatorname{comp} ^ {\flat} (A) = (2, 2): (8, 4 8).
$$

Let’s justify that $\mathsf { c o m p } ^ { \flat } ( A )$ is, in fact, a complement of A. 

Lemma 2.1.6.20. Suppose A is a flat layout. If A is a complementable, then 

$$
A \perp \operatorname{comp} ^ {\flat} (A).
$$

Proof. Lets write 

$$
\operatorname{sort} (\text { squeeze } (A)) = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m}),
$$

so that com $\mathsf { p } ^ { \flat } ( A ) = \mathsf { c o a l } ^ { \flat } ( C )$ where 

$$
C = \left(d _ {1}, \frac {d _ {2}}{s _ {1} d _ {1}}, \frac {d _ {3}}{s _ {2} d _ {2}}, \dots , \frac {d _ {m}}{s _ {m - 1} d _ {m - 1}}\right): \left(1, s _ {1} d _ {1}, s _ {2} d _ {2}, \dots , s _ {m - 1} d _ {m - 1}\right).
$$

By Proposition 2.1.6.10, it sufices to prove that C is a complement of sort(squeeze(A)). This is the case since the concatenation 

$$
\operatorname{sort} (\text { squeeze } (A)) \star C
$$

is equal to 

$$
\left(s _ {1}, \dots , s _ {m}, d _ {1}, \frac {d _ {2}}{s _ {1} d _ {1}}, \dots , \frac {d _ {m}}{s _ {m - 1} d _ {m - 1}}\right): (d _ {1}, \dots , d _ {m}, 1, s _ {1} d _ {1}, \dots , s _ {m - 1} d _ {m - 1}),
$$

and its sorting is equal to 

$$
\left(d _ {1}, s _ {1}, \frac {d _ {2}}{s _ {1} d _ {1}}, \dots , \frac {d _ {m}}{s _ {m - 1} d _ {m - 1}}, s _ {m}\right): (1, d _ {1}, s _ {1} d _ {1}, \dots , s _ {m - 1} d _ {m - 1}, d _ {m})
$$

which is column-major. 

We have shown that if A is complementable, then A admits a complement. Next, we prove that the converse also holds. 

Proposition 2.1.6.21. Suppose A is a flat layout. Then there exists a complement B of A if and only if A is complementable. 

Proof. If A is complementable, then by Lemma 2.1.6.20 the layout $B = \mathsf { c o m p } ^ { \flat } ( A )$ is a complement of A. Conversely, suppose there exists a complement B of A, and consider the flat layout 

$$
\begin{array}{l} L = \text { sort } \big (\text { squeeze } (A) \star \text { squeeze } (B) \big) \\ = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {n}). \end{array}
$$

Since $\Phi _ { L } ( 0 ) = 0 \mathrm { { } } $ , and $\Phi _ { L }$ is injective, we know that $d _ { 1 } \neq 0$ . We will argue that $d _ { i } = s _ { 1 } \cdot \cdot \cdot s _ { i - 1 }$ , i.e., that L is column-major. Since 

$$
\Phi_ {L} ^ {\text { cosize } (L)}: [ 0, \text { size } (L)) \to [ 0, \text { cosize } (L))
$$

is a bijection, we know that 1 is in the image of $\Phi _ { L } .$ , which implies that $d _ { 1 } = 1$ . Suppose $1 < i \le m$ and suppose we have proved that $d _ { j } = s _ { 1 } \cdot \cdot \cdot s _ { j - 1 }$ for all $j < i$ . Consider the stride $d _ { i }$ . We know that there is no $( x _ { 1 } , \ldots , x _ { i - 1 } , 0 , \ldots , 0 )$ such that $( x _ { 1 } , \ldots , x _ { i - 1 } , 0 , \ldots , 0 ) \cdot ( d _ { 1 } , \ldots , d _ { m } ) = s _ { 1 } \cdot \cdot \cdot s _ { i - 1 }$ , since the largest possible value of such an expression is 

$$
\sum_ {j = 1} ^ {i - 1} (s _ {j} - 1) (s _ {1} \dots s _ {j - 1}) = s _ {1} \dots s _ {i} - 1.
$$

Since $\Phi _ { L }$ is surjective, and $d _ { i } \leq d _ { i + 1 } \leq \cdot \cdot \cdot \leq d _ { m }$ , it follows that the next largest value of $\Phi _ { L }$ is $d _ { i } ,$ , so we must have $d _ { i } = s _ { 1 } \cdot \cdot \cdot s _ { i - 1 }$ , as claimed. 

Returning to our main goal, consider the layout 

$$
\operatorname{sort} (\operatorname{squeeze} (A)) = \left(s _ {1} ^ {\prime}, \dots , s _ {m ^ {\prime}} ^ {\prime}\right): \left(d _ {1} ^ {\prime}, \dots , d _ {m ^ {\prime}} ^ {\prime}\right).
$$

Then there exist $j _ { 1 } < \cdots < j _ { m ^ { \prime } }$ such that $s _ { i } ^ { \prime } = s _ { j _ { i } }$ and $d _ { i } ^ { \prime } = d _ { j _ { i } }$ for each $1 \leq i \leq m ^ { \prime }$ . If $1 \leq i < m ^ { \prime }$ then 

$$
s _ {i} ^ {\prime} d _ {i} ^ {\prime} = s _ {j _ {i}} d _ {j _ {i}} = s _ {j _ {i}} s _ {1} \cdot \cdot \cdot s _ {j _ {i} - 1}
$$

divides 

$$
d _ {i + 1} ^ {\prime} = s _ {1} \dots s _ {j _ {i + 1} - 1},
$$

so we conclude that A is complementable. 

Our next goal is to give an abstract characterization of the complement $\mathsf { c o m p } ^ { \flat } ( A )$ of a flat layout A. In order to do so, we need the following lemma. 

Lemma 2.1.6.22. Suppose A is a flat layout. If A is complementable and sorted, then the layout function 

$$
\Phi_ {A}: [ 0, \operatorname{size} (A)) \to \mathbb {Z}
$$

is increasing. 

Proof. Write 

$$
A = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m}).
$$

If $1 \leq k \leq m$ , we claim that 

$$
d _ {1} (s _ {1} - 1) + d _ {2} (s _ {2} - 1) + \dots + d _ {k - 1} (s _ {k - 1} - 1) \leq d _ {k}.
$$

If $k = 1$ , this holds vacuously, and by induction on $k ,$ we have 

$$
\begin{array}{r l} d _ {1} (s _ {1} - 1) + \dots + d _ {k - 2} (s _ {k - 2} - 1) + d _ {k - 1} (s _ {k - 1} - 1) & \leq d _ {k - 1} + d _ {k - 1} (s _ {k} - 1) \\ & = d _ {k - 1} s _ {k - 1} \\ & \leq d _ {k}. \end{array}
$$

Now, suppose we have $x , y \in [ 0 , \mathsf { s i z e } ( A ) )$ with $x \leq y .$ . These integers correspond, under the colexicographic isomorphism, to tuples. 

$$
(x _ {1}, \dots , x _ {m}), (y _ {1}, \dots , y _ {m}) \in [ 0, s _ {1}) \times \dots \times [ 0, s _ {m})
$$

Since $x \leq y ;$ , we know there is some maximal $1 \leq k \leq m$ such that $x _ { k } < y _ { k }$ , and $x _ { \ell } = y _ { \ell }$ for all $k < \ell \leq m$ . Now we can compute 

$$
\begin{array}{l} \Phi_ {A} (x) = d _ {1} x _ {1} + \dots + d _ {k - 1} x _ {k - 1} + d _ {k} x _ {k} + d _ {k + 1} x _ {k + 1} + \dots + d _ {m} x _ {m} \\ \qquad = d _ {1} x _ {1} + \dots + d _ {k - 1} x _ {k - 1} + d _ {k} x _ {k} + d _ {k + 1} y _ {k + 1} + \dots + d _ {m} y _ {m} \\ \qquad \leq d _ {1} (s _ {1} - 1) + \dots + d _ {k - 1} (s _ {k - 1} - 1) + d _ {k} x _ {k} + d _ {k + 1} y _ {k + 1} + \dots + d _ {m} y _ {m} \\ \qquad \leq d _ {k} + d _ {k} x _ {k} + d _ {k + 1} y _ {k + 1} + \dots + d _ {m} y _ {m} \\ \qquad = d _ {k} (x _ {k} + 1) + d _ {k + 1} y _ {k + 1} + \dots + d _ {m} y _ {m} \\ \qquad \leq d _ {k} y _ {k} + d _ {k + 1} y _ {k + 1} + \dots + d _ {m} y _ {m} \\ \qquad \leq d _ {1} y _ {1} + \ldots d _ {m} y _ {m} \\ \qquad = \Phi_ {A} (y). \end{array}
$$

Proposition 2.1.6.23. Suppose A and B are flat layouts. If 

1. $A \perp B ,$ 

2. $\mathsf { s i z e } ( B ) = \mathsf { s i z e } ( \mathsf { c o m p } ^ { \flat } ( A ) ) .$ 4 

3. B is coalesced, and 

4. B is sorted, 

then $B = \mathsf { c o m p } ^ { \flat } ( A )$ 

Proof. Conditions 1 and 2 imply that $\Phi _ { B }$ and $\Phi _ { \mathsf { c o m p } ^ { \flat } ( A ) }$ have the same image. Since B and $\mathsf { c o m p } ^ { \flat } ( A )$ are sorted, we know by Lemma 2.1.6.22 that $\Phi _ { B }$ and $\Phi _ { \mathsf { c o m p } ^ { \flat } ( A ) }$ are increasing. Combining these two facts, it follows that $\Phi _ { B } = \Phi _ { \mathsf { c o m p } ^ { \flat } ( A ) }$ . Proposition 2.1.4.18 and condition 3 then imply that 

$$
B = \operatorname{coal} ^ {\flat} (B) = \operatorname{coal} ^ {\flat} (\operatorname{comp} ^ {\flat} (A)) = \operatorname{comp} ^ {\flat} (A).
$$

Definition 2.1.6.24. Suppose A and B are flat layouts, and N is a positive integer. We say B is a N-complement of A if B is a complement of A and 

$$
\operatorname{size} (A) \cdot \operatorname{size} (B) = N.
$$

Definition 2.1.6.25. Suppose A is a flat layout, and write 

$$
\operatorname{sort} (\text { squeeze } (A)) = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m}).
$$

We say A is N-complementable if 

1. for all $1 \leq i < m$ , the integer $s _ { i } d _ { i }$ divides $d _ { i + 1 }$ , and 

2. the integer $s _ { m } d _ { m }$ divides N. 

Observation 2.1.6.26. If A is complementable, and $s _ { m } : d _ { m }$ is the last mode in the layout sort(squeeze(A)), then A is N-complementable exactly when N is a positive integer multiple of $s _ { m } d _ { m }$ 

Observation 2.1.6.27. N-complements are not unique. For example, if $A = ( 2 , 2 ) : ( 1 , 5 0 )$ and $N = 1 0 0$ , then each of the layouts $B _ { 1 } = ( 2 5 ) : ( 2 )$ , and $B _ { 2 } = ( 5 , 5 ) : ( 2 , 1 0 )$ is a N-complement of A. As a more general example, if B is a N-complement of A, then ${ \mathsf { c o a l } } ^ { \flat } ( B )$ is also a N-complement of A. 

Remark 2.1.6.28. Suppose A is a flat layout and $B _ { 1 }$ and $B _ { 2 }$ are N-complements of A. Then the layout functions $\Phi _ { B _ { 1 } }$ and $\Phi _ { B _ { 2 } }$ need not be equal, but they necessarily have the same image. For example, if $A = ( 4 ) : ( 6 3 )$ and $N = 2 5 2$ then $B _ { 1 } = ( 7 , 9 ) : ( 1 , 7 )$ and $B _ { 2 } = ( 9 , 7 ) : ( 7 , 1 )$ are N-complements of A, and $\Phi _ { B _ { 1 } } \neq \Phi _ { B _ { 2 } }$ , since 

$$
\Phi_ {B _ {1}} (1) = 1 \neq 7 = \Phi_ {B _ {2}} (1).
$$

As a more general example, if B is a N-complement of A, then sort(B) is also a N-complement of A. 

Construction 2.1.6.29. Suppose A is a flat layout, N is a positive integer, and A is N-complementable. If we write 

$$
\operatorname{sort} (\text { squeeze } (A)) = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m}),
$$

then we define a flat layout $\mathsf { c o m p } ^ { \flat } ( A , N )$ by 

$$
\operatorname{comp} ^ {\flat} (A, N) = \operatorname{coal} ^ {\flat} (C)
$$

where 

$$
C = \left(d _ {1}, \frac {d _ {2}}{s _ {1} d _ {1}}, \frac {d _ {3}}{s _ {2} d _ {2}}, \dots , \frac {N}{s _ {m} d _ {m}}\right): \left(1, s _ {1} d _ {1}, s _ {2} d _ {2}, \dots , s _ {m} d _ {m}\right).
$$

Example 2.1.6.30. If $A = ( 3 , 1 0 ) : ( 8 0 , 4 )$ and $N = 2 4 0 0$ , then 

$$
\mathsf {c o m p} ^ {\flat} (A, N) = (4, 2, 1 0): (1, 4 0, 2 4 0).
$$

Lemma 2.1.6.31. Suppose A is a flat layout, N is a positive integer, and A is N-complementable. Then comp<sup>♭</sup>(A, N) is a N-complement of A. 

Proof. Lets write 

$$
\operatorname{sort} (\text { squeeze } (A)) = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m}),
$$

so that com $\mathfrak { o } ^ { \flat } ( A , N ) = \mathsf { c o a l } ^ { \flat } ( C )$ where 

$$
C = \left(d _ {1}, \frac {d _ {2}}{s _ {1} d _ {1}}, \frac {d _ {3}}{s _ {2} d _ {2}}, \dots , \frac {N}{s _ {m} d _ {m}}\right): (1, s _ {1} d _ {1}, s _ {2} d _ {2}, \dots , s _ {m} d _ {m}).
$$

First, we compute 

$$
\begin{array}{l} \operatorname{size} (A) \cdot \operatorname{size} (B) = \left(\prod_ {i = 1} ^ {m} s _ {i}\right) \cdot \left(d _ {1} \cdot \left(\prod_ {i = 2} ^ {m} \frac {d _ {i}}{s _ {i - 1} d _ {i - 1}}\right) \cdot \frac {N}{s _ {m} d _ {m}}\right) \\ = \frac {\left(\prod_ {i = 1} ^ {m} s _ {i}\right) \left(\prod_ {i = 1} ^ {m} d _ {i}\right)}{\left(\prod_ {i = 1} ^ {m} s _ {i} d _ {i}\right)} \cdot N \\ = N. \end{array}
$$

We need to check that $A \star B$ is compact. Equivalently, we need to check that $\Phi _ { A \star B } ^ { N }$ is an isomorphism. By Lemma 2.1.5.6, it sufices to prove that 

$$
\text { squeeze } (A) \star \text { squeeze } (B)
$$

is compact. This is the case since this layout is equal to 

$$
\left(s _ {1}, \dots , s _ {m}, d _ {1}, \frac {d _ {2}}{s _ {1} d _ {1}}, \dots , \frac {N}{s _ {m} d _ {m}}\right): (d _ {1}, \dots , d _ {m}, 1, s _ {1} d _ {1}, \dots , s _ {m} d _ {m})
$$

and so its sorting 

$$
\operatorname{sort} (\text { squeeze } (A) \star \text { squeeze } (B))
$$

is equal to 

$$
\left(d _ {1}, s _ {1}, \frac {d _ {2}}{s _ {1} d _ {1}}, \ldots , \frac {d _ {m}}{s _ {m - 1} d _ {m - 1}}, s _ {m}, \frac {N}{s _ {m} d _ {m}}\right): (1, d _ {1}, s _ {1} d _ {1}, \ldots , s _ {m - 1} d _ {m - 1}, d _ {m}, s _ {m} d _ {m})
$$

which is column-major. 

Proposition 2.1.6.32. Suppose A is a flat layout and N is a positive integer. Then there exists a N-complement B of A if and only if A is N-complementable. 

Proof. If A is N-complementable, then by Lemma 2.1.6.31 the layout $B = { \mathsf { c o m p } } ^ { \flat } ( L , N )$ is a Ncomplement of A. 

On the other hand, suppose there exists a N-complement B of A. Consider the flat layout 

$$
\begin{array}{l} L := \text { sort } \big (\text { squeeze } (A) \star \text { squeeze } (B) \big) \\ = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {n}). \end{array}
$$

Since $\Phi _ { L } ( 0 ) = 0 ;$ , and $\Phi _ { L }$ is injective, we know that $d _ { 1 } \neq 0$ . We will argue that $d _ { i } = s _ { 1 } \cdot \cdot \cdot s _ { i - 1 }$ , i.e., that L is column-major. Since 

$$
\Phi_ {L} ^ {N}: [ 0, N) \to [ 0, N)
$$

is a bijection, we know that 1 is in the image of $\Phi _ { L }$ , which implies that $d _ { 1 } = 1$ . Suppose $1 < i \le m$ and suppose we have proved that $d _ { j } = s _ { 1 } \cdot \cdot \cdot s _ { j - 1 }$ for all $j < i$ . Consider the stride $d _ { i }$ . We know that there is no $( x _ { 1 } , \ldots , x _ { i - 1 } , 0 , \ldots , 0 )$ such that $( x _ { 1 } , \ldots , x _ { i - 1 } , 0 , \ldots , 0 ) \cdot ( d _ { 1 } , \ldots , d _ { m } ) = s _ { 1 } \cdot \cdot \cdot s _ { i - 1 }$ , since the largest possible value of such an expression is 

$$
\sum_ {j = 1} ^ {i - 1} (s _ {j} - 1) \left(s _ {1} \dots s _ {j - 1}\right) = s _ {1} \dots s _ {i} - 1.
$$

Since $\Phi _ { L }$ is surjective, and $d _ { i } \leq d _ { i + 1 } \leq \cdot \cdot \cdot \leq d _ { m }$ , it follows that the next largest value of $\Phi _ { L }$ is $d _ { i }$ , so we must have $d _ { i } = s _ { 1 } \cdot \cdot \cdot s _ { i - 1 }$ , as claimed. 

Returning to our main goal, consider the layout 

$$
\operatorname{sort} (\text { squeeze } (A)) = \left(s _ {1} ^ {\prime}, \dots , s _ {m ^ {\prime}} ^ {\prime}\right): \left(d _ {1} ^ {\prime}, \dots , d _ {m ^ {\prime}} ^ {\prime}\right).
$$

Then there exist $j _ { 1 } < \cdots < j _ { m ^ { \prime } }$ such that $s _ { i } ^ { \prime } = s _ { j _ { i } }$ and $d _ { i } ^ { \prime } = d _ { j _ { i } }$ for each $1 \leq i \leq m ^ { \prime } . { \mathrm { ~ I f ~ } } 1 \leq i < m ^ { \prime }$ 2 then 

$$
s _ {i} ^ {\prime} d _ {i} ^ {\prime} = s _ {j _ {i}} d _ {j _ {i}} = s _ {j _ {i}} s _ {1} \dots s _ {j _ {i} - 1}
$$

divides 

$$
d _ {i + 1} ^ {\prime} = s _ {1} \dots s _ {j _ {i + 1} - 1}.
$$

If $i = m ^ { \prime } ,$ , then 

$$
s _ {m ^ {\prime}} ^ {\prime} d _ {m ^ {\prime}} ^ {\prime} = s _ {j _ {m ^ {\prime}}} d _ {j _ {m ^ {\prime}}} = s _ {j _ {m ^ {\prime}}} s _ {1} \dots s _ {j _ {m ^ {\prime}} - 1}
$$

divides 

$$
N = s _ {1} \cdot \cdot \cdot s _ {m}.
$$

We conclude that A is N-complementable. 

Proposition 2.1.6.33. Suppose N is a positive integer, and A is a N-complementable flat layout. If B is a flat layout such that 

1. B is a N-complement of L, 

2. B is coalesced, and 

3. B is sorted, 

then $B = { \mathsf { c o m p } } ^ { \flat } ( A , N )$ 

Proof. Conditions 1 and 2 imply that $\Phi _ { B }$ and $\Phi _ { \mathsf { c o m p } ^ { \flat } ( A , N ) }$ have the same image. Since B and $\mathsf { c o m p } ^ { \flat } ( A , N )$ are sorted, we know by Lemma 2.1.6.22 that $\Phi _ { B }$ and $\Phi _ { \mathsf { c o m p } ^ { \flat } ( A , N ) }$ are increasing. Combining these two facts, it follows that $\Phi _ { B } = \Phi _ { \mathsf { c o m p } ^ { \flat } ( A , N ) }$ . Proposition 2.1.4.18 and condition 3 then imply that 

$$
B = \operatorname{coal} ^ {\flat} (B) = \operatorname{coal} ^ {\flat} (\operatorname{comp} ^ {\flat} (A, N)) = \operatorname{comp} ^ {\flat} (A, N).
$$

Lemma 2.1.6.34. Suppose A is a flat layout. $I f \ N _ { 1 } \ \leq \ N _ { 2 }$ are positive integers such that A is N<sub>1</sub>-complementable and A is $N _ { 2 }$ -complementable, then 

$$
\Phi_ {\mathsf {c o m p} ^ {\flat} (A, N _ {2})} \mid_ {[ 0, N _ {1})} = \Phi_ {\mathsf {c o m p} ^ {\flat} (A, N _ {1})}.
$$

Proof. Write 

$$
\begin{array}{c} \text {sort(squeeze(A)) = (s_{1} ,\ldots,s_{m}):(d_{1} ,\ldots,d_{m}) ,} \\ C = \left(d _ {1}, \frac {d _ {2}}{s _ {1} d _ {1}}, \frac {d _ {3}}{s _ {2} d _ {2}}, \ldots , \frac {d _ {m}}{s _ {m - 1} d _ {m - 1}}\right): \left(1, s _ {1} d _ {1}, s _ {2} d _ {2}, \ldots , s _ {m - 1} d _ {m - 1}\right) \end{array}
$$

and write 

$$
\begin{array}{l} E _ {1} = \left(\frac {N _ {1}}{s _ {m} d _ {m}}\right): (s _ {m} d _ {m}), \\ E _ {2} = \left(\frac {N _ {2}}{s _ {m} d _ {m}}\right): (s _ {m} d _ {m}), \\ C _ {1} = C \star E _ {1}, \\ C _ {2} = C \star E _ {2}, \end{array}
$$

so that 

$$
\begin{array}{c} \mathsf {c o m p} ^ {\flat} (A) = \mathsf {c o a l} ^ {\flat} (C) \\ \mathsf {c o m p} ^ {\flat} (A, N _ {1}) = \mathsf {c o a l} ^ {\flat} (C _ {1}) \\ \mathsf {c o m p} ^ {\flat} (A, N _ {2}) = \mathsf {c o a l} ^ {\flat} (C _ {2}). \end{array}
$$

Then we have a commuting diagram 

$$
\begin{array}{c} [ 0, \text {size} (C _ {1})) \xrightarrow {\text {colex} _ {(\text {size} (C) , N _ {1})} ^ {- 1}} [ 0, \text {size} (C)) \times [ 0, N _ {1}) \xrightarrow {\Phi_ {C} \times s _ {m} d _ {m}} \mathbb {Z} \times \mathbb {Z} \xrightarrow {+} \mathbb {Z} \\ \Big \downarrow \subseteq \\ [ 0, \text {size} (C _ {2})) \xrightarrow {\text {colex} _ {(\text {size} (C) , N _ {2})} ^ {- 1}} [ 0, \text {size} (C)) \times [ 0, N _ {2}) \xrightarrow {\Phi_ {C} \times s _ {m} d _ {m}} \mathbb {Z} \times \mathbb {Z} \xrightarrow {+} \mathbb {Z} \end{array}
$$

where, by Proposition 2.1.3.40, the composite of the top row is the layout function of $C _ { 1 } = C \star E _ { 1 }$ ， and the composite of the bottom row is the layout function of $C _ { 2 } = C \star E _ { 2 }$ . This tells us that the restriction of $\Phi _ { C _ { 2 } } ~ \mathrm { t o } ~ [ 0 , \mathsf { s i z e } ( C _ { 2 } ) )$ is $\Phi _ { C _ { 1 } }$ , and the result follow from the fact that 

$$
\begin{array}{l} \Phi_ {\mathsf {c o m p} ^ {\flat} (A, N _ {1})} = \Phi_ {C _ {1}} \\ \Phi_ {\mathsf {c o m p} ^ {\flat} (A, N _ {2})} = \Phi_ {C _ {2}}. \end{array}
$$

## 2.1.7 Further operations

In this section, we define several further operations on flat layouts, namely composition, flat division, and flat products. These are the flattened variants of more natural operations on (nested) layouts. We do not often work with these operations, but include them anyway for completeness. 

## 2.1.7.1 Composition

If A and B are flat layouts, then the composite $B \circ A$ is a flat layout whose layout function is the composite of the layout functions of A and B. More precisely, we have the following definition. 

Definition 2.1.7.1. Suppose A and B are flat layouts. We say the flat layout C is the composition of A and B, and write $C = B \circ A .$ , if 

1. C is non-degenerate, 

2. sh $\mathsf { i a p e } ( A ) = { \mathsf { s h a p e } } ( R )$ 

3. $\Phi _ { R } = \Phi _ { B } \circ \Phi _ { A } ^ { \mathsf { s i z e } ( B ) }$ 

Remark 2.1.7.2. Note that condition 2 in our definition ensures that $\Phi _ { R }$ and $\Phi _ { A }$ have the same domain, and condition 3 implies $\mathsf { c o s i z e } ( A ) \leq \mathsf { s i z e } ( B )$ 

Example 2.1.7.3. If $A = ( 2 , 3 ) : ( 5 , 6 )$ and $B = ( 8 0 ) : ( 1 0 )$ , then 

$$
B \circ A = (2, 3): (5 0, 6 0).
$$

More generally, if 

$$
A = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m})
$$

is a non-degenerate flat layout, and 

$$
B = (t): (e)
$$

is a rank 1 flat layout with $t \geq \mathsf { c o s i z e } ( A )$ , then A and B are composable, and 

$$
B \circ A = (s _ {1}, \ldots , s _ {m}): (t d _ {1}, \ldots , t d _ {m}).
$$

Example 2.1.7.4. If $A = ( 1 2 8 , 1 2 8 ) : ( 0 , 0 )$ and $B = ( 6 4 , 3 2 ) : ( 1 , 6 4 )$ , then 

$$
B \circ A = (1 2 8, 1 2 8): (0, 0).
$$

More generally, if A is a flat layout each of whose stride entries is zero, and B is any flat layout, then A and B are composable with $B \circ A = A$ 

Example 2.1.7.5. If $A = ( 6 4 , 3 2 ) : ( 2 , 2 5 6 )$ and $B = ( 2 0 4 8 , 2 0 4 8 ) : ( 1 , 2 0 4 8 )$ , then 

$$
B \circ A = (6 4, 3 2): (2, 2 5 6).
$$

More generally, if A is any flat layout, and B is a column-major flat layout with cosize $( A ) \leq \mathsf { s i z e } ( B )$ then $B \circ A = A$ 

Example 2.1.7.6. If $A = ( 4 ) : ( 2 )$ and $B = ( 2 , 2 , 6 ) : ( 1 2 , 6 , 1 )$ , then there is no flat layout R with $R = B \circ A$ 

Remark 2.1.7.7. If $B ^ { \prime }$ and B have the same layout function, then $B \circ A = B ^ { \prime } \circ A$ 

Remark 2.1.7.8. Flat layouts are a special case of the more general notion of layouts (Definition 2.3.1.1). It turns out that there are cases (such as Example 2.1.7.6) where there does not exist a flat layout C with $C = B \circ A$ , but there does exist a (nested) layout C with $C = B \circ A$ (see Example 2.3.7.6). For this reason, we postpone further discussion and analysis of composition until we have defined layouts in their full generality. 

## 2.1.7.2 Flat division

If A and B are flat layouts, then the flat division of A by B is a flattened version of the more natural logical division of layouts. See Section 2.3.8 for details. 

Definition 2.1.7.9. Suppose A and B are flat layouts, and that B is size(A)-complementable, with 

$$
B ^ {c} = \operatorname{comp} ^ {\flat} (B, \text { size } (A)).
$$

We define the flat division of A by B to be the flat layout 

$$
A \oslash^ {\flat} B = A \circ (B \star B ^ {c}).
$$

Example 2.1.7.10. If $A = ( 2 , 2 , 2 , 2 )$ : (1, 4, 2, 8) and $B = ( 2 , 2 ) : ( 4 , 2 )$ , then 

$$
A \oslash^ {\flat} B = (2, 2, 2, 2): (4, 2, 1, 8).
$$

Example 2.1.7.11. If $A = ( 3 , 5 , 9 , 6 )$ : (54, 0, 6, 1) and $B = ( 6 , 3 ) : ( 1 3 5 , 1 )$ , then 

$$
A \oslash^ {\flat} B = (6, 3, 5, 9): (1, 5 4, 0, 6).
$$

Example 2.1.7.12. If A is any flat layout and $B = ( ) : ( )$ is the empty layout, then 

$$
A \oslash^ {\flat} B = A.
$$

## 2.1.7.3 Flat products

If A and B are flat layouts, then the flat product $A \otimes ^ { \flat }$ B of A and B is a flattened version of the more natural logical product of layouts. See Section 2.3.9 for details. 

Definition 2.1.7.13. Suppose A and B are flat layouts, and that A is size(A)·cosize(B)-complementable, with 

$$
A ^ {c} = \operatorname{comp} ^ {\flat} (A, \text { size } (A) \cdot \text { cosize } (B)).
$$

We define the flat product of A and B by 

$$
A \otimes^ {\flat} B = A \star (A ^ {c} \circ B).
$$

Example 2.1.7.14. If $A = ( 2 , 2 , 2 ) : ( 1 , 2 , 4 )$ and $B = ( 2 , 2 , 2 ) : ( 1 , 2 , 4 )$ , then 

$$
A \otimes^ {\flat} B = (2, 2, 2, 2, 2, 2): (1, 2, 4, 8, 1 6, 3 2).
$$

Example 2.1.7.15. If $A = ( 2 , 2 , 2 ) : ( 1 , 2 , 4 )$ and $B = ( 3 , 5 ) : ( 5 , 1 )$ , then 

$$
A \otimes^ {\flat} B = (2, 2, 2, 3, 5): (1, 2, 4, 4 0, 8).
$$

Example 2.1.7.16. If A is any flat layout and $B = ( ) : ( )$ is the empty layout, then 

$$
A \otimes^ {\flat} B = A.
$$

## 2.1.8 Tractable flat layouts

In this section we define an especially well-behaved class of flat layouts, called tractable flat layouts. Tractable flat layouts include the most important examples of interest, such as row-major, columnmajor, compact, and complementable layouts. Later on, we will see that tractable flat layouts are precisely the layouts which arise from a certain category Tuple. 

Definition 2.1.8.1. Suppose L is a flat layout, and write 

$$
\operatorname{sort} (L) = \left(s _ {1}, \dots , s _ {m}\right): \left(d _ {1}, \dots , d _ {m}\right).
$$

We say L is tractable if for each $1 \leq i < m$ , we have 

1. $d _ { i } = 0 ,$ or 

2. $s _ { i } d _ { i }$ divides $d _ { i + 1 }$ 

Example 2.1.8.2. The flat layout 

$$
L = (1 2): (1 7)
$$

is tractable. More generally, any flat layout of rank 1 is tractable. 

Example 2.1.8.3. The flat layout 

$$
L = (2, 4, 3 2): (1, 2, 8)
$$

is tractable. More generally, any column-major layout 

$$
L = (s _ {1}, \dots , s _ {m}): (1, s _ {1}, \dots , s _ {1} \dots s _ {m - 1})
$$

is tractable. 

Example 2.1.8.4. The flat layout 

$$
L = (2, 4, 3 2): (1 2 8, 3 2, 1)
$$

is tractable. More generally, any row-major layout 

$$
L = (s _ {1}, \ldots , s _ {m}): (s _ {2} \dots s _ {m}, \ldots , s _ {m}, 1)
$$

is tractable. 

Example 2.1.8.5. The flat layout 

$$
L = (3, 3, 1, 3, 3, 1, 3): (8 1, 1, 0, 9, 3, 0, 2 7)
$$

is tractable. More generally, any compact flat layout is tractable. 

Example 2.1.8.6. The flat layout 

$$
L = (3, 7, 7): (0, 1 5, 0)
$$

is tractable. More generally, any flat layout with exactly one non-zero stride is tractable. 

Example 2.1.8.7. The flat layout 

$$
L = (2, 2, 2, 2): (1, 2 0 4 8, 1 6, 6 4)
$$

is tractable. More generally, any complementable flat layout is tractable. 

Example 2.1.8.8. Suppose L is a flat layout. If L is tractable and $I \subset \langle m \rangle$ is any subset, then the restriction $\textit { L } | _ { I }$ is tractable. In particular, if L is tractable, then squeeze(L) and filter(L) are tractable. 

Example 2.1.8.9. The flat layout 

$$
L = (4, 8): (3, 3)
$$

is not tractable. In particular, this shows that the concatenation $L _ { 1 } \star L _ { 2 }$ of tractable flat layouts $L _ { 1 }$ and $L _ { 2 }$ need not be tractable. 

Observation 2.1.8.10. If L is a tractable flat layout and no entry of stride(L) is equal to 0, then L is complementable. In particular, if L is tractable, then filter(L) is complementable. 

We conclude this section by enumerating a family of equivalent conditions for a flat layout L to be tractable. 

Proposition 2.1.8.11. Suppose L is a flat layout. Then the following conditions are equivalent. 

1. L is tractable. 

2. sort(L) is tractable. 

3. filter(L) is tractable. 

4. filter(L) is complementable. 

Proof. Suppose L is a flat layout. 

$( 1 \Leftrightarrow 2 )$ : This follows from the fact that 

$$
\operatorname{sort} (\operatorname{sort} (L)) = \operatorname{sort} (L).
$$

$( 1 \Leftrightarrow 3 )$ : This follows from the fact that 

$$
\operatorname{sort} (\text { filter } (L)) = \text { filter } (\operatorname{sort} (L)).
$$

$( 3 \Leftrightarrow 4 )$ : This follows from the fact that if 

$$
L = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m})
$$

is a flat layout such that each of the stride entries $d _ { i }$ is nonzero, then the definition of tractability coincides with that of complementability. 

## 2.2 Nested Tuples

In this section, we introduce nested tuples, which are the generalization of tuples needed to define layouts in full generality. 

## 2.2.1 Profiles

A nested tuple S is determined by its flattening, which is an ordinary tuple, and its ${ \it p r o f i l e }$ , which describes parenthesization pattern on S. We define profiles precisely as follows. 

Definition 2.2.1.1. A profile P is either 

1. $P = * ,$ or 

2. a tuple $P = ( P _ { 1 } , \ldots , P _ { r } )$ of profiles $P _ { 1 } , \ldots , P _ { r }$ for some $r \geq 0$ 

We write Profile for the set of profiles. 

Example 2.2.1.2. Here are some examples of profiles. 

$$
\begin{array}{l} P _ {1} = (*, *) \\ P _ {2} = (*, (*, *)) \\ P _ {3} = ((*, *), (*, *)) \\ P _ {4} = ((*, *, *), (*, ()) \\ P _ {5} = () \\ P _ {6} = * \end{array}
$$

Let’s define some important attributes of profiles. 

Definition 2.2.1.3. Suppose P is a profile. 

• The rank of X is 

$$
\operatorname{rank} (P) = \left\{ \begin{array}{l l} 1 & P = * \\ r & P = (P _ {1}, \ldots , P _ {r}) \text {   is   a   tuple   of   profiles. } \end{array} \right..
$$

• The length of P is 

$$
\mathsf {l e n} (P) = \left\{ \begin{array}{l l} 1 & P = * \\ \sum_ {i = 1} ^ {r} \mathsf {l e n} (P _ {i}) & P = (P _ {1}, \ldots , P _ {r}) \text {   is   a   tuple   of   profiles. } \end{array} \right.
$$

• The depth of P is 

$$
\mathsf {d e p t h} (P) = \left\{ \begin{array}{l l} 0 & P = * \\ 1 + \max _ {1 \leq i \leq r} (\mathsf {d e p t h} (P _ {i})) & P = (P _ {1}, \ldots , P _ {r}) \text {   is   a   tuple   of   profiles. } \end{array} \right.
$$

Example 2.2.1.4. Here are some examples of profiles, together with their rank, length, and depth : 

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

Definition 2.2.1.5. Suppose P is a profile with rank $( P ) = r . { \mathrm { ~ I f ~ } } 1 \leq i \leq r ,$ then the ith mode of P is 

$$
\mathsf {m o d e} _ {i} (P) = \left\{ \begin{array}{l l} P & \mathsf {d e p t h} (P) = 0 (\text { hence } i = r = 1), \\ P _ {i} & P = (P _ {1}, \ldots , P _ {r}) \text { has   depth } \geq 1. \end{array} \right.
$$

Example 2.2.1.6. If $P = ( ( * , * ) , ( ( ) ) , ( ( * , ( * , * ) ) ) )$ then the modes of $P$ are 

$$
\begin{array}{l} \text {mode} _ {1} (P) = (*, *) \\ \text {mode} _ {2} (P) = (()) \\ \text {mode} _ {3} (P) = (*, (*, *)) \end{array}
$$

The following notation will be useful. 

Notation 2.2.1.7. Suppose P is a profile of depth $> 0 .$ For any $1 \leq j \leq \mathsf { r a n k } ( P )$ , we write 

$$
\begin{array}{l} \mathsf {l e n} _ {j} (X) = \mathsf {l e n} (\mathsf {m o d e} _ {j} (P)), \\ \mathsf {l e n} _ {<   j} (P) = \sum_ {i = 1} ^ {j - 1} \mathsf {l e n} _ {i} (X), \\ \mathsf {l e n} _ {\leq j} (X) = \mathsf {l e n} _ {<   j} (P) + \mathsf {l e n} _ {j} (P) \end{array}
$$

The most important operation supported by profiles is substitution: If $Q$ is a profile of length $m ,$ and $P _ { 1 } , \ldots , P _ { m }$ are profiles, then we can obtain a new profile $( P _ { 1 } , \ldots , P _ { m } ) _ { Q }$ by substituting the ith entry of $Q$ with the profile $P _ { i } ,$ for each $1 \leq i \leq m$ . More precisely, we have the following definition. 

Definition 2.2.1.8. Suppose $Q$ is a profile of length $m ,$ and suppose $P _ { 1 } , \ldots , P _ { m }$ are profiles. Then the Q-substitution of $P _ { 1 } , \ldots , P _ { m }$ is the profile 

$$
(P _ {1}, \dots , P _ {m}) _ {Q}
$$

defined as follows. Write depth $\boldsymbol { \mathscr { l } } ( Q ) = d$ and rank $( Q ) = r .$ 

• If $d = 0$ , then $m = 1$ , and we define 

$$
(P _ {1}) _ {Q} = P _ {1}.
$$

• Suppose next that $d > 0$ , and that we have defined $Q ^ { \prime } \cdot$ -substitution for all profiles $Q ^ { \prime }$ of depth $< d .$ We can write 

$$
Q = (Q _ {1}, \dots , Q _ {r})
$$

where each mode $Q _ { i } = { \mathsf { m o d e } } _ { i } ( Q )$ has depth $< d .$ If for each $1 \leq i \leq r .$ , we set 

$$
\ell_ {i} = \operatorname{len} (P _ {1}) + \dots + \operatorname{len} (P _ {i - 1}),
$$

then we define 

$$
(P _ {1}, \dots , P _ {r}) _ {Q} = ((P _ {1}, \dots , P _ {\ell_ {2}}) _ {Q _ {1}}, \dots , (P _ {\ell_ {r} + 1}, \dots , P _ {\ell_ {r + 1}}) _ {Q _ {r}}).
$$

Example 2.2.1.9. If $Q = ( * , * )$ and $P _ { 1 } = ( * , * ) , P _ { 2 } = ( * , * , * )$ 

$$
(P _ {1}, P _ {2}) _ {Q} = ((*, *), (*, *, *))
$$

More generally, if $Q = ( * , \ldots , * )$ is the profile with depth $( Q ) = 1$ and len $( Q ) = { \mathsf { r a n k } } ( Q ) = r ,$ , then 

$$
(P _ {1}, \dots , P _ {r}) _ {Q} = (P _ {1}, \dots , P _ {r})
$$

is ordinary concatenation. 

Aside 2.2.1.10. There is an operadic interpretation of Q-substitution. The set Profile of profiles has the structure of a (non-symmetric) operad: the set 

$$
\operatorname{Profile} (n) = \{P \in \operatorname{Profile} | \operatorname{len} (P) = n \}
$$

forms the collection of n-ary operations of Profile, and if $n = m _ { 1 } + \cdot \cdot \cdot + m _ { r }$ , then the structure map 

$$
\operatorname{Profile} \left(m _ {1}\right) \times \dots \times \operatorname{Profile} \left(m _ {r}\right) \times \operatorname{Profile} (n) \longrightarrow \operatorname{Profile} \left(m _ {1} + \dots + m _ {r}\right)
$$

$$
(P _ {1}, \dots , P _ {r}), Q \longmapsto (P _ {1}, \dots , P _ {r}) _ {Q}
$$

is given by Q-substitution. One can also form the cofree symmetric operad on this non-symmetric operad, which amounts to endowing the sets of n-ary operations with trivial symmetric group action. 

## 2.2.2 Basic definitions

Having defined profiles and their basic properties, we can now define nested tuples. 

Definition 2.2.2.1. If V is a set, then a nested tuple X with entries in V is a pair $( X ^ { \flat } , P )$ consisting of 

1. a tuple $X ^ { \flat } = ( x _ { 1 } , \dots , x _ { m } )$ with entries in V, called the flattening of X, and 

2. a profile $\mathsf { p r o f } ( X ) = P$ of length m, called the profile of X. 

We write Nest(V) for the set of all nested tuples with entries in a set V. 

Example 2.2.2.2. Here are some examples of nested tuples, together with their flattening and profile. 

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

Notation 2.2.2.3. We sometimes write 

$$
X = (x _ {1}, \ldots , x _ {m}) _ {P}
$$

to denote a nested tuple with $X ^ { \flat } = ( x _ { 1 } , \dots , x _ { m } )$ and profile prof(X) = P. 

Observation 2.2.2.4. If V is any set, then by definition, we have a pullback square 

$$
\begin{array}{c} \text {Nest} (V) \xrightarrow {\text {prof} (-)} \text {Profile} \\ (-) ^ {b} \Big \downarrow \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \text {len} (-) \\ \text {Tuple} (V) \xrightarrow {\text {len} (-)} \mathbb {N}. \end{array}
$$

Remark 2.2.2.5. Given the recursive definition of profiles, we could equivalently define a nested tuple with entries in V to be either 

1. an element of V , or 

2. a tuple of nested tuples with entries in V . 

Let’s define some important attributes of nested tuples. Each such attribute of a nested tuple X is inhereted by its flattening $X ^ { \flat }$ or its profile prof(X). 

Definition 2.2.2.6. Suppose X is a nested tuple with entries in V. 

• The rank of X is 

$$
\operatorname{rank} (X) = \operatorname{rank} (P)
$$

• The length of X is 

$$
\operatorname{len} (X) = \operatorname{len} (P) = \operatorname{len} (X ^ {\flat})
$$

• The depth of X is 

$$
\operatorname{depth} (X) = \operatorname{depth} (P)
$$

• If V = Z, then the size of X is 

$$
\operatorname{size} (X) = \operatorname{size} (X ^ {\flat}).
$$

Example 2.2.2.7. Here are some examples of nested tuples of integers, together with their rank, length, depth, and size: 

$$
\begin{array}{l l} X = 2 7 & \text {rank} (X) = 1, \quad \text {len} (X) = 1, \quad \text {depth} (X) = 0, \quad \text {size} (X) = 2 7 \\ X = (2, 1 0, 5) & \text {rank} (X) = 3, \quad \text {len} (X) = 3, \quad \text {depth} (X) = 1, \quad \text {size} (X) = 1 0 0 \\ X = (((3, 4), 2, 2), 8, 9), & \text {rank} (X) = 3, \quad \text {len} (X) = 6, \quad \text {depth} (X) = 3, \quad \text {size} (X) = 3 0 9 6 \\ X = ((((),()), (2, (5, 5))), & \text {rank} (X) = 2, \quad \text {len} (X) = 3, \quad \text {depth} (X) = 3, \quad \text {size} (X) = 5 0 \end{array}
$$

Example 2.2.2.8. A nested tuple of integers with depth 0 is simply an integer. 

Example 2.2.2.9. A nested tuple of integers with depth 1 is simply a tuple of integers. If X is such a nested tuple, then $\mathsf { r a n k } ( X ) = \mathsf { l e n } ( X )$ 

Definition 2.2.2.10. Suppose $X = ( x _ { 1 } , \dots , x _ { m } ) _ { P }$ is a nested tuple with rank $( X ) = r . { \mathrm { ~ I f ~ } } 1 \leq i \leq r ,$ then the ith mode of X to be the nested tuple 

$$
\operatorname{mode} _ {i} (X) = \left(x _ {\text { len } _ {<   i} (P) + 1}, \dots , x _ {\text { len } _ {\leq i} (P)}\right) _ {\operatorname{mode} _ {i} (P)}.
$$

Example 2.2.2.11. If 

$$
X = ((3), 4, ((1 0, 1 0), 1 2)),
$$

then the modes of X are 

$$
\begin{array}{l} \text { mode } _ {1} (X) = (3) \\ \text { mode } _ {2} (X) = 4 \\ \text { mode } _ {3} (X) = ((1 0, 1 0), 1 2) \end{array}
$$

Example 2.2.2.12. If X = (32, 5, 6, 64), then the modes of X are 

$$
\begin{array}{l} \mathrm{mode} _ {1} (X) = 3 2 \\ \mathrm{mode} _ {2} (X) = 5 \\ \mathrm{mode} _ {3} (X) = 6 \\ \mathrm{mode} _ {4} (X) = 6 4 \end{array}
$$

It will be convenient to introduce the following notation. 

Notation 2.2.2.13. Suppose X is a nested tuple of integers with depth $( X ) > 0$ . For any $1 \le j \le$ rank(X), we write 

$$
\begin{array}{c} \mathsf {l e n} _ {j} (X) = \mathsf {l e n} (\mathsf {m o d e} _ {j} (X)), \\ \mathsf {l e n} _ {<   j} (X) = \sum_ {i = 1} ^ {j - 1} \mathsf {l e n} _ {i} (X), \\ \mathsf {l e n} _ {\leq j} (X) = \mathsf {l e n} _ {<   j} (X) + \mathsf {l e n} _ {j} (X) \end{array}
$$

and similarly, we write 

$$
\begin{array}{c} \text {size} _ {j} (X) = \text {size} (\text {mode} _ {j} (X)), \\ \text {size} _ {<   j} (X) = \prod_ {i = 1} ^ {j - 1} \text {size} _ {j} (X), \text {and} \\ \text {size} _ {\leq j} (X) = \text {size} _ {<   j} (X) \cdot \text {size} _ {j} (X). \end{array}
$$

Definition 2.2.2.14. If $X = ( x _ { 1 } , \ldots , x _ { m } ) _ { P }$ is a nested tuple and $1 \leq i \leq m$ , then the ith entry of X is 

$$
\operatorname{entry} _ {i} (X) = \operatorname{entry} _ {i} \left(X ^ {\flat}\right) = x _ {i}.
$$

Example 2.2.2.15. If 

$$
X = ((3), 4, ((1 0, 1 0), 1 2)),
$$

then the entries of X are 

$$
\begin{array}{l} \text {entry} _ {1} (X) = 3 \\ \text {entry} _ {2} (X) = 4 \\ \text {entry} _ {3} (X) = 1 0 \\ \text {entry} _ {4} (X) = 1 0 \\ \text {entry} _ {5} (X) = 1 2. \end{array}
$$

Example 2.2.2.16. If X = (32, 5, 6, 64), then the entries of X are 

$$
\begin{array}{l} \text {entry} _ {1} (X) = 3 2 \\ \text {entry} _ {2} (X) = 5 \\ \text {entry} _ {3} (X) = 6 \\ \text {entry} _ {4} (X) = 4. \end{array}
$$

Example 2.2.2.17. If X is a nested tuple with depth 1, then mode $\mathsf { \Omega } _ { i } ( X ) = \mathsf { e n t r y } _ { i } ( X )$ for all $1 \leq i \leq$ $\mathsf { r a n k } ( X ) = \mathsf { l e n } ( X )$ 

Observation 2.2.2.18. If X is a nested tuple of integers, then the entries of X are integers, while the modes of X are themselves nested tuples of integers. 

Finally, we introduce the notion of congruence of nested tuples, which indicates when nested tuples have the same profile. 

Definition 2.2.2.19. If $X _ { 1 }$ and $X _ { 2 }$ are nested tuples, we say $X _ { 1 }$ and $X _ { 2 }$ are congruent, if 

$$
\operatorname{prof} (X _ {1}) = \operatorname{prof} (X _ {2}).
$$

Example 2.2.2.20. Here are some examples of nested tuples $X _ { 1 }$ and $X _ { 2 } ,$ , and whether or not they are congruent 

$$
\begin{array}{l l l} X _ {1} = 2 7 & X _ {2} = 1 0 0 & \text {congruent} \\ X _ {1} = (2, 2) & X _ {2} = (8, 6 4) & \text {congruent} \\ X _ {1} = ((4, 8), (4, 8)) & X _ {2} = ((1, 1), (5, 1 0)) & \text {congruent} \\ X _ {1} = ((6 4, (8, 8)), (2 5, (5, 5))) & X _ {2} = ((2, (3, 5)), (7, (1 1, 1 3))) & \text {congruent} \\ X _ {1} = 2 7 & X _ {2} = (1 0 0) & \text {not congruent} \\ X _ {1} = (2, 2) & X _ {2} = (8, 6 4, 1 2 8) & \text {not congruent} \\ X _ {1} = ((4, 8), (4, 8)) & X _ {2} = (((1, 1), (5, 1 0))) & \text {not congruent} \end{array}
$$

## 2.2.3 Substitution

Recall that if Q is a profile of length r and $P _ { 1 } , \ldots , P _ { r }$ are profiles, then we defined a profile 

$$
(P _ {1}, \ldots , P _ {r}) _ {Q}
$$

called the Q-substitution of $P _ { 1 } , \ldots , P _ { r }$ . This profile is obtained from $Q$ by replacing the ith entry of $Q$ with the profile $P _ { i }$ . We can extend this to an operation on nested tuples as follows. 

Definition 2.2.3.1. Suppose $X _ { 1 } , \ldots , X _ { m }$ are nested tuples with profiles $P _ { 1 } , \ldots , P _ { m }$ , and suppose Q is a profile of length m. We define the Q-substitution 

$$
(X _ {1}, \ldots , X _ {m}) _ {Q}
$$

of $X _ { 1 } , \ldots , X _ { m }$ to be the nested tuple with flattening 

$$
(X _ {1}, \ldots , X _ {m}) _ {Q} ^ {\flat} = X _ {1} ^ {\flat} \star \dots \star X _ {m} ^ {\flat}
$$

and profile 

$$
(P _ {1}, \ldots , P _ {m}) _ {Q}.
$$

More generally, if $X _ { 1 } , \ldots , X _ { m }$ are nested tuples and Y is a nested tuple of length $m ,$ we define 

$$
(X _ {1}, \dots , X _ {m}) _ {Y} = (X _ {1}, \dots , X _ {m}) _ {\operatorname{prof} (Y)}.
$$

Example 2.2.3.2. If $( X _ { 1 } , X _ { 2 } , X _ { 3 } ) = ( 6 4 , 1 6 , 4 )$ and $Q = ( * , ( * , * ) )$ , then 

$$
(X _ {1}, X _ {2}, X _ {3}) _ {Q} = (6 4, (3 2, 4))
$$

Example 2.2.3.3. If $( X _ { 1 } , X _ { 2 } , X _ { 3 } , X _ { 4 } ) = ( ( 2 , 2 ) , ( 3 , 3 ) , ( 5 , 5 ) , ( 7 , 7 ) )$ and $Q = ( ( * , * ) , ( * , * ) )$ ), then 

$$
(X _ {1}, X _ {2}, X _ {3}, X _ {4}) _ {Q} = (((2, 2), (3, 3)), ((5, 5), (7, 7))).
$$

Example 2.2.3.4. If X = (12) and $Q = *$ , then 

$$
(X) _ {Q} = 1 2.
$$

Example 2.2.3.5. If $X _ { 1 } = 2 , X _ { 2 } = 2 , X _ { 3 } = ( 5 , 5 )$ , and $Q = ( * , * , * )$ , then 

$$
(X _ {1}, X _ {2}, X _ {3}) _ {Q} = (2, 2, (5, 5)) = (X _ {1}, X _ {2}, X _ {3}).
$$

More generally, if $X _ { 1 } , \ldots , X _ { m }$ are any nested tuples and $P = ( * , \ldots , * )$ then 

$$
(X _ {1}, \ldots , X _ {m}) _ {Q} = (X _ {1}, \ldots , X _ {k})
$$

is the concatenation of $X _ { 1 } , \ldots , X _ { m }$ 

Aside 2.2.3.6. There is an operadic interpretation of substitutions of nested tuples. The set Nest(Z) of nested tuples of integers is an algebra over the operad Profile, with structure maps given by Q-substitution: 

$$
\operatorname{Nest} (\mathbb {Z}) \times \dots \times \operatorname{Nest} (\mathbb {Z}) \times \operatorname{Profile} (n) \longrightarrow \operatorname{Nest} (\mathbb {Z})
$$

$$
(X _ {1}, \dots , X _ {m}), Q \longmapsto (X _ {1}, \dots , X _ {m}) _ {Q}.
$$

## 2.2.4 Refinement

In this section, we introduce an important relation on nested tuples called refinement. Intuitively, if $X ^ { \prime }$ and X are nested tuples of integers, we say $X ^ { \prime }$ refines X if $X ^ { \prime }$ may be obtained from X by replacing each entry of $X$ with some nested tuple of the same size. More precisely, we have the following definition. 

Definition 2.2.4.1. If $X ^ { \prime }$ and X are nested tuples, then we say $X ^ { \prime }$ refines X if either 

1. $X = \mathsf { s i z e } ( X ^ { \prime } )$ , or 

2. (a) $\mathsf { d e p t h } ( X ^ { \prime } ) , \mathsf { d e p t h } ( X ) > 0 ,$ 

(b) rank $( X ^ { \prime } ) = \mathsf { r a n k } ( X )$ , and 

(c) for each $1 \leq i \leq \mathsf { r a n k } ( X )$ , mode<sub>i</sub>(X<sup>′</sup>) refines mode<sub>i</sub>(X). 

Notation 2.2.4.2. We write 

$$
X ^ {\prime} \twoheadrightarrow X
$$

to indicate that $X ^ { \prime }$ refines $X$ . 

Example 2.2.4.3. Here are some examples of refinements of nested tuples. 

$$
\begin{array}{c} (2, (2, 2)) \twoheadrightarrow 8 \\ ((2, 2), (3, 3), (5, 5)) \twoheadrightarrow (4, 9, 2 5) \\ (6 4) \twoheadrightarrow 6 4 \\ (8, ((2, 2, 2), ((1, 4), (2, 2)))) \twoheadrightarrow (8, (8, 8)) \end{array}
$$

Observation 2.2.4.4. Refinement of nested tuples is reflexive, transitive, and antisymmetric, so refinement specifies a partial ordering on the collection of nested tuples of positive integers. 

If $X ^ { \prime }$ refines $X ,$ then we can think of $X ^ { \prime }$ as being obtained from X by replacing each entry $x _ { i }$ of $X$ with some nested tuple $X _ { i } ^ { \prime }$ of size $x _ { i }$ . We refer to the nested tuple $X _ { i } ^ { \prime }$ as the ith mode of $X ^ { \prime }$ relative to $X .$ . More precisely, we have the following definition. 

Construction 2.2.4.5. Suppose X is a nested tuple of integers of length $m ,$ and suppose $X ^ { \prime }$ refines X. For any $1 \leq i \leq m$ , we define a nested tuple 

$$
X _ {i} ^ {\prime} = \mathsf {m o d e} _ {i} (X ^ {\prime}, X),
$$

called the ith mode of $X ^ { \prime }$ relative to $X ,$ by the formula 

$$
\mathsf {m o d e} _ {i} (X ^ {\prime}, X) = \left\{ \begin{array}{l l} X ^ {\prime} & \mathsf {d e p t h} (X) = 0 (\text {hence} i = \ell = 1) \\ \mathsf {m o d e} _ {i - N} (\mathsf {m o d e} _ {j} (X ^ {\prime}), \mathsf {m o d e} _ {j} (X)) & j \text {is the largest integer such that} \\ & N := \mathsf {l e n} _ {<   j} (X) <   i. \end{array} \right.
$$

Example 2.2.4.6. If $X = ( ( 4 , 9 ) , ( 2 5 , 3 6 ) )$ , and $X ^ { \prime } = ( ( ( 2 , 2 ) , ( 3 , 3 ) ) , ( 2 5 , ( 6 , ( 2 , 3 ) ) ) )$ , then $X ^ { \prime }$ refines X and the modes of $X ^ { \prime }$ relative to X are 

$$
\begin{array}{l} \text {mode} _ {1} (X ^ {\prime}, X) = (2, 2) \\ \text {mode} _ {2} (X ^ {\prime}, X) = (3, 3) \\ \text {mode} _ {3} (X ^ {\prime}, X) = 2 5 \\ \text {mode} _ {4} (X ^ {\prime}, X) = (6, (2, 3)). \end{array}
$$

Example 2.2.4.7. If X is any nested tuple, then X refines X, and for any $1 \leq i \leq \mathsf { l e n } ( X )$ we have 

$$
\operatorname{mode} _ {i} (X, X) = \operatorname{entry} _ {i} (X).
$$

Example 2.2.4.8. If $X = X ^ { \flat }$ is a tuple, and $X ^ { \prime }$ refines X, then for any $1 \leq i \leq \mathsf { l e n } ( X )$ , we have 

$$
\operatorname{mode} _ {i} \left(X ^ {\prime}, X\right) = \operatorname{mode} _ {i} \left(X ^ {\prime}\right).
$$

Example 2.2.4.9. If $X ^ { \prime }$ is a nested tuple with ${ \mathsf { s i z e } } ( X ^ { \prime } ) = N$ , then $X ^ { \prime }$ refines N, and the only mode of $X ^ { \prime }$ relative to $N$ is 

$$
\operatorname{mode} _ {1} \left(X ^ {\prime}, N\right) = X ^ {\prime}.
$$

Notation 2.2.4.10. If $X ^ { \prime } \twoheadrightarrow X$ is a refinement and $1 \leq i \leq \mathsf { l e n } ( X )$ , then we write 

$$
\begin{array}{c} \operatorname{len} _ {i} (X ^ {\prime}, X) = \operatorname{len} (\operatorname{mode} _ {i} (X ^ {\prime}, X)) \\ \operatorname{len} _ {<   i} (X ^ {\prime}, X) = \sum_ {j <   i} \operatorname{len} _ {j} (X ^ {\prime}, X) \\ \operatorname{len} _ {\leq i} (X ^ {\prime}, X) = \sum_ {j \leq i} \operatorname{len} _ {j} (X ^ {\prime}, X) \end{array}
$$

Definition 2.2.4.11. Suppose $X ^ { \prime }$ refines $X ,$ , and write $X _ { i } ^ { \prime } = { \mathsf { m o d e } } _ { i } ( X ^ { \prime } , X )$ . Then the flattening of $X ^ { \prime }$ relative to X is the nested tuple 

$$
\operatorname{flat} \left(X ^ {\prime}, X\right) = \left(X _ {1} ^ {\prime}, \dots , X _ {m} ^ {\prime}\right).
$$

Example 2.2.4.12. If $X ^ { \prime } = ( ( ( 2 , 2 ) , ( 3 , 3 ) ) , ( ( 5 , 5 ) , ( 7 , 7 ) ) ) { \mathrm { ~ a n d ~ } } X = ( ( 4 , 9 ) , ( 2 5 , 4 9 ) )$ , then 

$$
\operatorname{flat} \left(X ^ {\prime}, X\right) = ((2, 2), (3, 3), (5, 5), (7, 7)).
$$

Example 2.2.4.13. If X is any nested tuple, then the flattening of X relative to X is 

$$
\operatorname{flat} (X, X) = X ^ {\flat}.
$$

Example 2.2.4.14. If $X = X ^ { \flat }$ is a tuple, and $X ^ { \prime }$ refines $X$ , then the flattening of $X ^ { \prime }$ relative to $X$ is 

$$
\operatorname{flat} \left(X ^ {\prime}, X\right) = X ^ {\prime}.
$$

Example 2.2.4.15. If $X ^ { \prime }$ is a nested tuple with size $( X ^ { \prime } ) = N$ , then $X ^ { \prime }$ refines $N _ { ; }$ , and the flattening of $X ^ { \prime }$ relative to $N$ is 

$$
\operatorname{flat} \left(X ^ {\prime}, N\right) = (N).
$$

Observation 2.2.4.16. If $X ^ { \prime }$ refines $X$ , then $\mathsf { f l a t } ( X ^ { \prime } , X )$ refines $X ^ { \flat }$ . 

## 2.3 Layouts

Having developed the necessary background on nested tuples, we turn our attention to layouts. These are a generalization of flat layouts in which shapes and strides are allowed to be nested tuples, rather than (flat) tuples. 

## 2.3.1 Basic definitions

Definition 2.3.1.1. A layout is a pair 

$$
L = S: D
$$

consisting of a nested tuple of positive integers 

$$
\operatorname{shape} (L) = S
$$

called the shape of $L ,$ and a nested tuple of non-negative integers 

$$
\operatorname{stride} (L) = D
$$

called the stride of $L ,$ such that S and D are congruent. 

Definition 2.3.1.2. If $L = S : D$ is a layout, then the rank, length, depth, size, and profile of L are defined to be the rank, length, depth, size, and profile of S, respectively. 

Example 2.3.1.3. The layout $L = ( 3 , ( 3 , 2 ) ) : ( 3 , ( 1 , 1 0 ) )$ may be pictured as follows. 

<table><tr><td>0</td><td>1</td><td>2</td><td>10</td><td>11</td><td>12</td></tr><tr><td>3</td><td>4</td><td>5</td><td>13</td><td>14</td><td>15</td></tr><tr><td>6</td><td>7</td><td>8</td><td>16</td><td>17</td><td>18</td></tr></table>

Example 2.3.1.4. The layout L = ((2, 2), (2, 2)) : ((1, 4), (2, 8)) may be pictured as follows. 

<table><tr><td>0</td><td>2</td><td>8</td><td>10</td></tr><tr><td>1</td><td>3</td><td>9</td><td>11</td></tr><tr><td>4</td><td>6</td><td>12</td><td>14</td></tr><tr><td>5</td><td>7</td><td>13</td><td>15</td></tr></table>

Example 2.3.1.5. The layout 

$$
L = 1 0: 4
$$

has rank(L) = 1, len(L) = 1, depth(L) = 0, size(L) = 10, and prof(L) = ∗. 

Example 2.3.1.6. The layout 

$$
L = (7, (2, 1 0, 4), (3, 7)): (1, (7, 1 4, 1 4 0), (5 6 0, 1 6 8 0))
$$

$$
\text { has   } \operatorname{rank} (L) = 3, \text {   len } (L) = 6, \text {   depth } (L) = 2, \text {   size } (L) = 1 1 7 6 0, \text {   and   } \operatorname{prof} (L) = (*, (*, *, *), (*, *))
$$

Example 2.3.1.7. The layout 

$$
L = ((2, 2, 2, (2, 2))): ((1, 0, 8, (0, 1 6)))
$$

$$
\text { has   } \operatorname{rank} (L) = 1, \operatorname{len} (L) = 5, \operatorname{depth} (L) = 3, \operatorname{size} (L) = 3 2, \text {   and   } \operatorname{prof} (L) = ((*, *, *, (*, *))).
$$

Example 2.3.1.8. The pair 

$$
S: D = (2, (2, 2)): (1, 2, 4)
$$

is NOT a layout because S and D are not congruent. 

Definition 2.3.1.9. If $L = S : D$ is a layout, then for any $1 \leq i \leq \mathsf { r a n k } ( L )$ we define the ith mode of L to be the layout 

$$
\operatorname{mode} _ {i} (L) = \operatorname{mode} _ {i} (S): \operatorname{mode} _ {i} (D),
$$

and for any $1 \leq i \leq \mathsf { l e n } ( L )$ , we define the ith entry of L to be the layout 

$$
\operatorname{entry} _ {i} (L) = \operatorname{entry} _ {i} (S): \operatorname{entry} _ {i} (D).
$$

Example 2.3.1.10. If $L = \left( ( 2 , 2 ) , 9 \right) : \left( ( 3 , 6 ) , 1 2 \right)$ , then the modes of L are 

$$
\begin{array}{l} \text { mode } _ {1} (L) = (2, 2): (3, 6) \\ \text { mode } _ {2} (L) = 9: 1 2 \end{array}
$$

and the entries of L are 

$$
\begin{array}{l} \text {entry} _ {1} (L) = 2: 3 \\ \text {entry} _ {2} (L) = 2: 6 \\ \text {entry} _ {3} (L) = 9: 1 2. \end{array}
$$

Remark 2.3.1.11. If L is a layout, then the modes of L are also layouts, and the entries of L are layouts of depth 0. 

Remark 2.3.1.12. A flat layout L is precisely a layout of depth 1. On the other hand, if L is a layout, we may obtain a flat layout $L ^ { \flat }$ as follows. 

Definition 2.3.1.13. If $L = S : D$ is a layout, we define the flattening of L to be the flat layout 

$$
L ^ {\flat} = S ^ {\flat}: D ^ {\flat}.
$$

Example 2.3.1.14. The flattening of $L = 1 0 : 4 { \mathrm { ~ i s ~ } } L ^ { \flat } = ( 1 0 ) : ( 4 )$ 

Example 2.3.1.15. The flattening of 

$$
L = \left((2, 2, 2, (2, 2))\right): \left((1, 0, 8, (0, 1 6))\right)
$$

is 

$$
L ^ {\flat} = (2, 2, 2, 2, 2): (1, 0, 8, 0, 1 6).
$$

Remark 2.3.1.16. If L is a layout then len ${ \mathfrak { a } } ( L ) = { \mathsf { r a n k } } ( L ^ { \flat } )$ , and for any $1 \leq i \leq \mathsf { l e n } ( L )$ , we have 

$$
\operatorname{entry} _ {i} (L) = \operatorname{mode} _ {i} \left(L ^ {\flat}\right).
$$

We can use the flattening construction above to extend many concepts from flat layouts to nested layouts. For example: 

Construction 2.3.1.17 (Layout function). If L is a nested layout, we define the layout function $\Phi _ { L }$ of L by 

$$
\Phi_ {L} = \Phi_ {L ^ {\flat}},
$$

where $\Phi _ { L ^ { \flat } }$ is the layout function of Construction 2.1.2.19. Similarly, if N is such that $\mathsf { I m a g e } ( \Phi _ { L } ) \subset$ $[ 0 , N )$ , we define 

$$
\Phi_ {L} ^ {N} = \Phi_ {L ^ {\flat}} ^ {N}
$$

to be the factorization of $\Phi _ { L }$ through the inclusion $[ 0 , N ) \subset \mathbb { Z } .$ 

Example 2.3.1.18. If $L = ( ( 2 , 2 ) , 2 ) : ( ( 3 , 0 ) , 1 0 )$ , then the layout function 

$$
\Phi_ {L}: [ 0, 8) \to \mathbb {Z}
$$

of $L$ is given by 

$$
\Phi_ {L} \begin{array}{c c c c c c c c c} & 0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 \\ & \Big \downarrow & \Big \downarrow & \Big \downarrow & \Big \downarrow & \Big \downarrow & \Big \downarrow & \Big \downarrow & \Big \downarrow \\ & 0 & 3 & 0 & 3 & 1 0 & 1 3 & 1 0 & 1 3 \end{array}
$$

Given a layout $L ,$ we can obtain a flat layout $L ^ { \flat }$ , and a profile $P = { \mathsf { p r o f } } ( L )$ . Conversely, if we are given a flat layout L and a profile P with the same length as $L ,$ then we can construct a layout with flattening L and profile $P$ as follows. 

Construction 2.3.1.19. If L is a flat layout, and P is a profile with len $( P ) = { \mathsf { l e n } } ( L )$ , then we can define 

$$
L = L _ {P}
$$

to be the layout with shape 

$$
\operatorname{shape} (L) = \operatorname{shape} (L) _ {P}
$$

and stride 

$$
\operatorname{stride} (L) = \operatorname{stride} (L) _ {P}
$$

where $( - ) _ { P }$ is the P-substitution operation of Definition 2.2.1.8. 

Example 2.3.1.20. If $L = ( 8 , 8 , 8 ) : ( 1 , 6 4 , 8 )$ and $P = ( * , ( * , * ) )$ , then 

$$
L _ {P} = (8, (8, 8)), (1, (6 4, 8)).
$$

Example 2.3.1.21. If $L = ( 1 2 8 ) : ( 2 )$ and $P = *$ , then 

$$
L _ {P} = 1 2 8: 2.
$$

Proposition 2.3.1.22. If L<sup>′</sup> is a flat layout and P is a profile with len $\left( L ^ { \prime } \right) = \mathsf { l e n } ( P )$ , then there exists a unique layout L whose flattening is $L ^ { \flat } = L ^ { \prime }$ and whose profile is prof $( L ) = P$ , namely $L = L _ { P } ^ { \prime }$ 

Proof. This follows from the definition of nested tuples, since a nested tuple is uniquely determined by its flattening and its profile. □ 

Observation 2.3.1.23. The previous proposition tells us that we have a pullback square 

$$
\begin{array}{c} \text {Layout} \xrightarrow {\text {prof(-)}} \text {Profile} \\ (-) ^ {b} \Biggl \downarrow \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \text {len(-)} \\ \text {FlatLayout} \xrightarrow {\text {len(-)}} \mathbb {N} \end{array}
$$

We can extend the notion of non-degeneracy to the nested case as follows. 

Definition 2.3.1.24. Suppose L is a layout. We say L is non-degenerate if for all $1 \leq i \leq \mathsf { l e n } ( L )$ , the following condition holds: 

$$
\operatorname{entry} _ {i} (\operatorname{shape} (L)) \quad \Rightarrow \quad \operatorname{entry} _ {i} (\operatorname{stride} (L))
$$

Example 2.3.1.25. The layouts 

$$
\begin{array}{l} L _ {1} = ((2, 2), 1): ((1, 2), 0) \\ L _ {2} = ((8, 8), (1, 1 6)): ((2, 3 2), (0, 1 2 8)) \end{array}
$$

are non-degenerate, while the layouts 

$$
\begin{array}{l} L _ {3} = ((2, 2), 1): ((1, 2), 4) \\ L _ {4} = ((8, 8), (1, 1 6)): ((2, 3 2), (1 0 2 4, 1 2 8)) \end{array}
$$

are degenerate. 

## 2.3.2 Basic operations

Having established the basic vocabulary for layouts, we turn to the operations they support. In this section, we define basic operations that will be needed to construct more sophisticated operations such as coalesce, complement, composition, logical division, and logical product. 

## 2.3.2.1 Flattening

If L is a layout, then we may obtain a flat layout $L ^ { \flat }$ by flattening the shape and stride of $L .$ 

Definition 2.3.2.1. If $L = S : D$ is a layout, we define the flattening of $L$ to be the flat layout 

$$
L ^ {\flat} = S ^ {\flat}: D ^ {\flat}.
$$

Example 2.3.2.2. The flattening of 

$$
L = ((2, 2, 2, (2, 2))): ((1, 0, 8, (0, 1 6)))
$$

is 

$$
L ^ {\flat} = (2, 2, 2, 2, 2): (1, 0, 8, 0, 1 6).
$$

Example 2.3.2.3. The flattening of $L = 1 0 : 4$ is $L ^ { \flat } = ( 1 0 ) : ( 4 )$ 

Example 2.3.2.4. Suppose L is a layout. Then depth $( L ) = 1$ if and only if $L = L ^ { \flat }$ 

## 2.3.2.2 Concatenate

We can concatenate layouts by concatenating their shapes and concatenating their strides. 

Definition 2.3.2.5. If $L = S : D$ and $L ^ { \prime } = S ^ { \prime } : D ^ { \prime }$ are layouts, then the concatenation of $L$ and $L ^ { \prime }$ is the layout (L, L<sup>′</sup>) 

$$
(L, L ^ {\prime}) = (S, S ^ {\prime}): (D, D ^ {\prime}).
$$

More generally, if $L _ { 1 } , \ldots , L _ { k }$ is any finite collection of layouts, with $L _ { i } = S _ { i } : D _ { i }$ , then the concatenation of $L _ { 1 } , \ldots , L _ { k }$ is the layout 

$$
(L _ {1}, \dots , L _ {k}) = (S _ {1}, \dots , S _ {k}): (D _ {1}, \dots , D _ {k}).
$$

Remark 2.3.2.6. Concatenation of nested tuples (and hence of layouts) is not associative. For example, take $L _ { 1 } = 3 : 4 , L _ { 2 } = 2 : 2$ , and $L _ { 3 } = 5 : 1$ . Then 

$$
\left(L _ {1}, \left(L _ {2}, L _ {3}\right)\right) = (3, (2, 5)): (4, (2, 1)) \neq ((3, 2), 5): ((4, 2), 1) = \left(\left(L _ {1}, L _ {2}\right), L _ {3}\right).
$$

Moreover, neither of these layouts is equal to the “three-fold” concatenation $( L _ { 1 } , L _ { 2 } , L _ { 3 } ) = ( 3 , 2 , 5 )$ $( 4 , 2 , 1 )$ . However, we see that each of these layouts has the same flattening, so each of these layouts has the same layout function. 

Example 2.3.2.7. If $L = ( 3 , 7 , 2 ) : ( 1 , 3 , 6 )$ and $L ^ { \prime } = ( 2 , ( 2 , ( 4 , 3 ) ) ) : ( 5 , 3 , ( 2 , 2 ) )$ , then 

$$
(L, L ^ {\prime}) = ((3, 7, 2), (2, (2, (4, 3)))): ((1, 3, 6), (5, (3, (2, 2))))
$$

Remark 2.3.2.8. Concatenation increases the depth of layouts. More precisely, we have 

$$
\operatorname{depth} (L, L ^ {\prime}) = 1 + \max (\operatorname{depth} (L), \operatorname{depth} (L ^ {\prime})).
$$

Remark 2.3.2.9. When L and $L ^ { \prime }$ are flat layouts, the concatenation of Definition 2.3.2.5 does NOT agree with the concatenation of flat layouts of Definition 2.1.3.36. Instead, these operations are related by the formula 

$$
L \star L ^ {\prime} = (L, L ^ {\prime}) ^ {\flat}.
$$

Remark 2.3.2.10. If L is any layout with depth $( L ) > 0$ and $\mathsf { r a n k } ( L ) = r$ , then we may write 

$$
L = (\operatorname{mode} _ {1} (L), \dots , \operatorname{mode} _ {r} (L))
$$

as the concatenation of its modes. 

Example 2.3.2.11. If 

$$
L = ((5, (7, 7)), 2, (4, 5)): ((1, (3 5, 5)), 0, (1, 8))
$$

then $L = \left( L _ { 1 } , L _ { 2 } , L _ { 3 } \right)$ where 

$$
\begin{array}{l} L _ {1} = \big (5, (7, 7) \big): \big (1, (3 5, 5) \big), \\ L _ {2} = 2: 0, \text {and} \\ L _ {3} = (4, 5): (1, 8). \end{array}
$$

## 2.3.2.3 Substitution

Recall that if $X _ { 1 } , \ldots , X _ { k }$ are nested tuples and P is a profile with $\mathsf { l e n } ( P ) = k ,$ then we may form the P-substitution 

$$
(X _ {1}, \ldots , X _ {k}) _ {P}
$$

which is obtained by replacing the ithe entry of P with the nested tuple $X _ { i }$ . We can extend this construction from nested tuples to layouts as follows. 

Definition 2.3.2.12. Suppose $L = S : D$ is a layout, and suppose P is a profile with len $( P ) = \mathsf { r a n k } ( L )$ We define 

$$
L _ {P} = S _ {P}: D _ {P}
$$

where $S _ { P }$ and $D _ { P }$ are the P-substitutions of (the modes of) S and D. 

Example 2.3.2.13. If $P = ( * , ( * , * ) )$ and $L = ( 8 , 8 , 8 ) : ( 1 , 8 , 6 4 )$ , then 

$$
L _ {P} = (8, (8, 8)): (1, (8, 6 4)).
$$

Example 2.3.2.14. If $P = ( * , ( * , * ) ) )$ and 

$$
L = ((2, 2), (3, 3), (5, 5)): ((2, 1), (1 2, 4), (1 8 0, 3 6)),
$$

then 

$$
L _ {P} = ((2, 2), ((3, 3), (5, 5))): ((2, 1), ((1 2, 4), (1 8 0, 3 6))).
$$

Example 2.3.2.15. If $L = ( 1 6 ) : ( 1 )$ and $P = *$ , then 

$$
L _ {P} = 1 6: 1.
$$

## 2.3.3 Coalesce

Recall that if L is a flat layout, then ${ \mathsf { c o a l } } ^ { \flat } ( L )$ is a the unique flat layout of minimal rank whose layout function is $\Phi _ { L }$ . We can make a similar construction in the setting of arbitrary (nested) layouts. We begin by defining the notion of a coalesced layout. 

Definition 2.3.3.1. Suppose L is a layout. We say L is coalesced if one of the following conditions holds. 

1. $L = 1 : 0 ,$ 

2. d $\mathsf { e p t h } ( L ) = 0$ and $\mathsf { s h a p e } ( L ) > 1$ , or 

3. depth $( L ) = 1 , \mathsf { r a n k } ( L ) > 1$ , and L is coalesced in the sense of Definition 2.1.4.1. 

Example 2.3.3.2. The layout 

$$
L = (2, (2, 2)): (1, (1 6, 5 1 2))
$$

is not coalesced since depth $( L ) > 1$ 

Example 2.3.3.3. The layout 

$$
L = (6 4): (2)
$$

is not coalesced, while the layout 

$$
L ^ {\prime} = 6 4: 2
$$

is coalesced. 

Example 2.3.3.4. The layout 

$$
L = 1: 8
$$

is not coalesced, while the layout 

$$
L ^ {\prime} = 1: 0
$$

is coalesced. 

Example 2.3.3.5. The empty layout 

$$
E = (): ()
$$

is not coalesced. 

Observation 2.3.3.6. Recall that a layout L is non-degenerate if 

$$
\operatorname{entry} _ {i} (\operatorname{shape} (L)) = 1 \quad \Rightarrow \quad \operatorname{entry} _ {i} (\operatorname{stride} (L)) = 0.
$$

If L is coalesced, then L is non-degenerate. 

If L is any layout, we can obtain a coalesced layout coal(L) as follows. 

Construction 2.3.3.7. Suppose L is a layout, and write 

$$
\operatorname{coal} ^ {\flat} \left(L ^ {\flat}\right) = \left(s _ {1}, \dots , s _ {m}\right): \left(d _ {1}, \dots , d _ {m}\right).
$$

1. If $m > 1$ , we define 

$$
\operatorname{coal} (L) = \operatorname{coal} ^ {\flat} \left(L ^ {\flat}\right)
$$

2. If m = 1, we define 

$$
\operatorname{coal} (L) = s _ {1}: d _ {1}
$$

3. If $m = 0 .$ , we define 

$$
\operatorname{coal} (L) = 1: 0.
$$

Example 2.3.3.8. If $E = ( ) : ( )$ is the empty layout, then 

$$
\operatorname{coal} (E) = 1: 0.
$$

Example 2.3.3.9. If $L = ( 1 , 1 ) : ( 2 , 4 )$ , then 

$$
\operatorname{coal} (L) = 1: 0.
$$

Example 2.3.3.10. If $L = ( 5 1 2 ) : ( 4 )$ , then 

$$
\operatorname{coal} (L) = 5 1 2: 4.
$$

Example 2.3.3.11. If $L = ( 2 , 2 , 2 ) : ( 1 , 2 , 4 )$ , then 

$$
\operatorname{coal} (L) = 8: 1.
$$

Example 2.3.3.12. If $L = ( ( 2 , 2 , 2 ) , ( 5 , 5 ) ) : ( ( 1 , 2 , 4 ) , ( 1 0 , 5 0 ) )$ , then 

$$
\operatorname{coal} (L) = (8, 2 5): (1, 1 0).
$$

Remark 2.3.3.13. If L is a layout, then coal(L) has depth 0 or 1. 

Proposition 2.3.3.14. If A and B are layouts, then 

$$
\Phi_ {A} = \Phi_ {B} \quad \Leftrightarrow \quad \operatorname{coal} (A) = \operatorname{coal} (B).
$$

Proof. Using Proposition 2.1.4.18, we have 

$$
\begin{array}{l l l} \Phi_ {A} = \Phi_ {B} & \Leftrightarrow & \Phi_ {A ^ {\flat}} = \Phi_ {B ^ {\flat}} \\ & \Leftrightarrow & \operatorname{coal} ^ {\flat} (A ^ {\flat}) = \operatorname{coal} ^ {\flat} (B ^ {\flat}) \\ & \Leftrightarrow & \operatorname{coal} (A) = \operatorname{coal} (B). \end{array}
$$

Definition 2.3.3.15. If L is a layout, define the complexity of L to be the integer 

$$
\text { complexity } (L) = \text { len } (L) + \text { depth } (L).
$$

Proposition 2.3.3.16. If L is a layout and $\mathsf { s i z e } ( L ) > 1$ , then $\mathsf { c o a l } ( L )$ is the unique complexity minimizing layout whose layout function is $\Phi _ { L }$ 

Proof. Suppose $L ^ { \prime }$ is a layout with the same layout function as L, and suppose ${ \mathsf { c o a l } } ( L ^ { \prime } ) \neq 1 : 0$ . Then 

$$
\operatorname{len} \left(L ^ {\prime}\right) \geq \operatorname{len} (\operatorname{coal} \left(L ^ {\prime}\right)) = \operatorname{len} (\operatorname{coal} (L)).
$$

There are two cases to consider. 

• (Case 1): Suppose len $\mathsf { \Omega } _ { 1 } ( L ^ { \prime } ) \mathsf { \Omega } > \mathsf { \Omega } 1$ . Then depth $( L ^ { \prime } ) \ge 1 \ge \mathsf { d e p t h } ( \mathsf { c o a l } ( L ) )$ ). Combining these inequalities, we observe that 

$$
\text { complexity } (L ^ {\prime}) \geq \text { complexity } (\text { coal } (L)),
$$

where equality holds if and only if $L ^ { \prime } = { \mathsf { c o a l } } ( L ^ { \prime } ) = { \mathsf { c o a l } } ( L )$ 

• (Case 2): Suppose len $( L ^ { \prime } ) = 1$ . Then $L ^ { \prime } = ( s ) : ( d )$ or $L ^ { \prime } = s : d$ for some integers $s > 1$ and $d \geq 0$ . In either case, we have ${ \mathsf { c o a l } } ( L ^ { \prime } ) = s : d ,$ and 

$$
\text { complexity } (L ^ {\prime}) \geq \text { complexity } (\text { coal } (L)),
$$

where equality holds if and only if $L ^ { \prime } = s : d = { \mathsf { c o a l } } ( L )$ 

Remark 2.3.3.17. The only reason that we need to exclude the case size $( L ) = 1$ is that if $\mathsf { s i z e } ( L ) = 1$ then 1 : 0 and the empty layout $( ) : ( )$ are distinct layouts with minimal complexity, and the same layout function as L (namely the trivial layout function $0 \mapsto 0 )$ 

## 2.3.4 Relative coalesce

There is an important invariant of coalesce called relative coalesce, denoted coal $( L , { \bar { S } } )$ . This operation receives as an additional input a nested tuple S<sup>¯</sup> which is refined by $\mathsf { s h a p e } ( L )$ . In this case, the relative coalesce operation simplifies the layout L has much as possible, while ensuring that the resulting shape still refines $\bar { S } .$ 

Definition 2.3.4.1. Suppose $L = S : D$ is a layout, and suppose $\bar { S }$ is some nested tuple of length m which is refined by S. Recall that for any $1 \leq i \leq m$ , we may consider the ith mode of $S$ relative to ${ \bar { S } } ,$ denoted 

$$
\operatorname{mode} _ {i} (S, \bar {S}).
$$

Since $S$ and D are congruent, there is a nested tuple 

$$
\mathsf {m o d e} _ {i} (D, \bar {S})
$$

corresponding to mode $\mathbf { \Omega } _ { i } ( S , \bar { S } )$ , and we define the ith mode of L relative to $\bar { S }$ to be the layout 

$$
\operatorname{mode} _ {i} (L, \bar {S}) = \operatorname{mode} _ {i} (S, \bar {S}): \operatorname{mode} _ {i} (D, \bar {S}).
$$

Example 2.3.4.2. If ${ \bar { S } } = ( 4 , ( 9 , 2 5 ) )$ and 

$$
L = ((2, 2), ((3, 3), (5, (1, 5)))): ((1, 2), ((6, 1 8), (9 0, (0, 4 5 0))))
$$

then 

$$
\begin{array}{l} \text {mode} _ {1} (L, \bar {S}) = (2, 2): (1, 2) \\ \text {mode} _ {2} (L, \bar {S}) = (3, 3): (6, 1 8) \\ \text {mode} _ {3} (L, \bar {S}) = (5, (1, 5)): (9 0, (0, 4 5 0)). \end{array}
$$

Observation 2.3.4.3. Suppose $L = S : D$ is a layout, and suppose $\bar { S }$ is a nested tuple of length m and profile $P$ which is refined by S. If for any $1 \leq i \leq m$ , we write 

$$
L _ {i} = \operatorname{mode} _ {i} (L, \bar {S}),
$$

then 

$$
L = (L _ {1}, \dots , L _ {m}) _ {P}
$$

is the P-substitution of its relative modes. 

Definition 2.3.4.4. Suppose $L = S : D$ is a layout, and suppose $\bar { S }$ is a nested tuple of length m and profile $P$ which is refined by S. We say L is coalesced over $\bar { S }$ if each relative mode 

$$
\mathsf {m o d e} _ {i} (L, \bar {S})
$$

is coalesced. 

Observation 2.3.4.5. In the setting of Definition 2.3.4.4, if L is coalesced over ${ \bar { S } } ,$ then $L$ is nondegenerate. 

Example 2.3.4.6. If L is a layout, then L is coalesced over shape(L) if and only if L is non-degenerate, i.e. 

$$
\operatorname{entry} _ {i} (\operatorname{shape} (L)) = 1 \quad \Rightarrow \quad \operatorname{entry} _ {i} (\operatorname{stride} (L)) = 0.
$$

Definition 2.3.4.7 (Relative coalesce). Suppose $L = S : D$ is a layout, and suppose $\bar { S }$ is a nested tuple of length m and profile $P$ which is refined by S. We define 

$$
\operatorname{coal} (L, \bar {S}) = (\operatorname{coal} (L _ {1}), \dots , \operatorname{coal} (L _ {m})) _ {P}.
$$

Remark 2.3.4.8. In the setting of Definition 2.3.4.7, the shape of $\mathsf { c o a l } ( L , { \bar { S } } )$ refines $\bar { S } .$ 

Lemma 2.3.4.9. If $L = S : D$ is a layout and $S$ refines ${ \bar { S } } ,$ then 

$$
\Phi_ {\mathrm{coal} (L, \bar {S})} = \Phi_ {L}.
$$

Proof. As above, let 

$$
L _ {i} = \operatorname{mode} _ {i} (L, S)
$$

denote the ith mode of L relative to $S ,$ and set $\bar { L } _ { i } = \mathsf { c o a l } ( L _ { i } )$ . Then 

$$
\begin{array}{r l} \Phi_ {\mathsf {c o a l} (L, \bar {S})} & = \Phi_ {(\bar {L} _ {1}, \dots , \bar {L} _ {m}) _ {\bar {S}}} \\ & = \Phi_ {(\bar {L} _ {1}, \dots , \bar {L} _ {m})} \\ & = \Phi_ {\mathsf {c o a l} ((\bar {L} _ {1}, \dots , \bar {L} _ {m}))} \\ & = \Phi_ {\mathsf {c o a l} ((L _ {1}, \dots , L _ {m}))} \\ & = \Phi_ {(L _ {1}, \dots , L _ {m})} \\ & = \Phi_ {(L _ {1}, \dots , L _ {m}) _ {\bar {S}}} \\ & = \Phi_ {L}. \end{array}
$$

Proposition 2.3.4.10. Suppose A and B are layouts, and suppose $\bar { S }$ is a nested tuple of length m such that shape(A) refines $\bar { S }$ and shape(B) refines S<sup>¯</sup>. Then 

$$
\Phi_ {A} = \Phi_ {B} \quad \Leftrightarrow \quad \mathsf {c o a l} (A, \bar {S}) = \mathsf {c o a l} (B, \bar {S})
$$

Proof. If $\mathsf { c o a l } ( A , { \bar { S } } ) = \mathsf { c o a l } ( B , { \bar { S } } )$ , the using Lemma 2.3.4.9, we have 

$$
\Phi_ {A} = \Phi_ {\mathsf {c o a l} (A, \bar {S})} = \Phi_ {\mathsf {c o a l} (B, \bar {S})} = \Phi_ {B}.
$$

Conversely, suppose that $\Phi _ { A } = \Phi _ { B }$ . We will argue that coa $| ( A , \bar { S } ) = \mathsf { c o a l } ( B , \bar { S } )$ . Set $P = { \mathsf { p r o f } } ( { \bar { S } } )$ , and for any $1 \leq i \leq m$ , set 

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

it sufices to prove that coa $| ( A _ { i } ) = { \mathsf { c o a l } } ( B _ { i } )$ for all $1 \leq i \leq m$ . By the associativity of colexicographic isomorphisms, we can write the layout function $\Phi _ { A }$ of A as 

$$
[ 0, \operatorname{size} (A)) \xrightarrow {\operatorname{colex} ^ {- 1}} \prod_ {j = 1} ^ {m} [ 0, \operatorname{size} (A _ {j})) \xrightarrow {\prod \Phi_ {A _ {j}}} \prod_ {j = 1} ^ {m} \mathbb {Z} \xrightarrow {+} \mathbb {Z}
$$

and we can write the layout function $\Phi _ { B }$ of $B$ as 

$$
[ 0, \operatorname{size} (B)) \xrightarrow {\operatorname{colex} ^ {- 1}} \prod_ {j = 1} ^ {m} [ 0, \operatorname{size} (B _ {j})) \xrightarrow {\prod \Phi_ {B _ {j}}} \prod_ {j = 1} ^ {m} \mathbb {Z} \xrightarrow {+} \mathbb {Z}
$$

For a fixed $1 \leq i \leq m$ , consider the subset 

$$
[ 0, \operatorname{size} (A _ {i})) \subset \prod_ {j = 1} ^ {m} [ 0, \operatorname{size} (A _ {j}))
$$

and its image 

$$
\operatorname{colex} ([ 0, \operatorname{size} (A _ {i}))) \subset [ 0, \operatorname{size} (A)).
$$

Since ${ \mathsf { s i z e } } ( A _ { j } ) = { \mathsf { s i z e } } ( B _ { j } )$ for all $1 \leq j \leq m$ , this is the same as the image 

$$
\operatorname{colex} ([ 0, \operatorname{size} (B _ {j}))) \subset [ 0, \operatorname{size} (B)) = [ 0, \operatorname{size} (B)).
$$

The restriction of $\Phi _ { A }$ to this subset is $\Phi _ { A _ { i } }$ , and the restriction of B to this subset is $\Phi _ { B _ { i } }$ , so it follows that $\Phi _ { A _ { i } } = \Phi _ { B _ { i } }$ , and by Proposition 2.3.3.14, we have coa $( A _ { i } ) = { \mathsf { c o a l } } ( B _ { i } )$ . We deduce that 

$$
\operatorname{coal} (A, \bar {S}) = \operatorname{coal} (B, \bar {S}),
$$

as desired. 

## 2.3.5 Compact layouts

We can easily extend the concept of compact layouts to the nested case. Again, in terms of the standard grid diagrams depicting layouts, a layout L is compact if each integer $0 \leq i < \mathsf { s i z e } ( L )$ appears exactly once. More preciesly, we have the following definition. 

Definition 2.3.5.1. Suppose L is a layout. We say L is compact if the layout function 

$$
\Phi_ {L} ^ {\text { cosize } (L)}: [ 0, \text { size } (L)) \to [ 0, \text { cosize } (L))
$$

is an isomorphism. 

Example 2.3.5.2. The layout 

$$
A = ((2, 2), (2, 2)): ((1, 4), (2, 8)) =
$$

<table><tr><td>0</td><td>2</td><td>8</td><td>10</td></tr><tr><td>1</td><td>3</td><td>9</td><td>11</td></tr><tr><td>4</td><td>6</td><td>12</td><td>14</td></tr><tr><td>5</td><td>7</td><td>13</td><td>15</td></tr></table>

is compact, while the layouts 

$$
B = ((2, 2), (2, 2)): ((1, 4), (2, 3 2)) =
$$

<table><tr><td>0</td><td>2</td><td>32</td><td>34</td></tr><tr><td>1</td><td>3</td><td>33</td><td>35</td></tr><tr><td>4</td><td>6</td><td>36</td><td>38</td></tr><tr><td>5</td><td>7</td><td>37</td><td>39</td></tr></table>

and 

$$
C = ((2, 2), (2, 2)): ((1, 4), (2, 0)) = \begin{array}{c c c c} \hline 0 & 2 & 0 & 2 \\ \hline 1 & 3 & 1 & 3 \\ \hline 4 & 6 & 4 & 6 \\ \hline 5 & 7 & 5 & 7 \\ \hline \end{array}
$$

are not compact. 

Example 2.3.5.3. The following layouts are compact: 

$$
\begin{array}{l} L _ {1} = (2, (2, 2)): (8, (1, 4)) \\ L _ {2} = ((8, 1), (8, 3 2)): ((2, 0), (1 6, 1 2 8)) \\ L _ {3} = 6 4: 1 \end{array}
$$

Example 2.3.5.4. The layout 

$$
L = (2, (2, 2)): (4, (8, 1 6))
$$

is not compact since the integer $1 \in [ 0 , 2 9 ) = [ 0 , \mathsf { c o s i z e } ( L ) )$ is not in the image of $\Phi _ { L }$ . More generally, if $\mathsf { s i z e } ( L ) \neq \mathsf { c o s i z e } ( L )$ , then L is not compact. 

We conclude this section by listing some equivalent conditions for a layout L to be compact. 

Proposition 2.3.5.5. Suppose L is a layout. Then the following conditions are equivalent. 

1. L is compact. 

2. $L ^ { \flat }$ is compact. 

3. coal(L) is compact. 

Proof. The equivalence of these conditions follows from the fact that 

$$
\Phi_ {L} = \Phi_ {L ^ {\flat}} = \Phi_ {\text { coal } (L)}.
$$

## 2.3.6 Complements

We can easily extend the concept of complement to the nested case as follows. 

Definition 2.3.6.1. Suppose A and B are layouts. We say B is a complement of A, and write $A \perp B$ if the concatenated layout $( A , B )$ is compact. 

Lemma 2.3.6.2. Suppose A and B are layouts. Then 

$$
A \perp B \quad \Leftrightarrow \quad A ^ {\flat} \perp B ^ {\flat}.
$$

Proof. This follows from the observation that $( A , B ) ^ { \flat } = A ^ { \flat } \star B ^ { \flat }$ 

Definition 2.3.6.3. Suppose A is a layout. We say A is complementable if $A ^ { \flat }$ is complementable. 

Lemma 2.3.6.4. Suppose A is a layout. Then there exists a complement B of A if and only if A is complementable. 

Proof. If A is complementable, then $A ^ { \flat }$ is complementable, so there exists a flat layout B such that the flat concatenation $A ^ { \flat } \star B$ is compact. It follows that the concatenation $( A , B )$ is also compact, so A admits a complement. Conversely, suppose there exists a layout B such that $( A , B )$ is compact. Then $B ^ { \flat }$ is a complement of $A ^ { \flat }$ , so by Proposition $2 . 1 . 6 . 2 1 , A ^ { \flat }$ is complementable, hence, by definition, so is A. □ 

Definition 2.3.6.5. Suppose A is a layout. If A is complementable, then we define 

$$
\operatorname{comp} (A) = \operatorname{coal} (\operatorname{comp} ^ {\flat} (A ^ {\flat})),
$$

as in Construction 2.1.6.16. If N is a positive integer and A is N-complementable, then we define 

$$
\operatorname{comp} (A, N) = \operatorname{coal} (\operatorname{comp} ^ {\flat} (A ^ {\flat}, N))
$$

as in Construction 2.1.6.29. 

Remark 2.3.6.6. Suppose A is a complementable layout. Then we almost always have ${ \mathsf { c o m p } } ( A ) =$ $\mathsf { c o m p } ^ { \flat } ( A ^ { \flat } )$ . More precisely, ${ \mathrm { i f ~ } } \mathsf { c o m p } ^ { \flat } ( A ^ { \flat } )$ has length $> 1$ , then 

$$
\operatorname{comp} (A) = \operatorname{comp} ^ {\flat} (A ^ {\flat}),
$$

if ${ \mathsf { c o m p } } ^ { \flat } ( A ^ { \flat } ) = ( s ) : ( d )$ has length 1, then 

$$
\operatorname{comp} (A) = s: d,
$$

and if ${ \mathsf { c o m p } } ^ { \flat } ( A ^ { \flat } ) = ( ) : ( )$ , then 

$$
\operatorname{comp} (A) = 1: 0.
$$

Definition 2.3.6.7. Suppose A is a layout and N is a positive integer. We say a layout B is a N-complement of A if $A \perp B$ , and 

$$
\operatorname{size} (A) \cdot \operatorname{size} (B) = N.
$$

Definition $\mathbf { 2 . 3 . 6 . 8 } .$ . Suppose A is a layout and N is a positive integer. We say A is N-complementable if the flat layout $A ^ { \flat }$ is N-complementable, as in Definition 2.1.6.24. 

Proposition 2.3.6.9. Suppose A is a layout. Then there exists a N-complement of A if and only if A is N-complementable. 

Proof. If B is a N-complement of A, then $B ^ { \flat }$ is a N-complement $A ^ { \flat } .$ , and so by Proposition 2.1.6.32, $A ^ { \flat }$ is N-complementable, hence, so is A. Conversely, if A is N-complementable, then $\mathsf { c o m p } ( A , N )$ is a N-complement of A. □ 

Example 2.3.6.10. If A = ((4, 2), (2, 2)) : ((3, 24), (192, 96)) and N = 768 then 

$$
\operatorname{comp} (A, N) = (3, 2, 2, 2): (1, 1 2, 4 8, 3 8 4).
$$

Example 2.3.6.11. If $A = \left( ( 1 6 , 4 ) , 6 4 \right) : \left( ( 1 , 1 6 ) , 6 4 \right)$ and N = 4096 then 

$$
\begin{array}{c} \mathsf {c o m p} (A, N) = \mathsf {c o a l} (\left(\left(\right): \left(\right)\right) \\ = 1: 0. \end{array}
$$

Example 2.3.6.12. If $A = \left( ( 1 6 , 4 ) , 6 4 \right) : \left( ( 1 , 1 6 ) , 6 4 \right) \mathrm { { a n d } } \ N = 8 1 9 2 \mathrm { { t h e n } }$ 

$$
\begin{array}{c} \mathsf {c o m p} (A, N) = \mathsf {c o a l} ((2): (4 0 9 6)) \\ = 2: 4 0 9 6. \end{array}
$$

Example 2.3.6.13. If $A = \left( ( 1 6 , 4 ) , 6 4 \right) : ( ( 8 , 1 ) , 1 2 8 )$ and $N = 1 6 3 8 4$ , then 

$$
\begin{array}{c} \mathsf {c o m p} (A, N) = \mathsf {c o a l} ((2, 2): (4, 8 1 9 2)) \\ = (2, 2): (4, 8 1 9 2). \end{array}
$$

## 2.3.7 Composition

In this section, we discuss the most important operation on layouts, namely composition. If A and B are layouts, then the composition of A and B is a layout $B \circ A$ whose layout function is the composite of the layout functions of A and B. More precisely, we have the following definition. 

Definition 2.3.7.1 (Composition of layouts). Suppose A and B are layouts. The composite of A and B is the unique layout $B \circ A$ satisfying the following properties. 

1. shape(B ◦ A) refines shape(A), 

2. $B \circ A$ is coalesced over shape(A), and 

3. $\Phi _ { B \circ A } = \Phi _ { B } \circ \Phi _ { A } ^ { \mathsf { s i z e } ( B ) }$ 

Remark 2.3.7.2. In order for $B \circ A$ to exist, we must have 

$$
\operatorname{Image} \left(\Phi_ {A}\right) \subseteq [ 0, \text { size } (B)).
$$

Remark 2.3.7.3. There is an implicit assertion in the definition of layout composition, namely that there is at most one layout satisfying the three conditions. This is justified by Proposition 2.3.4.10. We might define a weak composite of A and B to be a layout C satisfying conditions 1. and 3. (but not necessarily 2.), in which case 

$$
B \circ A = \operatorname{coal} (C, \text { shape } (A))
$$

We will see later on that when attempting to compute compositions of layouts, it is useful to compute any weak composite C of A and B, then coalesce over shape(A) to form the actual composite $B \circ A$ Remark 2.3.7.4. Note that, by Observation 2.3.4.5, condition 2. in the definition of composition implies that $B \circ A$ is non-degenerate. 

Example 2.3.7.5. If $A = ( 3 , 5 ) : ( 1 0 , 2 )$ and $B = ( 1 0 0 ) : ( 7 )$ , then 

$$
B \circ A = (3, 5): (7 0, 1 4).
$$

Example 2.3.7.6. If $A = ( 4 ) : ( 2 )$ and $B = ( 2 , 2 , 6 ) : ( 1 2 , 6 , 1 )$ , then the composition of A and B is 

$$
B \circ A = ((2, 2)): ((6, 1)).
$$

Remark 2.3.7.7. Example 2.3.7.6 illustrates the fact that the composition of flat layouts A and B need not be flat. 

Example 2.3.7.8. If $A = ( ( 2 , 4 ) , 8 ) : ( ( 4 , 8 ) , 8 )$ and $B = ( 4 , 4 , 4 , 4 ) : ( 2 , 4 , 8 , 1 6 )$ , then 

$$
B \circ A = ((2, (2, 2)), (2, 4)): ((4, (8, 8)), (8, 8)).
$$

<table><tr><td>0</td><td>4</td><td>8</td><td>12</td><td>16</td><td>20</td><td>24</td><td>28</td></tr><tr><td>1</td><td>5</td><td>9</td><td>13</td><td>17</td><td>21</td><td>25</td><td>29</td></tr><tr><td>2</td><td>6</td><td>10</td><td>14</td><td>18</td><td>22</td><td>26</td><td>30</td></tr><tr><td>3</td><td>7</td><td>11</td><td>15</td><td>19</td><td>23</td><td>27</td><td>31</td></tr></table>

Example 2.3.7.9. If A = ((3, (2, 2)), 24) : ((3, (9, 18)), 72) and B = (9, 8, 3, 8) : (24, 3, 1, 384) then 

$$
B \circ A = ((3, (2, 2)), (3, 8)): ((7 2, (3, 6)), (1, 3 8 4))
$$

Next, we develop some useful properties for computing the composition of layouts. 

Proposition 2.3.7.10. Suppose A is a layout, and suppose B and $\tilde { B }$ are layouts such that 

$\mathsf { s i z e } ( B ) \leq \mathsf { s i z e } ( \tilde { B } )$ , and 

$\Phi _ { \tilde { B } } \ | _ { \mathsf { s i z e } ( B ) } = \Phi _ { B }$ 

If A and B are composable, then 

$$
B \circ A = \tilde {B} \circ A.
$$

Proof. Suppose A and B are composable. Then cosize $( A ) \leq \mathsf { s i z e } ( B )$ , and the fact that B ◦ A is the composite of A and $\tilde { B }$ follows from the equality 

$$
\begin{array}{r l} & {\Phi_ {\tilde {B}} \circ \Phi_ {A} ^ {\mathrm{size} (\tilde {B})} = (\Phi_ {\tilde {B}}) | _ {\mathrm{size} (B)} \circ \Phi_ {A} ^ {\mathrm{size} (B)}} \\ & {\qquad = \Phi_ {B} \circ \Phi_ {A} ^ {\mathrm{size} (B)}.} \end{array}
$$

Corollary 2.3.7.11. If A and B are layouts, then A and B are composable if and only if A and coal(B) are composable, and 

$$
B \circ A = \operatorname{coal} (B) \circ A.
$$

Now that we have developed the basic properties of layout composition, we turn our attention to the two most important instances of composition, namely logical division and logical products. 

## 2.3.8 Logical division

In this section, we define the logical division of layouts. As a motivating example, consider the layout 

For various purposes, we may want to tile the layout A. For example, here are the tilings of A by various layouts B. 

<table><tr><td>0</td><td>4</td><td>8</td><td>12</td><td>16</td><td>20</td><td>24</td><td>28</td></tr><tr><td>1</td><td>5</td><td>9</td><td>13</td><td>17</td><td>21</td><td>25</td><td>29</td></tr><tr><td>2</td><td>6</td><td>10</td><td>14</td><td>18</td><td>22</td><td>26</td><td>30</td></tr><tr><td>3</td><td>7</td><td>11</td><td>15</td><td>19</td><td>23</td><td>27</td><td>31</td></tr></table>

$$
B = \begin{array}{c c} \hline 0 & 4 \\ \hline 1 & 5 \\ \hline \end{array}
$$

<table><tr><td>0</td><td>4</td><td>8</td><td>12</td><td>16</td><td>20</td><td>24</td><td>28</td></tr><tr><td>1</td><td>5</td><td>9</td><td>13</td><td>17</td><td>21</td><td>25</td><td>29</td></tr><tr><td>2</td><td>6</td><td>10</td><td>14</td><td>18</td><td>22</td><td>26</td><td>30</td></tr><tr><td>3</td><td>7</td><td>11</td><td>15</td><td>19</td><td>23</td><td>27</td><td>31</td></tr></table>

<table><tr><td>0</td><td>4</td><td>8</td><td>12</td></tr><tr><td>2</td><td>6</td><td>10</td><td>14</td></tr></table>

<table><tr><td>0</td><td>4</td><td>8</td><td>12</td><td>16</td><td>20</td><td>24</td><td>28</td></tr><tr><td>1</td><td>5</td><td>9</td><td>13</td><td>17</td><td>21</td><td>25</td><td>29</td></tr><tr><td>2</td><td>6</td><td>10</td><td>14</td><td>18</td><td>22</td><td>26</td><td>30</td></tr><tr><td>3</td><td>7</td><td>11</td><td>15</td><td>19</td><td>23</td><td>27</td><td>31</td></tr></table>

<table><tr><td>0</td><td>4</td><td>16</td><td>20</td></tr><tr><td>2</td><td>6</td><td>18</td><td>22</td></tr></table>

When working with such tiled layouts, we would like to index into our layout with coordinates of the form (tile coordinate, tile) where tile specifies which tile we are working with, and tile coordinate specifies a coordinate within the specified tile. For example, if both A and B have rank 2, we would like to write $( ( i , j ) , ( k , \ell ) )$ as the index of the $( i , j )$ th entry of the $( k , \ell ) \mathrm { t h }$ tile of A. The logical division of $A \oslash B$ is precisely the layout which afords us this ability. 

Definition 2.3.8.1. Suppose A and B are layouts, and suppose 

$$
B ^ {c} = \operatorname{comp} (B, \text { size } (A))
$$

is the complement of B with respect to size(A). Then the logical division of A by B is the layout 

$$
\begin{array}{c} A \oslash B = A \circ (B, B ^ {c}) \\ = (A \circ B, A \circ B ^ {c}). \end{array}
$$

Example 2.3.8.2. If $A = ( 4 , 8 ) : ( 1 , 4 )$ and $B = ( 2 , 2 ) : ( 1 , 4 )$ , then 

$$
A \oslash B = ((2, 2), (2, 4)): ((1, 4), (2, 8)),
$$

as depicted below. 

<table><tr><td>0</td><td>4</td><td>8</td><td>12</td><td>16</td><td>20</td><td>24</td><td>28</td></tr><tr><td>1</td><td>5</td><td>9</td><td>13</td><td>17</td><td>21</td><td>25</td><td>29</td></tr><tr><td>2</td><td>6</td><td>10</td><td>14</td><td>18</td><td>22</td><td>26</td><td>30</td></tr><tr><td>3</td><td>7</td><td>11</td><td>15</td><td>19</td><td>23</td><td>27</td><td>31</td></tr></table>

$$
B = \begin{array}{c c} \hline 0 & 4 \\ \hline 1 & 5 \\ \hline \end{array}
$$

<table><tr><td>0</td><td>2</td><td>8</td><td>10</td><td>16</td><td>18</td><td>24</td><td>26</td></tr><tr><td>1</td><td>3</td><td>9</td><td>11</td><td>17</td><td>19</td><td>25</td><td>27</td></tr><tr><td>4</td><td>6</td><td>12</td><td>14</td><td>20</td><td>22</td><td>28</td><td>30</td></tr><tr><td>5</td><td>7</td><td>13</td><td>15</td><td>21</td><td>23</td><td>29</td><td>31</td></tr></table>

Remark 2.3.8.3. The color of each entry in $A \oslash B$ indicates the tile to which it belongs, and the opacity of each entry in $A \oslash B$ indicates which entry of the tile it represents. This is why each column of $A \oslash B$ has the same color, and each row of $A \oslash B$ has the same opacity. 

Example 2.3.8.4. If $A = ( 4 , 8 ) : ( 1 , 4 )$ and $B = ( 2 , 2 ) : ( 4 , 1 )$ , then 

$$
A \oslash B = ((2, 2), (2, 4)): ((4, 1), (2, 8)),
$$

as depicted below. 

<table><tr><td>0</td><td>4</td><td>8</td><td>12</td><td>16</td><td>20</td><td>24</td><td>28</td></tr><tr><td>1</td><td>5</td><td>9</td><td>13</td><td>17</td><td>21</td><td>25</td><td>29</td></tr><tr><td>2</td><td>6</td><td>10</td><td>14</td><td>18</td><td>22</td><td>26</td><td>30</td></tr><tr><td>3</td><td>7</td><td>11</td><td>15</td><td>19</td><td>23</td><td>27</td><td>31</td></tr></table>

$$
B = \begin{array}{c c} \hline 0 & 1 \\ \hline 4 & 5 \\ \hline \end{array}
$$

<table><tr><td>0</td><td>2</td><td>8</td><td>10</td><td>16</td><td>18</td><td>24</td><td>26</td></tr><tr><td>4</td><td>6</td><td>12</td><td>14</td><td>20</td><td>22</td><td>28</td><td>30</td></tr><tr><td>1</td><td>3</td><td>9</td><td>11</td><td>17</td><td>19</td><td>25</td><td>27</td></tr><tr><td>5</td><td>7</td><td>13</td><td>15</td><td>21</td><td>23</td><td>29</td><td>31</td></tr></table>

Remark 2.3.8.5. Note the diference between the previous two examples. The tiling of A in each of the two examples is identical, but the layout of each tile is diferent. In first example, the tiles have column-major layouts, while in the second example, the tiles have row-major layouts. This results in diferent layouts when one performs logical division. 

Example 2.3.8.6. If $A = ( 4 , 8 ) : ( 1 , 4 )$ and $B = ( 2 , 4 ) : ( 2 , 4 )$ , then 

$$
A \oslash B = ((2, 4), (2, 2)): ((2, 4), (1, 1 6)).
$$

Example 2.3.8.7. If $A = ( 4 , 6 ) : ( 1 , 4 0 )$ and $B = 6 : 4$ , then 

$$
A \oslash B = (6, 4): (4 0, 1).
$$

Example 2.3.8.8. If $A = ( 4 , 6 , 2 , 4 , 2 , 5 )$ : (36, 1, 18, 0, 0, 144) and $B = ( 4 , 1 0 ) : ( 1 , 1 9 2 )$ , then 

$$
A \oslash B = (((4, (2, 5)), (6, 2, 4)): ((3 6, (0, 1 4 4)), (1, 1 8, 0))
$$

Example 2.3.8.9. If $A = ( 8 , ( 4 , 4 ) )$ and $B = ( 2 , ( 8 , 1 6 ) )$ , then 

$$
A \oslash B = ((2, 2), (2, (4, 4))): ((4, 8), (2, (8, 1 6))).
$$

## 2.3.9 Logical product

In this section, we define the logical product of layouts. 

Definition 2.3.9.1. Suppose A and B are layouts, and suppose 

$$
A ^ {c} = \operatorname{comp} (A, \text { size } (A) \cdot \text { cosize } (B))
$$

is the complement of A with respect to $\mathsf { s i z e } ( A )$ · cosize(B). Then the logical product of A and B is the layout 

$$
A \otimes B = (A, A ^ {c} \circ B).
$$

Observation 2.3.9.2. By Proposition 2.3.7.10 and Proposition 2.3.7.11, if we let 

$$
\widetilde {A} ^ {c} = \operatorname{comp} (A, N)
$$

for any valid $N \geq \mathsf { s i z e } ( A ) \cdot \mathsf { c o s i z e } ( B )$ , then 

$$
A ^ {c} \circ B = \tilde {A} ^ {c} \circ B.
$$

This means that when computing $A \otimes B ,$ we can take $A ^ { c }$ to be any suficiently large (sorted) complement of A. 

Example 2.3.9.3. If $A = ( 2 , 2 ) : ( 5 , 1 0 )$ and $B = ( 3 , 5 ) : ( 5 , 1 )$ are the layouts 

$$
A = \begin{array}{c c} \hline 0 & 1 0 \\ \hline 5 & 1 5 \\ \hline \end{array}
$$

<table><tr><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td></tr><tr><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td></tr><tr><td>10</td><td>11</td><td>12</td><td>13</td><td>14</td></tr></table>

then $A \otimes B$ is the layout 

$$
A \otimes B = ((2, 2), (3, 5)): ((5, 1 0), (2 0, 1))
$$

as depicted below. 

<table><tr><td>0</td><td>20</td><td>40</td><td>1</td><td>21</td><td>41</td><td>2</td><td>22</td><td>42</td><td>3</td><td>23</td><td>43</td><td>4</td><td>24</td><td>44</td></tr><tr><td>5</td><td>25</td><td>45</td><td>6</td><td>26</td><td>46</td><td>7</td><td>27</td><td>47</td><td>8</td><td>28</td><td>48</td><td>9</td><td>29</td><td>49</td></tr><tr><td>10</td><td>30</td><td>50</td><td>11</td><td>31</td><td>51</td><td>12</td><td>32</td><td>52</td><td>13</td><td>33</td><td>53</td><td>14</td><td>34</td><td>54</td></tr><tr><td>15</td><td>35</td><td>55</td><td>16</td><td>36</td><td>56</td><td>17</td><td>37</td><td>57</td><td>18</td><td>38</td><td>58</td><td>19</td><td>39</td><td>59</td></tr></table>

Example 2.3.9.4. If $A = ( 3 , 3 ) : ( 6 , 1 ) { \mathrm { ~ a n d ~ } } B = ( 1 0 , 1 2 ) : ( 2 4 , 2 ) , { \mathrm { t h e n } }$ 

$$
A \otimes B = ((3, 3), (1 0, 1 2)): ((6, 1), (2 1 6, 1 8)).
$$

Example 2.3.9.5. If A = (2, 10) : (1680, 4) and B = (4, 9) : (2, 56), then 

$$
A \otimes B = ((2, 1 0), ((2, 2), (3, 3))): ((1 6 8 0, 4), ((2, 4 0), (5 6 0, 3 3 6 0))).
$$

Example 2.3.9.6. If A = (4, (2, 2)) : (9, (1, 3)) and B = ((2, 4), 8) : ((1, 4), 2), then 

$$
A \otimes B = ((4, (2, 2)), ((2, 4), 8)): ((9, (1, 3)), ((3 6, 1 4 4), 7 2)).
$$

## 2.3.10 Tractable layouts

In this section we define an especially well-behaved class of layouts, called tractable layouts. We will see that tractable layouts are precisely the layouts which arise from a certain category Nest. 

Definition 2.3.10.1. We say a layout L is tractable if the flat layout $L ^ { \flat }$ is tractable, in the sense of Definition 2.1.8.1. Explicitly, L is tractable if the flat layout 

$$
\operatorname{sort} \left(L ^ {\flat}\right) = \left(s _ {1}, \dots , s _ {m}\right): \left(d _ {1}, \dots , d _ {m}\right)
$$

is such that for each $1 \leq i < m$ , we have 

1. $d _ { i } = 0 ,$ or 

2. $s _ { i } d _ { i }$ divides $d _ { i + 1 }$ 

Example 2.3.10.2. The layout 

$$
L = (((1 2))): ((1 7))
$$

is tractable. More generally, any layout L of length 1 is tractable. 

Example 2.3.10.3. The layout 

$$
L = ((2, 4), 3 2): ((1, 2), 8)
$$

is tractable. More generally, any column-major layout is tractable. 

Example 2.3.10.4. The layout 

$$
L = (2, (4, 3 2)): (1 2 8, (3 2, 1))
$$

is tractable. More generally, any row-major layout L is tractable. 

Example 2.3.10.5. The layout 

$$
L = ((3, 3), (1, 3), (3, 1, 3)): ((8 1, 1), (0, 8), (3, 0, 2 7))
$$

is tractable. More generally, any compact layout is tractable. 

Example 2.3.10.6. The layout 

$$
L = ((3, 7, 7)): ((0, 1 5, 0))
$$

is tractable. More generally, any layout with exactly one non-zero stride entry is tractable. 

Example 2.3.10.7. The layout 

$$
L = (2, (2, (2, 2))): (1, (2 0 4 8, (1 6, 6 4)))
$$

is tractable. More generally, any complementable layout is tractable. 

Example 2.3.10.8. The layout 

$$
L = ((8, 8), (5, 5)): ((8, 1), (1 0, 2))
$$

is not tractable. In particular, this shows that the concatenation $( L _ { 1 } , L _ { 2 } )$ of tractable layouts $L _ { 1 }$ and $L _ { 2 }$ need not be tractable. 

## Chapter 3

# Categories of layouts

Having thoroughly explored the algebra of layouts, we now turn our attention to the mathematical heart of this work: realizing layouts as morphisms in suitably-defined categories. Along the way, we develop a graphical calculus of layout diagrams that afords more straightforward computation of layout operations. 

## 3.1 The category Tuple

In this section, we define a category Tuple whose objects are tuples of positive integers, and whose morphisms we call tuple morphisms. Each tuple morphism $f : S  T$ encodes a flat layout $L _ { f } .$ Composition of tuple morphisms is compatible with layout composition, in that if f and g are composable tuple morphisms, then 

$$
L _ {g \circ f} = L _ {g} \circ L _ {f}.
$$

We define a realization functor (Theorem 3.1.4.4) 

$$
| \cdot |: \text { Tuple } \to \text { FinSet }
$$

which recovers the layout function of $L _ { f }$ via the formula 

$$
| f | = \Phi_ {L _ {f}} ^ {\text { size } (T)}.
$$

We develop an “algebra of tuple morphisms” which includes operations such as sort (Section 3.1.5.3), coalesce (Section 3.1.5.4), complement (Section 3.1.5.6), concatenate (Section 3.1.5.5), flat division (Section 3.1.5.7), and flat products (Section 3.1.5.8), which are compatible with the corresponding operations on flat layouts. 

## 3.1.1 Basic definitions

Definition 3.1.1.1. Let Fin<sub>∗</sub> denote the category whose objects are the pointed finite sets 

$$
\langle m \rangle_ {*} = \{*, 1, 2, \ldots , m \}
$$

for $m \geq 0$ , and whose morphisms $\alpha : \langle m \rangle _ { * } \to \langle n \rangle$ <sub>∗</sub> are functions satisfying $\alpha ( * ) = *$ . We call these morphisms pointed maps, or simply maps. 

Aside 3.1.1.2. Fin<sub>∗</sub> is a skeleton of the category FinSet<sub>∗</sub> of finite pointed sets. 

Notation 3.1.1.3. If the codomain of a pointed map $\alpha : \langle m \rangle _ { * } \to \langle n \rangle$ <sub>∗</sub> is understood, we sometimes write 

$$
\alpha = (\alpha (1), \dots , \alpha (m))
$$

as a tuple of length m with entries in $\langle n \rangle$ ∗ 

Example 3.1.1.4. There is a morphism $\alpha : \langle 4 \rangle _ { * } \to \langle 6 \rangle$ * in Fin<sub>∗</sub> given by 

$$
\alpha = (2, 1, *, 6),
$$

which we can visualize using the following diagram. 

![image](Imgaes/categorical-foundations-cute-layouts-paper/1d932e5197de984f50f44364a441c51e8c01176ee6a20812587959cf1a948276.jpg)


Note that the bullet corresponding to entry 3 does not support an arrow, reflecting the fact that it gets sent to ∗. 

Example 3.1.1.5. There is a morphism $\beta : \langle 5 \rangle _ { * } \to \langle 3 \rangle$ ∗ in Fin<sub>∗</sub> given by 

$$
\beta = (*, 1, 2, 3, *) ,
$$

which we can visualize using the following diagram. 

![image](Imgaes/categorical-foundations-cute-layouts-paper/3d8222e56590b7b24aedccd4c39e86cbdf4bfaac12964a9f9932f0bb2c12467e.jpg)


Example 3.1.1.6. For any $m \geq 0$ , there is a unique morphism in Fin<sub>∗</sub> of the form $\pi : \langle { m } \rangle _ { * } \to \langle { 0 } \rangle { } _ { : }$ ∗ namely 

$$
\pi = (*, \dots , *).
$$

Example 3.1.1.7. For any $n \geq 0 .$ , there is a unique morphism in Fin<sub>∗</sub> of the form $\delta : \langle 0 \rangle _ { * } \to \langle m \rangle ,$ <sub>∗</sub>, namely 

$$
\delta = ().
$$

Aside 3.1.1.8. The category Fin<sub>∗</sub> is the category of operators for the commutative operad, so we sometimes write 

$$
\mathbf {F i n} _ {*} = \mathbf {C o m m} ^ {\otimes}.
$$

We are especially interested in tractable morphisms in $\mathsf { F i n } _ { * } .$ , which we define below. 

Definition 3.1.1.9. We say a pointed map $\alpha : \langle m \rangle _ { * } \to \langle n \rangle$ <sub>∗</sub> is tractable if for any $j \in \langle n \rangle \subset \langle n \rangle ,$ <sub>∗</sub>, the preimage $\alpha ^ { - 1 } ( j )$ is empty or consists of a single element. 

Example 3.1.1.10. The maps 

![image](Imgaes/categorical-foundations-cute-layouts-paper/9398fc32796fb475da78edf997e9c71a347748e7b38219e6a1b23134782a8f27.jpg)


are tractable, while the maps 

![image](Imgaes/categorical-foundations-cute-layouts-paper/aced812813ea3f1c753a16585b6b4601e2167dc9cbf0dcade1b6fd4688edc022.jpg)


are not tractable 

Remark 3.1.1.11. If we represent a morphism $\alpha : \langle m \rangle _ { * } \to \langle n \rangle$ <sub>∗</sub> in Fin<sub>∗</sub> as a tuple, i.e. 

$$
\alpha = (\alpha (1), \dots , \alpha (m))
$$

then α is tractable if and only if no positive integer occurs more than once in α. Aside 3.1.1.12. The wide subcategory 

$$
\mathbf {E} _ {0} ^ {\otimes} \subset \mathbf {C o m m} ^ {\otimes} = \mathbf {F i n} _ {*}
$$

on the tractable pointed maps is the category of operators for the $\mathsf { E } _ { 0 }$ operad. 

Definition 3.1.1.13. Let Tuple denote the category whose objects are tuples 

$$
S = (s _ {1}, \ldots , s _ {m})
$$

of positive integers, where a morphism 

$$
f: (s _ {1}, \dots , s _ {m}) \to (t _ {1}, \dots , t _ {n})
$$

is specified by a tractable pointed map $\alpha : \langle m \rangle _ { * } \to \langle n \rangle$ satisfying the property that 

${ \mathrm { i f ~ } } 1 \leq i \leq m$ and $\alpha ( i ) \neq *$ , then $s _ { i } = t _ { \alpha ( i ) }$ 

We say that such a morphism $f$ lies over α, and refer to $f$ as a tuple morphism. 

Notation 3.1.1.14. If $f : ( s _ { 1 } , \ldots , s _ { m } ) \to ( t _ { 1 } , \ldots , t _ { n } )$ is a tuple morphism which lies over $\alpha ,$ then we sometimes depict f as 

$$
(s _ {1}, \ldots , s _ {m}) \xrightarrow [ \alpha ]{f} (t _ {1}, \ldots , t _ {n}).
$$

The graphical calculus of layouts we develop is based on the natural visualizations of morphisms in Tuple, as exemplified below. 

Example 3.1.1.15. The tuple morphism 

$$
(3, 1 2 8, 1 2 8) \xrightarrow [ (1 , 3 , 5) ]{f} (3, 2, 1 2 8, 2, 1 2 8)
$$

can be visualized using the following diagram. 

![image](Imgaes/categorical-foundations-cute-layouts-paper/88914cb5525127454469911d9442cff03c6cb2459b991fb5f4596d5f5be2c551.jpg)


Example 3.1.1.16. The tuple morphism 

$$
(3, 1 2 8, 1 2 8) \xrightarrow [ (* , 2 , 1) ]{g} (1 2 8, 1 2 8)
$$

can be visualized using the following diagram. 

$$
\begin{array}{c} 1 2 8 \\ 1 2 8 \\ 3 \end{array} \xrightarrow {} \begin{array}{c} 1 2 8 \\ 1 2 8 \\ 1 2 8 \end{array}
$$

Example 3.1.1.17. The tuple morphism 

$$
(1 6, 1 6, 1 6, 1, 3 2) \xrightarrow [ (* , * , 1 , * , 2) ]{h} (1 6, 3 2, 1, 1)
$$

can be visualized using the following diagram. 

$$
\begin{array}{c} 3 2 \\ 1 \\ 1 6 \\ 1 6 \\ 1 6 \end{array} \begin{array}{c} 1 \\ 1 \\ 3 2 \\ 1 6 \end{array} h
$$

Observation 3.1.1.18. We can relate the category Tuple to some well-known operads as follows. Let $\mathbb { Z } _ { > 0 } ^ { \mathrm { d i v } }$ denote the poset of positive integers under the divisibility relation, considered as a symmetric monoidal category with product given by multiplication of integers. Let $( \mathbb { Z } _ { > 0 } ^ { \mathsf { d i v } } ) ^ { \otimes }$ denote the category of operators of $\mathbb { Z } _ { > 0 } ^ { \mathrm { d i v } }$ . Then there are evident functors 

$$
\mathbf {T u p l e} \rightarrow (\mathbb {Z} _ {> 0} ^ {\mathrm{div}}) ^ {\otimes},
$$

and 

$$
\mathbf {T u p l e} \to \mathbf {E} _ {0} ^ {\otimes},
$$

such that the diagram 

$$
\begin{array}{c} \text { Tuple } \longrightarrow (\mathbb {Z} _ {> 0} ^ {\mathrm{div}}) ^ {\otimes} \\ \Big \downarrow \qquad \qquad \qquad \qquad \qquad \qquad \Big \downarrow \\ E _ {0} ^ {\otimes} \longrightarrow \text { Comm } ^ {\otimes} \end{array}
$$

commutes. This exhibits Tuple as the wide subcategory of the pullback operad 

$$
\mathbf {T u p l e} \subset \mathbf {E} _ {0} ^ {\otimes} \times_ {\mathbf {C o m m} ^ {\otimes}} (\mathbb {Z} _ {> 0} ^ {\text { div }}) ^ {\otimes}
$$

on the morphisms 

$$
(s _ {1}, \ldots , s _ {m}) \xrightarrow [ \alpha ]{f} (t _ {1}, \ldots , t _ {n})
$$

satisfying 

$$
\alpha (i) \neq 1 \quad \Rightarrow \quad s _ {i} = t _ {\alpha (i)}.
$$

## 3.1.2 From tuple morphisms to flat layouts

The impetus for working with the category Tuple is that each tuple morphism f encodes a flat layout $L _ { f }$ . Moreover, each tractable layout L gives rise to a tuple morphism $f _ { L }$ . We prove as Theorem 3.1.2.10 that these constructions are in some sense inverses, and that tractable layouts are precisely those encoded by tuple morphisms. 

Construction 3.1.2.1. Suppose 

$$
(s _ {1}, \ldots , s _ {m}) \xrightarrow [ \alpha ]{f} (t _ {1}, \ldots , t _ {n})
$$

is a tuple morphism. We define $L _ { f }$ to be the flat layout whose shape 

$$
\operatorname{shape} \left(L _ {f}\right) = \left(s _ {1}, \dots , s _ {m}\right)
$$

is the domain of $f ,$ and whose stride 

$$
\operatorname{stride} \left(L _ {f}\right) = \left(d _ {1}, \dots , d _ {m}\right)
$$

is defined by the formula 

$$
d _ {i} = \left\{ \begin{array}{l l} 0 & \alpha (i) = * \\ \prod_ {j <   \alpha (i)} t _ {j} & \alpha (i) \neq *. \end{array} \right.
$$

We refer to $L _ { f }$ as the layout encoded by f or the layout associated to $f .$ 

Example 3.1.2.2. The tuple morphism 

![image](Imgaes/categorical-foundations-cute-layouts-paper/edd5f63b063a9d495f4bec1bf4bd779a308d682eca01695c6559e4cbab6f0e20.jpg)


of Example 3.1.1.15 encodes the layout 

$$
L _ {f} = (3, 1 2 8, 1 2 8): (1, 6, 1 5 3 6).
$$

Note that computing the stride via the formula in Theorem 3.1.2.1 amounts to following the arrow from a specific shape entry to its target entry and multiplying together all entries below that one (taking the empty product to equal 1). 

Example 3.1.2.3. The tuple morphism 

$$
(3, 1 2 8, 1 2 8) \xrightarrow [ (* , 2 , 1) ]{g} (1 2 8, 1 2 8)
$$

of Example 3.1.1.16 encodes the layout 

$$
L _ {g} = (3, 1 2 8, 1 2 8): (0, 1 2 8, 1).
$$

Example 3.1.2.4. The tuple morphism 

$$
(1 6, 1 6, 1 6, 1, 3 2) \xrightarrow [ (* , * , 1 , * , 2) ]{h} (1 6, 3 2, 1, 1)
$$

of Example 3.1.1.17 encodes the layout 

$$
L _ {h} = (1 6, 1 6, 1 6, 1, 3 2): (0, 0, 1, 0, 1 6).
$$

We have seen how to compute the flat layout $L _ { f }$ encoded by a tuple morphism $f .$ On the other hand, if $L$ is tractable, then we can go in the other direction, constructing a tuple morphism $f$ which encodes L. Recall from Definition 2.1.8.1 that a flat layout L is tractable if 

$$
\operatorname{sort} (L) = \left(s _ {1}, \dots , s _ {m}\right): \left(d _ {1}, \dots , d _ {m}\right)
$$

satisfies the following property: 

$$
\text {   If   } 1 \leq i <   m, \text {   then   } d _ {i} = 0, \text {   or   } s _ {i} d _ {i} \text {   divides   } d _ {i + 1}.
$$

Construction 3.1.2.5. Suppose $L = ( s _ { 1 } , \ldots , s _ { m } ) : ( d _ { 1 } , \ldots , d _ { m } )$ is tractable, and set 

$$
\operatorname{sort} (L) = \left(s _ {1} ^ {\prime}, \dots , s _ {m} ^ {\prime}\right): \left(d _ {1} ^ {\prime}, \dots , d _ {m} ^ {\prime}\right),
$$

so there is some permutation $\sigma \in \Sigma _ { m }$ such that sor $\boldsymbol { \mathbf { \ell } } \cdot ( L ) = L ^ { \sigma }$ . In other words, $s _ { i } ^ { \prime } = s _ { \sigma ( i ) }$ and $d _ { i } ^ { \prime } = d _ { \sigma ( i ) }$ for each $1 \leq i \leq m$ . If each $d _ { i } ^ { \prime }$ is nonzero, then let $k = 0$ . Otherwise, let k be the largest integer such that $d _ { k } ^ { \prime } = 0$ . Let $\ell = 2 ( m - k )$ , and let 

$$
(t _ {1} ^ {\prime}, \ldots , t _ {\ell} ^ {\prime}) = \left(d _ {k + 1} ^ {\prime}, s _ {k + 1} ^ {\prime}, \frac {d _ {k + 2} ^ {\prime}}{s _ {k + 1} ^ {\prime} d _ {k + 1} ^ {\prime}}, s _ {k + 2} ^ {\prime}, \frac {d _ {k + 3} ^ {\prime}}{s _ {k + 2} ^ {\prime} d _ {k + 2} ^ {\prime}}, \ldots , \frac {d _ {m} ^ {\prime}}{s _ {m - 1} ^ {\prime} d _ {m - 1} ^ {\prime}}, s _ {m} ^ {\prime}\right).
$$

We define 

$$
f _ {L} ^ {\prime}: (s _ {1}, \ldots , s _ {m}) \to (t _ {1} ^ {\prime}, \ldots , t _ {\ell} ^ {\prime})
$$

to be the tuple morphism lying over the map $\alpha : \langle m \rangle _ { * } \to \langle \ell \rangle$ <sub>∗</sub> given by 

$$
\alpha^ {\prime} (i) = \left\{ \begin{array}{l l} * & \sigma^ {- 1} (i) \leq k \\ 2 (\sigma^ {- 1} (i) - k) & k + 1 \leq \sigma^ {- 1} (i) \leq m. \end{array} \right.
$$

Let $J = \{ j _ { 1 } < \cdots < j _ { n } \} \subset \langle \ell \rangle$ denote the collection of indices such that $j _ { i }$ is even or $t _ { j _ { i } } \neq 1$ . Let 

$$
(t _ {1}, \dots , t _ {n}) = (t _ {j _ {1}} ^ {\prime}, \dots , t _ {j _ {n}} ^ {\prime}),
$$

and let $\iota : \langle n \rangle _ { * } \to \langle \ell \rangle$ <sub>∗</sub> be the inclusion map $i \mapsto j _ { i }$ . Then by construction, the map $\alpha ^ { \prime }$ factors as $\alpha ^ { \prime } = \iota \circ \alpha$ , and we define the standard representation of $L$ to be the tuple morphism 

$$
(s _ {1}, \ldots , s _ {m}) \xrightarrow [ \alpha ]{f _ {L}} (t _ {1}, \ldots , t _ {n}).
$$

$$
L = (2, 2): (3, 3 0),
$$

then L is tractable, and the standard representation of L is the tuple morphism 

![image](Imgaes/categorical-foundations-cute-layouts-paper/eeb21ca167f0010e78bd086fdb3527cf25c9f4ead3d053ddd8ec4f1ab2ffbde6.jpg)


Note that, informally, computing $f _ { L }$ via Theorem 3.1.2.5 amounts to 

• initializing the codomain as $( )$ , 

• traversing the non-zero strides of L in increasing order, 

• if $d _ { j }$ is the current stride, and $d _ { i }$ is the previously visited stride, appending 

– (s<sub>j</sub> ) if s<sub>i</sub>d<sub>i</sub> = d<sub>j</sub> , or 

$$
\left(\frac {d _ {j}}{s _ {i} d _ {i}}, s _ {j}\right) \text {   if   } s _ {i} d _ {i} <   d _ {j},
$$

and 

• mapping $s _ { j } \mapsto s _ { j }$ 

Example 3.1.2.7. If 

$$
L = (1 2 8, 1 2 8): (1 2 8, 1),
$$

then $L$ is tractable, and the standard representation of $L$ is the tuple morphism 

$$
\begin{array}{c} 1 2 8 \\ 1 2 8 \end{array} \xrightarrow {} \begin{array}{c} 1 2 8 \\ 1 2 8 \end{array} f _ {L}
$$

Example 3.1.2.8. If 

$$
L = (2, 2, 2, 2): (2 4, 0, 3, 4 8 0),
$$

then $L$ is tractable, and the standard representation of L is the tuple morphism 

![image](Imgaes/categorical-foundations-cute-layouts-paper/314302fe5e5cc602aec203f347381b3cb9c1947434ce71e1695cd4190814a4bb.jpg)


Let’s justify that the tuple morphism $f _ { L }$ of Theorem 3.1.2.5 does, in fact, encode the layout $L .$ 

Lemma 3.1.2.9. Suppose L is a tractable flat layout, and $f = f _ { L }$ is the standard representation of L. Then the layout encoded by f is 

$$
L _ {f} = L.
$$

Proof. Suppose $L = ( s _ { 1 } , \ldots , s _ { m } ) : ( d _ { 1 } , \ldots , d _ { m } )$ is tractable, and let 

$$
(s _ {1}, \ldots , s _ {m}) \xrightarrow [ \alpha ]{f} (t _ {1}, \ldots , t _ {n})
$$

be the standard representation of L. Clearly 

$$
\operatorname{shape} \left(L _ {f}\right) = \left(s _ {1}, \dots , s _ {m}\right) = \operatorname{shape} (L).
$$

We need to check that stride $( L _ { f } ) = { \mathsf { s t r i d e } } ( L )$ . In other words, we need to check that for any $1 \leq i \leq m$ 2 we have 

$$
d _ {i} = \left\{ \begin{array}{l l} 0 & \alpha (i) = * \\ \prod_ {j <   \alpha (i)} t _ {j} & \alpha (i) \neq *. \end{array} \right.
$$

We borrow the notation of Theorem 3.1.2.5. If $\alpha ( i ) = *$ , then $\alpha ^ { \prime } ( i ) = *$ , and so $\sigma ^ { - 1 } ( i ) \leq k$ . This implies 

$$
d _ {i} = d _ {\sigma^ {- 1} (i)} ^ {\prime} = 0.
$$

Suppose otherwise that $\alpha ( i ) \neq *$ . Then $\alpha ^ { \prime } ( i ) \neq * ,$ and so $k + 1 \leq \sigma ^ { - 1 } ( i ) \leq m$ . We compute 

$$
\begin{array}{c}\prod_{j <   \alpha (i)}t_{j} = \prod_{\substack{j^{\prime} <   \alpha^{\prime}(i)\\ t_{j^{\prime}}^{\prime}\neq 1}}t_{j^{\prime}}^{\prime} = \prod_{j^{\prime} <   \alpha^{\prime}(i)}t_{j^{\prime}}^{\prime} = \prod_{j^{\prime} <   2(\sigma^{-1}(i) - k)}t_{j^{\prime}}^{\prime}\\ = d_{\sigma^{-1}(i)}^{\prime}\\ = d_{i}. \end{array}
$$

We have proved that if L is a tractable flat layout, then there exists a tuple morphism f which encodes L. Next, we prove the converse, which implies that tractable flat layouts are precisely the layouts encoded by tuple morphisms. 

Proposition 3.1.2.10. Suppose L is a flat layout. Then there exists a tuple morphism $f$ encoding L $i f$ and only $i f L$ is tractable. 

Proof. First, suppose L is a flat layout, and $f : ( s _ { 1 } , \ldots , s _ { m } ) \to ( t _ { 1 } , \ldots , t _ { n } )$ is a tuple morphism with $L _ { f } = L$ . We want to show that $L _ { f }$ is tractable. Let 

$$
\operatorname{sort} (L) = \left(s _ {1} ^ {\prime}, \dots , s _ {m} ^ {\prime}\right): \left(d _ {1}, \dots , d _ {m}\right)
$$

be the sorting of $L ,$ and suppose that $1 \leq i < m$ . We will argue that $d _ { i } = 0$ , or $s _ { i } ^ { \prime } d _ { i }$ divides $d _ { i + 1 }$ . If $d _ { i } = 0$ , then we are done. Suppose otherwise that $d _ { i } \neq 0$ . Then 

$$
d _ {i} = \prod_ {j <   k} t _ {j}
$$

for some $1 \leq k \leq n$ with $s _ { i } ^ { \prime } = t _ { k }$ . Since $d _ { i + 1 } \geq d _ { i }$ , we know that $d _ { i + 1 } \neq 0$ , so $d _ { i + 1 }$ has the form 

$$
d _ {i + 1} = \prod_ {j <   \ell} t _ {j}
$$

for some $1 \leq \ell \leq n$ . There are two cases to consider: 

• (Case 1) If $\ell > k ,$ , then 

$$
d _ {i + 1} = \prod_ {j <   \ell} t _ {j} = \left(\prod_ {j \leq k} t _ {j}\right) \left(\prod_ {k <   j <   \ell} t _ {j}\right) = s _ {i} ^ {\prime} d _ {i} \left(\prod_ {k <   j <   \ell} t _ {j}\right),
$$

so $s _ { i } ^ { \prime } d _ { i }$ divides $d _ { i + 1 }$ 

• (Case 2) If $\ell \leq k ,$ then since 

$$
\prod_ {j <   \ell} t _ {j} = d _ {i + 1} \geq d _ {i} = \prod_ {j <   k} t _ {j},
$$

we must have 

$$
t _ {\ell} = \dots = t _ {k - 1} = 1,
$$

and 

$$
d _ {i + 1} = d _ {i}.
$$

In particular, we have $s _ { i + 1 } ^ { \prime } = t _ { \ell } = 1$ . But since $\mathsf { s o r t } ( L _ { f } )$ is sorted and $d _ { i + 1 } = d _ { i }$ , we have $s _ { i } ^ { \prime } \leq s _ { i + 1 } ^ { \prime } = 1$ , so $s _ { i } ^ { \prime } = 1$ . We deduce that 

$$
s _ {i} ^ {\prime} d _ {i} = d _ {i + 1},
$$

so in particular, $s _ { i } ^ { \prime } d _ { i }$ divides $d _ { i + 1 }$ 

We conclude that L is tractable. 

Next, suppose that L is tractable. Then we can take $f = f _ { L }$ to be the standard representation of $L$ (see Construction 3.1.2.5), in which case, by Lemma 3.1.2.9, we have $L = L _ { f }$ □ 

Remark 3.1.2.11. It is important to note that there are many diferent tuple morphisms which give rise to the same layout. For example, each of the tuple morphisms shown below 

![image](Imgaes/categorical-foundations-cute-layouts-paper/a92bb77975979d7562e9ea3f6d5357c37198cf733f0c8ca2ca97437f19394b94.jpg)


encodes the layout 

$$
L _ {f} = L _ {g} = L _ {h} = (4, 4, 4): (1 4, 5 6, 5 6 0 0).
$$

Among these, $f$ is the simplest: There are no extraneous entries lying above the image of f (unlike $g )$ , and the entries not hit by $f$ are condensed (unlike h). To make precise the simplicity of $f$ among these morphisms, we introduce the notion of standard form. 

Definition 3.1.2.12. Suppose 

$$
(s _ {1}, \ldots , s _ {m}) \xrightarrow [ \alpha ]{f} (t _ {1}, \ldots , t _ {n})
$$

is a tuple morphism. We say f has standard form if the following conditions hold: 

1. $\mathrm { I f } \ n > 1$ , then $n \in { \mathsf { I m a g e } } ( \alpha )$ 

2. If $1 \leq j < n$ , then 

$$
j \notin \operatorname{Image} (\alpha) \quad \Rightarrow \quad \begin{array}{c} t _ {j} \neq 1, \text {   and   } \\ j + 1 \in \operatorname{Image} (\alpha) \end{array}
$$

Example 3.1.2.13. The tuple morphisms f of Remark 3.1.2.11 has standard form, while g and h do not. 

Example 3.1.2.14. The tuple morphisms 

![image](Imgaes/categorical-foundations-cute-layouts-paper/b2f29e6a84fc1aff2c1e4b314ec81e745dafe56b153845100c0af8bdccedce4f.jpg)


have standard form, while the tuple morphisms 

![image](Imgaes/categorical-foundations-cute-layouts-paper/26f697f8ef845f294e88f8d0c7256cac88290eeedadfc5a2f8686ba22151e045.jpg)


do not. 

Example 3.1.2.15. If L is a tractable layout, then by construction, the standard representation $f _ { L }$ of L has standard form. 

If we restrict to tuple morphisms of standard form, then there is almost a one-to-one correspondence with tractable layouts. However, there is one problematic case we need to exclude, as explicated in the following example. 

Example 3.1.2.16. Consider the tuple morphisms f and g shown below. 

![image](Imgaes/categorical-foundations-cute-layouts-paper/38bcd5b34ee35a5babce05ae699b69605e029da991131507e2699d00b815af14.jpg)


Both $f$ and g have standard form, and 

$$
L _ {f} = (8, 1, 1): (1, 8, 8) = L _ {g}.
$$

This example illustrates that the presence of entries of the form $s _ { i } = 1$ and $\alpha ( i ) \neq *$ can lead to non-uniqueness of a representing tuple morphism of standard form. On the layout side, this corresponds to shape entries $s _ { i } = 1$ with stride $d _ { i } \neq 0$ . In order to exclude such pathological examples, we introduce the notion of non-degeneracy. 

Definition 3.1.2.17. Suppose 

$$
(s _ {1}, \ldots , s _ {m}) \xrightarrow [ \alpha ]{f} (t _ {1}, \ldots , t _ {n})
$$

is a tuple morphism and 

$$
L = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m})
$$

is a flat layout. 

1. We say f is non-degenerate if 

$$
s _ {i} = 1 \quad \Rightarrow \quad \alpha (i) = *.
$$

2. We say L is non-degenerate if 

$$
s _ {i} = 1 \quad \Rightarrow \quad d _ {i} = 0.
$$

Observation 3.1.2.18. If f is a non-degenerate tuple morphism, then the layout $L _ { f }$ encoded by $f$ is non-degenerate. Conversely, if $L$ is a non-degenerate flat layout, then the standard representation $f _ { L }$ of $L$ is non-degenerate. 

Observation 3.1.2.19. Restricting to non-degenerate flat layouts is no real loss of generality. If L is an arbitrary flat layout, then filter(L) is a non-degenerate flat layout with the same coordinate function and layout function as $L .$ 

The essential property of non-degenerate tuple morphisms of standard form is that they are characterized by the layouts which they encode. This is made precise as follows. 

Lemma 3.1.2.20. Suppose $f$ and g are non-degenerate tuple morphisms of standard form. $I f L _ { f } = L _ { g }$ 9 then $f = g$ 

Proof. Suppose 

$$
(s _ {1}, \ldots , s _ {m}) \xrightarrow [ \alpha ]{f} (t _ {1}, \ldots , t _ {n})
$$

and 

$$
\left(s _ {1}, \ldots , s _ {m}\right) \xrightarrow [ \beta ]{g} \left(u _ {1}, \ldots , u _ {p}\right)
$$

are non-degenerate tuple morphisms of standard form with 

$$
L _ {f} = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m}) = L _ {g}.
$$

We want to show that $f = g .$ . First, we will argue that $( t _ { 1 } , \ldots , t _ { n } ) = ( u _ { 1 } , \ldots , u _ { p } )$ . Let 

$$
\begin{array}{l} X = \left\{t _ {1} \dots t _ {j} \mid 1 \leq j \leq n \right\} \\ Y = \left\{u _ {1} \dots u _ {k} \mid 1 \leq k \leq p \right\} \end{array}
$$

denote the sets of prefix products of $\left( t _ { 1 } , \ldots , t _ { n } \right)$ and $( u _ { 1 } , \ldots , u _ { p } )$ , respectively. We claim $X = Y$ , since each of these sets is equal to 

$$
Z = \left\{d _ {i}, s _ {i} d _ {i} \mid 1 \leq i \leq m \text {   and   } d _ {i} \neq 0 \right\}.
$$

Lets argue that $X = Z .$ . Suppose $1 \leq j \leq n$ . If there exists some $i \in \langle m \rangle$ with $\alpha ( i ) = j ,$ then $t _ { 1 } \cdot \cdot \cdot t _ { j } = s _ { i } d _ { i }$ . On the other hand, if j is not in the image of $\alpha ,$ then since $f$ has standard form, there exists some $i \in \langle m \rangle$ such that $\alpha ( i ) = j + 1$ , in which case $t _ { 1 } \cdot \cdot \cdot t _ { j } = d _ { i }$ . This proves that $X \subseteq Z$ 

Conversely, if $1 \leq i \leq m$ and $d _ { i } \neq 0 .$ , then $d _ { i } = t _ { 1 } \cdot \cdot \cdot t _ { \alpha ( i ) - 1 }$ and $s _ { i } d _ { i } = t _ { 1 } \cdot \cdot \cdot t _ { \alpha ( i ) }$ , which proves $Z \subseteq X$ We deduce that $X = Z$ . The same argument proves $Y = Z$ 

Since f and g are non-degenerate of standard form, we know that each $t _ { j }$ and each $u _ { k }$ is greater than 1, which implies 

$$
\begin{array}{c} t _ {1} <   t _ {1} t _ {2} <   \dots <   t _ {1} \dots t _ {n}, \\ u _ {1} <   u _ {1} u _ {2} <   \dots <   u _ {1} \dots u _ {p}, \end{array}
$$

and since $X = Y$ , it follows that $n = p ,$ and $t _ { 1 } \cdot \cdot \cdot t _ { j } = u _ { 1 } \cdot \cdot \cdot u _ { j }$ for each $1 \leq j \leq n$ . We deduce that $( t _ { 1 } , \ldots , t _ { n } ) = ( u _ { 1 } , \ldots , u _ { p } )$ 

Next, we need to argue that $\alpha = \beta$ . Suppose for contradiction that there exists some $i \in \langle m \rangle$ with $\alpha ( i ) \neq \beta ( i )$ . There are two cases to consider. 

• If $\alpha ( i ) = * \neq \beta ( i )$ , then 

$$
0 = d _ {i} = t _ {1} \dots t _ {\beta (i) - 1},
$$

a contradiction. The case $\alpha ( i ) \neq * = \beta ( i )$ is analogous. 

• If $\alpha ( i ) \neq * \neq \beta ( i )$ , then without loss of generality we may assume $\alpha ( i ) < \beta ( j )$ , in which case 

$$
d _ {i} = t _ {1} \dots t _ {\alpha (i) - 1} <   t _ {1} \dots t _ {\beta (i) - 1} = d _ {i},
$$

a contradiction. 

We deduce that $\alpha = \beta ,$ so $f = g$ 

We are now ready to prove our correspondence theorem, which identifies non-degenerate tuple morphisms of standard form with non-degenerate tractable flat layouts. 

Theorem 3.1.2.21. The maps 

![image](Imgaes/categorical-foundations-cute-layouts-paper/cb27ef7256faf20a6511162738e53c87a9371bb307ef00382e58e652d5c915b3.jpg)


![image](Imgaes/categorical-foundations-cute-layouts-paper/8363bd889d8770d8444df2fc42acbf1f5c828f27072b413e1fb882f89e9809eb.jpg)


![image](Imgaes/categorical-foundations-cute-layouts-paper/2d694e90302cff836f2269cc4c344e49b99229762880ce5d6254ca4f694f4452.jpg)


of Constructions 3.1.2.1 and 3.1.2.5 determine a one-to-one correspondence between non-degenerate tuple morphisms of standard form, and non-degenerate tractable flat layouts. 

Proof. We want to show that the constructions $f \mapsto L _ { f }$ and $L \mapsto f _ { L }$ are inverses, when restricted to tuple morphisms and layouts of the stated form. If L is a non-degenerate tractable flat layout, then by Lemma 3.1.2.9 we have $L _ { f _ { L } } = L$ . Suppose next that f is a non-degenerate tuple morphism of standard form and $L = L _ { f }$ is the layout encoded by $f .$ Since $f$ and $f _ { L _ { f } }$ are non-degenerate tuple morphisms of standard form, and the layouts encoded by these tuple morphsims are equal, it follows from Lemma 3.1.2.20 that $f = f _ { L _ { f } }$ □ 

## 3.1.3 Examples

In this section, we introduce some important families of tuple morphisms, and describe the flat layouts to which they give rise. 

Example 3.1.3.1 (Identity morphisms). We say a tuple morphism $f$ is an identity morphism if $f = \operatorname { i d } _ { S }$ for some tuple S. If $f = \operatorname { i d } _ { S }$ is an identity morphism, then $L _ { f }$ is the column-major layout with shape S. For instance, here is an example of an identity morphism f together with its associated layout $L _ { f }$ 

$$
\begin{array}{c c c}4 \longmapsto 4\\4 \longmapsto 4\\2 \longmapsto 2\\2 \longmapsto 2\\2 \longmapsto 2\\f\end{array}\qquad \rightsquigarrow \qquad L _ {f} = (2, 2, 2, 4, 4): (1, 2, 4, 8, 3 2)
$$

Example 3.1.3.2 (Isomorphisms). A tuple morphism $f : S  T$ is an isomorphism if there is a tuple morphism $g : T  S$ such that $g \circ f = \mathsf { i d } _ { S }$ and $f \circ g = \mathsf { i d } _ { T }$ . If f is an isomorphism, then its associated layout $L _ { f }$ is compact. For instance, here is an isomorphism $f$ together with its associated layout $L _ { f }$ 

$$
\begin{array}{c c}4&2\\4&4\\2&4\\2&2\\2&2\\f\end{array}\qquad \rightsquigarrow \qquad L _ {f} = (2, 2, 2, 4, 4): (2, 1, 6 4, 4, 1 6)
$$

Observation 3.1.3.3. Note that if a tuple morphism 

$$
(s _ {1}, \ldots , s _ {m}) \xrightarrow [ \alpha ]{f} (t _ {1}, \ldots , t _ {n})
$$

is an isomorphism, then $\alpha : \langle m \rangle _ { * } \to \langle m \rangle ,$ <sub>∗</sub> is a bijection, and so α $| _ { \langle m \rangle } \in \ \Sigma _ { m }$ is a permutation. Conversely, if $\sigma \in \Sigma _ { m }$ is a permutation, and $\left( s _ { 1 } , \ldots , s _ { m } \right)$ is a tuple of positive integers, then we may construct the isomorphism 

$$
\left(s _ {1}, \ldots , s _ {m}\right) \xrightarrow [ \sigma_ {*} ]{f} \left(s _ {\sigma (1)}, \ldots , s _ {\sigma (m)}\right).
$$

We conclude that there is a one-to-one correspondence between tuple isomorphisms $f$ with domain $\left( s _ { 1 } , \ldots , s _ { m } \right)$ , and permutations in $\Sigma _ { m }$ 

Example 3.1.3.4 (Projections). Suppose $S = ( s _ { 1 } , \ldots , s _ { m } ) $ is a shape, and suppose 

$$
\left\{i _ {1} <   \dots <   i _ {r} \right\} \subset \langle m \rangle
$$

is some subset. Let 

$$
\left(s _ {1}, \ldots , s _ {m}\right) \xrightarrow [ \alpha ]{p} \left(s _ {i _ {1}}, \ldots , s _ {i _ {r}}\right)
$$

be the tuple morphism lying over the map α with 

$$
\alpha (x) = \left\{ \begin{array}{l l} j & x = i _ {j} \\ * & \text { else. } \end{array} \right.
$$

We call $p$ the projection of $\left( s _ { 1 } , \ldots , s _ { m } \right)$ onto $( s _ { i _ { 1 } } , \ldots , s _ { i _ { r } } )$ . The layout encoded by $p$ is 

$$
L _ {p} = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m}),
$$

where 

$$
d _ {i} = \left\{ \begin{array}{l l} s _ {i _ {1}} \dots s _ {i _ {j - 1}} & i = i _ {j} \text {   for   some   } 1 \leq j \leq r \\ 0 & \text { otherwise. } \end{array} \right.
$$

For instance, here is a projection p of (64, 64, 3, 8) onto (64, 3), together with its associated layout. 

$$
\begin{array}{c c c}8&\\3&\\6 4&\longrightarrow&3\\6 4&\longmapsto&6 4\\&p\end{array}\qquad \rightsquigarrow \qquad L _ {p} = (6 4, 6 4, 3, 8): (1, 0, 6 4, 0)
$$

Example 3.1.3.5 (Dilations). Suppose $\boldsymbol { S } = \left( s _ { 1 } , \ldots , s _ { m } \right)$ is a shape, and suppose $c _ { 1 } , \ldots , c _ { m }$ are positive integers. The tuple morphism 

$$
\left(s _ {1}, \ldots , s _ {m}\right) \xrightarrow [ (* , 2 , * , 4 , \ldots , * , 2 m) ]{f} \left(c _ {1}, s _ {1}, \ldots , c _ {m}, s _ {m}\right)
$$

is called the dilation of $\left( s _ { 1 } , \ldots , s _ { m } \right) \ b y \ \left( c _ { 1 } , \ldots , c _ { m } \right)$ . The layout $L _ { f }$ associated to this morphism is $L _ { f } = ( s _ { 1 } , \ldots , s _ { m } ) : ( d _ { 1 } , \ldots , d _ { m } )$ , where 

$$
d _ {i} = \prod_ {j <   i} c _ {j} s _ {j}.
$$

For instance, here is the dilation f of (512, 512) by (2, 4), together with its associated layout. 

$$
\begin{array}{c c c}&5 1 2\\&\nearrow&4\\5 1 2&5 1 2\\5 1 2&\nearrow&2\\&f\end{array}\qquad \rightsquigarrow \qquad L _ {f} = (5 1 2, 5 1 2): (2, 4 0 9 6)
$$

Example 3.1.3.6 (Expansions). Suppose $S = ( s _ { 1 } , \ldots , s _ { m } ) $ is a tuple of positive integers, and suppose $1 \leq i \leq m ^ { \prime }$ , so that $S ^ { \prime } = ( s _ { 1 } , \ldots , s _ { m ^ { \prime } } )$ divides S. Then the tuple morphism 

$$
\left(s _ {1}, \ldots , s _ {m ^ {\prime}}\right) \xrightarrow [ (1 , 2 , \ldots , m ^ {\prime}) ]{e} \left(s _ {1}, \ldots , s _ {m ^ {\prime}}, \ldots , s _ {m}\right)
$$

is called the expansion of $S ^ { \prime } \ t o \ S$ . The layout encoded by e is the column-major layout with shape $\left( s _ { 1 } , \ldots , s _ { m ^ { \prime } } \right)$ . For instance, here is the expansion of $S ^ { \prime } = ( 4 , 4 ) \ \mathrm { t o } \ S = ( 4 , 4 , 8 , 8 )$ 

$$
\begin{array}{c c c}8&\\8&\\4 \longmapsto 4&\rightsquigarrow&L _ {e} = (4, 4): (1, 4)\\4 \longmapsto 4&\\e&\end{array}
$$

An important property of expansions is that if $f : S  T$ is any tuple morphism and $e : T  T ^ { \prime }$ is an expansion, then 

$$
L _ {e \circ f} = L _ {f}.
$$

In other words, post-composing $f$ with an expansion does not change the layout encoded by $f .$ 

Example 3.1.3.7 (Restrictions). Suppose 

$$
(s _ {1}, \dots , s _ {m}) \xrightarrow [ \alpha ]{f} (t _ {1}, \dots , t _ {n})
$$

is a tuple morphism, and suppose 

$$
I = \left\{i _ {1} <   \dots <   i _ {r} \right\} \subset \langle m \rangle
$$

is a subset of indices. Then the tuple morphism 

$$
(s _ {i _ {1}}, \ldots , s _ {i _ {r}}) \xrightarrow [ \alpha \circ \iota ]{f | _ {I}} (t _ {1}, \ldots , t _ {n})
$$

is called the restriction of f to I. If the layout encoded by $f$ is 

$$
L _ {f} = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m}),
$$

then the layout encoded by $f \mid _ { I }$ is 

$$
L _ {f | _ {I}} = (s _ {i _ {1}}, \dots , s _ {i _ {r}}): (d _ {i _ {1}}, \dots , d _ {i _ {r}}).
$$

For instance, here is the restriction $f \mid _ { I }$ of a tuple morphism $f ,$ where $I = \{ 2 , 4 \}$ 

$$
\begin{array}{c c}4&\\8&\rightarrow 4\\1 6&\rightarrow 1 6\\2&\rightarrow 8\\f&\end{array}\qquad \rightsquigarrow \qquad L _ {f} = (2, 1 6, 8, 4): (0, 8, 1, 1 2 8)
$$

$$
\begin{array}{c c}4&\rightarrow 4\\1 6&\rightarrow 1 6\\f | _ {I}&\sim\end{array}\qquad \qquad L _ {f | _ {I}} = (1 6, 4): (8, 1 2 8)
$$

Example 3.1.3.8 (Entry inclusions). An important special case of the previous construction is as follows. If $f : ( s _ { 1 } , \ldots , s _ { m } )  ( t _ { 1 } , \ldots , t _ { m } )$ is a tuple morphism and $1 \leq i \leq m$ , then the ith entry $f _ { i }$ of $f$ is 

$$
(s _ {i}) \xrightarrow [ (i) ]{f _ {i}} (t _ {1}, \ldots , t _ {n})
$$

If the layout encoded by $f$ is 

$$
L _ {f} = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m}),
$$

then the layout encoded by $f _ { i }$ is 

$$
L _ {f _ {i}} = (s _ {i}): (d _ {i}).
$$

For instance, here is a tuple morphism $f ,$ and its fourth entry $f _ { 4 }$ . 

![image](Imgaes/categorical-foundations-cute-layouts-paper/89d60f9cfafe6b3e3578f8f76806cc3e3f8aa42ba5c7b4bf967498eab4542bbc.jpg)


$$
\begin{array}{c c c}&4\\4&1 6\\&8\end{array}\quad \rightsquigarrow \quad L _ {f _ {4}} = (4): (1 2 8)
$$

Remark 3.1.3.9. Given an $\langle n \rangle _ { * } \in \mathsf { F i n } _ { * }$ , there is a morphism $\varphi _ { i } : \langle 1 \rangle _ { * } \to \langle n \rangle$ <sub>∗</sub> for each $i \in \langle n \rangle$ sending $* \mapsto *$ and $1 \mapsto i .$ . For a tuple morphism $f : ( s _ { 1 } , \ldots , s _ { m } ) \to ( t _ { 1 } , \ldots , t _ { n } )$ lying over $\alpha : \langle m \rangle _ { * } \to \langle n \rangle { } _ { : }$ <sub>∗</sub>, the i-th entry lies over the composite $\alpha \circ \varphi _ { i } : \langle 1 \rangle _ { * } \to \langle n \rangle .$ <sub>∗</sub>. 

Example 3.1.3.10 (Factorizations). Suppose 

$$
(s _ {1}, \ldots , s _ {m}) \xrightarrow [ \alpha ]{f} (t _ {1}, \ldots , t _ {n})
$$

is a tuple morphism, and suppose 

$$
J = \left\{j _ {1} <   \dots <   j _ {\ell} \right\} \subset \langle n \rangle
$$

is a subset such that Im $\mathsf { a g e } ( \alpha ) \subseteq J \cup \{ * \}$ . If we write $\iota : \langle \ell \rangle _ { * } \to \langle n \rangle$ <sub>∗</sub> for the map k 7→ j<sub>k</sub>, then α factors as $\alpha = \iota \circ \bar { \alpha }$ for a unique map $\bar { \alpha } : \langle m \rangle _ { * } \to \langle \ell \rangle _ { * } ,$ , and we define the factorization of f through J to be the tuple morphism 

$$
(s _ {1}, \ldots , s _ {m}) \xrightarrow [ \bar {\alpha} ]{f | ^ {J}} (t _ {j _ {1}}, \ldots , t _ {j _ {\ell}}).
$$

If the layout encoded by $f$ is 

$$
L _ {f} = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m}),
$$

then the layout encoded by $f \mid ^ { J }$ is 

$$
L _ {f | ^ {J}} = (s _ {1}, \ldots , s _ {m}): (d _ {1} ^ {\prime}, \ldots , d _ {m} ^ {\prime}),
$$

where 

$$
d _ {i} ^ {\prime} = \frac {d _ {i}}{\left(\prod_ {k <   \alpha (i) \text { and } k \notin J} t _ {j}\right)}.
$$

For instance, here is the factorization $f \mid ^ { J }$ of a tuple morphism $f ,$ where $J = \{ 2 , 4 , 5 \}$ 

![image](Imgaes/categorical-foundations-cute-layouts-paper/10c7cbca3c76f69e661144fa79637d1745ef824ea6766e4c358705d881ecbdb5.jpg)


$$
\begin{array}{c c}1 0&\\8 \xrightarrow {} 8&\rightsquigarrow\\8 \xrightarrow {} 8&\\f | ^ {J}&\end{array}\qquad L _ {f | ^ {J}} = (8, 8): (8, 1)
$$

Remark 3.1.3.11. There is a categorical interpretation of factorizations. Borrowing the notation of Example 3.1.3.10, we may observe that there is a tuple morphism $i : ( t _ { j _ { 1 } } , \dots , t _ { j _ { \ell } } ) \to ( t _ { 1 } , \dots , t _ { n } )$ lying over $\iota ,$ and $f \mid ^ { J }$ is the pullback of f along i: 

$$
\begin{array}{c} (s _ {1}, \ldots , s _ {m}) \xrightarrow {f | ^ {J}} (t _ {j _ {1}}, \ldots , t _ {j _ {\ell}}) \\ \mathrm{id} \Biggl \downarrow \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \Biggl \downarrow i \\ (s _ {1}, \ldots , s _ {m}) \xrightarrow [ f ]{} (t _ {1}, \ldots , t _ {n}) \end{array}
$$

## 3.1.4 Realization of tuple morphisms

As we have seen, a tuple morphism $f : S  T$ encodes a flat layout $L _ { f }$ . In this section, we will construct a realization functor 

$$
| \cdot |: \text { Tuple } \to \text { FinSet. }
$$

which makes this encoding explicit. The realization functor $| \cdot |$ sends a tuple morphism $f$ to the layout function $| f |$ of $L _ { f }$ . In order to construct our realization functor | · |, we first construct an auxiliary functor 

$$
F: \mathbf {T u p l e} \rightarrow \mathbf {F i n S e t}
$$

which we will use in our construction. 

Construction 3.1.4.1. We define a functor 

$$
F: \text { Tuple } \to \text { FinSet }
$$

as follows. 

• For an object $S = ( s _ { 1 } , \ldots , s _ { m } ) $ in Tuple, we define 

$$
F S = [ 0, S) = \prod_ {i = 1} ^ {m} [ 0, s _ {i}).
$$

• For a morphism $f : ( s _ { 1 } , \ldots , s _ { m } ) \to ( t _ { 1 } , \ldots , t _ { n } )$ in Tuple lying over $\alpha ,$ , we define $F f$ to be the map 

$$
[ 0, S) \xrightarrow {F f} [ 0, T)
$$

given by 

$$
(F f) (x _ {1}, \ldots , x _ {m}) = (y _ {1}, \ldots , y _ {n})
$$

where 

$$
y _ {j} = \left\{ \begin{array}{l l} x _ {i} & \text { there   exists } 1 \leq i \leq m \text { with } \alpha (i) = j, \\ 0 & \text { else }. \end{array} \right.
$$

One may easily verify that $F ( g \circ f ) = F g \circ F f$ and $F \mathsf { i d } _ { S } = \mathsf { i d } _ { F S }$ , so F is in fact a functor. 

Example 3.1.4.2. Suppose $f : ( 4 , 4 )  ( 4 , 4 , 4 )$ is the tuple morphism lying over $\alpha = ( 1 , 3 )$ . Then 

$$
F f: [ 0, (4, 4)) \to [ 0, (4, 4, 4))
$$

is given by 

$$
(F f) (x _ {1}, x _ {2}) = (x _ {1}, 0, x _ {2}).
$$

Example 3.1.4.3. Suppose $g : ( 3 , 2 5 6 , 2 5 6 , 5 1 2 )  ( 3 , 2 5 6 , 2 5 6 )$ is the tuple morphism lying over $\beta = ( * , 3 , 2 , * )$ . Then 

$$
F g: [ 0, (3, 2 5 6, 2 5 6, 5 1 2)) \to [ 0, (3, 2 5 6, 2 5 6))
$$

is given by 

$$
(F g) (x _ {1}, x _ {2}, x _ {3}, x _ {4}) = (0, x _ {3}, x _ {2}).
$$

Construction 3.1.4.4. We define a functor 

$$
| \cdot |: \text { Tuple } \to \text { FinSet }
$$

as follows. 

• For an object $S = ( s _ { 1 } , \ldots , s _ { m } ) $ in Tuple, we define 

$$
| S | = [ 0, \operatorname{size} (S)) = \{0, 1, \dots , \operatorname{size} (S) - 1 \}.
$$

• For a tuple morphism $f : S  T$ , we define 

$$
| f | = \operatorname{colex} _ {T} \circ F f \circ \operatorname{colex} _ {S} ^ {- 1}
$$

(recall Theorem 2.1.2.18). 

If $f : S  T$ and $g : T  U$ are composable tuple morphisms then 

$$
\begin{array}{r l} & {| g \circ f | = \mathsf {c o l e x} _ {U} \circ F (g \circ f) \circ \mathsf {c o l e x} _ {S} ^ {- 1}} \\ & {\qquad = \mathsf {c o l e x} _ {U} \circ F g \circ F f \circ \mathsf {c o l e x} _ {S} ^ {- 1}} \\ & {\qquad = \mathsf {c o l e x} _ {U} \circ F g \circ \mathsf {c o l e x} _ {T} ^ {- 1} \circ \mathsf {c o l e x} _ {T} \circ F f \circ \mathsf {c o l e x} _ {S} ^ {- 1}} \\ & {\qquad = | g | \circ | f |} \end{array}
$$

and if $f = \operatorname { i d } _ { S }$ is an identity morphism, then 

$$
\begin{array}{r l} | \mathsf {i d} _ {S} | & = \mathsf {c o l e x} _ {S} \circ F \mathsf {i d} _ {S} \circ \mathsf {c o l e x} _ {S} ^ {- 1} \\ & = \mathsf {c o l e x} _ {S} \circ \mathsf {i d} _ {S} \circ \mathsf {c o l e x} _ {S} ^ {- 1} \\ & = \mathsf {c o l e x} _ {S} \circ \mathsf {c o l e x} _ {S} ^ {- 1} \\ & = \mathsf {i d} _ {| S |}, \end{array}
$$

so $| \cdot |$ does in fact specify a functor. Next, we observe that for a morphism $f$ in Tuple, the map $| f |$ is the layout function of $L _ { f }$ . This allows us to easily deduce that composition of morphisms in Tuple is compatible with composition of flat layouts (see Corollary 3.1.4.6). 

Lemma 3.1.4.5. If $f : S  T$ is a tuple morphism, then the realization $| f |$ of f is the layout function of $L _ { f } .$ 

$$
| f | = \Phi_ {L _ {f}} ^ {\mathrm{size} (T)}
$$

Proof. Let $S = ( s _ { 1 } , \ldots , s _ { m } ) , T = ( t _ { 1 } , \ldots , t _ { n } )$ , and let 

$$
L _ {f} = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m})
$$

denote the layout associated to $f ,$ whose strides $d _ { i }$ are defined by the formula 

$$
d _ {i} = \left\{ \begin{array}{l l} 0 & \alpha (i) = * \\ \prod_ {j <   \alpha (i)} t _ {j} & \text {else.} \end{array} \right.
$$

By precomposing with colex<sub>S</sub> : $\Pi _ { i = 1 } ^ { m } [ 0 , s _ { i } ) \to [ 0 , \mathsf { s i z e } ( S ) )$ , it sufices to prove that for any $( x _ { 1 } , \ldots , x _ { m } ) \in$ $\textstyle \prod _ { i = 1 } ^ { m } [ 0 , s _ { i } )$ , we have 

$$
\left(\operatorname{colex} _ {T} \circ F f\right) \left(x _ {1}, \dots , x _ {m}\right) = \left(x _ {1}, \dots , x _ {m}\right) \cdot \left(d _ {1}, \dots , d _ {m}\right).
$$

For a general input $\begin{array} { r } { ( x _ { 1 } , \ldots , x _ { m } ) \in \prod _ { i = 1 } ^ { m } [ 0 , s _ { i } ) } \end{array}$ , we have 

$$
(F f) (x _ {1}, \dots , x _ {m}) = (y _ {1}, \dots , y _ {n})
$$

where $y _ { j }$ is equal to $x _ { i } { \mathrm { ~ i f ~ } } \alpha ( i ) = j$ , and 0 otherwise. It follows that 

$$
\begin{array}{l} (\mathsf {c o l e x} _ {T} \circ F f) (x _ {1}, \ldots , x _ {m}) = (y _ {1}, \dots , y _ {n}) \cdot (1, t _ {1}, \ldots , t _ {1} \dots t _ {n - 1}) \\ \qquad = \sum_ {j = 1} ^ {n} y _ {j} \cdot t _ {1} \dots t _ {j - 1} \\ \qquad = \sum_ {i = 1} ^ {m} x _ {i} d _ {i} \\ \qquad = (x _ {1}, \ldots , x _ {m}) \cdot (d _ {1}, \ldots , d _ {m}), \end{array}
$$

as desired. 

Corollary 3.1.4.6. If f and g are non-degenerate composable tuple morphisms, then 

$$
L _ {g \circ f} = L _ {g} \circ L _ {f}
$$

Proof. Suppose $f : S  T$ and $g : T  U$ are morphisms in Tuple lying over α and $\beta ,$ respectively. Write $S = ( s _ { 1 } , \ldots , s _ { m } ) $ and $T = ( t _ { 1 } , \ldots , t _ { n } ) $ . We need to check that 

1. shape $( L _ { g \circ f } )$ refines $\mathsf { s h a p e } ( L _ { f } )$ : This holds since the shape of $L _ { f }$ and $L _ { g \circ f }$ are both equal to $S .$ 

2. ${ \cal L } _ { g \circ f }$ is coalesced over shape $\left( L _ { f } \right)$ : This holds since the tuple morphism $g \circ f$ is non-degenerate, hence so is the layout ${ \cal L } _ { g \circ f }$ 

3. $\Phi _ { L _ { g \circ f } } = \Phi _ { L _ { g } } \circ \Phi _ { L _ { f } } ^ { \mathsf { s i z e } ( L _ { g } ) }$ : Using Lemma 3.1.4.5, we have 

$$
\begin{array}{c} \Phi_ {L _ {g} \circ f} ^ {\text {size} (U)} = | g \circ f | \\ = | g | \circ | f | \\ = \Phi_ {L _ {g}} ^ {\text {size} (U)} \circ \Phi_ {L _ {f}} ^ {\text {size} (T)}. \end{array}
$$

and by postcomposing with the inclusion $[ 0 , \mathsf { s i z e } ( U ) ) \subset \mathbb { Z }$ , and observing that $\mathsf { s i z e } ( T ) = \mathsf { s i z e } ( L _ { g } )$ ， the result follows. 

## 3.1.5 Operations on tuple morphisms

Our next goal is to develop an “algebra of tuple morphisms”, which includes operations such as coalesce, complement, composition, flat division, and flat products. We will prove that each of these operations is compatible with a corresponding operation on flat layouts. 

## 3.1.5.1 Sum

The sum $f \oplus g$ of tuple morphisms $f$ and $g$ is obtained by concatenating the domains and codomains of $f$ and $g .$ In order to define this operations precisely, we first define a corresponding operation on morphisms in $\mathsf { F i n } _ { * }$ 

Definition 3.1.5.1. Suppose $\alpha : \langle m \rangle _ { * } \to \langle n \rangle$ <sub>∗</sub> and $\beta : \langle p \rangle _ { * } \to \langle q \rangle$ are morphisms in $\mathsf { F i n } _ { * }$ . We define the sum of $\alpha$ and $\beta$ to be the morphism 

$$
\alpha \oplus \beta : \langle m + p \rangle_ {*} \rightarrow \langle n + q \rangle_ {*}
$$

given by 

$$
(\alpha \oplus \beta) (x) = \left\{ \begin{array}{l l} \alpha (x) & 1 \leq x \leq m \\ n + \beta (x - m) & m + 1 \leq x \leq m + p \\ * & x = *. \end{array} \right.
$$

This operation is associative, so we can consider the sum $\alpha _ { 1 } \oplus \cdot \cdot \cdot \oplus \alpha _ { k }$ for any finite collection of morphisms $\alpha _ { 1 } , \ldots , \alpha _ { k }$ in $\mathsf { F i n } _ { * }$ 

Remark 3.1.5.2. If α and $\beta$ are tractable pointed maps, then $\alpha \oplus \beta$ is tractable. 

Now we can define the sum of morphisms in Tuple. 

Definition 3.1.5.3. Suppose $f : S  T$ and $g : U \to V$ are tuple morphisms lying over α and $\beta ,$ respectively. We define the sum of $f$ and $g$ to be the tuple morphism 

$$
f \oplus g: S \star U \to T \star V
$$

lying over α ⊕ $\beta .$ This operation is associative, so we can consider the sum $f _ { 1 } \oplus \cdots \oplus f _ { k }$ for any finite collection of morphisms $f _ { 1 } , \ldots , f _ { k }$ in Tuple. 

Example 3.1.5.4. Here is an example of the sum $f \oplus g$ of tuple morphisms f and $g .$ 

$$
\begin{array}{c c c} 3 2 \longmapsto 3 2 & 4 & 4 \\ 1 6 \longmapsto 1 6 & 4 \xrightarrow {} 2 & 4 \\ f & g & f \oplus g \end{array}
$$

Example 3.1.5.5. Here is another example of the sum $f \oplus g$ of tuple morphisms f and $g$ 

$$
\begin{array}{c c c} f & g & f \oplus g \\ \hline \end{array}
$$

Remark 3.1.5.6. There is a categorical interpretation of the sum of tuple morphisms: if $f : S  T$ and $g : U \to V$ are tuple morphisms, then 

$$
f \oplus g: S \star U \to T \star V
$$

is the coproduct of f and $g$ in the arrow category Ar(Tuple). 

## 3.1.5.2 Squeeze

It is often the case that we want to remove any instances of the integer 1 from our tuples. This is accomplished by the squeeze functor. 

Definition 3.1.5.7. We define a functor 

$$
\text { Tuple } \xrightarrow {\text { squeeze } (-)} \text { Tuple }
$$

as follows. If $S = ( s _ { 1 } , \ldots , s _ { m } ) $ is an object in Tuple, we define 

$$
\operatorname{squeeze} (S) = \left(s _ {i _ {1}}, \dots , s _ {i _ {k}}\right)
$$

where $\{ i _ { 1 } < \dots < i _ { k } \} \subset \langle m \rangle$ are the indices with $s _ { i _ { j } } \neq 1$ . If $f : ( s _ { 1 } , \ldots , s _ { m } ) \to ( t _ { 1 } , \ldots , t _ { n } )$ is a tuple morphism, we define 

$$
\operatorname{squeeze} (f): \operatorname{squeeze} (S) \to \operatorname{squeeze} (T)
$$

to be the tuple morphism 

$$
\operatorname{squeeze} (f) = \left(f \mid_ {I}\right) | ^ {J}
$$

where $f \mid _ { I }$ is the restriction of $f$ to 

$$
I = \{i \in \langle m \rangle \mid s _ {i} \neq 1 \}
$$

as in Definition 3.1.3.7, and where $\left( f \mid _ { I } \right) \mid ^ { J }$ is be the factorization of $f \mid _ { I }$ through 

$$
J = \{j \in \langle n \rangle \mid t _ {j} \neq 1 \},
$$

as in Definition 3.1.3.10. 

Example 3.1.5.8. Here is an example of a morphism $f$ and the corresponding morphism squeeze $( f )$ 

![image](Imgaes/categorical-foundations-cute-layouts-paper/821c58baf4aff53588c7863f04be451fcc5c07f2bcca6dd32e17fd5502257b49.jpg)


Example 3.1.5.9. If $f : ( s _ { 1 } , \ldots , s _ { m } ) \to ( t _ { 1 } , \ldots , t _ { n } )$ is a tuple morphism, then 

$$
f = \operatorname{squeeze} (f) \quad \Leftrightarrow \quad \text {   no   } s _ {i}, t _ {j} \text {   is   equal   to   1.   }
$$

Proposition 3.1.5.10. If f is a tuple morphism, then 

$$
L _ {\text { squeeze } (f)} = \text { squeeze } (L _ {f}).
$$

Proof. Suppose $f : ( s _ { 1 } , \ldots , s _ { m } )  ( t _ { 1 } , \ldots , t _ { m } )$ is a tuple morphism, and let 

$$
L _ {f} = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m})
$$

be the flat layout associated to $f .$ Let $I = \{ i _ { 1 } < \cdots < i _ { m ^ { \prime } } \} \subset \langle m \rangle$ denote the subset of indices with $s _ { i _ { k } } \neq 1$ . Then 

$$
\begin{array}{l} L _ {f | _ {I}} = (s _ {i _ {1}}, \ldots , s _ {i _ {k}}): (d _ {i _ {1}}, \ldots , d _ {i _ {k}}) \\ = \text { squeeze } (L _ {f}). \end{array}
$$

Let $J = \{ j _ { 1 } < \cdots < j _ { n ^ { \prime } } \} \subset \langle n \rangle$ denote the subset of indices with $t _ { j _ { k } } \neq 1$ , so that squeeze $( f ) = ( f \mid _ { I } ) \mid ^ { J }$ Let $\beta$ denote the map over which squeeze(f) lies. Then 

$$
L _ {\text { squeeze } (f)} = L _ {(f | _ {I}) | ^ {J}} = (s _ {i _ {1}}, \ldots , s _ {i _ {k}}): (d _ {i _ {1}} ^ {\prime}, \ldots , d _ {i _ {k}} ^ {\prime})
$$

where 

$$
\begin{array}{c} d _ {i _ {k}} ^ {\prime} = \frac {d _ {i _ {k}}}{\left(\prod_ {\ell <   \beta (k) \text { and } \ell \notin J} t _ {\ell}\right)} \\ = d _ {i _ {k}} \end{array}
$$

since $t _ { \ell } = 1$ for any $\ell \not \in J$ . We conclude that 

$$
\begin{array}{c} L _ {\text { squeeze} (f)} = (s _ {i _ {1}}, \ldots , s _ {i _ {k}}): (d _ {i _ {1}}, \ldots , d _ {i _ {k}}) \\ = \text { squeeze } (L _ {f}). \end{array}
$$

Observation 3.1.5.11. If f is a tuple morphism, then 

$$
\operatorname{squeeze} (\operatorname{squeeze} (f)) = \operatorname{squeeze} (f),
$$

so 

$$
\text { Tuple } \xrightarrow {\text { squeeze } (-)} \text { Tuple }
$$

is an idempotent functor. 

## 3.1.5.3 Sort

The sort operation $f \mapsto \mathsf { s o r t } ( f )$ permutes the domain of f so that the resulting morphism is sorted, in the following sense. 

Definition 3.1.5.12. We say a tuple morphism 

$$
(s _ {1}, \ldots , s _ {m}) \xrightarrow [ \alpha ]{f} (t _ {1}, \ldots , t _ {n})
$$

is sorted if for any $1 \leq i , j \leq m$ , the following conditions hold. 

1. If $\alpha ( i ) = * \neq \alpha ( j )$ , then $i < j$ 

2. If $\alpha ( i ) = * = \alpha ( j )$ , then 

$$
i \leq j \quad \Rightarrow \quad s _ {i} \leq s _ {j}.
$$

3. If $\alpha ( i ) \neq * \neq \alpha ( j )$ , then 

$$
i \leq j \quad \Rightarrow \quad \alpha (i) \leq \alpha (j).
$$

Example 3.1.5.13. The morphisms $f _ { 1 } , f _ { 2 }$ , and $f _ { 3 }$ shown below 

$$
\begin{array}{c c c} 1 2 8 & \xrightarrow {} & 1 2 8 \\ 5 1 2 & \xrightarrow {} & 5 1 2 \\ 3 & \xrightarrow {} & f _ {1} \end{array} \quad \begin{array}{c c c} 4 & \xrightarrow {} & 4 \\ 1 & \xrightarrow {} & 1 \\ 1 & \xrightarrow {} & 8 \\ 1 & \xrightarrow {} & 6 4 \end{array} \quad \begin{array}{c c c} 6 0 & \xrightarrow {} & 6 0 \\ 2 0 & \xrightarrow {} & 2 \\ 3 2 & \xrightarrow {} & 2 0 \\ 8 & \xrightarrow {} & 4 \end{array}
$$

are sorted, while the morphisms $g _ { 1 } , g _ { 2 }$ , and $g _ { 3 }$ shown below 

![image](Imgaes/categorical-foundations-cute-layouts-paper/894632f74a4d4d369335e4a5be37bf17744424d745a127d47d936bfc74b4c31f.jpg)


are not sorted. The morphisms $g _ { 1 } , g _ { 2 } .$ , and $g _ { 3 }$ violate conditions 3, 1, and 2, respectively. 

Proposition 3.1.5.14. If f is a sorted tuple morphism, then the flat layout $L _ { f }$ is sorted. 

Proof. Suppose 

$$
(s _ {1}, \ldots , s _ {m}) \xrightarrow [ \alpha ]{f} (t _ {1}, \ldots , t _ {n})
$$

is sorted, and consider the layout 

$$
L _ {f} = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m}).
$$

Suppose $1 \leq i < m$ . We want to show that $d _ { i } < d _ { i + 1 } , \mathrm { o r } d _ { i } = d _ { i + 1 }$ and $s _ { i } \leq s _ { i + 1 }$ . There are two cases to consider. 

• (Case 1) Suppose that $\alpha ( i ) = * ,$ so that $d _ { i } = 0$ . If $\alpha ( i + 1 ) = *$ , then $d _ { i + 1 } = 0$ and since $f$ is sorted we have $s _ { i } \leq s _ { i + 1 } . \mathrm { ~ I f ~ } \alpha ( i + 1 ) \neq *$ , then $d _ { i + 1 } \geq 1 > 0 = d _ { i }$ 

• (Case 2) Suppose that $\alpha ( i ) \neq *$ , in which case $\alpha ( i + 1 ) \neq *$ and $\alpha ( i ) < \alpha ( i + 1 )$ . Then 

$$
d _ {i} = \prod_ {j <   \alpha (i)} t _ {j} \leq \prod_ {j <   \alpha (i + 1)} = d _ {i + 1},
$$

where equality holds only if $s _ { i } = t _ { \alpha ( i ) } = 1$ , which implies $s _ { i } \leq s _ { i + 1 }$ 

We conclude that $L _ { f }$ is sorted. 

Next, we define our $\mathsf { s o r t } ( - )$ operation on Tuple. If $f$ is a tuple morphism, then $\mathsf { s o r t } ( f )$ will be obtained by precomposing $f$ with an appropriate permutation g. 

Construction 3.1.5.15. Suppose 

$$
(s _ {1}, \ldots , s _ {m}) \xrightarrow [ \alpha ]{f} (t _ {1}, \ldots , t _ {n})
$$

is a tuple morphism. We define a permutation $\sigma \in \Sigma _ { m }$ as follows. Set 

$$
\begin{array}{l} P = \{i \in \langle m \rangle | \alpha (i) = * \}, \\ Q = \{i \in \langle m \rangle | \alpha (i) \neq * \}, \end{array}
$$

so $\langle m \rangle$ is the disjoint union of P and $Q .$ We define a linear ordering of P by $i _ { 1 } \preceq _ { P } i _ { 2 }$ if 

1. $s _ { i _ { 1 } } < s _ { i _ { 2 } }$ , or 

2. $s _ { i _ { 1 } } = s _ { i _ { 2 } }$ and $i _ { 1 } \le i _ { 2 }$ 

We define a linear ordering on $Q$ by $j _ { 1 } \preceq _ { Q } j _ { 2 } \mathrm { i f } \alpha ( i _ { 1 } ) \leq \alpha ( i _ { 2 } )$ . We define a linear ordering on $\langle m \rangle$ by $i _ { 1 } \preceq i _ { 2 }$ if 

1. $i _ { 1 } \in P$ and $i _ { 2 } \in Q$ 2 

2. $i _ { 1 } , i _ { 2 } \in P$ and $i _ { 1 } \preceq _ { P } i _ { 2 } , \mathrm { o r }$ 

3. $i _ { 1 } , i _ { 2 } \in Q$ and $i _ { 1 } \preceq _ { Q } i _ { 2 }$ 

Let $\sigma$ be permutation associated to the linear ordering ⪯ of ⟨m⟩, and let $\sigma ^ { - 1 }$ be its inverse. The map $\sigma _ { * } ^ { - 1 } : \langle m \rangle _ { * } \to \langle m \rangle$ <sub>∗</sub> is covered by a tuple morphism 

$$
g: \big (s _ {\sigma^ {- 1} (1)}, \ldots , s _ {\sigma^ {- 1} (m)} \big) \to \big (s _ {1}, \ldots , s _ {m} \big),
$$

and we define $\mathsf { s o r t } ( f )$ to be the composite 

$$
\operatorname{sort} (f) = f \circ g.
$$

Example 3.1.5.16. The sortings of the morphisms $g _ { 1 } , \ g _ { 2 }$ , and $g _ { 3 }$ of Example 3.1.5.13 are shown 

below. 

![image](Imgaes/categorical-foundations-cute-layouts-paper/2fca18bb84010c2fd808045c17da471790bedec76f4c35ec31f3e85c36f1a5a0.jpg)



Lemma 3.1.5.17. Suppose $f : S  T$ is a tuple morphism. Then f is sorted if and only if $\mathsf { s o r t } ( f ) = f$ Proof. Our construction of sort(−) guarantees that sort(f) is sorted for any tuple morphism $f .$ In particular, if $f = { \mathsf { s o r t } } ( f )$ , then $f$ is sorted. Conversely, if f is sorted, then the permutation $\sigma \in \Sigma _ { m }$ from Construction 3.1.5.15 is the identity permutation, so $g = \mathsf { i d } _ { S }$ , and so


$$
\operatorname{sort} (f) = f \circ \mathrm{id} _ {S} = f.
$$

Proposition 3.1.5.18. If f is a tuple morphism, then 

$$
L _ {\text { sort } (f)} = \text { sort } (L _ {f}).
$$

Proof. Borrowing our notation form Construction 3.1.5.15, we have sort $( f ) = f \circ g$ where 

$$
g: \left(s _ {\sigma^ {- 1} (1)}, \ldots , s _ {\sigma^ {- 1} (m)}\right) \to \left(s _ {1}, \ldots , s _ {m}\right)
$$

lies over $\sigma _ { * } ^ { - 1 } : \langle m \rangle _ { * }  \langle m \rangle _ { * } . \mathrm { ~ I f ~ } L _ { f } = ( s _ { 1 } , \dots , s _ { m } ) : ( d _ { 1 } , \dots , d _ { m } )$ , then 

$$
\begin{array}{c} L _ {\mathsf {s o r t} (f)} = (s _ {1} ^ {\prime}, \ldots , s _ {m} ^ {\prime}): (d _ {1} ^ {\prime}, \ldots , d _ {m} ^ {\prime}) \\ = (s _ {\sigma^ {- 1} (1)}, \ldots , s _ {\sigma^ {- 1} (m)}): (d _ {\sigma^ {- 1} (1)}, \ldots , d _ {\sigma^ {- 1} (m)}). \end{array}
$$

Since the modes of $L _ { \mathsf { s o r t } ( f ) }$ are a permutation of the modes of $L _ { f } .$ , it sufices to prove that $L _ { \mathsf { s o r t } ( f ) }$ is sorted. Suppose $1 \leq i < m$ . Suppose first that $\sigma ^ { - 1 } ( i ) \in P ,$ , so that $d _ { i } ^ { \prime } = d _ { \sigma ^ { - 1 } ( i ) } = 0$ . If $\sigma ^ { - 1 } ( i + 1 ) \in P _ { \mathrm { : } }$ then $d _ { i + 1 } ^ { \prime } = d _ { \sigma ^ { - 1 } ( i + 1 ) } = 0$ . By construction of $\sigma _ { \mathrm { { : } } }$ we have $s _ { i } ^ { \prime } = s _ { \sigma ^ { - 1 } ( i ) } \le s _ { \sigma ^ { - 1 } ( i + 1 ) } = s _ { i + 1 } ^ { \prime }$ . If instead $\sigma ^ { - 1 } ( i + 1 ) \in Q$ , then $d _ { i + 1 } ^ { \prime } = d _ { \sigma ^ { - 1 } ( i + 1 ) } ^ { \prime } > 0 = d _ { i } ^ { \prime }$ . Suppose next that $\sigma ^ { - 1 } ( i ) \in Q .$ . Then by construction of $\sigma ,$ we have $\sigma ^ { - 1 } ( i + 1 ) \in Q$ and $\overset { \cdot } { \alpha } ( \overset { \cdot } { \sigma } ^ { - 1 } ( i ) ) < \alpha ( \sigma ^ { - 1 } ( i + 1 ) )$ , and we have 

$$
\begin{array}{r l} & d _ {i} ^ {\prime} = d _ {\sigma^ {- 1} (i)} = \prod_ {j <   \alpha (\sigma^ {- 1} (i))} t _ {j} \\ & \qquad \leq \prod_ {j <   \alpha (\sigma^ {- 1} (i + 1))} t _ {j} \\ & \qquad = d _ {\sigma^ {- 1} (i + 1)} \\ & \qquad = d _ {i + 1} ^ {\prime}, \end{array}
$$

where equality holds if and only if $t _ { \alpha ( \sigma ^ { - 1 } ( i ) ) } = \cdot \cdot \cdot = t _ { \alpha ( \sigma ^ { - 1 } ( i + 1 ) ) - 1 } = 1$ . In particular, we have $s _ { i } =$ $s _ { \sigma ^ { - 1 } ( i ) } = t _ { \alpha ( \sigma ^ { - 1 } ( i ) ) } = 1$ , and so $s _ { i } ^ { \prime } \leq s _ { i + 1 } ^ { \prime }$ . We conclude that $L _ { \mathrm { s o r t ( f ) } }$ is sorted, so $L _ { \mathsf { s o r t } ( f ) } = { \mathsf { s o r t } } ( L _ { f } )$ □ Remark 3.1.5.19. The operation $\mathsf { s o r t } ( - )$ is not functorial. For example, consider the tuple morphisms $( 2 , 3 ) \xrightarrow { f } ( 3 , 2 )$ and (10, 25) $\frac { g } { \left( 2 , 1 \right) } \ \left( 2 5 , 1 0 \right)$ 

Then $f$ and g are composable with $g \circ f = \mathsf { i d } _ { ( 2 5 , 1 0 ) }$ , but the sorted morphisms $( 1 0 , 2 5 ) \xrightarrow [ { ( 1 , 2 ) } ] { \mathsf { s o r t } ( f ) } ( 1 0 , 2 5 )$ and (25, 10) $\xrightarrow { \mathsf { s o r t } ( g ) }$ (25, 10) 

are not composable. 

## 3.1.5.4 Coalesce

We begin by introducing the notion of a coalesced tuple morphism. 

Definition 3.1.5.20. Suppose $f : S  T$ is a tuple morphism lying over α. We say f is coalesced if 

1. $S = { \mathsf { s q u e e z e } } ( S )$ and 

2. for any $1 \leq i < \mathsf { l e n } ( S )$ , exactly one of the following conditions holds: 

(a) $\alpha ( i ) = * \neq \alpha ( i + 1 )$ 

(b) $\alpha ( i ) \neq * = \alpha ( i + 1 )$ 2 

(c) $\alpha ( i ) > \alpha ( i + 1 ) , \mathrm { o r }$ 

(d) $\alpha ( i ) < \alpha ( i + 1 )$ , and there exists $\alpha ( i ) < j < \alpha ( i + 1 )$ with $t _ { j } > 1$ 

Example 3.1.5.21. If there exists some $1 \leq i < \mathsf { l e n } ( S )$ with $\alpha ( i + 1 ) = \alpha ( i ) + 1$ , then $f$ is not coalesced. 

Remark 3.1.5.22. If $f : S  T$ is a tuple morphism such that $f = { \mathsf { s q u e e z e } } ( f )$ , then $f$ is coalesced if and only if for any $1 \leq i < \mathsf { l e n } ( S )$ , one of the following conditions holds: 

1. $\alpha ( i ) = * \neq \alpha ( i + 1 ) .$ 

2. $\alpha ( i ) \neq * = \alpha ( i + 1 )$ 

3. $\alpha ( i ) > \alpha ( i + 1 ) , \mathrm { o r }$ 

4. $\alpha ( i + 1 ) \neq \alpha ( i ) + 1 .$ 

Example 3.1.5.23. The morphisms 

![image](Imgaes/categorical-foundations-cute-layouts-paper/1a1fe11033b7a1889b2604ab64a658ecbbfb108168c09eaa54240a839c5a05b5.jpg)


are coalesced, while the morphisms 

![image](Imgaes/categorical-foundations-cute-layouts-paper/2efb74c510834ce37156da0d1be47639531b68afa2fe22e190513eadc6e15f8b.jpg)


are not coalesced. 

Proposition 3.1.5.24. Suppose f is a tuple morphism. Then f is coalesced if and only $i f ~ L _ { f }$ is coalesced. 

Proof. Suppose $f : ( s _ { 1 } , \ldots , s _ { m } ) \to ( t _ { 1 } , \ldots , t _ { n } )$ is a tuple morphism, and let 

$$
L _ {f} = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m})
$$

be the layout encoded by $f .$ 

Suppose first that f is coalesced. Then no entry of shape $( L _ { f } ) = \mathsf { d o m a i n } ( f )$ is equal to 1. Suppose $1 \leq i < m$ . We want to show that $s _ { i } d _ { i }$ is not equal to $d _ { i + 1 }$ . If $d _ { i } = 0$ , then $\alpha ( i ) = *$ , and we have 

$$
\begin{array}{r l} s _ {i} d _ {i} = d _ {i + 1} & \Leftrightarrow \quad d _ {i + 1} = 0 \\ & \Leftrightarrow \quad \alpha (i + 1) = * \end{array}
$$

but by our assumption that $f$ is coalesced, we have $\alpha ( i + 1 ) \neq *$ , hence $s _ { i } d _ { i } \neq d _ { i + 1 }$ . If $d _ { i } \neq 0$ , then $\alpha ( i ) \neq * . \mathrm { I f } \alpha ( i + 1 ) = * .$ , then $d _ { i + 1 } = 0$ , so $s _ { i } d _ { i } \neq d _ { i + 1 }$ . If $\alpha ( i + 1 ) < \alpha ( i )$ , then $d _ { i } \geq d _ { i + 1 }$ , and since $s _ { i } \neq 1$ , we have $s _ { i } d _ { i } > d _ { i + 1 }$ . Finally, ${ \mathrm { i f ~ } } \alpha ( i ) < \alpha ( i + 1 )$ , then 

$$
\begin{array}{c} s _ {i} d _ {i} = s _ {i} \cdot \left(\prod_ {j <   \alpha (i)} t _ {j}\right) = \prod_ {j \leq \alpha (i)} t _ {j} \\ <   \prod_ {j <   \alpha (i + 1)} t _ {j} \\ = d _ {i + 1}. \end{array}
$$

We conclude that $L _ { f }$ is coalesced. 

Suppose next that the layout $L _ { f }$ is coalesced. Then no entry in domain $( f ) = { \mathsf { s h a p e } } ( L _ { f } )$ is equal to 1. Suppose $1 \leq i < m$ . If $\alpha ( i ) = *$ , then $d _ { i } = 0$ , and since $L _ { f }$ is coalesced, we must have ${ d _ { i + 1 } } \neq s _ { i } { d _ { i } } = 0$ hence $\alpha ( i + 1 ) \neq *$ . Suppose $\alpha ( i ) \neq *$ , and $\alpha ( i ) < \alpha ( i + 1 )$ . Since $L _ { f }$ is coalesced, we have $s _ { i } d _ { i } \neq d _ { i + 1 }$ But if we write 

$$
s _ {i} d _ {i} = \prod_ {j \leq \alpha (i)} t _ {j},
$$

and 

$$
d _ {i + 1} = \prod_ {j <   \alpha (i + 1)} t _ {j},
$$

this implies that $\textstyle \prod _ { \alpha ( i ) < j < \alpha ( i + 1 ) } t _ { j } \neq 1$ . In particular, there exists some $\alpha ( i ) < j < \alpha ( i + 1 )$ with $t _ { j } > 1$ We conclude that $f$ is coalesced. □ 

Next, we define our coal(−) operation on tuple morphisms. 

Construction 3.1.5.25. Suppose f is a tuple morphism. We define a morphism coal(f) as follows: 

1. First, we set $g = { \tt s q u e e z e } ( f )$ , and we write $\beta : \langle m \rangle _ { * } \to \langle n \rangle$ <sub>∗</sub> for the map over which g lies. 

2. Next, we define an equivalence relation $\sim \mathrm { o n } \ \langle m \rangle$ where $i \sim i ^ { \prime }$ if either 

(a) $\beta ( i ^ { \prime \prime } ) = * \mathrm { f o r } i \le i ^ { \prime \prime } \le i ^ { \prime } ,$ or 

(b) $\beta ( i ^ { \prime \prime } ) = \beta ( i ) + ( i ^ { \prime \prime } - i ) \mathrm { f o r } i \le i ^ { \prime \prime } \le i ^ { \prime } .$ 

The quotient $\langle m \rangle / \sim$ is ordered by $[ i _ { 1 } ] \le [ i _ { 2 } ] \ \mathrm { i f } \ i _ { 1 } \le i _ { 2 }$ , so we can identify this quotient with ⟨m¯ ⟩ where ¯m is the size of $\langle m \rangle / \sim .$ 

3. Next, define an equivalence relation ∼ on ⟨n⟩ where $j \sim j ^ { \prime }$ if there exists $i \in \langle m \rangle$ such that 

$$
\beta (i + (j ^ {\prime \prime} - j)) = \beta (i) + (j ^ {\prime \prime} - j)
$$

for all $j \le j ^ { \prime \prime } \le j ^ { \prime }$ . The quotient $\langle n \rangle / \sim$ is ordered by $[ j _ { 1 } ] \le [ j _ { 2 } ] \ \mathrm { i f } \ j _ { 1 } \le j _ { 2 }$ , so we can identify this quotient with ⟨n¯⟩ where ¯n is the size of $\langle n \rangle / \sim$ 

4. Next, we observe that the map $\beta : \langle m \rangle _ { * } \to \langle n \rangle$ ∗ descends to a map 

$$
\bar {\beta}: \langle \bar {m} \rangle_ {*} \to \langle \bar {n} \rangle_ {*}
$$

given by $\bar { \beta } ( [ i ] ) = [ \beta ( i ) ]$ 

5. The domain $\bar { S } = ( \bar { s } _ { 1 } , \dots , \bar { s } _ { \bar { m } } )$ of coal(f) is defined by setting 

$$
\bar {s} _ {i} = \prod_ {i ^ {\prime} \in I} s _ {i ^ {\prime}}
$$

if $i \in \langle \bar { m } \rangle$ corresponds to the equivalence class $I \in \langle m \rangle / \sim$ . The codomain $\bar { T } = ( \bar { t } _ { 1 } , \dots , \bar { t } _ { \bar { n } } )$ of coal(f) is defined by setting 

$$
\bar {t} _ {j} = \prod_ {j ^ {\prime} \in J} t _ {j ^ {\prime}}
$$

if $j \in \langle \bar { n } \rangle$ corresponds to the equivalence class $J \in \langle n \rangle / \sim$ . We then define 

$$
\operatorname{coal} (f): \bar {S} \to \bar {T}
$$

to be the tuple morphism lying over ${ \bar { \beta } } .$ 

Example 3.1.5.26. Here is an example of a tuple morphism f and the coalesced morphism coal(f). 

![image](Imgaes/categorical-foundations-cute-layouts-paper/d8b19fdd4e539c59e482168f12f48ca24e61269e6a41636aa90f580bb6ec80a4.jpg)


Example 3.1.5.27. We can coalesce the morphism f of Example 3.1.5.8 as follows 

![image](Imgaes/categorical-foundations-cute-layouts-paper/d2a71ed8b2bdc41bf93b7c1828b5503cca6f7404673d54f2513733831c096f6f.jpg)



Proposition 3.1.5.28. If f is a tuple morphism, then


1. coal(f) is coalesced, and 

2. $L _ { \mathsf { c o a l } ( f ) } = { \mathsf { c o a l } } ( L _ { f } ) .$ 

Proof. First, we will argue that coal(f) is coalesced. This is immediate from our construction, since applying squeeze eliminates all modes equal to 1, and passing to the quotient in our construction consolidated all adjacent modes with $\alpha ( i + 1 ) = \alpha ( i ) + 1$ 

Next, we will prove that $L _ { \mathsf { c o a l } ( f ) } = \mathsf { c o a l } ( L _ { f } )$ . In light of Proposition 2.1.4.18 and Proposition 3.1.5.24, it sufices to prove that $\Phi _ { \mathsf { c o a l } ( f ) } = \Phi _ { f }$ . Certainly applying squeeze(−) to f has no impact on the associated layout function, so we need to argue that passing to the quotient in our construction does not change the layout function of the associated layout. This follows from the fact that forming our quotient can be formed in steps, where in each step we combine adjacent modes with either $\alpha ( i ) = * = \alpha ( i + 1 )$ , or $\alpha ( i + 1 ) = \alpha ( i ) + 1$ . These correspond to replacing adjacent modes of the form $s _ { i } , s _ { i + 1 } : 0 , 0$ with $s _ { i } s _ { i + 1 } : 0 ,$ , and $s _ { i } , s _ { i + 1 } : d _ { i } , s _ { i } d _ { i }$ with $s _ { i } s _ { i + 1 } : d _ { i }$ , respectively. Neither such operation changes the layout function of a layout, and so we conclude that $\Phi _ { L _ { \mathsf { c o a l } ( f ) } } = \Phi _ { \mathsf { c o a l } ( L _ { f } ) }$ , as desired. 

## 3.1.5.5 Concatenate

Next, we will define a concatenation operation on tuple morphisms. This operation may be performed on tuple morphisms satisfying a “disjointness” condition, which we specify below. 

Definition 3.1.5.29. Suppose $\alpha : \langle m \rangle _ { * } \to \langle n \rangle$ <sub>∗</sub> and $\beta : \langle p \rangle _ { * } \to \langle n \rangle$ <sub>∗</sub> are morphisms in Fin<sub>∗</sub> with the same codomain. We say α and $\beta$ have disjoint images if 

$$
\operatorname{Image} (\alpha) \cap \operatorname{Image} (\beta) = \{* \}.
$$

Construction 3.1.5.30. If $\alpha : \langle m \rangle _ { * } \to \langle n \rangle$ <sub>∗</sub> and $\beta : \langle p \rangle _ { * } \to \langle n \rangle$ <sub>∗</sub> have disjoint images, then we have a well-defined morphism 

$$
\alpha \star \beta : \langle m + p \rangle_ {*} \rightarrow \langle n \rangle_ {*}
$$

given by 

$$
(\alpha \star \beta) (i) = \left\{ \begin{array}{l l} * & i = * \\ \alpha (i) & 1 \leq i \leq m \\ \beta (i - m) & m + 1 \leq i \leq m + p. \end{array} \right.
$$

This operation is associative, so we can consider $\alpha _ { 1 } \star \cdot \cdot \star \alpha _ { k }$ for any collection of morphisms $\alpha _ { 1 } , \ldots , \alpha _ { k }$ in Fin<sub>∗</sub> with pairwise disjoint images. 

Remark 3.1.5.31. If $\alpha$ and $\beta$ are tractable pointed maps and α and $\beta$ have disjoint images, then $\alpha \star \beta$ is tractable. 

Definition 3.1.5.32. Suppose 

$$
f: S \to T
$$

and 

$$
g: U \to T
$$

are tuple morphisms lying over α and $\beta ,$ respectively. We say $f$ and $g$ have disjoint images if the morphisms α and $\beta$ have disjoint images. 

Example 3.1.5.33. Consider the tuple morphisms $f , g ,$ and h shown below. 

![image](Imgaes/categorical-foundations-cute-layouts-paper/bbc295a80b3292b94b3c5dfb231f40bcb1036c4fa27780a91a0d5bd608d7769f.jpg)


Then $f$ and $g$ have disjoint images, while $h$ and $g$ do not have disjoint images. 

Construction 3.1.5.34. Suppose 

$$
f: S \to T, \text { and } g: U \to T
$$

are tuple morphisms lying over α and $\beta ,$ respectively, and that $f$ and $g$ have disjoint images. We define the concatenation of $f$ and $g$ to be the morphism 

$$
f \star g: S \star U \to T
$$

lying over α ⋆ $\beta .$ This operation is associative, so we can consider $f _ { 1 } \cdot \cdot \cdot f _ { k }$ for any finite collection of morphisms $f _ { i }$ with pairwise disjoint images. 

Example 3.1.5.35. If $f$ and $g$ are the morphisms in Tuple from Example 3.1.5.33, then the concatenation of $f$ and $g$ is the morphism shown below. 

![image](Imgaes/categorical-foundations-cute-layouts-paper/23b5376f57658573396c48d14914ecdb3afd5e61c8de10a9ae66e3fb56356181.jpg)



f ⋆ g


Example 3.1.5.36. Suppose $f : ( s _ { 1 } , \ldots , s _ { m } ) \to ( t _ { 1 } , \ldots , t _ { n } )$ is a tuple morphism, and for any $1 \leq i \leq m$ let 

$$
f _ {i}: (s _ {i}) \to (t _ {1}, \dots , t _ {n})
$$

denote the ith entry of $f ,$ as in Example 3.1.3.8. Then we can write 

$$
f = f _ {1} \star \dots \star f _ {m}
$$

as the concatenation of its entries. 

Lemma 3.1.5.37. Suppose $f _ { 1 } : S _ { 1 } \to T$ and $f _ { 2 } : S _ { 2 } \to T$ are tuple morphisms with disjoint images. $I f g : T \to U$ is any tuple morphism, then 

$$
g \circ (f _ {1} \star f _ {2}) = (g \circ f _ {1}) \star (g \circ f _ {2}).
$$

Proof. Suppose $f _ { 1 } , \ f _ { 2 }$ , and $g$ lie over $\alpha _ { 1 } : \langle m _ { 1 } \rangle _ { * } \to \langle n \rangle , \alpha _ { 2 } : \langle m _ { 2 } \rangle _ { * } \to \langle n \rangle$ , and $\beta : \langle n \rangle  \langle p \rangle$ ， respectively. The two maps in question have the same domains and the same codomains, so it sufices to prove that 

$$
\beta \circ (\alpha_ {1} \star \alpha_ {2}) = (\beta \circ \alpha_ {1}) \star (\beta \circ \alpha_ {2}).
$$

We compute 

$$
\begin{array}{l} (\beta \circ (\alpha_ {1} \star \alpha_ {2})) (i) = \beta ((\alpha_ {1} \star \alpha_ {2}) (i)) \\ = \left\{ \begin{array}{l l} \beta (*) & i = * \\ \beta (\alpha_ {1} (i)) & 1 \leq i \leq m _ {1} \\ \beta (\alpha_ {2} (i - m _ {1})) & m _ {1} + 1 \leq i \leq m _ {1} + m _ {2} \end{array} \right. \\ = \left\{ \begin{array}{l l} * & i = * \\ (\beta \circ \alpha_ {1}) (i) & 1 \leq i \leq m _ {1} \\ (\beta \circ \alpha_ {2}) (i - m _ {1}) & m _ {1} + 1 \leq i \leq m _ {1} + m _ {2} \end{array} \right. \\ = ((\beta \circ \alpha_ {1}) \star (\beta \circ \alpha_ {2})) (i). \end{array}
$$

Proposition 3.1.5.38. Suppose $f _ { 1 } , \ldots , f _ { k }$ are morphisms in Tuple with the same codomain and with pairwise disjoint images. Then the layouts $L _ { f _ { 1 } } , \ldots , L _ { f _ { k } }$ satisfy 

$$
L _ {f _ {1} \star \dots \star f _ {k}} = L _ {f _ {1}} \star \dots \star L _ {f _ {k}}.
$$

Proof. First, we prove the result for $k = 2$ . Suppose 

$$
f = (s _ {1}, \ldots , s _ {m}) \rightarrow (t _ {1}, \ldots , t _ {n}), \text {and} g: (u _ {1}, \ldots , u _ {p}) \rightarrow (t _ {1}, \ldots , t _ {n})
$$

have disjoint images, and write 

$$
L _ {f} = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m}), \text {   and   } L _ {g} = (u _ {1}, \dots , u _ {p}): (d _ {1} ^ {\prime}, \dots , d _ {p} ^ {\prime}).
$$

Then the layout $L _ { f \star g }$ is given by 

$$
L _ {f \star g} = (s _ {1}, \ldots , s _ {m}, u _ {1}, \ldots , u _ {p}): (e _ {1}, \ldots , e _ {m + m ^ {\prime}})
$$

where 

$$
\begin{array}{l} e _ {i} = \prod_ {j <   (\alpha \star \beta) (i)} t _ {j} \\ = \left\{ \begin{array}{l l} \prod_ {j <   \alpha (i)} t _ {j} & 1 \leq i \leq m \\ \prod_ {j <   \beta (i - m)} t _ {j} & m + 1 \leq i \leq m + m ^ {\prime}. \end{array} \right. \\ = \left\{ \begin{array}{l l} d _ {i} & 1 \leq i \leq m \\ d _ {i - m} ^ {\prime} & m + 1 \leq i \leq m + m ^ {\prime}. \end{array} \right. \end{array}
$$

This concludes the proof of the result when $k = 2 .$ . The general case follows from the associativity of concatenation of tuple morphisms, and the associativity of concatenation of flat layouts. □ 

## 3.1.5.6 Complement

We begin by defining the notion of complementary tuple morphisms. 

Definition 3.1.5.39. Suppose $f : S  T$ and $g : U \to T$ are tuple morphisms. We say $g$ is a complement of f if 

1. f and g have disjoint images, and 

2. the concatenation 

$$
f \star g: S \star U \xrightarrow {\cong} T
$$

is an isomorphism. 

Example 3.1.5.40. If f and g are the morphisms shown below 

$$
\begin{array}{c}1 6\\\rightarrow 3 2\\3 2 \xrightarrow {} 3 2\\3 2 \xrightarrow {} 1 0\end{array}f \quad \text {   g   } \quad\begin{array}{c}1 6\\\rightarrow 3 2\\1 0 \xrightarrow {} 3 2\\1 6\end{array}
$$

then g is a complement of $f .$ 

Example 3.1.5.41. If f is the morphism shown below 

$$
\begin{array}{c} 2 5 6 \\ 1 2 8 \\ 1 2 8 \end{array} \xrightarrow {} \begin{array}{c} 1 2 8 \\ 2 5 6 \end{array}
$$

then f does not admit a complement. 

Next, we prove that complementary tuple morphisms give rise to complementary flat layouts. 

Proposition 3.1.5.42. $I f f : S \to T$ is a tuple morphism and g is a complement of f, then $L _ { g }$ is a size(T)-complement of $L _ { f }$ . 

Proof. Write $S = \mathsf { d o m a i n } ( f ) , U = \mathsf { d o m a i n } ( g )$ , and $T = \mathsf { c o d o m a i n } ( f ) = \mathsf { c o d o m a i n } ( g )$ . First, we note that 

$$
\begin{array}{r l} \mathsf {s i z e} (L _ {f}) \cdot \mathsf {s i z e} (L _ {g}) & = \mathsf {s i z e} (L _ {f} \star L _ {g}) \\ & = \mathsf {s i z e} (L _ {f \star g}) \\ & = \mathsf {s i z e} (S \star U) \\ & = \mathsf {s i z e} (T). \end{array}
$$

Next, we note that $f \star g$ is an isomorphism, hence so is 

$$
| f \star g | = \Phi_ {L _ {f \star g}} ^ {\mathrm{size} (T)}
$$

where we have used the identification of $\Phi _ { L _ { f \star g } } ^ { \mathsf { s i z e } ( T ) }$ of Lemma 3.1.4.5. 

Proposition 3.1.5.43. If f is an injective tuple morphism, then 

$$
\operatorname{coal} ^ {\flat} \left(L _ {f ^ {c}}\right) = \operatorname{comp} ^ {\flat} \left(L _ {f}, \operatorname{size} (T)\right).
$$

Proof. By Proposition 3.1.5.42, we know that $L _ { f ^ { c } }$ is a size $( T )$ -complement of $L _ { f }$ . Since $f ^ { c }$ is sorted, so is $L _ { f ^ { c } }$ and it follows from Proposition 2.1.6.33, it follows that 

$$
\operatorname{coal} ^ {\flat} \left(L _ {f ^ {c}}\right) = \operatorname{comp} ^ {\flat} \left(L _ {f}, \operatorname{size} (T)\right),
$$

since both of these layouts are flat, sorted, coalesced complements of $L _ { f }$ of the same size. □ 

Proposition 3.1.5.44. If $f : ( s _ { 1 } , \ldots , s _ { m } ) \to ( t _ { 1 } , \ldots , t _ { n } )$ is an injective tuple morphism of standard form, then 

$$
L _ {f ^ {c}} = \mathsf {c o m p} ^ {\flat} (L _ {f}).
$$

Proof. Write 

$$
L _ {f} = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m})
$$

for the layout encoded by f. By Proposition 3.1.5.42, we know that $L _ { f ^ { c } }$ is a size $( T )$ -complement of $L _ { f }$ . Where 

$$
\begin{array}{c} \text {size} (T) = t _ {1} \dots t _ {n} = (t _ {1} \dots t _ {n - 1}) t _ {n} \\ = d _ {m} s _ {m}. \end{array}
$$

By construction, $f ^ { c }$ is sorted, hence so is $L _ { f ^ { c } }$ . Moreover, since $f$ has standard form, it follows that $f ^ { c }$ is coalesced. By Proposition 2.1.6.23, we deduce that 

$$
L _ {f ^ {c}} = \mathsf {c o m p} ^ {\flat} (L _ {f}).
$$

Definition 3.1.5.45. Suppose $f$ is a tuple morphism lying over $\alpha : \langle m \rangle _ { * } \to \langle n \rangle$ <sub>∗</sub>. We say $f$ is complementable if α is injective. 

Construction 3.1.5.46. Suppose $f : ( s _ { 1 } , \ldots , s _ { m } ) \to ( t _ { 1 } , \ldots , t _ { n } )$ is a complementable tuple morphism. Let $j _ { 1 } < \cdots < j _ { n - m }$ denote the collection of indices in $\langle n \rangle$ which are not in the image of $\alpha .$ We define the complement of $f$ to be the tuple morphism 

$$
f ^ {c}: (t _ {j _ {1}}, \dots , t _ {j _ {k}}) \to (t _ {1}, \dots , t _ {n})
$$

lying over the map complement $( \alpha ) : \langle n - m \rangle _ { * } \to \langle n \rangle$ <sub>∗</sub> given by $k \mapsto j _ { k }$ . By construction, we may observe that $f ^ { c }$ is a complement of $f ,$ in the sense of Definition 3.1.5.39 

Example 3.1.5.47. Below is an example of a morphism f and its complement $f ^ { c }$ 

![image](Imgaes/categorical-foundations-cute-layouts-paper/7013fe7d3b12b3129acd1049b70e42107baefe0668239d91c17cef334011b740.jpg)


Proposition 3.1.5.48. $I f f$ is a tuple morphism and $g$ is a complement of f, then 

$$
\operatorname{sort} (g) = f ^ {c}.
$$

Proof. Suppose f lies over $\alpha : \langle m \rangle _ { * } \to \langle n \rangle _ { * } , { \sf s o r t } ( g )$ lies over $\beta : \langle n - m \rangle _ { * } \to \langle n \rangle$ <sub>∗</sub> and $f ^ { c }$ lies over $\alpha ^ { c } : \langle n - m \rangle _ { * } \to \langle n \rangle _ { }$ <sub>∗</sub>. Then $\beta$ and $\alpha ^ { c }$ are increasing maps with the same image, namely 

$$
\operatorname{Image} (\beta) = \langle n \rangle \setminus \operatorname{Image} (\alpha) = \operatorname{Image} \left(\alpha^ {c}\right),
$$

hence $\beta = \alpha ^ { c }$ , and hence sort $\operatorname { \rho } ( g ) = f ^ { c }$ 

Proposition 3.1.5.49. Suppose $f$ is a tuple morphism. Then $f$ admits a complement if and only if $f$ is complementable, in the sense of Definition $\ 3 . 1 . 5 . 4 5$ 

Proof. If f lies over a map α which is not injective, then for any morphism $f ^ { * }$ such that f and $f ^ { * }$ have disjoint images, the morphism $f \star f ^ { * }$ lies over a map which is not injective, hence $f \star f ^ { * }$ is not an isomorphism. Conversely, if $f$ lies over an injective map, then the morphism $f ^ { c }$ of Construction 3.1.5.46 is a complement of $f .$ . □ 

Proposition 3.1.5.50. $I f f$ is a complementable tuple morphism, then 

$$
\operatorname{sort} (f) = \left(f ^ {c}\right) ^ {c}.
$$

Proof. Both maps are increasing, injective, and have the same image, so they are equal. 

## 3.1.5.7 Flat division

In this section, we define a division operation on tuple morphisms. 

Definition 3.1.5.51. If $f$ and $g$ are tuple morphisms, we say g divides f if g and $f$ are composable. In other words, 

$$
\operatorname{codomain} (g) = \operatorname{domain} (f).
$$

Definition 3.1.5.52. Suppose $g : S  T$ and $f : T  U$ are tuple morphisms. The flat division of $f$ by $g$ is the tuple morphism 

$$
f \oslash^ {\flat} g = f \circ (g \star g ^ {c}).
$$

Example 3.1.5.53. Here is an example of tuple morphisms f and g together with their flat quotient $f O ^ { \flat } g .$ 

$$
\begin{array}{c c c} 1 2 8 \longmapsto 1 2 8 & 1 2 8 \longmapsto 1 2 8 \\ g & 2 & f \\ \hline \end{array} \quad \begin{array}{c c c} 1 2 8 \longmapsto 1 2 8 \\ 2 \longmapsto 2 \\ \hline \end{array} \quad \begin{array}{c c c} 2 & 1 2 8 \\ 1 2 8 \longmapsto 1 2 8 \\ \hline \end{array} \quad \begin{array}{c c c} f \otimes^ {b} g \\ \hline \end{array}
$$

Example 3.1.5.54. Here is an example of tuple morphisms f and g together with their flat quotient $f O ^ { \flat } g .$ 

![image](Imgaes/categorical-foundations-cute-layouts-paper/6039b08bd0e80764f025e6b91c9d166f2ca197703d5d0d6105688dbb6788b82b.jpg)


Example 3.1.5.55. Here is an example of tuple morphisms f and g together with their flat quotient $f \oslash ^ { \flat } g$ 

![image](Imgaes/categorical-foundations-cute-layouts-paper/83b48eae2c5cdc04bde54607429487a128e20ab497181a086a7a3408e46ab53e.jpg)


Proposition 3.1.5.56. If f and g are non-degenerate composable tuple morphisms, then 

$$
\operatorname{coal} ^ {\flat} \left(L _ {f \oslash^ {\flat} g}\right) = \operatorname{coal} ^ {\flat} \left(L _ {f} \oslash^ {\flat} L _ {g}\right)
$$

Proof. By Proposition 3.2.6.20, we have 

$$
\operatorname{coal} ^ {\flat} \left(L _ {g ^ {c}}\right) = \operatorname{comp} ^ {\flat} \left(L _ {g}, \operatorname{size} \left(L _ {f}\right)\right),
$$

and we compute 

$$
\begin{array}{l} \text {coal} ^ {\flat} (L _ {f} \oslash^ {\flat} L _ {g}) = \text {coal} ^ {\flat} (L _ {f} \circ (L _ {g} \star \text {comp} (L _ {g}, \text {size} (L _ {f})))) \\ \qquad = \text {coal} (L _ {f} \circ (L _ {g} \star L _ {g ^ {c}})) \\ \qquad = \text {coal} (L _ {f} \circ L _ {g \star g ^ {c}}) \\ \qquad = \text {coal} (L _ {f \circ (g \star g ^ {c})}) \\ \qquad = \text {coal} (L _ {f \oslash^ {\flat} g}). \end{array}
$$

## 3.1.5.8 Flat products

In this section we define a product operation on tuple morphisms. 

Definition 3.1.5.57. Suppose f and g are tuple morphisms. We say $f$ and g are product admissible if codomain $\boldsymbol { \mathsf { \Pi } } ( g ) = \mathsf { d o m a i n } ( f ^ { c } )$ . If f and g are product admissible, then we define flat product of f and $g$ to be 

$$
f \otimes^ {\flat} g = f \star (f ^ {c} \circ g).
$$

Example 3.1.5.58. If f and g are the tuple morphisms shown below 

$$
\begin{array}{c c} & 1 6 \\ & 1 6 \\ 1 6 \longmapsto 1 6 & 8 \longmapsto 8 \\ 1 6 \longmapsto 1 6 & 8 \longmapsto 8 \\ g & f \end{array}
$$

then $f$ and g are product-admissible, and $f \otimes ^ { \flat } g$ is the tuple morphism shown below. 

$$
\begin{array}{c} 1 6 \longmapsto 1 6 \\ 1 6 \longmapsto 1 6 \\ 8 \longmapsto 8 \\ 8 \longmapsto 8 \\ f \otimes^ {b} g \end{array}
$$

Example 3.1.5.59. If $f$ and g are the tuple morphisms shown below 

$$
\begin{array}{c} \text {   g   } \\ \text {   f   } \end{array}
$$

then $f$ and $g$ are product-admissible, and $f \otimes ^ { \flat } g$ is the tuple morphism shown below. 

![image](Imgaes/categorical-foundations-cute-layouts-paper/c4ec3b33c0519c4ad77a7a7ba38e561c8c59789c5a50525e3762e59b51ea4772.jpg)


Lemma 3.1.5.60. If f and g are product admissible and $g$ is injective, then $f \otimes ^ { \flat } g$ is injective and 

$$
(f \otimes^ {\flat} g) ^ {c} = f ^ {c} \circ g ^ {c}.
$$

Proof. The tuple morphisms $( f \otimes ^ { \flat } g ) ^ { c }$ and $f ^ { c } \circ g ^ { c }$ are injective, increasing, and have the same codomain, so it sufices to show that they have the same image. The image of $( f \otimes ^ { \flat } g ) ^ { c } = ( f \star ( f ^ { c } \circ g ) ) ^ { c }$ consists of those entries which are not in the image of $f ,$ and not in the image of $f ^ { c } \circ g$ . The image of $f ^ { c }$ consists of those entries which are not in the image of $f ,$ and so the image of the composition $f ^ { c } \circ g ^ { c }$ consists of those entries which are not in the image of $f ,$ , and not in the image of $f ^ { c } \circ g$ □ 

Proposition 3.1.5.61. Suppose $f$ and g are product admissible, and g and h are product admissible. Then 

1. $f \otimes ^ { \flat } g$ and h are product admissible, 

2. $f$ and $g \otimes ^ { \flat } h$ are product admissible, and 

3. $( f \otimes ^ { \flat } g ) \otimes ^ { \flat } h = f \otimes ^ { \flat } ( g \otimes ^ { \flat } h )$ 

Proof. Using Lemma 3.1.5.37 and Lemma 3.1.5.60, we compute 

$$
\begin{array}{l} f \otimes^ {\flat} (g \otimes^ {\flat} h) = f \star (f ^ {c} \circ (g \otimes^ {\flat} h)) \\ \qquad = f \star (f ^ {c} \circ (g \star (g ^ {c} \circ h))) \\ \qquad = f \star ((f ^ {c} \circ g) \star (f ^ {c} \circ (g ^ {c} \circ h))) \\ \qquad = f \star ((f ^ {c} \circ g) \star ((f ^ {c} \circ g ^ {c}) \circ h)) \\ \qquad = f \star (f ^ {c} \circ g) \star ((f \otimes^ {\flat} g) ^ {c} \circ h) \\ \qquad = (f \otimes^ {\flat} g) \star ((f \otimes^ {\flat} g) ^ {c} \circ h) \\ \qquad = (f \otimes^ {\flat} g) \otimes^ {\flat} h. \end{array}
$$

Proposition 3.1.5.62. Suppose f and g are non-degenerate tuple morphisms and that f and $g$ are product admissible. Then 

$$
L _ {f \otimes^ {\flat} g} = L _ {f} \otimes^ {\flat} L _ {g}.
$$

Proof. Suppose $f : S  T$ and $g : U \to V$ are product admissible, and set 

$$
L _ {f} ^ {*} = \operatorname{comp} ^ {\flat} (L _ {f}, \operatorname{size} (L _ {f}) \cdot \operatorname{cosize} (L _ {g})).
$$

Since $f$ is injective and the codomain of $g$ is the domain of $f ^ { c } ,$ , it follows that 

$$
\operatorname{size} \left(L _ {f}\right) \cdot \operatorname{cosize} \left(L _ {g}\right) \leq \operatorname{size} (S) \cdot \operatorname{size} (V) = \operatorname{size} (T).
$$

Using this fact, and the fact that 

$$
\Phi_ {\mathrm{comp} (L _ {f}, \mathrm{size} (T))} = \Phi_ {L _ {f ^ {c}}},
$$

we have 

$$
\begin{array}{c} L _ {f} ^ {*} \circ L _ {g} = \mathsf {c o m p} (L _ {f}, \mathsf {s i z e} (T)) \circ L _ {g} \\ = L _ {f c} \circ L _ {g}. \end{array}
$$

$$
\begin{array}{r l} L _ {f} \otimes^ {\flat} L _ {g} & = L _ {f} \star (L _ {f} ^ {*} \circ L _ {g}) \\ & = L _ {f} \star (L _ {f ^ {c}} \circ L _ {g}) \\ & = L _ {f} \star L _ {f ^ {c} \circ g} \\ & = L _ {f \star (f ^ {c} \circ g)} \\ & = L _ {f \otimes^ {\flat} g} \end{array}
$$

Using this fact, we compute 

## 3.2 The category Nest

In the previous section, we introduced a category Tuple, whose morphisms encode flat tractable layouts. In this section, we introduce a category Nest, whose morphisms encode tractable layouts with arbitrary nesting. 

## 3.2.1 Basic definitions

Recall that for a nested tuple $S ,$ we write $S ^ { \flat }$ for the flattening of S. For example, if $S = ( 6 4 , ( 8 , 8 ) )$ , then $S ^ { \flat } = ( 6 4 , 8 , 8 )$ 

Definition 3.2.1.1. Let Nest denote the category whose objects are nested tuples of positive integers, and in which a morphism 

$$
f: S \to T
$$

in Nest is specified by a tuple morphism 

$$
f ^ {\flat}: S ^ {\flat} \to T ^ {\flat}.
$$

In other words, 

$$
\operatorname{Hom} _ {\mathbf {N e s t}} (S, T) = \operatorname{Hom} _ {\mathbf {T u p l e}} (S ^ {\flat}, T ^ {\flat}).
$$

Explicitly, a morphism $f : S  T$ in Nest is specified by a tractable pointed map $\alpha : \langle \mathsf { l e n } ( S ) \rangle _ { * } \to$ $\langle \mathsf { l e n } ( T ) \rangle$ ⟩<sub>∗</sub> satisfying the following property: 

$\mathrm { ~ I f ~ } 1 \leq i \leq \mathsf { l e n } ( S )$ and $\alpha ( i ) \neq *$ , then entry $\mathsf { \Omega } _ { i } ( S ) = \mathsf { e n t r y } _ { \alpha ( i ) } ( T )$ 

We say such a morphism f lies over $\alpha ,$ and refer to $f$ as a nested tuple morphism. 

Notation 3.2.1.2. If $f : S  T$ is a nested tuple morphism which lies over α, we depict $f$ as 

$$
S \xrightarrow [ \alpha ]{f} T
$$

Example 3.2.1.3. Here are some examples of nested tuple morphisms. 

$$
(6 4, (8, 8)) \xrightarrow [ (1 , 2 , 3) ]{f} (6 4, 8, 8)
$$

$$
((2, 2), 2) \xrightarrow [ (* , 5 , 2) ]{g} (1 0, 2, 2, (3, 2, 3))
$$

$$
6 4 \xrightarrow [ (2) ]{h} ((6 4, 6 4), 5 1 2).
$$

Observation 3.2.1.4. If X is a set, lets write $X ^ { \mathrm { i n d } }$ for the indiscrete category on $X$ . This is the category whose objects are the elements of $X .$ , and in which there is a unique (iso)morphism between any two objects. Then by definition of Nest, we have a pullback square 

$$
\begin{array}{c} \text {Nest} \xrightarrow {\text {prof(-)}} \text {Profile} ^ {\text {ind}} \\ (-) ^ {b} \Biggl \downarrow \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \text {len(-)} \\ \text {Tuple} \xrightarrow {\text {len(-)}} \mathbb {N} ^ {\text {ind}} \end{array}
$$

We may view this as a categorification of the pullback square 2.2.2.4. 

Example 3.2.1.5. Suppose $S$ is a nested tuple of length m. ${ \mathrm { I f ~ } } 1 \leq i \leq m$ then there is a nested tuple morphism 

$$
\operatorname{entry} _ {i} (S) \to S
$$

lying over the map $\langle 1 \rangle _ { * } \to \langle m \rangle$ <sub>∗</sub> given by $1 \mapsto i .$ For instance, if $S = ( 6 4 , ( 8 , 8 ) )$ and $i = 1$ , then we have a nested tuple morphism 

$$
6 4 \xrightarrow [ (1) ]{} (6 4, (8, 8)).
$$

Example 3.2.1.6. Suppose $S$ is a nested tuple of rank r. If $1 \leq i \leq r .$ , then there is a canonical nested tuple morphism 

$$
\operatorname{mode} _ {i} (S) \to S
$$

lying over the map $\langle \mathsf { l e n } _ { i } ( S ) \rangle _ { * } \to \langle \mathsf { l e n } ( S ) \rangle$ <sub>∗</sub> given by $j \mapsto j + \mathsf { l e n } _ { < i } ( S )$ . For instance, if $S = ( 6 4 , ( 8 , 8 ) )$ , then we have a nested tuple morphism 

$$
(8, 8) \xrightarrow [ (2 , 3) ]{} (6 4, (8, 8)).
$$

Observation 3.2.1.7. There are functors relating the categories Nest and Tuple. First, there is an inclusion functor 

$$
\text { Tuple } \xrightarrow {\subset} \text { Nest }
$$

which considers a tuple morphism $f : S  T$ as a nested tuple morphism. Next, there is a flattening functor 

$$
\text { Nest } \xrightarrow {(-) ^ {b}} \text { Tuple }
$$

which sends a nested tuple morphism $f : S  T$ to the underlying tuple morphism $f ^ { \flat } : S ^ { \flat } \to T ^ { \flat }$ . The composite 

$$
\text { Tuple } \xrightarrow {\subset} \text { Nest } \xrightarrow {(-) ^ {b}} \text { Tuple }
$$

is the identity functor on Tuple, so Tuple is a retractive subcategory of Nest. Moreover, these functors form an adjoint equivalence of categories. 

Remark 3.2.1.8. One might wish to consider some category C whose morphisms encode tractable layouts, but which is not equivalent to Tuple. The authors have considered several such examples, but leave their investigation to future work. 

## 3.2.2 From nested tuple morphisms to layouts

The key feature of the category Nest is that if $f : S  T$ is a nested tuple morphism, then $f$ encodes a layout $L _ { f }$ . This layout is obtained by equipping the flat layout $L _ { f ^ { \flat } }$ with the nesting profile of S. More precisely, we have the following construction. 

Construction 3.2.2.1. Suppose 

$$
f: S \to T
$$

is a nested tuple morphism, and suppose $P = { \mathsf { p r o f } } ( S )$ . We define $L _ { f }$ to be the layout 

$$
L _ {f} = (L _ {f ^ {\flat}}) _ {P}
$$

where $( - ) _ { P }$ is the P-substitution operation of Definition 2.3.1.19. We refer to $L _ { f }$ as the layout encoded by $f .$ 

Construction 3.2.2.2. Suppose 

$$
(s _ {1}, \ldots , s _ {m}) _ {P} \xrightarrow [ \alpha ]{f} (t _ {1}, \ldots , t _ {n}) _ {Q}
$$

is a nested tuple morphism. We define $L _ { f }$ to be the layout whose shape 

$$
\mathsf {s h a p e} (L _ {f}) = (s _ {1}, \dots , s _ {m}) _ {P}
$$

is the domain of $f ,$ and whose stride 

$$
\mathsf {s t r i d e} (L _ {f}) = (d _ {1}, \dots , d _ {m}) _ {P}
$$

has entries defined by the formula 

$$
d _ {i} = \left\{ \begin{array}{l l} 0 & \alpha (i) = * \\ \prod_ {j <   \alpha (i)} t _ {j} & \alpha (i) \neq *. \end{array} \right.
$$

We refer to $L _ { f }$ as the layout encoded by $f .$ 

Example 3.2.2.3. The layout encoded by 

$$
((8, 8), (4, 4)) \xrightarrow [ (1 , 4 , 3 , 2) ]{f} (8, 4, 4, 8)
$$

is 

$$
L _ {f} = ((8, 8), (4, 4)): ((1, 1 2 8), (3 2, 8)).
$$

Example 3.2.2.4. The layout encoded by 

$$
(1 2 8, (4, 4, 2)) \xrightarrow [ (3 , 1 , 2 , *) ]{g} ((4, 4), 1 2 8)
$$

is 

$$
L _ {g} = (1 2 8, (4, 4, 2)): (1 6, (1, 4, 0)).
$$

Observation 3.2.2.5. The flattening functor 

$$
\text { Nest } \xrightarrow {(-) ^ {b}} \text { Tuple }
$$

is compatible with flattening of layouts, in that if f is a nested tuple morphism, then 

$$
(L _ {f}) ^ {\flat} = L _ {f ^ {\flat}}.
$$

If $L$ is a tractable layout, then we can construct a nested tuple morphism which encodes $L$ as follows. 

Construction 3.2.2.6. Suppose L is a tractable layout. We define the standard representation of L to be the nested tuple morphism 

$$
f _ {L}: S \to T
$$

where $( f _ { L } ) ^ { \flat } = f _ { L ^ { \flat } }$ <sub>♭</sub> is the standard representation of $L ^ { \flat } , S = { \mathsf { s h a p e } } ( L )$ is the shape of $L ,$ and $T$ is the codomain of $f _ { L ^ { \flat } }$ 

Example 3.2.2.7. If 

$$
L = (3 2, (2, 2)): (1 9 2, (2 4, 3))
$$

then the standard representation of $L$ is 

$$
(3 2, (2, 2)) \xrightarrow [ (6 , 4 , 2) ]{f _ {L}} (3, 2, 4, 2, 4, 3 2).
$$

Lemma 3.2.2.8. If L is a tractable layout, and $f = f _ { L }$ is the standard representation of $L ,$ then 

$$
L _ {f} = L.
$$

Proof. We have 

$$
(L _ {f}) ^ {\flat} = L _ {f ^ {\flat}} = L ^ {\flat}
$$

and 

$$
\operatorname{shape} (L _ {f}) = \operatorname{shape} (L).
$$

Proposition 3.2.2.9. Suppose $L$ is a layout. Then there exists a nested tuple morphism $f$ encoding $L$ $i f$ and only if L is tractable. 

Proof. Suppose first that $L = L _ { f }$ for some nested tuple morphism $f .$ Then $( L _ { f } ) ^ { \flat } = L _ { f ^ { \flat } }$ , and by Proposition 3.1.2.10, we know that $L ^ { \flat }$ is tractable, hence so is $L .$ Conversely, if $L$ is tractable, then we can take $f = f _ { L }$ to be the standard representation of $L ,$ and by Lemma 3.2.2.8, we have $L _ { f } = L . \quad \bigsqcup$ 

In order to establish a one-to-one correspondence between tractable layouts and certain nested tuple morphisms, we introduce the notion of standard form for nested tuple morphisms. 

Definition 3.2.2.10. Suppose $f : S  T$ is a nested tuple morphism. We say $f$ has standard form if 

1. $f ^ { \flat }$ has standard form, as in Definition 3.1.2.12, and 

2. $T$ is flat. 

Example 3.2.2.11. The nested tuple morphism 

$$
((2, 2), (3, 3)) \xrightarrow [ (4 , 6 , 2 , 3 ]{f} (1 0, 3, 3, 2, 1 0, 2)
$$

has standard form. 

Example 3.2.2.12. The nested tuple morphism 

$$
((2, 2), (3, 3)) \xrightarrow [ (4 , 6 , 2 , 3 ]{f} ((1 0, 3, 3), (2, 1 0, 2))
$$

does not have standard form since the codomain of $g$ is not flat. 

Just as in the flat case, we need to exclude non-degenerate nested tuple morphisms and nondegenerate layouts in order to obtain a one-to-one correspondence between nested tuple morphisms of standard form and tractable layouts. To this end, we make the following definition. 

Definition 3.2.2.13. Suppose 

$$
S \xrightarrow [ \alpha ]{f} T
$$

is a nested tuple morphism, and suppose 

$$
L = S: D
$$

is a layout. 

1. We say f is non-degenerate if 

$$
\operatorname{entry} _ {i} (S) = 1 \quad \Rightarrow \quad \alpha (i) = *.
$$

2. We say L is non-degenerate if 

$$
\operatorname{entry} _ {i} (S) = 1 \quad \Rightarrow \quad \operatorname{entry} _ {i} (D) = 0.
$$

Remark 3.2.2.14. If f is a nested tuple morphism, then $f$ is non-degenerate if and only if $f ^ { \flat }$ is non-degenerate. If L is a layout, then L is non-degenerate if and only if $L ^ { \flat }$ is non-degenerate. 

Proposition 3.2.2.15. The maps 

![image](Imgaes/categorical-foundations-cute-layouts-paper/a17f588a85286fcaa472b07cbf359622cddc4aa1ca3895b34532dcc3def61e64.jpg)


$$
\left\{ \begin{array}{c} N o n - d e g e n e r a t e \\ n e s t e d t u p l e m o r p h i s m s \\ o f s t a n d a r d f o r m \end{array} \right\} \longleftrightarrow \left\{ \begin{array}{c} N o n - d e g e n e r a t e \\ t r a c t a b l e l a y o u t s \end{array} \right\}
$$

![image](Imgaes/categorical-foundations-cute-layouts-paper/3542866dadbdea60b846dca05941d9207df40ba143754fcb6995a1d8cefb5891.jpg)


of Constructions 3.2.2.2 and 3.2.2.6 determine a one-to-one correspondence between nested tuple morphisms of standard form, and tractable layouts. 

Proof. We have already shown in Proposition 3.2.2.9 that if L is a tractable layout and $f = f _ { L }$ is the standard form of $L ,$ , then $L _ { f } = L$ . Suppose next that $f$ has standard form, and let $L = L _ { f }$ be the layout encoded by $f .$ We want to show that $f$ is equal to the standard representation $f _ { L }$ of $L .$ . By Proposition 3.1.2.21, we know that $f ^ { \flat }$ is equal to the standard representation $f _ { L ^ { \flat } }$ of $L ^ { \flat }$ , and since 

$$
\operatorname{domain} (f) = \operatorname{shape} (L) = \operatorname{domain} \left(f _ {L}\right),
$$

and 

$$
\operatorname{codomain} (f) = \operatorname{codomain} \left(f ^ {\flat}\right) = \operatorname{codomain} \left(f _ {L ^ {\flat}}\right) = \operatorname{codomain} \left(f _ {L}\right),
$$

we deduce that $f = f _ { L }$ 

## 3.2.3 Examples

In this section, we list some important families of nested tuple morphisms. 

Example 3.2.3.1 (Reparenthesizations). Suppose $S _ { 1 }$ and $S _ { 2 }$ are nested tuples with the same flattening 

$$
S _ {1} ^ {\flat} = S _ {2} ^ {\flat}.
$$

Then there is a reparenthesization isomorphism 

$$
\mathsf {i d} _ {S _ {1}} ^ {S _ {2}}: S _ {1} \xrightarrow {\cong} S _ {2}
$$

lying over the identity. These morphisms are transitive, in that 

$$
\mathsf {i d} _ {S _ {2}} ^ {S _ {3}} \circ \mathsf {i d} _ {S _ {1}} ^ {S _ {2}} = \mathsf {i d} _ {S _ {1}} ^ {S _ {3}},
$$

and compatible with identities, in that 

$$
\mathrm{id} _ {S} ^ {S} = \mathrm{id} _ {S}.
$$

If $\boldsymbol { f } = \mathrm { i d } _ { S _ { 1 } } ^ { S _ { 2 } }$ is a reparenthesization isomorphism, then $L _ { f }$ is the column major layout with shape $S _ { 1 }$ 

Example 3.2.3.2 (Flattenings). As a special case of the previous example, if $S$ is any nested tuple, then we have a flattening isomorphism 

$$
\mathrm{id} _ {S} ^ {S ^ {\flat}}: S \xrightarrow {\cong} S ^ {\flat}
$$

and an unflattening isomorphism 

$$
\mathrm{id} _ {S ^ {\flat}} ^ {S}: S ^ {\flat} \xrightarrow {\cong} S
$$

Observation 3.2.3.3. If $f : S  T$ is a nested tuple, then $f$ is equal to the composite 

$$
S \xrightarrow {\mathrm{id} _ {S} ^ {S ^ {b}}} S ^ {b} \xrightarrow {f ^ {b}} T ^ {b} \xrightarrow {\mathrm{id} _ {T ^ {b}} ^ {T}} T.
$$

In other words, we have a canonical factorization 

$$
f = \mathsf {i d} _ {T ^ {\flat}} ^ {T} \circ f ^ {\flat} \circ \mathsf {i d} _ {S} ^ {S ^ {\flat}}.
$$

Example 3.2.3.4 (Entries). Suppose 

$$
S \xrightarrow [ \alpha ]{f} T
$$

is a nested tuple morphism. Suppose $1 \leq i \leq \mathsf { l e n } ( S )$ , and write $j = \alpha ( i )$ . Then we refer to the nested tuple morphism 

$$
\operatorname{entry} _ {i} (S) \xrightarrow [ (j) ]{\operatorname{entry} _ {i} (f)} T
$$

as the ith entry of $f .$ The layout encoded by $\mathsf { e n t r y } _ { i } ( f )$ is 

$$
L _ {\text { entry } _ {i} (f)} = \text { entry } _ {i} (L _ {f}).
$$

Example 3.2.3.5 (Entry inclusions). As a special case of the previous example, if $S$ is a nested tuple and $1 \leq i \leq \mathsf { l e n } ( S )$ , we can take $f = \operatorname { i d } _ { S }$ , in which case 

$$
\operatorname{entry} _ {i} \left(\operatorname{id} _ {S}\right): \operatorname{entry} _ {i} (S) \longrightarrow S
$$

is the inclusion of the ith entry of $S .$ 

Example 3.2.3.6 (Modes). Suppose 

$$
S \xrightarrow [ \alpha ]{f} T
$$

is a nested tuple morphism. Suppose $1 \leq i \leq \mathsf { r a n k } ( S )$ and, write 

$$
\begin{array}{c} N = \mathsf {l e n} _ {<   i} (S) \\ \ell = \mathsf {l e n} _ {i} (S). \end{array}
$$

Then we refer to the nested tuple morphism 

$$
\operatorname{mode} _ {i} (S) \xrightarrow [ (N + 1 , \dots , N + \ell) ]{\operatorname{mode} _ {i} (f)} T
$$

as the ith mode of S. The layout encoded by mode $\left( L _ { f } \right)$ is 

$$
L _ {\text { mode } _ {i} (f)} = \text { mode } _ {i} (L _ {f}).
$$

Example 3.2.3.7 (Mode inclusions). As a special case of the previous example we may take $f = \operatorname { i d } _ { S }$ ， in which case 

$$
\operatorname{mode} _ {i} (\operatorname{id} _ {S}): \operatorname{mode} _ {i} (S) \to S
$$

is the inclusion of the ith mode of S. We sometime denote this map by 

$$
\operatorname{incl} _ {i} (S) = \operatorname{mode} _ {i} (\operatorname{id} _ {S}).
$$

## 3.2.4 Realization of nested tuple morphisms

In the flat case, we constructed a realization functor 

$$
\text { Tuple } \xrightarrow {| \cdot |} \text { FinSet }
$$

which sends a tuple morphism $f$ to the layout function of $L _ { f }$ . We can extend this to a realization functor 

$$
\text { Nest } \xrightarrow {| \cdot |} \text { FinSet }
$$

by precomposing with the flattening functor Nest → Tuple. 

Definition 3.2.4.1. We define the realization functor 

$$
\text { Nest } \xrightarrow {| \cdot |} \text { FinSet }
$$

to be the composite 

$$
\mathbf {N e s t} \xrightarrow {(-) ^ {b}} \mathbf {T u p l e} \xrightarrow {| \cdot |} \mathbf {F i n S e t}
$$

Lemma 3.2.4.2. $I f f : S \to T$ is a nested tuple morphism, then the realization $| f |$ of $f$ is the layout function of $L _ { f } .$ 

$$
| f | = \Phi_ {L _ {f}} ^ {\mathrm{size} (T)}.
$$

Proof. This follows immediately from 3.1.4.5, since 

$$
| f | = | f ^ {\flat} | = \Phi_ {L _ {f ^ {\flat}}} ^ {\text { size } (T)} = \Phi_ {L _ {f}} ^ {\text { size } (T)}
$$

## 3.2.5 Refinements

In this section, we revisit the refinement of nested tuples from a categorical perspective. Recall from section 2.2.4 that a nested tuple $S ^ { \prime }$ refines $S ,$ , denoted 

$$
S ^ {\prime} \longrightarrow S
$$

if $S ^ { \prime }$ may be obtained from $S$ by replacing each entry of $S$ with some nested tuple of the same size. For example, 

$$
(2, (2, 2)) \twoheadrightarrow 8,
$$

and 

$$
((2, 2), (3, 3), (5, 5)) \twoheadrightarrow (4, 9, 2 5).
$$

If len $| ( S ) = m$ and $\mathsf { p r o f } ( S ) = P$ , then we can write 

$$
S ^ {\prime} = (S _ {1} ^ {\prime}, \ldots , S _ {m} ^ {\prime}) _ {P}
$$

as the P-substitution of the relative modes 

$$
S _ {i} ^ {\prime} = \operatorname{mode} _ {i} (S ^ {\prime}, S).
$$

We refer to the ordinary concatenation 

$$
(S _ {1} ^ {\prime}, \ldots , S _ {m} ^ {\prime}) = \mathsf {f l a t} (S ^ {\prime}, S)
$$

as the flattening of $S ^ { \prime }$ relative to $S .$ 

Let Ref denote the poset category of nested tuples of positive integers under refinement, so that a morphism in Ref is a refinement $S ^ { \prime }  S$ . If S is a nested tuple, let 

$$
\mathbf {R e f} (S) = \{S ^ {\prime} \mid S ^ {\prime} \text {   refines   } S \}
$$

denote the poset of nested tuples refining S. Equivalently, Ref(S) is the slice category $\mathsf { R e f } _ { / S }$ 

Construction 3.2.5.1. [Relative mode inclusions] Suppose $S ^ { \prime } \to S$ is a refinement, and write 

$$
S _ {i} ^ {\prime} = \operatorname{mode} _ {i} (S ^ {\prime}, S)
$$

for the modes of $S ^ { \prime }$ relative to $S .$ Then $S ^ { \prime }$ and $\left( S _ { 1 } ^ { \prime } , \ldots , S _ { m } ^ { \prime } \right)$ have the same flattening, so we have a reparenthesization isomorphism 

$$
\mathsf {i d} _ {(S _ {1} ^ {\prime}, \dots , S _ {m} ^ {\prime})} ^ {S ^ {\prime}}: (S _ {1} ^ {\prime}, \dots , S _ {m} ^ {\prime}) \xrightarrow {\cong} S ^ {\prime}
$$

and we define 

$$
\operatorname{incl} _ {i} (S ^ {\prime}, S): S _ {i} ^ {\prime} \to S ^ {\prime}
$$

to be the composite 

$$
S _ {i} ^ {\prime} \xrightarrow {\operatorname{incl} _ {i} ((S _ {1} ^ {\prime} , \ldots , S _ {m} ^ {\prime}))} (S _ {1} ^ {\prime}, \ldots , S _ {m} ^ {\prime}) \xrightarrow {\operatorname{id} _ {(S _ {1} ^ {\prime} , \ldots , S _ {m} ^ {\prime})} ^ {S ^ {\prime}}} S ^ {\prime}
$$

of the ith mode inclusion of $\left( S _ { 1 } ^ { \prime } , \ldots , S _ { m } ^ { \prime } \right)$ with the reparenthesization isomorphism $( S _ { 1 } ^ { \prime } , \ldots , S _ { m } ^ { \prime } ) \cong S ^ { \prime }$ Example 3.2.5.2. If $S = ( 4 , ( 9 , 2 5 ) )$ and $S ^ { \prime } = ( ( 2 , 2 ) , ( ( 3 , 3 ) , 2 5 ) )$ ), then $S ^ { \prime }$ refines $S ,$ and $\mathsf { i n c l } _ { 2 } ( S ^ { \prime } , S )$ is the nested tuple morphism 

$$
(3, 3) \xrightarrow [ (3 , 4) ]{\operatorname{incl} _ {2} (S ^ {\prime} , S)} ((2, 2), ((3, 3), 2 5)).
$$

Construction 3.2.5.3. [Relative modes] Suppose $f ^ { \prime } : S ^ { \prime } \to T ^ { \prime }$ is a nested tuple morphism, and suppose $S ^ { \prime }$ refines S. We define the ith mode of $f ^ { \prime }$ relative to S, denoted 

$$
\operatorname{mode} _ {i} \left(f ^ {\prime}, S\right) = f ^ {\prime} \circ \operatorname{incl} _ {i} \left(S ^ {\prime}, S\right): S _ {i} ^ {\prime} \rightarrow T ^ {\prime}
$$

to be the composite 

$$
S _ {i} ^ {\prime} \xrightarrow {\operatorname{incl} _ {i} (S ^ {\prime} , S)} S ^ {\prime} \xrightarrow {f ^ {\prime}} T ^ {\prime}
$$

In particular, we have 

$$
\operatorname{mode} _ {i} \left(\operatorname{id} _ {S ^ {\prime}}, S\right) = \operatorname{incl} _ {i} \left(S ^ {\prime}, S\right).
$$

Example 3.2.5.4. Suppose $S = ( 4 , ( 9 , 2 5 ) )$ and $S ^ { \prime } = ( ( 2 , 2 ) , ( ( 3 , 3 ) , 2 5 ) )$ , so that $S ^ { \prime }$ refines S. If $f ^ { \prime }$ is the nested tuple morphism 

$$
((2, 2), ((3, 3), 2 5)) \xrightarrow [ (1 , 3 , 2 , * , 4) ]{f ^ {\prime}} (2, 3, 2, 2 5).
$$

then mode $_ 2 ( f ^ { \prime } , S )$ is the nested tuple morphism 

$$
(3, 3) \xrightarrow [ (2 , *) ]{\operatorname{mode} _ {2} \left(f ^ {\prime} , S\right)} (2, 3, 2, 2 5).
$$

Construction 3.2.5.5 (Pullbacks). Suppose $f : S  T$ is a nested tuple morphism lying over $\alpha ,$ and suppose $T ^ { \prime } \to T$ is a refinement. Let 

$$
T _ {j} ^ {\prime} = \operatorname{mode} _ {j} (T ^ {\prime}, T)
$$

denote the jth mode of $T ^ { \prime }$ relative to $T ,$ and for any $1 \leq i \leq \mathsf { l e n } ( S )$ , set 

$$
S _ {i} ^ {\prime} = \left\{ \begin{array}{l l} \text {entry} _ {i} (S) & \alpha (i) = * \\ T _ {j} ^ {\prime} & \alpha (i) = j. \end{array} \right.
$$

We define the pullback of $T ^ { \prime }$ along f to be the nested tuple 

$$
S ^ {\prime} = f ^ {*} T ^ {\prime} = \operatorname{sub} (S, (S _ {1} ^ {\prime}, \dots , S _ {m} ^ {\prime})).
$$

For any $1 \leq i \leq m$ , let 

$$
f _ {i} ^ {\prime}: S _ {i} ^ {\prime} \to T ^ {\prime}
$$

be the trivial map if $\alpha ( i ) = *$ , and the inclusion 

$$
\operatorname{incl} _ {j} (T ^ {\prime}, T): S _ {i} ^ {\prime} = T _ {j} ^ {\prime} \to T ^ {\prime}
$$

if $\alpha ( i ) = j$ . The maps $f _ { 1 } ^ { \prime } , \ldots , f _ { m } ^ { \prime }$ have disjoint images, so we form the concatenation 

$$
(f _ {1} ^ {\prime}, \ldots , f _ {m} ^ {\prime}): (S _ {1} ^ {\prime}, \ldots , S _ {m} ^ {\prime}) \to T ^ {\prime}.
$$

We define $f ^ { \prime } = T ^ { \prime \ast } f$ to be the composite 

$$
S ^ {\prime} \xrightarrow {\mathsf {i d} _ {S ^ {\prime}} ^ {(S _ {1} ^ {\prime} , \ldots , S _ {m} ^ {\prime})}} (S _ {1} ^ {\prime}, \ldots , S _ {m} ^ {\prime}) \xrightarrow {(f _ {1} ^ {\prime} , \ldots , f _ {m} ^ {\prime})} T ^ {\prime}.
$$

We refer to $f ^ { \prime }$ as the pullback of f along $T ,$ and depict such a pullback as a square 

$$
\begin{array}{c c c} S ^ {\prime} & \xrightarrow {f ^ {\prime}} & T ^ {\prime} \\ \Big \downarrow & \searrow & \Big \downarrow \\ S & \xrightarrow {f} & T. \end{array}
$$

Example 3.2.5.6. Suppose $f : ( 6 4 , 3 2 )  ( 4 , 6 4 , 4 , 3 2 )$ lies over $\alpha = ( 2 , 4 )$ . Then we have a pullback square 

$$
\begin{array}{c} ((1 6, 4), (1 6, 2)) \xrightarrow {f ^ {\prime}} ((2, 2), (1 6, 4), (2, 2), (1 6, 2)) \\ \Big \downarrow \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \Big \downarrow \\ (6 4, 3 2) \xrightarrow {f} (4, 6 4, 4, 3 2) \end{array}
$$

where $f ^ { \prime }$ lies over $\alpha ^ { \prime } = ( 3 , 4 , 7 , 8 )$ 

Example 3.2.5.7. Suppose $S$ is a nested tuple with flattening 

$$
S ^ {\flat} = (s _ {1}, \dots , s _ {m}),
$$

and suppose $S ^ { \prime }  S$ is a refinement with relative flattening 

$$
(S _ {1} ^ {\prime}, \ldots , S _ {m} ^ {\prime}).
$$

Then the pullback of $S ^ { \prime } \to S$ along the unflattening isomorphism 

$$
\mathsf {i d} _ {(s _ {1}, \dots , s _ {m})} ^ {S}: (s _ {1}, \dots , s _ {m}) \to S
$$

is the reparenthesization isomorphism 

$$
\begin{array}{c} (S _ {1} ^ {\prime}, \ldots , S _ {m} ^ {\prime}) \xrightarrow {\mathsf {i d} _ {(S _ {1} ^ {\prime} , \ldots , S _ {m} ^ {\prime})} ^ {S ^ {\prime}}} S ^ {\prime} \\ \Big \downarrow \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \Big \downarrow \\ (s _ {1}, \ldots , s _ {m}) \xrightarrow {\mathsf {i d} _ {(s _ {1} , \ldots , s _ {m})} ^ {S}} S. \end{array}
$$

Example 3.2.5.8. Suppose $S ^ { \prime }  S$ is a refinement, and consider the ith entry inclusion 

$$
s _ {i} \rightarrow S.
$$

Then the pullback of $S ^ { \prime } \to S$ along $s _ { i } \to S$ is the ith relative mode inclusion 

$$
\begin{array}{c} S _ {i} ^ {\prime} \xrightarrow {\text {incl} _ {i} (S ^ {\prime} , S)} S ^ {\prime} \\ \Big \downarrow \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \end{array}
$$

Observation 3.2.5.9. The pullback construction above specifies a contravariant functor 

$$
\mathbf {N e s t} ^ {\mathrm{op}} \longrightarrow \mathbf {C a t}
$$

$$
\begin{array}{c} S \longmapsto \mathbf {R e f} (S) \\ f \Big \downarrow \qquad \qquad \qquad \qquad \qquad \qquad \qquad \uparrow_ {f ^ {*}} \\ T \longmapsto \mathbf {R e f} (T) \end{array}
$$

$$
\begin{array}{c} f ^ {*} T ^ {\prime} \leftarrow f ^ {*} T ^ {\prime \prime} \\ \Big \uparrow \\ T ^ {\prime} \leftarrow T ^ {\prime \prime} \end{array}
$$

The key property of pullbacks is that the layout function of $f ^ { \prime }$ is equal to that of $f .$ 

Lemma 3.2.5.10. Suppose 

$$
\begin{array}{c c c} S ^ {\prime} & \xrightarrow {f ^ {\prime}} & T ^ {\prime} \\ \Big \downarrow & \searrow & \Big \downarrow \\ S & \xrightarrow {f} & T \end{array}
$$

is a pullback square, where f lies over α. Let 

$$
f _ {i} ^ {\prime}: S _ {i} ^ {\prime} \to T
$$

denote the ith mode of f<sup>′</sup> relative to $S _ { i }$ , and let 

$$
(L _ {f}) ^ {\flat} = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m}).
$$

Then for any $1 \leq i \leq m$ , we have 

$$
\operatorname{coal} (L _ {f _ {i} ^ {\prime}}) = s _ {i}: d _ {i}.
$$

Proof. Suppose $1 \leq i \leq m$ . If $\alpha ( i ) = *$ , then $f _ { i } ^ { \prime }$ is the trivial map, so 

$$
L _ {f _ {i} ^ {\prime}} = s _ {i}: 0 = s _ {i}: d _ {i}.
$$

In particular, coal $( L _ { f _ { i } ^ { \prime } } ) = s _ { i } : 0 = s _ { i } : d _ { i }$ . Suppose next that $\alpha ( i ) = j \neq *$ . By construction of $f ^ { \prime } { } _ { ; }$ , we have that 

$$
f _ {i} ^ {\prime} = \operatorname{incl} _ {j} (T ^ {\prime}, T): T _ {j} ^ {\prime} \to T ^ {\prime}.
$$

which lies over the map $\alpha _ { i } ^ { \prime }$ given by $t \mapsto \mathsf { l e n } _ { < j } ( T ^ { \prime } , T ) + t .$ . For each $1 \leq t < \mathsf { l e n } ( T _ { i } ^ { \prime } )$ , we have $\alpha _ { i } ^ { \prime } ( t ) = \alpha _ { i } ^ { \prime } ( t + 1 )$ , so $L _ { f _ { i } ^ { \prime } }$ is a column major layout with size size $( T _ { j } ^ { \prime } ) = t _ { j } = s _ { i }$ . This implies that coal $\left( L _ { f _ { i } ^ { \prime } } \right)$ is a depth 0 layout of the form 

$$
\operatorname{coal} (L _ {f _ {i} ^ {\prime}}) = s _ {i}: e
$$

for some integer $e \geq 0$ . We claim that $e = d _ { i }$ . If we write $t _ { j ^ { \prime } } ^ { \prime } = \mathsf { e n t r y } _ { j ^ { \prime } } ( T ^ { \prime } )$ , then we have 

$$
\begin{array}{l} e = \mathsf {e n t r y} _ {1} (\mathsf {s t r i d e} (L _ {f _ {i} ^ {\prime}})) = \prod_ {j ^ {\prime} <   \alpha_ {i} ^ {\prime} (1)} t _ {j ^ {\prime}} ^ {\prime} \\ \qquad = \prod_ {j ^ {\prime} \leq \mathsf {l e n} _ {<   j} (T ^ {\prime}, T)} t _ {j ^ {\prime}} ^ {\prime} \\ \qquad = \prod_ {j ^ {\prime} <   j} \mathsf {s i z e} (T _ {j ^ {\prime}} ^ {\prime}) \\ \qquad = \prod_ {j ^ {\prime} <   j} t _ {j ^ {\prime}} \\ \qquad = d _ {i}. \end{array}
$$

Proposition 3.2.5.11. If 

$$
\begin{array}{c c c} S ^ {\prime} & \xrightarrow {f ^ {\prime}} & T ^ {\prime} \\ \Big \downarrow & \searrow & \Big \downarrow \\ S & \xrightarrow {f} & T \end{array}
$$

is a pullback square, then $\Phi _ { L _ { f } } = \Phi _ { L _ { f ^ { \prime } } }$ 

Proof. We begin by fixing notation. Let $m = \mathsf { l e n } ( S )$ , and let 

$$
\begin{array}{c} S ^ {\flat} = (s _ {1}, \ldots , s _ {m}), \\ S _ {i} ^ {\prime} = \mathsf {m o d e} _ {i} (S ^ {\prime}, S), \\ T _ {j} ^ {\prime} = \mathsf {m o d e} _ {j} (T ^ {\prime}, T), \\ (L _ {f}) ^ {\flat} = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m}). \end{array}
$$

Consider the reparenthesization isomorphism 

$$
\mathsf {i d} _ {(S _ {1} ^ {\prime}, \dots , S _ {m} ^ {\prime})} ^ {S ^ {\prime}}: (S _ {1} ^ {\prime}, \dots , S _ {m} ^ {\prime}) \to S ^ {\prime}
$$

The composite of this map with $f ^ { \prime }$ is the concatenation $\left( f _ { 1 } ^ { \prime } , \ldots , f _ { m } ^ { \prime } \right)$ where $f _ { i } ^ { \prime }$ is the trivial map if $\alpha ( i ) = *$ , and the relative mode inclusion 

$$
\operatorname{incl} _ {i} (T ^ {\prime}, T): S _ {i} ^ {\prime} = T _ {j} ^ {\prime} \to T ^ {\prime}
$$

otherwise. Using Lemma 3.2.5.10, and the fact that $L _ { f ^ { \prime } } = L _ { ( f _ { 1 } ^ { \prime } , \dots , f _ { m } ^ { \prime } ) }$ , we compute 

$$
\begin{array}{l} \mathsf {c o a l} (L _ {f ^ {\prime}}) = \mathsf {c o a l} (L _ {(f _ {1} ^ {\prime}, \ldots , f _ {m} ^ {\prime})}) \\ \qquad = \mathsf {c o a l} ((L _ {f _ {1} ^ {\prime}}, \ldots , L _ {f _ {m} ^ {\prime}})) \\ \qquad = \mathsf {c o a l} ((\mathsf {c o a l} (L _ {f _ {1} ^ {\prime}}), \ldots , \mathsf {c o a l} (L _ {f _ {m} ^ {\prime}}))) \\ \qquad = \mathsf {c o a l} ((s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m})) \\ \qquad = \mathsf {c o a l} (L _ {f}). \end{array}
$$

By Proposition 2.3.3.14, we deduce that Φ $\mathbf { \Phi } _ { L _ { f ^ { \prime } } } = \Phi _ { L _ { f } }$ 

Construction 3.2.5.12 (Pushforwards). Suppose $f : S  T$ is a nested tuple morphism lying over $\alpha ,$ , and suppose $S ^ { \prime }  S$ is a refinement. Let 

$$
S _ {i} ^ {\prime} = \operatorname{mode} _ {i} (S ^ {\prime}, S)
$$

denote the ith mode of $S ^ { \prime }$ relative to $S ,$ and for any $1 \leq j \leq \mathsf { l e n } ( T )$ , set 

$$
T _ {j} ^ {\prime} = \left\{ \begin{array}{l l} \mathsf {e n t r y} _ {j} (T) & j \notin \mathsf {I m a g e} (\alpha) \\ S _ {i} ^ {\prime} & \alpha (i) = j. \end{array} \right.
$$

We define the pushforward of $S ^ { \prime }$ along $f$ to be the nested tuple 

$$
T ^ {\prime} = f _ {*} S ^ {\prime} = \operatorname{sub} (T, (T _ {1} ^ {\prime}, \dots , T _ {n} ^ {\prime})).
$$

For any $1 \leq i \leq m$ , let 

$$
f _ {i} ^ {\prime}: S _ {i} ^ {\prime} \to T ^ {\prime}
$$

be the trivial map if $\alpha ( i ) = *$ , and the relative mode inclusion 

$$
\operatorname{incl} _ {j} \left(T ^ {\prime}, T\right): S _ {i} ^ {\prime} = T _ {j} ^ {\prime} \rightarrow T ^ {\prime}
$$

if $\alpha ( i ) = j$ . The morphisms $f _ { 1 } ^ { \prime } , \ldots , f _ { m } ^ { \prime }$ have disjoint images, so we can form the concatenation 

$$
(f _ {1} ^ {\prime}, \ldots , f _ {m} ^ {\prime}): (S _ {1} ^ {\prime}, \ldots , S _ {m} ^ {\prime}) \to T ^ {\prime}.
$$

We define $f ^ { \prime } = S _ { \ast } ^ { \prime } f$ to be the composite 

$$
S ^ {\prime} \xrightarrow {\mathsf {i d} _ {S ^ {\prime}} ^ {(S _ {1} ^ {\prime} , \ldots , S _ {m} ^ {\prime})}} (S _ {1} ^ {\prime}, \ldots , S _ {m} ^ {\prime}) \xrightarrow {(f _ {1} ^ {\prime} , \ldots , f _ {m} ^ {\prime})} T ^ {\prime}.
$$

We refer to $f ^ { \prime }$ as the pushforward of f along T. We depict such a pushforward as 

$$
\begin{array}{c c c} S ^ {\prime} & \xrightarrow {f ^ {\prime}} & T ^ {\prime} \\ \Big \downarrow & & \Big \downarrow \\ S & \xrightarrow {f} & T \end{array}
$$

Example 3.2.5.13. If $f : ( 6 4 , 3 2 )  ( 4 , 6 4 , 4 , 3 2 )$ lies over $\alpha = ( 2 , 4 )$ , then we have a pushforward square 

$$
\begin{array}{c} ((1 6, 4), (1 6, 2)) \xrightarrow {f ^ {\prime}} (4, (1 6, 4), 4, (1 6, 2)) \\ \Big \downarrow \\ (6 4, 3 2) \xrightarrow [ f ]{} (4, 6 4, 4, 3 2) \end{array}
$$

The key property of pullbacks is that the layout function of $f ^ { \prime }$ is equal to that of $f .$ 

Lemma 3.2.5.14. Suppose 

$$
\begin{array}{c} S ^ {\prime} \xrightarrow {f ^ {\prime}} T ^ {\prime} \\ \Big \downarrow \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \Big \downarrow \\ S \xrightarrow {f} T \end{array}
$$

is a pushforward square, where $f$ lies over α. Let 

$$
f _ {i} ^ {\prime}: S _ {i} ^ {\prime} \to T
$$

denote the ith mode of $f ^ { \prime }$ relative to S, and let 

$$
(L _ {f}) ^ {\flat} = (s _ {1}, \ldots , s _ {m}): (d _ {1}, \ldots , d _ {m}).
$$

Then for any $1 \leq i \leq m$ , we have 

$$
\operatorname{coal} (L _ {f _ {i} ^ {\prime}}) = s _ {i}: d _ {i}.
$$

Proof. The proof is identical to that of Lemma 3.2.5.10 

Proposition 3.2.5.15. If 

$$
\begin{array}{c} S ^ {\prime} \xrightarrow {f ^ {\prime}} T ^ {\prime} \\ \Big \downarrow \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \Big \downarrow \\ S \xrightarrow {f} T \end{array}
$$

is a pushforward square, then Φ $\mathbf { \Phi } _ { L _ { f } } = \Phi _ { L _ { f ^ { \prime } } }$ 

Proof. The proof is identical to that of Proposition 3.2.5.11. 

Observation 3.2.5.16. The pushforward construction defined above specifies a covariant functor 

Nest Cat 

$$
\begin{array}{c c} S \longmapsto \mathbf {R e f} (S) & \qquad S ^ {\prime \prime} \to S ^ {\prime} \\ \Big \downarrow_ {f} \qquad \qquad \Big \downarrow_ {f _ {*}} \\ T \longmapsto \mathbf {R e f} (T) & \qquad f _ {*} S ^ {\prime \prime} \to f _ {*} S ^ {\prime} \end{array}
$$

Observation 3.2.5.17. If $f : S  T$ is an isomorphism of nested tuples, then 

$$
\operatorname{Ref} (T) \xrightarrow {f ^ {*}} \operatorname{Ref} (S)
$$

and 

$$
\mathbf {R e f} (S) \xrightarrow {f _ {*}} \mathbf {R e f} (T)
$$

are inverse isomorphisms of categories. Specifically, 

$$
(f ^ {- 1}) ^ {*} = f _ {*} \quad \text { and } \quad (f ^ {- 1}) _ {*} = f ^ {*}.
$$

Observation 3.2.5.18. If $S _ { 1 }$ and $S _ { 2 }$ are nested tuples with $\mathsf { f l a t } ( S _ { 1 } ) = \mathsf { f l a t } ( S _ { 2 } )$ , then there is a canonical nested tuple isomorphism $S _ { 1 } \cong S _ { 2 }$ , and hence, a canonical isomorphism of categories 

$$
\mathbf {R e f} (S _ {1}) \cong \mathbf {R e f} (S _ {2}).
$$

There is one more concept we need to specify, called mutual refinements. The importance of this concept will be come clear in Chapter 4, when we use this concept in our layout composition algorithm. 

Definition 3.2.5.19. Suppose $T$ and U are nested tuples. A mutual refinement of $( T , U )$ is a diagram of the form 

![image](Imgaes/categorical-foundations-cute-layouts-paper/cc1b31a6974bd0438b26bd4d4bd636df42891252b07223ac829b12291b50fff4.jpg)


Explicitly, this is a pair of nested tuples $( T ^ { \prime } , U ^ { \prime } )$ such that 

1. $T ^ { \prime }$ refines $T ,$ 

2. $U ^ { \prime }$ refines U, and 

3. $T ^ { \prime }$ divides $U ^ { \prime } .$ 

In addition to the definition of mutual refinements, we need the following fact. 

Lemma 3.2.5.20. Suppose T and U are nested tuples. Then there is a one-to-one correspondence between mutual refinements of $( T , U )$ , and mutual refinements of $( T ^ { \flat } , U ^ { \flat } )$ 

Proof. If $( T ^ { \prime } , U ^ { \prime } )$ is a mutual refinement of $( T , U )$ , then pulling back along the unflattening isomorphisms $\mathsf { i d } _ { T ^ { \flat } } ^ { T }$ <sub>♭</sub> and $\mathsf { i d } _ { U ^ { \flat } } ^ { U }$ yields a mutual refinement 

$$
\begin{array}{c c} (\mathsf {i d} _ {T ^ {\flat}} ^ {T}) ^ {*} T ^ {\prime} & \longrightarrow (\mathsf {i d} _ {U ^ {\flat}} ^ {U}) ^ {*} U ^ {\prime} \\ \Big \downarrow & \Big \downarrow \\ T ^ {\flat} & U ^ {\flat} \end{array}
$$

of $( T ^ { \flat } , U ^ { \flat } )$ . Conversely, if $( ( T ^ { \flat } ) ^ { \prime } , ( U ^ { \flat } ) ^ { \prime } )$ is a mutual refinement of $T ^ { \flat } , U ^ { \flat }$ , then pulling back along the flattening isomorphisms ${ \mathrm { i d } } _ { T } ^ { T ^ { \flat } }$ and $\mathsf { i d } _ { U } ^ { U ^ { \flat } }$ yields a mutual refinement 

$$
\begin{array}{c c} (\mathsf {i d} _ {T} ^ {T ^ {\flat}}) ^ {*} (T ^ {\flat}) ^ {\prime} \longmapsto (\mathsf {i d} _ {U} ^ {U ^ {\flat}}) ^ {*} (U ^ {\flat}) ^ {\prime} \\ \Big \downarrow & \Big \downarrow \\ T & U \end{array}
$$

of $( T ^ { \flat } , U ^ { \flat } )$ 

## 3.2.6 Operations on nested tuple morphisms

Our next task is to develop an “algebra of nested tuple morphisms”. Since we have already developed such an “algebra” for tuple morphisms, we can extend to the nested case by equipping the outputs of our various operations with an appropriate profile. 

## 3.2.6.1 Concatenate

Next, we define a concatenation operation on nested tuple morphisms, which is compatible with concatenation of layouts, in that 

$$
L _ {(f, g)} = (L _ {f}, L _ {g}).
$$

We concatenate nested tuple morphisms $f$ and $g$ by concatenating the domains of $f$ and $g .$ In order for this to be well-defined, we need $f$ and g to satisfy a disjointness condition, which we specify below. 

Definition 3.2.6.1. Suppose $f$ and $g$ are nested tuple morphisms with the same codomain. We say $f$ and $g$ have disjoint images if $f ^ { \flat }$ and $g ^ { \flat }$ have disjoint images, as in Definition 3.1.5.32. 

Example 3.2.6.2. If 

$$
f: (3, (5 1 2, 5 1 2)) \to (2, 5 1 2, 2, 5 1 2)
$$

lies over (∗, 2, 4) and 

$$
g: (2, 2) \to (2, 5 1 2, 2, 5 1 2)
$$

lies over (1, 3), then $f$ and g have disjoint images. 

Example 3.2.6.3. If 

$$
f: (2, (3 2, 6 4)) \rightarrow (3 2, (2, 2, 2), 6 4)
$$

lies over $\alpha = ( 3 , 1 , 5 )$ and 

$$
g: ((2, 2)) \to (3 2, (2, 2, 2), 6 4)
$$

lies over $\beta = ( 2 , 4 )$ , then $f$ and $g$ have disjoint images. 

Construction 3.2.6.4. Suppose $f : S  T$ and $g : U \to T$ are nested tuple morphisms lying over α and $\beta ,$ , respectively, and that $f$ and $g$ have disjoint images. We define the concatenation of $f$ and $g$ to be the nested tuple morphism 

$$
(f, g): (S, U) \to T
$$

with 

$$
\operatorname{flat} ((f, g)) = f ^ {\flat} \star g ^ {\flat}.
$$

More generally, if $f _ { i } : S _ { i }  T$ are nested tuple morphisms for $1 \leq i \leq k$ , and $f _ { 1 } , \ldots , f _ { k }$ have pairwise disjoint images, then we define the concatenation 

$$
(f _ {1}, \dots , f _ {k}): (S _ {1}, \dots , S _ {k}) \to T.
$$

to be the nested tuple morphism with 

$$
(f _ {1}, \dots , f _ {k}) ^ {\flat} = f _ {1} ^ {\flat} \star \dots \star f _ {k} ^ {\flat}.
$$

Example 3.2.6.5. The concatenation of the morphisms $f$ and $g$ of Example 3.2.6.2 is the nested tuple morphism 

$$
(f, g): ((3, (5 1 2, 5 1 2)), (2, 2)) \rightarrow (2, 5 1 2, 2, 5 1 2)
$$

lying over α ⋆ $\beta = ( * , 2 , 4 , 1 , 3 )$ 

Example 3.2.6.6. The concatenation of the morphisms $f$ and g of Example 3.2.6.3 is the nested tuple morphism 

$$
(f, g): ((2, (3 2, 6 4)), ((2, 2))) \to (3 2, (2, 2, 2), 6 4)
$$

lying over α ⋆ $\beta = ( 3 , 1 , 5 , 2 , 4 )$ 

Example 3.2.6.7. If 

$$
f: (2, 2) \to (2, 3, 5, 2, 3, 5)
$$

lies over $\alpha = ( 1 , 4 )$ 

$$
g: (3, 3) \to (2, 3, 5, 2, 3, 5)
$$

lies over $\beta = ( 2 , 5 )$ , and 

$$
h: (5, 5) \to (2, 3, 5, 2, 3, 5)
$$

lies over $\gamma = ( 3 , 6 )$ , then $f , g$ and h have pairwise disjoint images, and the concatenation 

$$
(f, g, h): ((2, 2), (3, 3), (5, 5)) \to (2, 3, 5, 2, 3, 5)
$$

lies over $\alpha \star \beta \star \gamma = ( 1 , 4 , 2 , 5 , 3 , 6 )$ 

Example 3.2.6.8. Suppose $f : S  T$ is a nested tuple morphism, and suppose 

$$
S ^ {\flat} = (s _ {1}, \dots , s _ {m}).
$$

Recall from example 3.2.3.4 that for any $1 \leq i \leq m$ , there is a nested tuple morphism 

$$
f _ {i}: s _ {i} \to T.
$$

called the ith entry of $f .$ These morphisms have pairwise disjoint images, and the concatenation 

$$
(f _ {1}, \ldots , f _ {m}): S ^ {\flat} \to T
$$

is the composite 

$$
(f _ {1}, \dots , f _ {m}) = f \circ \mathrm{id} _ {S ^ {\flat}} ^ {S}
$$

of Example 3.2.3.2 

Example 3.2.6.9. Suppose $f : S  T$ is a nested tuple morphism, and suppose 

$$
S = (S _ {1}, \dots , S _ {r}).
$$

Recall from example 3.2.3.6 that for any $1 \leq i \leq r ,$ , there is a nested tuple morphism 

$$
f _ {i}: S _ {i} \to T.
$$

called the ith mode of $f .$ These morphisms have pairwise disjoint images, and the concatenation 

$$
(f _ {1}, \dots , f _ {r}): S \to T
$$

is equal to $f .$ In other words, every nested tuple morphism f may be written as the concatenation of its modes: 

$$
f = (f _ {1}, \dots , f _ {r}).
$$

Proposition 3.2.6.10. If $f _ { 1 } , \ldots , f _ { k }$ are nested tuple morphisms with the same codomain and with pairwise disjoint images, then 

$$
L _ {(f _ {1}, \dots , f _ {k})} = (L _ {f _ {1}}, \dots , L _ {f _ {k}}).
$$

Proof. By construction, we have 

$$
\begin{array}{c} \text {shape} ((L _ {f _ {1}}, \ldots , L _ {f _ {k}})) = (\text {shape} (L _ {f _ {1}}), \ldots , \text {shape} (L _ {f _ {k}})) \\ = \text {shape} (L _ {(f _ {1}, \ldots , f _ {k})}). \end{array}
$$

and using Proposition 3.1.5.38, we have 

$$
\begin{array}{r l} (L _ {f _ {1}}, \ldots , L _ {f _ {k}}) ^ {\flat} & = L _ {f _ {1}} ^ {\flat} \star \dots \star L _ {f _ {k}} ^ {\flat} \\ & = L _ {f _ {1} ^ {\flat}} \star \dots \star L _ {f _ {k} ^ {\flat}} \\ & = L _ {f _ {1} ^ {\flat} \star \dots \star f _ {k} ^ {\flat}} \\ & = L _ {(f _ {1}, \ldots , f _ {k}) ^ {\flat}} \\ & = (L _ {(f _ {1}, \ldots , f _ {k})}) ^ {\flat}. \end{array}
$$

## 3.2.6.2 Coalesce

If $f$ is a nested tuple morphism, then we might define coal(f) to be ${ \mathsf { c o a l } } ^ { \flat } ( f ^ { \flat } )$ . Theoretically, this is a sound definition. However, in order to make our definitions compatible with the cute implementation, we make a small modification to our definition of coal(f). 

Definition 3.2.6.11. Suppose $f : S  T$ is a nested tuple morphism, and write 

$$
\operatorname{coal} ^ {\flat} \left(f ^ {\flat}\right): \left(s _ {1}, \dots , s _ {m}\right)\rightarrow \left(t _ {1}, \dots , t _ {n}\right).
$$

• (Case 1): If $m > 1$ , we define 

$$
\operatorname{coal} (f) = \operatorname{coal} ^ {\flat} \left(f ^ {\flat}\right).
$$

• (Case 2): If m = 1, we define coal(f) to be the composite 

$$
s _ {1} \xrightarrow [ (1) ]{} (s _ {1}) \xrightarrow {\operatorname{coal} ^ {b} (f ^ {b})} (t _ {1}, \dots , t _ {n}).
$$

• (Case 3): If $m = 0$ , we define coal(f) to be the composite 

$$
1 \xrightarrow [ (*) ]{} () \xrightarrow {\operatorname{coal} ^ {b} (f ^ {b})} (t _ {1}, \dots , t _ {n}).
$$

Example 3.2.6.12. If 

$$
f: ((2, 2), (3, 3), (5, 5)) \to (5, 5, 3, 3, 2, 2)
$$

lies over $\alpha = ( 5 , 6 , 3 , 4 , 1 , 2 )$ , then 

$$
\operatorname{coal} (f): (4, 9, 2 5) \rightarrow (2 5, 9, 4)
$$

lies over $\alpha ^ { \prime } = ( 3 , 2 , 1 )$ ). 

Proposition 3.2.6.13. $\lg f \colon S  T$ is a nested tuple morphism, then 

$$
\operatorname{coal} \left(L _ {f}\right) = L _ {\operatorname{coal} (f)}.
$$

Proof. Let’s again write 

$$
(s _ {1}, \ldots , s _ {m}) \xrightarrow [ \alpha ]{\text {coal} ^ {\flat} (f ^ {\flat})} (t _ {1}, \ldots , t _ {n}).
$$

There are three cases to consider. 

• (Case 1): Suppose $m > 1$ . Then 

$$
\begin{array}{l} L _ {\text { coal } (f)} = L _ {\text { coal } ^ {\flat} (f ^ {\flat})} \\ \quad = \text { coal } ^ {\flat} (L _ {f ^ {\flat}}) \\ \quad = \text { coal } ((L _ {f}) ^ {\flat}) \\ \quad = \text { coal } (L _ {f}). \end{array}
$$

• (Case 2): Suppose $m = 1$ . Then 

$$
\begin{array}{l} L _ {\mathsf {c o a l} (f)} = s _ {1}: t _ {1} \ldots , t _ {\alpha (1) - 1} \\ \qquad = \mathsf {c o a l} ((s _ {1}): (t _ {1} \dots t _ {\alpha (1) - 1})) \\ \qquad = \mathsf {c o a l} (L _ {\mathsf {c o a l} ^ {\flat} (f ^ {\flat})}) \\ \qquad = \mathsf {c o a l} (\mathsf {c o a l} ^ {\flat} (L _ {f ^ {\flat}})) \\ \qquad = \mathsf {c o a l} ((L _ {f}) ^ {\flat}) \\ \qquad = \mathsf {c o a l} (L _ {f}). \end{array}
$$

• (Case 3): Suppose $m = 0$ . Then 

$$
\begin{array}{l} L _ {\mathsf {c o a l} (f)} = 1: 0 \\ \qquad = \mathsf {c o a l} (\mathbf {\Omega}): (\mathbf {\Omega}) \\ \qquad = \mathsf {c o a l} (L _ {\mathsf {c o a l} ^ {\flat} (f ^ {\flat})}) \\ \qquad = \mathsf {c o a l} (\mathsf {c o a l} ^ {\flat} (L _ {f ^ {\flat}})) \\ \qquad = \mathsf {c o a l} ((L _ {f}) ^ {\flat}) \\ \qquad = \mathsf {c o a l} (L _ {f}). \end{array}
$$

## 3.2.6.3 Complement

In this section, we define the notion of complementary nested tuple morphisms. 

Definition 3.2.6.14. Suppose $f : S  T$ and $g : U \to T$ are nested tuple morphisms with disjoint images. We say g is a complement of f if 

$$
(f, g): (S, U) \to T
$$

is an isomorphism. 

Remark 3.2.6.15. If $f : S  T$ and $g : U \to T$ are nested tuple morphisms, then $g$ is a complement of f if and only if $g ^ { \flat }$ is a complement of $f ^ { \flat }$ , since $( f , g ) ^ { \flat } = f ^ { \flat } \star g ^ { \flat }$ 

Proposition 3.2.6.16. If $f : S  T$ is a nested tuple morphism and $g : U \to T$ is a complement of f, then $L _ { g }$ is a size(T)-complement of $L _ { f }$ . 

Proof. Observation 3.2.2.5 implies that 

$$
\begin{array}{l} (L _ {f}) ^ {\flat} = L _ {f ^ {\flat}}, \text {and} \\ (L _ {g}) ^ {\flat} = L _ {g ^ {\flat}} \end{array}
$$

and Lemma 2.3.6.2 allows us to reduce to the flat case (Proposition 3.1.5.42). 

Construction 3.2.6.17. Suppose $f : S  T$ is a nested nested tuple morphism. We define the complement of f to be the composite 

![image](Imgaes/categorical-foundations-cute-layouts-paper/913fbe69c4fc455d52ffa86e8273f962b26e3b2fbca8bd6f7aae9197e8fd03ce.jpg)


where $( f ^ { \flat } ) ^ { c }$ , is as defined in Construction 3.1.5.46, and $\mathsf { i d } _ { T ^ { \flat } } ^ { T } : T ^ { \flat } \cong T$ is the unflattening isomorphism. 

Example 3.2.6.18. The complement of the nested tuple morphism 

$$
((2, 2), (5, 5)) \xrightarrow [ (1 , 4 , 2 , 5) ]{f} ((2, 5, 7), (2, 5, 7))
$$

is 

$$
(7, 7) \xrightarrow [ (3 , 6) ]{f ^ {c}} ((2, 5, 7), (2, 5, 7)).
$$

Proposition 3.2.6.19. Suppose $f : S  T$ and $g : U \to T$ are nested tuple morphisms. If f is injective and g is a complement of f, then $L _ { g }$ is a size(T)-complement of $L _ { f }$ . 

Proof. This follows from Proposition 3.1.5.42 and Lemma 2.3.6.2 since 

$$
\begin{array}{l} (L _ {f}) ^ {\flat} = L _ {f ^ {\flat}} \\ (L _ {g}) ^ {\flat} = L _ {g ^ {\flat}}. \end{array}
$$

Proposition 3.2.6.20. $I f f : S \to T$ is an injective nested tuple morphism, then 

$$
\operatorname{coal} \left(L _ {f ^ {c}}\right) = \operatorname{comp} \left(L _ {f}, \text { size } (T)\right).
$$

Proof. Since $f ^ { c }$ is obtained from $( f ^ { \flat } ) ^ { c }$ by post-composing with a reparenthesization isomorphism, it follows that 

$$
L _ {f ^ {c}} = L _ {(f ^ {\flat}) ^ {c}}
$$

so by Proposition 3.2.6.20, it follows that 

$$
\operatorname{coal} ^ {\flat} \left(L _ {f ^ {c}}\right) = \operatorname{comp} ^ {\flat} \left(L _ {f}, \operatorname{size} (T)\right).
$$

Applying coal(−) to both sides yields the result. 

## 3.2.6.4 Composition

We can use the realization functor of Section 3.2.4 to prove that composition of nested tuple morphisms is compatible with composition of the associated layouts. 

Theorem 3.2.6.21. If f and g are non-degenerate composable nested tuple morphisms, then 

$$
L _ {g \circ f} = L _ {g} \circ L _ {f}.
$$

Proof. Suppose $f : S  T$ and $g : T  U$ are non-degenerate nested tuple morphisms. We need to check that 

1. shape ${ } : ( L _ { g \circ f } )$ refines shape $\left( L _ { f } \right)$ : This holds since 

$$
\operatorname{shape} \left(L _ {f}\right) = S = \operatorname{shape} \left(L _ {g \circ f}\right).
$$

2. ${ \cal L } _ { g \circ f }$ is coalesced over $\mathsf { s h a p e } ( L _ { f } ) ;$ : This holds since the nested tuple morphism $g \circ f$ is nondegenerate, hence so is the layout ${ \cal L } _ { g \circ f }$ 

3. Φ $ \cdot _ { L _ { g \circ f } } = \Phi _ { L _ { g } } \circ \Phi _ { L _ { f } } ^ { \mathsf { s i z e } ( L _ { g } ) }$ : Using Lemma 3.2.4.2, we have 

$$
\begin{array}{r l} \Phi_ {L _ {g \circ f}} ^ {\text {size} (U)} & = | g \circ f | \\ & = | g | \circ | f | \\ & = \Phi_ {L _ {g}} ^ {\text {size} (U)} \circ \Phi_ {L _ {f}} ^ {\text {size} (T)} \end{array}
$$

and by postcomposing with the inclusion $[ 0 , \mathsf { s i z e } ( U ) ) \subset \mathbb { Z }$ , and observing that siz $\mathsf { \Omega } _ { \mathsf { \Omega } } ^ { \mathsf { \Omega } } ( T ) = \mathsf { S i z e } ( L _ { g } )$ ， the result follows. 

## 3.2.6.5 Logical division

Next, we introduce logical division of nested tuple morphisms. This construction is obtained from flat division by introducing nesting profiles, with no compatibility constraints. 

Definition 3.2.6.22. Suppose $f$ and $g$ are nested tuple morphisms. We say $g$ divides f if g and $f$ are composable. In other words, 

$$
\operatorname{codomain} (g) = \operatorname{domain} (f).
$$

Definition 3.2.6.23. Suppose $g : S  T$ and $f : T  U$ are nested tuple morphisms. We define the logical division of $f$ by $g$ to be the nested tuple morphism 

$$
f \oslash g = f \circ (g, g ^ {c}).
$$

Example 3.2.6.24. The logical division of 

$$
((2, 2), 2) \xrightarrow [ (2 , 4 , *) ]{f} ((4, 2), (4, 2))
$$

by 

$$
(2, 2) \xrightarrow [ (1 , 3) ]{g} ((2, 2), 2)
$$

is 

$$
((2, 2), 2) \xrightarrow [ (2 , * , 4) ]{f \oslash g} ((4, 2), (4, 2)).
$$

Example 3.2.6.25. The logical division of 

$$
(8, 8, 5 1 2, 5 1 2, 5 1 2) \xrightarrow [ (* , * , 1 , 2 , 3) ]{f} (5 1 2, 5 1 2, 5 1 2)
$$

b 

$$
(8, 5 1 2) \xrightarrow [ (1 , 5) ]{g} (8, 8, 5 1 2, 5 1 2, 5 1 2)
$$

is 

$$
((8, 5 1 2), (8, 5 1 2, 5 1 2)) \xrightarrow [ (* , 1 , * , 2 , 3) ]{f \oslash g} ((4, 2), (4, 2)).
$$

Proposition 3.2.6.26. If $g : S  T$ and $f : T  U$ are non-degenerate nested tuple morphisms, then 

$$
\operatorname{coal} \left(L _ {f \oslash g}\right) = \operatorname{coal} \left(L _ {f} \oslash L _ {g}\right).
$$

Proof. By Proposition 3.2.6.20, we have 

$$
\operatorname{coal} \left(\operatorname{comp} \left(L _ {g}, \text { size } \left(L _ {f}\right)\right)\right) = \operatorname{coal} \left(L _ {g ^ {c}}\right)
$$

and we compute 

$$
\begin{array}{l} \text {coal} (L _ {f} \oslash L _ {g}) = \text {coal} (L _ {f} \circ (L _ {g}, \text {comp} (L _ {g}, \text {size} (L _ {f})))) \\ \qquad = \text {coal} (L _ {f} \circ (L _ {g}, L _ {g ^ {c}})) \\ \qquad = \text {coal} (L _ {f} \circ L _ {(g, g ^ {c})}) \\ \qquad = \text {coal} (L _ {f} \circ L _ {(g, g ^ {c})}) \\ \qquad = \text {coal} (L _ {f \circ (g, g ^ {c})}) \\ \qquad = \text {coal} (L _ {f \oslash g}). \end{array}
$$

Proposition 3.2.6.27. If f and $g$ are nested tuples and $g$ divides $f ,$ then 

$$
(f \oslash g) ^ {\flat} = f ^ {\flat} \oslash^ {\flat} g ^ {\flat}.
$$

Proof. We compute 

$$
\begin{array}{l} (f \oslash g) ^ {\flat} = (f \circ (g, g ^ {c})) ^ {\flat} \\ \qquad = f ^ {\flat} \circ (g, g ^ {c}) ^ {\flat} \\ \qquad = f ^ {\flat} \circ (g ^ {\flat} \star (g ^ {c}) ^ {\flat}) \\ \qquad = f ^ {\flat} \circ (g ^ {\flat} \star (g ^ {\flat}) ^ {c}) \\ \qquad = f ^ {\flat} \oslash^ {\flat} g ^ {\flat}. \end{array}
$$

## 3.2.6.6 Logical products

In this section, we define the logical product of nested tuple morphisms. 

Definition 3.2.6.28. Suppose $f$ and g are nested tuple morphisms. We say $f$ and $g$ are product admissible if codomain $( g )$ = domain(f<sup>c</sup>). If $f$ and $g$ are product admissible we define the logical product of $f$ and $g$ to be the nested tuple morphism 

$$
f \otimes g = (f, f ^ {c} \circ g).
$$

Example 3.2.6.29. The nested tuple morphisms 

$$
(8, 8) \xrightarrow [ (1 , 2) ]{f} (8, 8, 1 6, 1 6)
$$

and 

$$
(1 6, 1 6) \xrightarrow [ (1 , 2) ]{g} (1 6, 1 6)
$$

are product admissible, and their logical product is 

$$
((8, 8), (1 6, 1 6)) \xrightarrow [ (1 , 2 , 3 , 4) ]{f \otimes g} (8, 8, 1 6, 1 6).
$$

Example 3.2.6.30. The nested tuple morphisms 

$$
(1 2 8, 1 2 8) \xrightarrow [ (3 , 4) ]{f} (3 2, 3 2, 1 2 8, 1 2 8)
$$

and 

$$
(3 2) \xrightarrow [ (2) ]{g} (3 2, 3 2)
$$

are product admissible, and their logical product is 

$$
((1 2 8, 1 2 8), (3 2)) \xrightarrow [ (3 , 4 , 2) ]{f \otimes g} (3 2, 3 2, 1 2 8, 1 2 8).
$$

Proposition 3.2.6.31. Suppose f and g are non-degenerate nested tuple morphisms and that f and $g$ are product-admissible. Then 

$$
L _ {f \otimes g} = L _ {f} \otimes L _ {g}.
$$

Proof. Suppose $f : S  T$ and $g : U \to V$ are product admissible, and set 

$$
L _ {f} ^ {*} = \operatorname{comp} (L _ {f}, \text { size } (L _ {f}) \cdot \text { cosize } (L _ {g}))
$$

Since f is injective and codomain $( g ) = \mathsf { d o m a i n } ( f ^ { c } )$ , it follows that 

$$
\operatorname{size} \left(L _ {f}\right) \cdot \operatorname{cosize} \left(L _ {g}\right) \leq \operatorname{size} (S) \cdot \operatorname{size} (V) = \operatorname{size} (T).
$$

Using this fact, and the fact that 

$$
\Phi_ {\mathrm{comp} (L _ {f}, \mathrm{size} (T))} = \Phi_ {L _ {f c}},
$$

we have 

$$
\begin{array}{c} L _ {f} ^ {*} \circ L _ {g} = \mathsf {c o m p} (L _ {f}, \mathsf {s i z e} (T)) \circ L _ {g} \\ = L _ {f ^ {c}} \circ L _ {g}. \end{array}
$$

Using this fact, we compute 

$$
\begin{array}{r l} L _ {f} \otimes L _ {g} & = (L _ {f}, L _ {f} ^ {*} \circ L _ {g}) \\ & = (L _ {f}, L _ {f ^ {c}} \circ L _ {g}) \\ & = (L _ {f}, L _ {f ^ {c} \circ g}) \\ & = L _ {(f, f ^ {c} \circ g)} \\ & = L _ {f \otimes g} \end{array}
$$

## Chapter 4

## Computations

The categories Tuple and Nest ofer a powerful framework for computing with tractable layouts. It is frequently the case that in practice, however, one comes across tractable layouts A and B that are composable in the context of cute but whose standard representations are neither composable in Tuple nor Nest. This chapter is dedicated to the explication of how one may nevertheless use the categories Tuple and Nest to compute the composition, logical division, and logical product of tractable layouts, using the notion of mutual refinement. We introduce this notion in Section 4.1.1, present an algorithm for computing mutual refinements in Algorithm 4.1.1, and work through many explicit examples. 

## 4.1 Composition of tractable layouts

Suppose we want to compute the composition $B \circ A$ of the tractable layouts 

$$
\begin{array}{l} A = (6, 6): (6, 1), \\ B = (1 2, 3, 6): (1, 7 2, 1 2). \end{array}
$$

We might try to compute B ◦ A by computing the composite of the standard representations $f$ and $g$ of A and B: 

$$
\begin{array}{c c} 6 & 6 \\ 6 & 6 \end{array} f \quad \begin{array}{c c} 6 & 3 \\ 3 & 6 \\ 1 2 & 1 2 \end{array}
$$

However, these morphisms are not composable, since the codomain $( 6 , 6 )$ of $f$ is not equal to the domain (12, 3, 6) of $g .$ . This means that we can not use the morphisms $f$ and $g$ to compute the composite $B \circ A$ directly. We can, however, proceed with our computation by finding a mutual refinement of (6, 6) and (12, 3, 6), as depicted below 

$$
\begin{array}{c} 6 \searrow \\ 3 \searrow 6 \\ 6 \swarrow 2 \searrow 3 \\ 6 \searrow 6 \searrow 1 2 \end{array}
$$

This is a device which converts $f$ and $g$ into composable morphisms $f ^ { \prime }$ and $g ^ { \prime } { \mathrm { : } }$ 

![image](Imgaes/categorical-foundations-cute-layouts-paper/5c4ef193a4ea19eea993819b984cc2b799503da03b57d5990b2af3c18a4657af.jpg)


![image](Imgaes/categorical-foundations-cute-layouts-paper/d01caf72cc2adb90b6ea1bc6f69469c2ae5511a6112d5acc8ba96abe60b2e941.jpg)


The morphisms $f ^ { \prime }$ and $g ^ { \prime }$ are composable, so we may form the composite 

![image](Imgaes/categorical-foundations-cute-layouts-paper/9ff8b66369fd1e2be3045cca8238fb721e72bb3bdcb07eeed852067563ec5352.jpg)


and computing the encoded layout yields 

$$
B \circ A = L _ {g ^ {\prime} \circ f ^ {\prime}} = ((2, 3), 6): ((6, 7 2), 1).
$$

The goal of this section is to formalize this computational process into an algorithm for computing the composite of tractable layouts A and B. As we saw in our example, the non-trivial steps in our computation were 

1. finding a mutual refinement of certain (nested) tuples, and 

2. using the mutual refinement to convert f and g into composable morphisms $f ^ { \prime }$ and $g ^ { \prime }$ . 

We dedicate the following two sections to the explication of these steps. 

## 4.1.1 Mutual refinements

Before giving a precise definition of mutual refinements using the categorical framework of Chapter $s ,$ we give an informal overview. Consider the tuples (6, 6) and (12, 3, 6) of our motivating example. We asserted that the diagram 

$$
\begin{array}{c} 6 \searrow \\ 3 \searrow 6 \\ 6 \swarrow 2 \searrow 3 \\ 6 \searrow 6 \searrow 1 2 \end{array}
$$

is a mutual refinement of (6, 6) and (12, 3, 6). We can give a more precise description of this mutual refinement as follows. The left half of the diagram represents the refinement $( 6 , 6 )  ( 6 , ( 2 , 3 ) )$ , and the right half of the diagram represents the refinement ((6, 2), 3, 6) ↠ (12, 3, 6): 

$$
\begin{array}{l} 6 \xrightarrow {\angle} 2 \\ 6 - 6 \end{array} \qquad \qquad \leftrightarrow \qquad \qquad (6, 6) \ll - (6, (2, 3))
$$

$$
\begin{array}{l} 6 \\ 3 \searrow 6 \\ 2 \searrow 3 \\ 6 \searrow 1 2 \end{array} \quad \leftrightarrow \quad ((6, 2), 3, 6) \longrightarrow (1 2, 3, 6)
$$

The fact that the two halves of the diagram may be glued together corresponds to the fact that the nested tuple (6, (2, 3)) divides ((6, 2), 3, 6), which we denote 

$$
(6, (2, 3)) \succrightarrow ((6, 2), 3, 6).
$$

Putting these observations together, we may express our mutual refinement precisely as 

$$
\begin{array}{c c c} 6 & \\ 3 & \searrow & 6 \\ 6 \swarrow & 2 & \searrow & 3 \\ 6 & - & 6 & - & 1 2 \end{array} \qquad \leftrightarrow \qquad \begin{array}{c c c} (6, (2, 3)) \longmapsto ((6, 2), 3, 6) \\ \downarrow & & \downarrow \\ (6, 6) & & (1 2, 3, 6) \end{array}
$$

where we opt to depict the refinements $( 6 , 6 )  ( 6 , ( 2 , 3 ) )$ and $( ( 6 , 2 ) , 3 , 6 )  ( 1 2 , 3 , 6 )$ vertically. We can now give a precise definition of mutual refinements. 

Definition 4.1.1.1. Suppose T and U are nested tuples. A mutual refinement of $( T , U )$ is a diagram of the form 

$$
\begin{array}{c c c} T ^ {\prime} & \longrightarrow & U ^ {\prime} \\ \Big \downarrow & & \Big \downarrow \\ T & & U \end{array}
$$

Explicitly, this is a pair of nested tuples $( T ^ { \prime } , U ^ { \prime } )$ such that 

1. $T ^ { \prime }$ refines T, 

2. $U ^ { \prime }$ refines $U ,$ and 

3. $T ^ { \prime }$ divides U<sup>′</sup>. 

Example 4.1.1.2. A mutual refinement of $T = ( 6 , 6 )$ and $U = ( 2 , 6 , 3 )$ is given by 

$$
\begin{array}{c c} ((2, 3), (2, 3)) \longrightarrow & (2, (3, 2), 3) \\ \Big \downarrow & \Big \downarrow \\ (6, 6) & (2, 6, 3) \end{array}
$$

We depict this mutual refinement as follows. 

$$
\begin{array}{c} 3 \\ 6 \text {   \text {   }   } 2 \text {   \text {   }   } 3 \\ 3 \text {   \text {   }   } 6 \\ 6 \text {   \text {   }   } 2 \text {   \text {   }   } 2 \end{array}
$$

Example 4.1.1.3. A mutual refinement of $T = ( 8 , 8 , 8 )$ and $U = ( 2 , 8 , 8 , 8 )$ is given by 

$$
\begin{array}{c} ((2, 4), (2, 4), (2, 4)) \longrightarrow (2, (4, 2), (4, 2), (4, 2)) \\ \Big \downarrow \\ (8, 8, 8) \end{array}
$$

We depict this mutual refinement as follows. 

$$
\begin{array}{c} 2 \\ 4 \text { —— } 8 \\ 8 \text { —— } 2 \\ 4 \text { —— } 8 \\ 8 \text { —— } 2 \\ 4 \text { —— } 8 \\ 8 \text { —— } 2 \end{array}
$$

Example 4.1.1.4. A mutual refinement of $T = ( 4 , 2 , 2 , 3 2 )$ and $U = ( 3 2 , 3 2 )$ is given by 

$$
\begin{array}{c c} (4, 2, 2, (2, 1 6)) \xrightarrow {} ((4, 2, 2, 2), (1 6, 2)) \\ \Big \downarrow & \Big \downarrow \\ (4, 2, 2, 3 2) & (3 2, 3 2) \end{array}
$$

We depict this mutual refinement as follows. 

$$
\begin{array}{c} 2 \\ 1 6 - 3 2 \\ 3 2 - 2 \\ 2 - 2 \\ 2 - 2 \\ 4 - 4 - 3 2 \end{array}
$$

Example 4.1.1.5. If $T = ( 8 , 8 )$ and $U = ( 3 , 8 , 8 )$ , then there does not exist a mutual refinement of $T$ and $U _ { ☉ }$ . 

Example 4.1.1.6. If T and U are tuples with size $( T ) = 2 ^ { k }$ and $\mathsf { s i z e } ( L ) = 2 ^ { \ell }$ with $k \leq \ell ,$ then there exists a mutual refinement of $T$ and $U .$ . More generally, if T and U are tuples where size $\cdot ( T ) \leq \mathsf { s i z e } ( U )$ are powers of some fixed integer, then there exists a mutual refinement of $T$ and $U .$ . 

Observation 4.1.1.7. In each of the previous examples, we have considered mutual refinements of flat tuples $T$ and $U .$ The definition of mutual refinement, however, allows $T$ and $U$ to be any nested tuples. In any case, restricting to the flat case is no loss of generality, because there is a one-to-one correspondence between mutual refinements of a pair of nested tuples $( T , U )$ , and mutual refinements of their flattenings $( T ^ { \flat } , U ^ { \flat } )$ (see Lemma 3.2.5.20). In particular, there exists a mutual refinement of $( T , U )$ if and only if there exists a mutual refinement of $( T ^ { \flat } , U ^ { \flat } )$ 

Having made the appropriate definitions, we provide an algorithm for computing a mutual refinement of $( T , U )$ 

Algorithm 4.1.1: Mutual refinement algorithm

1 Input: Nested tuples T and U.
2 Output: A mutual refinement (T',U') of (T,U), if one exists, else None.

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

## 4.1.2 From mutual refinements to composable morphisms

Recall that in order to compute the composition $B \circ A$ of 

$$
\begin{array}{l} A = (6, 6): (6, 1) \text {and} \\ B = (1 2, 3, 6): (1, 7 2, 1 2), \end{array}
$$

we constructed tuple morphisms 

$$
\begin{array}{c c} 6 & 6 \\ 6 & 6 \end{array} f \quad \begin{array}{c c} 6 & 3 \\ 3 & 6 \\ 1 2 & 1 2 \end{array}
$$

and a mutual refinement. 

$$
\begin{array}{c} 6 \searrow \\ 3 \searrow 6 \\ 6 \swarrow 2 \searrow 3 \\ 6 \swarrow 6 \searrow 1 2 \end{array}
$$

The next step in our computation is to use our mutual refinement to convert $f$ and $g$ into composable morphisms $f ^ { \prime }$ and $g ^ { \prime } .$ Before giving a formal, categorical definition of this process, let’s illustrate the process with an example. 

We construct $f ^ { \prime }$ from $f$ and the left half of our mutual refinement: 

![image](Imgaes/categorical-foundations-cute-layouts-paper/72bb5a90c7e07be2f4a2f1d29bd010cc38c7bbd8daa6b355c866fd1179336d9e.jpg)


This construction is made by making the replacement 

$$
\begin{array}{c c c}6&\rightsquigarrow&6 - 6\\\searrow&6 - 6&\searrow\\6&\end{array}
$$

and making the replacement 

![image](Imgaes/categorical-foundations-cute-layouts-paper/34dadeeb02a970f2f730d0bc28f2341da19e333fb89add6927a94d357266f0a2.jpg)


More generally, we make the replacement 

![image](Imgaes/categorical-foundations-cute-layouts-paper/d0ae701daa1cd82349a5e2a64d1667902227253b4bcbfbfb6b3f86fa50f82283.jpg)


The process for constructing g<sup>′</sup> from g, and the right half of our mutual refinement is similar. 

$$
\begin{array}{c c}6&\\3&\searrow\\2&\searrow\\6&\longrightarrow\\\hline\end{array}\begin{array}{c c}6&\\3&\swarrow\\2&\longrightarrow\\6&\longmapsto\\\hline\end{array}\begin{array}{c c}3&\\6&\\\hline\end{array}\quad \rightsquigarrow \quad\begin{array}{c c}6&\\3&\swarrow\\2&\longmapsto\\6&\longmapsto\\\hline\end{array}\begin{array}{c c}3&\\6&\\\hline\end{array}\quad g ^ {\prime}
$$

This construction is made by making the replacements 

$$
6 \longrightarrow 6 \longmapsto 6 \quad \rightsquigarrow \quad 6 \longmapsto 6
$$

$$
3 \longrightarrow 3 \longmapsto 3 \quad \rightsquigarrow \quad 3 \longmapsto 3
$$

$$
\begin{array}{c c c}2&&2 \longmapsto 2\\6 \xrightarrow {\quad} 1 2 \longmapsto 1 2&\rightsquigarrow&6 \longmapsto 6\end{array}
$$

More generally, we make the replacement 

![image](Imgaes/categorical-foundations-cute-layouts-paper/6427b8e7b6ed709583725f07d8691938d622561723cfbc6a7d5b0357edb81551.jpg)


Having given an informal description of our procedure, we make things precise as follows. 

Construction 4.1.2.1. Suppose $f : S  T$ and $g : U \to V$ are nested tuple morphisms, and $( T ^ { \prime } , U ^ { \prime } )$ is a mutual refinement of $( T , U )$ . Then we may use the pullback and pushforward constructions of section 3.2.5 to form the diagram: 

$$
\begin{array}{c c c c c} S ^ {\prime} & \xrightarrow {\tilde {f}} & T ^ {\prime} & \xrightarrow {i} & U ^ {\prime} \\ \Big \downarrow & \searrow & \Big \downarrow & & \Big \downarrow \\ S & \xrightarrow [ f ] & T & & U \end{array} \xrightarrow [ g ]{\quad} V
$$

If we set $f ^ { \prime } = i \circ \tilde { f }$ and $g ^ { \prime } = \tilde { g }$ , then 

$$
S ^ {\prime} \xrightarrow {f ^ {\prime}} U ^ {\prime} \xrightarrow {g ^ {\prime}} V ^ {\prime}
$$

are composable nested tuple morphisms. 

## 4.1.3 The composition algorithm



Algorithm 4.1.2: Tractable Layout Composition Algorithm 



1 Input: Tractable layouts A and B. 

Algorithm 4.1.2 (continued): Tractable Layout Composition Algorithm 

2 Output: A weak composite $C$ of A and $B ,$ if one exists, else None.. 

3 Take the standard representations 

![image](Imgaes/categorical-foundations-cute-layouts-paper/38be5817b4cffe82fc1e5a8539ce54c67754b2a9016ece511a1f9214cd8b4669.jpg)


of A and coa $| ( B )$ , respectively. 4 Use Algorithm 4.1.1 to produce a mutual refinement 

![image](Imgaes/categorical-foundations-cute-layouts-paper/23c6f7a2e3c3ffcc404c6670eabed1e2cb75b4267c91bdfe736c1345a68aca9b.jpg)


of $( T , U )$ . If there does not exist a mutual refinement of $( T , U )$ , return None. 5 Use Construction 4.1.2.1 to obtain the composable nested tuple morphisms 

$$
S ^ {\prime} \xrightarrow {f ^ {\prime}} U ^ {\prime} \xrightarrow {g ^ {\prime}} V ^ {\prime}
$$

6 Compose $f ^ { \prime }$ and $g ^ { \prime } ,$ and compute the encoded layout 

$$
C = L _ {g ^ {\prime} \circ f ^ {\prime}}
$$

7 return C 

Theorem 4.1.3.1. If A and B are tractable layouts, then the output $C$ of the previous algorithm is a weak composite of A and B. Consequently, 

$$
B \circ A = \operatorname{coal} (C, \operatorname{shape} (A)).
$$

Proof. Proposition 3.2.5.15 and tells us that 

$$
\Phi_ {L _ {g ^ {\prime}}} = \Phi_ {L _ {g}} = \Phi_ {\mathsf {c o a l} (B)} = \Phi_ {B},
$$

and Proposition 3.2.5.11 and Example 3.1.3.6 tell us that 

$$
\Phi_ {L _ {f ^ {\prime}}} = \Phi_ {L _ {f}} = \Phi_ {A}.
$$

Theorem 3.2.6.21 then implies that 

$$
\begin{array}{r} \Phi_ {C} = \Phi_ {L _ {g ^ {\prime} \circ f ^ {\prime}}} = \Phi_ {g ^ {\prime}} \circ \Phi_ {f ^ {\prime}} ^ {\mathrm{size} (U ^ {\prime})} \\ = \Phi_ {B} \circ \Phi_ {A} ^ {\mathrm{size} (B)}. \end{array}
$$

By construction, the shape $S ^ { \prime }$ of $L _ { f ^ { \prime } }$ refines the shape S of A, so we conclude that $C$ is a weak composite of A and B. □ 

## 4.1.4 Examples

In this section we illustrate how Algorithm 4.1.3 may be used to compute the composition $B \circ A$ of tractable layouts A and B. 

Example 4.1.4.1. Suppose $A = ( 4 ) : ( 1 )$ , and $B = ( 2 , 2 ) : ( 2 , 1 )$ 

1. Take the standard representations of A and coa $| ( B ) = B$ 

$$
\begin{array}{c} 4 \longmapsto 4 \\ f \end{array} \qquad \qquad \begin{array}{c} 2 \\ 2 \\ g \end{array}
$$

2. Apply Algorithm 4.1.1 to obtain the mutual refinement 

$$
4 \leq \begin{array}{l} 2 - 2 \\ 2 - 2 \end{array}
$$

3. Form the diagram 

![image](Imgaes/categorical-foundations-cute-layouts-paper/478d9766571f93615b10ac14083520b0240943330cba7f649e47e4a3b76d42aa.jpg)


4. Resolve the diagram 

![image](Imgaes/categorical-foundations-cute-layouts-paper/f4636f6a12862b6eb86fde915fcc4e0d414090928b5791c23b69962cce7a0ce3.jpg)


5. Compose $f ^ { \prime }$ and $g ^ { \prime }$ to obtain 

$$
4 \leq \begin{array}{c} 2 \\ 2 \end{array} \bigotimes_ {g ^ {\prime} \circ f ^ {\prime}} ^ {2}
$$

6. Compute the associated layout 

$$
L _ {g ^ {\prime} \circ f ^ {\prime}} = ((2, 2)): ((2, 1)).
$$

7. $L _ { g ^ { \prime } \circ f ^ { \prime } }$ is coalesced over (4), so 

$$
B \circ A = ((2, 2)): ((2, 1)).
$$

Example 4.1.4.2. Suppose $A = ( 6 , 6 ) : ( 6 , 1 )$ , and $B = \left( 1 2 , 3 , 6 \right) : \left( 1 , 7 2 , 1 2 \right)$ 

1. Take the standard representations of A and ${ \mathsf { c o a l } } ( B ) = B$ 

$$
\begin{array}{c} 6 \\ 6 \end{array} \xrightarrow {} \begin{array}{c} 6 \\ 6 \end{array} f
$$

$$
\begin{array}{c} 6 \xrightarrow {} 3 \\ 3 \xrightarrow {} 6 \\ 1 2 \longmapsto 1 2 \end{array} g
$$

2. Apply Algorithm 4.1.1 to obtain the mutual refinement 

$$
\begin{array}{c} 6 \searrow \\ 3 \searrow 6 \\ 6 \swarrow 2 \searrow 3 \\ 6 \text { - - - } 6 \text { - - - } 1 2 \end{array}
$$

3. Form the diagram 

![image](Imgaes/categorical-foundations-cute-layouts-paper/99b486f6093c99a488f4090be8890226bbda3df622ec2626f44d704090199c93.jpg)


4. Resolve the diagram to obtain 

![image](Imgaes/categorical-foundations-cute-layouts-paper/1d2ef844619fecd333562d183a93783fd95954dd9e15235177a1a9765b13a279.jpg)


5. Compose $f ^ { \prime }$ and $g ^ { \prime }$ to obtain 

![image](Imgaes/categorical-foundations-cute-layouts-paper/0c00f7033a1449ca32fbab952a19b920e467c39d61449bbdf6650b804a913618.jpg)


6. Compute the associated layout 

$$
L _ {g ^ {\prime} \circ f ^ {\prime}} = ((2, 3), 6): ((6, 7 2), 1).
$$

7. $L _ { g ^ { \prime } \circ f ^ { \prime } }$ is coalesced over (6, 6), hence 

$$
B \circ A = ((2, 3), 6): ((6, 7 2), 1).
$$

Example 4.1.4.3. Suppose $A = ( 8 , 8 ) : ( 8 , 1 )$ , and $B = ( 1 6 , 1 6 ) : ( 1 6 , 1 )$ 

1. Take the standard representations of A and coal(B) = B. 

$$
\begin{array}{c c} 8 & 8 \\ 8 & 8 \end{array} \quad f \quad g
$$

2. Apply Algorithm 4.1.1 to obtain the mutual refinement 

$$
\begin{array}{c} 4 \\ 4 \searrow \\ 8 \text {   - - -   } 2 \\ 8 \text {   - - -   } 8 \end{array} \begin{array}{c} 1 6 \\ \text {   - - -   } 1 6 \end{array}
$$

3. Form the diagram 

![image](Imgaes/categorical-foundations-cute-layouts-paper/c9e81d362a0eb52da1c86556a59d0fb75a9d1def74a0047229d3a181337f51e9.jpg)


4. Resolve the diagram to obtain 

![image](Imgaes/categorical-foundations-cute-layouts-paper/7e81ced34411756f8f8618c0b7ab47560190eaa478aa79c6a2982a9c57e5d983.jpg)


5. Compose $f ^ { \prime }$ and $g ^ { \prime }$ to obtain 

![image](Imgaes/categorical-foundations-cute-layouts-paper/f64e5081c96a6af0ed5114c1c39cff2a79d5ab49aac928052c4fe820d41127e7.jpg)


6. Compute the associated layout 

$$
L _ {g ^ {\prime} \circ f ^ {\prime}} = ((2, 4), 8): ((1 2 8, 1), 1 6)
$$

7. $L _ { g ^ { \prime } \circ f ^ { \prime } }$ is coalesced over (8, 8), hence 

$$
B \circ A = ((2, 4), 8): ((1 2 8, 1), 1 6)
$$

Example 4.1.4.4. Suppose $A = ( 1 6 , 1 6 ) : ( 1 6 , 1 )$ , and $B = ( 8 , 8 , 8 ) : ( 6 4 , 8 , 1 )$ . 

1. Take the standard representations of A and coa $| ( B ) = B$ 

$$
\begin{array}{c c} 1 6 & 1 6 \\ 1 6 & \text {   f   } \end{array} \quad \begin{array}{c c} 8 & 8 \\ 8 & 8 \\ 8 & 8 \end{array} \quad g
$$

2. Apply Algorithm 4.1.1 to obtain the mutual refinement 

$$
\begin{array}{c} 2 \\ 4 \bigwedge \\ 4 \bigwedge \\ 1 6 \bigwedge \\ 1 6 \bigwedge \\ 8 \bigwedge \end{array}
$$

3. Form the diagram 

![image](Imgaes/categorical-foundations-cute-layouts-paper/e52ae499578c8522d5d0c2b63041ce71770284b331657062661766274d9adeb6.jpg)


4. Resolve the diagram to obtain 

![image](Imgaes/categorical-foundations-cute-layouts-paper/ec5303983a24b9b0047976708abeffe6351743f6b97284cc53687b13b82fa7b1.jpg)


5. Compose $f ^ { \prime }$ and $g ^ { \prime }$ to obtain 

![image](Imgaes/categorical-foundations-cute-layouts-paper/db842ccdb277087b96b26390c12c9bd7a192e57005cfd4a02bcbf5bcb7074362.jpg)


6. Compute the associated layout 

$$
L _ {g ^ {\prime} \circ f ^ {\prime}} = ((4, 4), (8, 2)): ((1 6, 1), (6 4, 8)).
$$

7. $L _ { g ^ { \prime } \circ f ^ { \prime } }$ is coalesced over (16, 16), hence 

$$
B \circ A = ((4, 4), (8, 2)): ((1 6, 1), (6 4, 8)).
$$

Example 4.1.4.5. Suppose $A = ( 6 , 6 ) : ( 5 , 6 0 )$ , and $B = ( 1 0 , 3 6 0 ) : ( 2 , 6 0 )$ 

1. Take the standard representations of A and ${ \mathsf { c o a l } } ( B ) = B$ 

![image](Imgaes/categorical-foundations-cute-layouts-paper/e710f9a6f7d1b9a1dc4d1a5e2def91babfa7df1bde9201ee7c416d7758425b00.jpg)


2. Apply algorithm 4.1.1 to obtain the mutual refinement 

![image](Imgaes/categorical-foundations-cute-layouts-paper/f5bce72e34cb284e93156a4287404f658358b8c590be9aad70b59a38dfbdd082.jpg)


3. Form the diagram 

![image](Imgaes/categorical-foundations-cute-layouts-paper/f85d4163b58431cdda0599f0720b327482805416b0a0f55cbc0c773ee3eb3084.jpg)


4. Resolve the diagram to obtain 

![image](Imgaes/categorical-foundations-cute-layouts-paper/b2786a69cd901056bbdc5ae3b2b777e3792f5ff2d7c1bb3d7ad3b5ce9a555973.jpg)


5. Compose $f ^ { \prime }$ and $g ^ { \prime }$ to obtain 

![image](Imgaes/categorical-foundations-cute-layouts-paper/904fd32192aa2609bd17b30dc0fd7df3f8df9275f0e3cd1b46f3263b4a281cb1.jpg)


6. Compute the associated layout 

$$
L _ {g ^ {\prime} \circ f ^ {\prime}} = ((2, 3), 6): ((1 0, 6 0), 3 6 0).
$$

7. The layout $L _ { g ^ { \prime } \circ f ^ { \prime } }$ is coalesced over (6, 6), so 

$$
B \circ A = ((2, 3), 6): ((1 0, 6 0), 3 6 0).
$$

## 4.1.5 More general compositions

The graphical calculus we have developed naturally extends to compute the composition $B \circ A$ of a tractable layout A with an arbitrary CuTe layout B. Informally, we do this by allowing our tuples to have entries in $\mathbb { Q } _ { > 0 } \supset \mathbb { Z } _ { > 0 }$ . We illustrate this extension with an example computation. 

Consider the layouts $A = ( 4 , 4 ) : ( 4 , 1 )$ and $B = ( 8 , 8 ) : ( 3 , 7 )$ . The layout A is tractable, and its standard representation is the tuple morphism f shown below. 

![image](Imgaes/categorical-foundations-cute-layouts-paper/44661e5bb33ec3c4fd85d70dd8ce442c618dcf95a29b0c239001998a894fc966.jpg)


The layout B is not tractable, but we may still depict B using the diagram 

![image](Imgaes/categorical-foundations-cute-layouts-paper/2f38e4e464e72f5b9a7ce9eb2ad211ccfd4907fd7c9bb1d4b390d0b14b428a0f.jpg)


This diagram does not correspond to an honest tuple morphism since the “codomain tuple” $\textstyle ( 3 , 8 , { \frac { 7 } { 2 4 } } , 8 )$ has non-integer entries. However, it still encodes the layout B via the usual prefix product formula, and is still admissible as an input to our composition algorithm: We can apply Algorithm 4.1.1 to obtain the mutual refinemen 

![image](Imgaes/categorical-foundations-cute-layouts-paper/cded2d62505106d2381d302b78707e83d0add4ff5419f65c4a1c7bad6fc19a8e.jpg)


form the diagram 

![image](Imgaes/categorical-foundations-cute-layouts-paper/ea1da707dd7e1d5b2190229d334e5e230866a46ab8e957c29e7fbe2655bfa6cf.jpg)


resolve this diagram to obtain 

![image](Imgaes/categorical-foundations-cute-layouts-paper/70c51112cd6ee6414fa650d5e1354c98509dc7fb661b5019cb3f0a0e8d005fbe.jpg)


and compose $f ^ { \prime }$ and $g ^ { \prime }$ to obtain the diagram 

![image](Imgaes/categorical-foundations-cute-layouts-paper/022a308c1dc8246bb03b09558d81b99121ac6b20563ad27e99dabca8f8be98af.jpg)


The encoded layout is $( ( 2 , 2 ) , 4 ) : ( ( 1 2 , 7 ) , 3 )$ , which is coalesced over (4, 4), so we conclude that 

$$
B \circ A = ((2, 2), 4): ((1 2, 7), 3).
$$

## 4.1.6 Admissibility for composition

In [16], the author introduces the notion of admissibility for composition, which is a suficient condition for the composition $B \circ A$ of layouts A and B to exist. Let’s recall the definition of admissibility for composition. As in [16], we restrict our attention to flat layouts with no shape entries equal to 1, and we assume that the first layout in our composition has no strides equal to 0. 

Definition 4.1.6.1. Suppose 

$$
A = (s _ {1}, \dots , s _ {m}): (d _ {1}, \dots , d _ {m})
$$

is a flat layout with no $s _ { i } = 1$ and no $d _ { i } = 0$ . Suppose B is a flat layout with 

$$
\operatorname{shape} (B) = \left(u _ {1}, \dots , u _ {p}\right).
$$

We say A and B are admissible for composition if the following conditions hold. 

1. For each $1 \leq i \leq m$ , there exists $1 \leq k \leq \ell \leq p$ such that 

(a) $u _ { 1 } \cdots u _ { k - 1 }$ divides d<sub>i</sub>, $d _ { i }$ 

(b) $d _ { i }$ divides $u _ { 1 } \cdots u _ { k }$ (properly if $k < p )$ 7 

(c) $u _ { 1 } \cdots u _ { \ell - 1 }$ divides $s _ { i } d _ { i }$ 2 

(d) $s _ { i } d _ { i }$ divides $u _ { 1 } \cdot \cdot \cdot u _ { \ell }$ (properly if $\ell < p )$ 

2. The intervals 

$$
\left[ d _ {i}, d _ {i} \left(s _ {i} - 1\right) \right] \cap \left[ 1, s _ {1} \dots s _ {m - 1}\right)
$$

are pairwise disjoint. 

Remark 4.1.6.2. The indices $k , \ell$ in the definition above are referred to as “division indices” in [16]. Remark 4.1.6.3. In [16], the author works with the “extended layout function” of a layout, which may be considered as the layout function of the layout obtained by replacing the final shape entry $s _ { m }$ with ∞. The definition we give here is the appropriate analogue for working with ordinary layout functions. 

Lemma 4.1.6.4. Suppose $T = ( t _ { 1 } , \dots , t _ { n } )$ and $U = ( u _ { 1 } , \dotsc , u _ { p } )$ are tuples of positive integers, and suppose $( T , U )$ admits a mutual refinement. Then for any prefix products $t _ { 1 } \cdots t _ { j }$ and $u _ { 1 } \cdot \cdot \cdot u _ { k } \ o f T$ and $U$ , respectively, either 

1. $t _ { 1 } \cdot \cdot \cdot t _ { j }$ is greater than $u _ { 1 } \cdot \cdot \cdot u _ { k } , o r$ 

2. $t _ { 1 } \cdot \cdot \cdot t _ { j }$ divides $u _ { 1 } \cdots u _ { k }$ 

Proof. Let’s choose some mutual refinement $( T ^ { \prime } , U ^ { \prime } )$ of $( T , U )$ , and write $( u _ { 1 } ^ { \prime } , \ldots , u _ { p ^ { \prime } } ^ { \prime } )$ for the flattening of $U ^ { \prime }$ . Any prefix product of $T$ or U is also a prefix product of $U ^ { \prime }$ , and since prefix products of a fixed tuple of positive integers satisfy $x \leq y \Rightarrow x \mid y .$ , the result follows. □ 

Theorem 4.1.6.5. Suppose A is a flat tractable layout with no shape entries equal to 1 and no stride entries equal to 0. Suppose B is a flat tractable layout. Let $f : S  T$ and $g : U \to V$ denote the standard representation of A and coal(B), respectively. If T and U admit a mutual refinement, then A and B are admissible for composition. 

Proof. Let’s write $S = ( s _ { 1 } , \ldots , s _ { m } ) , T = ( t _ { 1 } , \ldots , t _ { n } ) , U = ( u _ { 1 } , \ldots , u _ { p } )$ , and let’s write $\alpha : \langle m \rangle _ { * } \to \langle n \rangle$ ∗ for the map over which f lies. We need to check that the conditions from Definition 4.1.6.1 hold. 

1. Suppose $1 \leq i \leq m$ . Then $d _ { i } = t _ { 1 } \cdot \cdot \cdot t _ { j - 1 }$ for some $j ,$ namely $j = \alpha ( i )$ . Suppose we have a mutual refinement $( T ^ { \prime } , U ^ { \prime } )$ of $( T , U )$ , and write $( T ^ { \prime } ) ^ { \flat } = ( t _ { 1 } ^ { \prime } , \ldots , t _ { n ^ { \prime } } ^ { \prime } )$ and $( U ^ { \prime } ) ^ { \flat } = ( u _ { 1 } ^ { \prime } , \ldots , u _ { p ^ { \prime } } ^ { \prime } )$ 

• (a) and (b): Since $T ^ { \prime }$ refines T, there there exists some $1 \leq a \leq n ^ { \prime }$ such that 

$$
d _ {i} = t _ {1} \dots t _ {j - 1} = t _ {1} ^ {\prime} \dots t _ {a} ^ {\prime} = u _ {1} ^ {\prime} \dots u _ {a} ^ {\prime}.
$$

Take the maximal $k \in \langle p \rangle$ such that $u _ { 1 } \cdot \cdot \cdot u _ { k - 1 } \leq u _ { 1 } ^ { \prime } \cdot \cdot \cdot u _ { a } ^ { \prime }$ 

Suppose $k < p$ . We observe that 

$$
u _ {1} \cdot \cdot \cdot u _ {k - 1} \leq d _ {i} <   u _ {1} \cdot \cdot \cdot u _ {k}.
$$

where the second inequality holds by maximality of $k \in \langle p \rangle$ . Lemma 4.1.6.4 implies that $u _ { 1 } \cdots u _ { k - 1 }$ divides $d _ { i }$ and $d _ { i }$ divides $u _ { 1 } \cdots u _ { k }$ properly. 

Suppose $k = p$ . We observe that 

$$
u _ {1} \dots u _ {k - 1} \leq d _ {i} = t _ {1} \dots t _ {j - 1} <   t _ {1} \dots t _ {n} \leq u _ {1} \dots u _ {p} = u _ {1} \dots u _ {k}.
$$

Lemma 4.1.6.4 implies that $u _ { 1 } \cdots u _ { k - 1 }$ divides $d _ { i }$ and $d _ { i }$ divides $u _ { 1 } \cdots u _ { k }$ (properly, though we don’t require this). 

• (c) and (d): Again, since $T ^ { \prime }$ refines $T ,$ there exists some $1 \leq b \leq n ^ { \prime }$ such that 

$$
s _ {i} d _ {i} = t _ {1} \dots t _ {j} = t _ {1} ^ {\prime} \dots t _ {b} ^ {\prime} = u _ {1} ^ {\prime} \dots u _ {b} ^ {\prime}.
$$

Take the maximal $\ell \in \langle p \rangle$ such that $u _ { 1 } \cdot \cdot \cdot u _ { \ell - 1 } \leq u _ { 1 } ^ { \prime } \cdot \cdot \cdot u _ { b } ^ { \prime } .$ 

Suppose $\ell < p .$ . We observe that 

$$
u _ {1} \dots u _ {\ell - 1} \leq s _ {i} d _ {i} <   u _ {1} \dots u _ {\ell}.
$$

where the second inequality holds by maximality of $\ell \in \langle p \rangle$ . Lemma 4.1.6.4 implies that $u _ { 1 } \cdots u \ell - 1$ divides $s _ { i } d _ { i }$ and $s _ { i } d _ { i }$ divides $u _ { 1 } \cdot \cdot \cdot u _ { \ell }$ properly. 

– Suppose $\ell = p$ . We observe that 

$$
u _ {1} \dots u _ {\ell - 1} \leq s _ {i} d _ {i} = t _ {1} \dots t _ {j} \leq t _ {1} \dots t _ {n} \leq u _ {1} \dots u _ {p} = u _ {1} \dots u _ {k}.
$$

Lemma 4.1.6.4 implies that $u _ { 1 } \cdot \cdot \cdot u _ { k - 1 }$ divides $d _ { i }$ and $d _ { i }$ divides $u _ { 1 } \cdots u _ { k }$ 

2. For any $i \neq i ^ { \prime }$ in ⟨m⟩, we have $d _ { i } = t _ { 1 } \cdot \cdot \cdot t _ { j - 1 } , s _ { i } = t _ { j } , d _ { i ^ { \prime } } = t _ { 1 } \cdot \cdot \cdot t _ { j ^ { \prime } - 1 }$ , and $s _ { i ^ { \prime } } = t _ { j ^ { \prime } }$ , where $j = \alpha ( i )$ and $j ^ { \prime } = \alpha ( i ^ { \prime } )$ . We then have 

$$
[ d _ {i}, d _ {i} (s _ {i} - 1) ] = [ t _ {1} \dots t _ {j - 1}, t _ {1} \dots t _ {j - 1} (t _ {j} - 1) ]
$$

and 

$$
[ d _ {i ^ {\prime}}, d _ {i ^ {\prime}} (s _ {i ^ {\prime}} - 1) ] = [ t _ {1} \dots t _ {j ^ {\prime} - 1}, t _ {1} \dots t _ {j ^ {\prime} - 1} (t _ {j ^ {\prime}} - 1) ]
$$

If $j ^ { \prime } > j$ , then 

$$
t _ {1} \dots t _ {j - 1} (t _ {j} - 1) <   t _ {1} \dots t _ {j ^ {\prime} - 1}
$$

so the intervals do not overlap, and similarly if $j < j ^ { \prime }$ 

## 4.2 Logical division and logical product

In this section we illustrate how the composition algorithm 4.1.3 can be used to compute logical division and logical product. 

## 4.2.1 Logical division examples

Recall that if A and B are layouts, the logical division $A \oslash B$ is defined as 

$$
A \oslash B = A \circ (B, B ^ {c})
$$

where 

$$
B ^ {c} = \operatorname{comp} (B, \text { size } (A)).
$$

Example 4.2.1.1. Suppose we want to compute the logical division $A \oslash B$ where $A = ( 8 , 8 ) : ( 8 , 1 )$ and $B = ( 2 , 2 ) : ( 1 , 4 )$ . Then we can write $A = L _ { g } , B = L _ { h }$ and $B ^ { c } = L _ { h ^ { c } }$ c where f and $f ^ { c }$ are the tuple morphisms shown below. 

![image](Imgaes/categorical-foundations-cute-layouts-paper/29cd8e36011b3e73d2052609c8baedc358baee60c2a147728110fc2194dcc829.jpg)


It follows that $( B , B ^ { c } )$ is encoded by the nested tuple morphism $f = \left( h , h ^ { c } \right)$ shown below. 

![image](Imgaes/categorical-foundations-cute-layouts-paper/4fa71e21d0fd94aaf12e70f46b663049a8bf22892a989793e56835faa36d52b7.jpg)


We then proceed with our composition algorithm as before. We use algorithm 4.1.1 to find the mutual refinement 

![image](Imgaes/categorical-foundations-cute-layouts-paper/021f589259d5c234ce128c749af632a0d867d109e87d4e94e2a3a30e07185274.jpg)


form the diagram 

![image](Imgaes/categorical-foundations-cute-layouts-paper/634d1f736f363ed0ca8b505fd103b0c4fa3b28b8a7c07163ff69bdb77237d445.jpg)


resolve the diagram to obtain 

![image](Imgaes/categorical-foundations-cute-layouts-paper/dfd3d4e6c03fcb9209a49e8caf98ce8fb79bc3178d512a3d577edec9dcaccfff.jpg)


and compose f and $g ^ { \prime }$ to obtain 

![image](Imgaes/categorical-foundations-cute-layouts-paper/92fc0604a9403d86eb6cbc9901718825a403ff5bc10e8833240af8538f0b1a8b.jpg)


The layout encoded by this nested tuple morphism is 

$$
L _ {g ^ {\prime} \circ f} = ((2, 2), (2, 2)): ((8, 3 2), (1 6, 1))
$$

which is coalesced over ((2, 2), (2, 2)), so we conclude that 

$$
A \oslash B = ((2, 2), (2, 2)): ((8, 3 2), (1 6, 1)).
$$

## 4.2.2 Logical product examples

Recall that if A and B are layouts, the logical product A ⊗ B is defined as 

$$
A \otimes B = (A, A ^ {c} \circ B)
$$

where 

$$
A ^ {c} = \operatorname{comp} (A, \text { size } (A) \cdot \text { cosine } (B)).
$$

In particular, if we want to compute $A \otimes B$ by hand, it sufices to compute $A ^ { c } \circ B$ , and then concatenate the result with A. 

Example 4.2.2.1. Suppose we want to compute the logical product A ⊗ B where $A = ( 2 , 2 ) : ( 1 , 2 )$ and $B = ( 5 , 5 ) : ( 5 , 1 )$ . Then 

$$
\begin{array}{r l} A ^ {c} & = \text { comp } (A, \text { size } (A) \cdot \text { cosize } (B)) \\ & = \text { comp } (A, 1 0 0) \\ & = (2 5): (4). \end{array}
$$

We proceed as in the previous section. 

1. Take the standard representations of B and coa $| ( A ^ { c } ) = A ^ { c }$ 

![image](Imgaes/categorical-foundations-cute-layouts-paper/7a4ec91297944d91a02f45ede18a8ebf85b61e0b03a9e694e3b87b43387f482f.jpg)


2. Apply Algorithm 4.1.1 to obtain the mutual refinement 

$$
\begin{array}{l} 5 - 5 \\ 5 - 5 \end{array} \searrow 2 5
$$

3. Form the diagram 

![image](Imgaes/categorical-foundations-cute-layouts-paper/f37c36d6be520cc4080dc474921ae202fa7e7b8792fee6f9450248f6a31ed35f.jpg)


4. Resolve the diagram to obtain 

![image](Imgaes/categorical-foundations-cute-layouts-paper/5867b5a4106b1f562bf5b5868ef6ce46a629117dc825a51d2ba6a015073b44c9.jpg)


5. Compose $f ^ { \prime }$ and $g ^ { \prime }$ to obtain 

![image](Imgaes/categorical-foundations-cute-layouts-paper/35bc8fa41c3f2d1377826a9de4897a9b71a9749f8238da972a17f217b8ac2673.jpg)


6. Compute the encoded layout 

$$
L _ {g ^ {\prime} \circ f ^ {\prime}} = (5, 5): (2 0, 4).
$$

7. The layout $L _ { g ^ { \prime } \circ f ^ { \prime } }$ is coalesced over (5, 5), so 

$$
A ^ {c} \circ B = (5, 5): (2 0, 4).
$$

We conclude that 

$$
A \otimes B = ((2, 2), (5, 5)): ((1, 2), (2 0, 4)).
$$

## Appendix A

# An introduction to categories

Throughout this work, we freely use the language of categories which are mathematical objects which abstract the notion of morphisms and their composition. The purpose of this appendix is to provide a concise and user-friendly introduction to the basics of categories. In particular, we aim to the answer the following questions: 

1. What is a category? 

2. What is a functor? 

Those capable of answering these questions with confidence, and with examples in mind, will be able to understand the most important conepts and constructions in the current work. For those interested in learning the more advanced concepts from category theory, such as natural transformations, adjunctions, and (co)limits, we recommend [13]. 

## A.1 What is a category?

We begin by addressing the first question. Before giving a definition, let’s consider a motivating example. Suppose X and $Y$ are sets. A function $f : X \to Y$ assigns to each element $x \in X$ some element $f ( x ) \in Y$ . We refer to X as the domain of f and to g as the codomain of Y. 

Example A.1.0.1. There is a function $f : \mathbb { Z } \to \mathbb { Z }$ given by 

$$
f (x) = 2 x.
$$

Example A.1.0.2. There is a function $g : \mathbb { Z } \to \mathsf { B o o l }$ , where $\mathsf { B o o l } = \{ \mathbf { T r u e } , \mathbf { F a l s e } \}$ , given by 

$$
g (x) = \left\{ \begin{array}{l l} \text { True } & x \text {   is   even }, \\ \text { False } & x \text {   is   odd }. \end{array} \right..
$$

If $f : X \to Y$ and $g : Y  Z$ are functions, then we can compose $f$ and $g \colon$ The composite of $f$ and $g$ is the function $g \circ f : X \to Z$ given by 

$$
(g \circ f) (x) = g (f (x)).
$$

Example A.1.0.3. If f and g are the functions of Examples A.1.0.1 and A.1.0.2, then the composite $g \circ f : \mathbb { Z } \to$ Bool is given by 

$$
(g \circ f) (x) = \mathbf {T r u e}.
$$

Composition of functions satisfies two essential properties. First, composition is associative: if f and g are composable, and g and h are composable, then 

$$
h \circ (g \circ f) = (h \circ g) \circ f.
$$

Second, every set X has an identity function id $x : X \to X$ given by 

$$
\operatorname{id} _ {X} (x) = x.
$$

If $f : X \to Y$ is any function, then precomposing with $\mathsf { i d } _ { X }$ or post-composing with id<sub>Y</sub> leaves the function f unchanged: 

$$
f \circ \mathrm{id} _ {X} = f = \mathrm{id} _ {Y} \circ f.
$$

In pure and applied mathematics, there are many instances where we have some collection of objects, and morphisms between those objects, which have the same formal behavior of sets and functions: morphisms can be composed in an associative fashion, and objects admit identity morphisms. While functions between sets are the prototypical example, the objects in a category need not be sets, and the morphisms in a category need not be functions. We will see many such examples later on. To capture this recurring structure, we define the notion of a category. 

## Definition A.1.0.4. A category C consists of

1. a collection of objects: 

$$
\mathsf {o b} (\mathbf {C}) = \{X, Y, Z, \dots \}.
$$

These objects may be sets, tuples, numbers, vector spaces, matrices, or some other mathematical structure, depending on the category C. 

2. a collection of morphisms between those objects: 

$$
\operatorname{mor} (\mathbf {C}) = \{f, g, h, \dots \}.
$$

Each morphism $f : X \to Y$ in C has a domain X and a codomain Y, which are objects in C. 

3. a composition rule: If $f : X \to Y$ and $g : Y  Z$ are morphisms in C, then there is a morphism 

$$
g \circ f: X \to Z
$$

called the composite of f and g. Composition of morphisms in C is associative, in that 

$$
h \circ (g \circ f) = (h \circ g) \circ f,
$$

when defined. 

4. identity morphisms: If X is an object in C, then there is a morphism 

$$
\mathrm{id} _ {X}: X \to X
$$

called the identity morphism on X. If $f : X \to Y$ is any morphism in C, then 

$$
f \circ \mathrm{id} _ {X} = f = \mathrm{id} _ {Y} \circ f.
$$

Let’s take a look at some important examples of categories. We begin with the motivating example. 

Example A.1.0.5. There is a category Set whose objects are sets, and whose morphisms are functions. The composition of morphisms is given by functional composition: 

$$
(g \circ f) (x) = g (f (x))
$$

and the identity morphism on a set X is the identity function 

$$
\operatorname{id} _ {X} (x) = x.
$$

Example A.1.0.6. There is a category Vect whose objects are the vector spaces $\mathbb { R } ^ { n }$ for $n \geq 0$ , and whose morphisms are matrices. Specifically, a morphism 

$$
A: \mathbb {R} ^ {n} \to \mathbb {R} ^ {m}
$$

in Vect is ${ \mathrm { ~ a ~ } } m \times n$ matrix A. Composition in Vect is given by taking matrix products: 

$$
B \circ A = B A,
$$

and the identity morphism on $\mathbb { R } ^ { n }$ is the $n \times n$ matrix 

$$
\mathsf {i d} _ {\mathbb {R} ^ {n}} = I _ {n} = \left[ \begin{array}{c c c c c} 1 & 0 & \dots & & 0 \\ 0 & 1 & & & \\ \vdots & & \ddots & & \vdots \\ & & & 1 & 0 \\ 0 & & \dots & 0 & 1 \end{array} \right].
$$

Example A.1.0.7. There is a category Div whose objects are integers $a \ge 1$ , and in which there is a unique morphism 

$$
\mathsf {d i v} _ {a} ^ {b}: a \to b
$$

if a divides b. If a divides b and b divides c, then a divides c, which means that we have a well defined composition rule 

$$
\mathsf {d i v} _ {b} ^ {c} \circ \mathsf {d i v} _ {a} ^ {b} = \mathsf {d i v} _ {a} ^ {c},
$$

and the identity morphism 

$$
\mathsf {i d} _ {a} = \mathsf {d i v} _ {a} ^ {a}
$$

exists since every positive integer a divides itself. 

In addition to the definition of a category, there are a few important categorical concepts that we need to understand. For instance, it is important to understand the notion of an isomorphism, which generalizes the notion of a bijection of sets. 

Definition A.1.0.8. Suppose C is a category, and suppose $f : X \to Y$ is a morphism in C. We say f is an isomorphism if there exists a morphism $f ^ { - 1 } : Y \to X$ in C such that 

1. $f ^ { - 1 } \circ f = \mathsf { i d } _ { X }$ , and 

2. $f \circ f ^ { - 1 } = \mathsf { i d } _ { Y } .$ 

Example A.1.0.9. In the category Set, an isomorphism is a bijection: a function $f : X \to Y$ such that for each $y \in Y$ , there exists a unique $x \in X$ with $f ( x ) = y$ . For example, the function $f : \mathbb { Z } \to \mathbb { Z }$ given by 

$$
f (x) = x + 1 0
$$

is a bijection, with inverse $f ^ { - 1 } : \mathbb { Z } \to \mathbb { Z }$ given by 

$$
f ^ {- 1} (x) = x - 1 0.
$$

Example A.1.0.10. In the category Vect, an isomorphism is an invertible matrix. For example, the matrix 

$$
A = \left[ \begin{array}{c c} 3 & 2 \\ 1 & 1 \end{array} \right]
$$

is invertible with inverse 

$$
A ^ {- 1} = \left[ \begin{array}{c c} 1 & - 2 \\ - 1 & 3 \end{array} \right]
$$

since 

$$
A ^ {- 1} A = \left[ \begin{array}{c c} 1 & 0 \\ 0 & 1 \end{array} \right] = A A ^ {- 1}
$$

Example A.1.0.11. In the category Div, the only isomorphisms are the identity morphisms 

$$
\mathsf {i d} _ {a} = \mathsf {d i v} _ {a} ^ {a}.
$$

This is because if a divides b and b divides $^ { a , }$ then $a = b$ 

## A.2 What is a functor?

Next, we turn our attention to the second question. 

Definition A.2.0.1. Suppose C and D are categories. A functor $F : { \mathsf { C } } \to { \mathsf { D } }$ consists of 

1. for each object X in C, an object FX in D, and 

2. for each morphism $f : X \to Y$ in C, a morphism 

$$
F f: F X \to F Y
$$

in D, 

satisfying the following properties: 

1. $F$ is compatible with composition: If $f$ and g are composable morphisms in $\mathbf { c } ,$ then 

$$
F (g \circ f) = F g \circ F f.
$$

2. $F$ is compatible with identities: If X is an object in C, then 

$$
F \mathrm{id} _ {X} = \mathrm{id} _ {F X}.
$$

Example A.2.0.2. There is a functor F : Div → Set defined as follows. On objects, F is given by 

$$
F a = [ 0, a ] = \{x \in \mathbb {R} \mid 0 \leq x \leq a \}.
$$

and on morphisms, $F$ is given by 

$$
F \mathsf {d i v} _ {a} ^ {b} (x) = \frac {b}{a} \cdot x.
$$

Let’s verify that F is a functor. 

1. $F$ is compatible with composition: If a divides b and b divides $^ { c , }$ then 

$$
(F \mathsf {d i v} _ {b} ^ {c} \circ F \mathsf {d i v} _ {a} ^ {b}) (x) = F \mathsf {d i v} _ {b} ^ {c} (F \mathsf {d i v} _ {a} ^ {b} (x)) = \frac {c}{b} \cdot (\frac {b}{a} \cdot x) = \frac {c}{a} \cdot x = F \mathsf {d i v} _ {a} ^ {c} (x).
$$

2. $F$ is compatible with identities: If $a \ge 1$ , then 

$$
F \mathsf {i d} _ {a} (x) = F \mathsf {d i v} _ {a} ^ {a} (x) = \frac {a}{a} \cdot x = \mathsf {i d} _ {F a} (x).
$$

## Bibliography



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

