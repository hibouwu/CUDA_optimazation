## A note on the algebra of CuTe Layouts Jay Shah†

## 1 INTRODUCTION

The core abstraction of NVIDIA’s CUTLASS library for high-performance linear algebra is a specific notion of layout, introduced as part of its new backend core library CuTe in version 3.0 [1]. CuTe Layouts comprise a convenient formalism for describing and manipulating data of a multi-dimensional nature, such as the values of a matrix or tensor. The goal of this technical note is to study CuTe Layouts from a rigorous, mathematical point of view. Currently, the focus is on articulating sufficient conditions for the operations of complementation and composition to be well-defined, and also to provide explicit closed formulae for them. These operations play an important role in and of themselves, but also jointly in defining the operation of logical division. This operation, as well as its relatives such as zipped division, plays a critical role in various tiling and slicing operations for CuTe Layouts and Tensors (which are essentially Layouts together with pointers into memory). 

This note should be read as complementary to the discussion of layout operations in the CuTe documentation [2]. However, we think that certain portions of that documentation are mathematically vague or false if interpreted literally, which spurred the writing of this note. Most significantly, no discussion of necessary conditions for the operation of composition to be well-defined is given there. This becomes problematic when, for example, it is claimed that composition is left-distributive with respect to concatenation.1 In code, this is given as a definition of composition in the general case. But consider the simple example of

```txt
Layout A = make_layout(make_shape(_6{},_2{}),make_stride(_1{},_7{}));  
Layout B = make_layout(make_shape(_3{},_2{}),make_stride(_2{},_3{}));  
Layout C = composition(A,B); 
```

Then when running with CUTLASS 3.3, the layout $C$ evaluates to

$(\_3, \_2): (\_2, \_3)$ 

since $C$ is defined according to the supposed left-distributivity property. But note that $C$ doesn’t actually describe the composition of $A$ and $B$ in terms of the associated layout functions $f_A$, $f_B$, and $f_C$. Indeed, we have that

$$
f _ {C} (5) = f _ {C} (2) + f _ {C} (3) = 4 + 3 = 7,
$$

whereas 

$$
(f _ {A} \circ f _ {B}) (5) = f _ {A} (7) = f _ {A} (1) + f _ {A} (6) = 1 + 7 = 8.
$$

Actually, in this case $A \circ B$ will not be well-defined as a layout, even though for the separate modes $B_0$ and $B_1$ of $B$, the compositions $A \circ B_0$ and $A \circ B_1$ are well-defined. This “overflow” issue occurs since a certain disjointness condition is violated, which we articulate as Definition 2.17. Of course, in practice the programmer would not consider such a composition to begin with, but we hope that our note can serve as an all-purpose reference for when such operations are meant to be valid. However, we emphasize that the treatment of layouts given in this note is entirely implementation-agnostic.

The contents of the current note form a self-contained body of work as it stands, although it will appear unmotivated if the reader doesn’t already have prior experience working with CuTe Layouts. We anticipate adding to this document as the need arises, or if elaborations of other aspects of layout algebra are desired by CUTLASS/CuTe developers. 

1Item (3) in “Rules for computing composition” from [2]. 

## 2 LAYOUT ALGEBRA

Definition 2.1. A layout $L$ is a pair of positive integer tuples $\mathbf{S}$ and $\mathbf{D}$ of matching dimensions. We call $\mathbf{S}$ the shape and $\mathbf{D}$ the stride. We write $L=\mathbf{S}:\mathbf{D}$.

From now on in this note, we assume that layouts are flattened (i.e., internal parentheses for $\mathbf{S}$ and $\mathbf{D}$ have been removed); this won’t change the semantics of the operations that we consider. Let’s first introduce some basic terminology:

Definition 2.2. Let $\alpha \geq 0$ be an integer and $L = \mathbf { S } : \mathbf { D } = \left( M _ { 0 } , . . . , M _ { \alpha } \right) : \left( d _ { 0 } , . . . , d _ { \alpha } \right)$ be a layout. Then: 

• The size of $L$ is the product $M=M_0\cdot\ldots\cdot M_\alpha$.

• The length of $L$ is the integer $\alpha+1$.

• A mode of $L$ is one of the entries $(M_k):(d_k)$ for $0\leq k\leq\alpha$. We may regard this as a length 1 layout.

Given two layouts $L=\mathbf{S}:\mathbf{D}$ and $L'=\mathbf{S}':\mathbf{D}'$, let $\mathbf{S}''$ and $\mathbf{D}''$ be the shape and stride tuples given by the flattening of $(\mathbf{S},\mathbf{S}')$ and $(\mathbf{D},\mathbf{D}')$. Then the concatenation of $L$ and $L'$ is given by the layout

$$
(L, L ^ {\prime}) := \mathbf {S} ^ {\prime \prime}: \mathbf {D} ^ {\prime \prime},
$$

and we say that $(L,L')$ is decomposed by $L$ and $L'$. Inductively, given layouts $L_1,\ldots,L_N$, we can form the concatenated layout $(L_1,\ldots,L_N)$. Conversely, a layout $L$ is maximally decomposed by its modes.

To each layout $L$, we can associate a function as follows. Let $\mathbf{S}=(M_0,\ldots,M_\alpha)$ and $\mathbf{D}=(d_0,\ldots,d_\alpha)$ be the respective shape and stride tuples for $L$. Let $M=M_0M_1\cdots M_\alpha$ be the size of $L$ and let $[0,M)\subset\mathbb{N}$ be the subset $\{0,\ldots,M-1\}$. Then we have an isomorphism

$$
\iota : [ 0, M) \cong [ 0, M _ {0}) \times [ 0, M _ {1}) \times \dots \times [ 0, M _ {\alpha})
$$

given by $x\mapsto(x\bmod M_0,\lfloor x/M_0\rfloor\bmod M_1,\ldots,\lfloor x/(M_0\cdots M_{\alpha-1})\rfloor\bmod M_\alpha)$.

Definition 2.3. Given a layout $L$, its layout function $f_L:[0,M)\to\mathbb{N}$ is defined to be the composite

$$
[ 0, M) \cong [ 0, M _ {0}) \times \ldots \times [ 0, M _ {\alpha}) \subset \mathbb {N} ^ {\times (\alpha + 1)} \xrightarrow {(\cdot d _ {0} , \ldots , \cdot d _ {\alpha})} \mathbb {N} ^ {\times (\alpha + 1)} \xrightarrow {+} \mathbb {N}.
$$

In other words, $f _ { L }$ is the composition of the multi-linear function 

$$
\left[ 0, M _ {0}\right) \times \ldots \times \left[ 0, M _ {\alpha}\right) \to \mathbb {N}, \quad \left(x _ {0}, \dots , x _ {\alpha}\right) \mapsto d _ {0} x _ {0} + \dots + d _ {\alpha} x _ {\alpha},
$$

determined by the stride, with the isomorphism $\iota$, determined by the shape.

We then let $\widehat { f _ { L } } : \mathbb { N } \to$ N be the extension of $f _ { L }$ defined by replacing $M _ { \alpha }$ by $\infty ,$ i.e., the composite 

$$
\mathbb {N} \cong [ 0, M _ {0}) \times \ldots \times [ 0, M _ {\alpha - 1}) \times \mathbb {N} \subset \mathbb {N} ^ {\times (\alpha + 1)} \xrightarrow {(\cdot d _ {0} , . . . , \cdot d _ {\alpha})} \mathbb {N} ^ {\times (\alpha + 1)} \xrightarrow {+} \mathbb {N}
$$

where the first isomorphism is the extension $\bar\iota$ of $\iota$ given by

$$
x \mapsto (x \bmod M _ {0}, \left\lfloor \frac {x}{M _ {0}} \right\rfloor \bmod M _ {1}, \dots , \left\lfloor \frac {x}{M _ {0} \cdot \dots \cdot M _ {\alpha - 2}} \right\rfloor \bmod M _ {\alpha - 1}, \left\lfloor \frac {x}{M _ {0} \cdot \dots \cdot M _ {\alpha - 1}} \right\rfloor).
$$

## 2.1 Complementation

In this subsection, we define the notion of the complement of a layout $A$ with respect to a given integer $M$, under certain assumptions.

Definition 2.4. Let $A=(N_0,\ldots,N_\alpha):(d_0,\ldots,d_\alpha)$ be a layout. We say that $A$ is sorted if $d_0\leq\ldots\leq d_\alpha$ and for every $i<j$ such that $d_i=d_j$, $N_i\leq N_j$.

Definition 2.5. Let $A=(N_0,\ldots,N_\alpha):(d_0,\ldots,d_\alpha)$ be a layout and $M$ a positive integer. Suppose without loss of generality that $A$ is sorted; if not, replace $A$ with a permutation of itself that is sorted. Then we say that the pair $\{A,M\}$ is admissible for complementation (or simply admissible) if:

• For all $1 \leq i \leq \alpha$ , the product $N _ { i - 1 } d _ { i - 1 }$ divides $d _ { i }$ . 

• The product $N_\alpha d_\alpha$ divides $M$.

Definition 2.6. Let $A=(N_0,\ldots,N_\alpha):(d_0,\ldots,d_\alpha)$ be a layout and $M$ a positive integer. Suppose that $\{A,M\}$ is admissible for complementation and reindex $A$ so that it is sorted. Then the complement of $\{A,M\}$ is defined to be the layout

$$
\operatorname{complement} (A, M) = \left(d _ {0}, \frac {d _ {1}}{N _ {0} d _ {0}}, \frac {d _ {2}}{N _ {1} d _ {1}}, \dots , \frac {M}{N _ {\alpha} d _ {\alpha}}\right): \left(1, N _ {0} d _ {0}, N _ {1} d _ {1}, \dots , N _ {\alpha} d _ {\alpha}\right).
$$

Note that by definition, the complement of $A$ (taken with respect to some integer $M$) is insensitive to permutations of $A$. Moreover, its layout function is strictly increasing.

The following proposition explains the sense in which Definition 2.6 is taking a complement. 

Proposition 2.7. Let $\{A=(N_0,\ldots,N_\alpha):(d_0,\ldots,d_\alpha),M\}$ be an admissible pair and $B=\operatorname{complement}(A,M)$. Let $C=(A,B)$ be the concatenated layout. Then the size of $C$ is $M$ and $f_C:[0,M)\to\mathbb{N}$ restricts to a bijection $[0,M)\cong[0,M)$.

Proof. Since $\operatorname{size}(A)\operatorname{size}(B)=M$, the domain of $f_C$ is indeed $[0,M)$. The image of $f_C$ is the same as that of $f_{C'}$ for any permutation $C'$ of $C$. Therefore, when computing the image of $f_C$, we may sort $C$ so that the strides are in non-decreasing order, as well as reindex $A$ so that it is sorted. After reindexing $C$, let

$$
C ^ {\prime} = (d _ {0}, N _ {0}, \frac {d _ {1}}{N _ {0} d _ {0}}, N _ {1}, \frac {d _ {2}}{N _ {1} d _ {1}}..., N _ {\alpha}, \frac {M}{N _ {\alpha} d _ {\alpha}}): (1, d _ {0}, N _ {0} d _ {0}, d _ {1}, N _ {1} d _ {1},..., d _ {\alpha}, N _ {\alpha} d _ {\alpha}).
$$

Then we may write 

$$
C ^ {\prime} = \left(r _ {0}, r _ {1}, r _ {2},..., r _ {\beta}\right): \left(1, r _ {0}, r _ {0} r _ {1},..., r _ {0}... r _ {\beta - 1}\right)
$$

for $\beta = 2 \alpha + 1$ , and the maximum value that $f _ { C }$ attains is computed as 

$$
(r _ {0} - 1) + r _ {0} (r _ {1} - 1) + (r _ {0} r _ {1}) (r _ {2} - 1) + \dots + (r _ {0} \dots r _ {\beta - 1}) (r _ {\beta} - 1) = r _ {0} r _ {1} \dots r _ {\beta} - 1 = M - 1.
$$

To establish the bijectivity assertion, it then suffices to show that $f _ { C ^ { \prime } }$ is injective. For this, suppose that $x , y \in [ 0 , M )$ so that $f _ { C ^ { \prime } } ( x ) = f _ { C ^ { \prime } } ( y )$ , and let $( x _ { 0 } , . . . , x _ { \beta } )$ and $( y _ { 0 } , . . . , y _ { \beta } )$ be their coordinate vectors with respect to $C ^ { \prime }$ . Expanding the terms in the equality we get 

$$
x _ {0} + r _ {0} x _ {1} + (r _ {0} r _ {1}) x _ {2} + \ldots + (r _ {0} \ldots r _ {\beta - 1}) x _ {\beta} = y _ {0} + r _ {0} y _ {1} + (r _ {0} r _ {1}) y _ {2} + \ldots + (r _ {0} \ldots r _ {\beta - 1}) y _ {\beta}.
$$

We show by induction that $x _ { i } = y _ { i }$ for all $i \in \{ 0 , . . . , \beta \}$ , which will complete the proof. Firstly, taking both sides mod $r _ { 0 }$ shows that $x _ { 0 } = y _ { 0 }$ since both lie in $[ 0 , r _ { 0 } )$ . Now suppose by induction that given $0 < i \leq \beta ,$ , for all $j < i$ we have $x _ { j } = y _ { j }$ . Then we can reduce the expression to 

$$
(r _ {0} \dots r _ {i - 1}) x _ {i} + \dots + (r _ {0} \dots r _ {\beta - 1}) x _ {\beta} = (r _ {0} \dots r _ {i - 1}) y _ {i} + \dots + (r _ {0} \dots r _ {\beta - 1}) y _ {\beta}.
$$

Taking this equation mod $r _ { 0 } . . . r _ { i }$ and dividing by $\left( r _ { 0 } . . . r _ { i - 1 } \right)$ shows that $x _ { i } = y _ { i }$ , since we know both lie in $[ 0 , r _ { i } )$ . □ 

Corollary 2.8. In the setting of Proposition 2.7, let $ I = [ 0 , \mathrm { s i z e } ( A ) ) = [ 0 , N _ { 0 } . . . N _ { \alpha } )$ be the domain of $\dot { \boldsymbol { f } _ { A } }$ . Then 

$$
f _ {A} (I) \cap \widehat {f _ {B}} (I) = \{0 \}.
$$

In other words, $\widehat { f } _ { A }$ and $\widehat { f } _ { B }$ have disjoint image when restricted to the domain of $f _ { A }$ , apart from 0. 

Proof. Let $J = [ 0 , \mathsf { s i z e } ( B ) ) = [ 0 , M / ( N _ { 0 } . . . N _ { \alpha } ) )$ . By Proposition 2.7, we have that 

$$
f _ {A} (I \cap J) \cap f _ {B} (I \cap J) = \{0 \}.
$$

It remains to consider values of the extended function $\widehat { f } _ { B }$ on integers that might lie in $I$ but not $J .$ But $\widehat { f } _ { B }$ is a strictly increasing function, ${ \widehat { f } } _ { B } ( { \mathrm { s i z e } } ( B ) ) = M$ , and the largest value attained by $f _ { A }$ satisfies the inequality 

$$
\begin{array}{r l} & {(N _ {0} - 1) d _ {0} + (N _ {1} - 1) d _ {1} + \ldots + (N _ {\alpha} - 1) d _ {\alpha} <   d _ {1} + (N _ {1} - 1) d _ {1} + (N _ {2} - 1) d _ {2} + \ldots + (N _ {\alpha} - 1) d _ {\alpha}} \\ & {\qquad \leq d _ {2} + (N _ {2} - 1) d _ {2} + \ldots + (N _ {\alpha} - 1) d _ {\alpha} \leq \ldots} \\ & {\qquad \leq d _ {\alpha} + (N _ {\alpha} - 1) d _ {\alpha} \leq M.} \end{array}
$$

□ 

Remark 2.9. The CuTe documentation [2] stipulates that the complement $B$ of a layout $A$ with respect to an integer $M$ should satisfy three properties: 

(1) $A$ and $B$ are disjoint in the sense that $f _ { A } ( x ) \neq f _ { B } ( x )$ for all $x \neq 0$ in the domain of $f _ { A } { \mathrm { ; } }$ ; 

(2) $B$ is ordered in the sense that $f _ { B }$ is a strictly increasing function; 

(3) $B$ is bounded by $M$ in the sense that size $( B ) \geq M / \mathrm { s i z e } ( A )$ and cosize $\begin{array} { r } { ( B ) \leq \left\lfloor \frac { M } { \mathrm { c o s i z e } ( A ) } \right\rfloor } \end{array}$ · cosize($A$). Here, we let the cosize of a layout $A$ be given by $f _ { A } ( { \mathrm { s i z e } } ( A ) - 1 ) + 1$ 

We observe that all these properties are satisfied by the definition of complement given in Definition 2.6 for $\{ A , M \}$ admissible. (1) follows from Corollary $2 . 8 . ^ { 3 } \left( 2 \right)$ follows by definition of the complement as we noted above. Finally, for (3) we have that size $( B ) = M / \mathrm { s i z e } ( A )$ and 

$$
\begin{array}{r l} \mathrm{cosize} (B) & = 1 + (d _ {0} - 1) + \left(\frac {d _ {1}}{N _ {0} d _ {0}} - 1\right) N _ {0} d _ {0} + \ldots + \left(\frac {M}{N _ {\alpha} d _ {\alpha}} - 1\right) N _ {\alpha} d _ {\alpha} \\ & = d _ {0} + (d _ {1} - N _ {0} d _ {0}) + \ldots + (d _ {\alpha} - N _ {\alpha - 1} d _ {\alpha - 1}) + M - N _ {\alpha} d _ {\alpha} \\ & = M - ((N _ {0} - 1) d _ {0} + \ldots + (N _ {\alpha} - 1) d _ {\alpha}) \\ & = M - (\mathrm{cosize} (A) - 1), \end{array}
$$

where we reindexed $C$ according to its sort for the intermediate terms; this doesn’t change the final equality. Therefore, the inequality to check for the cosizes becomes 

$$
\frac {M}{\operatorname{cosize} (A)} - 1 + \frac {1}{\operatorname{cosize} (A)} \leq \left\lfloor \frac {M}{\operatorname{cosize} (A)} \right\rfloor ,
$$

which holds for any pair of positive integers. 

Example 2.10. We give two examples in CUTLASS 3.3 for when CuTe’s complement method can be evaluated but has potentially undesired behavior. Consider the layout $A = ( 4 ) : ( 2 )$ and $M = 1 9 ,$ , so we don’t have that $\{ A , M \}$ is admissible. Then complement $( \mathsf { A } , \mathsf { M } )$ evaluates to 

$$
(_ {2}, _ {3}): (_ {1}, _ {8})
$$

However, in this case cosize($B$) = 18, whereas cosize $( A ) = 7$ and thus 

$$
\left\lfloor \frac {M}{\mathrm{cosize} (A)} \right\rfloor \cdot \mathrm{cosize} (A) = \left\lfloor \frac {1 9}{7} \right\rfloor \cdot 7 = 2 \cdot 7 = 1 4.
$$

Now consider $A = ( 2 , 2 ) : ( 2 , 3 )$ and $M = 1 9$ . Then complement(A, M)$ evaluates to 

$(\_2, \_0, \_4): (\_1, \_4, \_6)$ 

which is the empty layout (with size $( B ) = 0 )$ , since 0 occurs in its shape tuple. 

## 2.2 Composition

We next discuss the operation of composition of layouts $A$ and $B$. For simplicity, we suppose that the shape tuples contain no integers equal to 1; stripping out these modes doesn’t change the associated layout function. The goal here is to produce a layout, denoted $A \circ B ,$ , whose associated function $f _ { A \circ B }$ identifies with the composition ${ \widehat { f } } _ { A } \circ f _ { B }$ . In general, we need conditions in order to be able to define $A \circ B$ 

Definition 2.11. Let $M , d > 0$ be positive integers and let $M = M _ { 0 } \cdot M _ { 1 } \cdot \ldots \cdot M _ { \alpha }$ be a given factorization of $M$ by integers $M _ { k } > 1$ . Replacing $M _ { \alpha }$ by ∞, let 

$$
\widehat {M} = M _ {0} \cdot M _ {1} \cdot \ldots \cdot M _ {\alpha - 1} \cdot \infty
$$

and consider ∞ to be divisible by every positive integer. We say that $M$ is left divisible by $d$ (implicitly, with respect to the given factorization) if there exists $0 \leq i \leq \alpha$ such that: 

(1) $M _ { 0 } . . . M _ { i - 1 }$ divides $d . ^ { 4 }$ 

(2) Supposing (1), let $c = d / ( M _ { 0 } . . . M _ { i - 1 } ) . ^ { 5 }$ Then if $\dot { \iota } < \alpha ,$ , we require in addition that $1 \leq c < M _ { i }$ 

(3) For (2) in the case $i < \alpha ,$ we require in addition that $c$ also divides $M _ { i }$ 

Note that $i$ is necessarily unique if it exists. In this case, we will refer to $i$ as the division index and write ${ \widehat { M } } = d \cdot { \widehat { M ^ { \prime } } }$ Moreover, we will endow $\widehat { M ^ { \prime } }$ with the following induced factorization: 

(a) I $\begin{array} { r } { 0 \leq i < \alpha , } \end{array}$ then $\widehat { M ^ { \prime } } = M _ { 0 } ^ { \prime } \cdot \ldots \cdot M _ { \alpha - i - 1 } ^ { \prime }$ · ∞ with $M _ { 0 } ^ { \prime } = M _ { i } / c > 1$ and $M _ { j } ^ { \prime } = M _ { i + j }$ for $0 < j < \alpha - i .$ 

(b) $\operatorname { I f } i = \alpha ,$ , then $\widehat { M } = d \cdot \infty$ and we will let $\widehat { M ^ { \prime } } = \infty$ 

Furthermore, we say that $M$ is weakly left divisible by $d$ if there exists $0 \leq i \leq \alpha$ such that the above conditions (1) and (2) hold, but not necessarily (3). Then we still call the (necessarily unique) $i$ the division index as before, but we no longer have the factorization of ${ \widehat { M } } .$ 

Note that in Definition 2.11, the term $\widehat { M ^ { \prime } }$ with its induced factorization can itself be considered for left divisibility or weak left divisibility (with the step of replacing the last factor by ∞ now being superfluous). 

We first consider composition in the restricted case of length 1 layouts for the second layout. To this end, we have the following notion of “admissibility for composition”: 

Definition 2.12. Let $\mathbf { S } = \left( M _ { 0 } , . . . , M _ { \alpha } \right)$ be a shape tuple, let $M = M _ { 0 } . . . M _ { \alpha }$ , and let $B = \left( N \right) : \left( r \right)$ be a layout of length 1. Then we say that the pair $\{ \mathsf { S } , B \}$ is admissible for composition (or simply admissible) if: 

(1) $M$ is left divisible by $r$. Write ${ \widehat { M } } = r \cdot { \widehat { M ^ { \prime } } }$ 

(2) With respect to its induced factorization, $\widehat { M ^ { \prime } }$ is weakly left divisible by $N .$ 

The idea of admissibility is that the composition $A \circ B$ of layouts will entail “dividing $B$ along the modes of $A$”. More precisely, we have the following: 

Definition 2.13. Suppose that $\mathbf { S } = \left( M _ { 0 } , . . . , M _ { \alpha } \right)$ is a shape tuple and $B = \left( N \right) : \left( r \right)$ is a layout of length 1 such that $\{ \mathsf { S } , B \}$ is admissible. Let $\mathbf { D } = \left( d _ { 0 } , . . . , d _ { \alpha } \right)$ be any stride tuple and let $A = \mathbf { S } : \mathbf { D }$ 

As in Definition 2.11, let $M = M _ { 0 } \cdot \ldots \cdot M _ { \alpha }$ and ${ \widehat { M } } = r \cdot { \widehat { M ^ { \prime } } }$ with division index $0 \leq i \leq \alpha .$ . We separate the definition of $A \circ B$ into two cases. First suppose that $0 \leq i < \alpha$ , so that 

$$
r = M _ {0} \cdot \ldots \cdot M _ {i - 1} \cdot c, \quad \widehat {M ^ {\prime}} = M _ {i} / c \cdot \ldots \cdot \infty .
$$

Then if $N \leq M _ { i } / c _ { \ast }$ , we let $A \circ B = ( N ) : ( c d _ { i } )$ . Otherwise, we have that $N = M _ { i } / c \cdot . . . \cdot M _ { j - 1 } \cdot c ^ { \prime }$ (where $c ^ { \prime } < M _ { j }$ if $j \neq \alpha )$ , and we let 

$$
A \circ B = \left\{ \begin{array}{l l} (M _ {i} / c, M _ {i + 1},..., M _ {j - 1}, c ^ {\prime}): (c d _ {i}, d _ {i + 1},..., d _ {j - 1}, d _ {j}) & \text {if} c ^ {\prime} > 1; \\ (M _ {i} / c, M _ {i + 1},..., M _ {j - 1}): (c d _ {i}, d _ {i + 1},..., d _ {j - 1}) & \text {if} c ^ {\prime} = 1. \end{array} \right.
$$

If instead $i = \alpha ,$ then we have $r = M _ { 0 } \cdot \ldots \cdot M _ { \alpha - 1 } \cdot c$ as before but $\widehat { M ^ { \prime } } = \infty$ , and we let $A$ \circ $B = \left( N \right) : \left( c d _ { \alpha } \right)$ 

Note that by definition the size of $A \circ B$ always equals that of $B$. We then have the following soundness proposition for Definition 2.13. In the proof, we will use the following notation: for a given index $0 \leq k \leq \alpha$ , let $\bar { \boldsymbol \delta } _ { k } \doteq \bar { \mathbb { N } } ^ { \times ( \alpha + 1 ) }$ denote the coordinate that is zero everywhere except in the $k$th position, where it is 1. 

Proposition 2.14. In the situation of Definition 2.13, we have that $f _ { A \circ B } = \widehat { f } _ { A } \circ f _ { B }$ 

Proof. We carry over notation from Definition 2.13. Then with respect to the isomorphism 

$$
\widehat {\iota}: \mathbb {N} \cong [ 0, M _ {0}) \times \dots \times [ 0, M _ {\alpha - 1}) \times \mathbb {N}
$$

of Definition 2.3, we have that $x$ is sent to $\boldsymbol { c } \cdot \delta _ { i }$ . Thus, we see that 

$$
(\widehat {f _ {A}} \circ f _ {B}) (1) = c d _ {i} = f _ {A \circ B} (1).
$$

In the cases of $i < \alpha$ and $N \leq M _ { i } / c \ \mathrm { o r } \ i = \alpha .$ , this already suffices to show $f _ { A \circ B } = \widehat { f } _ { A } \circ f _ { B }$ . In the remaining case $i < \alpha$ and $N = M _ { i } / c \cdot \ldots \cdot M _ { j - 1 } \cdot c ^ { \prime } ;$ , note that 

$$
\widehat {\iota} ((M _ {i} / c) r) = \delta_ {i + 1}, \widehat {\iota} (M _ {i + 1} (M _ {i} / c) r) = \delta_ {i + 2},..., \widehat {\iota} \bigl (M _ {j - 1}... M _ {i + 1} (M _ {i} / c) r \bigr) = \delta_ {j}.
$$

Therefore, we see that $f _ { A \circ B }$ and ${ \widehat { f } } _ { A } \circ f _ { B }$ agree on values $\{ 1 , \ M _ { i } / c , \ M _ { i + 1 } ( M _ { i } / c ) , \ . . . , \ M _ { j - 1 } . . . . M _ { i + 1 } ( M _ { i } / c ) \}$ (or drop the last term if $c ^ { \prime } = 1 )$ . In view of the multi-linearity properties of both functions,6 this implies that $f _ { A \circ B } = \widehat { f _ { A } } \circ f _ { B }$ . □ 

Example 2.15. Let $A = \left( M _ { 0 } , . . . , M _ { \alpha } \right) : \left( d _ { 0 } , . . . , d _ { \alpha } \right)$ be any layout. For $i = 0 ,$ , let $B _ { 0 } = \left( M _ { 0 } \right) : ( 1 )$ , and for $0 < i \leq \alpha$ let $B _ { i } = \left( M _ { i } \right) : \left( M _ { 0 } \cdot \ldots \cdot M _ { i - 1 } \right)$ . Then $A \circ B _ { i } = \left( M _ { i } \right) : \left( d _ { i } \right)$ 

To extend from the case of length 1 layouts to general layouts for the term $B$ in a putative composition $A \circ B ,$ we will write $B = ( B _ { 0 } , . . . , B _ { \beta } )$ as a concatenation of its modes and then concatenate the resulting compositions $A \circ B _ { 0 } , . . . , A \circ B _ { \beta }$ . For this to yield a correct result in general, we need to avoid potential collisions. 

Definition 2.16. In the situation of Definition 2.12, let $f _ { B } ~ : ~ [ 0 , N ) ~ $ N be the layout function, and let $I = [ r , r ( N - 1 ) ]$ be the interval given by the convex closure of the image $f _ { B } \left( \left[ 1 , N \right) \right)$ . Let $M ^ { \prime } = M _ { 0 } . . . M _ { \alpha - 1 }$ and $J = I \cap [ 1 , M ^ { \prime } ) ( \thinspace \mathrm { s o } \ J = \emptyset \ \mathrm { i f } \ \alpha = 0 )$ . Then the interval of definition for {$\mathbf{S}$, $B$} is $J$. 

Definition 2.17. Let $\mathbf { S } = ( M _ { 0 } , . . . , M _ { \alpha } )$ be a shape tuple, let $B = ( N _ { 0 } , . . . , N _ { \beta } ) : ( r _ { 0 } , . . . , r _ { \beta } )$ be a layout, and let $B _ { k } = \left( N _ { k } \right) : \left( r _ { k } \right)$ for $0 \leq k \leq \beta .$ . Then we say that the pair $\{ \mathsf { S } , B \}$ is admissible for composition if: 

(1) For all $0 \leq k \leq \beta ,$ the pair $\{ \mathsf { S } , B _ { k } \}$ is admissible for composition in the sense of Definition 2.12. 

(2) The intervals of definition for the pairs $\{ \mathsf { S } , B _ { k } \} _ { 0 \leq k \leq \beta }$ are disjoint. 

In this case, $\mathrm { i f } \ \mathbf { D } = \left( d _ { 0 } , . . . , d _ { \alpha } \right)$ is any stride tuple and $A = \mathbf { S } : \mathbf { D } $ , then we define the composition $A \circ B$ to be the concatenated layout 

$$
A \circ B := (A \circ B _ {0}, A \circ B _ {1},..., A \circ B _ {\beta})
$$

where each $A \circ B _ { k }$ is defined as in Definition 2.13. 

We have the following soundness theorem to validate Definition 2.17. 

Theorem 2.18. In the situation of Definition 2.17, we have that $f _ { A \circ B } = \widehat { f } _ { A } \circ f _ { B }$ 

Proof. By Proposition 2.14, we have that for all $0 \leq k \leq \beta ,$ the equality $f _ { A \circ B _ { k } } = \widehat { f } _ { A } \circ f _ { B _ { k } }$ of functions holds on the domain $[ 0 , \mathrm { s i z e } ( B _ { k } ) )$ . By Lemma 2.19, we have that the following diagram commutes: 

![image](Imgaes/layout-algebra-paper/c66332bf92820eaf6a45f180fc8d0da05cd4e4c633a57695c12cf6db99369b29.jpg)


It then suffices to see that the analogous diagram with ${ \widehat { f } } _ { A } \circ f _ { B }$ commutes, i.e. for the diagram 

$$
\begin{array}{c} [ 0, \text {size} (B)) \xrightarrow [ \cong ]{\iota} [ 0, \text {size} (B _ {0})) \times ... \times [ 0, \text {size} (B _ {\beta})) \\ \widehat {f _ {A}} \circ f _ {B} \Biggl \downarrow \qquad \qquad \qquad \qquad \qquad \qquad \qquad \Biggl \downarrow (\widehat {f _ {A}} \circ f _ {B _ {0}},..., \widehat {f _ {A}} \circ f _ {B _ {\beta}}) \\ \mathbb {N} \xleftarrow {} ^ {+} \mathbb {N} \times ... \times \mathbb {N}. \end{array}
$$

Breaking out the composition, we may factor this diagram as 

![image](Imgaes/layout-algebra-paper/26a08f65149056049dbc9560744bfbd7ea6729d57a0923c8607e5e16e6910017.jpg)


where the upper square commutes, again by Lemma 2.19. Note that the bottom square does not commute in general (i.e., the function ${ \widehat { f _ { A } } } : \mathbb { N } $ N itself is not generally additive). However, with respect to the factorization 

$$
\widehat {f} _ {A}: \mathbb {N} \xrightarrow {\cong} [ 0, M _ {0}) \times \ldots \times [ 0, M _ {\alpha - 1}) \times \mathbb {N} \xrightarrow {(d _ {0} , . . . , d _ {\alpha})} \mathbb {N} \times \ldots \times \mathbb {N} \xrightarrow {+} \mathbb {N},
$$

our assumption of disjoint intervals of definition ensures that the images of the maps $f _ { B _ { 0 } } , . . . , f _ { B _ { \beta } }$ are disjoint when intersected with $[ 0 , M _ { 0 } ) \times \ldots \times [ 0 , M _ { \alpha - 1 } ) - \{ 0 \}$ . For additivity, it now suffices to check that there do not exist distinct $B _ { k } , B _ { l }$ and non-zero $x$ \in im ${ \left( { f _ { B _ { k } } } \right) } , y \in \mathrm { i m } \left( f _ { B _ { l } } \right)$ that have coordinates $x _ { i } , y _ { i } \in [ 0 , M _ { i } )$ for some $0 \leq i < \alpha$ such that $x _ { i } + y _ { i } \ge M _ { i } ;$ if not, we may have that 

$$
\widehat {f} _ {A} (x + y) \neq \widehat {f} _ {A} (x) + \widehat {f} _ {A} (y)
$$

due to overflow in the $i$th coordinate, because the strides for the layout $A$ can be arbitrary. Now let $w _ { i _ { 0 } }$ and $z _ { j _ { 0 } }$ be the leftmost non-zero coordinates of $f _ { B _ { k } } ( 1 )$ and $f _ { B _ { l } } ( 1 )$ , respectively. If either of the indices $i _ { 0 }$ or $j _ { 0 }$ equal $\alpha$ then we are already done. Otherwise, we have that $w _ { i _ { 0 } } \leq M _ { i _ { 0 } } / 2$ and $z _ { j _ { 0 } } \leq M _ { j _ { 0 } } / 2$ from the left divisibility assumption. Moreover, the coordinates of subsequent values of $f _ { B _ { k } }$ and $f _ { B _ { l } }$ will increment by multiples of $w _ { i _ { 0 } }$ and $z _ { j _ { 0 } }$ in indices $i _ { 0 }$ and $j _ { 0 } ,$ by increments of 1 for indices greater than $i_0$ and $j _ { 0 }$ up to that occupied by the maximum value, and zero elsewhere. Finally, by disjointness7 we have that either $f _ { B _ { l } } ( 1 )$ is strictly greater than the maximum value attained by $f _ { B _ { k } }$ or vice-versa. Putting this all together, we see that disjointness of the intervals of definition rules out the possibility of overflow. 

We conclude that when restricted to the image of $( f _ { B _ { 0 } } , . . . , f _ { B _ { \beta } } )$ , we do have that $\widehat { f _ { A } }$ distributes over addition, which completes the proof. □ 

We used the following lemma about concatenated layouts in the proof of Theorem 2.18. 

Lemma 2.19. Let $C = ( C _ { 0 } , . . . , C _ { \gamma } )$ be a concatenated layout. Let 

$$
\iota : [ 0, \mathrm{size} (C)) \cong [ 0, \mathrm{size} (C _ {0})) \times ... \times [ 0, \mathrm{size} (C _ {\gamma}))
$$

be the usual isomorphism (as in Definition 2.3). Then the following diagram commutes: 

$$
\begin{array}{c} [ 0, \text {size} (C)) \xrightarrow [ \cong ]{\iota} [ 0, \text {size} (C _ {0})) \times ... \times [ 0, \text {size} (C _ {\gamma})) \\ f _ {C} \Big \downarrow \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \Big \downarrow (f _ {C _ {0}},..., f _ {C _ {\gamma}}) \\ \mathbb {N} \xleftarrow {} ^ {+} \mathbb {N} \times ... \times \mathbb {N} \end{array}
$$

Proof. If $C _ { 0 } , . . . , C _ { \gamma }$ are all length 1 layouts, then this is immediate from the definition. In general, we can take the maximal decomposition $C = ( C _ { 0 } ^ { \prime } , . . . , C _ { \gamma ^ { \prime } } ^ { \prime } )$ where all the $C _ { j } ^ { \prime }$ are length 1 layouts and $\gamma ^ { \prime } + 1$ is the length of $C .$ Then the $C _ { i }$ will be decomposed by disjoint and convex collections of the $C _ { j } ^ { \prime }$ in order, and we may place the diagram in question into the larger diagram 

$$
\begin{array}{c} [ 0, \text {size} (C)) \xrightarrow [ \cong ]{\iota} [ 0, \text {size} (C _ {0})) \times ... \times [ 0, \text {size} (C _ {\gamma})) ^ {\frac {(t _ {0} , \ldots , t _ {\gamma})}{\cong}} [ 0, \text {size} (C _ {0} ^ {\prime})) \times ... \times [ 0, \text {size} (C _ {\gamma^ {\prime}} ^ {\prime})) \\ f _ {C} \Biggl \downarrow \qquad \qquad \qquad \Biggl \downarrow (f _ {C _ {0}},..., f _ {C _ {\gamma}}) \\ \mathbb {N} \xleftarrow {+} \mathbb {N} ^ {\times (\gamma + 1)} \xleftarrow {(+, \ldots , +)} \mathbb {N} ^ {\times (\gamma^ {\prime} + 1)}. \end{array}
$$

Here, the maps $\iota _ { 0 } , . . . , \iota _ { \gamma }$ are the usual isomorphisms mapping the intervals $[ 0 , \mathrm { s i z e } ( C _ { i } ) )$ to their corresponding decompositions in terms of products of the intervals $[ 0 , \mathrm { s i z e } ( C _ { j } ^ { \prime } ) )$ . Now observe that the composite map $( \iota _ { 0 } , . . . , \iota _ { \gamma } )$ \circ \iota is also the usual isomorphism with respect to the maximal decomposition of $C$. Therefore, by definition the outer rectangle and righthand square commute, hence the lefthand square commutes. □ 

Example 2.20. As in Example 2.15, let $A = \mathbf { S } : \mathbf { D } = \left( M _ { 0 } , . . . , M _ { \alpha } \right) : \left( d _ { 0 } , . . . , d _ { \alpha } \right)$ be an arbitrary layout and 

$$
B _ {0} = (M _ {0}): (1), B _ {1} = (M _ {1}): (M _ {0}), \dots , B _ {\alpha} = (M _ {\alpha}): (M _ {0}... M _ {\alpha - 1}).
$$

Let $U \subset [ 0 , \alpha ]$ be any nonempty subset. Then for the collection of pairs $\{ \mathsf { S } , B _ { k } \} _ { k \in U }$ , the intervals of definition will be disjoint. Therefore, if we let $B _ { U }$ be the concatenation of the $B _ { k }$ for $k \in U$ , then the pair $\{ \mathsf { S } , B _ { U } \}$ is admissible for composition. Explicitly , if we write $U = \{ i _ { 0 } , . . . , i _ { \gamma } \}$ , then we have 

$$
A \circ B _ {U} = (M _ {i _ {0}},..., M _ {i _ {\gamma}}): (d _ {i _ {0}},..., d _ {i _ {\gamma}}).
$$

We may think of precomposition with $B _ { U }$ as a projector to the modes of $A$ with indices in $U .$ 

Warning 2.21. The conditions articulated in Definition 2.12 for single-mode admissibility are more relaxed than the static assert checks carried out in CUTLASS itself.8 Namely, our condition (1) is identical to a condition checked by CUTLASS, whereas for condition (2), our requirement of weak left divisibility is substituted by (ordinary) left divisibility in CUTLASS. For example, consider the layouts $A = \left( 4 , 6 , 8 , 1 0 \right) : \left( 2 , 3 , 5 , 7 \right)$ and $B = ( 6 ) : ( 1 2 )$ . Then attempting to compute the composition $C = A \circ B$ yields the error message “static assertion failed with "Static shape_div failure"” in CUTLASS, whereas according to our rules we would compute $C$ as (2, 3) : (9, 5). 

## 2.3 Logical Division

With these preliminaries in place, we can define the operation of logical division. 

Definition 2.22. Let $A = \mathsf { S } : \mathbf { D }$ and $B$ be layouts, and let $M$ be the size of $A$. Suppose that the pairs $\{ B , M \}$ and $\{ \mathsf { S } , B \}$ are admissible (for complementation and composition, respectively). Then we define the logical division $A / B$ to be the layout 

$$
A / B := A \circ (B, \text { complement } (B, M)).
$$

Implicit in Definition 2.22 is the following lemma: 

Lemma 2.23. Suppose $A = \mathbf { S } : \mathbf { D } , M = \mathrm { s i z e } ( A )$ , and $B$ are as in Definition 2.22. Then {$\mathbf{S}$, ($B$, complement($B$, $M$))} is admissible for composition. 

$$
\text { PROOF.   Write } A = \mathbf {S}: \mathbf {D} = (M _ {0},..., M _ {\alpha}): (d _ {0},..., d _ {\alpha}) \text { and } B = (N _ {0},..., N _ {\beta}): (r _ {0},..., r _ {\beta}). \text { Let }
$$

$$
\varphi : [ 0, \beta ] \stackrel {\cong} {\to} [ 0, \beta ]
$$

be the automorphism such that $B ^ { \varphi } : = \left( N _ { \varphi ( 0 ) } , . . . , N _ { \varphi ( \beta ) } \right) : \left( r _ { \varphi ( 0 ) } , . . . , r _ { \varphi ( \beta ) } \right)$ is sorted. Then by definition, 

$$
\operatorname{complement} (B, M) = \left(r _ {\varphi (0)}, \frac {r _ {\varphi (1)}}{N _ {\varphi (0)} r _ {\varphi (0)}},..., \frac {M}{N _ {\varphi (\beta)} r _ {\varphi (\beta)}}\right): \left(1, N _ {\varphi (0)} r _ {\varphi (0)},..., N _ {\varphi (\beta)} r _ {\varphi (\beta)}\right).
$$

Now write 

$$
B _ {0} ^ {\prime} = \left(r _ {\varphi (0)}\right): (1), B _ {1} ^ {\prime} = \left(\frac {r _ {\varphi (1)}}{N _ {\varphi (0)} r _ {\varphi (0)}}\right): \left(N _ {\varphi (0)} r _ {\varphi (0)}\right), \ldots , B _ {\beta} ^ {\prime} = \left(\frac {M}{N _ {\varphi (\beta)} r _ {\varphi (\beta)}}\right): \left(N _ {\varphi (\beta)} r _ {\varphi (\beta)}\right)
$$

for the length 1 layouts that comprise complement($B$, $M$). We first claim that the pairs $\{ \mathsf { S } , B _ { k } ^ { \prime } \}$ for $0 \le k \le \beta$ are all admissible for composition. By assumption, we have that $M$ is left divisible by $r _ { \varphi ( k ) }$ and its remainder is then weakly left divisible by $N _ { \varphi ( k ) }$ , for all $0 \leq k \leq \beta .$ But since $r _ { \varphi ( k ) } N _ { \varphi ( k ) } | r _ { \varphi ( k + 1 ) }$ for all $0 \le k < \beta$ and $M = { \mathrm { s i z e } } ( A )$ , the additional divisibility condition $( 3 )$ in Definition 2.11 needed to promote weak left divisibility to left divisibility is necessarily satisfied for all the $N _ { \varphi ( k ) }$ terms. Therefore, we deduce that the pairs $\{ \mathsf { S } , B _ { k } ^ { \prime } \}$ are indeed all admissible. Now by Proposition 2.7, we see that the additional disjointness assumption is satisfied so that {$\mathbf{S}$, ($B$, complement($B$, $M$))} is admissible for composition. □ 

This concludes our current treatment of logical division. For the time being, we leave further discussion of examples of logical division to the CuTe documentation. 

## 3 PERMUTATIONS EXPRESSIBLE AS LAYOUT FUNCTIONS

In this section, we explain how to retrieve all permutations that are expressible as layout functions in a structured way (for some more precise motivation, we refer to Remark 3.16 below). We will assume that the reader is familiar with the basic language of category theory, which is convenient for describing the algebraic structure of “ordered factorizations” that naturally appears here. 

Definition 3.1. We define the set ob(Fact) of ordered factorizations to consist of all expressions $\left[ \hbar \cdot \cdot \cdot \cdot \hbar \epsilon \right]$ where $k \geq 0$ and the $p_i$ are primes (not necessarily distinct). The case $k = 0$ corresponds to the empty factorization, which we denote as []. 

Example 3.2. The set ob(Fact) includes expressions such as [], [2], [3], [22], [23], [32], [232], etc. 

Notation 3.3. Let $\underline { { \boldsymbol { k } } }$ denote the set $\{ 1 , 2 , . . . , k \}$ consisting of $k$ elements. $( \mathrm { I f } k = 0$ , then ${ \underline { { 0 } } } = \emptyset$ is the empty set.) 

Definition 3.4. We define the category Fact of ordered factorizations as follows: 

(1) ob(Fact) is the set of objects of Fact. 

(2) For every expression $E = \left[ p _ { 1 } p _ { 2 } . . . p _ { k } \right]$ in ob(Fact) and every morphism of finite sets $\alpha : \underline { n } \to \underline { k } ,$ we have a morphism 

$$
E ^ {\alpha} = [ p _ {\alpha (1)} p _ {\alpha (2)} \dots p _ {\alpha (n)} ] \xrightarrow {\alpha_ {E}} E = [ p _ {1} p _ {2} \dots p _ {k} ]
$$

in Fact. This defines the set of all morphisms with codomain $E ,$ and ranging over all $E$ thus defines the set of all morphisms in Fact. 

(3) The composition of morphisms is defined as follows. Suppose we have morphisms of finite sets $\alpha : \underline { n } \to \underline { k }$ and $\beta : { \underline { { m } } }  { \underline { { n } } }$ and an expression $E = \left[ p _ { 1 } p _ { 2 } . . . p _ { k } \right]$ . Write 

$$
E ^ {\alpha} = \left[ p _ {\alpha (1)} p _ {\alpha (2)}... p _ {\alpha (n)} \right] = \left[ q _ {1}... q _ {n} \right].
$$

Let $\gamma = \alpha \circ \beta : \underline { m }  \underline { k } .$ Then the composition of the morphisms 

$$
\alpha_ {E}: E ^ {\alpha} = [ p _ {\alpha (1)} p _ {\alpha (2)} \dots p _ {\alpha (n)} ] \rightarrow E = [ p _ {1} \dots p _ {k} ], \quad \beta_ {E ^ {\alpha}}: (E ^ {\alpha}) ^ {\beta} = [ q _ {\beta (1)} \dots q _ {\beta (m)} ] \rightarrow E ^ {\alpha} = [ q _ {1} \dots q _ {n} ]
$$

is given by $\gamma _ { E } : E ^ { \gamma }  E ,$ , where we use that $\left[ q _ { \beta ( 1 ) } . . . q _ { \beta ( m ) } \right] = \left[ p _ { Y ( 1 ) } . . . p _ { Y ( m ) } \right] .$ 

It’s easy to check that the composition of morphisms in Fact is associative and has identities, so Definition 3.4 really does define a category. 

Notation 3.5. Let $\Sigma _ { k }$ denote the symmetric group on $k$ letters. Given an element $\varphi \in \Sigma _ { k }$ , we also denote the associated automorphism of $\underline { { \boldsymbol { \cdot } } } \underline { { \boldsymbol { k } } }$ by $\varphi .$ 

Example 3.6. Suppose $E = \left[ 2 2 2 \right]$ . Then every permutation $\varphi \in \Sigma _ { 3 }$ defines an automorphism $E ^ { \varphi } = E  E$ in Fact. Conversely, every automorphism of [222] uniquely corresponds to an element of $\Sigma _ { 3 }$ 

Suppose $E = \left[ 2 3 2 \right]$ . Then the transposition $\sigma = \left( 1 3 \right) \in \Sigma _ { 3 }$ defines an automorphism of $E$ since $E ^ { \sigma } = E$ On the other hand, the transposition $\tau = ( 1 2 ) \in \Sigma _ { 3 }$ defines a morphism $E ^ { \tau } = [ 3 2 2 ]  E = [ 2 3 2 ]$ 

Remark 3.7. Let FinSet denote the category of finite sets (or rather a skeleton, with objects given by the sets $\underline{n}$ for $n \geq 0 )$ . Given an object $\underline { { k } } \in \mathbf { F i n S e t }$ , let $\mathrm { F i n S e t } ^ { \prime \mathrm { f } } \mathrm { \underline { { { \varepsilon } } } }$ denote the overcategory, whose objects are morphisms $[ \alpha : \underline { { n } }  \underline { { k } } ]$ and whose morphisms are commuting triangles. Recall that this category has a final object given by $[ \mathrm { i d } _ { \underline { { k } } } ]$ 

Then for every expression $E = \left[ { p _ { 1 } . . . p _ { k } } \right]$ of length $k ,$ we have a functor 

$$
F _ {E}: \mathbf {F i n S e t} ^ {\underline {{/ k}}} \to \mathbf {F a c t}
$$

that sends the object $[ \alpha : \underline { { n } } \to \underline { { k } } ] \mathrm { t o } E ^ { \alpha }$ and the unique morphism $[ \alpha ]  [ \mathrm { i d } _ { \underline { { { k } } } } ] \mathrm { t o } \alpha _ { E } : E ^ { \alpha }  E$ . This functor has every morphism in Fact with codomain $E$ in its image. 

Remark 3.8. In fact, we can identify Fact itself as a certain overcategory (or rather, a full subcategory thereof). Namely, let $\mathcal { P }$ denote the infinite set of primes $\{ 2 , 3 , 5 , \ldots \}$ , let Set be the category of sets, and let FinSet/P be the full subcategory of $\mathbf { S e t } ^ { \mathcal { P } }$ on those morphisms $X \to { \mathcal { P } }$ where $X$ is a finite set. Then we have an equivalence of categories 

$$
\mathbf {F a c t} \simeq \mathbf {F i n S e t} ^ {\mathcal {P}}
$$

that sends an expression $E = \left[ \pmb { p } _ { 1 } . . . \pmb { p } _ { k } \right]$ to the morphism $E _ { \bullet } : \underline { { k } } \to \mathcal { P }$ given by $i \mapsto p _ { i }$ . Under this equivalence, the functor $F _ { E }$ of Remark 3.7 identifies with the functor 

$$
\operatorname{FinSet} ^ {\underline {{k}}} \simeq \left(\operatorname{FinSet} ^ {\mathcal {P}}\right) ^ {/ E _ {\bullet}} \rightarrow \operatorname{FinSet} ^ {\mathcal {P}}
$$

that forgets the map to $E _ { \bullet }$ . 

We now explain how to associate a layout to every morphism in Fact. 

Definition 3.9. Suppose $E = \left[ \pmb { p } _ { 1 } . . . \pmb { p } _ { k } \right]$ and $\alpha : \underline { n } \to \underline { k } .$ . We define a layout $L _ { ( E , \alpha ) }$ as follows:9 

(1) Its shape tuple is $( p _ { \alpha ( 1 ) } , p _ { \alpha ( 2 ) } , . . . , p _ { \alpha ( n ) } )$ 

(2) Its stride tuple is $( d _ { 1 } , d _ { 2 } , . . . , d _ { n } )$ where $\begin{array} { r } { d _ { i } = \prod _ { j < \alpha ( i ) } p _ { j } . ^ { 1 0 } } \end{array}$ 

We also let $f _ { ( E , \alpha ) }$ denote the associated layout function. 

Example 3.10. Suppose $E = \left[ 2 3 \right]$ and $\varphi = \left( 1 2 \right) \in \Sigma _ { 2 }$ is the nontrivial transposition. Then $L _ { \left( E , \varphi \right) } = \left( 3 , 2 \right) : \left( 2 , 1 \right)$ Suppose $E = \left( 2 2 2 \right)$ and $\varphi = ( 2 3 1 ) \in \Sigma _ { 3 }$ , so $\varphi$ is a cycle of order 3 with $\varphi ( 1 ) = 2 , \varphi ( 2 ) = 3 , \varphi ( 3 ) = 1$ . Then $L _ { ( E , \varphi ) } = ( 2 , 2 , 2 ) : ( 2 , 4 , 1 )$ 

Remark 3.11. Let $E = \left[ { p _ { 1 } . . . p _ { k } } \right]$ and $\alpha : \underline { n }  \underline { k } .$ Let $N = p _ { 1 } \cdot \ldots \cdot p _ { k }$ and $N ^ { \alpha } = p _ { \alpha ( 1 ) } \cdot \ldots \cdot p _ { \alpha ( n ) }$ . In what follows, consider the canonical isomorphisms 

$$
[ 0, N) \cong [ 0, p _ {1}) \times [ 0, p _ {2}) \times \ldots \times [ 0, p _ {k}),
$$

$$
\left[ 0, N ^ {\alpha}\right) \cong \left[ 0, p _ {\alpha (1)}\right) \times \left[ 0, p _ {\alpha (2)}\right) \times \ldots \times \left[ 0, p _ {\alpha (n)}\right)
$$

Then the associated layout function $f _ { ( E , \alpha ) } : [ 0 , N ^ { \alpha } ) \to [ 0 , N ) \subset \mathbb { N }$ can be described as the multilinear function 

$$
[ 0, p _ {\alpha (1)}) \times [ 0, p _ {\alpha (2)}) \times \dots \times [ 0, p _ {\alpha (n)}) \rightarrow [ 0, p _ {1}) \times [ 0, p _ {2}) \times \dots \times [ 0, p _ {k})
$$

that sends the basis vector $\delta _ { i }$ for $1 \leq i \leq n \mathrm { t o } \delta _ { \alpha ( i ) }$ , and which restricts to an isomorphism $[ 0 , p _ { \alpha ( i ) } ) \stackrel { \cong } { \longrightarrow } [ 0 , p _ { \alpha ( i ) } )$ for all $1 \leq i \leq n .$ . In particular, if $\alpha$ is itself a bijection, then $f _ { ( E , \alpha ) }$ restricts to an automorphism of [0, $N$). 

Elaborating on Remark 3.11, we have the following lemma, which indicates that composition in the category Fact is compatible with the composition of layout functions. 

Lemma 3.12. Suppose we have morphisms of finite sets $\alpha : \underline { { n } } \to \underline { { k } } , \beta : \underline { { m } } \to \underline { { n } }$ and an expression $E = \left[ \ p _ { 1 } p _ { 2 } . . . \ p _ { k } \right]$ Write $\gamma = \alpha \circ \beta .$ . Consider the composition 

$$
\gamma_ {E}: E ^ {\gamma} = (E ^ {\alpha}) ^ {\beta} \xrightarrow {\beta_ {E ^ {\alpha}}} E ^ {\alpha} \xrightarrow {\alpha_ {E}} E
$$

in Fact. Then the associated layoutfunctions satisfy the composition equality 

$$
f _ {(E, \gamma)} = f _ {(E, \alpha)} \circ f _ {(E ^ {\alpha}, \beta)}.
$$

Proof. Let $N = p _ { 1 } \cdot . . . \cdot p _ { k } , N ^ { \alpha } = p _ { \alpha ( 1 ) } \cdot . . . \cdot p _ { \alpha ( k ) }$ , and $N ^ { \gamma } = p _ { \gamma ( 1 ) } \cdot . . . \cdot p _ { \gamma ( m ) }$ . We use the canonical isomorphisms 

$$
[ 0, N) \cong [ 0, p _ {1}) \times [ 0, p _ {2}) \times \dots \times [ 0, p _ {k}),
$$

$$
[ 0, N ^ {\alpha}) \cong [ 0, p _ {\alpha (1)}) \times [ 0, p _ {\alpha (2)}) \times \dots \times [ 0, p _ {\alpha (n)})
$$

$$
[ 0, N ^ {\gamma}) \cong [ 0, p _ {\gamma (1)}) \times [ 0, p _ {\gamma (2)}) \times \ldots \times [ 0, p _ {\gamma (m)})
$$

to write the domains and codomains of the layout functions in question (noting that $f _ { ( E ^ { \alpha } , \beta ) }$ has codomain lying inside $[ 0 , N ^ { \alpha } ) )$ . We are trying to equate the multilinear function 

$$
f _ {(E, \gamma)}: [ 0, p _ {\gamma (1)}) \times [ 0, p _ {\gamma (2)}) \times \dots \times [ 0, p _ {\gamma (m)}) \rightarrow [ 0, p _ {\alpha (1)}) \times [ 0, p _ {\alpha (2)}) \times \dots \times [ 0, p _ {\alpha (n)})
$$

with the composition of the two multilinear functions 

$$
f _ {(E ^ {\alpha}, \beta)}: [ 0, p _ {\gamma (1)}) \times [ 0, p _ {\gamma (2)}) \times \ldots \times [ 0, p _ {\gamma (m)}) \to [ 0, p _ {\alpha (1)}) \times [ 0, p _ {\alpha (2)}) \times \ldots \times [ 0, p _ {\alpha (n)})
$$

$$
f _ {(E, \alpha)}: [ 0, p _ {\alpha (1)}) \times [ 0, p _ {\alpha (2)}) \times \dots \times [ 0, p _ {\alpha (n)}) \to [ 0, p _ {1}) \times [ 0, p _ {2}) \times \dots \times [ 0, p _ {k}).
$$

But since basis vectors are mapped to basis vectors by Remark 3.11, it suffices to check the desired equality on basis vectors, which is straightforward. □ 

Warning 3.13. In Lemma 3.12, the per-mode condition of admissibility for composition (Definition 2.12) is obviously satisfied. However, the disjointness condition in Definition 2.17 may be violated in the case where $\beta : { \underline { { m } } }  { \underline { { n } } }$ is not an injective function. This isn’t a contradiction with the prior analysis carried out in the proof of Theorem 2.18, since there we were concerned with the composition being well-defined in the situation of arbitrary strides for the second layout. 

We now define a “realization” functor from Fact to FinSet that sends morphisms of ordered factorizations to their associated layout functions 

Definition 3.14. Let $R$: Fact → FinSet be the functor defined as follows: 

(1) Let $E = \left[ \pmb { p } _ { 1 } . . . \pmb { p } _ { k } \right]$ be an object of Fact and let $N = p _ { 1 } \cdot \ldots \cdot p _ { k }$ . Then $R ( E ) = [ 0 , N ) . ^ { 1 }$ 11 

(2) For every morphism $\alpha _ { E } : E ^ { \alpha } \to E .$ let $R ( \alpha _ { E } ) = f _ { ( E , \alpha ) } : [ 0 , N ^ { \alpha } )  [ 0 , N )$ be as in Definition 3.9. 

By Lemma 3.12, $R$: Fact → FinSet does indeed define a functor since it respects the composition of morphisms (and identities as well, obviously). 

We note that $R$ doesn’t contain every possible function expressible as a layout function in its image. However, it does contain every automorphism $[ 0 , N ) \stackrel { \cong } { \longrightarrow } [ 0 , N )$ expressible as a layout function in its image. 

Proposition 3.15. Let $N > 0$ be a positive integer and let $f : [0,N) \to [0,N)$ be an automorphism such that there exists a layout $L$ of size $N$ with $\hat f = f_L$ Then $f _ { L }$ is in the image of the realization functor $R$. 

Proof. Without loss of generality, we may suppose that the shape tuple of $L$ is given by $\left( p _ { 1 } , p _ { 2 } , . . . , p _ { k } \right)$ where the $p_i$ are all prime numbers and $N=p_1\cdots p_k$ So we may write $L = \left( p _ { 1 } , p _ { 2 } , . . . , p _ { k } \right) : \left( d _ { 1 } , d _ { 2 } , . . . , d _ { k } \right)$ . Then the sort of $L$ must be of the form 

$$
L ^ {\varphi} := \left(p _ {\varphi (1)}, p _ {\varphi (2)},..., p _ {\varphi (k)}\right): \left(1, p _ {\varphi (1)}, p _ {\varphi (1)} p _ {\varphi (2)},..., \Pi_ {1 \leq i <   k} p _ {\varphi (i)}\right)
$$

for some permutation $\varphi \in \Sigma _ { k }$ , in order for $f _ { L }$ to be an automorphism of $[ 0 , N )$ . But this means that if we let $\psi = \varphi ^ { - 1 }$ be the inverse permutation, then 

$$
\psi_ {E}: E ^ {\psi} = \left[ p _ {1} p _ {2}... p _ {k} \right] = \left[ p _ {\psi (\varphi (1))} p _ {\psi (\varphi (2))}... p _ {\psi (\varphi (k))} \right]\rightarrow E = \left[ p _ {\varphi (1)} p _ {\varphi (2)}... p _ {\varphi (k)} \right]
$$

is a morphism in Fact such that $R ( \psi _ { E } ) = f _ { L } = f .$ 

Remark 3.16. One way to interpret Proposition 3.15 is that if we take the maximal subgroupoid $\mathbf { F a c t } ^ { \simeq }$ inside Fact (i.e., the subcategory on all invertible morphisms), then 

$$
R: \operatorname{Fact} ^ {\simeq} \rightarrow \operatorname{FinSet}
$$

carves out exactly those permutations expressible as layouts. Our motivation for this description is that for a fixed integer $N > 0$ , the subset $\Sigma _ { N } ^ { L }$ of $\Sigma _ { N }$ on those automorphisms expressible as layout functions is typically not a subgroup (being not generally closed under the group multiplication, i.e. composition). Instead, if we let 

$$
\mathbf {F a c t} _ {N} ^ {\simeq} \subset \mathbf {F a c t} ^ {\simeq}
$$

be the full subgroupoid on those objects $\left[ \hbar \cdot \cdot \cdot \cdot \hbar _ { k } \right]$ with $N = p _ { 1 } \cdot \ldots \cdot p _ { k }$ , then $\Sigma _ { N } ^ { L }$ consists of those morphisms in the image of $R$ on $\mathbf { F a c t } _ { N } ^ { \simeq }$ . However, we see that $\Sigma _ { N } ^ { L }$ is closed under the operation of taking the group inverse. Moreover, in the special case that $N$ is a prime power $p ^ { k }$ , then $\Sigma _ { N } ^ { L }$ is in fact a subgroup and is isomorphic to $\Sigma _ { k }$ This corresponds to $\mathbf { F a c t } _ { p ^ { k } } ^ { \simeq }$ being a groupoid with one object $[ p p . . . p ]$ , i.e., a group. 

## REFERENCES

[1] CUTLASS — CUDA Templates for Linear Algebra Subroutines. https://github.com/NVIDIA/cutlass. 

[2] CuTe Layout Operations. https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/02_layout_operations.md. 
