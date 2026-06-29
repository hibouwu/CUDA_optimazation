# 大语言模型量化

## 1. 大模型量化 Scaling Law

### 1.1 标准大模型 Scaling Law

大模型的标准 Scaling Law 公式为：

$$
L(N,D)=aN^{-\alpha}+bD^{-\beta}+E
$$

其中：

- $N$：模型参数量。参数量越大，loss 越低。
- $D$：训练 token 数。训练 token 数越大，loss 越低。
- $E$：不可避免的误差。
- $\alpha,\beta$：扩大模型和数据时 loss 的下降速度，通常与模型架构设计、tokenizer、数据质量、训练方式相关。

此公式描述了大模型在高精度训练和高精度推理情况下的精度表现。

### 1.2 低精度训练和 PTQ 的 Scaling Law（Scaling Law for Precision）

#### 1.2.1 统一公式

引入低精度训练和 PTQ（训练后量化）后，模型性能通常不如高精度模型。[Scaling Laws for Precision](https://arxiv.org/pdf/2411.04330) 通过实验统计建模，提出了同时考虑低精度训练和 PTQ 的 Scaling Law：

$$
\begin{aligned}
L(N,D,P_\text{train},P_\text{post})
={}&AN_\text{eff}^{-\alpha}
+BD^{-\beta}
+E \\
&+\delta_\text{PTQ}
(N_\text{eff},D,P_\text{train},P_\text{post})
\end{aligned}
$$

其中：

- 前三项为标准 Scaling Law：模型越大、数据越多，loss 越低。
- 由于训练时引入了低精度量化，参数量 $N$ 变为有效参数量 $N_\text{eff}$。量化噪声使同样参数量的低精度模型在 loss 上表现得像一个更小的高精度模型。
- 第四项表示 PTQ 对模型精度的影响。

#### 1.2.2 低精度训练时的有效参数量（QAT 对精度的影响）

有效参数量表示为：

$$
\begin{aligned}
N_\text{eff}(P)
={}&N
(1-e^{-P_w/\gamma_w})
(1-e^{-P_a/\gamma_a})
(1-e^{-P_{kv}/\gamma_{kv}}) \\
={}&N
\prod_{x\in\{w,a,kv\}}
(1-e^{-P_x/\gamma_x})
\end{aligned}
$$

其中：

- $(1-e^{-P_w/\gamma_w})$ 表示训练时权重量化对有效参数量的影响。$P_w$ 为量化后的 bit 数；$\gamma_w$ 为权重灵敏度，是通过预先规划并探索不同 $D$、$N$、$P$ 组合的实验拟合得到的参数。
- $(1-e^{-P_a/\gamma_a})$ 和 $(1-e^{-P_{kv}/\gamma_{kv}})$ 分别表示训练时量化 activation 和 KV cache 对有效参数量的影响。
- 权重、激活和 KV cache 量化对有效参数量的影响近似独立，总体影响为各项影响的乘积。

**指导意义：**在降低训练精度 $P$ 时，该公式可以估算需要增加多少参数量 $N$ 才能保持误差不变。

> **图片待补充：不同精度下的有效参数量与等误差线**
>
> 左图表示不同精度下有效参数量的差异。对 KV cache 进行低比特量化时，有效参数量下降最少；对权重量化时下降最大。
>
> 中图和右图表示降低训练精度后，需要增加多少参数量才能保持误差不变。中图为实测结果，右图为公式推导结果。每条线均为等误差线，颜色越深表示误差越低。大部分模型在 7～8 bit 训练精度下只需少量增加参数，即可保持总体误差不变。

> **图片待补充：权重、激活和 KV cache 三个影响因子的独立性**
>
> 该图主要说明以下三个影响因子的独立性：
>
> $$
> N
> (1-e^{-P_w/\gamma_w})
> (1-e^{-P_a/\gamma_a})
> (1-e^{-P_{kv}/\gamma_{kv}})
> $$
>
> 训练时可以分别量化权重、激活和 KV cache，拟合各自的 $\gamma$，进而预测不同 W/A/KV 精度组合下的 loss。真正的最优 $P$ 还需要结合训练与推理成本、硬件吞吐和部署约束共同决定。

#### 1.2.3 PTQ 对模型精度的影响

统一公式中的第四项为：

$$
\delta_\text{PTQ}
=
C_T e^{-P_\text{post}/\gamma_\text{post}}
\left(
\frac{D^{\gamma_D}}{N_\text{eff}^{\gamma_N}}
\right)
\prod_{x\in\{w,a,kv\}}
[1-e^{-C_x(P_x-P_\text{post})}]
$$

该公式可拆分为以下三部分。

##### 1. 量化本身造成的损失

$$
C_T e^{-P_\text{post}/\gamma_\text{post}}
$$

- $P_\text{post}$ 是部署或 PTQ 后的 bit 数，例如 8 bit、4 bit、3 bit。
- $P_\text{post}$ 越大，指数项 $e^{-P_\text{post}/\gamma_\text{post}}$ 越小，PTQ 损失越小。
- $P_\text{post}$ 越小，例如从 8 bit 降到 4 bit、3 bit、2 bit，该项会快速增大。这说明低 bit 的损害并非线性增加，而可能呈指数式恶化。
- $\gamma_\text{post}$ 表示模型或量化方法对推理 bit 数的敏感程度。$\gamma_\text{post}$ 越小，减少 bit 数造成的损害越剧烈。
- $C_T$ 是整体尺度常数，吸收了数据集、模型家族、量化算法、loss 单位等因素。

##### 2. 模型的过训练程度

$$
\frac{D^{\gamma_D}}{N_\text{eff}^{\gamma_N}}
$$

- $D$ 是训练 token 数。
- $N_\text{eff}$ 是有效参数量，而不是原始参数量。低精度训练会使 $N_\text{eff}<N$。
- $D^{\gamma_D}$ 位于分子，表示训练数据越多，PTQ 退化越可能增大。
- $N_\text{eff}^{\gamma_N}$ 位于分母，表示有效模型越大，越能吸收量化扰动。

训练数据越多，模型拟合效果通常越好；但对训练充分的模型进行 PTQ 时，性能反而可能下降。这是因为模型内部的特征表达变得更加精细和显著，参数离群值效应更强，量化误差也随之增大。论文将这种现象称为 **overtraining effect**。

这并不表示训练数据多本身有害，而是说“模型经过大量训练后，再强制 PTQ 到极低 bit”可能更加危险。

##### 3. 训练精度与推理精度的差距修正

$$
\prod_{x\in\{w,a,kv\}}
[1-e^{-C_x(P_x-P_\text{post})}]
$$

$x$ 分别表示：

- $w$：weights，权重。
- $a$：activations，激活。
- $kv$：KV cache 或 attention 相关中间量。

$P_x$ 是训练时对应部分使用的精度，$P_\text{post}$ 是最终 PTQ 或部署精度。

如果 W、A、KV 的训练精度与推理精度均相同，该项为 0；如果 PTQ 推理精度与训练精度相差不大，该项较小，表示 PTQ 造成的额外误差较小。

##### 临界训练 token 数

$D_\text{crit}$ 表示“继续增加训练数据带来的原始 loss 边际下降”恰好等于“PTQ degradation 边际增加”时的临界 token 数：

$$
D_\text{crit}
=
\left(
\frac{\beta B N^{\gamma_N}e^{P_\text{post}/\gamma_\text{post}}}
{\gamma_D C_T}
\right)^{\frac{1}{\gamma_D+\beta}}
$$

- 当 $D<D_\text{crit}$ 时，继续增加训练数据，PTQ 后的模型整体仍会变好。
- 当 $D=D_\text{crit}$ 时，继续增加训练数据，PTQ 后的模型性能不再提升。
- 当 $D>D_\text{crit}$ 时，继续增加训练数据，PTQ 后的模型性能反而可能下降。

推导过程略。

**指导意义：**

- 如果模型最终需要以低精度部署，应在训练阶段引入低精度训练，以提高模型的稳定性和可控性。
- 传统非量化部署通常直接交付 loss 收敛时的模型；对于量化部署，可以在训练过程中保存多个 checkpoint 并分别进行 PTQ。某些尚未完全收敛的 checkpoint 在量化部署后的效果可能优于已收敛的 checkpoint。
- 如果模型在 $D=D_\text{crit}$ 时，部署模型的 loss 仍未达到要求，应适当增加参数量 $N$，而不是继续堆叠训练数据。
- $D_\text{crit}$ 的影响主要体现在极低 bit（如 4 bit）下。对于 8 bit PTQ，通常不需要考虑训练数据增加导致最终部署性能下降的问题。

> **图片待补充：训练时长与 PTQ 退化**
>
> BF16 模型训练越久，再进行 PTQ 后退化的可能性越大，特别是在极低 bit 下。

> **图片待补充：模型规模与精度的关系**
>
> 大模型搭配低精度的 loss 优于小模型搭配高精度。

> **图片待补充：PTQ bit 数与 loss 的关系**
>
> loss 在 4～6 bit PTQ 时会剧烈增长。

#### 1.2.4 总结

- 训练精度应该作为 Scaling Law 的变量参与模型设计，而不是默认使用 BF16/FP16。低精度训练会降低 $N_\text{eff}$，因此使用 FP8/FP4 训练时，不能直接照搬高精度训练配方。
- 如果训练精度降低，应重新分配模型规模和训练 token。在固定算力下，低精度训练通常更适合增加参数量，而不是继续增加数据。
- 如果模型未来需要以 FP4、INT4 或 NVFP4 部署，训练阶段最好引入相近的量化噪声或 QAT，让模型提前适应低精度推理。
- 应谨慎处理过度训练的模型。训练 token 很多时，模型可能对 PTQ 更敏感，因此训练过程中应定期进行目标格式的量化评估，而不应只观察 BF16 validation loss。
- PTQ 损失不是常数，而是与 $D/N$、模型大小和推理 bit-width 共同相关。过度训练、小模型和低 bit 会叠加风险。
- 训练低精度与推理低精度之间存在“鲁棒化”关系。训练精度越接近部署精度，PTQ 的额外损失可能越小。
- 权重、激活和 KV cache 的低精度影响近似可乘，但敏感度不同。量化策略不能只围绕权重设计。
- FP4/NVFP4 不应只用于最终 PTQ，最好配合低精度训练 recipe、随机舍入、细粒度 scaling 或混合精度保留。

### 1.3 混合精度 Scaling Law

#### 1.3.1 Scaling Law for Precision

[Scaling Laws for Mixed-Precision Quantization](https://arxiv.org/pdf/2410.06722) 是对 1.2 节 Scaling Law for Precision 核心公式的混合精度扩展，主要替代标准公式中的第四项：

$$
\begin{aligned}
L(N,D,P_\text{train},P_\text{post})
={}&AN_\text{eff}^{-\alpha}
+BD^{-\beta}
+E \\
&+\delta_\text{PTQ}
(N_\text{eff},D,P_\text{train},P_\text{post})
\end{aligned}
$$

论文首先提出 weak law：

$$
\delta^\text{opt}(N,Q_r,Q_b)
=
C e^{AQ_r}N^{-\gamma_N}
$$

其中：

- $\delta$：loss 的退化，用于替代 1.2 节公式中的第四项。
- $Q_r$：低精度参数的比例。该值越大，采用低精度的参数越多，模型退化越快。
- $e^{AQ_r}$：表示低精度比例造成的 loss 增长是指数式的，而不是线性的。
- $N$：模型参数量。模型越大，通常越能抵抗量化影响。
- $N^{-\gamma_N}$：表示大模型可以吸收一部分量化误差。

上述公式未考虑 block size，即多少个 element 共享一个 scale 参数，例如 NVFP4 为 16，MXFP8 为 32。进一步引入 block size $Q_b$ 后，strong law 为：

$$
\delta^\text{opt}(N,Q_r,Q_b)
=
C e^{AQ_r}N^{-\gamma_N}(Q_b+d)^{\gamma_c}
$$

其中，$d$ 为拟合常数，$\gamma_c$ 为 block size 敏感度。$Q_b$ 越小，误差项越小，并且 $Q_b$ 对误差的影响呈幂次关系。

实验表明，该 Scaling Law 可以用于不同的量化方法，但每种方法需要分别拟合参数 $C$、$\gamma_N$、$\gamma_c$、$A$ 和 $d$。具体论证和实验过程略。

**指导意义：**

- 如果模型考虑使用混合精度部署，应在训练阶段根据 loss 目标确定模型参数量。
- 如果 PTQ 部署后的效果仍未达到目标，并且 $Q_r$ 无法继续调整，可以减小 block size，以降低量化影响。

#### 1.3.2 敏感度分析与低精度层选取

1.3.1 节给出了混合精度量化达到目标精度时，低精度层比例和 block size 应满足的约束，但没有解决哪些层使用高精度、哪些层使用低精度的问题。常用的低精度层选取方法如下。

##### 单层消融法

逐一将每层设置为低精度并进行试推理，观察模型效果下降幅度。下降最大的层对量化噪声最敏感，应采用高精度；选择影响最大的 $n$ 层使用高精度，其余层使用低精度。

##### 贪心恢复法

先将所有层设置为低精度，再逐层恢复为高精度。每轮选择恢复后 loss 改善最多的层，并保留其高精度设置，直到达到预定的 $Q_r$。

##### Activation 感知法（AWQ）

参考：[AWQ](https://arxiv.org/abs/2306.00978)

对于 $A\times B=D$，如果 $A$ 的值较大，对 $B$ 施加的噪声会被 $A$ 放大，使结果 $D$ 的误差增大；同理，对 $A$ 施加噪声时，$B$ 的大小也会影响 $D$ 的误差。

可以先使用少量数据试推理，得到每层权重 $W$ 的输入激活 $A$ 的平均能量。输入激活能量较大的层，其权重使用高精度，以尽量降低输出误差；能量较小的层则使用低精度。

##### HAWQ：Hessian 判别法

参考：[HAWQ](https://www.stat.berkeley.edu/~mmahoney/pubs/HAWQ_ICCV_2019_paper.pdf)

模型训练收敛后，大部分参数的一阶导数接近 0，无法用于判断参数扰动对整体 loss 的影响；但二阶导数通常不为 0，可以描述参数在当前极小值点附近的曲率。

> **图片待补充：二阶导数与参数扰动造成的 loss 偏移**

二阶导数越大，参数受到扰动后造成的 loss 偏移越大。因此，可以根据二阶导数判断参数重要性：二阶导数越大，参数越重要，应使用更高精度。

对于一个 tensor，需要通过 Hessian 矩阵描述各参数的二阶偏导。直接计算 Hessian 矩阵需要二次反向传播，时间复杂度很高，尤其是对距离 loss 较远的深层参数，几乎不可行。

因此，可以使用 Hessian 矩阵绝对值最大的特征值（谱半径）近似表示每个 tensor 的 Hessian 能量。谱半径越大，表示该 tensor 的二阶导数整体越大、抗扰动能力越低，应使用更高精度。Hessian 谱半径可以通过幂迭代法在有限时间内求得。

> **图片待补充：使用幂迭代法计算 Hessian 谱半径**

得到所有参数 tensor 的 Hessian 谱半径后，即可据此决定哪些层使用高精度、哪些层使用低精度。

### 1.4 Compression Scaling Law

[Compression Scaling Law](https://arxiv.org/pdf/2502.16440) 提出了一种与前述方法类似的 Scaling Law。它将量化、剪枝等压缩方式统一抽象为压缩效率 $\mathrm{eff}$，核心公式为：

$$
L(N,D,C)
=
\frac{a}{(N\cdot \mathrm{eff}(C))^b}
+
\frac{c}{D^d}
+
e
$$

其中：

- $N$：真实参数量。
- $D$：训练数据量。
- $C$：压缩配置，包括权重、激活、KV cache、量化 bit 数、稀疏率、scale 粒度和校准算法等。
- $\mathrm{eff}(C)$：压缩配置对应的压缩效率。效率越高，整体误差越小。
- $a$：模型规模项系数。
- $b$：参数量对 loss 的影响指数。$b$ 越大，增加模型参数量越有效。
- $c$：数据不足对 loss 的影响系数。
- $d$：训练数据量对 loss 的影响指数。
- $e$：固有误差。

该公式假设压缩只影响有效参数量，而不影响训练数据量，是比 1.2 节更宽泛的 Scaling Law。本质上，它在标准 Scaling Law 中加入了 $\mathrm{eff}$ 项。每种压缩配置都需要通过多次实验和 loss 统计来拟合其 $\mathrm{eff}$，具体过程略。

> **图片待补充：BF16 训练配置迁移至 3 bit 的实验结果**
>
> 论文将 BF16 训练配置直接迁移到 3 bit，发现二者的训练走势非常相似。这表明高精度训练的学习率、batch size 等超参数通常也适用于低比特训练，不必针对每个 bit-width 进行大规模调参。

**指导意义：**

- 进行 NVFP4/MXFP8 小规模 sweep 时，可以先沿用 BF16/MXFP8 baseline 的学习率和 batch size。
- 这并不意味着永远不需要调参。在 EPM 初筛阶段，可以减少超参数维度，优先研究 $N$、$D$、$C$。
- 对 NVFP4 full training 等极低 bit 训练，仍建议检查 learning rate、loss scaling、stochastic rounding 和 activation scaling 是否稳定。

## 2. 低比特训练与量化感知训练

### 2.1 主要区别

低比特训练的主要目标是节省训练成本。其 forward 和 backward 中的 GEMM 使用低精度，但权重梯度的记录和权重参数的更新仍使用高精度。

QAT 的主要目标是让模型提前适应低精度量化部署。它在模型中加入伪量化节点，使模型在训练阶段感知量化噪声，从而降低量化部署时的误差。QAT 的 forward 和 backward 仍使用高精度。

**QAT：**在部署时需要量化的位置插入伪量化节点，并使用 STE（Straight-Through Estimator，直通估计器）传播梯度。量化误差会反映到训练 loss 中，参数更新时也会考虑该误差。

> **图片待补充：QAT 训练流程**

**低比特训练：**训练包含三个 GEMM 和一次参数更新：

1. 前向传播并记录计算图：

   $$
   Y=XW
   $$

2. 反向传播，计算输入梯度：

   $$
   dX=dY\,W^T
   $$

3. 反向传播，计算权重梯度：

   $$
   dW=X^T dY
   $$

4. 更新参数：

   $$
   W\leftarrow W-\mathrm{lr}\cdot dW
   $$

前三个 GEMM 使用低精度，以提高吞吐和计算速度。更新参数时，$dW$ 以高精度存储，$W$ 反量化为高精度后再应用梯度，更新完成后重新量化。

另一类方案始终以高精度存储 $W$，仅在 forward 和 backward 时临时将其量化为低精度。

一般来说，LLM 会先训练一个 base model，再对其进行低比特微调或 QAT。

### 2.2 FP8 Formats for Deep Learning

参考：[FP8 Formats for Deep Learning](https://arxiv.org/pdf/2209.05433)

#### 2.2.1 整体流程

FP8 training 的前向传播和两次反向传播均使用 FP8。反向传播得到的 $dW$ 以高精度格式更新 master weight；master weight 更新后，同时生成 `weight_fp8`，供下一次前向传播和反向传播使用。

> **图片待补充：FP8 training 流程**

#### 2.2.2 双格式系统

FP8 训练采用双格式系统：

- 权重和前向传播的 activation 使用 E4M3，因为前向传播更需要精度。
- Activation gradient 使用 E5M2，因为反向传播梯度的动态范围更大，并且梯度更关注方向，对具体数值精度的敏感度较低。

### 2.3 DeepSeek-V3 FP8 Training

参考：[DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437)

> **图片待补充：DeepSeek-V3 FP8 训练方案**

DeepSeek-V3 采用了 FP8 Formats for Deep Learning 的训练思路，并进行了以下改进：

1. **更细粒度的 scale：**使用更细粒度的 scale 缓解 outlier。与 MXFP8 的 block scale 不同，activation 使用 $1\times N_C$ tile scale，weight 使用 $N_C\times N_C$ block scale，其中 $N_C=128$。即 activation 每 $1\times128$ 个元素共享一个 scale，weight 每 $128\times128$ 个元素共享一个 scale。

2. **关键模块保留高精度：**前向传播和反向传播的三个主要 GEMM 使用 FP8，但以下模块保留高精度：

   - embedding module
   - output head
   - MoE gating modules
   - normalization operators
   - attention operators
   - master weights
   - weight gradients
   - optimizer states

3. **优化器精度调整：**改进 Adam optimizer，将一阶矩和二阶矩改为 BF16，gradients 保持 FP32。

4. **Dgrad 使用 E4M3：**由于 scale 粒度更细，outlier 的影响降低，因此 Dgrad 不再需要 E5M2，可改用 E4M3 提高精度。

5. **采用 online scaling：**

   - Delayed scaling：使用前几轮的 amax 计算当前 block 的 scale。
   - Online scaling：在量化时实时计算当前 block 的 scale。

6. **MoE 低精度通信：**MoE 训练中的 all-to-all communication 是主要瓶颈。DeepSeek-V3 采用以下方案：

   - MoE up-projection 前的 activation 先量化为 FP8，再进行 dispatch。
   - MoE down-projection 前的 activation gradient 采用类似的 FP8 量化。
   - Forward/backward combine 保留 BF16，以保护关键精度。
   - 部分通信相关 activation 的 scale 使用 2 的整数次幂。
