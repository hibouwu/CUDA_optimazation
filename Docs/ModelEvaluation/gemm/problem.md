# GEMM 问题定义

## 1. 计算定义

给定矩阵 A、B、C 和标量 α、β，计算 D：

\[
D=\alpha AB+\beta C,\qquad
A\in\mathbb{R}^{M\times K},\quad
B\in\mathbb{R}^{K\times N},\quad
C,D\in\mathbb{R}^{M\times N}.
\]

\[
D_{ij}=\alpha\sum_{k=0}^{K-1}A_{ik}B_{kj}+\beta C_{ij}.
\]

以单次稠密实数 GEMM 为基本计算单位，采用经典乘加算法。

## 2. 输入与输出要求

记 \(\mathbb N_+=\{1,2,\ldots\}\)、\(\mathbb N_0=\{0,1,\ldots\}\)，\(\mathbb F_t\subset\mathbb R\) 为格式 \(t\) 可表示的有限实数集合。当前候选浮点格式集合为：

\[
\mathcal F=\{\mathrm{FP64},\mathrm{FP32},\mathrm{FP16},\mathrm{BF16},
\mathrm{FP8\_E4M3},\mathrm{FP8\_E5M2},
\mathrm{FP6\_E2M3},\mathrm{FP6\_E3M2},\mathrm{FP4\_E2M1}\}.
\]

低精度格式名称采用 [NVIDIA PTX 的格式定义](https://docs.nvidia.com/cuda/parallel-thread-execution/#alternate-floating-point-data-formats)。TF32 作为允许的计算格式单独指定，不作为本表的独立存储类型。以下是问题的候选域，各参数还须同时满足表中的约束；具体硬件支持的子集在硬件文档中定义。

| 维度 | 参数 | 定义域与约束 |
|---|---|---|
| 问题集合 | 类型、任务数 \(L\) 或 \(G\) | 类型 \(\in\{\mathrm{Single},\mathrm{Batched},\mathrm{Grouped}\}\)。Single 为一项；Batched 的 \(L\in\mathbb N_+\)，各项 M、N、K 相同；Grouped 的 \(G\in\mathbb N_+\)，每项分别指定尺寸及其余输入输出参数。批量中的任务相互独立。 |
| 矩阵形状 | \(M,N,K\) | \((M,N,K)\in\mathbb N_+^3\)；Grouped 中每个 \((M_g,N_g,K_g)\) 同样取正整数。当前范围不含零尺寸退化问题，不要求尺寸为某个硬件分块或指令形状的整数倍。 |
| 数据类型 | \(t_A,t_B,t_C,t_D,t_\alpha,t_\beta\) | 每个类型分别取自 \(\mathcal F\)，不强制相同。\(\beta=0\) 且不提供 C 时，\(t_C\) 可记为 `null`。具体组合能否在某硬件上原生执行另行判断。 |
| 元素取值 | A、B、C 的元素及其取值范围 | 普通浮点的 \(X_{ij}\in\mathbb F_{t_X}\)；带缩放表示的元素按下行解码。问题实例可以进一步限定取值区间或集合，默认包含所选表示的全部有限值；NaN、Inf 输入不在当前范围内。 |
| 数值表示 | 各矩阵的表示方式、缩放格式、分组与布局 | 表示方式 \(\in\{\text{普通浮点},\text{带缩放浮点}\}\)。带缩放时 \(X_{ij}=s_{g(i,j)}q_{ij}\)，\(q_{ij}\in\mathbb F_{t_X}\)，\(s_g\in\mathbb F_{t_s}\cap\mathbb R_{\ge0}\)，\(t_s\in\mathcal F\cup\{\mathrm{UE8M0}\}\)。分组是元素索引集的非空分区，每组对应一个缩放因子；组大小为正整数。需给出分组映射及缩放数组的形状、布局；整矩阵缩放为一组，Block-Scaled 为多组。D 使用带缩放表示时，输出包含数值载荷和缩放因子。 |
| 计算精度 | 允许的运算模式集合 \(\mathcal A\) | \(\mathcal A\) 为非空集合。每个模式分别指定 A、B 的乘法操作数格式 \(\in\mathcal F\cup\{\mathrm{TF32}\}\)，累加及标量计算格式 \(\in\mathcal F\)，以及舍入、融合乘加和输出转换规则；输出表示须与 D 的要求一致。方案只能在 \(\mathcal A\) 中选择，并满足误差要求；尚未确定的约束记为待定，不视为无限制。 |
| 误差要求 | 误差函数 \(E\)、容差 \(\varepsilon\) | \(E:\mathbb R^{M\times N}\times\mathbb R^{M\times N}\to\mathbb R_{\ge0}\)，\(\varepsilon\in\mathbb R_{\ge0}\) 且有限；要求 \(E(\widehat D,D^\star)\le\varepsilon\)。需明确 E 的公式，例如最大绝对误差或相对 Frobenius 误差；使用相对误差时需定义参考值为零的处理。\(D^\star\) 为按输入表示解码后，以第 1 节实数公式定义的结果，\(\widehat D\) 为输出解码后的结果。 |
| 存储布局 | 各矩阵的顺序、\(ld_X\)、C/D 是否共用存储 | 顺序 \(\in\{\text{行主序},\text{列主序}\}\)。对形状为 \(r_X\times c_X\) 的矩阵，\(ld_X\in\mathbb N_+\)，行主序要求 \(ld_X\ge c_X\)，列主序要求 \(ld_X\ge r_X\)，单位为元素。C/D 关系 \(\in\{\text{独立存储},\text{完全共用存储}\}\)；共用时类型、表示和布局须兼容，不允许部分重叠或 D 覆盖 A/B。低于 8 bit 的格式另需声明实际打包方式。 |
| 批量寻址 | 寻址方式、批次步长或地址列表 | Batched 的寻址方式 \(\in\{\text{固定步长},\text{逐项地址}\}\)；Grouped 逐项给出地址与布局。固定步长 \(\mathrm{stride}_X\in\mathbb N_0\)，以该矩阵存储元素为单位；地址列表长度等于任务数，各地址须指向足够的有效存储。输入允许共享只读数据，输出间不得重叠，也不得覆盖其他任务仍需读取的数据；带缩放表示需同时给出缩放数组的寻址方式。 |
| 标量系数 | \(\alpha,\beta\) | \(\alpha\in\mathbb F_{t_\alpha}\)，\(\beta\in\mathbb F_{t_\beta}\)。\(\beta=0\) 时 C 可不提供，其类型与布局可记为 `null`，且无需读取 C；\(\beta\ne0\) 时必须提供 C。\(\alpha=1,\beta=0\) 时为 \(D=AB\)。 |
| 设备分布 | 执行范围、输入与输出的设备位置 | 执行范围 \(\in\{\text{单 GPU},\mathrm{Distributed}\}\)，与 Single/Batched/Grouped 独立。对参与设备集合 \(\mathcal G\)，单 GPU 要求 \(\operatorname{card}(\mathcal G)=1\)，Distributed 要求 \(\operatorname{card}(\mathcal G)\ge2\)。每个输入元素的初始位置及每个输出元素要求的最终位置，均为 \(\mathcal G\) 的非空子集，可表达分片或复制；带缩放表示同时指定缩放因子的位置。实际设备数量与互连能力在硬件文档中给出，计算划分与通信方式由方案选择。 |

## 3. 性能指标

按一次乘加计两次浮点运算，GEMM 主乘加部分的计算量为：

\[
W=2MNK\quad\text{FLOP}.
\]

Batched 的总工作量为 \(2LMNK\)，Grouped 为 \(\sum_g 2M_gN_gK_g\)。该口径只统计问题要求的主乘加工作；填充、重算、归约和结果处理的实际执行量在方案中另计，其耗时仍包含在完成时间中。Distributed GEMM 不因 GPU 数增加而重复计算工作量。

默认计时边界为：输入按约定布局在设备 GMEM 就绪并开始执行本次任务，到全部 D 在约定同步条件下可被后继消费者使用。完成任务所需的布局转换、计算、归约及写回均在边界内；不额外要求缓存内容强制刷新至物理外存。主机传输、分配和调用开销若属于研究目标，应另行声明端到端边界。Batched、Grouped 和 Distributed 均使用整个任务的完成时间。

设执行时间为 \(T\) 秒，报告 \(T\) 与有效吞吐量：

\[
P_{\mathrm{s}}=\frac{W}{T}\quad\text{FLOP/s}.
\]

若采用固定参考频率 \(f_{\mathrm{ref}}\)（Hz）归一化，则：

\[
N_{\mathrm{cycle,ref}}=T f_{\mathrm{ref}},\qquad
P_{\mathrm{cycle,ref}}=\frac{W}{N_{\mathrm{cycle,ref}}}\quad\text{FLOP/reference\ cycle}.
\]

注明参考频率及设备范围。这里是整个任务的工作量除以经过的参考周期，不再乘以 SM 数；频率变化时，参考周期不等于实际累计 SM 时钟周期。跨硬件比较同时保留秒与 FLOP/s。
