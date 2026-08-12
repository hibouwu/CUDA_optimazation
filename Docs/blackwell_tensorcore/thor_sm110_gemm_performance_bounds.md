# Thor/SM110 稠密 GEMM：条件性能上界、经验理想包络与实测最好值

> **研究目标**：回答“在 Thor/SM110 的物理约束下，一个没有可避免性能浪费的稠密 GEMM 最快可以到哪里”，而不是只预测仓库中的 `tc3`。
>
> **模型状态**：结构模型、证据分级、工作量计算、完整 GEMM 结果导入和缺口审计已经可执行；全精度硬件数值闭环尚未完成。
>
> **可信度纪律**：可证明上界、microbenchmark 经验包络和完整 GEMM 实测值分别报告，任何一层都不能冒充另一层。
>
> **范围**：单次、稠密、经典矩阵乘法；不包含稀疏、Strassen/近似算法、batched/grouped GEMM 和多 GPU 通信。

## 1. 先给出结论

“比 cuBLAS 还完美的 GEMM”有三种不同含义，必须先分开：

1. **条件性能上界**描述在一组明确的硬件容量上界和算法假设下，任何合法实现都不能超过的位置。
2. **经验理想包络**描述按照现有 microbenchmark 测得的组件能力和已枚举的合法 schedule，一个消除了已知实现损失的 GEMM 应该达到的位置。
3. **完整 GEMM 已观测最好值**描述 cuBLAS、cuBLASLt、CUTLASS 和仓库 kernel 中，已经通过 correctness 验证的最好实测性能。

定义 \(P_{\mathrm{obs}}\) 为完整 GEMM 已观测最好性能；对浮点模式其单位为
FLOP/s，对 S8/U8 整数模式其单位为 OP/s。定义 \(P^\star\) 为所有物理可实现
GEMM 中真实但未知的最好性能；定义 \(P_{\mathrm{ub}}\) 为条件性能上界，二者
使用与 workload 相同的性能单位。在上界假设全部成立且
workload 语义一致时，必须满足：

\[
P_{\mathrm{obs}}\le P^\star\le P_{\mathrm{ub}}.
\]

定义 \(\widehat P_{\mathrm{env}}\) 为 microbenchmark 驱动的经验理想包络，单位
同 workload。它不自动进入上面的不等式。原因是 microbenchmark 实测值只能证明
“硬件至少已经做到这么快”，不能证明“硬件绝不可能更快”。

因此本文不会把最高 microbenchmark TFLOP/s 直接称为硬件物理上限。如果完整
GEMM 超过 \(\widehat P_{\mathrm{env}}\)，说明经验模型需要重校准；如果完整 GEMM
在语义相同的前提下超过 \(P_{\mathrm{ub}}\)，则上界的容量、工作量或适用条件至少
有一项错误。

## 2. 冻结第一版 GEMM 语义

定义 \(M\) 为输出矩阵的行数，单位为 element；定义 \(N\) 为输出矩阵的列数，
单位为 element；定义 \(K\) 为 reduction 维度，单位为 element。第一版 workload
计算：

\[
D=\alpha\operatorname{op}(A)\operatorname{op}(B)+\beta C.
\]

在上式中，\(A\) 是输入矩阵 A，\(B\) 是输入矩阵 B，\(C\) 是可选的输入累加
矩阵，\(D\) 是输出矩阵；\(\operatorname{op}(\cdot)\) 表示保持原布局或转置；
\(\alpha\) 是矩阵乘积的标量系数；\(\beta\) 是矩阵 C 的标量系数。当
\(\beta=0\) 时，理想实现不必读取 C。

第一版固定以下边界：

- \(M,N,K\) 都是正整数，且 \(\alpha\ne0\)；
- workload schema 能声明 NN、NT、TN 和 TT 四种 A/B 转置组合；当前 v1 的最低
  工作量对四者相同，但 schedule manifest 尚未把布局与 leading dimension 的
  合法 data-movement 路径枚举完整，因此不能把当前搜索结果称为四种布局都完备；
- 独立 leading dimension 尚未进入可执行 schema；当前工作量公式只统计逻辑
  元素，不把行间 padding 当作必然传输量；
- v1 可执行数值模型只接受 `epilogue=none`。bias、ReLU、GELU、residual 和
  requant 是后续合同；在其工作量和 I/O 未实现前会 fail closed；
- 计时默认只包括设备端 GEMM，不包括 host-device copy、内存分配和一次性
  prepack；如果业务要求包含，必须在 workload 中单独声明；
- `cold_hbm` 表示模型入口只保证输入位于设备内存，`hot_l2` 表示输入工作集已
  预热到 L2，`compute_oracle` 表示操作数已经位于 Tensor Core 可消费位置；
- 经典 GEMM 的用户可见数学工作量固定为 \(2MNK\) 次标量操作，不允许用稀疏或快速矩阵
  乘法减少这一定义。

batched/grouped GEMM、跨算子 persistent reuse、稀疏 GEMM、Strassen、近似
矩阵乘法、多 GPU、PCIe/NVLink 和 host-device copy 不在第一版内。对声明范围
的“完备”不等于声称覆盖这些排除项。

## 3. 第一版精度合同

定义 \(s_{\mathrm{in}}\) 为单个 A/B 输入元素的平均存储字节数，单位为
B/element；定义 \(s_{\mathrm{acc}}\) 为单个 accumulator 元素的存储字节数，
单位为 B/element；定义 \(s_{\mathrm{out}}\) 为单个输出元素的存储字节数，单位
为 B/element；定义 \(K_{\mathrm{mma}}\) 为该精度 MMA 原子在 K 方向一次消费的
元素数，单位为 element/instruction。

| 模式 ID | \(s_{\mathrm{in}}\) | \(s_{\mathrm{acc}}\) | \(s_{\mathrm{out}}\) | \(K_{\mathrm{mma}}\) | scale 语义 |
| --- | ---: | ---: | ---: | ---: | --- |
| `fp16_f32` | 2 | 4 | 4 | 16 | 无 |
| `bf16_f32` | 2 | 4 | 4 | 16 | 无 |
| `tf32_f32` | 4 | 4 | 4 | 8 | 无 |
| `e4m3_f32` | 1 | 4 | 4 | 32 | 无 |
| `e5m2_f32` | 1 | 4 | 4 | 32 | 无 |
| `e3m2_f32` | 0.75 | 4 | 4 | 32 | 无 |
| `e2m3_f32` | 0.75 | 4 | 4 | 32 | 无 |
| `e2m1_f32` | 0.5 | 4 | 4 | 32 | 无 |
| `mxfp4_f32` | 0.5 | 4 | 4 | 64 | 每 32 个输入值 1 B UE8M0 scale |
| `nvfp4_f32` | 0.5 | 4 | 4 | 64 | 每 16 个输入值 1 B UE4M3 scale |
| `s8_s32` | 1 | 4 | 4 | 32 | signed INT8 → INT32 |
| `u8_s32` | 1 | 4 | 4 | 32 | unsigned INT8 → INT32 |

FP6/FP4 的分数字节只表示 workload 合同所声明的紧凑逻辑存储下界，不表示
SMEM/TMEM 或某条 copy 指令必然以 0.75 B/0.5 B 的物理 transaction 搬运。
例如 PTX ISA 9.0 的 `tcgen05.cp` FP6 解压格式是每 16 个 6-bit 元素（12 B）
再带 4 B padding；若 schedule 使用这条路径，其 issued traffic 必须按实际
16 B 统计，不能沿用 12 B 的逻辑下界。raw `f8f6f4` 直接从 SMEM 消费和先经
`tcgen05.cp` 解压到 TMEM 是两种不同 schedule，模型必须分别声明。

MXFP4/NVFP4 的 value bytes、input scale bytes、accumulator bytes、output
value bytes 和 output scale bytes 必须分开统计。特别是 FP32 accumulator 从
TMEM 回读时仍按 \(s_{\mathrm{acc}}=4\) B/element 统计，不能用 packed E2M1 的
0.5 B/element 代替。

## 4. 工作量：先证明必须做什么

### 4.1 用户工作与发射工作

定义 \(W_{\mathrm{use}}\) 为用户要求的经典 GEMM 标量计算工作量；浮点模式单位
为 FLOP，S8/U8 模式单位为 OP：

\[
W_{\mathrm{use}}=2MNK.
\]

这里一次 multiply-add 按一次乘法和一次加法计为 2 个标量操作；只有浮点
workload 才把它称为 2 FLOP，整数 workload 写成 2 OP。

定义 \(B_M\)、\(B_N\) 和 \(B_K\) 分别为 schedule 的 CTA tile 在 M、N、K
方向的元素数，单位均为 element；定义 \(N_M=\lceil M/B_M\rceil\)、
\(N_N=\lceil N/B_N\rceil\) 和 \(N_K=\lceil K/B_K\rceil\)，分别表示三个方向
的 tile 数。

若 schedule 用完整 tile padding 边界，定义 \(W_{\mathrm{issue}}\) 为实际发射
的计算工作，单位与 \(W_{\mathrm{use}}\) 相同：

\[
W_{\mathrm{issue}}
=2(N_MB_M)(N_NB_N)(N_KB_K)+W_{\mathrm{reduce}}.
\]

定义 \(W_{\mathrm{reduce}}\) 为 split-K 最终 reduction 增加的计算工作，单位
同 workload。若使用恰好覆盖边界的专用 tail kernel，则 padding 项可以消失，但 tail
kernel 的合法指令粒度和固定成本仍需单独建模。当前 v1 对非整除 shape 的 `exact`
schedule 直接拒绝；只有加入该 tail kernel 的显式 manifest 后才会放行，避免凭空
假设任意尺寸的无 padding 指令。

定义形状效率 \(\eta_{\mathrm{shape}}\) 为：

\[
\eta_{\mathrm{shape}}=\frac{W_{\mathrm{use}}}{W_{\mathrm{issue}}}.
\]

### 4.2 通用最小 I/O

定义 \(Q_{\mathrm{in,val}}^{\mathrm{LB}}\) 为至少读取一次 A/B value 的字节数下界，
单位为 B：

\[
Q_{\mathrm{in,val}}^{\mathrm{LB}}=(MK+KN)s_{\mathrm{in}}.
\]

对 block-scaled 精度，定义 \(b_s\) 为一个 scale 覆盖的 value 个数，单位为
element/scale；定义 \(s_s\) 为单个 scale 的字节数，单位为 B/scale。输入
scale 的字节数下界为：

\[
Q_{\mathrm{in,scale}}^{\mathrm{LB}}
=
\left\lceil\frac{MK}{b_s}\right\rceil s_s
+
\left\lceil\frac{KN}{b_s}\right\rceil s_s.
\]

无 scale 的精度令这一项为 0。

定义 \(Q_C^{\mathrm{LB}}\) 为 C 的最小读取量，单位为 B：

\[
Q_C^{\mathrm{LB}}=
\begin{cases}
0,&\beta=0,\\
MN s_{\mathrm{acc}},&\beta\ne0.
\end{cases}
\]

定义 \(Q_D^{\mathrm{LB}}\) 为最终输出至少写一次的字节数，单位为 B。普通
accumulator 输出时 \(Q_D^{\mathrm{LB}}=MN s_{\mathrm{out}}\)；packed quantized
输出还必须加 value packing 和 output scale。

上述公式只是对任何经典实现都成立的最小 I/O。具体 schedule 的 TMA 逻辑流量
可能因 output tile 重用边界、CTA group、split-K 和 cache 行为显著增加。不能
把 `TMA payload bytes`、`L2 request bytes` 和 `DRAM physical bytes` 当作同一
个量。

### 4.3 可执行输入参数合同

下面列出代码中出现、但不一定进入上述闭式公式的字段。字段首次在本表出现时即
给出定义与单位，避免同名参数靠读者猜测。

| workload 字段 | 定义与单位 |
| --- | --- |
| `workload_id` | workload 的稳定字符串标识，无单位 |
| `m`, `n`, `k` | 分别对应 \(M,N,K\)，单位 element |
| `precision_id` | 第 3 节某一精度合同的稳定标识，无单位 |
| `transpose_a`, `transpose_b` | 是否对 A/B 取转置的布尔值；v1 只执行二者均为 false 的 NN 路径 |
| `alpha`, `beta` | 分别对应 \(\alpha,\beta\) 的无量纲标量 |
| `epilogue` | 输出后处理语义枚举；v1 仅实现 `none` |
| `residency` | 入口数据驻留合同：`cold_hbm`、`hot_l2` 或 `compute_oracle` |
| `output_mode` | `accumulator` 或 `packed_quantized` 输出存储合同；v1 只实现前者 |
| `include_launch` | 是否允许经验层计入 launch/fixed time 的布尔值 |

| schedule 字段 | 定义与单位 |
| --- | --- |
| `schedule_id` | schedule 的稳定字符串标识，无单位 |
| `bm`, `bn`, `bk` | 对应 \(B_M,B_N,B_K\) 的 CTA tile 尺寸，单位 element |
| `stages` | pipeline 中同时驻留的 stage 数，单位 stage |
| `mma_m`, `mma_n` | 一条 MMA atom 的 M/N 尺寸，单位 element/instruction |
| `cta_group` | 一条操作协作的 CTA 数，只能为 1 或 2，单位 CTA/group；v1 只执行 1 |
| `split_k` | K 方向独立 partial 数，单位 partition；v1 只执行 1 |
| `tail_policy` | `pad` 或 `exact`；非整除 `exact` 在 v1 中 fail closed |
| `supported_precisions` | schedule 显式允许的 `precision_id` 集合 |
| `smem_limit_bytes` | 单 CTA 可用于该 schedule 的 SMEM 上限，单位 B/CTA |
| `tmem_columns` | 分配的 TMEM column 数，单位 column/CTA |
| `threads` | CTA 线程数，单位 thread/CTA |
| `registers_per_thread` | 可选寄存器占用，单位 32-bit register/thread |
| `uses_tma` | 是否声明使用 TMA data path 的布尔值 |
| `input_transport_layout` | 输入物理搬运布局；`logical_packed`、`byte_padded`、`b6x16_p32` 或 `b4x16_p64` |
| `persistent` | 是否声明 persistent 调度的布尔值；v1 对 true fail closed |
| `fixed_seconds` | 经验层已测固定成本，单位 s；严格层不使用实测固定成本 |

| hardware 字段 | 定义与单位 |
| --- | --- |
| `hardware_id` | 硬件配置的稳定标识，无单位 |
| `sm_count` | 可用 SM 数，单位 SM/GPU |
| `clock_hz` | 被记录的 GPU 时钟，单位 cycle/s；它是环境字段，不替代实测 elapsed time |

| capacity 字段 | 定义与单位 |
| --- | --- |
| `capacity_id` | 容量记录的稳定标识，无单位 |
| `resource` | 被约束资源的稳定标识，无单位 |
| `rate_per_second` | SI 基础单位下的服务率，单位由 `work_unit`/s 决定 |
| `work_unit` | `flop`、`operation` 或 `byte` |
| `evidence_kind` | 第 5.1 节定义的逻辑证据等级 |
| `source_id` | 来源记录的稳定标识，无单位 |
| `source_path`, `source_locator` | 仓库内文件路径及文件内可机械定位条件 |
| `source_url` | 可选外部一手来源 URL |
| `original_value`, `original_unit` | 来源中的未换算数值及单位 |
| `condition` | 容量成立所需的 workload、功耗、频率或工具条件 |
| `uncertainty_fraction` | 相对不确定度，范围 \([0,1)\)，无量纲；v1 保存但尚不传播到中心值 |
| `qualification` | `snapshot_only`、`closure_qualified` 或 `quarantined` |
| `trial_count` | 支持该记录的独立 trial 数，单位 trial |
| `artifact_paths` | closure 所依赖的源码、原始结果、SASS/NCU、环境或 hash 路径集合 |

## 5. 第一层：条件可证明性能上界

定义资源集合 \(\mathcal R\) 为模型采用的硬件资源约束集合。对其中每个资源
\(r\in\mathcal R\)，定义 \(Q_r^{\mathrm{LB}}\) 为任何合法实现至少需要在资源
\(r\) 上完成的工作，单位可能是 FLOP、B、instruction 或 transaction；定义
\(U_r\) 为资源 \(r\) 的服务率上界，单位与 \(Q_r^{\mathrm{LB}}\) 每秒对应。

只有当证据能支持“真实服务率不大于 \(U_r\)”时，才有：

\[
T_r^{\mathrm{LB}}=\frac{Q_r^{\mathrm{LB}}}{U_r}.
\]

定义 \(T_{\mathrm{resource}}^{\mathrm{LB}}\) 为全部独立资源工作下界：

\[
T_{\mathrm{resource}}^{\mathrm{LB}}
=\max_{r\in\mathcal R}T_r^{\mathrm{LB}}.
\]

取最大值代表允许资源完美重叠，是一个乐观时间下界。若两个资源共享端口或不能
同时达到各自峰值，必须增加联合容量约束，而不是把两个时间任意相加。

### 5.1 证据等级

每个容量参数都带 `evidence_kind`：

| 等级 | 含义 | 能否进入条件上界 |
| --- | --- | --- |
| `specified_upper` | 官方规格或明确架构合同给出的服务率上限 | 可以，需记录条件 |
| `derived_upper` | 从可复核的 issue、频率或端口约束推导 | 可以，需保留推导 |
| `profiler_model_peak` | NCU `%peak` 等工具内部模型峰值 | 只能形成带工具假设的条件上界 |
| `measured_sustained` | 独立 microbenchmark 实测持续值 | 不可以，只进入经验层 |
| `measured_joint` | 联合 microbenchmark 的实测工作点 | 不可以作为容量外边界 |
| `observed_gemm` | 完整 GEMM 实测 | 只用于已观测层和反证 |
| `derived_work` | 由 workload/schedule 推导的工作量 | 可用于对应层 |
| `unknown` | 当前无有效证据 | 不进入数值约束 |

选择 `unknown` 或省略约束会让条件上界变松，但仍保持逻辑方向正确。用一个测得的
sustained 值冒充 \(U_r\) 会产生虚假的低上界，可能被真实 GEMM 轻易超过。

### 5.2 有限并行与尾部

定义 \(n_t\) 为不可再分割的任务数，单位为 task；定义 \(p_i\) 为第 \(i\) 个
任务的最小服务时间，单位为 s；定义 \(U_t\) 为能同时服务这类任务的等价硬件
单元数，单位为 service unit。有限并行时间至少满足：

\[
T_{\mathrm{parallel}}^{\mathrm{LB}}
\ge
\max\left(
\frac{\sum_{i=1}^{n_t}p_i}{U_t},
\max_i p_i
\right).
\]

若全部任务同构且每个耗时 \(p\)，理想 makespan 才能写成：

\[
T_{\mathrm{parallel,identical}}
=\left\lceil\frac{n_t}{U_t}\right\rceil p.
\]

本文不统一乘一个“最后一波效率”。persistent、stream-K、多 resident CTA 和
专用 tail kernel 都会改变任务分解；将一个 wave 系数同时乘到 compute、TMA、
HBM 和 critical path 上容易重复计算尾部。

当前可执行 v1 只在经验层使用上述 per-group span/makespan。把全 GPU 规格峰值
平均除以 SM 数并不能单独证明单 CTA 的最大服务率，所以严格层在没有 per-SM 或
per-CTA 证据时不加这项约束；它保留全 GPU 总工作/总容量约束。代价是小 shape 的
严格上界更松，收益是不会用未经证明的“均匀切分”制造虚假上界。

### 5.3 因果关键路径

定义执行依赖图 \(G=(V,E)\)，其中 \(V\) 是 load、MMA、wait、TMEM readback、
epilogue 等阶段，\(E\) 是生产者到消费者的真实依赖。定义
\(T_{\mathrm{span}}^{\mathrm{LB}}\) 为图中任一合法执行都无法缩短的最长依赖链，
单位为 s。

例如，下一块 TMA load 和当前 MMA 可以在双缓冲稳态重叠；但 accumulator 完成
前不能读取最终 TMEM 结果。具体 schedule 可以用
`first load + steady max(load, compute) + last compute`，通用上界则用
work/span 形式，避免把 `tc3` 的固定阶段顺序误当成所有 GEMM 的唯一顺序。

### 5.4 联合容量区域

定义资源吞吐向量 \(\mathbf y\) 为同一时刻 TMA、L2、SMEM、TMEM、Tensor Core
等资源的服务率组合；定义矩阵 \(A\) 为联合资源线性约束的系数矩阵；定义向量
\(\mathbf b\) 为每条联合约束的容量上限。若能证明硬件容量外边界满足：

\[
A\mathbf y\le\mathbf b,
\]

则 \(A\) 和 \(\mathbf b\) 的每一行都能生成一条联合时间下界。定义
\(T_{\mathrm{joint}}^{\mathrm{LB}}\) 为所有已证明联合容量约束给出的最大时间下界，
单位为 s；没有有效外边界时该项为 0。

联合 microbenchmark 只提供一个可实现的 \(\mathbf y\) 点，即容量区域内点。它
能校准经验模型，却不能单独证明外边界。仓库现有 `tcgen05.cp`/MMA overlap
结果因此标为 `measured_joint`，不用于无条件上界。

### 5.5 总时间下界

定义 \(T_{\mathrm{fixed}}^{\mathrm{LB}}\) 为已经证明不可消除的 launch、分配或
同步固定时间下界，单位为 s。实测 launch latency 默认不是“不可更短”的证明；
没有可信下界时严格层令这一项为 0。

条件总时间下界为：

\[
T_{\mathrm{ub}}^{\mathrm{LB}}
=
\max\left(
T_{\mathrm{resource}}^{\mathrm{LB}},
T_{\mathrm{parallel}}^{\mathrm{LB}},
T_{\mathrm{span}}^{\mathrm{LB}},
T_{\mathrm{joint}}^{\mathrm{LB}},
T_{\mathrm{fixed}}^{\mathrm{LB}}
\right).
\]

条件性能上界为：

\[
P_{\mathrm{ub}}=\frac{W_{\mathrm{use}}}{T_{\mathrm{ub}}^{\mathrm{LB}}}.
\]

## 6. 第二层：microbenchmark 驱动的经验理想包络

定义 \(\widehat C_r\) 为资源 \(r\) 在匹配精度、shape、CTA group、频率、cache
状态和 occupancy 条件下测得的经验服务率，单位与资源工作每秒对应。

定义 workload 描述 \(w\) 为尺寸、精度、转置、\(\alpha/\beta\)、epilogue、
residency 和计时边界的集合；定义 schedule 描述 \(x\) 为 tile、MMA atom、
stage、CTA group、split/stream-K、persistent、tail 和资源 footprint 的集合。

对一个合法 schedule，定义 \(Q_r(x,w)\) 为 schedule \(x\) 执行 workload \(w\)
时向资源 \(r\) 发出的工作量，单位与 \(\widehat C_r\) 的分子一致。经验时间估计记为：

\[
\widehat T(x,w)
=
\max_r\frac{Q_r(x,w)}{\widehat C_r},
\]

其中还需加入该 schedule 的 dependency span、有限并行 makespan、固定成本和
已测得的联合资源修正。定义 \(\mathcal X_{\mathrm{manifest}}\) 为已经通过 PTX、
descriptor、SMEM/TMEM/register、occupancy 和 launch 合法性检查的 schedule
集合。经验理想包络为：

\[
\widehat T_{\mathrm{env}}(w)
=
\min_{x\in\mathcal X_{\mathrm{manifest}}}\widehat T(x,w),
\qquad
\widehat P_{\mathrm{env}}(w)
=
\frac{W_{\mathrm{use}}}{\widehat T_{\mathrm{env}}(w)}.
\]

这只是已枚举 schedule 中的理想预测。若 manifest 没有包含一种新算法，搜索
结果不能称为所有 GEMM 的绝对上界。经验层缺少任一必需 resource capacity 时
直接返回 `insufficient_evidence`，不输出用剩余资源拼出的“半个包络”；严格层则
可以保留单条已证明约束形成的合法但较松上界，并把状态标为 `partial`。

## 7. 第三层：完整 GEMM 已观测最好值

定义一个 eligible backend series 为同一 workload 上至少 10 个 trial、全部
`Matched=1`、没有 missing/timeout/launch failure 且 GFLOP/s 为正的完整 GEMM
结果序列。

定义 \(b\) 为 backend 的索引，定义 \(j\) 为同一 backend series 内 trial 的索引，
定义 \(P_{b,j}\) 为第 \(j\) 次 trial 的性能，单位与 workload 相同。为降低单个
噪声尖峰的影响，当前工具先计算每个 backend series 的 median，
再选择 median 最大的 backend 作为稳定最好实现；同时保留该 series 的 minimum
和 maximum。选择规则为：

\[
P_{\mathrm{obs,median}}
=
\max_b\operatorname{median}_{j\in\mathrm{valid\ trials}}P_{b,j}.
\]

最大单 trial 值只用于检查上界违规，不作为稳定性能中心值。

“Reference”字段当前主要表示性能 denominator，不必然是 correctness reference。
例如 NVFP4/MXFP4 CUTLASS 路径可以通过自己的 host correctness reference，但
CSV 的性能 denominator 仍是 FP16 cuBLAS。工具将这种情况标记为
`cross_precision_denominator`，禁止把 ratio 解读为同精度库胜负。

## 8. 当前仓库证据状态

下面状态来自可执行 `coverage` 命令，不是人工印象：

Thor T5000 的条件规格锚点来自 NVIDIA 的官方产品表：MAXN 下 dense FP4 为
1035 TFLOP/s、dense FP8 为 517 TFLOP/s、sparse FP16 为 517 TFLOP/s，内存带宽
273 GB/s、最大 GPU 频率 1.57 GHz。本文只把表中明确的 dense 数字当
`specified_upper`；但产品表没有把 FP8 拆成 E4M3/E5M2，也没有指明 FP4 的具体
encoding，因此映射到某个 PTX 精度合同仍是显式条件。BF16/FP16 的 258.5
TFLOP/s 是按 2:1 稀疏倍率推得的 `derived_upper`，不是官方直接列出的 dense 项。规格来源见
[NVIDIA Jetson Thor 官方介绍](https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/)。

| 精度 | compute 条件上界 | compute 实测 | 完整 GEMM | 同精度性能 denominator | 当前闭环 |
| --- | --- | --- | --- | --- | --- |
| FP16 | 有（推导） | 缺 | 有 | 有 | 否 |
| BF16 | 有 | 有 | 缺 | 缺 | 否 |
| TF32 | 缺 | 缺 | 缺 | 缺 | 否 |
| FP8 E4M3 | 有 | 有（快照） | 有（快照） | 有 | 否 |
| FP8 E5M2 | 有 | 缺 | 缺 | 缺 | 否 |
| FP6 E3M2 | 缺 | 缺 | 缺 | 缺 | 否 |
| FP6 E2M3 | 缺 | 缺 | 缺 | 缺 | 否 |
| raw FP4 E2M1 | 缺 | 缺 | 缺 | 缺 | 否 |
| MXFP4 | 缺 | 缺 | 有 | 跨精度 | 否 |
| NVFP4 | 有 | **隔离** | 有（快照） | 跨精度 | 否 |
| signed INT8 | 缺 | 缺 | 有 | 有 | 否 |
| unsigned INT8 | 缺 | 缺 | 缺 | 缺 | 否 |

这里的“快照”定义为：仓库只有一个汇总数字，或者没有同一运行合同下至少 10 次
原始 trial、源码/SASS/hash/环境的闭合证据。快照可以帮助提出假设，但不能获得
`closure_qualified` 资格。因此 E4M3 也没有数值闭环；整个经验 GEMM 数据流还缺少
同构 `tmem.readback` 和 epilogue capacity。

历史 NVFP4 的 1032.111 TFLOP/s 被明确隔离：旧生成器把 PTX ISA Table 42 中
raw E2M1 的 type code `5` 写进了 Table 44 的 `mxf4nvf4` descriptor；Table 44
对 block-scaled E2M1 的编码是 `1`。`ptxas` 接受并生成 `UTCOMMA.4X` 只能证明
静态 lowering，不能证明该数值对应声明的 NVFP4 语义。新 campaign 使用独立的
Table 42/Table 44 encoder 和反例测试，在重跑前模型把旧数字标为 `unknown`、
`quarantined`，不让它进入经验包络。

当前公共资源已有经验数据：

- HBM/LPDDR streaming read 和 write；
- L2 unique read 和 end-to-end store path；
- TMA L2-hit 和 DRAM-stream ingress。

上述 TMA 数字仍是旧 `max(clock64 per CTA)` 合同下的快照。新的 closure
campaign 已把计时改为整网格最早 `%globaltimer` start 到最晚 stop，并要求 20 个
不同 SM ID；Thor 返回结果前，旧数字不升级为 `closure_qualified`。

当前公共资源硬缺口：

- 同构 TMEM accumulator readback 的 Thor 结果；其新源码已静态编译为
  `LDTM.x8`/`LDTM.x16`，但静态 lowering 不等于带宽实测；
- 各输出语义的 epilogue；
- launch/TMEM alloc/barrier 固定成本；
- TMA+MMA、MMA+readback+store 等联合容量外边界或稳定经验模型；
- calibration/holdout 上的完整预测误差。

因此当前 `all_precisions_closed=false`、`all_common_resources_closed=false` 是
预期且正确的结果，目标尚未完成。

## 9. 自动化接口和反证规则

可执行模型位于
[`scripts/sm110_gemm_model`](../../scripts/sm110_gemm_model/README.md)。

证据审计：

```bash
python3 -m scripts.sm110_gemm_model.cli audit \
  --capacities scripts/sm110_gemm_model/profiles/capacities.json \
  --repo-root .
```

精度和公共资源覆盖：

```bash
python3 -m scripts.sm110_gemm_model.cli coverage \
  --repo-root . \
  --capacities scripts/sm110_gemm_model/profiles/capacities.json \
  --observed-input results/gemm_sm110/sm110_gemm_core_128_4096_10trials.csv \
  --observed-input results/quant_gemm_sm110/sm110_quant_gemm_1024_sweep.csv
```

模型测试：

```bash
python3 -m unittest -v scripts.sm110_gemm_model.test_model
```

自动审计至少执行以下规则：

- `measured_sustained` 和 `measured_joint` 不能进入条件上界；
- NCU model peak 没有适用条件时审计失败；
- 参数无 source path、locator 或单位时审计失败；
- source path 不存在时审计失败；
- \(\beta=0\) 时最小工作量不得读取 C；
- padding 的 issued compute work 不得小于 useful compute work；
- scale bytes、accumulator bytes 和 output bytes 必须分开；
- 增大一个有效 rate upper 不能降低性能上界；
- 完整 GEMM 最大 trial 超过同语义条件上界时审计失败；
- 完整 GEMM 超过经验包络时只触发重校准，不写成物理违规；
- 跨精度 denominator 不能用于同精度效率结论；
- 浮点路径用 FLOP/s，S8/U8 路径用 OP/s，二者禁止隐式互换；
- `closure_qualified` 必须有至少 10 次 trial 和可定位的原始 artifact；
- `unknown` 证据必须显式 `quarantined`，且不能参与任一数值层；
- residency 或 timed scope 不相同的观测不得直接作上界违规判定；
- 任何精度缺 compute、完整 GEMM 或同精度 denominator 时不得标为数值闭环。

## 10. 与 `tc3` 阶段模型的关系

[`thor_sm110_gemm_stage_model.md`](./thor_sm110_gemm_stage_model.md) 对固定 FP16
`tc3` kernel 的 load/compute/epilogue 顺序做了代码对应分析，它仍然是有价值的
schedule-specific case study。

本文不是把 `tc3` 的参数换成变量名，而是改变建模层级：

- `tc3` 模型问“这个固定 kernel 为什么花这些时间”；
- 经验包络问“当前已知合法 schedule 中哪个最理想”；
- 条件上界问“在声明的硬件容量上界下任何实现都不能超过哪里”。

固定 kernel 的 TMA/MMA 双缓冲公式可以成为 \(\widehat T(x,w)\) 的一个 schedule
实例，但不能直接代表所有 GEMM。

## 11. 还需要完成的硬件闭环

每一种精度必须依次通过：

1. PTX/descriptor 合法性；
2. 预期 SASS 指令；
3. compute-only 10-trial；
4. 同构 TMA+MMA mainloop 10-trial；
5. 同构 readback/epilogue 10-trial；
6. 独立 correctness reference 的完整 GEMM 10-trial；
7. 同语义 cuBLAS/cuBLASLt/CUTLASS 或明确的性能 denominator；
8. calibration 和 holdout 分离；
9. 无条件上界违规；
10. run spec、环境、源码、binary、SASS、NCU 和结果 hash 完整。

当前本地环境不能访问 NVIDIA driver，因此新的 SM110 数值不能在此机器生成。
后续 Thor 运行必须使用稳定 campaign ID、逐 case `result.json`、run fingerprint、
持久日志、PID/status 和安全 resume；目录存在或任务已启动不算完成。

## 12. Microbenchmark 与完整 GEMM 来源

本节是本文参数和验证数据的来源附录。路径均相对仓库根目录。

### 12.0 新的全精度 compute campaign（等待 Thor 回传）

- runner：
  [`microbench/sm110_gemm_campaign/run_compute_campaign.py`](../../microbench/sm110_gemm_campaign/run_compute_campaign.py)
- detached/resume launcher：
  [`microbench/sm110_gemm_campaign/launch_compute_campaign.sh`](../../microbench/sm110_gemm_campaign/launch_compute_campaign.sh)
- fail-closed 回传审计：
  [`microbench/sm110_gemm_campaign/audit_campaign.py`](../../microbench/sm110_gemm_campaign/audit_campaign.py)
- Git 往返说明：
  [`microbench/sm110_gemm_campaign/README.md`](../../microbench/sm110_gemm_campaign/README.md)
- descriptor 一致性实现：
  [`scripts/sm110_gemm_model/tcgen05_descriptors.py`](../../scripts/sm110_gemm_model/tcgen05_descriptors.py)
- descriptor 与 shape 的规范来源：
  [NVIDIA PTX ISA 9.0, tcgen05 instruction descriptor](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#tcgen05-instruction-descriptor)

manifest 固定为 12 个计算合同、3 个 N shape、2 种 launch，共 72 个 case：FP16、
BF16、TF32、E4M3、E5M2、E3M2、E2M3、raw E2M1、MXFP4、NVFP4、S8、U8。
当前 campaign 还把硬件合同固定为 20 SM；full-SM case 回读每个 block 的 `%smid`，
只有 20 个 block 覆盖 20 个不同 SM 时，才允许用 `blocks × warps × iterations`
计算全 GPU issued work。
FP6/raw FP4 的 direct-SMEM campaign 把逻辑值放在 8-bit container 中，matrix
descriptor 沿用该物理布局的 8/4 byte-offset 编码；逻辑 `input_bits` 与物理
`descriptor_storage_bits` 在原始输出中分别记录。它不同于紧凑 packed 数据先经
`tcgen05.cp` 的 `b6x16_p32`/`b4x16_p64` 解压路径，后者必须作为另一类 schedule
和 microbenchmark 单独测量。
每个 case 至少 10 次 trial；compute window 用 PTX `%globaltimer` 的 nanosecond
计时；full-SM aggregate elapsed time 取所有 CTA 的最早 start 到最晚 stop，包含
CTA 启动偏斜，不用单 CTA 最大 duration 冒充整网格时长。另存 CUDA-event 的
host-observed 整 kernel 时间作交叉检查，避免把单点频率采样直接折算成吞吐。若
counter 权限可用，每种
精度另选一个 full-SM M128N256 case 保存 NCU 报告。runner 保存实际源码、idesc 字段、精确编译
命令、SASS、原始 stdout、environment、binary/SASS/source hash 和不可变 run spec。
本地 CUDA 13.0 `sm_110a` 静态门禁已经 72/72 通过；这只证明生成和 SASS 路径，
不替代 Thor 上的吞吐与数值验证。

### 12.1 Tensor Core compute-only

- 入口与生成器：
  [`microbench/mma_compute_only/run_thor_tcgen05_report.py`](../../microbench/mma_compute_only/run_thor_tcgen05_report.py)
- 使用说明：
  [`microbench/mma_compute_only/README.md`](../../microbench/mma_compute_only/README.md)
- 原始整理报告：
  [`microbench/mma_compute_only/分析报告.txt`](../../microbench/mma_compute_only/分析报告.txt)
- 结构化结果：
  [`microbench/mma_compute_only/plots/benchmark_results.csv`](../../microbench/mma_compute_only/plots/benchmark_results.csv)
- NCU 结构化结果：
  [`microbench/mma_compute_only/plots/ncu_results.csv`](../../microbench/mma_compute_only/plots/ncu_results.csv)
- SASS/NCU 入口：同目录生成的 `benchmark_src/`、`build/`、
  `run_ncu_reports.sh` 和 `ncu_reports/`。
- 当前只作历史快照的参数：BF16 258.030 TFLOP/s、E4M3 516.059 TFLOP/s；
  对应报告时钟为 1.575 GHz，尚不满足新的 10-trial artifact closure 合同。
- 历史 NVFP4 1032.111 TFLOP/s 因 descriptor type code 错位被隔离，不能进入
  经验包络；只有新的 Table 44 descriptor 重跑通过后才能替换。
- 当前边界：timed window 排除 TMA、copy pipeline、TMEM readback、epilogue 和
  launch；现有 FP4 case 是 `mxf4nvf4.block_scale.block16`，不能代表 raw E2M1
  或 MXFP4。

基本命令：

```bash
cd microbench/mma_compute_only
./build_and_run.sh run --iters 10000
./build_and_run.sh ncu
./build_and_run.sh plot
```

### 12.2 L2

- 源码：
  [`microbench/L2throughtput/demo.cu`](../../microbench/L2throughtput/demo.cu)
- 说明：
  [`microbench/L2throughtput/README.md`](../../microbench/L2throughtput/README.md)
- 原始结果：
  [`microbench/L2throughtput/results/l2_throughput.csv`](../../microbench/L2throughtput/results/l2_throughput.csv)
- SASS：
  [`microbench/L2throughtput/results/sass_summary.txt`](../../microbench/L2throughtput/results/sass_summary.txt)
- NCU：
  [`microbench/L2throughtput/results/ncu/ncu_l2_validation_summary.csv`](../../microbench/L2throughtput/results/ncu/ncu_l2_validation_summary.csv)
- 对抗式审查：
  [`microbench/L2throughtput/results/adversarial_review.md`](../../microbench/L2throughtput/results/adversarial_review.md)
- 已引用实测：unique read 946.701239 B/cycle/GPU；end-to-end store path
  299.372706 B/cycle/GPU。
- 已引用条件峰值：NCU model peak read 1024 B/cycle/GPU、write
  512 B/cycle/GPU。二者不是 measured sustained。

基本命令：

```bash
cd microbench/L2throughtput
./build_and_run.sh run
./build_and_run.sh validate
./build_and_run.sh ncu
```

### 12.3 GMEM/DRAM streaming

- 源码：
  [`microbench/05_gmem_dram_bandwidth/gmem_dram_bandwidth.cu`](../../microbench/05_gmem_dram_bandwidth/gmem_dram_bandwidth.cu)
- 说明：
  [`microbench/05_gmem_dram_bandwidth/README.md`](../../microbench/05_gmem_dram_bandwidth/README.md)
- 原始结果：
  [`microbench/05_gmem_dram_bandwidth/results/gmem_dram_bandwidth.csv`](../../microbench/05_gmem_dram_bandwidth/results/gmem_dram_bandwidth.csv)
- SASS：
  [`microbench/05_gmem_dram_bandwidth/results/sass_summary.txt`](../../microbench/05_gmem_dram_bandwidth/results/sass_summary.txt)
- NCU：
  [`microbench/05_gmem_dram_bandwidth/results/ncu/ncu_gmem_summary.csv`](../../microbench/05_gmem_dram_bandwidth/results/ncu/ncu_gmem_summary.csv)
- 对抗式审查：
  [`microbench/05_gmem_dram_bandwidth/results/adversarial_review.md`](../../microbench/05_gmem_dram_bandwidth/results/adversarial_review.md)
- 已引用实测：read-stream 126.010672 B/cycle/GPU；write-stream
  70.429363 B/cycle/GPU。
- 边界：本机 NCU 可能缺直接 `dram__bytes*`，验证会使用 LTS miss-sector proxy。

基本命令：

```bash
cd microbench/05_gmem_dram_bandwidth
./build_and_run.sh run
./build_and_run.sh validate
./build_and_run.sh ncu
```

### 12.4 TMA GMEM→SMEM

- 源码：
  [`microbench/07_tma_gmem_smem_bandwidth/tma_gmem_smem_bandwidth.cu`](../../microbench/07_tma_gmem_smem_bandwidth/tma_gmem_smem_bandwidth.cu)
- 说明：
  [`microbench/07_tma_gmem_smem_bandwidth/README.md`](../../microbench/07_tma_gmem_smem_bandwidth/README.md)
- 原始结果：
  [`microbench/07_tma_gmem_smem_bandwidth/results/tma_gmem_smem_bandwidth.csv`](../../microbench/07_tma_gmem_smem_bandwidth/results/tma_gmem_smem_bandwidth.csv)
- SASS：
  [`microbench/07_tma_gmem_smem_bandwidth/results/sass_summary.txt`](../../microbench/07_tma_gmem_smem_bandwidth/results/sass_summary.txt)
- NCU：
  [`microbench/07_tma_gmem_smem_bandwidth/results/ncu/ncu_tma_summary.csv`](../../microbench/07_tma_gmem_smem_bandwidth/results/ncu/ncu_tma_summary.csv)
- 对抗式审查：
  [`microbench/07_tma_gmem_smem_bandwidth/results/adversarial_review.md`](../../microbench/07_tma_gmem_smem_bandwidth/results/adversarial_review.md)
- 已引用实测：L2-hit 773.443437 B/cycle/GPU；DRAM-stream
  155.779224 B/cycle/GPU。
- 边界：这是包含 issue、completion、mbarrier 和 SMEM destination 的端到端
  TMA ingress，不是纯 DRAM 或纯 SMEM port peak。历史结果用各 CTA 最大
  `clock64()` span；新的整卡 closure rate 必须来自后述 unified component
  campaign 的 `globaltimer_gbytes_per_second`。

基本命令：

```bash
cd microbench/07_tma_gmem_smem_bandwidth
./build_and_run.sh run
./build_and_run.sh validate
./build_and_run.sh ncu
```

### 12.5 TMEM ingress、consume、readback 与联合 overlap

- `tcgen05.cp` 说明与结果：
  [`microbench/06_tmem_cp_bandwidth/README.md`](../../microbench/06_tmem_cp_bandwidth/README.md)、
  [`results/tmem_cp_only_summary.csv`](../../microbench/06_tmem_cp_bandwidth/results/tmem_cp_only_summary.csv)
- TMEM consume 说明与结果：
  [`microbench/08_tmem_consume_bandwidth/README.md`](../../microbench/08_tmem_consume_bandwidth/README.md)、
  [`results/tmem_consume_results.csv`](../../microbench/08_tmem_consume_bandwidth/results/tmem_consume_results.csv)
- CP/MMA overlap 说明与结果：
  [`microbench/11_pipeline_overlap/README.md`](../../microbench/11_pipeline_overlap/README.md)、
  [`results/pipeline_overlap_results.csv`](../../microbench/11_pipeline_overlap/results/pipeline_overlap_results.csv)
- accumulator readback 源码与说明：
  [`tmem_readback_bandwidth.cu`](../../microbench/12_tmem_readback_bandwidth/tmem_readback_bandwidth.cu)、
  [`README.md`](../../microbench/12_tmem_readback_bandwidth/README.md)
- unified component campaign、运行合同和独立审计：
  [`sm110_gemm_component_campaign`](../../microbench/sm110_gemm_component_campaign/README.md)
- 各目录的 `results/sass_summary*` 和 `results/ncu/*` 保存 SASS/NCU 证据。
- 当前 headline：`tcgen05.cp` ingress 859.024 B/cycle/GPU；TS MMA consume
  115.699 B/cycle/GPU；steady CP/MMA pipeline 约 89% component-overlap
  efficiency。
- 边界：这些是特定 TS/CP 数据路径的需求率或经验工作点，不是 raw TMEM bank
  read/write peak，也不是联合容量外边界。`08_tmem_consume_bandwidth` 测的是
  TS MMA 从 TMEM 消费 A operand，不是 GEMM 尾部的 accumulator readback；两者
  不能共享参数。新 readback microbenchmark 明确发出 `tcgen05.ld` 并以 SASS
  `LDTM.x8`/`LDTM.x16` 为静态锚点，但只有 Thor 的 10-trial/20-SM 审计通过后
  才能进入经验包络。

基本命令：

```bash
cd microbench/06_tmem_cp_bandwidth && ./build_and_run.sh summarize
cd ../08_tmem_consume_bandwidth && ./build_and_run.sh run
cd ../11_pipeline_overlap && ./build_and_run.sh run
bash ../../microbench/sm110_gemm_component_campaign/launch_component_campaign.sh <run-id>
```

### 12.6 SMEM、L1、DSMEM 与拓扑补充

- L1：
  [`microbench/04_l1_bandwidth`](../../microbench/04_l1_bandwidth/README.md)
- DSMEM：
  [`microbench/03_dsmem_bandwidth`](../../microbench/03_dsmem_bandwidth/README.md)
- DSMEM topology：
  [`microbench/09_dsmem_topology_contention`](../../microbench/09_dsmem_topology_contention/README.md)
- SMEM bank/stride：
  [`microbench/10_smem_bank_stride_bandwidth`](../../microbench/10_smem_bank_stride_bandwidth/README.md)

这些结果用于 schedule 合法性、bank conflict 和 CTA group/cluster 经验修正。由于
部分路径缺 direct byte counter，必须保留其 app clock、SASS、wavefront 或
miss-sector proxy 的证据边界。

### 12.7 完整 GEMM 已观测值

- 全精度实现/正确性 reference/同精度 denominator 覆盖合同：
  [`support_manifest.json`](../../microbench/sm110_full_gemm_campaign/support_manifest.json)
- 覆盖合同审计：
  [`audit_support_manifest.py`](../../microbench/sm110_full_gemm_campaign/audit_support_manifest.py)
- 首批 closure runner：
  [`run_full_gemm_campaign.py`](../../microbench/sm110_full_gemm_campaign/run_full_gemm_campaign.py)
- detached/resume launcher：
  [`launch_full_gemm_campaign.sh`](../../microbench/sm110_full_gemm_campaign/launch_full_gemm_campaign.sh)
- 独立结果审计：
  [`audit_campaign.py`](../../microbench/sm110_full_gemm_campaign/audit_campaign.py)
- Git 往返运行合同：
  [`README.md`](../../microbench/sm110_full_gemm_campaign/README.md)

- FP16→FP32 10-trial sweep：
  [`results/gemm_sm110/sm110_gemm_core_128_4096_10trials.csv`](../../results/gemm_sm110/sm110_gemm_core_128_4096_10trials.csv)
- 量化 1024 sweep：
  [`results/quant_gemm_sm110/sm110_quant_gemm_1024_sweep.csv`](../../results/quant_gemm_sm110/sm110_quant_gemm_1024_sweep.csv)
- FP16 runner：
  [`scripts/run_sm110_gemm_core_sweep.sh`](../../scripts/run_sm110_gemm_core_sweep.sh)
- 量化 runner：
  [`GEMMquant_sm110/scripts/run_quant_gemm_1024.py`](../../GEMMquant_sm110/scripts/run_quant_gemm_1024.py)
- FP16 主报告：
  [`GEMMsm110/SM110_GEMM_OPTIMIZATION_REPORT.md`](../../GEMMsm110/SM110_GEMM_OPTIMIZATION_REPORT.md)
- 量化主报告：
  [`GEMMquant_sm110/SM110_QUANT_GEMM_OPTIMIZATION_REPORT.md`](../../GEMMquant_sm110/SM110_QUANT_GEMM_OPTIMIZATION_REPORT.md)

当前工具只接收至少 10 个 trial 且全部 matched 的 backend series，再按最高
median 选择稳定最好实现。NVFP4/MXFP4 的性能 denominator 是 FP16 cuBLAS，
因此只保留绝对 GFLOP/s 和 correctness 状态，不使用历史 ratio 证明同精度胜负。
覆盖合同当前只有 FP16→FP32、E4M3→FP32 和 S8→S32 达到“可启动 closure
campaign”的实现条件；BF16、TF32、E5M2、两种 FP6、raw E2M1、U8 尚无完整
路径，MXFP4/NVFP4 则因输出合同、外部生成源码留存和跨精度 denominator 只能
标为 `partial`。这里的 `ready_for_closure_campaign` 仍不表示硬件闭环完成。

首批新 campaign 把可闭环的三种合同冻结为 9 个 square `NN` case：每种精度
`N=1024,2048,4096`，其中前两点是 calibration，4096 是预先保留的 holdout。
FP16 使用 `tc5b`（1024）和 `tc5a`（2048/4096），E4M3 使用 `q7`，S8 使用
`q15`；这些是仓库历史 sweep 中的强自研候选，而不是把库实现冒充自研 GEMM。
每个外层 case 运行 10 个独立 trial；每个 trial 内候选和同精度库 reference
使用同一输入，FP16 内层计时 100 次、E4M3/S8 内层计时 10 次。runner 独立从
`2N^3/time` 重算吞吐，S8 明确使用 OP/s。

数值证据包含两级门禁：先用实际量化后的输入在 CPU 上抽样 64 个输出，检查
cuBLAS/cuBLASLt reference；再把候选的完整输出矩阵与该 reference 比较。S8→S32
要求 bit-exact，浮点累加按显式 `atol`/`rtol` 合同判断。静态证据也不是二进制级
mnemonic 搜索：审计在被测 kernel 的 SASS 函数块内检查 `UTCHMMA`、FP8
`HMMA.16816.F32` 或 `IMMA.16816.S8.S8` 及 store。当前 CUDA 13.0 本地静态
门禁 9/9 通过；在 Thor 返回 90 个 trial、环境和可选 NCU artifact 之前，不把
它们升级为新的已观测值。

compute、component、full-GEMM 三批使用同一非阻塞 GPU 文件锁，因此 Thor 上
只能串行运行；这把“不要并发争抢 GPU”从操作约定升级为 runner 的机械约束。

### 12.8 硬件和软件环境来源

- compute-only 报告记录 Thor、20 SM 和 1.575 GHz：
  [`microbench/mma_compute_only/分析报告.txt`](../../microbench/mma_compute_only/分析报告.txt)
- 当前模型硬件快照：
  [`scripts/sm110_gemm_model/profiles/thor_sm110.json`](../../scripts/sm110_gemm_model/profiles/thor_sm110.json)
- 当前参数与逐项 locator：
  [`scripts/sm110_gemm_model/profiles/capacities.json`](../../scripts/sm110_gemm_model/profiles/capacities.json)

后续复测必须另外保存 GPU 名称、SM/compute capability、driver、CUDA、NVCC、
NCU、时钟、功耗模式、温度、Git commit、编译命令、binary hash、SASS hash 和
运行时间戳。当前仓库快照尚未为每项历史结果提供完整统一的环境 manifest，
这也是目标未完成的原因之一。
