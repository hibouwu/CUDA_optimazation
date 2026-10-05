# tc5a：Thor/SM110 候选方案模型

[建模入口](../README.md) · [统一接口](../model/interfaces.md) · [Thor 参数](../hardware/thor_sm110/README.md)

本文件保留历史候选方案，当前通用建模尚未选用 tc5a。

关联的旧 workload 为 `n2048_fp16_cold_calibration`，计算 FP16 输入、FP32 累加与输出的 D=AB；输入布局及计时边界的对接事项见下文。

代码入口：[`Tc5aRunner = Tc5OverlapRunner<128, 256, 64, 4>`](../../../../GEMMsm110/include/backends/tc5_persistent.cuh)，对应 kernel 为 `tc5a_overlap_epilogue_1sm_kernel`。

## 计算组织与数据流

| 参数 | 值 |
|---|---|
| 旧方案 ID | `tc5a_m128n256k64_stage4` |
| CTA tile | M128 × N256 × K64 |
| SMEM 输入缓冲级数 | 4 |
| TMEM 累加缓冲数 | 2，供 mainloop 与前一输出块的结果处理重叠 |
| 线程数 | 192/CTA |
| CTA group / Split-K | 1 / 1 |
| 调度 | persistent |
| TMEM 分配 | 512 列 |
| TMEM 读回 | x8，4 个 warp |

kernel 数据路径：A / 预打包 B → TMA → SMEM → `tcgen05.mma` → TMEM → `tcgen05.ld` → 寄存器 → global store → GMEM 中的 FP32 D。没有额外结果运算仍需计入累加器读回和输出存储。

参数来源：[旧方案配置](../../../../scripts/sm110_gemm_model/examples/schedules.json)。

## 输入布局与计时边界

workload 要求逻辑 B[K,N] 按行主序存储；`Tc5OverlapRunner` 接收按 N×K 行主序存储的 `b_nk`，满足 \(B_{\mathrm{nk}}[n,k]=B[k,n]\)。两者数学值相同，物理布局不同。

现有 [main.cu](../../../../GEMMsm110/src/main.cu) 在 host 上生成 `h_b_half_nk` 并复制到设备，然后才构造 runner 和计时。该 benchmark 的结果描述预打包输入下的 kernel 执行，尚不覆盖原 workload 从行主序 B 开始的完整任务。后续应在完整任务中计入布局转换，或另定义允许预打包输入的 workload；若 B 被多次复用，应显式给出复用次数及转换费用的摊销方式。

[benchmark_kernel](../../../../GEMMsm110/include/gemm_benchmark.cuh) 先预热，再对多次 launch 进行 CUDA event 计时并取平均，期间未逐次驱逐缓存。因此不能仅凭 workload 名称将该结果归为冷缓存，也不能未经测量断言所有输入均命中 L2。

本模型描述多级输入流水线、双累加缓冲与 persistent 工作分配如何共同决定执行时间。方案配置、复用规则和联合服务参数保存在本文件，不回写为通用硬件规则。

## 范围与模型输入

当前对应 FP16 输入、FP32 累加与输出、无尾块的 tc5a 主 kernel。B 已处于实现所需的预打包布局；原 workload 所需的布局转换另按[输入布局与计时边界](#输入布局与计时边界)处理。初始缓存状态是模型输入，不能由 workload 名称推定。

记问题实例为 \(w\)，方案配置为 \(s\)，通用层服务与约束为 \(\mathcal H_0,\mathcal H_1,\mathcal H_2,\mathcal H_3\)，仅在本方案条件下成立的联合服务参数为 \(\theta_{\mathrm{tc5a}}\)：

\[
\widehat T_{\mathrm{tc5a}}
=F_{\mathrm{tc5a}}(w,s;\mathcal H_0,\mathcal H_1,\mathcal H_2,\mathcal H_3,
\theta_{\mathrm{tc5a}}).
\]

这是模型的输入关系；完成时间函数尚未完成数值校准。改变 stage、tile 或任务分配时，要更新本方案的需求与事件关系，并检查联合服务参数是否仍适用。

## 引用的通用服务

| 来源 | 引用内容 | tc5a 在本文件中定义的内容 |
|---|---|---|
| [L0 模型](../model/L0.md) / [Thor 参数](../hardware/thor_sm110/L0.md) | 选定 MMA 的计算服务、依赖条件与取数需求 | 指令工作量、累加链及发出次序 |
| [L1 模型](../model/L1.md) / [Thor 参数](../hardware/thor_sm110/L1.md) | SMEM、TMEM、寄存器、驻留、片上访问与同步约束 | 输入槽与累加槽的分配、warp 分工、读回与缓冲复用 |
| [L2 模型](../model/L2.md) / [Thor 参数](../hardware/thor_sm110/L2.md) | 协作流水线接口、TMA、global store、缓存及外存服务 | 各阶段通路需求、访问次序、输入槽与累加槽的完成及释放事件 |
| [L3 模型](../model/L3.md) / [Thor 参数](../hardware/thor_sm110/L3.md) | 协作阶段组合、任务调度及多 SM 共享约束 | worker 工作分配、跨输出块重叠与整卡完成事件 |

引用独立服务时保留其测量条件。若组合执行需要额外校准，所得参数归入 \(\theta_{\mathrm{tc5a}}\)，不能作为所有方案通用的峰值。

## 工作量与资源占用

设输出 tile 为 \(B_M\times B_N\)，K 分块长度为 \(B_K\)，输入槽数为 \(S\)，输入元素字节数为 \(b\)。以下计账要求 M、N、K 分别整除对应 tile 尺寸；cleanup 路径尚未纳入。

\[
N_{\mathrm{out}}=\frac{M}{B_M}\frac{N}{B_N},\qquad
N_K=\frac{K}{B_K},\qquad
q_{\mathrm{in}}=(B_M+B_N)B_Kb.
\]

\[
Q_{\mathrm{TMA,payload}}=N_{\mathrm{out}}N_Kq_{\mathrm{in}},\qquad
S_{\mathrm{dynamic}}=S q_{\mathrm{in}}.
\]

动态 SMEM 之外，还需计入静态 barrier 状态、对齐及其他分配。TMEM 的两个累加缓冲单独计量，不能与输入槽数合并。资源合法性使用 L1 的分配和驻留约束。

对当前 2048³、tile 128×256×64、4 级输入缓冲的配置，MiB、KiB 均按二进制单位：

| 数量 | 推导 | 结果及含义 |
|---|---|---|
| 有效主乘加工作量 | \(2\times2048^3\) | 17,179,869,184 FLOP |
| 输出块数量 | \((2048/128)(2048/256)\) | 128 个输出块 |
| 每个输出块的 K 分块数 | \(2048/64\) | 32 步 |
| 全 kernel 的 K 分块处理次数 | \(128\times32\) | 4096 次 |
| 单步输入 payload | \((128+256)\times64\times2\) | 48 KiB，A 为 16 KiB、B 为 32 KiB |
| 动态 SMEM 输入缓冲 | \(4\times48\) KiB | 192 KiB，未计静态状态与分配对齐 |
| 该调度的输入 TMA payload 总量 | \(4096\times48\) KiB | 192 MiB，包含不同输出块对同一矩阵数据的重复请求 |
| A、B 各自的逻辑数据量 | \(2048^2\times2\) | 各 8 MiB |
| D 的逻辑写出量 | \(2048^2\times4\) | 16 MiB |

输入逻辑数据共 16 MiB，调度发出的输入 TMA payload 共 192 MiB，实际外存读流量尚未确定。缓存复用、请求粒度及竞争共同影响最后一个量；A、B、D 共 32 MiB 也不能推出全部命中 L2。

单个输出块沿整个 K 维的 A/B 数据共 \(32\times48=1536\) KiB，需要按分块序列供给。按 [Thor L1](../hardware/thor_sm110/L1.md#存储与缓存容量)记录的 228 KiB/SM 上限，192 KiB 动态 SMEM 已将该方案限制为至多 1 个 CTA/SM；实际寄存器及分配量仍需结合编译产物核实。

## 两组缓冲的事件约束

每个 worker 内，按执行顺序将所有输出块的 K 分块连续编号为 \(j\)，将输出块编号为 \(o\)。当前实现中，循环 SMEM 槽跨输出块持续推进；TMEM 使用两个累加缓冲。

| 对象 | 事件 | 当前实现对应关系 |
|---|---|---|
| 输入槽 | 搬运发出 \(a_j\)、A/B 就绪 \(r^A_j,r^B_j\) | TMA warp 获取槽、发出搬运，计算 warp 等待 `tma_barrier` |
| 输入消费 | 计算发出 \(c_j\)、输入槽释放 \(u_j\) | `issue_mma` 后提交 `mma_barrier`，生产者等待其完成通知后复用 |
| 累加缓冲 | 输出块计算完成 \(m_o\)、结果处理释放 \(v_o\) | `mainloop_barrier` 通知结果处理；参与的结果处理 warp 经 `epilogue_barrier` 交还缓冲 |

采用 \(S\) 个循环输入槽时，槽复用至少要求：

\[
a_j\ge u_{j-S}\quad(j\ge S).
\]

每个分块的计算发出至少要求：

\[
c_j\ge\max(r^A_j,r^B_j,t^{\mathrm{acc}}_j,t^{\mathrm{issue}}_j).
\]

其中 \(t^{\mathrm{acc}}_j\) 是累加依赖允许本步开始的时刻，\(t^{\mathrm{issue}}_j\) 是发射服务允许开始的时刻。输出块 \(o\) 的首次计算使用复用的 TMEM 槽时，还须满足：

\[
c_{o,0}\ge v_{o-2}\quad(o\ge2).
\]

\(c_{o,0}\) 表示该输出块首个 K 分块的计算发出时刻。输入槽释放、TMEM 槽释放和 GMEM 输出完成是不同事件；不能把指令发出当作异步工作完成。上述约束来自该实现的循环缓冲和 barrier 组织，不是所有 L1/L2 模型都必须采用的结构。

完成时间仍需结合服务延迟、启动间隔、共享带宽及各 warp 的执行次序求解；这些不等式本身不是完整预测公式。

## Persistent worker 与完成时间

设实际启动 worker 数为 \(P\)。源码按 `work_id = blockIdx.x + q * gridDim.x` 分配输出块，因此 worker \(p\) 的任务集合为：

\[
\mathcal O_p=\{p+qP\mid q\in\mathbb Z_{\ge0},\ p+qP<N_{\mathrm{out}}\}.
\]

\(P\) 取 runner 的实际选择，不能一般化为 SM 数。对当前 2048³、20 SM 且未设置 `TC5H_WORKERS` 覆盖值的情况，\(P=20\)：8 个 worker 各处理 7 块，其余 12 个各处理 6 块。

各 worker 内部的计算、前一块结果处理可以重叠；各 worker 之间共享缓存与外存服务。完整 kernel 时间由包含初始化、工作处理、最终同步与释放的完成事件决定，不能用“7 块 × 独立 CTA 延迟”代替。

## 尚待完成的时间推导

- 在以上事件图中接入适用的 L0–L3 服务参数，并补齐发射与完成事件的递推。
- 确定仅适用于当前方案的联合服务参数及其缓存、并发条件。
- 计算多个 worker 竞争、启动与排空后的 kernel 完成时间；需要布局转换时再按任务边界组合。

## 数值预测与验证状态

尚无数值预测结果。代入时需记录 workload、方案版本、硬件服务参数及来源、初始缓存条件，以及本方案所需的联合服务参数；布局与计时边界应与 workload 对齐。

本建模目录尚无该方案的实验记录。接入历史 benchmark 时，须按前文对齐 B 的预打包与预热条件；workload 中参考结果、误差指标和容差仍待确定。后续实验按 run 保存条件、结果和分析，关联所用方案及预测版本，再比较正确性、预测/实测时间、波动、相对误差与候选排序。
