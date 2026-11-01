# Lean 形式化建模架构（ARCH）

关联文档：[建模目标（GOAL）](GOAL.md) · [推进计划（PLAN）](PLAN.md) · [操作、资源与事件接口](../Docs/ModelEvaluation/gemm/model/common/interfaces.md) · [CUTLASS 程序图](../Docs/ModelEvaluation/gemm/model/mechanism/program/cutlass_program_graph.md)

## 1. 两层组织结构：通用带秩区间图 + 有限槽流水线实例

为支撑 [GOAL.md](GOAL.md) 中的条件性上下界 $[L, U]$ 与不相容判据，避免为特定事件个数重复编写递推引理，采用两层组织：

```text
第一层：通用事件区间约束图（General Interval Constraint Graph）
  ├── 输入：节点集 V、源点区间 [t0_min, t0_max]、带偏移区间 [d_min, d_max] 的入边列表；观察集合 O 独立指定，完成时间由时间戳导出
  ├── 通用性质：ValidLower（必要等待 ≥ d_min）与 ValidUpper（合取就绪推进 ≤ max + d_max）
  ├── 一般图引理（无需秩证书）：路径下界引理、正权环无解引理
  └── 带秩 DAG 引理（需 IsRankedDAG）：显式含源基例的极值递推 τ_low / τ_high、极值可达性、L ≤ T ≤ U、定值唯一性、不相容判据
                                  ↓ 实例化
第二层：有限槽 S 流水线模型（Slot Pipeline Instance）
  ├── 节点与边：每个工作项 g < n 的五事件（acq, rdy, commit, cmp, rel）与四类局部边
  ├── 槽安全性质（无需无环假设）：同槽跨代活跃区间不相交、任意时刻在途槽数 ≤ S
  ├── 结构无解反证（S = 1）：正权环导致不存在满足 ValidLower 的完整排程
  └── 无环实例化（S ≥ 2, n ≥ 0）：显式秩函数 rank（5g, 5g+1, 5g+2, 5g+3, 5g+8），直接复用第一层 DAG 定理导出 22 / 21 / 17
```

### 关于拓扑秩（Topological Rank）与程序图的说明

- 有限有向图无环当且仅当存在拓扑秩函数 $\mathrm{rank} : V \to \mathrm{Nat}$ 使得每条入边 $u \to v$ 满足 $\mathrm{rank}(u) < \mathrm{rank}(v)$。以 $\mathrm{rank}$ 作为无环证书，可直接利用 Lean 4 的自然数良基归纳定义极值递推，无需从“无环路”出发手写拓扑排序存在性证明。
- 注意：[`cfg_b_program.py`](../microbench/gh200_resource_campaign/access_rules/cfg_b_program.py#L195-L216) 在 `epilogue_defer` 等分支中存在前向引用（如 `j1.math_handoff -> j0.epilogue_permit`），因此**节点的创建序号不是拓扑秩**；源码通过 `TopologicalSorter(nodes).static_order()` 导出的 `topological_order` 才提供拓扑序（若后续导入，仍须逐边校验 $\mathrm{rank}(u) < \mathrm{rank}(v)$）。

---

## 2. 第一层：通用事件区间约束图规范

设 $V$ 为有限事件节点集合，允许为空。依赖结构 $P$ 给出节点与入边，参数 $H=[H_{\min},H_{\max}]$ 给出以下自然数区间：

- 对每个节点 $v \in V$，给定有限入边列表 $\mathrm{preds}(v)$；每条入边记为 $(u, d^{\min}_{u \to v}, d^{\max}_{u \to v})$，其中 $u\in V$ 且 $d^{\min}_{u \to v} \le d^{\max}_{u \to v}$；
- 对无入边的源节点（$\mathrm{preds}(v) = \varnothing$），给定源起点区间 $[t^{\min}_0(v), t^{\max}_0(v)]$，要求 $t^{\min}_0(v) \le t^{\max}_0(v)$。源起点区间也是 $H$ 的组成部分。

**轨迹只定义为时间戳赋值 $\tau : V \to \mathrm{Nat}$。** 独立选择观察集合 $O\subseteq V$，定义其完成时间：
\[
T_O(\tau) := \max\bigl(\{0\}\cup\{\tau(v)\mid v\in O\}\bigr).
\]
$T$ 不作为独立变量或合法性前提保存；下文固定 $O$ 时，$T$ 是 $T_O(\tau)$ 的简写。$O$ 可以为空，此时 $T_O(\tau)=0$。观察节点不必是无出边节点。若称 $T_O$ 为全图完成时间，需满足每个节点都可沿零条或多条边到达某个 $O$ 中的节点；在下界合法性下，这保证 $T_O(\tau)=\max(\{0\}\cup\{\tau(v)\mid v\in V\})$。

### 2.1 双侧合法性谓词与边权重语义

以下谓词只约束时间戳，均不依赖观察集合；省略参数时始终指同一组 $P,H$。

1. **下界合法性 $\mathrm{ValidLower}(P,H_{\min},\tau)$（适用于任意有向图）**：
   - 若 $\mathrm{preds}(v) = \varnothing$，则 $t^{\min}_0(v) \le \tau(v)$；
   - 对每条入边 $(u, d^{\min}, d^{\max}) \in \mathrm{preds}(v)$，有 $\tau(u) + d^{\min} \le \tau(v)$。
2. **上界合法性 $\mathrm{ValidUpper}(P,H_{\max},\tau)$（合取就绪后的有界延迟推进）**：
   - 若 $\mathrm{preds}(v) = \varnothing$，则 $\tau(v) \le t^{\max}_0(v)$；
   - 若 $\mathrm{preds}(v) \neq \varnothing$，则 $\tau(v) \le \max_{(u, d^{\min}, d^{\max}) \in \mathrm{preds}(v)} \big(\tau(u) + d^{\max}\big)$。
   > **参数语义区分**：在多前驱节点处，$d^{\max}_{u \to v}$ 表示合取就绪规则中的偏移约束，并不要求每条入边单独满足 $\tau(v) - \tau(u) \le d^{\max}_{u \to v}$（先到达的前驱还需等待其他前驱就绪）。因此，不能将实测单边事件时间差直接等同于 $d^{\max}_{u \to v}$。
3. **拓扑无环证书 $\mathrm{IsRankedDAG}(\mathrm{rank})$**：
   - $\mathrm{rank}:V\to\mathrm{Nat}$，对每条入边 $u\to v$ 满足 $\mathrm{rank}(u) < \mathrm{rank}(v)$。一般图与合法性定义不要求提供该证书，只有 DAG 递推定理需要。

### 2.2 第一层通用定理

1. **通用路径下界引理（无需秩证书）**：
   在满足 $\mathrm{ValidLower}(\tau)$ 的任意图上，任意从源节点 $s$（满足 $\mathrm{preds}(s)=\varnothing$）到终点 $v \in O$ 的有向路径，源起点下界 $t^{\min}_0(s)$ 加路径最小边权 $d^{\min}$ 之和均不超过 $\tau(v)$ 及 $T$。
2. **正权环无解引理（无需秩证书）**：
   若图中存在从某节点 $v$ 回到自身的非空有向环且其 $d^{\min}$ 权重之和 $> 0$，则 $\neg \exists \tau,\ \mathrm{ValidLower}(\tau)$。
3. **带秩 DAG 极值排程可达性、上下界与唯一性定理（需 $\mathrm{IsRankedDAG}(\mathrm{rank})$）**：
   沿 $\mathrm{rank}$ 良基递归定义极值下界排程 $\underline{\tau}$ 与极值上界排程 $\bar{\tau}$：
   \[
   \underline{\tau}(v) := \begin{cases}
   t^{\min}_0(v), & \mathrm{preds}(v) = \varnothing,\\
   \max_{(u, d^{\min}, d^{\max}) \in \mathrm{preds}(v)} \big(\underline{\tau}(u) + d^{\min}\big), & \mathrm{preds}(v) \neq \varnothing,
   \end{cases}
   \]
   \[
   \bar{\tau}(v) := \begin{cases}
   t^{\max}_0(v), & \mathrm{preds}(v) = \varnothing,\\
   \max_{(u, d^{\min}, d^{\max}) \in \mathrm{preds}(v)} \big(\bar{\tau}(u) + d^{\max}\big), & \mathrm{preds}(v) \neq \varnothing,
   \end{cases}
   \]
   对同一观察集合 $O$，令 $L(P,H_{\min},O) := T_O(\underline{\tau})$，$U(P,H_{\max},O) := T_O(\bar{\tau})$，下文简写为 $L,U$：
   - **极值排程可达性**：在上述区间有序条件下，$\underline{\tau}$ 与 $\bar{\tau}$ 均满足同一个原始区间模型的双侧规范：
     \[
     \forall \sigma\in\{\underline{\tau},\bar{\tau}\},\quad
     \mathrm{ValidLower}(P,H_{\min},\sigma)\land\mathrm{ValidUpper}(P,H_{\max},\sigma).
     \]
     同时有 $\forall v,\ \underline{\tau}(v)\le\bar{\tau}(v)$。因此 $L,U$ 是该模型对观察集合 $O$ 的可达到极值，无需把区间改成两个不同的定值模型；
   - **双侧夹逼**：对任意满足 $\mathrm{ValidLower}(\tau)$ 的轨迹有 $\forall v,\ \underline{\tau}(v) \le \tau(v)$ 及 $L \le T$；对任意满足 $\mathrm{ValidUpper}(\tau)$ 的轨迹有 $\forall v,\ \tau(v) \le \bar{\tau}(v)$ 及 $T \le U$；
   - **定值唯一性**：当 $t^{\min}_0 = t^{\max}_0$ 且所有边 $d^{\min} = d^{\max}$ 时，$\underline{\tau} = \bar{\tau}$，满足双侧规范的 $\tau$ 唯一且 $T_O(\tau) = L = U$。
4. **候选与观测区间不相容判据**：
   在满足 $\mathrm{IsRankedDAG}(\mathrm{rank})$ 的图上，固定观察集合 $O$，对任意与 $[L, U]$ 不相交的观测区间 $I_{\mathrm{obs}}$（即 $\forall t \in I_{\mathrm{obs}},\ t < L \lor t > U$）：
   \[
   \neg\exists \tau,\quad \mathrm{ValidLower}(P,H_{\min},\tau) \land \mathrm{ValidUpper}(P,H_{\max},\tau) \land T_O(\tau) \in I_{\mathrm{obs}}.
   \]

---

## 3. 第二层：有限槽 $S$ 流水线模型规范

将第一层实例化为从程序图提取了**槽复用**与**跨拍提交后释放**的抽象有限槽流水线。

### 3.1 参数域、事件节点与四类边

- **规模与延迟参数**：槽数 $S : \mathrm{Nat}$（全局前提 $S \ge 1$），工作项数 $n : \mathrm{Nat}$（允许 $n=0$）；流水化延迟区间参数 $d^{\min}_{\mathrm{issue}} \le d^{\max}_{\mathrm{issue}}$、$d^{\min}_{\mathrm{load}} \le d^{\max}_{\mathrm{load}}$、$d^{\min}_{\mathrm{mma}} \le d^{\max}_{\mathrm{mma}}$（其中 $d_{\mathrm{mma}}$ 表示提交后的剩余计算完成延迟）。
- **槽与代次**：$\mathrm{slot}(g) = g \bmod S,\ \mathrm{gen}(g) = \lfloor g / S \rfloor$。
- **事件节点与观察集合**：
  - $V_n=\{g\in\mathrm{Nat}\mid g<n\}\times\{\mathrm{acq},\mathrm{rdy},\mathrm{commit},\mathrm{cmp},\mathrm{rel}\}$，即每项五个事件；时间戳 $t_e(g)$ 是 $\tau(e(g))$ 的简写；
  - 当 $n\ge1$ 时，唯一无入边源节点为 $\mathrm{acq}(0)$，源起点区间设为 $[0,0]$；
  - $O_n=\{\mathrm{rel}(g)\mid g<n\}$。每项内部事件均沿阶段链到达 $\mathrm{rel}(g)$，因此该集合覆盖全部事件，流水线完成时间统一定义为 $T=T_{O_n}(\tau)$；
  - 当 $n=0$ 时，$V_n$、边集与 $O_n$ 均为空；唯一轨迹是空函数，双侧规范自然成立，完成时间与极值均为 $T=L=U=0$。这由同一定义导出，不另设空流水线模型。
- **四类有向边与权重区间**：
  1. **单项阶段链**（$g < n$）：
     - $\mathrm{acq}(g) \to \mathrm{rdy}(g)$，权重 $[d^{\min}_{\mathrm{load}}, d^{\max}_{\mathrm{load}}]$；
     - $\mathrm{rdy}(g) \to \mathrm{commit}(g)$，权重 $[0, 0]$；
     - $\mathrm{commit}(g) \to \mathrm{cmp}(g)$，权重 $[d^{\min}_{\mathrm{mma}}, d^{\max}_{\mathrm{mma}}]$；
     - $\mathrm{cmp}(g) \to \mathrm{rel}(g)$，权重 $[0, 0]$。
  2. **相邻发射间隔边**（$g + 1 < n$）：$\mathrm{acq}(g) \to \mathrm{acq}(g + 1)$，权重 $[d^{\min}_{\mathrm{issue}}, d^{\max}_{\mathrm{issue}}]$。
  3. **同槽跨代复用边**（$g + S < n$）：$\mathrm{rel}(g) \to \mathrm{acq}(g + S)$，权重 $[0, 0]$。
  4. **跨拍提交后释放边**（$g + 1 < n$，提取自 [cutlass_program_graph.md §3](../Docs/ModelEvaluation/gemm/model/mechanism/program/cutlass_program_graph.md#L77) 中 `wait<1>` 释放先后）：$\mathrm{commit}(g + 1) \to \mathrm{rel}(g)$，权重 $[0, 0]$。

### 3.2 第二层核心定理与 $S=1 / 4 / 6$ 实例

1. **槽生命周期安全与在途容量上界（仅依赖 $\mathrm{ValidLower}$，对任意 $S \ge 1, n \in \mathrm{Nat}$ 成立）**
   - 每项满足最小占用时长 $t_{\mathrm{acq}}(g) + (d^{\min}_{\mathrm{load}} + d^{\min}_{\mathrm{mma}}) \le t_{\mathrm{rel}}(g)$；
   - 对任意 $g_1 < g_2 < n$，若 $g_1 \bmod S = g_2 \bmod S$，则 $t_{\mathrm{cmp}}(g_1) \le t_{\mathrm{rel}}(g_1) \le t_{\mathrm{acq}}(g_2)$，活跃区间 $[t_{\mathrm{acq}}(g_1), t_{\mathrm{rel}}(g_1))$ 与 $[t_{\mathrm{acq}}(g_2), t_{\mathrm{rel}}(g_2))$ 严格不相交；
   - 任意时刻 $t$，同时活跃的工作项数满足 $|\{g < n \mid t_{\mathrm{acq}}(g) \le t < t_{\mathrm{rel}}(g)\}| \le S$；
   - 对 $S$ 个模同余类的复用链求和，导出平均槽占用下界 $n(d^{\min}_{\mathrm{load}} + d^{\min}_{\mathrm{mma}}) \le S \cdot T$。

2. **$S = 1$ 的正权环无解定理**
   当 $S = 1,\ n \ge 2,\ d^{\min}_{\mathrm{load}} > 0$ 时，路径 $\mathrm{rel}(0) \xrightarrow{+0} \mathrm{acq}(1) \xrightarrow{+d^{\min}_{\mathrm{load}}} \mathrm{rdy}(1) \xrightarrow{+0} \mathrm{commit}(1) \xrightarrow{+0} \mathrm{rel}(0)$ 构成权重为 $d^{\min}_{\mathrm{load}} > 0$ 的有向环，由第一层正权环无解引理直接证明不存在满足 $\mathrm{ValidLower}$ 的完整排程。

3. **$S \ge 2,\ n \in\mathrm{Nat}$ 的显式拓扑秩定理（连接第一层通用 DAG）**
   当 $S \ge 2$ 时，定义显式拓扑秩函数 $\mathrm{rank}$：
   \[
   \begin{aligned}
   \mathrm{rank}(\mathrm{acq}(g)) &= 5g,&
   \mathrm{rank}(\mathrm{rdy}(g)) &= 5g + 1,&
   \mathrm{rank}(\mathrm{commit}(g)) &= 5g + 2,\\
   \mathrm{rank}(\mathrm{cmp}(g)) &= 5g + 3,&
   \mathrm{rank}(\mathrm{rel}(g)) &= 5g + 8.
   \end{aligned}
   \]
   四类有向边在 $S \ge 2$ 下全部严格递增：
   - 阶段边：$5g < 5g+1 < 5g+2 < 5g+3 < 5g+8$；
   - 发射边：$5g < 5(g+1)$；
   - 跨拍边：$\mathrm{rank}(\mathrm{commit}(g+1)) = 5g + 7 < 5g + 8 = \mathrm{rank}(\mathrm{rel}(g))$；
   - 同槽复用边：当 $S \ge 2$ 时，$\mathrm{rank}(\mathrm{rel}(g)) = 5g + 8 < 5g + 10 \le 5(g + S) = \mathrm{rank}(\mathrm{acq}(g+S))$。
   因此，对任意 $S \ge 2,\ n\in\mathrm{Nat}$，流水线图满足第一层的 $\mathrm{IsRankedDAG}(\mathrm{rank})$，直接继承极值上下界 $L \le T \le U$、定值唯一性与观测不相容判据。

4. **变槽数与变控制边的精确上下界实例**（取定值参数 $n = 12,\ d_{\mathrm{issue}} = 1,\ d_{\mathrm{load}} = 4,\ d_{\mathrm{mma}} = 2$，$d^{\min} = d^{\max} = d$）：
   - **$S = 4$（含跨拍边）的紧上下界 $L = U = 22$**（去掉跨拍边时 $L = U = 21$）：
     - 下界 $T \ge 22$ 由路径 $\mathrm{acq}(0) \xrightarrow{+6} \mathrm{rel}(0) \xrightarrow{+0} \mathrm{acq}(4) \xrightarrow{+4} \mathrm{commit}(4) \xrightarrow{+0} \mathrm{rel}(3) \xrightarrow{+0} \mathrm{acq}(7) \xrightarrow{+6} \mathrm{rel}(7) \xrightarrow{+0} \mathrm{acq}(11) \xrightarrow{+6} \mathrm{rel}(11)$ 给出；
     - 极值递推轨迹：
       \[
       \begin{aligned}
       t_{\mathrm{acq}} &= [0, 1, 2, 3, 6, 7, 8, 10, 12, 13, 14, 16],\\
       t_{\mathrm{rdy}}(g) &= t_{\mathrm{commit}}(g) = t_{\mathrm{acq}}(g) + 4,\\
       t_{\mathrm{cmp}}(g) &= t_{\mathrm{acq}}(g) + 6,\\
       t_{\mathrm{rel}}(g) &= \begin{cases} \max(t_{\mathrm{cmp}}(g),\ t_{\mathrm{commit}}(g+1)), & g < 11, \\ t_{\mathrm{cmp}}(11), & g = 11 \end{cases}
       = [6, 7, 8, 10, 12, 13, 14, 16, 18, 19, 20, 22]_g,
       \end{aligned}
       \]
       同时满足 $\mathrm{ValidLower}$ 与 $\mathrm{ValidUpper}$，唯一确定 $T = 22$。
   - **$S = 6$（含跨拍边）的紧上下界 $L = U = 17$**：
     - 下界 $T \ge 17$ 由路径 $\mathrm{acq}(0) \xrightarrow{+11} \mathrm{acq}(11) \xrightarrow{+6} \mathrm{rel}(11)$ 给出；
     - 极值递推轨迹 $t_{\mathrm{acq}}(g) = g,\ t_{\mathrm{rdy}}(g) = t_{\mathrm{commit}}(g) = g+4,\ t_{\mathrm{cmp}}(g) = t_{\mathrm{rel}}(g) = g+6$ 同时满足双侧约束，唯一确定 $T = 17$。在本组参数及四类约束下，加入 `commit(g+1) ≤ rel(g)` 不增加 $S = 6$ 的最优完成时间。

## 4. 明确省略的内容、上下界转移方向与适用边界

- **省略内容与动态服务差异**：
  - 第二层首轮实例未包含 [`cfg_b_program.py`](../microbench/gh200_resource_campaign/access_rules/cfg_b_program.py#L165-L179) 中消费者释放旧槽后才进入后续项查询的反向控制边（`rel(g-1) -> full_query(g+1)`）、`wait<1>` 本身的指令执行开销及尾部排空细节。
  - 静态程序依赖图本身不含动态排队、在途请求分摊（如 [`cfg_b_input_network.py`](../microbench/gh200_resource_campaign/access_rules/cfg_b_input_network.py#L42) 中的 `ReplyGroup`）和共享资源日历竞争。将通用区间 DAG 定理用于具体性能模型前，必须先将这些动态服务关系归约为显式顺序边或有依据的有效偏移区间。
- **对下界与上界的不同影响**（以下增删边比较固定节点集、观察集合、源节点及其起点区间，并保留共有边的偏移区间；若删边产生新源节点，须重新核验其起点约束，不能直接套用单调关系）：
  - 省略程序边会放松 $\mathrm{ValidLower}$ 约束，因此子图下界 $L_{\mathrm{sub}}$ 仍是包含更多边的完整图的合法下界（$L_{\mathrm{sub}} \le L_{\mathrm{full}}$）；
  - 省略前驱边或将未知控制等待设为 0 会收紧 $\mathrm{ValidUpper}$，可能导致子图上界 $U_{\mathrm{sub}} \le U_{\mathrm{full}}$。因此，当 $T_{\mathrm{obs}} > U_{\mathrm{sub}}$ 时，定理仅证伪“当前声明的子图边 $P_{\mathrm{sub}}$ 与偏移区间 $H_{\max}$ 的组合与观测相容”，其用途是发现不相容并限制后续解释，不能单独区分原因是漏边、区间偏小还是测量窗口差异。
