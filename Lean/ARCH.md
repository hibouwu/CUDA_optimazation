# Lean 形式化建模架构（ARCH）

关联文档：[建模目标（GOAL）](GOAL.md) · [推进计划（PLAN）](PLAN.md) · [操作、资源与事件接口](../Docs/ModelEvaluation/gemm/model/common/interfaces.md) · [CUTLASS 程序图](../Docs/ModelEvaluation/gemm/model/mechanism/program/cutlass_program_graph.md)

## 1. 两层组织结构：通用带秩区间图 + 有限槽流水线实例

为支撑 [GOAL.md](GOAL.md) 中的条件性上下界 $[L, U]$ 与不相容判据，避免为特定事件个数重复编写递推引理，采用两层组织：

```text
第一层：通用事件区间约束图（General Interval Constraint Graph）
  ├── 输入：节点集 V、源点区间 [t0_min, t0_max]、带偏移区间 [d_min, d_max] 的入边列表
  ├── 轨迹与完成时间：轨迹为 τ : V → Nat；观察完成时间为派生量 T_O(τ) := max({0} ∪ {τ(v) | v ∈ O})
  ├── 局部规范：ValidLower(τ)（必要等待 ≥ d_min）与 ValidUpper(τ)（合取就绪推进 ≤ max + d_max）
  ├── 一般图引理（无需秩证书）：路径下界引理、正权环无解引理
  └── 带秩 DAG 引理（需 IsRankedDAG）：极值递推 τ_low / τ_high 同属双侧规范、L ≤ T_O(τ) ≤ U、定值唯一性、不相容判据
                                  ↓ 实例化
第二层：有限槽 S 流水线模型（Slot Pipeline Instance）
  ├── 节点与终点集：V_n = {g < n} × {acq, rdy, commit, cmp, rel}，观察终点集 O_n = {rel(g) | g < n}
  ├── 槽安全性质（S ≥ 1, n ≥ 0）：同槽跨代活跃区间不相交、任意时刻在途槽数 ≤ S
  ├── 结构无解反证（S = 1, n ≥ 2, d_load_min > 0）：正权环导致不存在满足 ValidLower 的轨迹
  └── 无环实例化（S ≥ 2, n ≥ 0）：显式秩函数 rank（5g, 5g+1, 5g+2, 5g+3, 5g+8），直接复用第一层定理导出 22 / 21 / 17
```

### 关于拓扑秩（Topological Rank）与程序图的说明

- 有限有向图无环当且仅当存在拓扑秩函数 $\mathrm{rank} : V \to \mathrm{Nat}$ 使得每条入边 $u \to v$ 满足 $\mathrm{rank}(u) < \mathrm{rank}(v)$。以 $\mathrm{rank}$ 作为无环证书，可直接利用 Lean 4 的自然数良基归纳定义极值递推，无需从“无环路”出发手写拓扑排序存在性证明。
- 注意：[`cfg_b_program.py`](../microbench/gh200_resource_campaign/access_rules/cfg_b_program.py#L195-L216) 在 `epilogue_defer` 等分支中存在前向引用（如 `j1.math_handoff -> j0.epilogue_permit`），因此**节点的创建序号不是拓扑秩**；源码通过 `TopologicalSorter(nodes).static_order()` 导出的 `topological_order` 才提供拓扑序（若后续导入，仍须逐边校验 $\mathrm{rank}(u) < \mathrm{rank}(v)$）。

---

## 2. 第一层：通用事件区间约束图规范

设 $V$ 为有限事件节点集合：
- 对每个节点 $v \in V$，给定入边列表 $\mathrm{preds}(v)$，每条入边记为 $(u, d^{\min}_{u \to v}, d^{\max}_{u \to v})$，要求权重区间良定义：$d^{\min}_{u \to v} \le d^{\max}_{u \to v}$；
- 对无入边的源节点（$\mathrm{preds}(v) = \varnothing$），给定源起点区间 $[t^{\min}_0(v), t^{\max}_0(v)]$，要求 $t^{\min}_0(v) \le t^{\max}_0(v)$。

轨迹为时间戳赋值 $\tau : V \to \mathrm{Nat}$。给定关心的观察终点集 $O \subseteq V$，观察完成时间定义为 $\tau$ 的派生函数（不作为独立自由变量）：

\[
T_O(\tau) := \max\bigl(\{0\} \cup \{\tau(v) \mid v \in O\}\bigr).
\]

当 $O = \varnothing$ 时，$T_O(\tau) = 0$。若 $O$ 仅选取部分节点，则 $T_O(\tau)$ 表示该组观察端点的完成时间；若要代表全图完成时间，需另行满足图中所有节点均可到达 $O$。

### 2.1 双侧合法性谓词与边权重语义

1. **下界合法性 $\mathrm{ValidLower}_{H_{\min}}(\tau)$（适用于任意有向图，不要求提供秩证书）**：
   - 若 $\mathrm{preds}(v) = \varnothing$，则 $t^{\min}_0(v) \le \tau(v)$；
   - 对每条入边 $(u, d^{\min}_{u \to v}, d^{\max}_{u \to v}) \in \mathrm{preds}(v)$，有 $\tau(u) + d^{\min}_{u \to v} \le \tau(v)$。
2. **上界合法性 $\mathrm{ValidUpper}_{H_{\max}}(\tau)$（合取就绪后的有界延迟推进）**：
   - 若 $\mathrm{preds}(v) = \varnothing$，则 $\tau(v) \le t^{\max}_0(v)$；
   - 若 $\mathrm{preds}(v) \neq \varnothing$，则 $\tau(v) \le \max_{(u, d^{\min}, d^{\max}) \in \mathrm{preds}(v)} \big(\tau(u) + d^{\max}\big)$。
   > **参数语义区分**：在多前驱节点处，$d^{\max}_{u \to v}$ 表示**合取就绪规则中的偏移约束**，并不要求每条入边单独满足 $\tau(v) - \tau(u) \le d^{\max}_{u \to v}$（先到达的前驱还需等待其他前驱就绪）。因此，不能将实测单边事件时间差 $\tau(v) - \tau(u)$ 直接等同于 $d^{\max}_{u \to v}$。
3. **拓扑无环证书 $\mathrm{IsRankedDAG}(\mathrm{rank})$**：
   - 对所有 $v \in V$ 及 $(u, d^{\min}, d^{\max}) \in \mathrm{preds}(v)$，满足 $\mathrm{rank}(u) < \mathrm{rank}(v)$。

### 2.2 第一层通用定理

1. **通用路径下界引理（无需秩证书）**：
   在满足 $\mathrm{ValidLower}_{H_{\min}}(\tau)$ 的任意图上，任意从源节点 $s$（$\mathrm{preds}(s)=\varnothing$）到节点 $v \in O$ 的有向路径，源起点下界 $t^{\min}_0(s)$ 加路径最小边权 $d^{\min}$ 之和均不超过 $\tau(v)$ 及 $T_O(\tau)$。
2. **正权环无解引理（无需秩证书）**：
   若图中存在从某节点 $v$ 回到自身的非空有向环且其 $d^{\min}$ 权重之和 $> 0$，则 $\neg \exists \tau,\ \mathrm{ValidLower}_{H_{\min}}(\tau)$。
3. **带秩 DAG 极值排程可达性、上下界与唯一性定理（需 $\mathrm{IsRankedDAG}(\mathrm{rank})$）**：
   沿 $\mathrm{rank}$ 良基递归定义下极值排程 $\underline{\tau}$ 与上极值排程 $\bar{\tau}$：
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
   并由同一完成时间函数定义 $L := T_O(\underline{\tau})$，$U := T_O(\bar{\tau})$：
   - **同一区间模型内的极值可达性**：当所有区间满足 $t^{\min}_0 \le t^{\max}_0$ 且 $d^{\min} \le d^{\max}$ 时，$\underline{\tau}$ 与 $\bar{\tau}$ 均同时满足同一区间模型的双侧规范：
     \[
     \mathrm{ValidLower}_{H_{\min}}(\underline{\tau}) \land \mathrm{ValidUpper}_{H_{\max}}(\underline{\tau}),\qquad
     \mathrm{ValidLower}_{H_{\min}}(\bar{\tau}) \land \mathrm{ValidUpper}_{H_{\max}}(\bar{\tau}),
     \]
     且 $\forall v \in V,\ \underline{\tau}(v) \le \bar{\tau}(v)$；
   - **双侧夹逼**：对任意满足 $\mathrm{ValidLower}_{H_{\min}}(\tau)$ 的轨迹有 $\forall v,\ \underline{\tau}(v) \le \tau(v)$ 及 $L \le T_O(\tau)$；对任意满足 $\mathrm{ValidUpper}_{H_{\max}}(\tau)$ 的轨迹有 $\forall v,\ \tau(v) \le \bar{\tau}(v)$ 及 $T_O(\tau) \le U$；
   - **定值唯一性**：当 $t^{\min}_0 = t^{\max}_0$ 且所有边 $d^{\min} = d^{\max}$ 时，$\underline{\tau} = \bar{\tau}$，满足双侧规范的轨迹 $\tau$ 唯一且 $T_O(\tau) = L = U$。
4. **候选与观测区间不相容判据**：
   在满足 $\mathrm{IsRankedDAG}(\mathrm{rank})$ 的图上，对任意与 $[L, U]$ 不相交的观测区间 $I_{\mathrm{obs}}$（即 $\forall t \in I_{\mathrm{obs}},\ t < L \lor t > U$）：
   \[
   \neg\exists \tau,\quad \mathrm{ValidLower}_{H_{\min}}(\tau) \land \mathrm{ValidUpper}_{H_{\max}}(\tau) \land T_O(\tau) \in I_{\mathrm{obs}}.
   \]

---

## 3. 第二层：有限槽 $S$ 流水线模型规范

将第一层实例化为从程序图提取了**槽复用**与**跨拍提交后释放**的抽象有限槽流水线。

### 3.1 参数域、事件节点与四类边

- **规模与延迟参数**：槽数 $S : \mathrm{Nat}$（全局前提 $S \ge 1$），工作项数 $n : \mathrm{Nat}$（包含 $n = 0$ 的空工作负载）；流水化延迟区间参数 $d^{\min}_{\mathrm{issue}} \le d^{\max}_{\mathrm{issue}}$、$d^{\min}_{\mathrm{load}} \le d^{\max}_{\mathrm{load}}$、$d^{\min}_{\mathrm{mma}} \le d^{\max}_{\mathrm{mma}}$（其中 $d_{\mathrm{mma}}$ 表示提交后的剩余计算完成延迟）。
- **槽与代次**：$\mathrm{slot}(g) = g \bmod S,\ \mathrm{gen}(g) = \lfloor g / S \rfloor$。
- **事件节点与观察终点集（对任意 $n \ge 0$ 统一定义）**：
  \[
  V_n := \{g \mid g < n\} \times \{\mathrm{acq}, \mathrm{rdy}, \mathrm{commit}, \mathrm{cmp}, \mathrm{rel}\},\qquad
  O_n := \{\mathrm{rel}(g) \mid g < n\}.
  \]
  当 $n = 0$ 时，$V_0 = \varnothing, O_0 = \varnothing$，$T_{O_0}(\tau) = 0$，$L = U = 0$；当 $n \ge 1$ 时，$\mathrm{acq}(0)$ 为唯一无入边源节点（源起点区间 $[0, 0]$），且每项内部事件均沿阶段链到达 $\mathrm{rel}(g) \in O_n$。
- **四类有向边与权重区间**：
  1. **单项阶段链**（$g < n$）：
     - $\mathrm{acq}(g) \to \mathrm{rdy}(g)$，权重 $[d^{\min}_{\mathrm{load}}, d^{\max}_{\mathrm{load}}]$；
     - $\mathrm{rdy}(g) \to \mathrm{commit}(g)$，权重 $[0, 0]$；
     - $\mathrm{commit}(g) \to \mathrm{cmp}(g)$，权重 $[d^{\min}_{\mathrm{mma}}, d^{\max}_{\mathrm{mma}}]$；
     - $\mathrm{cmp}(g) \to \mathrm{rel}(g)$，权重 $[0, 0]$。
  2. **相邻发射间隔边**（$g + 1 < n$）：$\mathrm{acq}(g) \to \mathrm{acq}(g + 1)$，权重 $[d^{\min}_{\mathrm{issue}}, d^{\max}_{\mathrm{issue}}]$。
  3. **同槽跨代复用边**（$g + S < n$）：$\mathrm{rel}(g) \to \mathrm{acq}(g + S)$，权重 $[0, 0]$。
  4. **跨拍提交后释放边**（$g + 1 < n$，提取自 [cutlass_program_graph.md §3](../Docs/ModelEvaluation/gemm/model/mechanism/program/cutlass_program_graph.md#L77) 中 `wait<1>` 释放先后）：$\mathrm{commit}(g + 1) \to \mathrm{rel}(g)$，权重 $[0, 0]$。

### 3.2 各条结论的精确前提与 $S=1 / 4 / 6$ 实例

| 结论 | 所需前提 |
|---|---|
| 同槽跨代不覆盖、任意时刻在途槽数 $\le S$、平均槽占用下界 | $S \ge 1,\ n \ge 0,\ \mathrm{ValidLower}_{H_{\min}}(\tau)$ |
| 显式拓扑秩严格单调（$\mathrm{IsRankedDAG}(\mathrm{rank})$）及极值上下界可达 | $S \ge 2,\ n \ge 0$ |
| $\mathrm{acq}(0)$ 为唯一无入边源节点 | $n \ge 1$ |
| $S = 1$ 因正权环不存在满足 $\mathrm{ValidLower}_{H_{\min}}$ 的轨迹 | $S = 1,\ n \ge 2,\ d^{\min}_{\mathrm{load}} > 0$ |
| $S = 4$（$22 / 21$）与 $S = 6$（$17$）紧上下界实例 | $n = 12,\ d_{\mathrm{issue}} = 1,\ d_{\mathrm{load}} = 4,\ d_{\mathrm{mma}} = 2$ |

1. **槽生命周期安全与在途容量上界（$S \ge 1, n \ge 0$，仅依赖 $\mathrm{ValidLower}_{H_{\min}}$）**
   - 对任意 $g < n$，满足最小占用时长 $t_{\mathrm{acq}}(g) + (d^{\min}_{\mathrm{load}} + d^{\min}_{\mathrm{mma}}) \le t_{\mathrm{rel}}(g)$；
   - 对任意 $g_1 < g_2 < n$，若 $g_1 \bmod S = g_2 \bmod S$，则 $t_{\mathrm{cmp}}(g_1) \le t_{\mathrm{rel}}(g_1) \le t_{\mathrm{acq}}(g_2)$，活跃区间 $[t_{\mathrm{acq}}(g_1), t_{\mathrm{rel}}(g_1))$ 与 $[t_{\mathrm{acq}}(g_2), t_{\mathrm{rel}}(g_2))$ 严格不相交；
   - 任意时刻 $t$，同时活跃的工作项数满足 $|\{g < n \mid t_{\mathrm{acq}}(g) \le t < t_{\mathrm{rel}}(g)\}| \le S$；
   - 对 $S$ 个模同余类的复用链求和，导出平均槽占用下界 $n(d^{\min}_{\mathrm{load}} + d^{\min}_{\mathrm{mma}}) \le S \cdot T_{O_n}(\tau)$。

2. **$S = 1$ 的正权环无解定理（$S = 1, n \ge 2, d^{\min}_{\mathrm{load}} > 0$）**
   路径 $\mathrm{rel}(0) \xrightarrow{+0} \mathrm{acq}(1) \xrightarrow{+d^{\min}_{\mathrm{load}}} \mathrm{rdy}(1) \xrightarrow{+0} \mathrm{commit}(1) \xrightarrow{+0} \mathrm{rel}(0)$ 构成权重为 $d^{\min}_{\mathrm{load}} > 0$ 的有向环，由第一层正权环无解引理直接证明 $\neg \exists \tau,\ \mathrm{ValidLower}_{H_{\min}}(\tau)$。

3. **$S \ge 2, n \ge 0$ 的显式拓扑秩定理（连接第一层通用 DAG）**
   定义显式拓扑秩函数 $\mathrm{rank}$：
   \[
   \begin{aligned}
   \mathrm{rank}(\mathrm{acq}(g)) &= 5g,&
   \mathrm{rank}(\mathrm{rdy}(g)) &= 5g + 1,&
   \mathrm{rank}(\mathrm{commit}(g)) &= 5g + 2,\\
   \mathrm{rank}(\mathrm{cmp}(g)) &= 5g + 3,&
   \mathrm{rank}(\mathrm{rel}(g)) &= 5g + 8.
   \end{aligned}
   \]
   当 $S \ge 2$ 时，四类有向边全部满足 $\mathrm{rank}(u) < \mathrm{rank}(v)$，故流水线图满足 $\mathrm{IsRankedDAG}(\mathrm{rank})$，直接继承第一层极值可达性、上下界夹逼 $L \le T_{O_n}(\tau) \le U$、定值唯一性与观测不相容判据。

4. **变槽数与变控制边的精确上下界实例**（取定值参数 $n = 12,\ d_{\mathrm{issue}} = 1,\ d_{\mathrm{load}} = 4,\ d_{\mathrm{mma}} = 2$，$d^{\min} = d^{\max} = d$）：
   - **$S = 4$（含跨拍边）的紧上下界 $L = U = 22$**（去掉跨拍边时 $L = U = 21$）：
     - 下界 $T_{O_{12}}(\tau) \ge 22$ 由路径 $\mathrm{acq}(0) \xrightarrow{+6} \mathrm{rel}(0) \xrightarrow{+0} \mathrm{acq}(4) \xrightarrow{+4} \mathrm{commit}(4) \xrightarrow{+0} \mathrm{rel}(3) \xrightarrow{+0} \mathrm{acq}(7) \xrightarrow{+6} \mathrm{rel}(7) \xrightarrow{+0} \mathrm{acq}(11) \xrightarrow{+6} \mathrm{rel}(11)$ 给出；
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
       同时满足 $\mathrm{ValidLower} \land \mathrm{ValidUpper}$，唯一确定 $T_{O_{12}}(\tau) = 22$。
   - **$S = 6$（含跨拍边）的紧上下界 $L = U = 17$**：
     - 下界 $T_{O_{12}}(\tau) \ge 17$ 由路径 $\mathrm{acq}(0) \xrightarrow{+11} \mathrm{acq}(11) \xrightarrow{+6} \mathrm{rel}(11)$ 给出；
     - 极值递推轨迹 $t_{\mathrm{acq}}(g) = g,\ t_{\mathrm{rdy}}(g) = t_{\mathrm{commit}}(g) = g+4,\ t_{\mathrm{cmp}}(g) = t_{\mathrm{rel}}(g) = g+6$ 同时满足双侧约束，唯一确定 $T_{O_{12}}(\tau) = 17$。在本组参数及四类约束下，加入 `commit(g+1) ≤ rel(g)` 不增加 $S = 6$ 的最优完成时间。

## 4. 明确省略的内容、上下界转移方向与适用边界

- **省略内容与动态服务差异**：
  - 第二层首轮实例未包含 [`cfg_b_program.py`](../microbench/gh200_resource_campaign/access_rules/cfg_b_program.py#L165-L179) 中消费者释放旧槽后才进入后续项查询的反向控制边（`rel(g-1) -> full_query(g+1)`）、`wait<1>` 本身的指令执行开销及尾部排空细节。
  - 静态程序依赖图本身不含动态排队、在途请求分摊（如 [`cfg_b_input_network.py`](../microbench/gh200_resource_campaign/access_rules/cfg_b_input_network.py#L42) 中的 `ReplyGroup`）和共享资源日历竞争。将通用区间 DAG 定理用于具体性能模型前，必须先将这些动态服务关系归约为显式顺序边或有依据的有效偏移区间。
- **对下界与上界的不同影响**：
  - 以下增删边比较均假定固定节点集、观察集合 $O$、源节点集合（$\mathrm{preds}(v)=\varnothing$）及其源起点区间，并保留共有边的偏移区间；若删边使某节点失去最后一条入边而成为新源节点，须重新核验其源起点约束，不能直接套用下述单调关系（在 §3.1 中移除跨拍边 `commit(g+1) -> rel(g)` 时，`rel(g)` 仍保留入边 `cmp(g) -> rel(g)`，不产生新源节点）。
  - 在不产生新源节点的前提下，省略程序边会放松 $\mathrm{ValidLower}$ 约束，因此子图下界 $L_{\mathrm{sub}}$ 仍是包含更多边的完整图的合法下界（$L_{\mathrm{sub}} \le L_{\mathrm{full}}$）；
  - 省略前驱边或将未知控制等待设为 0 会收紧 $\mathrm{ValidUpper}$，可能导致子图上界 $U_{\mathrm{sub}} \le U_{\mathrm{full}}$。因此，当观测值超出 $U_{\mathrm{sub}}$ 时，定理仅证伪“当前声明的子图边 $P_{\mathrm{sub}}$ 与偏移区间 $H_{\max}$ 的组合与观测相容”，其用途是发现不相容并限制后续解释，不能单独区分原因是漏边、区间偏小还是测量窗口差异。

## 5. 真实程序短链的接入约定与当前缺口

### 5.1 选定对象与图边界

选用 [739011 历史条件](../results/gh200_resource_campaign/access_rules/Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011/cases.json) `v09d_cfg_b_G4_base`：M3328、N768、K512，tile 为 128×128×64，grid 为 `[1,132,1]`，S6。使用同目录 `static_setup.json` 和当前 `cfg_b_program.build_program / expand_cta`，取 CTA 0、输出 `j=0`（坐标 `[0,0]`，`g=0..7`）。本轮是历史程序输入的结构核验，不重新评分该历史留出条件，也不宣称当前 Python 与历史二进制已建立形式化等价。

`expand_cta` 使用默认 `register_quota=False, epilogue_defer=0, separate_empty_completion=False`。保留 `j0.main_end` 的全部静态前驱闭包，唯一源为 `entry`，观察集合取 `O={j0.main_end}`，相对起点取 `entry=0`。从完整图导出的 `topological_order` 构造秩，限制到闭包仍逐边递增。该选择保留首个输出的准备、8 组输入/计算、两次同槽复用及尾部排空；省略的寄存器分配细节及其他 CTA 的资源竞争不因此消失。

闭包包含 133 节点、174 边，完整 CTA 图为 275 节点、374 边；这些是当前 Python 实例的 CPU 核验结果，尚不是 Lean 导入证书。不能把闭包中的服务节点直接标为零延迟并声称其上界覆盖实测。

### 5.2 五事件投影与不能直接复用的条件

| 五事件 | 候选程序节点 | 对应关系与限制 |
|---|---|---|
| `acq(g)` | `g.acquire` | 表示获得槽后的节点；`entry → acquire(0)` 的准备时间存在，不能将源起点 `[0,0]` 原样移到真实 `acquire(0)` |
| `rdy(g)` | `g.full_ready` | 依赖 `expect_tx` 和 A/B0/B1 三次完成；不等于消费者已经执行完等待 |
| `commit(g)` | `g.commit` | 静态路径为 `full_ready → consumer_wait → compute_batch → commit`，并受 `full_query` 控制 |
| `cmp(g)` | `g.compute_complete` | 与 `commit` 同为 `compute_batch` 的后继；静态图没有 `commit → compute_complete` 路径。§5.3 已给出计算递推前提下的 Lean 局部边证书，尚未修改 Python 静态图或导入整张图 |
| `rel(g)` | `g.release` | `g<7` 由下一组 `wait1` 释放；`g=7` 由 `j0.wait0` 排空释放；默认图合并了释放发出与可见完成 |

按该投影，槽复用边 `release(g) → acquire(g+6)` 为直接边；`g<7` 的跨拍边由 `commit(g+1) → wait1(g+1) → release(g)` 路径支持；相邻 acquire 通过上一组 B1 发出路径推进。它们提供静态先后关系，非零偏移仍须另证。

第一阶段实例不能直接当作这个程序的双侧规范：

- 实例中 `rdy → commit` 是唯一入边且区间为 `[0,0]`，双侧规范强制 `rdy=commit`；真实路径保留消费者推进与计算提交。仅将 `rdy` 改名为 `consumer_wait` 仍未覆盖随后发射时间。
- `cmp → rel` 及复用控制边的零上偏移，也要求等待返回、释放和后续发射在相应前驱最大值处立即推进；源码先后关系不能给出这种上界。
- 跨输出不能把所有 Ktile 连成同一种 `wait1` 规则。当前图中不存在 `g8.commit → g7.release` 路径；每输出末组由 `wait0` 释放。所选单输出避开这处边界，但不能推广到整个 pingpong 工作列表。
- 释放事件最大值不是 `MAIN_END` 的已证等式。源码在 `mma_tail` 返回后打点，Python 默认图中仍有 `g7.release → j0.main_end` 边；观察开销或控制偏移尚未定界。

接入时优先将这条细图实例化到现有 `Core.Graph`，保留当前五事件模型作为独立抽象实例。无需改变通用轨迹定义，也不能通过删去真实等待来强行复用五事件实例的数值结论。

### 5.3 服务假设的资格

| 服务 | 现有依据 | 进入双侧规范仍需满足的条件 |
|---|---|---|
| 计算提交与完成 | `supply_model.wgmma_compute_step`：`E=max(M,Eprev)+c`，`C=max(M+f,E−q)`，`D=E+λ` | 在 `c≥f≥0,q≥0,λ≥0` 下可推出 `M≤C≤E≤D` 及 `λ≤D−C≤q+λ`；但 `D−C` 不是 `c`，且单个函数本身不校验这些前提。数学候选关系不等于硬件事实 |
| TMA 应答共享 | `cfg_b_input_network.ReplyGroup` 按活动请求数分摊处理速率，其他请求排队 | 必须限制其他请求的到达、剩余工作与排队量，或展开有依据的资源先后关系。每请求自身工作量不能直接充当完成偏移上界 |
| 本地输入/计算服务 | `LocalServiceCalendar` 预留多个可用片段，`JointLocalInputNetwork` 共享该日历 | 实际有效延迟依赖已有预留，不能仅用一次回放的服务顺序宣称所有允许执行均受同一固定 DAG 覆盖 |
| 控制与观测端点 | 程序源码给出调用顺序和打点位置 | 为 consumer wait、wait1/wait0、release、main_end 等保留可解释的推进上限；没有上限依据时仅可使用已证明其前提的下界结论 |

两个局部反例用于限制误用：`M=Eprev=0,c=10,q=3,λ=1,f=0` 时计算服务给出 `E=10,C=7,D=11`，提交后剩余延迟为 4；应答组中单个工作量 10 的请求在 10 完成，两个同时到达的同样请求各在 20 完成。均为现有 Python 候选的行为，不是新增硬件测量。

计算区间的代数依据为 `E≥M+c≥M+f`，故 `E−q≤C≤E`，代入 `D=E+λ` 得上述剩余延迟界。若另有 `λ∈[λmin,λmax]`、`q≤qmax`，则可选条件区间 `[λmin,qmax+λmax]`；这些参数界的硬件依据仍须单独建立。

该关系已在 [Pipeline/Compute.lean](GemmProofs/Pipeline/Compute.lean) 中形式化，依赖通用 `Core.Model`，不改变五事件实例：

- `signed_completion_window` 对整数 `M,Eprev,c,q,λ,f : Int` 在 `f≤c,q≥0` 下证明 `C+λ≤D≤C+(q+λ)`，其中 `E−q` 为普通带符号减法。此代数结论本身无需起点或 λ 非负；图的非负时间/偏移由后续 Nat 接口保证。
- `commit_cast` 证明非负整数输入下，Nat 递推的提交时刻嵌入 Int 后，与带符号递推完全相同。无需额外假设 `q≤E`：当 `E−q<0` 时，非负的 `M+f` 分支已经占优。
- `completion_window`、`residual_bounds` 对任意 Nat 输入及 `f≤c` 分别证明加法窗口和 `λ≤D−C≤q+λ`；前者保证 `C≤D`，不会利用自然数减法截断掩盖完成早于提交。
- `edge_bounds` 将已知参数范围放宽为 `[λmin,qmax+λmax]`，并证明区间有序；`graph_edge` 对任意现有 `Core.Graph/Trace`，在提交/完成时间满足递推且存在该入边时，证明此边的下界约束与完成节点的多前驱最大值上界约束。不假定整图已满足上界规范，也不要求完成节点只有这一条入边。

本轮补齐的是计算服务条件下的局部边依据；未将 Python 的浮点运算视为精确整数递推，未证明任意实数时间的整数化保持性。其他入边下界、源起点、该新增边与整图拓扑秩的相容性及参数硬件资格，仍须在具体图接入时分别核验。

对所选多 CTA 条件，当前未建立覆盖共享服务的有效上偏移区间，也未完成源准备/控制推进区间的资格证明。静态前驱闭包不包含其他 CTA，也不包含可能在首输出尾部排空期间运行的本 CTA 下一输出；这些外部工作仍可能竞争资源。因此不能把这条静态闭包的极值上界用于真实执行的排除结论。静态零权路径先后可以核验，非平凡性能下界同样需要相应最小延迟依据。

### 5.4 时间与观测窗口

本次查阅 [G4 dual-00 原始记录](../results/gh200_resource_campaign/access_rules/Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011/samples/v09d_cfg_b_G4_base/dual-00.json) 的实际 setup 为 `r18-dual-clock-events`、每 tile 10 字，不使用静态 setup 的 6 字布局解码。CTA 0 首输出坐标 `[0,0]`，溢出标记为 0，entry 与两消费者角色记录的 SM 相同。冻结 `v06_trace.hpp` 在每个打点先读 `clock64` 再读 `globaltimer`；collective 在首次 consumer wait 后、WGMMA 发射前记录 `FIRST_MMA`，kernel 在 `mma_tail` 后记录 `MAIN_END`。

本短链的候选观察口径为 **ENTRY → MAIN_END，globaltimer 整数 ns**，原始记录差值为 4800；同次 `clock64` 差值为 8401。历史主循环口径 **FIRST_MMA → MAIN_END** 则为 2816 ns / 4556 cycles。两种窗口不能互换，两个时钟也不能混用。4800 仅是一次带打点执行的观测，不是区间上限或校准参数；没有据此拟合服务偏移，也未构造用于硬件证伪的 `I_obs`。

若继续采用整数 ns，必须给出将物理/候选时间映射到整数轨迹的误差规则及测量误差范围，并向外放宽区间；现有 Python 的小数 ns 不可直接截断后当作已满足 Lean 约束。若改用主循环窗口，要保留 FIRST 之前的输入与资源状态，明确两个端点的差值或等价切分；不能把 FIRST 强制设为全图零源并丢弃先前工作。带打点窗口与 plain CUDA event 时间也不等价。
