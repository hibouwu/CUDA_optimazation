# 04 条件可证明性能上界

严格层只使用“真实服务率不可能高于它”的外上界，不使用 microbenchmark sustained 中位数。

## 1. 单资源时间下界

定义资源集合 \(\mathcal R\)。对资源 \(r\)，定义任何合法实现不可避免的最低工作 \(Q_r^{\mathrm{LB}}\)，定义 rate upper \(U_r\)。只有证据支持真实服务率 \(C_r\le U_r\) 时，才有：

\[
T_r^{\mathrm{LB}}=
\frac{Q_r^{\mathrm{LB}}}{U_r}.
\]

若同一资源有多个同时成立的 upper，取最小 \(U_r\)，即外约束交集。

## 2. 多资源理想重叠

定义独立资源时间下界：

\[
T_{\mathrm{resource}}^{\mathrm{LB}}
=
\max_{r\in\mathcal R}T_r^{\mathrm{LB}}.
\]

取最大而非求和，表示允许不同资源完美重叠。这是“没有可避免性能浪费”所需的乐观时间下界。

如果两个 rate 共享同一物理容量，必须增加 joint outer constraint；不能为了保守而任意相加，也不能为了乐观而假设两个 peak 可同时达到。

## 3. 当前严格资源

### 3.1 Compute

对 `tensor_core_classical`，定义 precision 对应 Tensor Core rate upper \(U_{\mathrm{TC},p}\)：

\[
T_{\mathrm{compute}}^{\mathrm{LB}}
=
\frac{W_{\mathrm{use}}}{U_{\mathrm{TC},p}}.
\]

对 `all_classical`，必须使用 aggregate `compute.total.<precision_id>` upper，不能只使用 Tensor Core pipe upper。

当前大约 7/12 precision 有某种条件 compute upper；TF32、FP6、raw E2M1 和 MXFP4 等仍缺独立外上界。

### 3.2 External memory

对 `cold_hbm`，定义 minimum external bytes：

\[
Q_{\mathrm{ext}}^{\mathrm{LB}}
=Q_{\mathrm{in,val}}^{\mathrm{LB}}
+Q_{\mathrm{in,scale}}^{\mathrm{LB}}
+Q_C^{\mathrm{LB}}
+Q_D^{\mathrm{LB}}.
\]

若产品规格给出共享 LPDDR/HBM total upper \(U_{\mathrm{ext,total}}\)：

\[
T_{\mathrm{ext,total}}^{\mathrm{LB}}
=\frac{Q_{\mathrm{ext}}^{\mathrm{LB}}}{U_{\mathrm{ext,total}}}.
\]

Thor profile 当前使用 273 GB/s LPDDR5X 条件上界，并显式声明 no-compression/no-external-reuse 条件。

### 3.3 GPU-wide L2

定义最低 L2 read 和 write：

\[
Q_{\mathrm{L2,read}}^{\mathrm{LB}}
=Q_{\mathrm{in,val}}^{\mathrm{LB}}
+Q_{\mathrm{in,scale}}^{\mathrm{LB}}
+Q_C^{\mathrm{LB}},
\]

\[
Q_{\mathrm{L2,write}}^{\mathrm{LB}}=Q_D^{\mathrm{LB}}.
\]

当前条件参数：

- \(U_{\mathrm{L2,read}}=1024\ \mathrm{B/cycle/GPU}\)；
- \(U_{\mathrm{L2,write}}=512\ \mathrm{B/cycle/GPU}\)。

在 1.575 GHz snapshot 下分别换算为 1.6128 TB/s 和 0.8064 TB/s。它们是整 GPU 共享 bus，不乘 `sm_count`。

当前没有可引用的 outer constraint 证明：

\[
\frac{R}{1024}+\frac{W}{512}\le1.
\]

因此严格层暂时把 read/write 作为两条独立外约束并允许理想重叠。这使 upper 可能更松，但不把 measured duplex 内点冒充外边界。参数来源见 [EXP-02](../experiments/EXP-02-l2-physical-bounds.md)。

## 4. 有限并行下界

定义不可再分割任务 \(i=1,\ldots,n_t\)，单任务最低服务时间 \(p_i\)，等价并行 service unit 数 \(U_t\)。任意调度满足：

\[
T_{\mathrm{parallel}}
\ge
\max\left(
\frac{\sum_i p_i}{U_t},
\max_i p_i
\right).
\]

同构 task 且理想调度时：

\[
T_{\mathrm{parallel,identical}}
=\left\lceil\frac{n_t}{U_t}\right\rceil p.
\]

严格层只有在 per-SM/per-CTA 服务率本身有外上界证据时才使用该约束；不能把 GPU aggregate peak 平均除以 SM 数制造单 CTA upper。

## 5. 因果 span 下界

定义执行依赖图 (G=(V,E))，节点为 load、MMA、wait、TMEM readback、epilogue 等事件，边为必须满足的 producer/consumer 依赖。定义图中最长不可绕过路径为 \(T_{\mathrm{span}}^{\mathrm{LB}}\)。

只有延迟本身具有下界证据时，它才能进入严格层。实测 latency 通常是内点，不自动成为任何实现都无法突破的 latency lower bound。

## 6. Joint outer region

若能证明吞吐向量 \(\mathbf y\) 满足：

\[
\mathbf H\mathbf y\le\mathbf c,
\]

则可由每一行推导 joint 时间下界 \(T_{\mathrm{joint}}^{\mathrm{LB}}\)。当前代码尚未实现任意 \(\mathbf H,\mathbf c\) 的通用输入 schema；现阶段只通过命名的 strict resources 表达已经支持的外约束。

联合 microbenchmark 提供容量区域内点，不能单独构造 \(\mathbf H\mathbf y\le\mathbf c\) 的外边界。

## 7. 总时间下界与性能上界

定义已经证明不可消除的 fixed time lower bound 为 \(T_{\mathrm{fixed}}^{\mathrm{LB}}\)。当前零浪费 upper 令未证明的 fixed lower bound 为 0；这不是“测得 fixed cost 为零”。

\[
T_{\mathrm{ub}}^{\mathrm{LB}}
=
\max\left(
T_{\mathrm{resource}}^{\mathrm{LB}},
T_{\mathrm{parallel}}^{\mathrm{LB}},
T_{\mathrm{span}}^{\mathrm{LB}},
T_{\mathrm{joint}}^{\mathrm{LB}},
T_{\mathrm{fixed}}^{\mathrm{LB}}
\right),
\]

\[
P_{\mathrm{ub}}
=\frac{W_{\mathrm{use}}}{T_{\mathrm{ub}}^{\mathrm{LB}}}.
\]

没有任何有效时间约束时返回 `insufficient_evidence`。只有部分资源 upper 时可返回 `partial`，并列出 `missing_resources`。

## 8. Domain upper 与 manifest upper

`domain_conditional_upper` 只接受：

- `tensor_core_classical` / `all_classical` scope 的 compute upper；
- `all_classical` memory hierarchy upper；
- 与当前 hardware/mode/clock 匹配的外约束。

它不依赖 schedule manifest。

`manifest_conditional_upper` 是所有合法 manifest schedule 的 performance upper 最大值。只要有一个合法 schedule 没有数值上界，整个 manifest upper fail closed；不能把该 schedule 从最大值中静默删掉。

下一章对具体 schedule 使用 measured capacities 构造经验资源包络。
