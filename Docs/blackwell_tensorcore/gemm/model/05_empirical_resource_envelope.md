# 05 经验资源包络

经验层回答：对一个已证明合法的具体 schedule \(x\) 和 workload \(w\)，使用同合同 microbenchmark 容量，一个消除已知实现浪费的 schedule 需要多长时间。

## 1. Capacity 不是只有一个数字

定义 \(\widehat C_r\(\kappa\)\) 为资源 \(r\) 在合同 \(\kappa\) 下测得的 sustained service rate。合同至少可能包含：

- precision；
- MMA shape；
- CTA group；
- hardware ID、SM count、mode、clock；
- residency；
- TMA payload 与 destination slots；
- TMEM load width 与 readback warps；
- threads/CTA 与 resident CTAs/SM；
- read:write ratio；
- schedule/workload ID；
- timed scope。

只有 \(\kappa_{\mathrm{capacity}}\) 与 \(\kappa_{x,w}\) 匹配时，容量才能被选择。同名或位宽相近不构成等价证明。

## 2. 单资源经验时间

对 schedule-issued work (Q_r\(x,w\))：

\[
\widehat T_r(x,w)
=\frac{Q_r(x,w)}{\widehat C_r(\kappa_{x,w})}.
\]

定义独立资源经验时间：

\[
\widehat T_{\mathrm{resource}}(x,w)
=\max_r\widehat T_r(x,w).
\]

最大值表示允许资源理想重叠；它不是 joint attainability 证明。集成包络还需要第 06 章的 exact causal profile。

## 3. Empirical compute

经验 compute 使用 \(W_{\mathrm{issue}}\) 和 shape-qualified resource：

```text
tensor.<format>.m<MM>n<NN>
```

例如 `tensor.bf16.m128n256` 只服务 MMA M128N256 schedule，不能替代 M128N64 或 M128N128。

若 full-GPU compute capacity 还绑定 threads/CTA 与 resident CTAs/SM，模型可推导 per-group task span 与整数 wave makespan；缺少这些 scope 时只保留 aggregate service time。

## 4. Ratio-qualified memory duplex

### 4.1 Cold external memory

定义理想 cold-entry read：

\[
Q_{\mathrm{HBM,R}}(x,w)
=Q_{\mathrm{TMA,unique}}(x,w)+Q_C^{\mathrm{LB}}.
\]

定义 external write：

\[
Q_{\mathrm{HBM,W}}(x,w)=Q_D^{\mathrm{LB}}.
\]

定义精确 read:write ratio：

\[
\rho_{\mathrm{HBM}}(x,w)
=
Q_{\mathrm{HBM,R}}(x,w):Q_{\mathrm{HBM,W}}(x,w).
\]

只有 `resource="hbm.duplex"`、ratio 匹配且 external read/write bytes 都得到相应证据的 capacity 才能给出：

\[
\widehat T_{\mathrm{HBM,duplex}}(x,w)
=
\frac{Q_{\mathrm{HBM,R}}+Q_{\mathrm{HBM,W}}}
     {\widehat C_{\mathrm{HBM,duplex}}(\rho_{\mathrm{HBM}})}.
\]

Thor `-i` cold campaign 只证明 external read miss 和 L2 write-path issue，导入资源是 `hbm.duplex.proxy`，不是 `hbm.duplex`。它不能满足上式，所以当前 physical HBM empirical layer仍 fail closed。详见 [EXP-04](../experiments/EXP-04-memory-duplex-surface.md)。

### 4.2 Shared L2 duplex

定义 schedule 对共享 L2 发出的 read request payload：

\[
Q_{\mathrm{L2,R}}(x,w)
=Q_{\mathrm{TMA,issued}}(x,w)+Q_C^{\mathrm{LB}}.
\]

定义 write payload：

\[
Q_{\mathrm{L2,W}}(x,w)=Q_D^{\mathrm{LB}}.
\]

定义：

\[
\rho_{\mathrm{L2}}(x,w)
=Q_{\mathrm{L2,R}}(x,w):Q_{\mathrm{L2,W}}(x,w),
\]

\[
\widehat T_{\mathrm{L2,duplex}}(x,w)
=
\frac{Q_{\mathrm{L2,R}}+Q_{\mathrm{L2,W}}}
     {\widehat C_{\mathrm{L2,duplex}}(\rho_{\mathrm{L2}})}.
\]

`l2.duplex` 是 GPU-wide shared service，不乘 SM 数。当前 `-i` 结果覆盖现有 workload/schedule manifest 所需的 L2 ratio surface。

## 5. Exact TMA service

模型把共享 L2/DRAM fabric 与每 SM TMA→SMEM 出口分开。

### 5.1 Resource identity

exact TMA resource 包含：

```text
tma.smem_ingress.contract.<family>.stride<ld>.per_sm
tma.hbm.contract.<family>.stride<ld>
```

`family` 冻结 tile、value/scale payload、request count、stage、thread 与 SMEM topology；`stride` 冻结共同 packed row stride。

当前 NN v1 中 A leading dimension 为 K、B leading dimension 为 N；现有 resource campaign 使用共同 A/B stride，所以只有 (K=N) 且 stride 已在 manifest 中时才能匹配。

历史 tc5a capacity 只有到 stride 2048 的单向 alias，不能覆盖 1024/4096 或其它 family。

### 5.2 Per-SM makespan

定义每 task TMA payload：

\[
Q_{\mathrm{TMA/task}}
=\frac{Q_{\mathrm{TMA,issued}}}{N_{\mathrm{task}}}.
\]

定义一个 SM 的 sustained ingress 为 \(\widehat C_{\mathrm{TMA,SM}}\)。CTA-group-1、每 SM 一个 service worker 时：

\[
\widehat T_{\mathrm{TMA,span}}
=\frac{Q_{\mathrm{TMA/task}}}{\widehat C_{\mathrm{TMA,SM}}},
\]

\[
\widehat T_{\mathrm{TMA,makespan}}
=
\left\lceil\frac{N_{\mathrm{task}}}{S}\right\rceil
\widehat T_{\mathrm{TMA,span}}.
\]

不能用 full-grid aggregate TMA rate 除以 20 推导 per-SM rate；测量可能已经被共享 L2 限速。per-SM evidence 必须由单 CTA/单 observed SM 隔离。详见 [EXP-03](../experiments/EXP-03-tma-payload-surface.md) 与 [EXP-05](../experiments/EXP-05-exact-tma-topology.md)。

## 6. TMEM 与其它 schedule resources

经验层还可能包含：

- `tmem.scale_ingress`；
- `tmem.readback` 或 `tmem.readback.x<registers>.warps<warps>`；
- `reduction.io`；
- 已证明的 fixed time。

block-scale schedule 缺 scale ingress capacity 时不能借 accumulator readback 或 TMA rate。tc5a 192-thread CTA 只有 4 个 readback warp，因此不能使用 6-warp 假设。

## 7. Hard upper intersection

经验 rate 是内点。对同一个 schedule，模型把所有适用 strict upper 重新加入 `hard_upper:*` 时间约束。例如 cold schedule 即使有 physical duplex capacity，仍必须满足共享 `hbm.total`、`l2.read` 和 `l2.write` 外上界。

这可以发现经验参数、工作量或作用域不一致，但不会把 measured rate 升级为物理 upper。

## 8. Resource layer 的 fail-closed 规则

经验层缺少任一必需 resource 时返回：

```text
status = insufficient_evidence
performance_per_second = null
missing_resources = [...]
```

它可以同时保留已经算出的 `resource_seconds` 作为诊断，但不能把剩余资源拼成“半个包络”。

## 9. Manifest resource envelope

定义合法 schedule 集 \(\mathcal X_{\mathrm{manifest}}\)。只有当每个合法 schedule 都有数值 resource layer 时，才定义：

\[
\widehat T_{\mathrm{resource,manifest}}(w)
=
\min_{x\in\mathcal X_{\mathrm{manifest}}}
\widehat T_{\mathrm{resource}}(x,w).
\]

忽略一个缺数值的合法 schedule 会低估潜在最优性能，所以整个 manifest resource envelope 必须 fail closed。

资源层仍没有证明独立 component capacities 能同时达到。下一章用 exact causal profile 建模 startup、steady interval、accumulator reuse 和 drain。
