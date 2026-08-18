# 06 因果流水线模型

独立 resource capacity 只能说明每条路径单独达到过某个 rate。要把资源层升级为完整经验理想包络，必须证明 TMA、MMA、TMEM readback 与 store 在同一 schedule 中的依赖和 joint timing。

## 1. Profile 作用域

一个 closure-qualified `PipelineProfile` 必须冻结：

- profile/resource/schedule ID；
- singleton 或显式 precision set；
- exact hardware ID、SM count、mode、clock；
- input residency 与 timed scope；
- stage 数与 accumulator buffer 数；
- resident CTAs/SM；
- calibration/holdout K-tile 和 output-task 范围；
- 每 case trial 数；
- source、expected commit、SASS/NCU/raw trial artifacts；
- 预声明拟合门禁和逐坐标验证。

当前 causal calibration 只接受 `hot_l2`。它不能直接关闭 cold-HBM residency。

## 2. Component timing 参数

定义：

- \(\lambda_T\)：TMA first completion latency；
- \(\iota_T\)：TMA steady completion interval；
- \(\lambda_M\)：MMA first completion latency；
- \(\iota_M\)：MMA steady completion interval；
- \(\lambda_J\)：joint pipeline first MMA completion；
- \(\iota_J\)：joint steady completion interval；
- \(\lambda_E\)：一个 output task 的 epilogue/readback/store drain latency。

单位均为 s。`measured_joint` 只进入经验层。

## 3. Persistent worker 递推

定义一个最慢 resident worker 负责 \(O\) 个 output task，每个 task 包含 \(K_t\) 个 K tile，accumulator buffer 数为 \(A\)。定义第 \(o\) 个 task 的首个 MMA 完成时刻 \(F_o\)、最后 MMA 完成时刻 \(L_o\) 和 epilogue 完成时刻 \(E_o\)。

第一个 task：

\[
F_0=\lambda_J.
\]

后续 task 的初始 joint 约束：

\[
F_o\ge L_{o-1}+\iota_J.
\]

当 accumulator buffer 将被复用时，还必须等待前 \(A\) 个 task drain：

\[
F_o\ge E_{o-A}+\iota_J,
\qquad o\ge A.
\]

因此：

\[
F_o=
\begin{cases}
\lambda_J,&o=0,\\
L_{o-1}+\iota_J,&0<o<A,\\
\max(L_{o-1}+\iota_J,E_{o-A}+\iota_J),&o\ge A.
\end{cases}
\]

一个 task 的最后 MMA：

\[
L_o=F_o+(K_t-1)\iota_J.
\]

readback/store 串行 drain：

\[
E_o=\max(L_o,E_{o-1})+\lambda_E,
\]

其中 \(E_{-1}=0\)。最慢 worker 时间为：

\[
T_{\mathrm{worker}}=E_{O-1}.
\]

这条递推显式保留 startup、joint interval、双 accumulator reuse 和 serialized drain；不能用简单 `first + (n-1)max(TMA,MMA)` 替代。

## 4. 从整卡 task 到最慢 worker

定义 service worker 数：

\[
U_{\mathrm{worker}}
=
\left\lfloor\frac{S}{\mathrm{cta\_group}}\right\rfloor
\times \mathrm{resident\_ctas\_per\_sm}.
\]

定义最慢 worker 的 output-task 数：

\[
O=
\left\lceil
\frac{N_{\mathrm{task}}}{U_{\mathrm{worker}}}
\right\rceil.
\]

若 \(K_t\) 或 \(O\) 超出 profile 的 calibration/holdout 范围，模型拒绝外推并返回 `profile_range` 缺口。

## 5. Causal layer

定义：

\[
\widehat T_{\mathrm{DAG}}(x,w)=T_{\mathrm{worker}},
\qquad
\widehat P_{\mathrm{DAG}}(x,w)
=\frac{W_{\mathrm{use}}}{\widehat T_{\mathrm{DAG}}(x,w)}.
\]

profile 是不可拆分的 joint evidence。若存在多次完整 campaign，可以选择最快的完整 profile；不能从不同 campaign 中各取一个最好 component 再组合。

## 6. 与资源层合并

具体 schedule 的集成经验时间：

\[
\widehat T(x,w)
=
\max\left(
\widehat T_{\mathrm{resource}}(x,w),
\widehat T_{\mathrm{DAG}}(x,w)
\right).
\]

经验性能为：

\[
\widehat P(x,w)=
\frac{W_{\mathrm{use}}}{\widehat T(x,w)}.
\]

资源层或 causal 层任一缺失时，`empirical_ideal_envelope` 没有数值。代码仍分别报告两个子层，便于识别缺口来源。

## 7. Manifest 集成包络

只有当每个合法 schedule 都有完整资源层与 causal 层时，才定义：

\[
\widehat T_{\mathrm{env}}(w)
=\min_{x\in\mathcal X_{\mathrm{manifest}}}
\widehat T(x,w),
\]

\[
\widehat P_{\mathrm{env}}(w)
=\frac{W_{\mathrm{use}}}{\widehat T_{\mathrm{env}}(w)}.
\]

manifest 不是所有可能 GEMM 算法的全集，因此 \(\widehat P_{\mathrm{env}}\) 不是绝对物理上界。

## 8. 当前证据

当前代码已有：

- persistent-worker DAG 求解器；
- tc5a FP16/BF16 分精度 runner；
- 每精度 91 case、合计 182 case 的冻结合同；
- calibration/holdout 与 SASS/NCU auditor；
- synthetic 1,820-trial 回归。

当前没有回传 closure-qualified Thor timing profile，所以 `causal_pipeline_closed_count=0`。详见 [EXP-07](../experiments/EXP-07-causal-pipeline.md)。

下一章说明完整 GEMM observation 如何验证、反证或重校准上述模型。
