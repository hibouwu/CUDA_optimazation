# 07 完整 GEMM、反证与重校准

完整 GEMM observation 是第三层证据。它不能作为 component capacity，也不能直接定义物理 rate upper；它用于报告当前最好实现、验证数值语义、反证错误上界并校准经验模型。

## 1. Observation 合同

一个 closure-qualified `ObservedBest` 必须冻结：

- precision、M/N/K；
- transpose、alpha、beta、epilogue、output mode；
- arithmetic path；
- hardware ID、SM count、operating mode；
- residency 与 timed scope；
- candidate backend；
- correctness reference 与 relation；
- performance denominator 与 relation；
- calibration/holdout split；
- trial count 与 matched count；
- candidate/reference min、median、max；
- source locator、run ID 和 artifact paths。

所有声明 trial 必须 matched；closure 至少要求 10 个外部 trial。

## 2. Correctness reference 与性能 denominator 分开

`correctness_reference` 回答输出是否满足同一数值合同；`reference` 回答性能 ratio 的 denominator 是谁。二者可以是同一个 backend，也可以不同。

只有 `performance_reference_relation="same_precision"` 时，candidate/reference ratio 才能解释为同精度库性能比较。FP4 candidate 对 FP16 cuBLAS 的 ratio 只能是 cross-precision diagnostic。

closure-qualified correctness 要求：

```text
correctness_reference_relation = independent_same_contract
```

不能用 candidate 自己的输出、只检查 CUDA error 的 binary 或单纯 SASS presence 代替数值 reference。

## 3. 已观测最好值

对相同 workload 合同，定义 eligible backend 集 \(\mathcal B\(w\)\)。定义：

\[
P_{\mathrm{obs}}(w)
=
\max_{b\in\mathcal B(w)}
P_{b,\mathrm{median}}(w).
\]

同时保留 winning backend 的 maximum trial，用于上界反证；只比较 median 会漏掉单个 trial 超过 upper 的情况。

## 4. 对条件上界的反证

给定容差 \(\epsilon_{\mathrm{ub}}\)，若：

\[
P_{\mathrm{obs,max}}
>
(1+\epsilon_{\mathrm{ub}})P_{\mathrm{ub}},
\]

至少有一项错误：

- workload/precision 语义不一致；
- 工作量分子错误；
- rate upper 数值或单位错误；
- hardware/clock/scope 错配；
- 把 measured point 当作 upper；
- 输出/denominator 不同合同；
- artifact 导入或计时错误。

不能通过提高容差或删除 observation 消除这种矛盾。

## 5. 对经验包络的重校准

给定经验容差 \(\epsilon_{\mathrm{env}}\)，若：

\[
P_{\mathrm{obs,max}}
>
(1+\epsilon_{\mathrm{env}})
\widehat P_{\mathrm{env}},
\]

这不违反物理定律，而说明经验模型至少存在一个问题：

- component rate 不是实际 schedule 的可用 service；
- issued work 计账过高；
- cache reuse/residency 假设错误；
- causal profile 不匹配；
- manifest 漏掉更优 schedule；
- independently measured resources 的 joint behavior 被误建模。

应回到 experiment contract 和 workload/schedule mapping 重校准。

## 6. 三层 closure

一个 workload 的 `absolute_three_layer_closure` 至少要求：

1. `domain_conditional_upper` 数值存在且完整；
2. 只使用 closure-qualified empirical capacities/profile；
3. `empirical_ideal_envelope` 数值存在；
4. 有 residency/timed-scope 对齐的完整 GEMM observation；
5. correctness 为 independent same contract；
6. observation 不超过 conditional upper；
7. observation 不超过经验包络的预声明重校准容差。

同精度性能 ratio 是额外比较维度，不应阻止 absolute performance closure；因此另报 `same_precision_ratio_closure`。

## 7. Calibration 与 holdout

workload manifest 每个 precision/domain 至少要有：

- 一个 calibration workload；
- 一个预声明 holdout workload。

当前 full-GEMM campaign 使用 N=1024/2048 calibration、N=4096 holdout。split 不证明 cache residency；residency 仍需独立 NCU 或构造证据。

一套参数在 calibration 上拟合良好但 holdout 超差，应标记为模型不完整或 profile quarantine，不能用 holdout 重新调参后仍称为留出验证。

## 8. 当前完整 GEMM 覆盖

当前 runner 已覆盖 6 个 closure-ready precision path：

- FP16；
- BF16；
- TF32；
- E4M3；
- E5M2；
- signed INT8。

当前历史 Thor 结果早于 E5M2 扩展，不能把 runner 就绪称为 E5M2 fresh observation。其余 FP6、raw E2M1、MXFP4、NVFP4 和 U8 仍缺完整 candidate/reference matrix 或同合同 denominator。

详见 [EXP-08](../experiments/EXP-08-full-gemm-validation.md)。下一章汇总当前所有机械门禁与剩余 Thor 实验。
