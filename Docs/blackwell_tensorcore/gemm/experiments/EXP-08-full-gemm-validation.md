# EXP-08：完整 GEMM candidate/reference validation

## 1. 研究问题

当前 candidate 在完整数据搬运、算术、readback、epilogue/store 路径下是否满足数值合同；其性能相对 same-contract reference、strict upper 和 empirical envelope 处于什么位置？

## 2. 对应模型对象

- `ObservedBest`；
- `numeric_closure`；
- `absolute_three_layer_closure`；
- `same_precision_ratio_closure`；
- upper contradiction 与 empirical recalibration finding。

## 3. Case matrix

当前 runner-ready precision path：

- FP16；
- BF16；
- TF32；
- E4M3；
- E5M2；
- signed INT8。

shape：N∈{1024,2048,4096} 的方阵；1024/2048 为 calibration，4096 为 holdout；每 case 10 external trials。

历史 Thor base suite 早于 E5M2 扩展，因此 E5M2 runner 就绪不等于已有 fresh Thor observation。

## 4. Correctness 合同

每个 precision 在 support manifest 中声明：

- input/accumulator/output type；
- candidate backend；
- independent numerical reference；
- same input precision；
- same output type；
- performance denominator；
- same-precision status；
- implementation source paths。

trial 必须报告 reference sample/mismatch 与 candidate/reference performance；只检查 CUDA error 不合格。

## 5. 性能与计时

candidate/reference 使用同一 workload 和设备端 kernel timing contract。模型保留各自 min/median/max，并定义：

\[
R_{\mathrm{cand/ref}}
=
\frac{P_{\mathrm{cand,median}}}
     {P_{\mathrm{ref,median}}}.
\]

只有 denominator 为 same precision 时，该 ratio 才能解释为同精度实现比较。

## 6. 接受门禁

- 10/10 matched trials；
- candidate 与 reference numerical contract 一致；
- function-scoped target SASS；
- selected holdout NCU；
- run spec、support manifest、source hash；
- compile command、binary hash、SASS；
- environment/SM/mode/clock/commit；
- trial timeout 与 process-group termination contract；
- independent portable auditor；
- COMPLETE marker 和 artifact hash。

## 7. 进入模型

standalone audited campaign 先导入为 `snapshot_only` observation；只有 compute/component/full 三批 suite linkage 证明相同 host/GPU/commit 后，才升级为 `closure_qualified` 并绑定 exact hardware scope。

observation 只进入第三层与反证逻辑，不进入 component capacities。

## 8. 当前状态

历史 Thor observation 覆盖 FP16、BF16、TF32、E4M3、S8；current runner 增加 E5M2，但尚无对应 fresh result。其它 precision 仍缺完整 candidate/reference 或同精度 denominator。

历史 FP16 N=2048 tc5a/cublas 数值仍可作为历史采集引用，但旧 128.436 TFLOP/s envelope 基于迁移前独立 component 组合，不能当作 current f06 integrated envelope。详见 [历史结果](../appendices/historical_results.md)。

## 9. 不能证明什么

- complete GEMM performance 不直接给出内部 component capacity；
- full-GEMM 超过 empirical envelope 说明模型需重校准，不自动反证物理 upper；
- static-only E5M2 不是 Thor numerical/performance observation；
- calibration/holdout split 不证明 L2/HBM residency；
- cross-precision denominator ratio 不能解释为同精度库胜负。

## 10. 源码与工件

- runner：[run_full_gemm_campaign.py](../../../../microbench/sm110_full_gemm_campaign/run_full_gemm_campaign.py)
- portable auditor：[audit_campaign.py](../../../../microbench/sm110_full_gemm_campaign/audit_campaign.py)
- support manifest：[support_manifest.json](../../../../microbench/sm110_full_gemm_campaign/support_manifest.json)
- FP16 entry：[main.cu](../../../../GEMMsm110/src/main.cu)
- extended precision entry：[extended_gemm_bench.cu](../../../../GEMMquant_sm110/src/extended_gemm_bench.cu)
- quant entry：[quant_gemm_bench.cu](../../../../GEMMquant_sm110/src/quant_gemm_bench.cu)
- observation importer：[observations.py](../../../../scripts/sm110_gemm_model/observations.py)
