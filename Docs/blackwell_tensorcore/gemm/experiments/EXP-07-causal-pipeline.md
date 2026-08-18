# EXP-07：exact causal pipeline

## 1. 研究问题

在同一 persistent-worker schedule 内，TMA、MMA、accumulator reuse、TMEM readback 与 store 的 startup、steady interval 和 drain 如何共同决定最慢 worker completion time？

## 2. 对应模型参数

- `tma_first_completion_seconds`；
- `tma_completion_interval_seconds`；
- `mma_first_completion_seconds`；
- `mma_completion_interval_seconds`；
- `joint_first_mma_completion_seconds`；
- `joint_completion_interval_seconds`；
- `epilogue_latency_seconds`；
- full-worker calibration/holdout validation。

## 3. Frozen schedule

当前 runner 只覆盖：

```text
tc5a_m128n256k64_stage4
```

并为 FP16 与 BF16 分别建立 singleton profile。共同 topology：

- BM128N256K64；
- 4 stages；
- A=16 KiB、B=32 KiB per K tile；
- two TMA requests/K tile；
- four MMA instructions/K tile；
- two accumulator buffers；
- 192 threads；
- one resident CTA/SM；
- hot-L2 input。

相同 16-bit payload 宽度不构成 FP16/BF16 timing 可复用证明；tensor-map type、instruction descriptor、SASS function 和 profile precision ID 分别审计。

## 4. Case matrix

每个 precision 91 case：

- TMA-only stage 1/2/4；
- MMA-only K-tile sweep；
- joint TMA+MMA；
- full persistent worker output-task sweep；
- calibration K tiles {1,2,4,8,16,32}；
- holdout K tile 64；
- output tasks {1,2,4,8,16,32}；
- 10 external trials/case；
- 4 份预声明 NCU/precision。

两 precision 合计 182 case、1,820 trials、8 NCU。

## 5. Fit 与门禁

component fit 记录 intercept/slope；joint profile 使用首个 MMA completion、joint interval 和单-output-task epilogue latency。门禁包括：

- component/joint R²；
- calibration 最大相对误差；
- holdout 最大相对误差；
- 每个 validation coordinate 由 importer/auditor重新计算；
- joint interval 不得快于孤立 component 的物理记录而无解释；
- profile qualification 与预声明门禁结果一致。

miss gate 的完整 bundle保留为 `quarantined`，不删除也不进入 envelope。

## 6. 当前状态

- DAG solver：已实现；
- runner/auditor/static contract：已实现并通过回归；
- FP16/BF16 frozen profile schema：已实现；
- fresh Thor 182-case timing bundle：尚未回传；
- current closure-qualified causal profile count：0。

## 7. 进入模型

profile 只有在 hardware/SM/mode/clock、schedule、precision、residency、timed scope、stage、resident CTA 和 K/output range 全匹配时进入 `causal_pipeline_envelope`。

资源层与 profile 层取时间最大值；缺任一层时 `empirical_ideal_envelope` fail closed。

## 8. 不能证明什么

- synthetic 1,820-trial regression 不是 Thor timing；
- static SASS 不证明 runtime protocol；
- hot-L2 profile 不关闭 cold-HBM；
- tc5a profile 不适用于 generic/FP6/block-scale schedule；
- profile 内点不是 strict latency/rate outer bound；
- full-GEMM observation不能替代内部 event DAG。

## 9. 源码与工件

- CUDA source：[tc5a_pipeline_dag.cu](../../../../microbench/16_tc5a_pipeline_dag/tc5a_pipeline_dag.cu)
- manifest：[contract_manifest.json](../../../../microbench/sm110_gemm_causal_campaign/contract_manifest.json)
- runner：[run_causal_campaign.py](../../../../microbench/sm110_gemm_causal_campaign/run_causal_campaign.py)
- campaign auditor：[audit_campaign.py](../../../../microbench/sm110_gemm_causal_campaign/audit_campaign.py)
- platform auditor：[audit_causal_suite.py](../../../../microbench/sm110_gemm_causal_campaign/audit_causal_suite.py)
- importer：[causal_import.py](../../../../scripts/sm110_gemm_model/causal_import.py)
