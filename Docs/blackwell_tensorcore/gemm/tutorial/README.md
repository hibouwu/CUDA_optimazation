# GEMM 性能模型教程入口

本目录提供学习顺序，不重新定义规范模型。遇到教程与 `model/` 冲突时，以 `model/` 和当前可执行代码为准。

## 1. 建议学习路线

1. [为什么需要 upper / envelope / observed 三层](../model/01_scope_and_claims.md)
2. [符号、单位与硬件作用域](../model/02_symbols_units_and_workload.md)
3. [从 GEMM 推导 useful/minimum/unique/issued work](../model/03_work_accounting.md)
4. [从时间下界得到性能上界](../model/04_strict_performance_upper.md)
5. [共享 L2、memory duplex 与 per-SM ingress](../model/05_empirical_resource_envelope.md)
6. [persistent-worker 因果 DAG](../model/06_causal_pipeline_model.md)
7. [完整 GEMM 如何反证或重校准模型](../model/07_observed_gemm_and_falsification.md)
8. [当前哪些门禁仍为 false](../model/08_current_coverage_and_gaps.md)

## 2. 配套实验阅读顺序

```text
EXP-01 compute
  → EXP-02 L2 scope
  → EXP-03 TMA payload
  → EXP-04 memory duplex
  → EXP-05 exact TMA topology
  → EXP-06 TMEM
  → EXP-07 causal pipeline
  → EXP-08 full GEMM
```

每份实验先读“不能证明什么”，再读结果；这样可以避免把 component rate 外推成完整 GEMM 结论。

## 3. 一个练习顺序

完整 worked example：[FP16 N=2048 current-model 手算](01_fp16_n2048_worked_example.md)。

以 FP16 N=2048、tc5a M128N256K64 为例：

1. 计算 \(W_{\mathrm{use}}\)；
2. 计算 minimum input/output；
3. 计算 128 output tasks、32 K tiles；
4. 推导 A=64 MiB、B=128 MiB schedule-issued request；
5. 区分 16 MiB unique cold-entry 与 192 MiB L2/TMA issued work；
6. 用 1024/512 B/cycle/GPU 构造 strict L2 constraints；
7. 构造 exact L2/HBM duplex ratios；
8. 检查 exact TMA family/stride；
9. 检查 causal profile 是否存在；
10. 在缺失 physical HBM duplex/profile 时得到 `insufficient_evidence`，而不是沿用历史 envelope。

## 4. Legacy 教程

常见错误与反证：[common failure modes](02_common_failure_modes.md)。

[旧单体教程](../../thor_sm110_gemm_performance_model_tutorial.md) 保留大量手算、预测题和历史说明。它生成于 current duplex/exact/causal schema 之前，包含旧独立 `hbm.read/write` empirical 叙述；在逐课迁移完成前只作为 legacy 教材，不作为 current 公式来源。

## 5. 掌握标准

读者应能：

- 说明为什么 rate upper 与 measured sustained 方向不同；
- 区分 minimum、unique、issued 和 physical bytes；
- 区分 GPU-wide L2 与 per-SM TMA；
- 为 exact ratio/topology 选择 capacity；
- 手算 task wave 与最慢 worker；
- 解释 observed 超过 upper 与超过 envelope 的不同后果；
- 在缺证据时主动输出 `insufficient_evidence`。
