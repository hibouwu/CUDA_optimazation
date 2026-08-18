# 历史结果与模型收紧记录

本附录保存仍有审计价值、但不再代表 current schema 的结果。历史数据不会因模型收紧而失效；其结论作用域必须保留。

## 1. 2026-08-14 composite closure

- code：`63630d10f239d4f725e19aed76c77c4186632c37` 最终文档/模型线；
- result：`ba651f0ebddd0983ceca5b352e65aa7ed5b7f32c`；
- composition：base compute/full-GEMM + component supplement；
- base interval `oc3_event_cnt +179` warning；
- component supplement OC increments 为 0。

FP16 N=2048 历史结果：

| 项目 | 数值 |
| --- | ---: |
| tc5a candidate median | 120.039 TFLOP/s |
| same-precision cuBLAS median | 130.633 TFLOP/s |
| candidate/reference | 91.89% |
| legacy empirical envelope | 128.436 TFLOP/s |
| candidate/legacy envelope | 93.46% |
| per-SM TMA makespan | 56.939 µs |
| legacy shared L2 read time | 133.762 µs |

在当时模型中，shared L2 read 是该 exact shape/schedule 的经验瓶颈。

## 2. 为什么不能作为 current envelope

历史 envelope 使用：

- 独立 `hbm.read` / `hbm.write` empirical points；
- 独立 `l2.read` / `l2.write` empirical points；
- legacy tc5a capacity alias；
- 未要求 exact causal profile；
- 较弱的 hardware/applicability schema。

current f06 模型改为：

- ratio-qualified `hbm.duplex` / `l2.duplex`；
- cold proxy 与 physical HBM duplex 分开；
- exact TMA family/stride；
- causal DAG/profile；
- hardware/mode/clock scope；
- 任一合法 schedule 缺数值时 manifest fail closed。

因此历史 `all_common_resources_closed=true` 只表示当时定义的独立公共资源完成，不表示 current joint/exact/all-precision closure。

## 3. 可继续引用的历史事实

以下事实仍受 raw artifacts 支持：

- 对应 candidate/reference trial 的完整 GEMM性能；
- 10-trial compute/component 中位数；
- exact tc5a FP16/BF16 stride2048 component measurement；
- SASS、binary hash、environment 和 OC warning；
- 审计器对当时 run contract 的通过状态。

引用时必须使用“historical”、“legacy envelope”或具体 result commit，不能写成 current f06 integrated envelope。

## 4. Legacy 文档

- [旧主文档](../../thor_sm110_gemm_performance_bounds.md)
- [旧单体教程](../../thor_sm110_gemm_performance_model_tutorial.md)
- [旧 current replay 名称的历史重放](../../thor_sm110_current_model_replay.md)
- [旧全精度矩阵](../../thor_sm110_all_precision_evidence_matrix.md)

这些文件保留链接兼容和历史审计；current 规范从 [GEMM 建模总入口](../README.md) 开始。
