# Thor/SM110 GEMM 性能建模总入口

本文档集回答一个明确问题：

> 在给定 GEMM 语义和 Thor/SM110 物理约束下，一个没有可避免性能浪费的经典稠密 GEMM，性能最多能到哪里；现有 microbenchmark 与完整 GEMM 又分别证明到了哪一层？

这里不把某个 `tc3`、`tc5a` 或 cuBLAS kernel 当作“所有 GEMM”的定义。模型把任何可证明的外上界、具体 schedule 的经验理想包络和已经观测到的完整 GEMM 分开报告。

## 1. 三个输出

定义 \(P_{\mathrm{obs}}\) 为语义匹配且通过 correctness 的完整 GEMM 已观测最好性能；定义 \(P^\star\) 为声明实现域中真实但未知的最优性能；定义 \(P_{\mathrm{ub}}\) 为当前假设和物理 rate upper 下的条件性能上界。只要 workload、实现域和上界假设一致，就必须满足：

\[
P_{\mathrm{obs}}\le P^\star\le P_{\mathrm{ub}}.
\]

定义 \(\widehat P_{\mathrm{env}}\) 为 microbenchmark 驱动的经验理想包络。它是具体 schedule 与已测容量的预测，不自动进入上式；如果完整 GEMM 超过它，应重校准经验模型，而不是宣布违反物理上界。

| 层 | 回答的问题 | 可使用的证据 |
| --- | --- | --- |
| 条件上界 | 任何合法实现都不能超过哪里 | `specified_upper`、`derived_upper`、带明确工具假设的 `profiler_model_peak` |
| 经验理想包络 | 当前合法 schedule 消除已知浪费后应达到哪里 | 同合同 `measured_sustained`、`measured_joint`、exact causal profile |
| 已观测最好值 | 现在真正运行并通过数值验证的最好完整 GEMM 是多少 | 完整 candidate/reference trial、correctness、SASS、环境与审计工件 |

## 2. 当前证据快照

本页当前基线：

- 模型代码：`f06f2cd917a4cb23806b5e1be06120be9152ed7b`
- parameter supplement 采集代码：`0c42cbb7987e204a2c8f78f17e4cce0096fbdef0`
- parameter supplement GPU 数据：`aa845dd9e70e2c541ae3a7d5293bf8de4bd55092`
- parameter supplement 结果分支：`thor-results/thor-t5000-parameter-plots-maxn-20260817-i`
- 目标完成状态：`complete=false`

| 证据面 | 当前状态 | 不能据此声称什么 |
| --- | --- | --- |
| 12 精度 full-SM compute surface | runner 定义完整，36 个 full-SM shape 合同 | measured compute rate 不是物理 rate upper |
| 公共 component surface | 定义完整 | 独立 component peak 不证明 joint attainability |
| TMA payload surface | 4/8/16/32/64 KiB × hot/cold 已测 | 尚不覆盖 block-scale 512 B/1 KiB scale request |
| hot-L2 duplex | 当前所需 ratio 已测 | 不证明外部 DRAM write bytes |
| cold memory proxy | 当前所需 ratio 的 external-read/L2-write-path proxy 已测 | `external_write_bytes_proven=false`，不能导入为 physical `hbm.duplex` |
| exact TMA topology | 2/28 schedule/precision pair | 不能把 tc5a FP16/BF16 capacity 借给其它 topology |
| causal pipeline | 求解器存在，tc5a FP16/BF16 runner 已冻结；Thor profile 尚无 | 不能输出普遍 joint causal envelope |
| 完整 GEMM runner | 6/12 precision path 已定义 | 不能称全部精度已闭合 |
| 严格 compute upper | 约 7/12 precision 有条件证据 | 其余精度只能由其它资源给出更松的 partial upper |

上述“缺失”是模型输出的一部分。严谨文档不要求所有门禁都是 true，但要求 false 的原因、作用域和需要的下一条证据都能机械追踪。

## 3. 正式模型章节

建议按以下顺序阅读；每章只承担一个建模责任。

```text
gemm/
├── README.md
├── model/        # 规范公式、假设、算法与当前 coverage
├── experiments/  # 每个 campaign/物理问题的短实验合同
├── appendices/   # schema、来源、current replay、历史与复现
└── tutorial/     # worked example、反例与学习路径
```

1. [范围、主张与证据边界](model/01_scope_and_claims.md)
2. [符号、单位、workload 与硬件作用域](model/02_symbols_units_and_workload.md)
3. [工作量与数据流计账](model/03_work_accounting.md)
4. [条件可证明性能上界](model/04_strict_performance_upper.md)
5. [经验资源包络](model/05_empirical_resource_envelope.md)
6. [因果流水线模型](model/06_causal_pipeline_model.md)
7. [完整 GEMM、反证与重校准](model/07_observed_gemm_and_falsification.md)
8. [当前覆盖与缺口](model/08_current_coverage_and_gaps.md)

模型论证链固定为：

```text
目标与范围
  → 符号、单位和作用域
  → 任何经典 GEMM 不可避免的最低工作量
  → 最低工作量 / 物理 rate upper
  → 时间下界与条件性能上界
  → 具体 schedule 的 issued work
  → ratio/topology 匹配的经验容量
  → causal DAG 与最慢 worker makespan
  → 经验理想包络
  → 完整 GEMM observation
  → 反证、重校准和证据缺口
```

## 4. 实验合同

实验文档以一个物理问题或一个 campaign family 为粒度；一条 case 不单独建文档。

| ID | 实验 | 进入模型的内容 |
| --- | --- | --- |
| EXP-01 | [12 精度 compute surface](experiments/EXP-01-compute-surface.md) | shape-qualified empirical compute rate；部分产品级 conditional upper 的来源交叉检查 |
| EXP-02 | [L2 物理边界](experiments/EXP-02-l2-physical-bounds.md) | GPU-wide `l2.read` / `l2.write` 条件上界与 L2 capacity |
| EXP-03 | [TMA payload surface](experiments/EXP-03-tma-payload-surface.md) | payload/residency-qualified TMA service surface |
| EXP-04 | [memory duplex surface](experiments/EXP-04-memory-duplex-surface.md) | ratio-qualified `l2.duplex` 与 `hbm.duplex.proxy` |
| EXP-05 | [exact TMA topology](experiments/EXP-05-exact-tma-topology.md) | schedule/precision/stride/topology-qualified TMA capacities |
| EXP-06 | [TMEM readback 与 scale](experiments/EXP-06-tmem-readback-and-scale.md) | `tmem.readback.*`、`tmem.scale_ingress` 与 epilogue diagnostics |
| EXP-07 | [causal pipeline](experiments/EXP-07-causal-pipeline.md) | exact persistent-worker timing profile |
| EXP-08 | [完整 GEMM validation](experiments/EXP-08-full-gemm-validation.md) | candidate/reference observation、correctness、calibration/holdout |

每份实验文档固定回答：研究问题、模型参数、case matrix、计时与工作量、接受门禁、当前结果、允许进入的模型层、明确不能证明的内容、源码与工件。

## 5. 附录

- [性能与带宽资源总表（最终版）](FINAL_PERFORMANCE_BANDWIDTH_REPORT.md)
- [可执行 schema reference](appendices/schema_reference.md)
- [microbenchmark 与完整 GEMM 来源](appendices/microbenchmark_sources.md)
- [当前模型重放状态](appendices/current_model_replay.md)
- [历史结果及模型收紧记录](appendices/historical_results.md)
- [审计与复现命令](appendices/audit_and_reproduction.md)

## 6. 教程

[教程入口](tutorial/README.md) 按学习顺序链接正式章节和实验。旧的 4656 行单体教程暂时保留为 legacy 教学材料，但不再作为当前公式或当前证据状态的规范来源。

## 7. 文档职责

| 文档类型 | 负责什么 | 不负责什么 |
| --- | --- | --- |
| `model/` | 规范定义、公式、假设、算法与反证规则 | runner 操作日志和逐 case 原始结果 |
| `experiments/` | 测量合同、门禁、结果作用域和 non-claims | 重新定义主模型公式 |
| `appendices/` | schema、来源、当前生成状态、历史和复现 | 改变模型语义 |
| `tutorial/` | 教学顺序、练习和直觉 | 作为唯一事实源 |
| `microbench/**/README.md` | 启动、恢复、审计与结果提交操作 | 给 microbenchmark rate 赋予超出实验合同的物理含义 |

任何当前数值结论都必须能沿以下链回溯：

```text
模型参数
  → experiment ID
  → runner/auditor
  → run ID 与 expected commit
  → raw trial / NCU / SASS / environment
  → imported Capacity 或 ObservedBest
  → coverage / target-completion 输出
```

无法完成这条回溯时，模型应输出 `insufficient_evidence` 或保留历史/诊断标签，而不是补一个邻近数字。
