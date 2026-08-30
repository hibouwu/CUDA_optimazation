# Phase 2：34 个 Official C++ Schedule Tag

## 阶段开始

证明固定 CUTLASS e05f953a5b3d38adc240df2ff928e0421c2abba3 中每个高层 Mainloop Schedule（59 个显式 Tensor Tag，加上按 11 个实现分组建立的 KernelScheduleAuto 控制项）如何经过 Builder、DispatchPolicy、Collective、Stage、Copy/Layout、TiledMMA 与 Atom，映射到同一 sm_110a 目标函数的 PTX/SASS；所有项目得到可审计的静态终态，静态证据绝不越级为 Thor 运行、数值或性能结论。

Phase 0 找到 39 个能从固定 CUTLASS C++ test/example 直接构造的 Tag；其中 5 个已经作为
Phase 1 control 重放。本阶段处理剩余 34 个，每个 Tag 只选一个官方 canonical point。一个
Tag 在其他类型、Tile、Cluster 或 Fusion 下的组合域不由这个点自动覆盖。

34 个配置分成两份可审计 spec：non-blockscale 15 项，block-scaled/sparse 19 项。Generator
把 spec 确定性展开成 `config.hpp`、唯一 `device_kernel` TU、type-witness TU 和 instance record。
所有实例仍固定 `cutlass::arch::Sm100` recipe 与 `compute_110a/sm_110a` binary target。

## 正向结果与数据

正式命令：

```text
python3 tools/run_codegen_v2.py --root . --all-phase2
```

34 个实例全部得到允许的正式终态：33 个 `STATIC_PASS`，1 个
`UNSUPPORTED_SM110A/ARCH_GUARD_FALLBACK`；没有 `UNEXPECTED_COMPILE_FAIL`、
`ATTRIBUTION_FAIL` 或遗漏。

| 实现分组 | 实例数 | `STATIC_PASS` | `UNSUPPORTED_SM110A` | Mainloop Stage 范围 |
|---|---:|---:|---:|---:|
| Dense | 3 | 3 | 0 | 12–24 |
| Pointer-array Dense | 2 | 2 | 0 | 5–7 |
| Blockwise | 4 | 4 | 0 | 5–7 |
| Planar Complex | 4 | 4 | 0 | 2–8 |
| Fast FP32 / 9xBF16 | 1 | 0 | 1 | 8 |
| Mixed-input | 1 | 1 | 0 | 6 |
| Dense Block-scaled | 6 | 6 | 0 | 3–8 |
| Pointer-array Block-scaled | 6 | 6 | 0 | 2–7 |
| 普通 Sparse | 2 | 2 | 0 | 11–14 |
| Sparse Block-scaled | 5 | 5 | 0 | 3–10 |

ProblemShape 分布为 dense 18、array 8、grouped 4、MoE 4。MMA fragment 的实际 operand
source 为：SMEM/SMEM 26 项、sparse-SMEM/SMEM 7 项、TMEM/SMEM 1 项。最后一项是
MixedInput ConvertOnly；logical A/B 为 BF16/INT4，Builder 交换后实际使用
`tuple<INT4>`/BF16，tuple arity 为 1。

这些数据还说明 Stage 不能从 Schedule 名称机械推出。同为 Planar family，四个点的 Mainloop
Stage 分别是 6、4、2、8；Dense mixed TMA+cp.async 的 1SM/2SM 是 12/24；Sparse
block-scaled 五点分布为 7、10、3、5、9。它们是固定 Epilogue carveout、Tile、Cluster 和
数据路径共同作用后的解析值，不是性能排名。

## FastFP32 的架构 Guard

`KernelTmaWarpSpecialized2SmFastFP32SmemSm100` 的类型链完整解析到
`SM100_MMA_F16BF16_2x1SM_SS_SCALED`，并生成唯一 PTX entry、唯一 ELF `STO_ENTRY` 与同名
SASS function。编译和全部派生命令都返回 0，但该目标函数没有 Tensor MMA：

```text
PTX target:  brkpt    = 9, tcgen05.mma = 0
SASS target: BPT.TRAP = 9, UTC*MMA     = 0
```

固定 CUTLASS 只在 SM100A/SM103A 路径定义
`CUTE_ARCH_TCGEN05_F16BF16_MMA_SCALED_ENABLED`；SM110A 路径没有该 macro，Atom wrapper
进入 `CUTE_INVALID_CONTROL_PATH`。因此 Builder 到 `FUNCTION_BINDING` 都是 `RESOLVED`，只有
`FUNCTION_CONTRACT` 为 `REJECTED`。同一合同、同一 CUTLASS、同一容器和同一 target 的
`dense_f16_2sm` fresh `STATIC_PASS` 是 legal control。

完整 nvdisasm 另有 3 个内部 helper，每个含 1 个 trap；全文件共 12 个。正式结果仍为目标
函数内 9 个，证明 helper 没有污染归因。`UNSUPPORTED_SM110A` 只适用于这个 canonical
FastFP32 实例，不能外推该 public Tag 的全部 Builder 偏特化。

## 对抗审查

证明固定 CUTLASS e05f953a5b3d38adc240df2ff928e0421c2abba3 中每个高层 Mainloop Schedule（59 个显式 Tensor Tag，加上按 11 个实现分组建立的 KernelScheduleAuto 控制项）如何经过 Builder、DispatchPolicy、Collective、Stage、Copy/Layout、TiledMMA 与 Atom，映射到同一 sm_110a 目标函数的 PTX/SASS；所有项目得到可审计的静态终态，静态证据绝不越级为 Thor 运行、数值或性能结论。

本阶段的正反例结果：

```text
PHASE2_OFFICIAL_GENERATION_ADVERSARIAL_PASS mutations=8 positive=34
CODEGEN_V2_MODEL_ADVERSARIAL_PASS mutations=99 positive=13
ATTRIBUTION_V2_ADVERSARIAL_PASS cases=33 positive=5
DEEP_REPLAY_V2_ADVERSARIAL_PASS mutations=3 positive=1
CODEGEN_V2_CAMPAIGN_ADVERSARIAL_PASS mutations=4 positive=2
SCHEDULE_TAG_ADVERSARIAL_PASS mutations=57 positive_helpers=2
```

Phase 2 专项红队还故意制造了：Fast target/helper trap 混淆、伪造 trap count、目标函数仍有
MMA、替换成另一段 hash 正确但错误的源码范围、historical/non-pass/cross-environment/multiple
control、summary 漏项/重排、普通实例强翻 unsupported、Fast 强翻 static pass，以及 archive
缺失的 offline replay。所有错误都在预期门禁被拒绝。

## 修复

证明固定 CUTLASS e05f953a5b3d38adc240df2ff928e0421c2abba3 中每个高层 Mainloop Schedule（59 个显式 Tensor Tag，加上按 11 个实现分组建立的 KernelScheduleAuto 控制项）如何经过 Builder、DispatchPolicy、Collective、Stage、Copy/Layout、TiledMMA 与 Atom，映射到同一 sm_110a 目标函数的 PTX/SASS；所有项目得到可审计的静态终态，静态证据绝不越级为 Thor 运行、数值或性能结论。

本阶段不是一次生成就通过。实际发现并修复了这些问题：

1. CUDA Core `cp.async` Mainloop 的完整 kernel 仍可能因 Epilogue 出现 TMA。函数合同改为要求
   classic `cp.async/LDGSTS`，Mainloop 是否用 TMA 由 resolved copy type 判断，不再错误地全函数
   禁止 TMA。
2. PtrArray unit test 的 Builder layout tag 不带 `*`，pointer-array 语义来自
   `ArrayProblemShape` 与 PtrArray schedule；只有 grouped NVF4/Blockwise 配置按官方源码使用
   pointer layout。Spec、Config 和文档已分开表达这两种路径。
3. PtrArray Planar 需要显式包含 array-planar Epilogue specialization；补齐实体 header 后，
   positive/negative TiledMMA pair 才能构造。
4. Grouped Blockwise collective没有 `ElementSFA/B` alias，scale value 实际采用
   `ElementAccumulator=float`。Reporter 增加这一 specialization，同时继续验证 LayoutSF、major
   与 granularity。
5. Sparse Mxf8f6f4 的 E sparsity 实际为 8，不是按其他 block-scaled sparse 路径类推的 16；
   spec 以 type witness 和固定 source 条件修正。
6. Sparse block-scaled 的 `PerRowLinCombPerRowBiasEltAct` 会展开 8、4 和 rounding 默认参数；
   validator只对白名单模板按真实默认规则展开，不无条件忽略尾参。
7. FastFP32 compile-success trap 不能写成普通 contract failure，也不能写成 `STATIC_PASS`。
   Harness 增加有 source range、同函数 trap、MMA absence 和 fresh control 的严格
   `ARCH_GUARD_FALLBACK` 终态。
8. Phase 2 首次切换为 `COMPLETE` 后，schedule 对抗测试的 clean fixture 因只复制 Phase 0/1
   报告而被 fail-closed 拒绝。Fixture 改为从 campaign 合同逐项复制所有已完成阶段的报告，
   避免以后新增阶段时再次漏掉 completion evidence；修复后本阶段全部反例从头重跑。

## 阶段完成

证明固定 CUTLASS e05f953a5b3d38adc240df2ff928e0421c2abba3 中每个高层 Mainloop Schedule（59 个显式 Tensor Tag，加上按 11 个实现分组建立的 KernelScheduleAuto 控制项）如何经过 Builder、DispatchPolicy、Collective、Stage、Copy/Layout、TiledMMA 与 Atom，映射到同一 sm_110a 目标函数的 PTX/SASS；所有项目得到可审计的静态终态，静态证据绝不越级为 Thor 运行、数值或性能结论。

```text
positive_validation: PASS
deliberate_breakage: PASS
adversarial_audit: PASS
repair: PASS
full_phase_rerun: PASS
deep_replay: PASS
unexplained_failures: 0
omissions: 0
evidence_promotions: 0
```

持久化结果：

- Phase 2 summary：`evidence/codegen-sm110a-v2/summary-phase2-official-20260830.json`，
  SHA-256 `ce2d7f21973aa335a7bc151f9bab8a3cff294c64d6999f4737efe25722c27c2e`；
- 34 result、34 fingerprint、34 artifact manifest、34 hash-chained journal；
- 136 份 Git 内 type/PTX/SASS/contract excerpt；
- 完整 Phase 2 archive 约 151 MiB，1156 个文件，位于被忽略的
  `artifacts/codegen-sm110a-v2/phase2-official-20260830/`；
- result-contract bundle：`6046f6f498fff30e78b7adde3ee05bf76229491579c6370c8d99cd582630b02e`。

59 个显式 Tag 的当前投影为：`STATIC_PASS=39`、`UNSUPPORTED_SM110A=1`、
`NOT_CHECKED=19`。本阶段没有执行 Thor kernel、数值计算或性能测量。
