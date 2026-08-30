# Phase 1：通用跨层 Codegen Harness 与六实例 Fresh Replay

## 阶段开始

证明固定 CUTLASS e05f953a5b3d38adc240df2ff928e0421c2abba3 中每个高层 Mainloop Schedule（59 个显式 Tensor Tag，加上按 11 个实现分组建立的 KernelScheduleAuto 控制项）如何经过 Builder、DispatchPolicy、Collective、Stage、Copy/Layout、TiledMMA 与 Atom，映射到同一 sm_110a 目标函数的 PTX/SASS；所有项目得到可审计的静态终态，静态证据绝不越级为 Thor 运行、数值或性能结论。

这一阶段先解决一个基础问题：什么样的证据，才足以把一个 CUTLASS 编译期实例写成 `STATIC_PASS`。固定输入是 CUTLASS `e05f953a5b3d38adc240df2ff928e0421c2abba3`、`cutlass::arch::Sm100` recipe、`compute_110a → sm_110a` target，以及 digest 固定的 CUDA 13.0 容器。阶段内没有 CUDA kernel launch。

每个实例只有一份 `config.hpp`，由它派生类型链和目标函数两条路径：

```text
Config
├─ MainloopBuilder / EpilogueBuilder
│  └─ type witness
│     └─ DispatchPolicy / Stage / Copy-Layout / TiledMMA / Atom
└─ GemmKernel
   └─ seeded dual-code FATBIN
      ├─ one compute_110a PTX entry
      └─ one sm_110a CUBIN STO_ENTRY
         └─ one exact-name nvdisasm function
```

目标函数先由唯一 PTX `.entry`、唯一 ELF `STO_ENTRY` 和唯一同名 nvdisasm function 绑定，之后才检查函数内 opcode。opcode 不参与选函数。

第一版 harness 可以覆盖六个既有实例，却把普通 Dense 的字段形状当成了所有 CUTLASS family 的共同形状。阶段因此重新打开，并用 PlanarComplex、FastFP32、MixedInput、普通 Sparse、Blockwise 和 mixed block-scaled pilot 检查通用性。最终合同同时记录：

- logical element/layout 与真正传入 Builder 的 element/layout；
- standard、array、grouped、MoE ProblemShape；
- static 与 dynamic Cluster type，以及固定的 default shape；
- classic copy、input/compute copy、block scale、blockwise scale、mixed scale 和 sparse metadata；
- main、Planar negative 和 scale-factor MMA role；
- mainloop、scheduler、accumulator、load-to-transform、transform-to-MMA、computation、transformation Stage；
- `SMEM_DESCRIPTOR`、`SPARSE_SMEM_DESCRIPTOR`、`TMEM_FRAGMENT` 三类实际 MMA operand source。

实例输入中原来手写的 `operand_source` 已删除。operand source 现在由 `TiledMma::FrgTypeA/B` 的真实编译类型得出，属于结果而不是声明。

## 正向验证

正式命令：

```text
python3 tools/run_codegen_v2.py \
  --root . \
  --all-phase1 \
  --run-id phase1-generalized-20260830
```

六个 canonical instance 都得到 fresh `STATIC_PASS`。下表只列 type witness 和同函数 PTX/SASS 中实际保存的值。

| Instance | 显式 Schedule Tag | Atom shape | MmaTile | Cluster | Stage：Main/Sched/Accum | MMA A/B source | 目标函数 MMA：PTX/SASS |
|---|---|---:|---:|---:|---:|---|---:|
| `dense_f16_1sm` | `KernelTmaWarpSpecialized1SmSm100` | 128×128×16 | 128×128×64 | 1×1×1 | 7/2/4 | SMEM / SMEM | 4/4 |
| `dense_f16_2sm` | `KernelTmaWarpSpecialized2SmSm100` | 256×128×16 | 256×128×64 | 2×1×1 | 8/2/4 | SMEM / SMEM | 4/4 |
| `dense_bs_nvfp4_1sm` | `KernelTmaWarpSpecialized1SmNvf4Sm100` | 128×128×64 | 128×128×256 | 1×1×1 | 6/2/2 | SMEM / SMEM | 4/4 |
| `dense_bs_mxf4_1sm` | `KernelTmaWarpSpecialized1SmMxf4Sm100` | 128×128×64 | 128×128×256 | 1×1×1 | 6/2/2 | SMEM / SMEM | 4/4 |
| `dense_bs_mxf8_1sm` | `KernelTmaWarpSpecialized1SmMxf8f6f4Sm100` | 128×128×32 | 128×128×128 | 1×1×1 | 6/2/2 | SMEM / SMEM | 4/4 |
| `sparse_bs_nvfp4_1sm` | `KernelSparseTmaWarpSpecialized1SmNvf4Sm100` | 128×128×128 | 128×128×256 | 1×1×1 | 6/2/2 | sparse SMEM / SMEM | 2/2 |

Sparse control 还解析出 A sparsity `2`、E sparsity `16`。三类 block-scaled control 分别解析出 UE4M3/SV16、UE8M0/SV32 和 UE8M0/SV32。所有实例同时满足：

- `MainloopBuilder::CollectiveOp == CollectiveMainloop`；
- `EpilogueBuilder::CollectiveOp == CollectiveEpilogue`；
- `Mainloop::DispatchPolicy`、`Mainloop::TiledMma`、`TiledMma::Atom` 的 producer/consumer view 相等；
- `Kernel::CollectiveMainloop`、`Kernel::CollectiveEpilogue` 与上游类型相等；
- PTX entry count、ELF `STO_ENTRY` count、nvdisasm exact-symbol match count 都是 1；
- CUBIN 的 nvdisasm JSON 都有 4 个函数：1 个目标函数和 3 个内部 guardrail；合同只检查目标函数。

## 对抗审查

证明固定 CUTLASS e05f953a5b3d38adc240df2ff928e0421c2abba3 中每个高层 Mainloop Schedule（59 个显式 Tensor Tag，加上按 11 个实现分组建立的 KernelScheduleAuto 控制项）如何经过 Builder、DispatchPolicy、Collective、Stage、Copy/Layout、TiledMMA 与 Atom，映射到同一 sm_110a 目标函数的 PTX/SASS；所有项目得到可审计的静态终态，静态证据绝不越级为 Thor 运行、数值或性能结论。

本阶段执行了“正向 fixture → 故意破坏 → 独立复核 → 修复 → 全量重跑”，没有把一次 runner 返回 0 当作完成。冻结时的专项结果是：

```text
ATTRIBUTION_V2_ADVERSARIAL_PASS cases=33 positive=5
CODEGEN_V2_MODEL_ADVERSARIAL_PASS mutations=84 positive=7
DEEP_REPLAY_V2_ADVERSARIAL_PASS mutations=3 positive=1
CODEGEN_V2_CAMPAIGN_ADVERSARIAL_PASS mutations=4 positive=2
SCHEDULE_TAG_ADVERSARIAL_PASS mutations=57 positive_helpers=2
```

84 个 model 反例除原有的跨函数拼接、错误 symbol、伪 journal、错误 Docker argv/mount、跨 attempt artifact、伪 FATBIN/CUBIN 和错误 Stage/Atom 外，又增加了以下 family-specific 破坏：

- required opcode 的 `min_count=0` 与 forbidden opcode 非零；
- Planar pair/negative role 为空或来自无关类型；
- FastFP32 用仅包含 `FastF32` 字样的伪 DispatchPolicy；
- MixedInput tuple arity、narrow/wide/scale/zero 类型与 swap 关系不一致；
- Blockwise granularity、major、ElementSF 和 `ScaleConfig` 五个模板参数不一致；
- Sparse metadata element、A/E sparsity、`SparseConfig` family 或 `.mma.sp` required contract 不一致；
- mixed block-scaled 的 `TiledMma_SF` 与 `Atom` 无关；
- `Shape<_N>`、subbyte、整数别名和 Fusion 默认模板参数被错误利用；
- MMA fragment type 与 operand-source 分类不一致。

正式结果随后执行：

```text
python3 tools/validate_codegen_v2.py \
  --root . \
  --require-results \
  --require-archive \
  --deep-replay
```

Deep replay 在新临时目录重新编译六个 type witness 和六个 FATBIN，重新提取 PTX/CUBIN，再从 CUBIN 重新生成 ELF symbols、code metadata 和 nvdisasm JSON/text。所有派生产物都与 sealed archive 逐字节一致：

```text
phase1_results=6
unsealed_results=0
result_contract_bundle_sha256=6046f6f498fff30e78b7adde3ee05bf76229491579c6370c8d99cd582630b02e
```

省略 `--run-id` 并执行 `--all-phase1 --resume` 时，runner 从冻结合同读取当前 run id，六项都
返回 `RESUMED_VERIFIED`，并再次通过完整 result/archive 校验；它不会回落到旧 campaign。

最后又从空的 `/tmp/sm110a-phase1-final.5VQYwP` 目录重新 configure/build，host CTest
`11/11` 全部通过；没有复用旧 `build-host`。

## 修复

证明固定 CUTLASS e05f953a5b3d38adc240df2ff928e0421c2abba3 中每个高层 Mainloop Schedule（59 个显式 Tensor Tag，加上按 11 个实现分组建立的 KernelScheduleAuto 控制项）如何经过 Builder、DispatchPolicy、Collective、Stage、Copy/Layout、TiledMMA 与 Atom，映射到同一 sm_110a 目标函数的 PTX/SASS；所有项目得到可审计的静态终态，静态证据绝不越级为 Thor 运行、数值或性能结论。

本轮实际发现并修复的主要问题如下：

1. 普通 Dense 专用的 `SmemCopyAtomA/B` 假设不能覆盖 FastFP32、MixedInput 和 InterleavedComplex。Reporter 改为 trait 分支，分别记录 classic copy 与 input/compute copy。
2. 单一 `TiledMma::Atom` 不能说明 Planar negative 或 mixed block-scale 的 scale-factor MMA。Reporter 和 validator 加入明确 role 及交叉绑定。
3. Blockwise 不是 hardware block-scaled。它现在单独记录 M/N/K granularity、SFA/SFB element/layout 和 major，不再复用 block-scale vector-size 字段。
4. 普通 Sparse 与 block-scaled Sparse 的 sparsity alias 不同。Reporter 同时处理 `ElementASparsity` 与 `ElementAMmaSparsity`，并把 E sparsity、metadata element、`SparseConfig` 一起绑定。
5. PtrArray/Grouped/MoE 的 Builder operand、pointer layout、ProblemShape 和 dynamic Cluster 不能压成普通 Dense 字段。Instance schema 已拆开 logical contract 与 Builder contract。
6. `typeid` demangle 会把 `cute::_N` 写成 `cute::C<N>`、`int4b_t` 展开成 `integer_subbyte`、`int8_t/int32_t` 展开为基础类型。Validator 只规范化已明确验证的 alias，不使用无边界 substring。
7. Fusion 默认模板参数不能无条件忽略。当前只对白名单 `LinearCombination` 按其真实五参数默认规则展开，伪造 C/Scalar/Round 会被拒绝。
8. 原实例中手填的 operand source 会把预期当成结果。该字段已删除，改由 MMA fragment type 分类。
9. 第一版六实例结果是在较窄合同下生成的。它们保留在 Git commit `5c19054` 中，没有被复用；本轮使用新 run id 从空 active evidence namespace 重新生成。
10. Phase 2 的 FastFP32 preflight 暴露了 compile-success 的架构 guard fallback。Harness 增加了
    `UNSUPPORTED_SM110A/ARCH_GUARD_FALLBACK` 终态及同环境 control；六个 Phase 1
    `STATIC_PASS` control 随新合同从空 namespace 重放，不能沿用旧 fingerprint。

两次中止的预运行只产生未 sealed 的 ignored archive，并在正式重放前移出 active namespace；没有 result、summary 或状态投影引用它们。

## 阶段完成

证明固定 CUTLASS e05f953a5b3d38adc240df2ff928e0421c2abba3 中每个高层 Mainloop Schedule（59 个显式 Tensor Tag，加上按 11 个实现分组建立的 KernelScheduleAuto 控制项）如何经过 Builder、DispatchPolicy、Collective、Stage、Copy/Layout、TiledMMA 与 Atom，映射到同一 sm_110a 目标函数的 PTX/SASS；所有项目得到可审计的静态终态，静态证据绝不越级为 Thor 运行、数值或性能结论。

阶段 1 的完成条件全部满足：

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

- current summary：`evidence/codegen-sm110a-v2/summary-phase1-generalized-20260830.json`，SHA-256 `be821150b2af8b857868b08ab20a2f87c7e2003d0cb6471d4764c54360a7e15d`；
- 6 份 result、6 份 fingerprint、6 份 artifact manifest、6 份 hash-chained journal；
- 24 份 Git 内 type/PTX/SASS/contract excerpt；
- 完整 archive 约 16 MiB，位于被忽略的 `artifacts/codegen-sm110a-v2/phase1-generalized-20260830/`，每个文件的 SHA-256 与 bundle checksum 已写入 manifest。

59 个显式 Tag 的当前投影是：

```text
STATIC_PASS: 6
HISTORICAL_STATIC_PASS: 0
NOT_CHECKED: 53
```

这只说明六个 canonical instance 的静态链路闭合。没有执行 Thor kernel，没有数值输入输出，没有 CUDA event/NCU，也没有延迟、吞吐或性能比较。剩余 53 个显式 Tag 和 11 个 Auto control 由后续阶段继续完成。
