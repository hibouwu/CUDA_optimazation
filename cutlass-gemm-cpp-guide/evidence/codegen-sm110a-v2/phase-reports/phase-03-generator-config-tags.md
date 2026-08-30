# 阶段 3：从 generator 配置闭合到函数级 codegen

## 阶段开始

证明固定 CUTLASS e05f953a5b3d38adc240df2ff928e0421c2abba3 中每个高层 Mainloop Schedule（59 个显式 Tensor Tag，加上按 11 个实现分组建立的 KernelScheduleAuto 控制项）如何经过 Builder、DispatchPolicy、Collective、Stage、Copy/Layout、TiledMMA 与 Atom，映射到同一 sm_110a 目标函数的 PTX/SASS；所有项目得到可审计的静态终态，静态证据绝不越级为 Thor 运行、数值或性能结论。

阶段 3 的分母不是所有含有 Python 枚举的 Tag，而是固定来源清单中 5 个确有完整
`generator.py` 配置的 Tag。每条实例同时绑定 generator 配置行与 `library.py` 的
enum→C++ Tag 映射行；缺其中一跳不能称为 generator 配置闭环。

| canonical instance | Schedule Tag | 固定配置 | 预期风险 |
|---|---|---|---|
| `interleaved_complex_tf32_1sm_generator` | `KernelTmaWarpSpecialized1SmInterleavedComplexTF32Sm100` | complex F32，A Row/B Col，`128×64×16`，cluster `1×1×1` | TF32 input-transform |
| `fast_fp32_complex_1sm_generator` | `KernelTmaWarpSpecialized1SmFastFP32Sm100` | complex F32，裸 Builder input，`128×64×16`，cluster `1×1×1` | scaled BF16 guard |
| `interleaved_complex_tf32_2sm_generator` | `KernelTmaWarpSpecialized2SmInterleavedComplexTF32Sm100` | complex F32，A Row/B Col，`256×64×16`，cluster `2×1×1` | 2CTA TF32 input-transform |
| `fast_fp32_complex_2sm_generator` | `KernelTmaWarpSpecialized2SmFastFP32Sm100` | complex F32，裸 Builder input，`256×64×16`，cluster `2×1×1` | 2CTA scaled BF16 guard |
| `dense_bs_mxf4_2sm_generator` | `KernelTmaWarpSpecialized2SmMxf4Sm100` | E2M1×E2M1、UE8M0/SV32，`256×128×256`，NoSmem Epilogue | 2CTA MXF4 |

这里固定 Default scheduler、identity transform 和静态 cluster。Generator 还会枚举 layout、
conjugate、C-null、Stream-K 和 dynamic cluster；这些组合不在一条 canonical instance 的
覆盖范围内。

## 数据与结果

5 个正式实例得到 3 个 `STATIC_PASS` 和 2 个 `UNSUPPORTED_SM110A`：

| instance | 终态 | resolved Atom / operand source | Stage | 同函数 codegen |
|---|---|---|---|---|
| Interleaved TF32 1SM | `STATIC_PASS` | `SM100_MMA_TF32_TS_INTERLEAVED...`，TMEM/SMEM | Mainloop 8，Computation 8，Transformation 4 | 2 STTM，8 UTCHMMA |
| Fast complex 1SM | `UNSUPPORTED_SM110A` | `SM100_MMA_F16BF16_TS_SCALED`，TMEM/SMEM | Mainloop 4，Load→Transform 4，Transform→MMA 4 | 36 trap，6 STTM，0 MMA |
| Interleaved TF32 2SM | `STATIC_PASS` | `SM100_MMA_TF32_2x1SM_TS_INTERLEAVED...`，TMEM/SMEM | Mainloop 10，Computation 10，Transformation 4 | 2 STTM，8 UTCHMMA.2CTA |
| Fast complex 2SM | `UNSUPPORTED_SM110A` | `SM100_MMA_F16BF16_2x1SM_TS_SCALED`，TMEM/SMEM | Mainloop 8，Load→Transform 8，Transform→MMA 5 | 36 trap，6 STTM，0 MMA |
| MXF4 2SM | `STATIC_PASS` | `SM100_MMA_MXF4_2x1SM_SS<...,UE8M0,...,32>`，SMEM/SMEM | `8/2/2` | 4 UTCOMMA.2CTA，4 UTCCP，2 LDTM |

Fast generator 的 Builder A/B 必须保持裸 `cutlass::complex<float>`。它先命中 compatibility
specialization，再由 resolved Mainloop 证明内部 TransformA/B 是 `cute::identity`；若为了
方便直接把 Config 改成 tuple，就不再是 generator emitter 的真实入口。Interleaved TF32
则相反，它的 Builder A/B 本来就是 `tuple<complex<float>, identity>`。

两个 Fast complex 实例的类型链、FATBIN、CUBIN、sole PTX entry、sole ELF `STO_ENTRY` 和
唯一同名 nvdisasm function都成功。失败只发生在 `FUNCTION_CONTRACT`：固定 SM110A 配置没有
定义 `CUTE_ARCH_TCGEN05_F16BF16_MMA_SCALED_ENABLED`，所以目标函数走 guard fallback。
结论只属于这两个 generator canonical instance；同一 public Tag 还存在 plain-float Builder
偏特化，不能据此宣布整个 Tag 类型域不支持。

Interleaved 2SM 的 MMA 必须严格为 2CTA，但 input-transform Builder 不能使用 2SM TMA load，
所以实际路径是 1CTA TMA 加 multicast。合同只对这一 mechanism 放宽 TMA load 的 CTA 形式，
没有放宽 MMA 的 `cta_group::2` / `.2CTA` 要求。

## 最终 harness 迁移

Phase 3 同时完成了一次性的语义迁移，避免 Phase 4/5 每新增一批实例就让旧 fingerprint失效：

- 具体运行批次移到 `tests/codegen/run-campaigns/*.json`，不进入 codegen fingerprint；summary
  用 sealed `campaign_ref` 绑定 member顺序与允许终态。
- static contract 不再硬编码 `--all-phaseN`。Runner 使用 `--campaign <id>`，validator 使用可
  重复的 `--require-campaign <id>`。
- architecture guard 改成 profile：实数 2SM Smem 是 9 trap；complex 1SM/2SM TS 各是
  36 trap，并各自绑定 Atom、macro、源码区间、精确函数计数和 legal control。
- 为 Phase 4/5 预先冻结 `EXPECTED_STATIC_REJECT` 的 stop-at-first-failure合同：返回码、诊断、
  infrastructure denylist、源码摘录、部分 journal/artifact、legal control 和 deep replay均有
  独立门禁。当前阶段没有借此提前升级任何 source-derived 或 Auto 状态。

这次迁移改变了 runner/model/schema/attribution，因此此前 40 份基线按设计失效。旧证据与
完整 archive分别移到 `/tmp` 备份；随后 Phase 1、Phase 2、Phase 3 control 和 Phase 3 main
全部在最终 bundle `b4d8852e9e962cb57dd718d45709272ef47cd6c228866f917e40be3edc6fda64`
下 clean replay。

## 对抗审查

证明固定 CUTLASS e05f953a5b3d38adc240df2ff928e0421c2abba3 中每个高层 Mainloop Schedule（59 个显式 Tensor Tag，加上按 11 个实现分组建立的 KernelScheduleAuto 控制项）如何经过 Builder、DispatchPolicy、Collective、Stage、Copy/Layout、TiledMMA 与 Atom，映射到同一 sm_110a 目标函数的 PTX/SASS；所有项目得到可审计的静态终态，静态证据绝不越级为 Thor 运行、数值或性能结论。

本阶段故意破坏了 generator factory、双 provenance anchor、Fast 裸 compatibility input、
complex mechanism、TF32 MMA/STTM 精确计数、MXF4 kind/SV32、guard profile、whole-binary trap
污染、campaign hash/member/status、拒绝返回码/诊断/infrastructure、PTX/SASS函数归属和
artifact派生关系。最终轻量门禁为：

```text
PHASE3_GENERATOR_GENERATION_ADVERSARIAL_PASS mutations=13 positive=5
CODEGEN_V2_MODEL_ADVERSARIAL_PASS mutations=103 positive=15
EXPECTED_STATIC_REJECT_ADVERSARIAL_PASS mutations=20 positive=3
CODEGEN_V2_CAMPAIGN_ADVERSARIAL_PASS mutations=13 positive=1
ATTRIBUTION_V2_ADVERSARIAL_PASS cases=34 positive=7
DEEP_REPLAY_V2_ADVERSARIAL_PASS mutations=3 positive=1
SCHEDULE_TAG_ADVERSARIAL_PASS mutations=57 positive_helpers=2
REPLAY_ATTESTATION_PASS records=1
REPLAY_ATTESTATION_ADVERSARIAL_PASS mutations=3 positive=1
```

完整 deep replay 还从源码重新生成 47 个 attempt 的所有决定性产物。机器可读 attestation：

`evidence/codegen-sm110a-v2/replay-attestations/phase3-final-deep-replay.json`

其结果是 44 个 `STATIC_PASS` attempt、3 个 `UNSUPPORTED_SM110A` attempt、0 个未封存结果。
其中 2 个额外 pass 是 Fast guard 引用的 Interleaved control；显式 Tag 分母仍是 45。

## 修复

证明固定 CUTLASS e05f953a5b3d38adc240df2ff928e0421c2abba3 中每个高层 Mainloop Schedule（59 个显式 Tensor Tag，加上按 11 个实现分组建立的 KernelScheduleAuto 控制项）如何经过 Builder、DispatchPolicy、Collective、Stage、Copy/Layout、TiledMMA 与 Atom，映射到同一 sm_110a 目标函数的 PTX/SASS；所有项目得到可审计的静态终态，静态证据绝不越级为 Thor 运行、数值或性能结论。

正式执行不是一次通过，至少修复了四个实质问题：

1. `tcgen05.st...b32[%r]` 的 opcode 后直接跟 `[`；旧 parser 不认这个 token终止符，导致
   PTX store假阴性。Parser与正例同步修复。
2. Interleaved 2SM 使用1CTA TMA+multicast，但MMA仍是2CTA。CTA合同拆成严格MMA group与
   instance-specific TMA policy，避免全局放宽。
3. Fast complex 实例不能复用 Phase 2 的 9-trap SS profile；1SM/2SM TS分别冻结36-trap
   profile，并拒绝把3个helper trap计入目标函数。
4. MXF4 pilot 的4条LDTM来自另一Epilogue分支；正式NoSmem canonical instance实际是2条。
   最终合同按同一Config、同一目标函数重新计数为MMA/copy/load `4/4/2`。
5. Schedule 对抗 fixture 最初只复制显式 Tag result，没有复制 guard control 的 sealed
   run-campaign/summary graph。Fixture 改为遍历所有当前 campaign 与 summary result graph，
   clean positive和57个破坏项才在同一证据模型下通过。
6. 终审发现每个 attempt 的 `full/type_witness.json` 是 tracked type excerpt 的未封存副本。
   Runner 现在删除这个临时副本，把 in-progress journal 移出 sealed archive；validator 同时要求
   archive 实际文件集合与 manifest 完全相等，并拒绝 orphan result、fingerprint、manifest、
   journal、excerpt、attempt、symlink 与非目录 attempt 节点。四个 campaign 随最终合同重新
   fresh replay，不能在旧 manifest 上补 hash。

每次修复后都从受影响的 clean campaign 重新执行，最后再从头 deep replay 全部47份结果。
从新的 `/tmp/sm110a-phase3-archiveclosure-ctest.nB9Xw3` 目录 fresh configure/build 后，
host CTest `16/16` 全部通过。

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

- Phase 3 main summary：`summary-phase3-generator-20260830.json`，SHA-256
  `b377ffb654c2d7b740c8a3015080e4bec22edc0febc1ac323e272caa40160522`；
- Phase 3 control summary：`summary-phase3-generator-controls-20260830.json`，SHA-256
  `3f7465bf828a1a22546a83a4f7ba42e12f6021e7eb39a043445c007d48250f56`；
- Phase 3 main 5 result / 5 manifest / 5 journal / 20 Git excerpt；control 2 result / 2 manifest /
  2 journal / 8 Git excerpt；
- 完整 archive：main 约 25 MiB、165 个文件；control 约 11 MiB、66 个文件；两者实际文件
  集合均与 manifest 逐项相等；
- 45个显式Tag当前投影：`STATIC_PASS=42`、`UNSUPPORTED_SM110A=3`、
  `NOT_CHECKED=14`；11个Auto control仍为`NOT_CHECKED`；
- 本阶段没有执行Thor kernel、数值计算或性能测量。
