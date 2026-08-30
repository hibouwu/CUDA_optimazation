# Phase 1：跨层 Codegen Harness 与六实例 Fresh Replay

## 阶段开始

证明固定 CUTLASS e05f953a5b3d38adc240df2ff928e0421c2abba3 中每个高层 Mainloop Schedule（59 个显式 Tensor Tag，加上按 11 个实现分组建立的 KernelScheduleAuto 控制项）如何经过 Builder、DispatchPolicy、Collective、Stage、Copy/Layout、TiledMMA 与 Atom，映射到同一 sm_110a 目标函数的 PTX/SASS；所有项目得到可审计的静态终态，静态证据绝不越级为 Thor 运行、数值或性能结论。

这一阶段先不扩展 59 个 Tag。先把“一个声明实例如何成为一份可信的函数级静态结论”做实，再重放 Phase 0 中仅有历史证据的 6 个 Tag。固定输入是 CUTLASS `e05f953a5b3d38adc240df2ff928e0421c2abba3`、`cutlass::arch::Sm100` recipe、`compute_110a → sm_110a` target 和 digest 固定的 CUDA 13.0 容器。整个阶段没有 CUDA kernel launch。

Harness 冻结在本地 commit `1482612`。每个实例由同一份 `config.hpp` 派生两条链：

```text
Config
├─ MainloopBuilder / EpilogueBuilder
│  └─ type_witness → resolved types / shapes / stages
└─ GemmKernel
   └─ one seeded dual-code FATBIN
      ├─ one compute_110a PTX entry
      └─ one sm_110a CUBIN STO_ENTRY
         └─ one exact-name nvdisasm function
```

函数先由唯一 PTX `.entry`、唯一 ELF `STO_ENTRY` 和唯一同名 nvdisasm function 绑定，再检查该函数的 opcode；opcode 从不参与选函数。

## 正向结果

正式命令：

```text
python3 tools/run_codegen_v2.py \
  --root . \
  --all-phase1 \
  --run-id phase1-fresh-20260830
```

六个 canonical instance 都得到 fresh `STATIC_PASS`。下表中的 Atom shape、Stage 和计数都来自提交的 type/function excerpt，而不是从 Tag 名推断。

| Instance | 显式 Schedule Tag | Atom shape | MmaTile | Cluster | Stage：Main/Sched/Accum | 机制字段 | 目标函数内 MMA：PTX/SASS |
|---|---|---:|---:|---:|---:|---|---:|
| `dense_f16_1sm` | `KernelTmaWarpSpecialized1SmSm100` | 128×128×16 | 128×128×64 | 1×1×1 | 7/2/4 | SS，CTA group 1 | 4/4 |
| `dense_f16_2sm` | `KernelTmaWarpSpecialized2SmSm100` | 256×128×16 | 256×128×64 | 2×1×1 | 8/2/4 | SS，CTA group 2 | 4/4 |
| `dense_bs_nvfp4_1sm` | `KernelTmaWarpSpecialized1SmNvf4Sm100` | 128×128×64 | 128×128×256 | 1×1×1 | 6/2/2 | UE4M3，SV16 | 4/4 |
| `dense_bs_mxf4_1sm` | `KernelTmaWarpSpecialized1SmMxf4Sm100` | 128×128×64 | 128×128×256 | 1×1×1 | 6/2/2 | UE8M0，SV32 | 4/4 |
| `dense_bs_mxf8_1sm` | `KernelTmaWarpSpecialized1SmMxf8f6f4Sm100` | 128×128×32 | 128×128×128 | 1×1×1 | 6/2/2 | UE8M0，SV32 | 4/4 |
| `sparse_bs_nvfp4_1sm` | `KernelSparseTmaWarpSpecialized1SmNvf4Sm100` | 128×128×128 | 128×128×256 | 1×1×1 | 6/2/2 | UE4M3，SV32，A sparsity 2 | 2/2 |

六个实例还同时满足：

- `MainloopBuilder::CollectiveOp == CollectiveMainloop`；
- `EpilogueBuilder::CollectiveOp == CollectiveEpilogue`；
- `Mainloop::DispatchPolicy`、`Mainloop::TiledMma`、`TiledMma::Atom` 的 consumer view 与独立输出相等；
- `Kernel::CollectiveMainloop`、`Kernel::CollectiveEpilogue` 与上游 resolved 类型相等；
- PTX entry count、ELF STO_ENTRY count、nvdisasm exact-symbol match count 均为 1；
- 每份 CUBIN 的 nvdisasm JSON 共含 4 个函数：1 个目标函数和 3 个内部 guardrail；合同只检查目标函数。

Stage 输入和解析结果分开保存。Dense/普通 block-scaled 的 resolved `StageCountAutoCarveout<N>` 中，`N` 必须等于 resolved Epilogue SharedStorage 字节数；Sparse 的 `StageCountAutoCarveoutEpi<...>` 必须真正绑定同一个 resolved Epilogue 类型。

## 对抗审查

证明固定 CUTLASS e05f953a5b3d38adc240df2ff928e0421c2abba3 中每个高层 Mainloop Schedule（59 个显式 Tensor Tag，加上按 11 个实现分组建立的 KernelScheduleAuto 控制项）如何经过 Builder、DispatchPolicy、Collective、Stage、Copy/Layout、TiledMMA 与 Atom，映射到同一 sm_110a 目标函数的 PTX/SASS；所有项目得到可审计的静态终态，静态证据绝不越级为 Thor 运行、数值或性能结论。

本阶段没有把一次 runner 成功当作完成。独立检查包括：

- attribution suite：`cases=33`，`positive=5`；
- record/model suite：`mutations=64`，`positive=6`；
- deep-replay suite：`mutations=3`，`positive=1`；
- campaign/summary suite：`mutations=4`，`positive=2`；
- 59-Tag/phase-gate suite：`mutations=57`，`positive_helpers=2`；
- fresh host CTest：11/11 PASS。

故意破坏覆盖了不同函数拼接、PTX/SASS symbol 不同、helper 指令污染、重复 JSON key、非有限 JSON 数字、1SM/2SM 交叉、错误 Stage count、错误 Atom shape、Config 只在注释中出现、错误 Docker argv/mount、跨 attempt excerpt、伪造 journal seal、错误 source snapshot、伪 FATBIN/CUBIN、跨层 type witness 伪造和 summary 跨 run 选取。

正式结果随后执行：

```text
python3 tools/validate_codegen_v2.py \
  --root . \
  --require-results \
  --require-archive \
  --deep-replay
```

Deep replay 从当前冻结源码重新编译六个 type witness 和六个 FATBIN，在全新的临时目录重提取 PTX/CUBIN，并从 CUBIN 重生成 ELF symbols、code metadata、nvdisasm JSON/text。六份派生产物与首次 sealed evidence 全部一致：

```text
phase1_results=6
unsealed_results=0
result_contract_bundle_sha256=ee0352d9c1887f650d33b17bf47c1b6e3a56b526878214fb6552701b4c579ce7
```

## 修复

证明固定 CUTLASS e05f953a5b3d38adc240df2ff928e0421c2abba3 中每个高层 Mainloop Schedule（59 个显式 Tensor Tag，加上按 11 个实现分组建立的 KernelScheduleAuto 控制项）如何经过 Builder、DispatchPolicy、Collective、Stage、Copy/Layout、TiledMMA 与 Atom，映射到同一 sm_110a 目标函数的 PTX/SASS；所有项目得到可审计的静态终态，静态证据绝不越级为 Thor 运行、数值或性能结论。

对抗审查中实际发现并修复的关键问题如下：

1. 旧 inspector 会按 opcode 取第一个函数，并允许 PTX 与 SASS 来自不同 symbol。新归因规则先绑定唯一 symbol，再执行函数内 full-match opcode contract。
2. 普通 executable 可能含空的额外 CUBIN。新链改用一次 seeded `nvcc --fatbin` 同时生成 PTX 与 CUBIN。
3. raw SASS 文本会混入内部 guardrail。新合同只消费 nvdisasm JSON 中 exact-name function 的 `sass-instructions[].opcode`。
4. `extern "C"` wrapper 会扰动真实 CUTLASS kernel codegen。正式对象改为显式实例化真实 `cutlass::device_kernel<GemmKernel>`。
5. 长 `GemmKernel` 的 host demangle 会失败。新规则不依赖 host `c++filt`，而由共同 Config、Builder/Kernel cross-view、真实 device-kernel ABI 外壳和 deep replay 联合归属。
6. Sparse 的 Epilogue、Stage policy 和 B scale-vector 原声明不准确；已改为真实 `TmaWarpSpecialized1SmNvf4`、`StageCountAutoCarveoutEpi<CollectiveEpilogue>` 和 A/B 均 SV32。
7. 只写 artifact parent ID 不能证明真实派生。新增 pinned-image deep replay，重新编译、提取和反汇编。
8. mutable status manifest 曾进入 fingerprint，结果投影会造成自失效。现在只冻结 Tag/group 与 Auto seed/hypothesis/source anchors；`status/result_id/resolved_*` 是可重算投影。
9. evidence 多文件发布不能原子完成。现在合法 summary 是唯一 commit marker；未被 summary 引用的文件只计为 unsealed staging，不参与证据结论。

修复后，真实 `dense_f16_2sm` 长 ABI 类型、五类 block-scaled/sparse path 和完整六实例 replay 均重新通过。

## 阶段完成

证明固定 CUTLASS e05f953a5b3d38adc240df2ff928e0421c2abba3 中每个高层 Mainloop Schedule（59 个显式 Tensor Tag，加上按 11 个实现分组建立的 KernelScheduleAuto 控制项）如何经过 Builder、DispatchPolicy、Collective、Stage、Copy/Layout、TiledMMA 与 Atom，映射到同一 sm_110a 目标函数的 PTX/SASS；所有项目得到可审计的静态终态，静态证据绝不越级为 Thor 运行、数值或性能结论。

阶段 1 的范围现已闭合：

```text
positive_validation: PASS
deliberate_breakage: PASS
adversarial_audit: PASS
full_phase_rerun: PASS
unexplained_failures: 0
omissions: 0
evidence_promotions: 0
```

持久化结果：

- current summary：`evidence/codegen-sm110a-v2/summary-phase1-fresh-20260830.json`，SHA-256 `1b5be5fffe28491f7a3b165ea8739de97e59ff81c8116c91e20e85e96cdf00b9`；
- 6 份 result、6 份 fingerprint、6 份 artifact manifest、6 份 hash-chained journal；
- 24 份 Git 内 type/PTX/SASS/contract excerpt；
- 完整 archive 约 16 MiB，位于被忽略的 `artifacts/codegen-sm110a-v2/phase1-fresh-20260830/`，每个文件的 SHA-256 与 bundle checksum 已进入 manifest。

59 个显式 Tag 的当前投影因此变为：

```text
STATIC_PASS: 6
HISTORICAL_STATIC_PASS: 0
NOT_CHECKED: 53
```

这只表示六个 canonical instance 的静态链路已闭合。没有执行 Thor kernel，没有数值输入输出，没有 CUDA event/NCU，也没有延迟、吞吐或性能比较。剩余 53 个显式 Tag 与 11 个 Auto control 仍由后续阶段完成。
