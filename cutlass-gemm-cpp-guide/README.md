# CUTLASS GEMM C++ Guide for NVIDIA Thor / SM110a

这是 `CUDA_optimazation` 仓库中的独立子项目，也是一个可执行的 CUTLASS GEMM C++ 指南。现有已实现 case绑定独立源码、机器可读
`case.json`、CTest 入口以及函数级 PTX/SASS contract；它不把源码存在、编译成功、
指令出现、数值正确和性能优秀混成一个结论。

## 当前结论

- 目标只承诺 `compute_110a → sm_110a`。
- CUTLASS 固定为 v4.6.1 / `e05f953a5b3d38adc240df2ff928e0421c2abba3`。
- CUDA 固定为 13.0.88，canonical container 固定到 digest。
- 历史 `static-20260817` snapshot仍保留 10 个旧 case 的聚合记录；它们与 fresh 结果分开
  展示，不因为 Tag 已重放就自动升级 BF16、FP8、Bias+ReLU 或 tail 等历史变体。
- Phase 1 已用通用结果合同 fresh 重放 6 个 canonical instance：Dense FP16 1SM/2SM、
  NVFP4、MXFP4、MXFP8 和 Sparse NVFP4。合同已经区分 logical/Builder 类型、不同 copy/layout
  role、多级 Stage、MMA fragment 的实际 SMEM/TMEM source，以及 Planar、FastFP32、MixedInput、
  Blockwise、Sparse 等 family-specific 结构。六份正式结果闭合了唯一 PTX/SASS 目标函数和
  函数内 opcode，并通过从源码重新编译开始的 deep replay。
- 59 个显式 Schedule Tag 当前为 `STATIC_PASS=6`、`NOT_CHECKED=53`；11 个
  `KernelScheduleAuto` 控制项仍全部 `NOT_CHECKED`。
- 当前机器没有可用 NVIDIA driver；所有 `runtime_correct` 仍是 `false`，不能称为
  Thor 数值闭环或 v0.1 release。

## 最短使用路径

```bash
git clone --recurse-submodules https://github.com/hibouwu/CUDA_optimazation.git
cd CUDA_optimazation/cutlass-gemm-cpp-guide

# 无 GPU：schema、shape/layout、独立 CPU oracle
cmake --preset host
cmake --build --preset host
ctest --preset host

# Thor runner：编译与运行
cmake --preset sm110a-gpu
cmake --build --preset sm110a-gpu
ctest --preset sm110a-gpu -L runtime-sm110a
```

单独查看/验证 case：

```bash
./build-sm110a-gpu/dense_f16_1sm_p128 --describe --json
./build-sm110a-gpu/dense_f16_1sm_p128 --verify --seed 20260817

./build-sm110a-gpu/bs_mxfp8_1sm_p128 --describe --json
./build-sm110a-gpu/bs_mxfp8_1sm_p128 --verify --seed 20260817
```

## 阅读顺序

1. [CUDA Core / SIMT GEMM Codegen 形式化合同验证](docs/01-cuda-core-simt-gemm-codegen-formal-validation.md)
2. [Tensor Core / TCGen05 GEMM Codegen 形式化合同验证](docs/02-tensor-core-tcgen05-gemm-codegen-formal-validation.md)

两篇文档都从高层 GEMM/Builder 配置出发，依次确认 Builder 偏特化、Collective、Stage、
TiledMMA、Atom/Wrapper 与函数级 PTX/SASS 的对应关系。`STATIC_PASS` 不表示 Thor runtime
correctness，也不包含性能结论。

Tensor Core Codegen 的固定分母是 CUTLASS v4.6.1 中 59 个显式 SM100 Schedule Tag。完整
清单位于
[`tests/codegen/sm110a_tensor_schedule_tags.json`](tests/codegen/sm110a_tensor_schedule_tags.json)，
来源清单位于
[`tests/codegen/sm110a_schedule_reference_inventory.json`](tests/codegen/sm110a_schedule_reference_inventory.json)。
11 个 `KernelScheduleAuto` 控制项位于
[`tests/codegen/sm110a_auto_control_inventory.json`](tests/codegen/sm110a_auto_control_inventory.json)。
阶段 0 给 59 个 Tag 固定了四类来源线索：39 个能在官方 C++ test/example 中找到显式或
条件式引用，1 个只在官方注释中给出 Auto 候选映射，5 个可由 `generator.py` 配置还原，
其余 14 个从相邻合法配置沿一个明确变化轴派生。这组数字不表示任何实例已经被
`sm_110a` 编译器接受。完整审计见
[`phase-00-workspace-source-inventory.md`](evidence/codegen-sm110a-v2/phase-reports/phase-00-workspace-source-inventory.md)。
跨层 harness、六实例数据和对抗审查见
[`phase-01-cross-layer-harness.md`](evidence/codegen-sm110a-v2/phase-reports/phase-01-cross-layer-harness.md)。
`host.schedule_tag_coverage` 会检查目标、源码 commit与clean状态、59项分组、case语义映射、
来源锚点和文档表格；
`host.schedule_tag_adversarial` 负责验证故意破坏不会被误判为通过。

## 重要边界

`128×128×128` 在文档中默认表示数学 problem shape；它不自动等于 CTA tile、
MMA tile、单条指令 shape、TMA tile或 scale tile。性能 benchmark、NCU、grouped、
batched、pointer-array、Stream-K 和跨架构可移植性均不属于 v0.1。

GitHub Actions workflow位于父仓库根目录的 `.github/workflows/cutlass-guide-*.yml`；
子项目内不保存一套无效的嵌套 workflow副本。
