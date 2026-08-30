# CUTLASS GEMM C++ Guide for NVIDIA Thor / SM110a

这是 `CUDA_optimazation` 仓库中的独立子项目，也是一个可执行的 CUTLASS GEMM C++ 指南。现有已实现 case绑定独立源码、机器可读
`case.json`、CTest 入口以及函数级 PTX/SASS contract；它不把源码存在、编译成功、
指令出现、数值正确和性能优秀混成一个结论。

## 当前结论

- 目标只承诺 `compute_110a → sm_110a`。
- CUTLASS 固定为 v4.6.1 / `e05f953a5b3d38adc240df2ff928e0421c2abba3`。
- CUDA 固定为 13.0.88，canonical container 固定到 digest。
- 历史 `static-20260817` snapshot记录 10/10 核心 case通过 container compile 与函数级
  PTX/SASS contract；完整函数产物仍需在新文档结构下 fresh 重放后逐项归档。
- FP16 dense `128³` 的目标函数包含 TMA、`tcgen05.mma.kind::f16` 和 `tcgen05.ld`，
  且不包含 `tcgen05.cp`。
- MXFP8 block-scaled `128³` 的同一目标函数包含
  `cp.async.bulk.tensor + tcgen05.cp + tcgen05.mma...block_scale + tcgen05.ld`。
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
并由 `host.schedule_tag_coverage` 检查数量、唯一性、分组、case映射和文档完整性。

## 重要边界

`128×128×128` 在文档中默认表示数学 problem shape；它不自动等于 CTA tile、
MMA tile、单条指令 shape、TMA tile或 scale tile。性能 benchmark、NCU、grouped、
batched、pointer-array、Stream-K 和跨架构可移植性均不属于 v0.1。

GitHub Actions workflow位于父仓库根目录的 `.github/workflows/cutlass-guide-*.yml`；
子项目内不保存一套无效的嵌套 workflow副本。
