# CUTLASS GEMM C++ Guide for NVIDIA Thor / SM110a

这是 `CUDA_optimazation` 仓库中的独立子项目，也是一个可执行的 CUTLASS GEMM C++ 指南。每个章节都绑定独立源码、机器可读
`case.json`、CTest 入口以及函数级 PTX/SASS contract；它不把源码存在、编译成功、
指令出现、数值正确和性能优秀混成一个结论。

## 当前结论

- 目标只承诺 `compute_110a → sm_110a`。
- CUTLASS 固定为 v4.6.1 / `e05f953a5b3d38adc240df2ff928e0421c2abba3`。
- CUDA 固定为 13.0.88，canonical container 固定到 digest。
- 10/10 核心 case 已通过 container compile 与函数级 PTX/SASS contract。
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

1. [范围与证据](docs/scope-and-evidence.md)
2. [构建与工具链](docs/build-and-toolchain.md)
3. [GEMM Shape Ledger](docs/gemm-shape-ledger.md)
4. [CUTLASS API 层级](docs/cutlass-api-layers.md)
5. [Dense GEMM](docs/dense-gemm.md)
6. [TMA、TMEM 与 TCGen05](docs/tma-tmem-tcgen05.md)
7. [2SM GEMM](docs/two-sm-gemm.md)
8. [Block-scaled GEMM](docs/block-scaled-gemm.md)
9. [Sparse GEMM](docs/sparse-gemm.md)
10. [Epilogue 与 tail](docs/epilogue-and-tails.md)
11. [测试与 codegen](docs/testing-and-codegen.md)
12. [能力矩阵](docs/capability-matrix.md)
13. [来源与许可](docs/provenance.md)

## 重要边界

`128×128×128` 在文档中默认表示数学 problem shape；它不自动等于 CTA tile、
MMA tile、单条指令 shape、TMA tile或 scale tile。性能 benchmark、NCU、grouped、
batched、pointer-array、Stream-K 和跨架构可移植性均不属于 v0.1。

GitHub Actions workflow位于父仓库根目录的 `.github/workflows/cutlass-guide-*.yml`；
子项目内不保存一套无效的嵌套 workflow副本。
