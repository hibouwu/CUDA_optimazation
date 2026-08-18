# CUTLASS API 层级

## Track A：应用调用

每个 full-GEMM case按同一生命周期：

```text
Element/Layout/Alignment
  -> CollectiveEpilogue
  -> CollectiveMainloop
  -> GemmUniversal
  -> GemmUniversalAdapter
  -> Arguments
  -> get_workspace_size
  -> can_implement
  -> initialize
  -> run
```

主入口分别在：

- [`cutlass_dense_case.hpp`](../include/guide/cutlass_dense_case.hpp)
- [`cutlass_blockscaled_case.hpp`](../include/guide/cutlass_blockscaled_case.hpp)
- [`cutlass_bias_relu_case.hpp`](../include/guide/cutlass_bias_relu_case.hpp)

每个 case仍有独立 `case.cu`，因此 codegen audit不会在一个包含多种 kernel 的大 binary
里误抓 mnemonic。

## Track B：机制下钻

Track B 不复制一套 raw PTX full GEMM。它追踪 Track A 最终实例化的 CuTe 类型：

```text
TMA copy atom
SMEM layout / descriptor
MMA atom / TiledMMA
TMEM allocation and accumulator
UTCCP scale copy
completion barrier
TMEM load
```

`tests/mechanism/test_cutlass_source_contracts.py` 把固定 submodule 中的 collective、atom 和
inline PTX wrapper连接起来；`inspect_codegen.py` 再验证这些路径确实进入目标函数。

## `Sm100` tag 与 `sm_110a`

CUTLASS builder仍使用 `cutlass::arch::Sm100`。真正启用 SM110 TCGen05/TMA/TMEM 的是
NVCC 的 `sm_110a` target及 CUTLASS `config.hpp` 中相应 feature macro。文档不得把 C++
tag 名称当成 binary architecture。
