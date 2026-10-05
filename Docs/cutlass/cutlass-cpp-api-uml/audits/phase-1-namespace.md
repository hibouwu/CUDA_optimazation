# 阶段1：命名空间绑定独立审查

本轮审查者没有编写或修改 `namespace_bindings.py` 的绑定实现，也没有修改核心命名空间分支。审查采用独立缩减源码，先用 Clang 验证实际 owner，再检查提取结果；不以零诊断或实体数量替代正确性。

## 版本边界与运行结果

本轮审查前后读取的源码SHA-256一致：

- `scripts/namespace_bindings.py`：`2bc874ca4339bffca691ad0943c1bf65ae4944133b0b85410512277d9d7dd4a1`
- `scripts/extract_declarations.py`：`0df5813ecb87117c6e3d9364ebad6de6ed82fee98a47d86a92fe9ff9d8baee8a`
- 独立测试 `tests/test_namespace_review.py`：`6a68b3c3051a3c578be5df5397fbc339675817317ffbb399c04544ad59394ad1`

执行入口：

```bash
.venv/bin/python -m unittest discover -s tests -p test_namespace_review.py -v
```

当前结果为 **9项测试、5通过、4失败**。失败测试未使用 `expectedFailure`；它们代表需要修复的开放缺陷。Clang版本为21.1.8；交叉检查使用 `-std=c++17 -pedantic-errors -fsyntax-only`，并以 `sizeof(expected_owner::X)`验证命名空间归属。普通fixture不使用系统头文件；`CUTE_STL_NAMESPACE`与`cutlass`的配置前导宏取自固定源中的相应定义形态。

## 开放缺陷

### NS01：compound namespace中的宏分量未展开，且零诊断

最小反例：

```cpp
#define API_NS a
namespace API_NS::b { struct X {}; }
```

Clang确认owner为 `a::b::X`。当前提取器输出 `API_NS::b::X`，其作用域被标为 `source_name`，诊断为空。

核心在 `extract_declarations.py:436` 将完整拼写 `API_NS::b` 传给resolver；resolver在 `namespace_bindings.py:32`按整个字符串匹配宏名。因此分量 `API_NS`没有参与替换，随后只按 `::`拆字面作用域。

同一问题影响真实库的 `namespace cutlass::detail`：提取结果只有默认字面owner，没有 `CUTLASS_NAMESPACE`已定义时的参数化owner。缩减用例中设置 `CUTLASS_NAMESPACE review`后，Clang确认owner应为 `cutlass_review::detail::X`，但源清单没有相应参数化分支。

固定源实例包括：

- `include/cutlass/detail/mma.hpp:38`
- `include/cutlass/detail/sm100_blockscaled_layout.hpp:43`
- `include/cutlass/gemm/dispatch_policy.hpp:42`
- `include/cutlass/transform/kernel/sparse_gemm_compressor.hpp:52`

对应失败测试：`test_macro_component_of_compound_namespace_expands`、`test_compound_cutlass_retains_parameterized_owner`。

修复应对每个预处理token分量进行宏替换与rescanning，再形成namespace组件；不能把整个compound拼写误当一个宏名。默认owner与参数化owner应共享同一源出现位置并保留各自条件。

### NS02：本地宏指向导入配置宏时，未rescan便宣布literal resolved

最小反例：

```cpp
#define API_NS CUTE_STL_NAMESPACE
namespace API_NS { struct X {}; }
```

分别定义 `CUTE_STL_NAMESPACE std` 和 `CUTE_STL_NAMESPACE cuda::std`后，Clang确认两个配置的owner为 `std::X`、`cuda::std::X`。当前提取器却输出字面 `CUTE_STL_NAMESPACE::X`，诊断为空。

`namespace_bindings.py:56`的rescan检测只使用 `local_defs`，未考虑 `config_defs`，然后60行直接调用 `_literal`。因此一个尚需配置宏替换的标识符被误报为已解析的具体namespace。

对应失败测试：`test_local_alias_to_imported_macro_is_rescanned_or_pending`。测试接受两种诚实结果：完成两个条件owner的rescanning，或保留明确pending；不接受零诊断的字面owner。

### NS03：重复使用同一配置宏被独立做笛卡尔积，产生不可能owner

最小反例：

```cpp
namespace CUTE_STL_NAMESPACE {
  namespace CUTE_STL_NAMESPACE { struct X {}; }
}
```

同一翻译单元内的两次宏展开必须使用同一个配置。Clang检查仅得到：

- 未定义 `__CUDACC_RTC__`时的 `std::std::X`；
- 定义 `__CUDACC_RTC__`时的 `cuda::std::cuda::std::X`。

当前核心对内外层分别枚举两选项，输出4种owner，还包含 `std::cuda::std::X`和`cuda::std::std::X`。这两项的条件列表同时要求 `defined(__CUDACC_RTC__)`及其否定，却没有 `unsatisfiable`或恒假标记。它们不能被当作可达的条件实现。

对应失败测试：`test_repeated_macro_uses_share_one_configuration`。测试允许保留不可能分支用于审计，但必须显式标为不可满足；或者直接共享一致配置环境，避免生成这两个组合。

这不同于删除原始 `#if 0` 源码：这里的混合组合不是任何一个实际宏环境所产生的源码分支，而是绑定器独立组合引入的结果。

## 已通过的针对性检查

- `#if 0`中的 `#undef CUTE_STL_NAMESPACE`不会禁用实际宏；std/cuda::std两种条件owner均保留。
- 多行注释中的伪`#undef`不会作为宏生命周期事件。
- `#undef API_\`换行`NS`经过行拼接后生效，恢复字面`API_NS`。
- `#define → #undef → #define`按使用点之前的新定义形成owner，未追溯改写旧位置。
- 未知条件下的`#undef`明确pending，没有无依据选择一种宏状态；Clang分别验证了条件0/1的两个具体结果。

对于普通 `namespace cutlass`，当前绑定保留token-paste/rescan表达式、三条固定宏定义证据和真实缺失绑定 `CUTLASS_NAMESPACE replacement tokens and any macros reached by argument/result rescanning`。本轮没有发现该表达式本身猜造具体字符串；开放问题是compound写法漏走这套绑定，以及重复配置选择缺乏一致性约束。

## 审查结论

当前命名空间绑定尚不能通过阶段1。三类开放缺陷对应四个实际失败测试，其中NS01已确认影响固定源的常见namespace写法。上述5项通过结果只支持相应缩减行为，不证明全库owner、宏状态或声明清单已经完整。

## 追加复审：NS01–NS03的已知反例关闭

根代理修复后，独立审查者重新执行原有9项测试，结果为 **9/9通过**。测试未删除、未改弱，也没有添加`expectedFailure`。

本次复审版本：

- `scripts/namespace_bindings.py`：`846a9792fac6e01f9ddecc7a4c68e66b1b750d350260f241b1a3d8ad0b0ae384`
- `scripts/extract_declarations.py`：`7ac761eb0b18840a6aedf426d225907e50b6e89b81f45c076a8b659034e345e3`

复审确认：compound namespace的宏分量进入绑定流程，`cutlass::detail`保留参数化owner；本地宏指向`CUTE_STL_NAMESPACE`时继续rescanning到两个条件owner；嵌套重复宏通过共享binding key/choice约束，不再产生两个互相矛盾的混合owner。因此 **NS01、NS02、NS03在上述缩减反例及版本边界内关闭**。

这不是全量命名空间覆盖结论，也不表示阶段1通过。原始失败记录保留在上文；后续若发现新反例，应新增记录并复开相应问题，而不是改写历史结果。
