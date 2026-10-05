# `cute_array` 第一版关系草稿：有界只读复审

复审对象是固定快照 `array.hpp` 的 source contracts、具体调用重载、tuple 命名空间、零长度特化、ADL 和隐式成员边界，不以计数吻合替代语义检查。本轮没有执行生成器，也没有修改 `author_draft.py`、`relations.json` 或全局数据。

本轮摘录对应的文件散列：

- `author_draft.py`：`6fb2c8022276ac9314e1d7e83236e70887cfd18cbf16b96e3e44e6e890a60b3d`。
- `relations.json`：`408d8c209d261872efa638ecffc687382610d35c4e103c6519815efff11e880e`。

生成器由作者并行修订，以下 ID 与结论以这些具体摘录为准；后续修复需要重新读取结果，而不能将本报告当作最新版本的失败判定。

## 发现 1：合并实体的顶层条件错误地只保留首个物理出现

优先级：P1，可能直接影响按配置筛选时的实体存在性。

`array.namespace.std`、`array.type.std.tuple_size`、`array.type.std.tuple_element` 及 `array.alias.std.tuple_element.type` 各自合并了两处物理声明：430–444 行的非 RTC `CUTE_STL_NAMESPACE -> std` 分支，以及 447–475 行受 `CUTE_STL_NAMESPACE_IS_CUDA_STD` 控制的 std 桥。它们的 `source_occurrences` 正确保留两处条件，但顶层 `preprocessor_conditions` 仍只有首次出现的 `!defined(__CUDACC_RTC__)`。

具体反例是 RTC 配置：`config.hpp` 定义 `CUTE_STL_NAMESPACE_IS_CUDA_STD`，因此第 464 行起的 std tuple 特化实际存在；顶层 `!RTC` 却为假。若消费者将该字段视为整个实体的存在条件，会错误排除真正存在的 std 桥。

建议用“各 occurrence 条件合取后再相互析取”的条件组表示实体存在性，或明确移除单一顶层条件的实体语义、强制消费者使用 occurrences。不要将多个互斥出现的条件平铺为一个 AND。对应 occurrence 的 `aliases`、`type_uses` 等关系也应保留来源分支，不能依赖当前错误的顶层条件来补足。

## 发现 2：预处理 `#if` 表达式被当作普通潜在求值表达式

优先级：P2，错误求值阶段与错误 owner。

`array.edge.0500.evaluates` 把源文件第 450 行 `__CUDACC_VER_MAJOR__ >= 13` 记为 `source = array.namespace.cute`、`evaluation = potentially_evaluated`、`condition = 无额外条件`。对应节点为 `array.expression.binary_expression.8671`。

该表达式实际属于 `#if`，位于第 446 行条件控制的 std 桥文本中，不是 cute 命名空间里的 C++ 求值。问题来自通用 `binary_expression` 处理路径没有排除预处理祖先节点，随后 `owners(ast)` 无普通声明 owner 时回退到 cute。

建议保留表达式原文，但标为 `preprocessing_condition` 一类，使用预处理阶段，并关联外层 `defined(CUTE_STL_NAMESPACE_IS_CUDA_STD)` 条件及其物理作用范围。不要为它产生普通 C++ `potentially_evaluated` 关系。

## 发现 3：条件 include 只登记文本，未登记选择条件

优先级：P2，文件依赖条件缺失。

三个 include obligation 的 `preprocessor_conditions` 均缺失：

- 第 396 行 `CUDA_STD_HEADER(tuple)`，实际需要 `defined(__CUDACC_RTC__)`。
- 第 398 行 `<tuple>`，实际需要 `!defined(__CUDACC_RTC__)`。
- 第 452 行 `<cuda/std/__tuple_dir/structured_bindings.h>`，实际需要 `defined(CUTE_STL_NAMESPACE_IS_CUDA_STD) && (__CUDACC_VER_MAJOR__ >= 13)`。

宏展开节点为第 396 行单独保存了 RTC 条件，但 include obligation 自身没有关联该条件；第 452 行没有可等价替代的条件记录。`conditions()` 目前只额外覆盖旧 RTC 前置声明区间，未覆盖第 452 行的 CUDA 13 分支。

建议在 include/file-dependency occurrence 上直接保存完整外层条件链，并将包含表达式的图引用连接到真实依赖或宏展开节点；不能仅以“预处理 directive 已计入另一个清单”替代依赖关系自身的条件。

## 发现 4：tuple 模板形参类型的 owner 回退到了 `cute`

优先级：P2，错误源实体/作用域。

以下 `type_uses` 的 target `array.external.size_t` 本身合理，但 source 全是 `array.namespace.cute`：

- `array.edge.0378.type_uses`：源行 433。
- `array.edge.0391.type_uses`、`array.edge.0392.type_uses`：源行 438。
- `array.edge.0403.type_uses`：源行 459。
- `array.edge.0404.type_uses`：源行 464。
- `array.edge.0411.type_uses`、`array.edge.0412.type_uses`：源行 469。

这些 `size_t` 位于 `CUTE_STL_NAMESPACE` 或 `std` 内的 tuple 模板形参，并不属于 cute 命名空间。模板形参文本位于其 `struct_specifier` AST 范围之前，`owners(ast)` 找不到声明 owner，过滤 namespace 后便错误使用 cute 兜底。

建议先把模板形参绑定到其 template subject 或已创建的 parameter node，再由该物理出现继承命名空间条件；对于宏命名空间中的模板，保留 std/cuda::std 两个配置绑定。不要用固定 cute 作为所有无法归属语法的默认 owner。

## 已核对、未发现错误的重点

所有列出的具体 `calls` 与 `implicit_calls` 已逐条对照原文。当前 mutable/const 的 `begin`、`data`、`end`、`operator[]` 目标没有发现选错；非 const `cbegin/cend` 先选 mutable 成员后转换返回类型的处理正确。

零长度成员空体、free clear 的 `T(0)` 要求、`operator==` 普通循环的实例化边界、`reverse` 聚合初始化/赋值/按值返回要求，与 14 个既有 Clang oracle 的证据一致。`get` 的 `a[I]` 与 `cute::move` 已分两层，`const&&` 回退与函数体内 `static_assert` 的说明正确。

成员 `swap` 保留了普通 using 引入与 ADL 的候选路径，没有错误指定唯一元素 swap；隐式特殊成员只作为依赖 T/N/语言模式的边界说明，没有伪造源文件中不存在的构造、析构或赋值 API。这些判断不扩展到未测试 CUDA/RTC 配置、用户新增特化、任意 T 实例或运行时行为。

## 定点复验追加：四项均已关闭

作者修复后，本轮仅重新读取上述四项对应字段与关系，没有运行生成器、扩大复审范围或改动关系数据。定点复验对象为：

- `author_draft.py` SHA-256：`16bfafae16ae02d9ea9d08d01a45e4a8bd5d42473110a9dcd3dad676ecc717df`。
- `relations.json` SHA-256：`977e6b76120fd15001e7aa76c54e8c3e9c2aa4a5fe7562c7d1b6e7a2d3a63883`。

**发现 1：CLOSED。** 四个被点名的合并实体均已移除误导性的顶层 `preprocessor_conditions`，改用 `availability.operator = any_of`，其两个 `occurrence_condition_groups` 分别保留非 RTC 宏命名空间出现与受第 446 行控制的 std 桥出现。各物理 occurrence 原条件仍在。std `tuple_element::type` 的第 441、472 行 alias 边也分别绑定到对应来源条件组，不再借用首次出现的实体条件。

**发现 2：CLOSED。** 第 450 行表达式现对应 `array.edge.0503.evaluates`，source 已是 `array.namespace.std`，`evaluation = preprocessing`，condition 与 `source_occurrence_condition_groups` 均保存第 446 行的 `defined(CUTE_STL_NAMESPACE_IS_CUDA_STD)`。不再被解释为 cute 内的普通潜在求值表达式。

**发现 3：CLOSED。** 第 396 行 include 保存 RTC 条件（directive 395），第 398 行保存非 RTC 条件（directive 397）；第 452 行同时保存 std 桥条件（directive 446）与 CUDA 主版本至少 13 的条件（directive 450）。三处被点名的条件链均已补齐。

**发现 4：CLOSED。** 第 433、438 行模板形参中的 `size_t` 引用分别绑定 std/cuda::std 的 tuple subject，并各带非 RTC/RTC 来源条件组；第 459 行绑定 `array.type.decl_460` 并保存完整旧 RTC 前置声明条件；第 464、469 行绑定对应 std tuple subject 并保存第 446 行桥条件。这些定点引用不再回退到 `array.namespace.cute`。

本轮关闭的是以上四项明确缺陷，不代表新版本所有图关系、CUDA/RTC 编译分支或全项目覆盖率已经得到独立验收。作者报告的其他 mutation 检查不在本轮复验范围内。
