# `cute::array`：六组有界编译反例

本报告只针对固定快照 `snapshot/include/cute/container/array.hpp`，已完整阅读其 476 行原文。头文件 SHA-256 为 `a5a6c8357cf4311bed395db7dfa8fcc5c28b3d295fdfec9ad2b00b75253ad8ee`。实际工具链为 Clang 21.1.8（Fedora 21.1.8-6.fc43，`x86_64-redhat-linux-gnu`），按 C++17 host 模式执行 `-fsyntax-only`，使用快照、CUDA 13.0 include 和 CCCL include 路径。

六组源码包含 14 个独立编译变体：8 个通过，6 个按预期被拒绝。每次的完整源码、实际命令、退出码、stdout、stderr 和诊断匹配结果都保存在 [oracle_results.json](oracle_results.json)。[oracle_array.py](oracle_array.py) 将源码通过 stdin 送给编译器，不生成 C++ 文件、目标文件或可执行文件，也不运行 C++ 程序。

复现命令（在仓库根目录执行，结果写向 stdout）：

```sh
python3 Docs/cutlass/cutlass-cpp-api-uml/data/module-drafts/cute_array/oracle_array.py
```

以下结果不构成 GPU、PTX/SASS、性能或一般运行时安全验证；常量表达式断言只验证给定良构样本。

## 1. `N=0` 成员 `clear()` 与 free `cute::clear(a)` 不等价

源位置：零长度特化的成员 `fill`、`clear` 在 326–332 行，free `clear` 在 351–355 行。

样本的 `NoZero` 删除了 `NoZero()` 和 `NoZero(int)`。`cute::array<NoZero, 0> a{}; a.clear();` 编译通过；将调用替换成 `cute::clear(a);` 后编译失败，诊断落在第 355 行 `a.fill(T(0));`：

```text
functional-style cast from 'int' to 'NoZero' uses deleted function
```

因此不能把 free `clear` 的关系画成“转发给成员 clear”。它实际构造 `T(0)` 并调用成员 `fill`；即使零长度 `fill` 是空体，调用参数表达式仍须良构。成员零长度 `clear` 自身为空体，不施加这项 `T(0)` 要求。

## 2. 零次普通循环仍检查 `operator==` 函数体中的 `!=`

源位置：339–349 行。

对 `array<Element, 0>` 执行 `static_assert(a == b)`：没有元素比较运算符时失败；仅定义 `Element` 的 `operator==` 时仍失败；提供 `operator!=` 时通过且结果是 `true`。失败都定位到第 344 行 `lhs[i] != rhs[i]`，诊断为 `invalid operands to binary expression`。

这是 C++17 模式下对该固定源码的结论。`for (size_t i = 0; i < N; ++i)` 不是 `if constexpr`；迭代次数为零不意味着实例化时可以忽略循环体的类型检查。应将要求写为“`const T` 元素之间的 `!=` 及其 `if` 条件使用必须良构”，不要只笼统写“T 支持相等比较”。

## 3. `reverse` 的要求应跟随实际初始化与赋值表达式

源位置：372–385 行。

这组包含五个变体：

- `N=0`：元素类型删除默认构造、复制构造、移动构造和复制赋值，`cute::reverse(a)` 仍通过。零长度数组没有存储元素，`if constexpr (N == 0u)` 选中 `return t`，非零分支被丢弃。
- `N=1`，空聚合类型删除默认构造：通过；样本同时断言 `std::is_aggregate_v<Element>` 为真、`std::is_default_constructible_v<Element>` 为假。在这里，C++17 聚合初始化使 `t_r{}` 仍然良构。这是对“必须满足默认可构造类型特征”的直接反例。
- `N=1`，非聚合类型删除默认构造：失败，诊断落在第 380 行 `cute::array<T,N> t_r{};`，指出省略初始化项的元素初始化调用了已删除构造函数。
- `N=1`，删除从 `const Element&` 的赋值：失败，诊断落在第 382 行 `t_r[k] = t[N - k - 1];`，为 `overload resolution selected deleted operator '='`。
- `N=1`，删除复制构造但允许移动构造和从 `const Element&` 的赋值：通过。因而不能额外宣称“reverse 总是要求元素复制构造”，也不能把源码中的赋值称为复制构造。

适合关系草稿的准确措辞是：非零分支先要求 `array<T,N> t_r{}` 良构，再要求目标元素能由源 `const T` 元素赋值，最后要求返回 `t_r` 的按值返回语义良构。本组不尝试枚举所有特殊成员函数组合，也不依据 syntax-only 结果声称实际发生了几次复制、移动或复制消除。

## 4. `get` 只有三个重载，`const&&` 实参不会获得 `const T&&`

源位置：404–426 行；`cute::move` 的定义在固定快照 `cute/util/type_traits.hpp` 194–199 行。

单组编译期类型断言验证：`array<int,1>&` 返回 `int&`，`const array<int,1>&` 返回 `const int&`，`array<int,1>&&` 返回 `int&&`，`const array<int,1>&&` 返回 `const int&`。源码确实没有单独的 `const&&` 重载，最后一种实参绑定到 `const&` 重载。样本还验证 `volatile&` 和 `const volatile&` 数组不匹配这组 `get`。

右值重载不是构造或搬出一个独立元素。源码先经 `a[I]` 取出元素引用，再调用 `cute::move`；后者返回 `static_cast<remove_reference_t<T>&&>(t)`。样本在常量表达式里对仍存活的具名数组调用右值 `get`，取得 `int&&`，通过该引用修改值，并断言引用地址等于 `a.data()`、原数组值同时改变。此断言通过，证据限定为“该右值引用仍指向原数组存储”。它不保证任何来自临时数组的引用生命周期，也不为越界访问提供运行时安全性。

## 5. `tuple_element` 不检查范围，`get` 的范围检查在函数体中

源位置：404–426 行、438–442 行。

对 `using Empty = cute::array<int, 0>`，以下三项可同时通过：`std::tuple_size<Empty>::value == 0`、`std::tuple_element<999, Empty>::type` 是 `int`、`decltype(cute::get<999>(std::declval<Empty&>()))` 是 `int&`。最后一个是未求值的返回类型查询，不实例化 `get` 的函数体。

加入实际调用表达式 `cute::get<999>(a)` 后，编译失败并定位至第 408 行：

```text
static assertion failed due to requirement '999UL < 0UL': Index out of range
```

因此应区分三层：`tuple_size` 报告长度；这里的 `tuple_element` 无条件提供元素类型；实例化 `get` 函数体时由 `static_assert(I < N)` 拒绝越界。不要把 `get` 描述成在重载决议阶段通过 SFINAE 排除越界参数。

## 6. `N=0` 是逻辑长度零，不是对象大小零

源位置：198–337 行。

样本对 `array<NoDefault, 0>` 验证 `std::is_empty_v` 为真，且数组可默认构造，尽管 `NoDefault()` 已删除。`empty()` 为真，`size()`、`max_size()` 为零，`data()`、`begin()`、`end()` 为 `nullptr`，这些常量表达式断言均通过。

`sizeof(Empty) > 0` 和本 host/ABI 下的 `sizeof(Empty) == 1` 也通过。准确写法是“该特化不含元素存储、逻辑长度为零”；不能写成“大小为零”而混淆对象表示与容器长度。数值 `1` 仅记录本次具体工具链样本，不作为跨 ABI 的 API 保证。本组不调用 `front()`、`back()` 或 `operator[]`，更不把这些访问接口的存在解释为零长度时可以安全解引用。
