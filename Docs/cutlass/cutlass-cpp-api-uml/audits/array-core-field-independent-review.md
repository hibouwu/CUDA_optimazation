# array 接入所需核心字段变更：独立有界复审

结论：**固定 array.hpp 的三项字段接入未发现静默字段丢失或调用限定符串入；新增 12 项独立测试全部通过。** 同时记录一项更早的宏／条件函数头组合未覆盖边界：它明确输出 pending，不是解析成功；本轮没有修改核心、降低诊断或重启全库生成。

审查对象是 `Extractor.declaration` 的返回 cv 分离、函数形参 index、字段 declared_type_spelling。不是对完整 C++ 类型系统、所有 field kind 分类或任意声明投影作全面证明。

读取的核心指纹：

```text
scripts/extract_declarations.py
2ff1b7be010b090f0ef57d093bfa062d6a113b04049abae5b1e2ca7d74b712c7

scripts/declaration_projection.py
daff166263c0e89f8549f9ee6099cd2a0e23cd6252854b13061a2752d7fb9ee6
```

固定原文为提交 `8f50b052e1099fb982392a622caab69b97b63128` 的 `include/cute/container/array.hpp`，SHA256 `a5a6c8357cf4311bed395db7dfa8fcc5c28b3d295fdfec9ad2b00b75253ad8ee`。本次只直接提取这个 header 和少量内存 fixture，未读取／写回全局 declarations.json。

## 1. 固定 array 的实际输出

新鲜单文件结果仍为 92 条条件绑定 occurrence、52 个可调用定义、19 个写出的函数形参、0 诊断；没有把本轮结果声称成新的全局账本已经完成。

- 每个函数 parameters 中的 index 是从 0 开始的连续序号；物理参数名、默认值与 raw 区间不因新增 index 改变。固定 array 的 19 个形参均回切原始字节一致。
- 52 个函数均保留 constexpr；恰有 22 个 class 成员的 callable qualifiers 含一个 const，没有重复的返回 const。全部函数 raw_signature 仍与物理 signature_range 回切一致。
- 两个 const data（签名起始 99／254 行）输出 `return_type=const T *`、`return_cv_qualifiers=[const]`、`qualifiers=[const,constexpr]`。
- free `get` 的 const-reference 重载（起始 412 行）输出 `return_type=const T &`、`return_cv_qualifiers=[const]`，但 callable qualifiers 只有 constexpr；没有虚构一个 free const 方法。
- `__elems_` 仍是唯一物理存储成员：`declared_type=element_type`、`declarator=__elems_[N]`、`declared_type_spelling=element_type [N]`、initializer=null。零长度特化没有新增字段。
- 固定 array 的语义字段与实体身份中没有 `__codex_parser_` 临时名字。

## 2. 返回 cv 与方法 cv 的独立对照

以下合法重载没有混合 receiver 与返回类型身份：

```cpp
struct S {
  const int* f();
  int* f() const;
  const volatile int* g() const volatile & noexcept;
};
```

两个 f 的 entity_id 不同。第一个 callable qualifiers 为空、return_cv 为 const；第二个 callable qualifiers 为 const、return_cv 为空。g 的 callable qualifiers 保留 const、volatile、`&`、noexcept，返回类型另保留 const volatile。独立 Clang 21.1.8、C++17 的成员函数指针类型断言实际通过。

新增字段的**明确边界**也纳入测试：

```cpp
struct S {
  int* const p();
  auto q() const -> const int*;
};
```

这里 p、q 的 `return_cv_qualifiers` 都为空，但 return_type 分别完整保留 `int * const` 和 `const int*`。该列表实际只描述函数 declaration 节点的直接 declaration-specifier cv，不是所有指针层、尾置返回、alias 展开后 cv 的汇总。空列表不能被下游解释为“返回类型不含 const”。Root 已确认按此实际合同编写字段语义文档；本轮不为补字段 role 打断冻结生成器。

类似地，`static constexpr int constants[2]` 的 declared_type_spelling 是 `int [2]`，constexpr 单独在 qualifiers 中；Clang 断言其实际 `decltype(S::constants)` 是 `const int[2]`。完整声明拼写组装不等于实例化／推导后的完整语义类型，不能丢掉 constexpr 后拿这个 spelling 独立做最终类型比较。

## 3. 字段形状、initializer 与投影

有界 fixture 核对了下列类型拼写与对象名字分离：

```cpp
struct S {
  int *pointer = nullptr, values[3] = {1,2,3};
  const int* const fixed = nullptr;
  int (*matrix)[3] = nullptr;
  int (&reference)[3];
  const int (*callback)(float x) = nullptr;
  int (*callbacks[2])(float x) = {};
};
```

pointer／values 没有共享错 initializer 或指针形状；返回的抽象声明拼写分别包含 `int *`、`int [3]`、`const int * const`、`int (*)[3]`、`int (&)[3]`、`const int (*)(float x)`、`int (*[2])(float x)`。callback／callbacks 保持字段类别，而非被当成可调用定义。Clang 对这些实际成员类型的 `__is_same` 断言通过。

这里仅验证“字段而非函数”，没有把既有 `member_constant` 分类提升为最终可变性语义。旧实现会因声明中出现 const 而使用该 kind，即便 const 修饰的是 pointee／函数返回类型；这是原有分类粒度，不是本次新增 declared_type_spelling 所证明的对象 const。新消费者仍需按照完整声明组合解释类型。

投影组合还核对了：

- `decltype(typename T::X{}) value{};` 恢复完整类型表达式和 `{}` initializer。
- 同一声明中重复 `typename T::X{}`／`sizeof(typename T::X)` 时，前者恢复不会改写后者。
- 匿名 `unsigned : 3;` 的 name 仍为 null，bit_width=3，declared_type_spelling 不含临时声明名或位宽拼接。
- 匿名默认参数 `int* = nullptr` 的 name 为 null，含 `{}` 的具名参数和 `...` 均有正确 index 与原默认值。
- 独立 macro 字段 `FIELDS(3)` 与另一个 class 内的条件函数头均保留新字段，宏展开字段的大小为 3，两函数变体的 index 各从 0 开始；semantic 字段未泄漏 parser placeholder。

## 4. 未覆盖但未伪装成功的组合边界

下例在 Clang C++17 的 `-DA=0`、`-DA=1` 两次编译中都通过：

```cpp
#define FIELDS(N) int slots[N]={}; const int* ptr=nullptr;
struct S {
  FIELDS(3) static const int* f(
#if A
  int x
#else
  float y
#endif
  ) { return nullptr; }
};
```

当前 header 识别把前方 `FIELDS(3)` 连同 `static const int* f(...)` 一起作为 conditional header 候选，使变体无法按函数头单独解析。结果明确保留 `conditional_function_variant_parse_pending` 与 parse_error，恢复得到的 f 没有标为 parsed。

这不是本次新增 index／类型字段造成的静默成功，也不出现在固定 array.hpp。最初组合正控暴露这一点后，测试改为：分别核对已支持 macro／conditional header 的字段，再要求这个尚未支持的相邻组合仍然明确 pending；没有跳过结果、改诊断或宣称该组合已覆盖。若后续要修复，应单独验证 conditional header 起点与前置宏成员边界，不能仅删除 FIELDS 文本或泛化允许此变体通过。

## 5. 可重复检查与收口

新增测试：[test_array_core_fields_independent_review.py](../tests/test_array_core_fields_independent_review.py)。共 12 项：固定 array 的分母／参数／字段／返回类型，cv-ref 重载，复杂字段与 initializer，匿名参数，投影恢复，macro／条件头，以及上述显式 pending 边界。Clang 使用 stdin、`-fsyntax-only`；没有生成或运行 C++ 程序，没有 GPU／ABI 跨平台保证。

```sh
.venv/bin/python -m unittest discover -s tests -p test_array_core_fields_independent_review.py -v
```

本次 12 项全部通过。通过的含义是新增字段在已检查的固定源码形状中保留了应有语义，且未覆盖组合没有被静默通过；不意味着所有声明类型、implicit const、return cv 层级或所有投影组合均已求解。未更改或重启 Root 正在进行的全量任务。
