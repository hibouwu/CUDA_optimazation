# 声明字段的语义边界

这些字段描述固定源码中的声明，不等于编译器已经解析完毕的具体实例类型。`raw_signature` 和相应字节范围保留原文；类型拼写中的别名、模板参数、auto 和条件仍需按实际配置解析。

## 函数

- `return_type` 保留提取器组合出的返回类型拼写，包括指针、引用和相应限定层级。声明说明符可能使用正规化顺序，因此 `const T*` 与原文 `T const*` 可以对应同一个语法类型；它们不能与 `T* const` 混为一谈。
- `qualifiers` 保存函数声明说明符与函数声明器上的限定信息，例如 constexpr、static、成员函数的 const/ref/noexcept。返回基础类型上的 const/volatile 不再混入此列表。
- **`return_cv_qualifiers` 只保存函数定义/声明节点上直接出现的返回声明说明符 cv。** 它不是返回类型所有层级的 cv 集合，也不展开别名。`int* const f()` 的 const 位于指针层；`auto f() -> const int*` 的 const 位于尾置返回类型；两者都可能具有空的此列表，而完整拼写仍在 `return_type` 中。不能由空列表推断返回类型没有 const。
- `parameters` 保留源码参数的顺序，`index` 为该列表的从零起序号。参数名、类型、默认值和物理位置另行保存；隐式 this 不伪造成源文件写出的参数。
- `template_parameters` 可能包括所在类的模板环境。多个物理声明合并到同一实体时，必须按具体 occurrence 及其词法祖先判断参数来源，不能将不同声明位置的同名 T/N 串成一个模板参数列表。

## 数据声明

- `declared_type` 是声明的基础类型说明符，而不是完整对象类型。对应的 `declared_type_role` 为 `declaration_type_specifier`。
- `declarator` 保留带名称的声明器，可能含指针、数组和初始化相关语法；不能把它简单当成变量名称。
- `declared_type_spelling` 组合基础类型、显式 cv 与移除名称/初始化后的抽象声明器。例如 `element_type __elems_[N]` 对应 `element_type [N]`。它仍是源码类型拼写，不执行 auto 推导、别名展开或数组长度推导；属性及存储说明另行保留，不能把它当作完整ABI类型证明。
- `initializer`、`qualifiers`、对齐属性及位域信息是独立记录。基础类型、声明器、完整拼写和原始声明必须相互一致，不能只校验其中一个字段。

## 同一实体的不同出现位置

`entity_id` 与 `declaration_occurrence_id` 分开。同一源码实体可以在多个位置、不同条件下出现；每处都必须保留完整签名、模板环境和条件。

图集中的 `selected_declaration_conditions` 只说明当前主显示位置。`declaration_availability` 以 any_of 保存各位置的预处理条件组，组内为合取、组间为析取；它不保证模板实参良构，也不代表硬件支持。不能以首处条件替代整个实体的条件，更不能把互斥出现平铺成一个AND。

## using 的已知身份缺口

当前部分 using 记录把目标表达式拼入所在namespace，产生并不存在的C++限定名。这是提取身份错误，不是模板尚未实例化。普通导入、namespace directive和继承构造器的源码声明、词法owner、查找作用及实际目标必须分开；修复方案与迁移要求见 [using接口模型](using-interface-model.md)。该缺口关闭前，相关图只能保留明确的源码导入/查找关系与未完成端点，不得绑定伪QName或宣称目标解析完成。
