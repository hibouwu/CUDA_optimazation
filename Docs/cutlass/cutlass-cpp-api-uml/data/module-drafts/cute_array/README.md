# `cute::array` 完整文件工作包草稿

范围只有固定提交 `8f50b052e1099fb982392a622caab69b97b63128` 的 `include/cute/container/array.hpp`，共 476 行，SHA-256 为 `a5a6c8357cf4311bed395db7dfa8fcc5c28b3d295fdfec9ad2b00b75253ad8ee`。本目录没有正式 module manifest，没有修改全局声明账本、提取器或站点。

这是可重复的源码账本和关系草稿，不是 API 覆盖、全部模板实例或运行时行为已经验收的声明。

## 源码分母与条件绑定

原文共有 **87 个物理声明**：52 个可调用定义、22 个别名、8 个 struct 声明、1 个数据成员和 4 个 namespace 出现位置。52 个可调用定义进一步分为主模板 22 个成员、`N=0` 特化 22 个成员、5 个 free 操作和 3 个 `get` 重载。8 个 struct 包含 6 个定义及 2 个旧 RTC 路径的条件前置声明。文件自身没有宏定义。

这里的“物理”不受当前预处理配置影响。`CUTE_STL_NAMESPACE` 的真实 `std`／`cuda::std` 两分支展开后形成 **92 条声明绑定**；这不应替换 87 个物理位置的分母。四个合并的 std 实体以及重开的 cute namespace 保留各自多个 `source_selectors` 与 `source_occurrences`。不同 std／cuda::std 实体绝不因为短名相同而合并。

共记录 54 个实际写出的形参：19 个函数形参和 35 个模板形参，无默认值。模板形参只在其物理位置计一次，类成员继承的模板环境引用这些参数，不制造新的形参位置。

[relations.json](relations.json) 目前有 262 个节点、509 条关系、5 个关系视图、6 组源码合同和 536 个源码构造义务。节点数包含参数、局部对象、表达式和外部边界，**不是 API 数量**。210 个类型拼写 token 分清声明名字、模板类型／值参数、别名引用、外部 typedef 与模板 family 引用；`size_t`、`ptrdiff_t` 不因为 Tree-sitter 的节点名称而被称作语言内建类型。

## 调用、表达式与对象边界

31 个普通显式 C++ 调用与 9 个重载下标表达式各有唯一源码调用点，共 40 条 `calls` 关系。另有 2 条 `implicit_calls`，专门记录主模板 `fill` 的 range-for 在 C++17 中选择成员 `begin`／`end`，其 `synthetic_expression` 明确不是源码字面调用。

以下构造不计为额外普通 API 调用：

- 两处 `T(0)` 是依赖 T 的函数式类型转换／初始化；T 为类时可能选择构造函数，T 为算术类型时不是一次用户 API 调用。
- 四处 `begin()[pos]` 是返回指针的内建下标；两个 `operator[]` token 则是显式调用里的函数名，并不是第二层下标操作。
- `CUDA_STD_HEADER(tuple)` 是 include 文件名宏展开。
- `e = value`、`t_r[k] = t[...]`、元素 `!=` 保留依赖操作节点和准确操作数合同，既不把所有情况称作内建，也不凭表达式名字虚构用户运算符目标。
- 局部循环变量、range-for 元素引用及 `t_r{}` 是局部对象／初始化义务，不进入 87 个显式声明的库 API 分母。源代码未显式声明构造、析构、复制／移动及赋值特殊成员；生成、删除、平凡性及聚合初始化的良构性由元素类型、长度和语言模式共同决定。

`calls` 的目标包括准确重载与有具体候选、实参绑定、查找路径及缺少绑定的依赖节点。例如，成员 `swap` 的局部 using 引入标准库候选，但调用仍参与 T 的 ADL；free `clear` 调用 `a.fill(T(0))`，不是成员 `clear`。

## 保留的关键合同

主模板只有 `element_type __elems_[N]` 一个数据成员。`element_type=T` 保留 cv，`value_type=remove_cv_t<T>` 不会改写存储。`N=0` 特化没有元素成员，指针接口返回 `nullptr`、逻辑长度为零，但这不等于 `sizeof(array<T,0>) == 0`，更不使 `front`／`back`／`operator[]` 可以安全执行。

非 const `cbegin`／`cend` 先调用 mutable 成员，再转换为 const pointer；const 重载则选择 const 成员。主模板 `back` 调用 `operator[](N-1)`，零长度 `back` 则解引用 `begin()`，注释里的 `rbegin` 不产生关系。

`get` 只有 `&`、`const&` 和 `&&` 三个重载。`get&&` 中具名参数 `a` 是左值，先调用 mutable 下标，再调用 CuTe 自己在 `type_traits.hpp` 定义的 `cute::move`；返回引用仍指向原数组存储。`const array&&` 绑定现有 `const&` 重载。`I<N` 检查在函数体 `static_assert` 内，而非签名 SFINAE；此处 `tuple_element` 自身没有范围检查。

`operator==` 的普通 for 即使 N 为零，也不自动跳过函数体依赖表达式的实例化检查。`reverse` 的 `if constexpr` 才能丢弃另一分支；非零路径应按 `t_r{}` 初始化、从 const 元素赋值、按值返回逐项陈述，不用粗略的 `is_default_constructible` 或 `is_copy_constructible` 替代源码要求。

所有边保存 `source_occurrence_condition_groups`；实体存在性用 `availability.operator=any_of` 组合不同物理出现的条件。`control_contexts` 另记普通 for、运行时 if 与 `if constexpr` 的分支，不能将这些条件混作同一类声明筛选。条件 include、CUDA 13 外部桥接 header、旧 RTC 前置声明及 std 桥接均保留。

## 重复生成与核验

以下命令在图集项目根目录运行：

```sh
.venv/bin/python data/module-drafts/cute_array/author_draft.py
.venv/bin/python data/module-drafts/cute_array/check_source.py
.venv/bin/python data/module-drafts/cute_array/check_source.py --extractor-fixture
.venv/bin/python -m unittest discover -s data/module-drafts/cute_array -p 'test_source_check.py' -v
python3 data/module-drafts/cute_array/oracle_array.py
```

`author_draft.py` 不读取全局账本或导入提取器；它从原始字节提取源码分母，只用等长空白替代 52 个 `CUTE_HOST_DEVICE` token 进行 Tree-sitter 解析，原始签名及真实宏定义另外保存。它仅原子更新本目录的 `relations.json`。

`check_source.py` 不导入作者脚本，独立遍历原文并核对物理声明、形参、构造义务、精确调用与类型引用、每条关系的源字节、namespace 条件和各视图端点。默认不访问全局提取器或账本。`--extractor-fixture` 只增加当前提取器的单文件对照：本次 92 条 selector 均唯一匹配、92 个不同 occurrence、0 诊断；没有把这个新鲜小样本伪装成已读全局 canonical。

32 项测试本次全部通过，其中 1 项正控和 31 项破坏测试。除一般错误重载、错误 owner、漏条件、伪造构造函数外，还包含同时删除调用／类型引用／alias／归属关系及其全部图引用的反例，防止单靠悬空引用检查通过后遗漏真实源码义务。

[oracle_array.py](oracle_array.py) 与 [oracle_results.json](oracle_results.json) 保存实际 Clang 21.1.8、C++17 host syntax-only 的 6 组样本、14 次编译，8 个通过、6 个预期拒绝，14/14 匹配。完整样本及其边界见 [oracle_review.md](oracle_review.md)。没有生成或运行可执行文件，也没有 GPU、PTX/SASS、性能或一般运行时安全验证。

[oracle_draft_review.md](oracle_draft_review.md) 保留独立审查的四项原始缺陷与追加关闭证据。它们涉及 merged std 实体条件、预处理表达式阶段、条件 include 与 tuple 形参 owner；不以只通过可调用数量来替代语义复审。

四项定点关闭后，作者又补了两条 source-driven 特化关系：第 465／470 行 std 桥接在旧 RTC 路径上关联第 457／460 行本文件明确写出的 primary 前置声明；其他路径仍关联外部 primary 边界，条件互斥。这两条增加关系及其删除测试已通过本目录检查，但尚未获得第二位审查者的定点复验，不冒用此前四项独立关闭结论。

## 接入前仍需处理

1. 将严格 selector 接入 canonical，并独立核对每处物理出现的 entity／occurrence／variant；正式 enrich 时不能用首个 occurrence 的条件覆盖整个合并实体的 `availability`。
2. `type_traits.hpp:92` 的 using 声明与引入的 `cute::remove_cv_t` lookup 名字必须分开。当前草稿只保存真实 using 位置、目标表达式和两条外部候选，不绑定已知错误的旧 canonical QName，也没有伪造 alias 定义。
3. 外部 std／CCCL 声明、ABI typedef 来源、CUDA 13 桥接 header 内容、RTC／CUDA 编译路径仍是明确边界；host 样本不验证这些配置。
4. 所有 T/N 实例与用户 ADL 候选仍依赖具体绑定。尚未 stage、生成正式逐边图或进行浏览器验收，不能宣称该工作包已经完整通过。
