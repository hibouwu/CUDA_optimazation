# using 原文分类层：实现、独立库存核对与边界

本轮只新增 `scripts/using_syntax.py`、`tests/test_using_syntax.py` 和本稿。没有修改 Extractor、Reconciler、全局数据或站点，也没有编辑冻结的独立 oracle。此层是物理源码的分类和分段，不是 C++ lookup 求解器，不构成 using 目标／全部 API 覆盖已完成的结论。

## 1. 接口与实际字段

```python
scan_source_usings(path, source: bytes, commit) -> {
    "schema_version": 1,
    "commit": ...,
    "path": ...,
    "source_sha256": ...,
    "sources": [...],
    "namespace_aliases": [...],
    "counts": {...},
    "diagnostics": [...],
    "scope_or_target_resolution_performed": False,
}
```

函数不读取调用者文件、不展开宏、不运行预处理或 Extractor，不写输出文件。输入应是与 path 对应的原始 bytes；Core 如在虚拟投影上使用它，必须另保存虚拟坐标系和源映射，不能把虚拟 offset 当作原始物理 ID／位置。

词法器通过相邻绝对路径 `spec_from_file_location` 加载，使用独立模块名，既不依赖 cwd，也不要求调用者把 scripts 放进 sys.path。独立测试以 `python -I`、cwd=/tmp、按绝对路径动态加载模块核对了这一接入方式。

每个 `sources` 记录包含：

- `source_using_id`：由 schema version、written-using family、commit、path、原始半开区间与原文 hash 构造；没有 owner、target binding、最终类别名称或 constructor 判断。同一物理位置重新绑定宏不应改变这个源码身份；同名 using 的不同出现不会合并。
- `kind`：`alias`、`import`、`directive`、`unsupported`；`syntax_status` 区分原文形状分类、未支持形状和 malformed。该字段不是编译器合法性／clean parse 状态。
- `source_range`、`raw_signature`、`raw_sha256`，以及 keyword／terminator 范围。所有区间采用原始 UTF-8 bytes 的半开边界和 1-based 行号。
- `declared_name`／`declared_name_range`：仅 alias 有声明名；普通 import／directive 不假装声明 target 名字对应的新实体。alias 属性通过 `attribute_ranges` 保留。
- `target_source_expression`／`target_source_range`：保留原始 target，包括写出的 typename。`target_name_source_expression`／range 去掉 leading typename，仅用于 qualifier 和末名分段；关键字另有 `typename_range`，并未丢弃。
- `leading_global_scope_range`：原文明确出现的前导 `::`。宏未来展开成全局限定的情况不在这里提前推断。
- `qualifier_segments`：逐段原文、范围、token spelling、语法类别与可用的 head name。只在模板／圆括号／方括号／花括号之外切分 `::`，不使用 `str.split('::')`。这些 qualifier 可能代表 namespace、class 或依赖类型；字段本身不判断其语义 owner 类别。
- `terminal_name`／range／kind：普通标识符、operator-function-id 或 RHS 的 template-id；一般 alias 类型表达式的末名允许为 null，不能把 `int`、`decltype(...)` 等硬造为模板名字。合法 `decltype(expr)::type` 的 qualifier 标为 `decltype_specifier`，没有伪造的 namespace head name。
- `constructor_spelling_candidate`：仅在非 typename import 的末标识符与紧邻 qualifier head 重复时为 true；证据字段明确写明仍需 owner／base binding。没有 `inherited_constructor_import`、构造器签名或最终 access 推断。

所有记录的 `semantic_resolution` 都是 `not_attempted_source_syntax_only`；没有 owner、qualified_name、entity_id、target_type、target entity 或 access 字段。

`namespace_aliases` 使用不同 family、`source_namespace_alias_id` 和 `kind=namespace_alias`，保留 alias 声明名与同样的 target 分段。它不增加 `counts.written_using`。固定实例的精确位置是 **mma_traits_sm100.hpp:428**，而不是早先模型文字里偏一行的 427。

实际样例：[type_traits.hpp:92](../snapshot/include/cute/util/type_traits.hpp:92) 的结果为：

```text
source_using_id = using_src_7e253424ae0f80a8987d4e91
kind = import
raw_signature = using CUTE_STL_NAMESPACE::remove_cv_t;
source_range = [3493, 3531)
target_source_expression = CUTE_STL_NAMESPACE::remove_cv_t
target_source_range = [3499, 3530)
qualifier_segments[0].head_name = CUTE_STL_NAMESPACE
terminal_name = remove_cv_t
terminal_name_range = [3519, 3530)
constructor_spelling_candidate = false
```

`using iter_adaptor<P, gmem_ptr<P>>::iter_adaptor;` 则只有一个 qualifier 段，末名为 iter_adaptor，candidate=true，但仍只分类为 import。

## 2. 固定原文库存与独立检查

实际逐文件读取 scope 的 824 个 snapshot 文件，并核对所有 SHA256 后得到：

| 计数 | 数值 | 边界 |
|---|---:|---|
| written using | 59,819 | 包含局部、模板和所有预处理分支 |
| alias | 59,136 | 不是 59,136 个独立库类型 |
| import | 499 | 不判定目标种类／重载集合 |
| directive | 184 | cute 172、detail 11、SM90 1 |
| unsupported | 0 | 仅说明固定库存形状均被记录 |
| namespace alias | 1 | 独立表，不加入 written using |
| constructor spelling candidate | 83 | 不是已确认的继承构造器或可调用构造器总数 |

固定库存没有新的 diagnostics。683 个非 alias using 分布在 162 文件；typename 六处仍精确位于 gemm_grouped_per_group_scale.h:73–78。

新测试导入**未修改**的冻结 oracle 的 `written_using_forms`，逐个比较 source start/end、分类和完整原文，而不让旧 oracle 导入新生产实现。两者共享已验证词法器，不共享新的 statement-terminator／qualifier 分段实现；没有以生产结果生成 expected。

## 3. 两个精确范围修正：计数不变，旧 oracle 的终止点不够完整

逐 span 核对发现唯一文件差异：[copy_sm90_desc.hpp:295–296](../snapshot/include/cute/arch/copy_sm90_desc.hpp:295)。源码为：

```cpp
using TmaDescriptor = struct alignas(64) { char bytes[128]; };
using Im2ColTmaDescriptor = struct alignas(64) { char bytes[128]; };
```

冻结 oracle 取遇到的第一个分号，因此停在内部 `bytes[128];`，没有包括 ` };`：

- 第一条原结束 offset=12067，完整声明结束=12070。
- 第二条原结束 offset=12138，完整声明结束=12141。

新实现按圆括号／方括号／花括号平衡寻找最外层分号，保留完整 alias 和内联类型表达式。新测试仅对这两条**预先写明的完整原文 literal**补正 expected，并核对旧区间确实比它少 3 bytes；其余 59,817 条继续与冻结 oracle 完全对照。没有修改旧测试／审查，没有修改 candidates.json 或重定义候选分母。

实际 Clang 21.1.8、C++17、stdin `-fsyntax-only` 样本接受这两个声明，并通过相应 sizeof/alignof 静态断言。它说明两条完整语法确实合法，不是为了让库存通过随意扩张区间；不构成设备 ABI 或运行时验证。

这里的 alias 还包含真实匿名 struct 和字段定义。生产源码层只保留完整 target 原文，**不声称替 Core 抽取了这些嵌套类型／成员**；后续 API 分母仍须由原候选和作用域层分别核对。

## 4. 已测试的负例和边界

测试覆盖 alias 属性与字符串内 `=`／`;`，`using Base::operator=`，operator `<`／`,`／`<<`／`[]` 与模板括号／列表逗号的区别，多层模板中 `::`，模板参数圆括号内的比较，typename，前导全局限定、注释／字符串／raw string、拆分 using／末名 token 的续行、CRLF 延续注释及稳定 ID。

对 `using enum` 和不在固定形状内的多 declarator import 列表，保留整个源码、kind=unsupported 和明确诊断；对缺分号、空 alias target、词法未闭合输入保留 malformed／诊断，不丢失 written using。

一般 alias RHS 可以是完整类型表达式，不要求总能取出 terminal name。`using X=decltype(...)`、`using X=int`、函数指针、匿名 struct 等保留完整 RHS，不把其中最后一个 identifier 或字段名当作 alias target 的末名。

本层不判断 C++ 合法性，不推断 namespace/class/function/block owner，不计算预处理条件，不展开 qualifier 宏，不判 base 是否存在，也不求重载、ADL、目标类型或最终 accessibility。它不会选择 `#if 0` 外的单一分支；条件和 Context 由 Core 接入。分类成功不能自行清除已有 parse_error／scope mismatch。

固定审查发现 written using 不位于宏定义／directive 正文。本层本质上枚举词法 using token；若后续输入出现该情况，调用者必须根据物理预处理区间归属它，而不能将其当成已经展开的普通 C++ 声明。虚拟宏展开同样需要独立来源映射。

## 5. 复现与交接

```sh
.venv/bin/python -m unittest discover -s tests -p test_using_syntax.py -v
```

本轮 21 项测试包括固定 824 文件及 59,819 个物理出现的对照、上述边界 fixture 和隔离 importlib 加载；源码 scan 不生成全局输出文件。Core 应以 source_using_id 保持位置身份，再绑定可信 Context、原文条件和后续 using interface，而非把本层末名拼成一个新 entity QName。

最终实跑 21/21 通过，耗时约 13.7 秒。冻结指纹：

```text
scripts/using_syntax.py
5704ebdc0cd6f7a628d855fa15ec93ba05d8b0fb230a381140284b7195bb4275

tests/test_using_syntax.py
9d79ec69c701507382ce29d4420cefaafcc41370e93b8245601bc1e53323e6e7

未修改的 tests/test_using_declaration_independent_review.py
6ff863482913c0866b52b42b9ef01ee997626aa37fd53bd96d724a3604adafa7
```
