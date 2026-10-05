# 固定快照的 using 接口数据模型方案

状态：**设计方案，尚未接入提取器、对账器或站点。** 本文只针对提交 `8f50b052e1099fb982392a622caab69b97b63128` 的实际源码形状；不实现通用 C++ 名称查找、不重建全库，也不把已知错误身份换一个显示名称后宣称修复。

依据是已冻结的 [独立审查](../audits/using-declaration-independent-review.md)、现有 [Extractor.alias](../scripts/extract_declarations.py:1018)、[Extractor.record](../scripts/extract_declarations.py:721) 以及 [Reconciler](../scripts/reconcile_candidates.py:1302)。本轮没有修改这些文件或既有数据。

## 1. 要改变的对象不是目标类型，而是“这处 using 做了什么”

`using src::f;`、`using namespace src;`、`using Base::Base;` 都有可定位的源码出现，但并不分别定义一个 `owner::src::f` 函数、`owner::src` namespace 或带任意签名的 derived 构造函数。它们首先是**有位置、有作用域、有条件的声明或名称查找动作**，之后才关联已经存在的目标。

因此建议保留一个独立的、以原文为身份依据的 `using_interfaces` 集合。它可以与现有 `occurrences` 共存并互相引用，但普通 import、directive、继承构造器的接口记录不进入目标实体的身份键，也不自动拥有 C++ `qualified_name`。类型 alias 和 namespace alias 有自己的声明名字，应继续作为有名声明处理，但不能因此新建一份底层类型或 namespace。

图页可以给每个 using 接口一个独立入口，其标题直接显示原始声明；这与为它虚构一个 C++ QName 是两件事。

## 2. 已核对的固定源码形状与不变量

冻结审查按 scope 的 824 个文件及各文件 SHA 扫描原文：59,136 个 alias declaration、499 个非 alias import、184 个 namespace directive，共 59,819 个 written using。683 个非 alias 出现分布在 162 个文件。这是**词法库存**，包括函数内声明、模板和未激活分支；不是 59,819 个库 API，也不替代现有 `candidates.json` 的候选分母。

本次方案必须保留以下区别：

| 原文形状 | 模型类别 | 能直接确定的内容 | 不得据此确定的内容 |
|---|---|---|---|
| `using X = T;`，含 alias template 与属性 | `alias_declaration` | 新的 alias 声明名 X、目标类型表达式 T | 新建一个与 T 不同的底层类型 |
| `using Q::name;` | `ordinary_import` 或待语义分类的 import | 原始 target、词法 owner、单个引入名字 | target 必为类型、target 数量必为 1 |
| `using typename Base::T;` | `dependent_type_import` | typename 关键字、引入名字 T、依赖 target | 已经绑定某个具体成员类型 |
| `using namespace Q;` | `namespace_directive` | 提名 namespace 的表达式、声明点和词法 scope | 新 namespace 名字、目标成员的静态复制列表 |
| class 中经基类证明的构造器导入 | `inherited_constructor_import` | derived owner、base 表达式／绑定、原 using 位置 | using 自己的构造函数签名及最终可访问性 |
| `namespace A = Q;` | `namespace_alias_declaration`，独立于 written using | alias 名 A 与 namespace target | 新 namespace 本体、directive 的查找语义 |

固定范围没有 `using enum`。分类器遇到该形式应保留一个明确的 unsupported/source-review 记录，而不是宣称支持。审查中的 83 项末名重复只是继承构造器**拼写候选**；确认类别必须再验证 class owner、实际基类及 alias/template 绑定。

`functional.h:990` 的 `using red [[deprecated("use atomic_add instead")]] = atomic_add<T>;` 要求识别 alias 名后的属性，不能要求 using 后第二个 token 必为 `=`。审查负例 `using Base::operator=;` 则说明不能看到任意 `=` 就当 alias。续行、注释、字符串中的 `;`／`using` 应沿用已核验词法器处理，不能改成逐行正则。

## 3. 五层记录：位置、上下文、作用、目标表达式、目标集合

下列字段名是建议的新 schema，不代表它们已经写入全局数据。

### 3.1 `source_using_occurrence`：不可被语义分支替代的原文位置

每个真正 written using 保存一条物理记录：

```text
source_using_id
snapshot_commit
path
source_range {start_byte, end_byte, start_line, end_line}
raw_signature / raw_sha256
syntax_form {alias | import | namespace_directive | unsupported}
keyword_ranges / declarator_ranges / attribute_ranges
target_source_range / target_source_expression
source_candidate_refs[]
lexical_classification_proof {lexer_fingerprint, rule_id, evidence_ranges[]}
```

`source_using_id` 建议由提交、path、物理范围、原文 hash、稳定的 written-using 类别组成，**不要含最终 target、owner QName、构造器候选或解析器生成名字**。同一位置展开为 std／cuda::std 时仍是一条物理记录。同一 namespace 里两次写相同 using 则是两个物理记录；它们的声明点与当时可见声明可能不同，不能按目标短名合并。

保留整个声明和 target 子区间；`typename`、前导 `::`、模板参数、属性各有自己的范围。原始 target 去掉的是 using 语法本身，不得因展示需要丢掉 typename 的独立语法证据或截断模板实参。目标限定符里出现 `::` 的模板实参不能用简单字符串 split 分段。

namespace alias 的物理记录使用独立 `source_namespace_alias_id`，不加入 59,819 个 written using 数量。固定例子为 [mma_traits_sm100.hpp:427](../snapshot/include/cute/atom/mma_traits_sm100.hpp:427) 的 `namespace TMEM = cute::TMEM;`。

### 3.2 `lexical_context_variant`：所在位置，不是目标名字的一部分

```text
context_variant_id / source_using_id
scope_ref
scope_kind {namespace | class | function | block | local_class}
scope_chain_refs[]
enclosing_function_ref? / lexical_block_ref?
scope_binding_conditions[]
source_preprocessor_conditions[]
template_environment_refs[]
declaration_point {path, byte}
lexical_access {value | conditional decision list | not_applicable}
scope_integrity {verified | pending | mismatch, proof_refs[]}
```

namespace／class 的 owner 引用经验证的实体或参数化 scope 记录；函数／块引用真实函数身份加物理块区间。局部 class 的成员 using 属于该局部 class，不能被“祖先中有函数”一概降成普通局部表达式。函数／块的显示路径可以帮助导航，但不可把 `<block@...>` 当作合法 C++ QName 组件。

词法 access 仅在 class 成员语境有意义。namespace、函数内 using 没有 public/private 成员访问控制，不从当前 Context 默认 public 制造这个属性。条件访问标签仍保留已建立的 last-active-label 决策证据，不折叠成一个默认标签。

若原始／投影树的 owner 受到 scope-integrity 警告，保留物理 occurrence，但该 context 不得标 verified。不能因为 `using` 这一行能被词法拆开，就信任损坏树中的 class/function 祖先。审查已给出原始树把 visitor directive 塞进 class、把 scheduler 构造器导入塞进函数的真实反例。

### 3.3 `using_interface_variant`：声明对查找的作用

```text
using_interface_id / source_using_id / context_variant_id
semantic_form
introduced_lookup
target_expression_ref
binding_conditions[]
availability
source_contract_status
relationship_obligations[]
```

`using_interface_id` 是接口记录身份，不是被导入实体身份。建议以物理 occurrence、context variant、实际源条件分支及语义类别建立，避免依赖尚未求出的 target 集合。一个 source import 最终解析出更多重载时，不应导致它的源码身份变化。

`introduced_lookup` 按类别区分：

- ordinary import：`{kind: ordinary_name, spelling: name, lookup_scope_ref, source_name_range}`。namespace/class 中可附 `qualified_lookup_spelling`，例如 `cute::remove_cv_t`；该字段表示一种查找写法，不作为“新函数／新类型”的 entity QName。若 owner 参数化、为函数块或尚未确定，就不生成该字段。
- dependent typename import：同样引入末名，另保存 `requires_type_target=true` 与 typename 源区间；这是一项语法要求，不是具体 target 已解析。
- namespace directive：`{kind: none}`，不保存伪造的 local name。它有独立 `nominated_namespace_expression`，不是 `introduced_lookup.spelling=target namespace`。
- inherited constructor：`{kind: constructor_family, derived_type_ref}`。可以显示“从 Base 继承构造器”，但没有名为 `Derived::Base::Base` 的导入函数；也不从某编译器 UsingDecl.name 复制一个派生构造器名字。
- alias declaration：自己的 alias entity 与 `declared_name` 合法保留；target 类型关系独立。namespace alias 则保留自己的 alias 声明身份和 nominated namespace 的关系，不与 directive 共用语义类别。

一个物理 target spelling 和可验证基类 binding 尚不足以分类时，使用 `semantic_form=import_classification_pending`，附明确候选及缺少的证明。不能把 83 个拼写候选强制全部标为 inherited constructor，更不能把这种分类失败改成“symbolic target 已完整”。

### 3.4 `target_expression`：原文、条件展开与来源映射

```text
target_expression_id
source_expression / source_range
has_typename_keyword / typename_range?
root_qualified_in_source
qualifier_source_segments[] / terminal_name_source_range
expansion_variants[] {
  conditions[], expanded_spelling, expanded_root_qualified,
  macro_definition_chain[], macro_bindings,
  source_mapping[], missing_external_bindings[], validation_status
}
```

原文 `CUTE_STL_NAMESPACE::remove_cv_t` 与展开的 `std::remove_cv_t`／`cuda::std::remove_cv_t` 分开保存。不能把展开字符的虚拟 offset 写成物理 offset；映射回宏 token 的区间与展开后的 spelling 也不能互相冒充。

同样要允许**合法的空宏展开**：[helper_macros.hpp:136–140](../snapshot/include/cutlass/detail/helper_macros.hpp:136) 在 `__CUDA_ARCH__` 下把 `CUTLASS_CMATH_NAMESPACE` 定义为空。`CUTLASS_CMATH_NAMESPACE :: isnan` 此时变成 `::isnan`，根限定是确定结果，不是“namespace 丢失”的 pending；另一分支为 `std::isnan`。用非空 namespace 字符串作为唯一成功条件会再次误判真实源码。

[platform.h:151–157](../snapshot/include/cutlass/platform/platform.h:151) 对 `CUTLASS_STL_NAMESPACE` 的定义受 `!defined(CUTLASS_STL_NAMESPACE)` 保护。未提供外部 override 时，必须保留“外部已绑定／本地默认两分支”这层区别，不能擅自只列 std／cuda::std 两个穷尽结果。宏先后生效、undef、本地遮蔽和多层定义来源仍由既有 namespace/macro 证据机制证明，不能对每个含宏 target 直接全局 replace。

namespace scope 的条件与 target 宏条件使用同一组可追溯条件原子，保留相关性；不要把两个选择组盲目笛卡尔展开，产生源代码不可能同时成立的 owner／target 组合。恒假源分支仍保留物理记录，明确 inactive；不能删除来减少分母。

### 3.5 `target_resolution`：零到多个目标，不是一个 `target_type`

```text
resolution_id / using_interface_id
lookup_phase {declaration | instantiation | use_site}
lookup_point / configuration_evidence / template_bindings
status {not_attempted | dependent | partial | complete | external_boundary |
        inactive | confirmed_rejection | extraction_error}
targets[] {
  target_ref: entity_ref OR external_ref,
  target_kind,
  declaration_occurrence_refs[], conditions[],
  resolution_basis, lookup_path_refs[], evidence_refs[]
}
membership_completeness {complete_for_bound_context, universe, proof_refs[]}
missing_bindings[] / blockers[]
```

`targets=[]` 必须和状态同时解释，不能默认“已证明没有目标”。它可能是未求解、dependent、外部定义未取得、未激活分支或前端已确认的非法声明；只有明确的证据才能使用 `complete`／`confirmed_rejection`。对 active ordinary import，完整解析却没有任何声明目标是需要复查或前端拒绝证据的情况，不应静默成功。

一个函数 import 可以关联多个 function entity，每个重载保留自己的完整签名与声明位置。`membership_completeness` 必须针对固定配置、声明点和具体模板绑定说明，不能因为找到一个 target 就宣称全重载集合完整。未取得外部标准库定义不等于 fixed-scope `missing_definition`，但 external boundary 也不是调用点已决议。

`target_kind` 不能从 using 语法或后缀猜测：实际库中有类型模板、alias template、函数模板／重载、变量模板、枚举值和数据成员。目标类别只有在真实声明或前端证据支持时才填写具体值；语法要求可另放 `expected_target_category`，不与已证 target_kind 混淆。

## 4. 几个真实位置如何落入模型

### 普通 import 与 overload

[type_traits.hpp:92](../snapshot/include/cute/util/type_traits.hpp:92) 对应一个 source occurrence，lexical owner 是该文件第 58 行重开的 cute namespace。target source 为 `CUTE_STL_NAMESPACE::remove_cv_t`，introduced lookup spelling 为 `remove_cv_t`，可表示的 namespace 查找写法为 `cute::remove_cv_t`。其两个 target expansion variant 依据 [config.hpp:106–111](../snapshot/include/cute/config.hpp:106) 的固定定义，原文 occurrence 不随宏分支复制。

这里不能设置 `qualified_name=cute::CUTE_STL_NAMESPACE::remove_cv_t`，也不能把 using 记录变成 `cute::remove_cv_t` 的新 alias template 定义。以后 array 的 `remove_cv_t<T>` 引用先关联这条导入路径，再关联实际 target／模板绑定；不用手工造 alias 节点绕开全局错误。

[complex.hpp:40–45](../snapshot/include/cute/numeric/complex.hpp:40) 的六个导入同理。`cutlass::conj` 的一个 source import 可能导入多个重载，不能只留下第一个。独立 Clang 缩减样本证明，一处 `using src::f;` 对应两条不同 FunctionDecl 的 shadow target，而 `using X=src::T;` 才产生 TypeAliasDecl；编译器的 shadow id 仅作为该次 oracle 证据，不能直接当持久 entity ID。

[copy_traits_sm100.hpp:464](../snapshot/include/cute/atom/copy_traits_sm100.hpp:464) 的真实 target 定义在 [copy_sm100.hpp:618](../snapshot/include/cute/arch/copy_sm100.hpp:618)。可以在 fixed source 上先实现这类“namespace import → 唯一已知非函数实体”的最窄正向绑定，但必须同时验证完整 qualifier、namespace variant、条件和真实声明类别；不能只比较尾部 `SM100_TMEM_LOAD_16dp256b1x`。

### namespace directive 与 namespace alias

[sm90_pipeline.hpp:52](../snapshot/include/cutlass/pipeline/sm90_pipeline.hpp:52) 应有 directive 接口记录、cutlass lexical scope、target expression `cute` 和 nominated namespace 的关系。没有 `cutlass::cute` 这个新 namespace entity。directive 不复制 cute 的全部 API，也不预先决定后面每个未限定名字的解析结果。

directive 的 effective lookup context 不能只用词法 owner 代替 C++ 查找规则。模型保留 lexical owner、提名 namespace、声明点、block extent，以及“最近公共 namespace 等规则尚未执行”的 resolution 状态；不做静态全员复制、不把后续成员声明纳入一个永久冻结快照。后续某个 use site 还需结合遮蔽、可见声明与 ADL 求解。

`namespace TMEM = cute::TMEM;` 则有 `cute::UMMA` owner、声明名 TMEM 和合法 alias 查找写法 `cute::UMMA::TMEM`；被别名引用的 namespace 仍是 `cute::TMEM`，不是再建一个同成员列表的 namespace。

### class 导入、局部导入与构造器继承

[sm90_visitor_tma_warpspecialized.hpp:602](../snapshot/include/cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp:602) 的 `using Impl::ops;` 有 derived owner，Impl 在 589 行绑定基类模板，真正存储在 583 行。图可以有 `member_import` 关系，但不能再建一份 `Derived::Impl::ops` 字段、资源或初始化事件。普通成员导入的 using access 影响该类中的名称可见性表示，而资源和实现所有权仍属于目标声明。

[gemm_grouped_per_group_scale.h:73–78](../snapshot/include/cutlass/gemm/kernel/gemm_grouped_per_group_scale.h:73) 的六项 typename import 应保存 Base 的第 70 行绑定与模板环境。当前 parser 错误必须显式处理：可以后续使用原文类别证明和 source-mapped 适配恢复这六种固定形状，但在未验证 owner、完整范围和投影前，不能仅去掉 name 中的 typename 就把 `contains_parse_error` 改成 clean。

[array.hpp:188](../snapshot/include/cute/container/array.hpp:188) 属于 `swap(array&)` 的函数块；source occurrence 及局部 lookup 关系不可遗漏，但不增加一个 namespace API。第 190 行未限定 `swap` 的解析义务引用该 using，再保留标准库普通查找候选和 T 的 ADL 候选。函数内 alias／ordinary import／directive 必须分别分类，不继续统称 `function_local_alias`。

[sm90_pipeline.hpp:161／165](../snapshot/include/cutlass/pipeline/sm90_pipeline.hpp:161) 为两个不同 derived owner 的继承构造器导入，保留两处源记录。它们的 `lexical_access=private` 来自 class 默认标签，**不推出最终 inherited constructor 为 private**。需要另外记录：

```text
constructor_inheritance {
  derived_type_ref,
  base_specifier_range,
  base_expression,
  base_binding {status, alias_chain[], template_bindings, target_type_ref?},
  source_using_access,
  constructor_candidates[] {
    base_constructor_ref, original_base_access, signature_ref,
    language_rule_evidence[], viability_status, accessibility_status,
    argument_bindings?, use_site_ref?, conditions[]
  },
  special_member_interactions {status, evidence_refs[]}
}
```

只有 use site 和配置／实参等证据足够时才给出可调用结论。固定 Clang oracle 已证明：private using 不阻止 public 带参基类构造器的外部调用；deleted default 仍拒绝；public using 也不开放 protected 基类构造器。不能把这个 oracle 推广为所有 copy/move/default 构造器都按统一规则继承，也不能从 source using 补写一个没有原文的函数定义。

## 5. 当前实现为何会把错误记录对账为成功

本轮读取的核心 SHA256 为：

```text
extract_declarations.py
da8d2007310e15cbb5abe6924e402e2d71789887419da94aab34d608395abff1

reconcile_candidates.py
9ea91ad5a2ebe41c8914b4418d3381131dd9baabeca2e05cb998730334b29a79

using-declaration-independent-review.md
0535743ffc78651480907382f467ef92a11b9a45088c52fce3b5dd60a34c48c6
```

`Extractor.alias` 对 using_declaration 取第一个 named child 的全文作为 name，再将同一文本放到 target_type。`record` 经 [qualified](../scripts/extract_declarations.py:687) 把全部 context scope 名字与这个 name 拼接；entity key 同样纳入该复合 name。因此错误不只是显示字段：既有 identity 的生成依据本身已经错误。

`record` 的 source_occurrence_id 还含 kind/name，不能假定修正 name 后自然保留旧 source ID；迁移需要物理范围桥接。它计算的 class access 可以保留为 lexical_access，但不适合作为 inherited constructor 最终 access。

Reconciler 的 `ALIASES` 目前包含 alias、typedef、using_declaration、namespace_alias；`head_kind` 把 written using 都归 alias，`compatible` 又允许该类别匹配 ALIASES 或 TYPES。`invalid_occurrence` 核对 QName 是否等于 scope_chain 加 name，以及 name token 是否在签名里出现，这只验证**同一错误构造过程的机械一致性**，并未证明 using 的语义。

`build_expression_contexts` 中的 alias RHS 处理依赖 `=`，不够覆盖普通 import、directive 和构造器导入；`slim_occurrence` 也只有统一 target_type，后续若不扩充，会在对账前丢掉新的分类／绑定证据。`record` 的 friend-type lookup 目前再从 `using_declaration.target_type.split('::')[-1]` 找候选，需要一起改为 introduced_lookup 与原 using 接口引用，否则 typename／operator／模板 qualifier 仍会误判。

本轮实际在内存中跑了三个自包含小样本，经当前 Extractor → slim_occurrence → Reconciler，没有写文件或读取全库账本：

| 输入中的 using | 当前 occurrence 结果 | 当前对账结果 |
|---|---|---|
| `namespace dst { using src::f; }`，src 有 int/double 两重载 | `name=src::f`，`qualified_name=dst::src::f`，parsed | `mapped_occurrences / phase1_declaration`，无 pending reason |
| `namespace dst { using namespace src; }` | `name=src`，`qualified_name=dst::src`，parsed | 同样映射成功，无 pending reason |
| `void g(){ using src::f; f(1); }` | using 不输出 API occurrence | `classified_non_api / phase2_expression / function_local_alias` |

最后一例没有新增 namespace API 是合理方向，但 `function_local_alias` 分类不准确，且不能替代后续 lookup／调用关系义务。前两例则说明只修改 extractor 的显示、却保留现行泛化 ALIASES 放行规则，无法形成有效修复门禁。

## 6. 候选分母与阶段义务如何保持

现有 `candidates.json` 保持原 candidate_id、path、范围、raw hash 和行数。新 written-using 库存是一层独立子区间索引，通过 `source_candidate_refs` 关联已有 syntax_interval；不能删旧候选、将一个宽候选换成数量更少的 using 记录，或用新分类数重新定义旧候选分母。

一个候选可能覆盖模板头与 using 声明等多个构造，一个 written using 也可能与多个既有区间交叠。映射保存关联表和 exact token coverage，不假设两套分母一一对应。继续要求现有 Reconciler.run 的每个候选恰有一条最终结果，同时在结果中明确附加 using 接口和关系义务。

建议分开两种验收状态：

- **源码接口合同**：物理范围、形状、lexical owner／access、参数环境、宏分支、introduced lookup 类别均有证据。ordinary/directive/constructor 不能靠一个伪 QName 标为通过。树损伤或无法证明分类保留阶段 1 blocker；固定依赖表达式已完整记录、仅缺具体实例时可以关闭结构提取义务，但必须保持后续 target resolution 未决。
- **名称查找与关系**：0..N target 集合、重载完整性、use-site 遮蔽／ADL、构造器可访问性等独立义务。source contract verified 不等于该集合已完整解析。函数内 using 可不计独立库 API，但必须有 source record、明确 local 类别和可追踪的阶段 2 obligation。

候选结果可暂保持 `mapped_occurrences` 或 `classified_non_api` 等外层状态，但新增 `source_using_ids`、`using_interface_ids`、`source_contract_status`、`relationship_obligation_refs` 与 `resolution_status`；分类名改为具体的 `function_local_using_import`、`function_local_using_directive`、`function_local_alias_declaration`。不能把有目标未决的 using 一概塞进 `phase2_expression` 后停止对账。

遇到类型未知不要默认 `target_type`；遇到错误 QName 也不要把它重命名为 symbolic。前者可能是精确依赖表达式，后者是已确认 extraction_error，需要撤销旧语义成功标记。

## 7. 旧实体与模块端点的升级

迁移前先锁定旧 canonical、候选数据、生成器及修复版各 SHA。建议建立单独的升级索引，而不是原位修改 JSON 中所有同短名实体：

```text
legacy_using_upgrade {
  old_entity_id, old_occurrence_id, old_source_occurrence_id?,
  old_kind, old_name, old_qualified_name,
  source_join {commit,path,physical_range,raw_hash,matching_rule},
  new_source_using_id,
  new_interface_variant_ids[],
  action {retain_alias_identity | supersede_fake_entity | reclassify_local |
          retain_pending_source_mismatch},
  reason, evidence_refs[], downstream_revalidation_refs[]
}
```

保持正确 alias 的既有身份应是默认方向，不为了改 using 表把 59,136 个 alias 全部重新编号；确实受属性／owner 解析错误影响的 alias 另按精确证据处理。普通 import、directive、继承构造器的旧伪 entity 不可直接复用为新的 lookup interface 身份，也不可当作其 target 的别名 ID。旧 ID 可保留历史入口和一对多迁移记录，但其旧错误 QName 必须明确 retired／superseded。

物理 join 必须核对原始范围、内容、类别与 scope 证据。旧 source_occurrence_id 会因 name 修正而变化，不能把旧 ID 相等当作唯一迁移条件，也不能只按行号／末名配对。若一处旧记录混淆多个语义类别、缺失范围或仍处于受损 scope，保留 pending 升级，不猜目标。

旧图的入边需要按实际语义重审：

- 若边表示“引用／经过这处导入”，改连新的 using interface，再关联已证 target。
- 若边声称调用具体函数，必须重做该 use site 的 overload/ADL 验证；不能把旧 fake using 节点简单重定向到任意一个 target。
- 若旧 directive 被伪造成 `cutlass::cute`，不能将全部入边批量改指向 `cute` namespace 后声称调用／类型都解析完成。
- 若旧 class import 被当作字段资源，删除的是伪造的新存储语义，实际基类字段保持原身份；相关 reads/writes/初始化边须重新定位，不静默合并两个资源。
- inherited constructor 的使用点只能关联已验证 base constructor／继承机制记录及对应实例，不能从 using 位置凭空补一个无来源派生构造器定义。

多个物理出现归到同一查找展示项时，availability 必须是各 occurrence 条件 conjunction 的 **OR**；每条 target／use-site 边又只使用其自身出处条件。不能沿用首个 occurrence 条件屏蔽后续分支，也不能把互斥定义变成 AND。

## 8. 有界实现顺序与回归门禁

最短路径是复用已经验证的 [written_using_forms 库存与 oracle](../tests/test_using_declaration_independent_review.py)，不要再复制一份弱化的 using 正则扫描器。后续实现可分三个有明确停止点的步骤：

1. 增加 source-backed 分类与接口记录，覆盖固定 alias／ordinary import／directive／六项 typename／继承构造器候选及 namespace alias。保持原 scope guard、所有条件及候选分母；尚不能确认构造器类别的具体位置保留 pending，不扩大成通用 lookup 求解器。
2. 给已知固定宏 target 做可追溯分支绑定，并在最窄正向案例上关联已知声明，例如 copy traits 的 namespace target、P90 明确基类构造器、Impl::ops。目标集合、外部边界和实例化缺失绑定保持结构化；不要求一轮求出所有标准库或模板实例。
3. 同时升级 slim_occurrence／Reconciler 规则、friend lookup 消费者和模块 selector／显示，再做全量候选不变对账与受影响模块端点复审。不能只输出新字段而让旧 QName 验证路径继续机械放行。

最低回归集合应包含以下可具体拒绝的破坏：

1. 独立库存仍为 59,136／499／184；所有 824 文件 hash 核对，alias 属性、注释、字符串、续行和 `operator=` 负例不误分。库存只是覆盖验证，不自动证明每处 owner／target 正确。
2. 把 `cute::remove_cv_t` 查找写法改为 `cute::CUTE_STL_NAMESPACE::remove_cv_t`，或把 `using ::cuda::std::swap` 接到 cutlass owner 后，必须拒绝。
3. 删除 std／cuda::std 任一 target 宏分支，吞掉 CUTLASS_STL_NAMESPACE override guard，或将合法空展开 `::isnan` 改为 unknown，必须拒绝。
4. 同一 using 的两个函数重载 target 只留一个，即使同步修改 count，也必须被有绑定范围的 oracle／目标集合完整性检查拒绝。一个 target 被输出不等于完整。
5. directive 增加伪 declared_name、复制成员列表、泄漏函数块作用域或忽略后续遮蔽，必须拒绝；namespace alias 不得落入 directive 规则。
6. Impl::ops 被生成第二份字段存储，或 member target kind 被统一改为 type，必须拒绝。
7. typename 六项丢掉关键字／Base 绑定／模板参数，或者 parse_error 未有新证明便被标 clean，必须拒绝。
8. 仅凭末名重复确认构造器，伪造 Derived::Base::Base QName，或从 private/public using 推导最终 constructor accessibility，必须拒绝。既有三项 Clang 正反样本继续通过；copy/move/default 边界不扩大。
9. 局部 using 被当 namespace API，或被删除且所有后续 call/lookup 边也一起删掉，原候选与局部关系义务仍必须检出缺口。local class 的成员 using 不被函数父作用域误吞。
10. 旧错误 entity ID 无证据沿用、仅以同短名迁移、丢掉重复物理位置、将一对多 target 升级压成一个目标，都必须拒绝。
11. 对齐与 array 的 selector、复数 occurrence availability、friend-type pending、条件 scope 和已修 P90 归属继续回归，防止 new using 层绕开既有的物理来源和 scope 验证。

本轮没有新增分类原型：现有冻结库存器已经覆盖这些词法形状，再写一份并不能解决当前真正的 identity／lookup 分层问题。本文提供字段合同、当前错误接受路径、迁移方案和回归要求；正式实现及全库升级仍由下一轮有界任务完成。

本轮额外复跑现有测试中的 `UsingClangSemanticOracles` 四项，全部通过；命令如下。它只运行 overload、宏目标、directive 和继承成员／构造器的自包含 Clang 样本，没有运行全库库存测试或全量重生成。上文三个当前 Extractor／Reconciler 小样本另外在内存执行，结果原样保留；它们是错误复现，不是新模型已实现的证明。

```sh
.venv/bin/python -m unittest discover -s tests -p test_using_declaration_independent_review.py -k UsingClangSemanticOracles -v
```
