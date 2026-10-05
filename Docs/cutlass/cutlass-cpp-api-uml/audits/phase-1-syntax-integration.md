# 阶段1：语法投影接入独立审查

本审查只修改独立测试与本审查记录，没有修改五个生成器模块。审查对象是根代理编写的核心投影接入、字段恢复及虚拟声明整合流程，不把模块自身的构造测试当成集成正确性证明。

## 版本边界

本次验证版本：

- `scripts/extract_declarations.py`：`7ac761eb0b18840a6aedf426d225907e50b6e89b81f45c076a8b659034e345e3`
- `scripts/namespace_bindings.py`：`846a9792fac6e01f9ddecc7a4c68e66b1b750d350260f241b1a3d8ad0b0ae384`
- `scripts/declaration_syntax.py`：`09642761655ecc7ad5997536ba99ef0b9ada1298f7a409f2d1a519f198b72b0b`
- `scripts/declaration_projection.py`：`bb2464afa35a159d6adaf45f87d9ff26d9ffcd3c1c1631a2f15a3b02abe52997`
- `scripts/macro_expansion.py`：`9e3e0f2b3debda4ba2cb1f59c14aa0ca3b45f1f29064035ae909bd63af6e9237`

入口：

```bash
.venv/bin/python -m unittest discover -s tests -p test_syntax_integration_review.py -v
```

测试中的合法缩减源码采用Clang 21.1.8、strict C++17交叉核对。当前共有10项测试；已知通过4项，另外6项暴露下列开放问题。最后一项专门输入三个不完整语法片段，要求工具诚实登记解析器的MISSING节点；它不主张这些片段是合法C++。

## SI01：无位置约束的全局字符串恢复污染其他语义字段

复现：

```cpp
template<class T> auto f()->decltype(typename T::X{});
template<class T> auto g()->decltype(sizeof(typename T::X));
```

源码通过Clang strict C++17。f需要解析器的依赖类型括号适配；g并不需要该适配。但接入后的g返回类型字段为 `decltype(sizeoftypename T::X)`，不是源中的 `decltype(sizeof(typename T::X))`，诊断为空。

另一个合法反例：

```cpp
template<class T> auto f()->decltype(typename T::X{});
const char* text="(typename T::X)";
```

常量的`initializer`字段被改成 `"typename T::X"`，同样零诊断。两个反例中的`raw_signature`仍能正确切回源文本，因此仅验证raw签名不能发现错误。

根因是 `extract_syntax_projection()`中的 `restore_strings()`对每个occurrence的全部字符串无条件调用`str.replace`，把只属于某个编辑区间的`(typename T::X)`恢复规则应用到了不相关位置和字符串字面量内部。

需要按字段的虚拟/物理范围恢复对应edit，而不是全局替换同名子串；字面量必须保持原值。对应两个失败测试：

- `test_restoration_cannot_rewrite_unrelated_sizeof_return_type`
- `test_restoration_cannot_rewrite_string_literal`

## SI02：placeholder没有携带访问区间，条件alias及宏生成类型丢失private

条件alias反例：

```cpp
struct S { private:
  using X =
#if A
    int
#else
    long
#endif
  ;
};
```

当前两个X变体均被记录为`access=public`，诊断为空。原声明被替换为`;`后，该位置没有命名AST子节点；`contexts_at()`因此取到`field_declaration_list`进入时的默认public上下文，没有取到后续`private:`建立的有效访问区间。

宏生成类型存在同一问题：

```cpp
#define MAKE(X) struct X {};
struct S { private: MAKE(A) };
```

生成A有正确宏origin和class owner，却仍是`access=public`，诊断为空。

对应失败测试：`test_conditional_alias_restores_private_access`、`test_generated_type_restores_private_access`。应按class内部访问控制区间恢复上下文，不能仅依赖placeholder处是否存在命名AST节点。

## SI03：简单成员声明宏没有展开，也没有明确pending

```cpp
#define MAKE(X) int X;
struct S { private: MAKE(x) };
```

Clang确认这是合法的private整数成员。当前输出x的`declared_type=MAKE`，没有`macro_origin`，诊断为空。宏主体只有`int X;`时没有进入现有声明宏展开候选，之后恢复树将调用误认作普通成员声明。

测试 `test_simple_member_macro_is_expanded_or_explicit_pending`接受真正展开为int并保存宏来源，或者保留明确pending；不接受伪造的普通成员声明作为成功覆盖。SI04的诊断修复可以先消除“零诊断成功”误导，但不等于已补全该宏生成接口。

## SI04：只遍历named_children会漏记MISSING标点

三个独立输入：

```cpp
struct S { int x };
void f(int x;
int x
```

分别单独提取时，Tree-sitter都返回 `normalized_parse_has_error=true`，且其树明确包含 `MISSING ";"`或`MISSING ")"`。当前提取器却返回空diagnostics。

根因是共享`descendants()`只遍历`named_children`；缺失的标点是匿名子节点，`record_parse_errors()`根本没有访问到它们。因此现有`missing_syntax`计数不是所有缺失语法节点的数量，不能用它作为完整解析缺口分母。

测试 `test_unnamed_missing_punctuation_is_a_real_diagnostic`要求三个片段都登记blocking `missing_syntax`。诊断遍历应覆盖`node.children`中的所有MISSING/ERROR节点；声明提取本身仍可使用命名节点路径。

## 已通过的四项接入检查

- `T const& value={}`恢复为真实默认值`{}`，实际签名中没有临时标识符。
- `__CUTE_REQUIRES`展开与默认值投影组合后，函数原始签名的物理字节回切正确，展开签名保留`enable_if`且没有parser临时名。
- 普通class内的两个条件alias分支使用同一个entity、两个variant、同一个源声明ID；逐片source segments能够重建各自virtual source。
- `CUTLASS_CONSTEXPR_IF_CXX17`保留constexpr/空两种条件定义，没有被写成无条件constexpr qualifier。

## 补充观察：template alias边界仍需恢复

`template<class T> using X = #if ... T #else T* #endif ;`的缩减形式当前产生一个明确`missing_syntax`，所以不属于零诊断成功；但生成的两条alias的`template_parameters`为空，原始签名也没有template前缀。模板头与被placeholder替代的声明分离后，所属模板上下文丢失。后续若关闭该语法诊断，需要同时恢复模板身份，不能只让分支virtual source独立解析成功。

## 结论

当前接入不能通过阶段1。SI01与SI02证明“原始签名回切正确、分支计数正确、单文件零诊断”仍不足以证明语义字段正确；SI04则要求先修正诊断节点分母，才能继续解释全量语法统计。

## 追加定点回归：原有SI01–SI04反例通过

在以下版本组合上，原有10个集成测试重新运行，结果为 **10/10通过**：

- 核心提取器：`2ee8e7201fe880caeadc907934bec247fe014798d2bc9fd97aba2927a1befbd8`
- 语义投影映射：`21504e059a7abefbc6c6f012ad037b5f3f1c8ca2afed7361f5f418179db0dbb4`
- 宏展开器：`e6e201372012c3d28f4bc27e8ef29986c190a604442bca0e428964d593a322d2`

SI03不是依靠“新增一条解析诊断”使宽松测试通过：专门读取输出确认 `MAKE(x)`现在产生 `declared_type=int`、`access=private`且带`macro_origin`的真实成员，诊断为空。宏模块新增16项定点测试，所有`test_macro*.py`共34项测试通过。

本次宏变更支持简单/多成员声明、直接函数声明、基本类型token-paste声明，并为调用点提供的分号单独保存`callsite_suffix`与`macro_invocation_end_byte`。对象式声明宏、嵌套声明宏、类型/表达式尚不明确的实参形式、缺少声明终止符、`__VA_OPT__`及未支持的prescan/rescan仍保留明确pending，未猜造成成功。

`Return_Status`的3个调用与平台`static_assert`的1个调用，现在在宏模块的`non_declaration_invocations`中保留原位置及定义依据，不伪造命名API。`cuda_host_adapter.hpp`的两个`CUTLASS_CUDA_DRIVER_WRAPPER_DECL`定义歧义仍然开放。

因此原SI01–SI04的已知缩减反例在该版本组合内关闭；这不是整个宏展开、全部语义字段或阶段1的完成声明。原有失败事实保留在上文，新反例应继续追加而不是改写历史。

## 追加独立复审：组合语义与条件访问通过，两个新边界开放

在上述核心`2ee8e720…`、投影`21504e05…`、宏`e6e20137…`冻结版本上，新增独立入口：

```bash
.venv/bin/python -m unittest discover -s tests -p test_semantic_integration_review.py -v
```

当前6项测试中4项通过、2项失败。四项通过结果比原反例进一步覆盖：

- 同一函数签名内两次相同`typename T::X{}`、`sizeof(typename T::X)`和包含相同文本的字符串，叠加`__CUTE_REQUIRES`及`{}`默认值后，实际返回类型和entity身份均与源语义一致，没有跨位置替换。
- `class S`中两个独立条件访问标签按最后一个生效标签决定权限。四个A/B配置的决策结果与Clang实际成员可访问性一致。
- 实际展开的`FIELD(x)`整数成员同样保留条件访问决策，与Clang两个配置一致。
- 条件private/protected标签不污染friend函数注入：friend声明与外围namespace定义使用同一实体且不是private成员。

### SI05：friend类型声明出现位置静默遗漏

合法缩减源码：

```cpp
namespace n {
  struct S { private: friend struct F; };
  struct F {};
}
```

Clang strict C++17通过，类型F属于namespace n。当前数据只有后一个`n::F`定义，没有前一个friend声明出现位置，诊断为空。Tree-sitter对该源码生成的是直接包含`type_identifier`的`friend_declaration`，不经过现有struct-specifier记录路径。

失败测试：`test_friend_type_forward_occurrence_is_not_silently_lost`。应保留两个occurrence，并按同一namespace类型身份连接，同时记录friend的词法所属class。不能把已通过的friend函数注入测试推广为全部friend声明覆盖。

### SI06：双层投影的中间parser文本仍出现在expanded_raw

```cpp
template<class T, __CUTE_REQUIRES(sizeof(T)>1)>
void f(T value={});
```

当前`parameter.raw`、`parameter.default`、类型及entity身份正确；但`parameter.expanded_raw`包含 `T value=__codex_parser_braced_default_63`。这是第一层投影的中间parser文本，不是实际宏展开后的API参数拼写。

失败测试：`test_two_projection_expanded_parameter_text_has_no_parser_identifier`。可以恢复真实`expanded_raw`，也可以将中间文本移到明确命名为`parser_projection_raw`的审计字段。测试不要求删除这段溯源信息；要求避免把临时parser标识符作为未加区分的展开后API文本交付。

SI05与SI06仍开放，阶段1不因此通过。

## 追加定点关闭：原SI05/SI06反例通过

根代理修复后，在核心`8d5eccceaf43d53b0ebdfa4f79870bdb05d25356f18922c0486876f1fba037f7`、投影`8458a5efe4adb286ff3e36d73a53a1d5b11fca40fca2bb068ec393de75fd45ae`上，原`test_semantic_integration_review.py`的6项独立用例重新运行，**6/6通过**。

普通namespace friend类型前置声明现在有独立出现位置，并与对应定义连接；双投影的`expanded_raw`不再被中间parser文本覆盖。因此原SI05、SI06在其原缩减反例内关闭。新加入的friend查找边界另记于`phase-1-friend-lookup.md`，其中的开放问题不因这里关闭窄反例而消失。
