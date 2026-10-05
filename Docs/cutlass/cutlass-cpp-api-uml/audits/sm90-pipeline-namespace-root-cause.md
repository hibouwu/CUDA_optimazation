# sm90_pipeline.hpp 外层 cutlass 作用域丢失：只读根因诊断

核查日期：2026-09-09。固定源码为 `include/cutlass/pipeline/sm90_pipeline.hpp`，提交 `8f50b052e1099fb982392a622caab69b97b63128`，SHA-256 `67b1c16b4bb071668082ed5697cd2e21c50a6b37eb509aa187f21a5c3100ec3c`。

本轮没有修改core、projection、snapshot、全局JSON或模块数据，没有重新生成全库。只在内存中对这一文件执行现有Extractor、Tree-sitter对照及Clang缩减样本，唯一新增产物为本报告。下面的探针成功不表示正式修复已接入。

## 1. 根因结论

**首因是732行合法的匿名指针默认参数被tree-sitter-cpp 0.23.4错误解析；影响扩散是提取器信任了已经被恢复过程截短的namespace/class边界。** 不是`namespace_bindings.py`选错cutlass绑定，也不是SourceProjection把后续位置移出了namespace。

原始声明片段为：

```cpp
template <>
class PipelineTmaStore< /* Stages_ = */ 0, /* UnacquiredStages = Stages_ - 1 = */ -1 > {
  // ...
  template<class ThisTemplateParameterExistsOnlyForDependentFalse = int>
  CUTLASS_DEVICE
    void producer_acquire(PipelineState /* state */,
      ThisTemplateParameterExistsOnlyForDependentFalse* /* unused */ = nullptr) {
    static_assert(cutlass::detail::dependent_false<ThisTemplateParameterExistsOnlyForDependentFalse>,
      "It is never valid to call PipelineTmaStore<0>::producer_acquire");
  }
  // ...
};
```

指针参数物理span为 `[25375,25447)`；`= nullptr`的等号在byte25438。该参数没有名字，默认值是nullptr。出错不是负数`-1`或注释：删除特化实参注释、给`-1`加括号均未修复；单独`S<0,-1>`在同版本Tree-sitter可解析。

经当前完整投影管线，Tree-sitter生成的namespace_definition范围是48–735，物理映射为 `[2415,25630)`。735其实是这个producer_acquire的函数body闭括号；真正`namespace cutlass {`在48行、`{`位于2433，真正namespace闭括号在1388行byte47638，完整定义应为 `[2415,47639)`。解析恢复把函数闭括号误当namespace结束，最后1388的真闭括号反而成为根级ERROR。

## 2. 根因如何进入现有提取管线

当前检查的关键代码指纹：

| 文件 | SHA-256 |
| --- | --- |
| `scripts/extract_declarations.py` | `2ad68f3b0c11fafb9464ee61e6c2b208a2501dd77fd40617d290581fcd585e8d` |
| `scripts/declaration_projection.py` | `daff166263c0e89f8549f9ee6099cd2a0e23cd6252854b13061a2752d7fb9ee6` |
| `scripts/declaration_syntax.py` | `24d877b79ec055ea047f82e31484edb9dab0e2273100595bcbca85a9c626361e` |
| `scripts/namespace_bindings.py` | `0cd14ea32380d9faca7accb451208f7c276a7abf0bd4508b6e08474038c21524` |

环境为tree-sitter 0.25.2、tree-sitter-cpp 0.23.4、Clang21.1.8；项目venv执行所有Python探针。

1. `Extractor.extract`先进入bitfield包装，再分析syntax/header，随后解析注解掩码后的字节。该文件无位域适配项，完整路径实际产生9项braced-parameter-default投影，位置为328两处、387、668、865、909、1062、1110、1155；没有732匿名指针默认参数适配项。`declaration_syntax.py:359–427`只扫描`= {`默认值；732为`= nullptr`。
2. `extract_declarations.py:137–145`已有的“给匿名指针临时命名”适配只处理`__CUTE_REQUIRES`展开后的指针NTTP，不覆盖这里的函数参数。这是相近的既有机制，不是已修复本处的证据。
3. `extract_declarations.py:192–231`的函数体预处理修复只对解析器已识别的函数body和其内预处理指令生效。732已被错误读成表达式/初始化声明，且首因不是body中的预处理分支，因此这层不能恢复真正类/namespace边界。
4. `walk`在751行把每个解析节点及当前Context存入context_intervals；761–807只在**解析器给出的**namespace body内压入cutlass作用域；对根节点的后续children则继续使用空Context。它没有独立验证namespace的闭合位置，也没有将“包围作用域遭恢复破坏”的不确定性传到后面的语法正常节点。
5. `record`在`extract_declarations.py:695–725`使用当时scope链创建entity identity，并仅以局部`node.has_error`设置parse_status。后面的PipelineAsync类和pipeline_init_wait函数局部节点无ERROR，因此出现**错误owner但parse_status=parsed**。这不是可以改名成symbolic的模板依赖，而是真实作用域提取错误。
6. `record_parse_errors`仍在1164–1175保留6项blocking诊断，文件并未被宣告整体完成；问题是这些诊断没有阻止受污染范围内产生看似确定的错误身份。`SourceProjection`只按位置映射和恢复语义文本，不能替一个已丢掉父namespace的AST恢复正确owner；因此修改显示名或projection偏移不触及首因。

原始未掩码源码直接Tree-sitter解析有132个ERROR/MISSING，namespace在526行就被恢复截断；经过注解和现有syntax适配后恢复到735行仍错误。不能以原始132对比最终6来认为其余范围的语义身份已经可靠。

## 3. 对照试验与真实影响范围

以下均是单文件内存探针，不修改原文件：

| 输入/开关 | 出现记录 | 诊断 | 外层namespace / pipeline_init_wait结果 |
| --- | ---: | ---: | --- |
| 当前完整Extractor默认路径 | 370 | 6 | namespace48–735；函数为裸`pipeline_init_wait` |
| 禁用syntax投影 | 370 | 15 | 仍48–735；仍丢cutlass |
| 同时禁用syntax与constraint投影 | 370 | 15 | 同上；排除变长投影是首因 |
| 只删除711行特化实参注释 | 370 | 6 | 未改善 |
| 只给711行的-1加括号 | 370 | 6 | 未改善 |
| 仅把732匿名指针参数临时改为有名`unused` | 508 | 0 | namespace48–1388，恢复默认及参数化cutlass两种绑定 |
| 通过现有SourceProjection零宽插入parser-only名字并恢复语义 | 508 | 0 | 同样恢复正确范围；原参数name=null、default=nullptr、原签名保留 |

最后一项不是写入式实现：在内存syntax_analysis.projection_edits追加byte25438的零宽编辑，parse_projection为`__scope_probe_parameter `，semantic_spelling和expanded均为空，parser_only=true，再调用现有`extract_syntax_projection`。结果中raw_signature、expanded_signature、parameters、name、qualified_name均没有临时名字泄漏。既有semantic_reader及parameter方法1020–1029已经能把该插入名字恢复为null；正式修复仍需要可靠检测、测试和常规接入，而不是固定给显示名加cutlass前缀。

作用域错误比三个manual override更广：

- **710–735：** `<0,-1>`特化本身没有形成正确类实体；Stages/UnacquiredStages被记为namespace常量，PipelineState被提升为namespace alias，Params及其成员挂到错误owner，两构造函数变成namespace function。当前14条记录，内存修复后18条。两种cutlass绑定均受到类owner丢失影响。
- **738–756：** 两个producer_commit重载和producer_tail被记为global function，private params_变成global variable。当前4条，修复后8条（默认与参数化cutlass各一份）。
- **765–1388：** PipelineTransactionAsync、两个PipelineDetail namespace片段、PipelineAsync、OrderedSequenceBarrier、pipeline_init_wait以及pipeline_init_arrive_relaxed都丢掉外层cutlass。当前130条记录；修复后260条，因为正确保留默认cutlass与`<cutlass_namespace(CUTLASS_NAMESPACE)>`两个绑定。类别包括class/struct、method/constructor/destructor/operator、alias、member/member_constant、enum/enumerator、namespace和function。
- **48–709：** 本次对照的222条记录数量不变。它们不因后面的失败自动被排除；仍需按各自源范围审查。

这些是本文件在固定版本提取器中的对照记录数，不是全库API总数或“新增138个API”的结论。特别是修复后的条件/参数化namespace会给同一物理源码带来多条出现记录。

当前6项诊断的物理位置为710–733、731–733、732等号、734右括号、757闭括号、1388闭括号，均为函数体外分类。即便后段method自身没有ERROR，也不能依它的parsed标记绕过scope污染。

## 4. Clang合法性缩减对照

以下自包含源通过 `clang++ -std=c++17 -x c++ -fsyntax-only -Xclang -ast-dump=json -`，退出0且无诊断：

```cpp
namespace cutlass {
template<int I,int J> class S;
template<> class S<0,-1> {
public:
  template<class T=int> void producer_acquire(int, T* = nullptr) {}
};
template<int Stages> class PipelineAsync {};
void pipeline_init_wait(int) {}
}
```

Clang将producer_acquire记为`cutlass::S::producer_acquire`方法，第二参数为无name的`T *` ParmVarDecl、默认值CXXNullPtrLiteralExpr；后面的类型与函数仍分别为`cutlass::PipelineAsync`及`cutlass::pipeline_init_wait`。

同一Tree-sitter版本连`void f(int* = nullptr) {}`都会产生ERROR；`template<class T=int> void f(T* = nullptr) {}`甚至被恢复成初始化声明，放进class/namespace后可破坏外围结构。`T const* = nullptr`也失败；给参数加名字则通过。对照中的`T& = *(T*)nullptr`在Tree-sitter与Clang均可解析——这里只验证声明合法性，没有运行或调用这个默认表达式。

因此不能将该处判为CUDA源码非法，也不能因某个普通前端的恢复树把类内容提升成全局接口。Clang只为缩减声明提供独立合法性证据；本轮没有用Clang重新编译整个CUDA头，更没有GPU执行。

## 5. 建议的最小正式修复

**第一层：在已有syntax projection中增加匿名指针默认参数适配。** 对固定源中已经识别、且能通过独立签名/参数探针确认的函数参数区间，在指针declarator与等号之间插入parser-only标识符；原始语义spelling为空，保留pointer类型、cv、默认表达式、注释和物理参数范围，恢复name=null。该文件可从已证实的732模式切入，不需另建通用C++解析器，也不应裸regex改写所有`* =`或把变量/乘法/赋值当成参数。

现有SourceProjection与extract_syntax_projection已在只读探针中证明能承载这种编辑。应复用其位置感知semantic_reader，而不是全局字符串删除；原raw signature保持原始文本，parser_projection_signature可以作为诊断投影记录，但不能进入API参数、类型、默认值或身份。

**第二层：加结构性防误归属门禁。** 已知namespace/class的解析结束位置若与独立确认的物理作用域边界不一致，应把受影响范围记为scope-recovery pending/错误，不允许仅因后续局部节点无ERROR就建立“确定global owner”实体。该门禁是防止下一次恢复污染，不替代第一层真正解析修复。独立括号/作用域检查必须保留预处理条件，不可在一般含条件括号的文件中盲数所有分支后直接补namespace；本文件48到1388的物理边界有直接源码证据。

错误scope参与了entity_id、template identity、internal-linkage判断及namespace绑定，正式修复应通过提取器重新生成正确身份，再让下游引用对账；不能只改qualified_name字段、复制现有manual节点或全局给路径内实体加`cutlass::`。

## 6. 必要回归及复核边界

至少需要：

1. 原始完整sm90_pipeline.hpp经正式管线保留namespace `[2415,47639)`，PipelineTmaStore特化、PipelineAsync和两个init函数处于正确scope；默认及参数化cutlass绑定均保留。后段原本parsed的错误global身份不得继续混入新账本。
2. 原732参数name必须为null、type包含真实指针类型、default为nullptr；raw signature及物理span精确回切，所有语义字段与identity中无临时名字。原711特化的-1、原注释与所有overload保持。
3. 缩减的int*/T*/T const*默认参数、嵌套class/namespace及其后续兄弟声明；用Clang AST核对匿名性质。已命名指针默认参数不得被清空名字；已可解析的引用默认参数不得被误改。
4. 拒绝把函数体中的乘法/赋值、直接初始化或其它非参数`*`片段当成此投影候选；无法确认参数区间时保留具体pending。
5. 与现有9项braced defaults和位置投影组合后仍保持后续偏移/来源；const/private、namespace宏配置、同名重载和局部作用域不得被同一次恢复污染。
6. 另一个未修复的作用域破坏反例应触发结构scope错误，而不是产出可被下游误信为正常global的parsed节点。不能把真实提取错误改名symbolic来满足模块门禁。

已存在的临时反例是 `contracts.json` 的 `contract.api.init_wait`（1364），以及 `tmem_protocol.json` 的三个manual方法：1129 `PipelineAsync::producer_tail`、1110与1185两个`PipelineAsync::producer_acquire`重载。它们证明当前全局身份不能直接信任，却没有修复其它受影响实体。只有正式源映射修复、相应回归和必要的全局重新对账完成后，才适合撤销这些override；本报告未修改它们。

本次只定位一个固定文件的首因及传播范围，未扫描其它824文件中该语法的分布，也未宣布全库声明或关系完整。正式修复可以先按这个已证实工作包推进，不以清除无关模块所有诊断为启动条件。
