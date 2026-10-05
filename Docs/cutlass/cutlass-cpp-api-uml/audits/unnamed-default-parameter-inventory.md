# 匿名指针/引用默认形参：固定824文件独立清单

## 结论与检查范围

固定snapshot原始文本中，确认的**直接匿名指针函数形参带默认值只有1处**：`include/cutlass/pipeline/sm90_pipeline.hpp:732`。本次没有找到直接匿名引用默认形参，也没有找到抽象函数指针/数组引用形参后接默认值的实例。另有2处同形文本是CuTe约束宏的替换列表，不是函数声明；71个实际`__CUTE_REQUIRES`调用均落在独立词法平衡的`template<...>`参数范围，属于NTTP生成模式，不能混入函数形参投影。

这是对固定源中相应声明写法及这两个约束宏模式的清单，不是任意宏展开或typedef背后全部语义指针类型的通用C++推断。824文件全部进入扫描，没有从Tree-sitter已提取成功的声明反向定义分母。未改snapshot、core、候选/声明全局数据；只新增本文。

- 固定提交：`8f50b052e1099fb982392a622caab69b97b63128`。
- `data/scope.json` SHA256：`2b8a0094ff40dcd0a47c4da64e309fda8af98dc1ecdeb9b28a6e8a8692c2970f`。
- 扫描前逐一核对824个snapshot文件的SHA，全部匹配scope记录。
- 使用独立候选扫描器的注释/字面量感知lex，再按物理字节区别复合运算符；另用raw Tree-sitter AST和窄抽象声明模式作反向漏项检查，但不将AST命中数当作分母。

## 真正的函数形参

```cpp
template<class ThisTemplateParameterExistsOnlyForDependentFalse = int>
CUTLASS_DEVICE
  void producer_acquire(PipelineState /* state */,
    ThisTemplateParameterExistsOnlyForDependentFalse* /* unused */ = nullptr) {
  static_assert(cutlass::detail::dependent_false<ThisTemplateParameterExistsOnlyForDependentFalse>,
    "It is never valid to call PipelineTmaStore<0>::producer_acquire");
}
```

原文为P90的729–735行，owner是`cutlass::PipelineTmaStore<0,-1>`特化内的public方法模板。静态断言是延迟到函数模板实例化时触发的接口诊断，不是将此接口从账本删除的依据。

| 部分 | 物理范围，半开字节区间 | 应记录的身份 |
| --- | --- | --- |
| 模板参数前缀 | 729；`[25228,25298)` | `class ThisTemplateParameterExistsOnlyForDependentFalse = int`是具名类型模板参数，不是函数指针形参，也不是NTTP |
| 第一个函数形参 | 731；`[25342,25367)`含注释 | `PipelineState`按值、name=null、无默认值；注释`state`不是参数名 |
| 第二个函数形参 | 732；`[25375,25447)` | `ThisTemplateParameterExistsOnlyForDependentFalse*`、name=null、default=`nullptr`；注释`unused`不是参数名 |
| 触发恢复错误的等号 | 732；`[25438,25439)` | 默认实参分隔符；不能作为赋值表达式或operator的一部分 |

独立Clang oracle直接复用了729–735原始函数片段，只添加最小外围类、PipelineState别名和dependent_false定义，不实例化会有意失败的方法。Clang21.1.8退出0；AST为CXXMethodDecl，两个ParmVarDecl都无name，第二个type为模板T指针，默认子节点为CXXNullPtrLiteralExpr。这只证明缩减上下文中的C++声明类别/默认值，不证明该诊断接口可以成功执行。

## 近似文本的排除与区别

独立lex第一轮找到399个“指针/引用符号后遇到等号”的粗候选。396处是连续字节的`*=`、`&=`等复合运算符或其宏操作数；排除后仅剩上面1个真实参数及下面2个宏定义。不能用忽略空白/边界的正则把复合运算符拆成匿名形参。

| 类别 | 实际源例 | 判定与投影要求 |
| --- | --- | --- |
| 约束宏替换列表 | `cute/util/type_traits.hpp:55`：`typename cute::enable_if<(__VA_ARGS__)>::type* = nullptr`；`:56`为decltype版本 | 两处都是宏定义，不是函数形参；分别生成匿名指针NTTP模式 |
| 约束宏实际调用 | `cute/algorithm/gemm.hpp:179–187`，184起的`__CUTE_REQUIRES(...)`在template参数列表内 | 共71个`__CUTE_REQUIRES`调用都在此类词法template范围；`__CUTE_REQUIRES_V`在固定824文件中无实际调用。函数形参投影应拒绝，原NTTP source-mapped展开规则另管 |
| 命名引用默认参数 | `cute/algorithm/axpby.hpp:54`：`PrdTensor const& p = {}` | p是真实名字，不是匿名；不能因reference_declarator没有某个预想字段就误判name=null |
| 命名指针默认参数 | `cutlass/gemm/device/gemm_universal_adapter.h:315`：`void* workspace = nullptr` | workspace是真实名字，应原样保留，不需注入语法临时名 |
| 普通运算符/表达式 | `cute/numeric/math.hpp:276`的`x &= x-1`；`cute/algorithm/functional.hpp:163`的宏操作数`*=` | 不属于parameter-list；不得生成参数或函数声明 |
| 抽象引用cast | `cute/arch/mma_sm80.hpp:368`的`reinterpret_cast<double(&)[2]>(...)`等 | 粗函数指针模式命中，但不是声明参数且后面没有该声明的默认实参等号 |
| 函数指针类型默认模板参数 | `cutlass/gemm/kernel/sm100_tile_scheduler_group.hpp:135`的`typename CallbackBeforeCommit = WorkTileInfo(*)(WorkTileInfo)` | 这里的`(*)`在类型模板参数的默认类型中；真正函数参数callback_before_commit另有名字，不能把类型表达式变成匿名函数形参 |
| 匿名非指针默认函数参数 | 如`gemm_universal_adapter.h:270`的`int /* smem_capacity */ = -1`，P90:1062的`InitBarriers = {}` | 不属于此指针/引用适配规则。括号默认值另有语法问题时应由对应规则处理，不能以本规则批量加名 |

补充筛查了`T&&=value`这种不能简单按最后两字节当`&=`处理的词法边界，以及指针后的属性写法、parenthesized abstract pointer/reference后的函数/数组后缀默认值；固定824文件中未找到额外实例。粗`(*)`/`(&)`扫描产生的33个近似命中经后续参数/default位置检查均不是新目标。

raw Tree-sitter遍历中，没有识别到一个显式abstract pointer/reference declarator的optional函数形参；它连P90真实目标也没有按该结构给出。因此“AST抽象默认参数数为0”不能当作不存在此API的证据。反向检查中常见的其它命中是具名引用参数、匿名scalar/tag参数，以及被错误恢复成optional_parameter的宏条件/比较表达式，均不构成新增匿名指针函数形参。

## 原始损伤与新适配小样必须分开

在同一P90源上禁用syntax/constraint/bitfield投影，以普通annotation mask解析，得到6个parse_error和9个missing_syntax。`namespace cutlass`的AST范围为48–735，而物理括号范围应到1388；731–733的producer_acquire被恢复为ERROR/init_declarator/binary_expression一带，732的等号自身是ERROR。下游实际实体出现错误owner：

| 源声明 | 未适配结果 | 正确默认namespace结果 |
| --- | --- | --- |
| 1109–1110 | `PipelineAsync::producer_acquire` | `cutlass::PipelineAsync::producer_acquire` |
| 1128–1129 | `PipelineAsync::producer_tail` | `cutlass::PipelineAsync::producer_tail` |
| 1184–1185 | `PipelineAsync::producer_acquire` | `cutlass::PipelineAsync::producer_acquire` |
| 1362–1364 | `pipeline_init_wait` | `cutlass::pipeline_init_wait` |

为隔离原因，另直接在内存中的相同annotation-masked文本25438处只插入一个检测用参数名，不做其它语法修复。6个ERROR消失，namespace重新为48–1388；9个其它missing仍保留。这证明该匿名指针默认形参足以触发namespace错误恢复；同时不能把其它braced-default等missing都归罪于这一参数。实验只修改内存探针，不写源文件，也不将该检测名作为API。

任务期间Root已接入新的source-mapped适配。下列稳定检查点（探针前后源码哈希一致）的默认Extractor对P90返回0诊断，第二个形参name=null/default=nullptr保留，上表四处默认namespace owner恢复，定制CUTLASS_NAMESPACE variant仍独立存在：

- extract_declarations.py：`2ad68f3b0c11fafb9464ee61e6c2b208a2501dd77fd40617d290581fcd585e8d`。
- declaration_syntax.py：`bbefbaeb974132c3375739bfa3754cc0032c3ff3e78b7dcd80bd120913d8bd89`。
- declaration_projection.py：`daff166263c0e89f8549f9ee6099cd2a0e23cd6252854b13061a2752d7fb9ee6`。
- macro_expansion.py：`e6e201372012c3d28f4bc27e8ef29986c190a604442bca0e428964d593a322d2`。

这是单文件当前实现检查点，不是全局declarations已重生成或旧manual overrides已可删除的证明。

## 该投影应拒绝的反例与保真条件

在上述稳定syntax检查点调用其候选适配函数，独立构造以下短反例，观察结果吻合预期：

- 具名`T* p=nullptr`、具名`T const& p=T{}`：0次投影，不改真实名字。
- `template<class T,T* = nullptr> struct S`：函数参数规则0次投影，不冒充函数形参；NTTP规则另管。
- `MACRO(T* = nullptr)`、`x *= 2; x &= 3;`、注释和字符串中的`void f(int* = nullptr)`：0次投影。
- `void (*fp)(int* = nullptr)`：0次投影，不能把函数指针类型中不合法的默认实参洗成有效函数声明。
- 已可干净解析的`void f(T& = value)`：0次多余投影，Extractor仍保留匿名引用和默认表达式。

正向控制为匿名指针默认形参1次、构造函数匿名指针参数1次、同一函数两个匿名指针参数2次；插入的语义拼写为空，结果不能泄漏parser-only identifier。对不完整`void f(int* = nullptr;`，当前规则可修复局部匿名指针形态，但Extractor仍报告missing_syntax；不能把局部grammar适配等同整段合法，更不能清除剩余诊断。

最终必要保真条件包括：仍输出原始name=null；保留`*`/`&`/`&&`及cv限定、default完整原文和物理字节；注释中的state/unused不变成参数名；函数自身类型模板参数与函数参数分开；所有条件配置仍保留；一处临时名字插入不能让其它namespace/class、后续声明或重载身份发生语义改变。对于这次固定源，独立真实目标仍是P90:732这一处，不能为了扩张规则而把近似文本一概视作同类API。
