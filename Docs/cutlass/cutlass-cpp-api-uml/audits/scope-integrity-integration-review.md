# Scope-integrity接入：独立反例与定点复审

## 结论与版本

本轮先发现两个实际接入漏洞，Root修复后用原反例重新检查，另核对P90默认/禁用syntax、合法条件分支、多层SourceProjection、类内宏和条件alias虚拟解析。**以下列明的接入反例在最终检查点关闭；不代表全局declarations已重生成、全部scope已可证明或所有C++语法已覆盖。**

最终独立探针开始/结束源码SHA一致：

| 文件 | SHA256 |
| --- | --- |
| `extract_declarations.py` | `54f1c36919ced4a4b437c78fc7fdf645dfe2f675eecd0f025c07f8f9bac68539` |
| `declaration_scope_integrity.py` | `3212d524ab351bd3a46cf37a70f61c12784fe64ab74b3edb223eabff17c719d1` |
| `reconcile_candidates.py` | `efe0b591ad10632ce2744cc9c4b0f99455536d396e7bed5070653afbe3f6ae2e` |
| `declaration_projection.py` | `daff166263c0e89f8549f9ee6099cd2a0e23cd6252854b13061a2752d7fb9ee6` |

初次两个失败探针结束时记录的extractor SHA为`e285e72309bd5f9ce7ec18c3581593ea810c001f9d4210770f43f189f7e2f862`，scope checker与reconciler指纹同表。初次运行时源码仍在制作，因此本稿将其作为发现检查点，不冒称它是已发布冻结版。未修改core、snapshot或全局数据；本文是本审查唯一新增文件。

固定源为提交`8f50b052e1099fb982392a622caab69b97b63128`。下文P90均指snapshot的`include/cutlass/pipeline/sm90_pipeline.hpp`。反例只在内存中构造，不写回源文件。

## 1. P90真实损坏/默认修复对照

禁用`project_syntax`、`project_constraints`和`project_bitfields`，复现原始匿名指针默认形参造成的错误恢复：解析器将cutlass namespace范围取为48–735，而独立literal closure为1388。最终接入报告：23个可识别scope，22个bounds verified、1个blocking mismatch；原有6个parse_error和9个missing_syntax仍保留。

`PipelineAsync::producer_tail`及`pipeline_init_wait`的猜测拼写此时仍可能缺cutlass，但occurrence不再被当作已核清global API：`parse_status=scope_closure_mismatch`、`scope_review_required=true`，Reconciler拒绝理由包含`occurrence_not_cleanly_parsed`与`enclosing_scope_requires_source_review`。这是隔离错误身份，不是根据括号擅自修复owner。

默认开启现有语法修复时，P90小样为508 occurrences/0诊断，24个可识别scope全部verified；默认namespace下相应限定名恢复为`cutlass::PipelineAsync::...`及`cutlass::pipeline_init_wait`。给Reconciler正常snapshot source_reader后，`cutlass::pipeline_init_wait`的invalid_occurrence结果为空；未提供外部source_reader时的namespace来源待核不被误报为此接入导致的scope失败。

## 2. I01：宏虚拟解析丢失外层scope不可信标记

### 初次反例

在P90前增加：

```cpp
#define AUDIT_DECL() void audit_generated();
```

再在最后的` } // end namespace cutlass`之前放`AUDIT_DECL()`，其余源保持原文，并禁用上述syntax修复。外层错误闭合被正确判为blocking，普通后续声明也已标记；**宏生成的audit_generated却是global拼写、parse_status=parsed、scope_review_required=false**。原因是虚拟macro extractor只检查自己干净的声明片段，没有继承调用点的scope不可信上下文。

### 修复复审：关闭

新Context.scope_reviews随调用点进入macro虚拟解析；宏结果继承`scope_closure_mismatch`和scope_review_required，不能再作为干净global声明。macro integration对相关结果使用`expanded_scope_review_pending`，不因片段自身可解析而清除外层疑点。展开记录还保留scope_integrity证据及virtual/invocation来源。

额外加入一个**真正位于namespace外**的`void audit_generated();`作身份冲突反例：新版本中真正global声明保持parsed且Reconciler通过；不可信的宏展开结果被拒绝，两者entity_id不同。作为反向控制，完全干净的global宏调用与同签名前置声明仍归并为同一entity_id。因此隔离只作用于不可信owner，没有禁止原本可证明的正常归并。

## 3. I02：uncertainty只覆盖parsed_body而漏掉潜在尾部

### 初次反例

在P90前定义两个互相平衡的scope-changing宏：

```cpp
#define AUDIT_OPEN namespace audit_inner {
#define AUDIT_CLOSE }
```

在`namespace cutlass {`后依次调用AUDIT_OPEN、AUDIT_CLOSE，再禁用syntax修复。两宏真实展开的净作用域作用平衡；独立checker见未展开scope宏，正确降为uncertainty而非proven mismatch。但初接入只把不可信区域延伸到解析器提前闭合的739行，没有覆盖finding给出的潜在尾部。后面的`PipelineAsync::producer_tail`与`pipeline_init_wait`仍以parsed、无review标记的猜测global身份入账。

### 修复复审：关闭

新region覆盖finding的潜在尾部，而不止parsed_body。两个后续方法现在为`parse_status=parsed`且`scope_review_required=true`，Reconciler拒绝理由为`enclosing_scope_requires_source_review`。**没有将uncertainty升级成scope_closure_mismatch，也没有产生“源码已经证明非法”的scope诊断。** 原本存在的parser/macro未解析诊断仍单独保留，不能反过来用它们冒充scope checker的证明。

## 4. 合法#if分支与“不确定不是源码非法”

独立构造三组短源码，分别以Clang在A=0/A=1编译，六次均成功：

| 场景 | Scope-integrity接入结果 |
| --- | --- |
| namespace内#if分别声明完整struct S/T，之后公共位置void f() | 三个scope bounds verified，0 mismatch/0 uncertainty；n::S、n::T、n::f无需scope复核，Reconciler通过 |
| namespace在#if/else各有一个物理右括号 | 不同合法closing候选，0 blocking mismatch、有uncertainty；相关声明要求scope review |
| #if A打开inner namespace，另一个#if A关闭它 | 未求解两个条件的关联，因此0 blocking mismatch、有uncertainty；不把保守条件过包当source-invalid |

后两组可能仍出现Tree-sitter自己的missing/parse诊断；这是解析器不能完整表达该条件结构的事实，不是Clang判源码无效。已检查相关干净方法保持parsed并另带scope_review_required，而不是被错误标为proven scope mismatch。

## 5. SourceProjection与虚拟macro/alias层

### 两层投影及UTF-8前缀

使用含中文注释前缀的：

```cpp
namespace n {
  template<class T, __CUTE_REQUIRES(sizeof(T)>1)>
  void f(T* = nullptr, int x = {});
}
```

同一源会经历匿名函数参数/braced-default语法投影和constraint展开投影。结果为0诊断、scope verified，记录的coordinate_space为mapped_source_bytes，source_sha256严格匹配原始UTF-8 bytes，并保留两层projection_input_hashes。scope opening/expected closing回切原始字节分别为`{`、`}`；f的匿名参数name=null/default=nullptr和x的`{}`默认值保留。没有把投影后字节拿来定位原始源。

### 类内macro片段与外层uncertainty传播

使用`#define DECL(T) void g(T* = nullptr, int x = {}) {}`和`namespace n {struct S { DECL(int) }; }`：原始source scope无误，宏内使用class-fragment parser壳并再次执行参数/default投影。生成方法为`n::S::g`，0诊断，无错误scope-review标记，没有把parser壳变成有效owner。

再将同一struct放入前述关联#if的inner namespace结构：生成g保留outer scope_review_required，仍只是uncertainty，不变成proven mismatch。宏g的physical signature范围准确回切为`DECL(int)`（而非虚拟函数体偏移），scope refs保留父source路径；虚拟片段自身干净不能洗掉父作用域疑点。

### 条件alias在回映后的上下文中仍继承guard

另在关联#if结构中加入`template<class T> using Alias =`与#if B的T/T*两个RHS，前面保留UTF-8注释。两个alias variant均继承scope_review_required，parse_status保持parsed；integration状态均为expanded_scope_review_pending。原始signature范围为同一物理声明，scope报告的source_sha256匹配原始bytes、coordinate_space为mapped_source_bytes。此项检查覆盖了restore_context之后继续集成虚拟声明的路径，不只是首次AST walk中的普通声明。

同类P90长度变化投影中，也检查了不可信region的原始范围及原source hash：guard end没有停留在投影后更长的坐标，projection旧范围另存证据。没有发现本轮修复后的坐标层错位反例。

## 6. 最终验证与边界

本轮先执行上面的独立实际P90改写及短源码反例，再运行既有测试：`test_scope_integrity.py`的16项与`test_scope_integrity_integration.py`的7项，共23项通过。没有以测试数量代替上述逐项证据。

| 发现/检查项 | 最终状态 |
| --- | --- |
| I01 宏展开丢父scope taint | 关闭；继承review，拒绝清洁global身份 |
| I02 uncertainty漏掉潜在尾部 | 关闭；扩大review范围但不变source-invalid |
| 猜测global与真正global误归并 | 关闭；不可信身份隔离，干净控制仍正常归并 |
| P90默认修后误阻断 | 未复现；508/0、24个scope verified |
| 合法#if不确定性误判blocking mismatch | 未复现；Clang正向控制与scope输出分别核对 |
| 多层投影/宏/alias回映错层 | 列明样本未复现；原始字节、hash、父review传播吻合 |

checker仍只检查parser已识别的namespace/class/struct/union body；未识别scope、未知外部宏及未完成声明继续由已有独立候选/宏义务处理。它不修owner、不选择预处理配置、不将uncertainty当源码错误；预算耗尽也只能保留复核义务。此次通过是该接入与列明反例的定点检查，不授权清空全库scope pending、不替代全量重生成对账，也不涉及HTML或浏览器验收。
