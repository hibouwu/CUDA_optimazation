# 阶段1：位域核心接入与class fragment独立审查

本审查没有修改生成器，仅新增`tests/test_bitfield_review.py`与本记录。核对对象包括真实descriptor字段、宏定义模式证据、宏调用实例、anonymous位域身份、C++20初始化值、访问控制、SourceProjection多层来源映射，以及条件化函数头共享body与位域投影的组合。

## 版本边界

最终复现版本：

- 核心：`1b6ad2b275d345aaef242ee7e814af4d75038ff2615f7269fced3fe7f33a992c`
- 投影：`daff166263c0e89f8549f9ee6099cd2a0e23cd6252854b13061a2752d7fb9ee6`
- 独立位域模块：`eb5375e0d96193c51188fd3da2ba320b35eec8ecc9ffeeff0be53bba8822b634`
- 条件头模块：`3a7c16588c7caa48f9599b800652cf7d714071f534fb17a6df6c9c22c0ea5081`
- 独立测试：`da12a06a981478dc06263d040c316ed61dbfbbf7b8d696d6efa334f53d7bc973`

审查开始时核心/投影为`ff18963a…`/`989ce1f3…`。期间根代理接入headers，加入外部条件的`directive_path`防错映射，并将宏模式证据切换为独立位域模块的`analyze_macro_bitfield_patterns`。最终测试依据上述新版本，不把早期结果冒充最新版本结论。

运行入口：

```bash
.venv/bin/python -m unittest discover -s tests -p test_bitfield_review.py -v
```

当前 **13个测试函数中12个通过、1个失败**；失败函数分三项契约检查，因此unittest显示3条失败。原`test_bitfield_integration.py`的6项测试也已实际通过，但它们没有覆盖下文BF01的组合。

## 真实descriptor字段逐项核对

本轮不是只比较总数。对每条输出检查物理`name_range`、`colon_range`、`bit_width_range`、`declarator_range`、`initializer_range`与`signature_range`能否切回相应源码。

- `include/cute/arch/mma_sm100_desc.hpp`：51个位域，其中12个匿名；覆盖SmemDescriptor、MaskAndShiftB、InstrDescriptor、InstrDescriptorBlockScaled的字段。
- `include/cute/arch/mma_sm90_desc.hpp`：11个位域，其中6个匿名。
- 总计62个位域、18个匿名。18个匿名entity身份互异；它们没有获得伪造C++名字。62项的initializer均为None，对应实际源文；访问均为public。

这些检查证明当前固定源的字段契约与位置表达，不证明GPU实际布局、端序、编译器ABI或运行时数值行为。

## 宏模式证据与调用实例的区分

采用含注释伪colon和续行的宏定义，逐项检查：

- `classified_colons`确实指向原宏定义中的一个`:`字节，没有误指注释中的colon；
- 对应模式ID存在于独立`definition_shape_proof.bitfield_patterns`中；
- 用`definition_shape_proof.source_segments`反向重建该位置，虚拟模式源码中仍然是同一colon；
- proof保持`is_concrete_api_instance=false`，不能拿模式字段数冒充调用产生的API数量；
- 宏调用的具体字段另有`macro_origin`，其虚拟width/colon范围能切回对应`virtual_source`；物理顶层范围指向调用点，不假装调用点本来就写了colon。

根代理将proof结构从早期重复probe切换为独立模块后，测试随结构调整到`definition_shape_proof`，但保留并加强了原colon、注释排除、逐片映射和非实例断言，没有删除对应检查。

当前带块注释和续行的缩减宏仍可能留下一个明确的原始预处理语法诊断；模式证明与具体字段出现不用于隐去这条诊断。

## 已通过的删除、重名、作用域与多层映射反例

- 在三个相继的投影层中插入长类型名、anonymous parser名与class语法外壳，`origin_span`仍把两个colon精确映射到原始单字节位置，语义读取器还原原文。
- `A::value`与`B::value`不会按短名合并；各自匿名邻居也有独立身份。
- 从宏源中删除命名位域后，输出相应减少，匿名邻居仍保留正确width；模式proof的真实colon数量也同步减少。
- 人为删除一个已生成匿名位域occurrence后，保留独立physical模型的检查能拒绝损坏输出。该破坏测试只说明检查发现此类删除，不代替全库候选对账。
- C++20的width与initializer分开：`a:3=1`为width 3、initializer 1；`b:4{2}`为width 4、initializer `{2}`。零宽匿名位域仍是匿名字段，不把parser临时名字当作合法C++名字。
- `FIELDS(N)`在实际class/struct上下文中保留private/protected权限，不生成语法wrapper实体；模型owner明确标记为grammar context，并连接实际scope entity。
- 同一命名字段的两个条件width保持同entity、独立variant和各自条件。
- 宏fragment中的`LOCAL`条件仍指调用文件第2行；`CUTLASS_NAMESPACE`条件仍指固定helper定义的第41行，没有被外壳的行号映射污染。
- 会提前闭合外壳并打开另一个class的合法宏扩展保留`macro_member_context_escape_pending`，不能标成`expanded_and_parsed`或用片段中前半段字段制造完成假象。

## BF01：条件函数头共享body再次解析已插入的anonymous名字，导致零诊断假成功

合法缩减源码：

```cpp
static int
#if A
f(void const* x)
#else
f(void const*)
#endif
{
  struct Local { unsigned value:1, :2; };
  return 0;
}
```

Clang strict C++20在A=0、A=1均通过。当前核心的两个header变体已经把Local类型连接到同一entity，但位域组合仍出错，且diagnostics为空。

### 错误1：匿名字段获得parser名字并污染entity

应有字段名字序列`value, None, value, None`，实际匿名字段变为：

```text
__codex_parser_anon_bitfield_170459d3e8cbbc5452b4e296
```

该字符串也进入entity。原本只有语法用途的临时名已经被当成真实C++成员名。

### 错误2：四个virtual source ID没有对应两个physical位域模型

文件的物理分析仍保存两个正确源模型：

```text
bitfield-source:999f64b67e46b203c66250c8
bitfield-source:170459d3e8cbbc5452b4e296
```

四个生成occurrence却使用另四个virtual `bitfield_source_id`，没有对应回这两个physical模型；与删除输出相同的源模型覆盖检查因此失败。字段colon范围能回到原位置，不等于源身份已经正确连接。

### 错误3：raw_signature/declarator仍含parser插名

原始物理签名是：

```cpp
unsigned value:1, :2;
```

字段的`raw_signature`却包含完整临时标识符。按`signature_range`切回真实源文不相等；匿名字段的declarator也包含该临时名字。

行为与以下机制一致：外层位域投影已给共享body插入anonymous parser名，随后条件头构造把这份parser body复制为虚拟source，再次位域分析将插入名认作真实名字。条件头的物理identity mapper修好了Local类型身份，但没有同时恢复位域的原始model、anonymous属性和源签名。

对应失败测试为`test_shared_conditional_header_body_preserves_anonymous_physical_model`，三个子检查分别拒绝匿名身份污染、physical模型未覆盖以及raw字段污染。没有删除这些失败断言或将其改成expectedFailure。

## 结论

固定62/18位域、独立class fragment以及指定宏模式证据通过本轮正向核查，但BF01说明“单项投影均通过”不能替代组合验证。当前不能宣称位域与条件头组合已完成，更不能据此让阶段1通过。

## 追加冻结版复审：BF01原例与前置/嵌套偏移反例通过

系统中断后，本审查分支没有仍在等待的工具句柄；未重启、终止或操作根代理继续运行的全量生成进程。只修改独立测试和本追加记录。

本次验证版本在测试前后保持一致：

- 核心：`2ad68f3b0c11fafb9464ee61e6c2b208a2501dd77fd40617d290581fcd585e8d`
- 投影：`daff166263c0e89f8549f9ee6099cd2a0e23cd6252854b13061a2752d7fb9ee6`
- 条件头：`3a7c16588c7caa48f9599b800652cf7d714071f534fb17a6df6c9c22c0ea5081`
- 位域模块：`eb5375e0d96193c51188fd3da2ba320b35eec8ecc9ffeeff0be53bba8822b634`
- 独立测试：`7186449f1212ef5163b9a3805829be6953b4adbd1b18ce90bc2e13766b4732da`

`test_bitfield_review.py`现在共15项测试，**15/15通过**；原BF01的三个失败子契约均未删除或改弱。

### BF01原例的实际修复

两个条件头变体下，4个field occurrence正确连接到原始2个physical bitfield model ID；匿名字段恢复`name=None`，不再把parser插名写入entity。两条路径的Local类型及其相同字段身份一致，raw_signature、declarator与全部物理token范围均能切回原文，diagnostics为空。因此BF01原缩减反例在本版本内关闭，不是仅改成pending。

### 前方Prefix匿名位域的偏移检查

在同一源前方增加 `struct Prefix {unsigned :3;};`，强制外层先产生一个anonymous-name投影。结果为5个field occurrence，对应3个physical model ID；Local在两个header变体中仍是同一entity，没有被前置插入导致的虚拟字节位移拆开。

每个字段的colon、width、declarator、name/anonymous状态、initializer和原始签名仍按最终物理字节核对；没有parser名进入语义身份。A=0与A=1的完整缩减源均通过Clang strict C++20。

### 中文前缀、零宽位域与嵌套条件body的组合检查

另一个缩减源同时包含：中文前缀注释；Prefix中的零宽匿名位域及普通命名位域；A控制的两个函数头；共享body中B/!B以及!B下C/!C的三个Local定义分支。

核对结果：

- 14个field occurrence恰好对应原始8个physical model ID，没有遗漏或额外virtual source ID；
- 两个Prefix模型各使用一次，六个body字段模型各在两个header变体中使用一次；
- 六个Local出现位置连接到同一个类型entity；
- 匿名名字保持None，字段物理token及raw_signature全部正确；
- body字段的条件包含对应A、B及必要的C条件，没有丢掉嵌套分支；
- A/B/C的八个配置均通过Clang strict C++20；提取器diagnostics为空。

本追加记录只关闭已执行的BF01及这两类位移/条件组合反例。宏wrapper逃逸仍应保留明确pending，其他宏展开、条件头语法和全库候选对账尚不能由这15项测试代替；阶段1没有因此通过。
