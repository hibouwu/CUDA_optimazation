# CuTe alignment：独立stage接入与三图静态复审

本轮只审查`stage_cute_alignment.py`、正式`module.json`、相关源码/数据及root指定的三张SVG。没有编辑stage、manifest、draft或canonical ledger。测试中运行真实enrich/stage逻辑，但以只读内存records替换ledger迭代器，并捕获全部dump调用；没有真正执行stage写入。图形检查使用rsvg-convert白底PNG与view_image原尺寸，不使用浏览器或HTTP。

## 1. v1 stage的5个真实放行反例

被审初版stage SHA为`6262f84f0d8712adcfdcf2750a884b4977380c0a15adec7e230e98e4de5ab45f`。baseline通过，错误类型qualified owner以及整项alignment_specifiers删除能被拒绝；因此不是所有门禁失效。但下列canonical内存损坏仍被接受，并会在捕获的输出中关闭adapter/identity issue：

| 反例 | 初版结果 | 原因 |
| --- | --- | --- |
| 删除一个canonical alignment的CUDA conditional_expansions分支 | ACCEPTED | 只比对attribute span与alignment_expression，未对账两个真实定义分支 |
| 将attribute.owner_entity_id / owner_occurrence_id改错 | ACCEPTED | 顶层owner查找成功后不核attribute内部回引用 |
| 将is_byte_aligned的canonical attributes及qualifiers清空 | ACCEPTED | raw signature仍匹配，但结构化声明属性丢失未被比对 |
| 将namespace记录qualified_name改为wrong::cute | ACCEPTED | namespace节点只有source_selectors复数形式；通用enrich对裸cute名字没有同等qualified-owner检查，最终full_name为wrong::cute |
| 删除array_aligned.hpp的namespace occurrence，重复alignment.hpp的namespace occurrence以保持总16 | ACCEPTED | len==16和occurrence集合subset不能证明16个独立物理位置；输出两个namespace位置都来自alignment.hpp |

最后一例尤其说明：两文件的namespace可以共享一个entity，但必须保留两个不同的source occurrence；重复一个不能补足另一个。16这个正确分母本身不能替代逐selector多重集对账。

## 2. 新stage的定点关闭证据

Root新增`verify_canonical_contract`后，审查版本SHA为`a21a3efdfe6e20f56908b23aeeb2cb45184e52e4e6bded2c0c3b4e7af946ef0b`。我直接读了实现，并运行新的独立测试，而不是将root的修复总结当成结果。

新增测试文件为`tests/test_alignment_stage_review.py`。它使用固定alignment.hpp、array_aligned.hpp以及两个小型依赖header的内存canonical records，经真实enrich_nodes与stage.main走完整门禁，所有写入始终被捕获。14项测试全部通过：

- 正控：保留16个声明、零manual覆盖、namespace两个不同文件位置和同一个cute实体。
- 原5个放行反例全部被拒绝。
- 错误类型owner、整项alignment缺失仍被拒绝。
- namespace同名但不同entity ID、alignment单位改成bits、expression span漂移、宏定义body篡改、条件directive位置漂移、重复同一个宏分支也被拒绝。

新实现不只核总数：它比对16个精确selector的Counter及唯一occurrence ID，检查namespace合并身份，确认主模板和9个特化的10个独立身份；声明参数/默认值/属性/qualifiers与源草稿比对；alignment还核owner回引用、语法owner范围、单位、spelling、表达式范围、两条件分支及其定义正文、源hash和条件位置。因此上述5项初版缺口在这个审查版本中关闭。

复跑：

```bash
.venv/bin/python -m unittest discover -s tests -p test_alignment_stage_review.py -v
```

这组测试只验证接入门禁对列明损坏的拒绝能力，不代替整个Extractor或全库语义审查；它不读取/重写多GB全局账本，也没有进行GPU运行。

## 3. manifest与分母判断

所审`data/modules/cute_alignment/module.json` SHA为`3bb025bd8f9b5eb9522b95d9f1c17b3ef3df236a49b9e953a24106828141108f`。它明确是`full_file_work_package_with_explicit_dependency_perimeter`，`full_file_scope`只有alignment.hpp与array_aligned.hpp，source_owner_module为cute_core，未把依赖的math.hpp/array.hpp加入完整文件分母。

物理声明16的组成正确：namespace出现2、is_byte_aligned函数模板1、aligned_struct主模板加9个偏特化共10、array_aligned主模板1、CUTE_ALIGNAS条件定义2。共享namespace实体不减少两个source occurrence；macro的两个定义不是两个不同全局宏名字；依赖API节点不追加到16。

manifest没有宣称全库或runtime完成，offline_use仍为generation_pending、独立会议使用未执行。该边界与本轮实际工作一致；stage source/identity门禁通过不自动等于浏览器可用性、完整模板实例或硬件语义验收通过。

## 4. 三张指定SVG的静态目视

检查了以下冻结文件；原尺寸文字可辨，无可见裁切、遮挡或签名被截断。XML中的两个API链接端点也分别核查，不通过浏览器点击验证。

| SVG | 尺寸 | SHA-256 |
| --- | --- | --- |
| `edge-alignment.edge.specializes_128_1a6f66fc.svg` | 1062×538 | `9a8c671cf16a9af95bc2ff801d33bfd8dd9a94bfa42e078ed933d468e30bbeae` |
| `edge-alignment.edge.expand_attribute_array_40_cpp_cbe61f7d.svg` | 946×536 | `16f3b5452fcb7abe2e6e2496e383ef67869e7b65292ab923a24651a8124fba1e` |
| `edge-alignment.edge.has_single_bit_f11bd398.svg` | 907×690 | `315e3a2cf37cb00704cdc3a0d7fa1c2c62ae5472533f3b53aa0b308e52665b8b` |

- **128偏特化图：** 67行的偏特化完整签名保留CUTE_ALIGNAS(128)、固定128和仍未绑定的Child；58行主模板保留Alignment以及Child=void。边为独立特化关系，没有把Child画成基类。链接分别指向alignment.type.aligned_128和alignment.type.aligned_primary。
- **CPP属性展开图：** array_aligned.hpp:40的调用端点与alignment.hpp:54的宏定义端点不同，`!defined(__CUDACC__)`和非运行call的求值标签可见，宏定义原文完整。链接分别指向alignment.attribute.array_40与alignment.macro.cpp。
- **has_single_bit图：** 源函数的模板参数、constexpr、CUTE_HOST_DEVICE以及`void const* const ptr`完整；目标has_single_bit签名没有吞入`{`；47行精确callsite可见，图中文字明确static_assert/T=int/不是运行期调用。两个函数链接正确。

## 5. 静态呈现尚待复验的两项

第一，旧图对macro和namespace级类型/函数显示`access: public`。这不是它们具有C++类成员public权限的证据，而是原ledger Context默认值被通用图注显示。Root已提出并实现区分access_scope，不更改全局声明语义。**本次目视的上述SHA仍是显示旧public的版本；新渲染版本需要另行核查，不能把代码改动当作图已通过。**

第二，CPP宏图虽然有调用实参Alignment与定义形式参数n，却没有显示数据中已有的`expanded_spelling = alignas(Alignment)`；读者要自行做n→Alignment替换。建议显式展示这项绑定或真实展开结果。数据未丢失该字段，但这张“逐边细视图”还没有把绑定输出直接呈现出来。此项是图的表达改进，不是源码API错误。

预览metadata保留过旧validate_evidence对行尾newline产生的12个假quote mismatch。Root已单独修复保留原文并检查byte/line一致，本报告不将那些已解释的旧校验消息重新判为draft缺陷。

本轮结论限于：新stage对列明反例通过定点拒绝；manifest分母与边界准确；三张指定旧SVG在静态原尺寸下无可见裁切且源码/API端点正确。不能由此宣称117张alignment图、实际浏览器交互或整个模块已经完成使用验收。

## 追加：显式macro绑定的stage版本

Root随后在stage122–124为alignment的expands_to边加入`macro_bindings={'n': alignment_expression}`。我已读该增量，并对stage SHA `abfc31d7ce72259355f052882c081e46f0b511c1308b1fa5f336b1f0ea6738f1` 重新执行本报告14项测试，全部通过；没有扩展新的语法反例，也没有修改stage。新access_scope与展开结果显示仍需以实际重新生成的SVG SHA进行目视复验，不能仅凭新字段存在关闭显示项。

## 正式重生成后的四图复验：显示项关闭

本节读取正式stage/build之后的新版SVG。先核SHA，再使用rsvg-convert白底PNG和view_image原尺寸逐张查看，并解析SVG XML中的可见text与API链接。检查前后四个SHA一致；没有使用浏览器、HTTP或修改代码。

| SVG | 本版尺寸 | 本版SHA-256 |
| --- | --- | --- |
| `edge-alignment.edge.specializes_128_1a6f66fc.svg` | 1062×518 | `4dac62245d6b12dcb66e4dceaace4c8dc2d554aefd01a5f6eaa74e724c4f5fff` |
| `edge-alignment.edge.expand_attribute_array_40_cpp_cbe61f7d.svg` | 946×551 | `1fac12e63b05fb5e874305a662a1e218df0435a2136c3318f710d458a7d334d5` |
| `edge-alignment.edge.has_single_bit_f11bd398.svg` | 914×649 | `3d245f0620defc5a5174e5bff5e3339f6d474b17a457f153b09de3dfd95f35b4` |
| `edge-alignment.edge.evaluate_assert_0dd18ea2.svg` | 825×590 | `70b575df7935d7e54a799e3046902f473c667c53e06174cc9a62ad86713fad14` |

**access_scope显示项关闭。** 四图的namespace级类型、函数和macro均不再出现`access: public`。同时完整原始声明没有因此被删减：偏特化保留CUTE_ALIGNAS(128)与Child，主模板保留Alignment和Child=void；两个函数仍保留constexpr、CUTE_HOST_DEVICE、参数名及指针cv限定。原尺寸下未发现可见裁切或遮挡。

**macro_bindings显示项关闭。** CPP展开图现在明确显示`宏参数：n := Alignment`及`展开结果：alignas(Alignment)`，并同时保留`!defined(__CUDACC__)`、40行调用来源和54行原宏定义。两端链接仍分别是alignment.attribute.array_40和alignment.macro.cpp；不是通过改写宏定义中的n来伪造实际源文件。

**C++双引号保留检查通过。** 新evaluate_assert图以表达式求值关系指向binding节点alignment.expr.assert，不是伪造static_assert函数调用。边标签和目标文本均显示原双引号字符串`"N must be a power of 2 in alignment check"`；没有换成单引号，也没有把`<U+0022>`渲染成可见原样标记。文本换行只用于排版；SVG解码text合并后的字符串与原C++字面量吻合。源47行和constant_evaluation_static_assert标签仍可见。

**端点与调用位置复验通过。** 偏特化图链接仍为aligned_128→aligned_primary，函数调用图仍为is_byte_aligned→has_single_bit，明确47行callsite和T=int的static_assert常量求值；函数签名没有吞入body开括号。新增表达式图链接是is_byte_aligned→expr.assert。这里核对的是SVG静态链接值，不声称已在浏览器点击并验证交互。

结论：上文两个待复验显示项在这四个固定SHA上关闭，附加双引号显示检查通过。本次未发现新的可见裁切、错误源码/API端点或声明文本损失。范围仍仅是这四张静态图，不代表117张图全部目视通过，也不替代实际浏览器或会议使用验收。
