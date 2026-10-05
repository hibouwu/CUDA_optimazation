# CUTE_ALIGNAS：字段对齐宏引发的作用域恢复损伤

日期：2026-09-09。固定提交：`8f50b052e1099fb982392a622caab69b97b63128`。本轮只读诊断、词法清点与内存投影试验；没有修改core、snapshot、canonical declarations、Dense数据或模块草稿，没有重新生成全库。唯一新增文件是本报告。

## 1. 首因不是scope-guard，而是尚未展开的对齐属性

目标文件为 `include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp`。216–217原文是：

```cpp
CUTE_ALIGNAS(128) cute::array<ElementBlockScale, cute::cosize_v<SmemLayoutSFA>> smem_SFA;
CUTE_ALIGNAS(128) cute::array<ElementBlockScale, cute::cosize_v<SmemLayoutSFB>> smem_SFB;
```

原始调用span分别为 `[10857,10874)` 与 `[10980,10997)`；两条完整字段声明span分别为 `[10857,10946)` 与 `[10980,11069)`。调用实参均为128，不能根据外层已继承`cute::aligned_struct<128>`就删除字段的alignment合同。

Tree-sitter将这些尚未展开的宏按普通C++声明片段恢复，216行甚至生成名为`cute`的member，smem_SFA与smem_SFB本身丢失。恢复过程中连续吞并外层闭括号，形成三个late_close：

| scope | 真实body闭合 | 旧解析body闭合 | 暴露出的结构损伤 |
| --- | --- | --- | --- |
| TensorStorage，213行开 | 218，end byte11102 | 222，end byte11216 | 吞入本应属于SharedStorage的PipelineStorage和pipeline |
| SharedStorage，212行开 | 222，end byte11216 | 1083，end byte49867 | 吞入Collective后续成员、Arguments、Params和函数 |
| CollectiveMma，92行开 | 1083，end byte49867 | 1087，end byte49970 | 将真实namespace闭括号当作自身闭合 |

当前单文件提取得到105条出现记录，其中104条被scope-guard隔离。9项诊断分别是4 parse_error、1 missing_syntax、3 scope_closure_mismatch，以及879行独立的function_scope_declaration_pending。目标namespace本身落入大ERROR，不能因个别内部节点仍能识别而信任其qualified owner。scope-guard只是揭露并隔离原损伤，不是引入错误的原因。

## 2. 固定宏定义和实际include来源

`CUTE_ALIGNAS`并非在cute/config.hpp定义。完整固定定义位于 **`include/cute/container/alignment.hpp:51–55`**：

```cpp
#if defined(__CUDACC__)
#  define CUTE_ALIGNAS(n) __align__(n)
#else
#  define CUTE_ALIGNAS(n) alignas(n)
#endif
```

两个definition的物理span分别为 `[2263,2301)` 和 `[2308,2344)`，formal parameter都是n。条件是`__CUDACC__`，不是`__CUDA_ARCH__`；不能在全配置提取中只保留Host的alignas拼写或只保留CUDA的__align__拼写。

目标文件有可直接核查的include链：目标45行→`cute/atom/mma_atom.hpp:36`→`cute/tensor_impl.hpp:49`→`cute/container/array_aligned.hpp:33`→`cute/container/alignment.hpp`。因此这不是缺失定义的库外宏。include链只证明此固定来源可达，不是运行调用边；局部override或不同include顺序如以后存在，仍必须检查有效定义，不能按宏名永久硬编码语义。

## 3. 固定824文件中全部实际调用

按scope.json逐文件读取并屏蔽注释/字面量，排除macro definition本身，得到**16个调用、5个文件、每次均一个实参**。这是物理调用清点，不是32个条件变体实体或完整API覆盖计数。

| 文件 | 原调用行 | 实参/位置 |
| --- | --- | --- |
| `cute/container/alignment.hpp` | 60–68，各一处 | 1、2、4、8、16、32、64、128、256；class-key后的9个aligned_struct特化属性 |
| `cute/container/array_aligned.hpp` | 40 | Alignment；class-key后的array_aligned模板属性 |
| `cutlass/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input.hpp` | 271、272 | SmemAlignmentA、SmemAlignmentB；字段前缀 |
| `cutlass/gemm/collective/sm90_mma_tma_gmma_rs_warpspecialized_mixed_input.hpp` | 291、292 | SmemAlignmentA、SmemAlignmentB；字段前缀 |
| 当前FP8 blockwise目标 | 216、217 | 128、128；字段前缀 |

上表共10个类型属性位置、6个字段属性位置。后缀检索没有排除.inl，分母来自固定824 manifest；没有只检索Dense已引用文件。

两个mixed_input文件的SmemAlignmentA/B来自`cutlass::detail::alignment_for_swizzle(SmemLayoutA/B{})`，分别见普通版本277–284、array版本261–265；源码断言要求至少128字节。它们不是一律等于128。array_aligned的Alignment也是模板实参，不能把这16处都实例化成一个常量属性。

另外4个文件存在同源损伤，但不一定破坏scope闭合：

- alignment.hpp：当前14条出现记录、11个parse_error；9个特化被恢复为名叫CUTE_ALIGNAS的struct，而不是正确的aligned_struct特化。
- array_aligned.hpp：3条记录、1个missing_syntax；类被误读成struct CUTE_ALIGNAS。
- 普通mixed_input：264条记录、2个missing_syntax；每个宏被拆成declared_type为CUTE_ALIGNAS的伪member SmemAlignmentA/B，而真正smem_A/B局部节点仍为parsed。
- array mixed_input：318条记录、3个missing_syntax；同样拆出伪member。额外语法诊断没有被归零或掩盖。

这些文件的scope-guard均无mismatch。这说明边界检查不能代替属性、类型名字和声明候选的语义对账；正式修复应覆盖全部16处及两类位置，不能只把目标文件3个late_close清掉。

## 4. 内存对照证明真正因果，也显示现有接入缺口

所有替换仅用于进程内探针，原snapshot未写入：

| 目标文件探针 | 结果 |
| --- | --- |
| 原始CUTE_ALIGNAS(128) | 105记录，3 scope mismatch，104隔离，两个真实scale字段未提取 |
| 只将两处展开为alignas(128) | 220记录，0 scope mismatch；两个真实字段及正确owner恢复；仍有879行独立pending |
| 只展开为CUDA __align__(128) | 仍105记录、3 scope mismatch，另加2 alignment_context_pending |
| 用GNU aligned语法作parser-only适配 | 220记录，0 scope mismatch，仍有879行独立pending |
| SourceProjection保留原CUTE_ALIGNAS语义spelling、只给parser GNU属性 | scope与字段范围恢复；仍有2 declaration_macro_expansion_pending加879行pending，且字段尚未挂接alignment结构化元数据 |

第三行失败的原因明确：`declaration_headers.py:311–350`现有alignment adapter只扫描字面`__align__`，而且342–343仅允许紧跟class/struct/union关键字的位置。它既没有处理CUTE_ALIGNAS的已知条件定义，也没有实现字段前缀位置。`CUTE_ALIGNAS`没有进入普通annotation allowlist，这一点应保留，不能为了过门禁把它加入可抹除列表。

源码映射探针保留了原完整字段签名、声明span和真实declared_type；但现有属性列表仍为空，alignment合同没有自动挂接到实体，macro pending也还在。这只是证明现有SourceProjection可以承载语法适配，不是完整实现完成。不能只凭scope诊断归零关闭alignment的语义义务。

## 5. 使用原CuTe header的Clang oracle：抹掉字段属性确实改变布局

Clang21.1.8、C++17执行以下自包含样本，直接include固定snapshot的原header：

```cpp
#include <cstddef>
#include "cute/container/array_aligned.hpp"
struct Probe : cute::aligned_struct<128> {
  char prefix;
  CUTE_ALIGNAS(128) cute::array<float,3> a;
  CUTE_ALIGNAS(128) cute::array<float,3> b;
};
static_assert(alignof(Probe)==128);
static_assert(offsetof(Probe,a)==128);
static_assert(offsetof(Probe,b)==256);
static_assert(sizeof(Probe)==384);
static_assert(alignof(cute::array_aligned<float,3,64>)==64);
```

有效命令（源从stdin输入）为：

```bash
clang++ -std=c++17 -I snapshot/include \
  -I /usr/local/cuda-13.0/targets/x86_64-linux/include \
  -I /usr/local/cuda-13.0/targets/x86_64-linux/include/cccl \
  -x c++ -fsyntax-only -
```

原样本退出0、无诊断。仅在该临时输入中删掉两个字段CUTE_ALIGNAS后，Clang退出1并给出具体反例：a偏移4而非128，b偏移16而非256，sizeof为128而非384；alignof仍为128，因为基类已施加整体对齐。**因此仅验证外层alignof会漏掉真实字段对齐损失。**

初次命令漏掉CCCL include路径，失败原因为找不到cuda/std/utility；补齐实际安装路径后才获得上述正反结果。该基础设施失败没有被当成源码错误或有效负例。此oracle核验普通Host分支的真实alignas合同，不声称已经编译CUDA的__align__分支，也不是目标FP8完整模板实例的offset测量；没有运行GPU。

## 6. 建议的最小source-mapped修复合同

建议在现有syntax/header适配层增加对固定CUTE_ALIGNAS定义链的有据识别，同时覆盖class-key后和字段前两种位置：

1. 为每个调用记录原始macro span、实参expression_span、定义来源/哈希、两个有效条件和替换拼写。CUDA支为__align__(n)，非CUDA支为alignas(n)。可以共享“requested_alignment，单位bytes”的语义属性，但不能伪造一个无条件实际展开拼写。
2. 给Tree-sitter使用可解析的对齐属性投影，例如`__attribute__((aligned(expression)))`；它只属于parser适配，不是把上游API改成GNU属性。semantic_spelling保持原宏调用，raw signature始终回切原字节；若显示expanded spelling，必须带相应定义条件。
3. 将结构化alignment属性挂到**真正的被修饰类型或字段出现记录**，包括alignment_expression、单位、原位置、来源宏及条件。Alignment/SmemAlignmentA/B保留依赖表达式和待绑定来源，不替换成128，也不丢弃外层已有alignment继承。
4. 完成字段/类型及alignment来源的双向绑定后，才能把相应宏候选从“未知声明宏”转为“已核验alignment属性义务”。不能将它简单分类成无API意义的非声明宏，也不能仅在文件级存一个属性表而不连接owner。
5. 如有局部#undef/重定义、未知宏body、实参arity/括号无法确认或属性位置不受当前适配支持，保留明确pending。不要把同名函数或任意大写调用当CUTE_ALIGNAS语义。

必要回归应包含：目标3个late_close正确恢复且104条隔离解除有真实身份依据；5文件16调用全入账；9个aligned_struct特化和array_aligned名字恢复；混合输入不再生成伪SmemAlignment member；两类位置的条件展开及属性owner保留；SourceProjection组合后签名/类型/偏移无临时属性泄漏；上述offset正反oracle；未覆盖CUDA编译分支和879等无关pending不能被默默关闭。

## 7. 本轮指纹与结论边界

- 目标源码：`d5a274f90d2f9a0f33c33c8def8c710ef9200097762df4c6752d4706746c33c8`。
- alignment.hpp：`b3c87930342925652ae4123f97ab716900b2d980e582a58b5abb37bde784c38f`。
- array_aligned.hpp：`4dd7cc1c6a73d52e6a74fb86f9d2e72c93c5edb721a228474660674c99151f3d`。
- extractor：`54f1c36919ced4a4b437c78fc7fdf645dfe2f675eecd0f025c07f8f9bac68539`。
- declaration_headers：`3a7c16588c7caa48f9599b800652cf7d714071f534fb17a6df6c9c22c0ea5081`。
- declaration_syntax：`bbefbaeb974132c3375739bfa3754cc0032c3ff3e78b7dcd80bd120913d8bd89`。
- scope_integrity：`3212d524ab351bd3a46cf37a70f61c12784fe64ab74b3edb223eabff17c719d1`。

开始和末次核对时上述核心/源码指纹未变。本轮定位了真实因果、全部固定调用位置、一个独立布局oracle及可用的source-map接入路径；没有实施正式修复，没有用alias/annotation masking藏掉alignment语义，也没有宣称全库解析清零或这5个文件的API关系已经完整。
