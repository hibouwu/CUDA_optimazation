# CUTE_ALIGNAS：独立原始语义与Clang反例

本稿先核准固定源码合同，不以待接入的alignment提取器结果定义分母。固定scope仍为824文件，逐文件SHA与scope.json匹配。只新增 `tests/test_cute_alignment_review.py` 和本文；没有改core、snapshot或全局数据。

**前5节保留原始语义检查点：11项独立source/Clang测试实际通过。** 适配器、occurrence归属与独立reconciliation的后续复审记录在第6节，不以最初的source测试代替接入验证。

## 1. 两个真实条件定义

定义在snapshot的 `include/cute/container/alignment.hpp:51–55`，不是cute/config.hpp：

```cpp
#if defined(__CUDACC__)
#  define CUTE_ALIGNAS(n) __align__(n)
#else
#  define CUTE_ALIGNAS(n) alignas(n)
#endif
```

条件轴是`__CUDACC__`，不能替换成`__CUDA_ARCH__`。两种配置的原始替换拼写不同；共用的对齐语义也不能据此变成无条件的同一实际宏展开。独立测试抽取这段**原始定义块**，用Clang预处理在定义/不定义__CUDACC__时分别得到 `__align__(Alignment)` / `alignas(Alignment)`，两个结果均实际核对。

这项预处理oracle不声称已在CUDA设备模式下编译__align__；后文布局oracle编译的是普通Host分支，不能把两种验证混合计作CUDA执行证据。

## 2. 固定16处使用与属性owner

独立lex排除宏定义、注释和字符串后，确认为5个文件中的16处调用；没有引用声明提取器的成功输出。以下范围是宏调用本身的半开物理字节区间。

| 文件、物理行 | 宏范围 | 原始alignment表达式 | 属性所属声明 |
| --- | --- | --- | --- |
| cute/container/alignment.hpp:60 | [2458,2475) | 1 | aligned_struct<1,Child>特化 |
| 同文件:61 | [2537,2554) | 2 | aligned_struct<2,Child>特化 |
| 同文件:62 | [2616,2633) | 4 | aligned_struct<4,Child>特化 |
| 同文件:63 | [2695,2712) | 8 | aligned_struct<8,Child>特化 |
| 同文件:64 | [2774,2791) | 16 | aligned_struct<16,Child>特化 |
| 同文件:65 | [2853,2870) | 32 | aligned_struct<32,Child>特化 |
| 同文件:66 | [2932,2949) | 64 | aligned_struct<64,Child>特化 |
| 同文件:67 | [3011,3028) | 128 | aligned_struct<128,Child>特化 |
| 同文件:68 | [3090,3107) | 256 | aligned_struct<256,Child>特化 |
| cute/container/array_aligned.hpp:40 | [2033,2056) | Alignment | array_aligned类模板，不能冒出类名CUTE_ALIGNAS |
| gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input.hpp:271 | [14787,14815) | SmemAlignmentA | SharedStorage::TensorStorage::smem_A字段 |
| 同文件:272 | [14898,14926) | SmemAlignmentB | SharedStorage::TensorStorage::smem_B字段 |
| gemm/collective/sm90_mma_tma_gmma_rs_warpspecialized_mixed_input.hpp:291 | [15108,15136) | SmemAlignmentA | SharedStorage::TensorStorage::smem_A字段 |
| 同文件:292 | [15219,15247) | SmemAlignmentB | SharedStorage::TensorStorage::smem_B字段 |
| gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp:216 | [10857,10874) | 128 | SharedStorage::TensorStorage::smem_SFA字段 |
| 同文件:217 | [10980,10997) | 128 | SharedStorage::TensorStorage::smem_SFB字段 |

表中cute路径以`include/`为前缀，其余以`include/cutlass/`为前缀。前10处位于class-key之后；后6处是字段类型之前的属性。后6处不能误归给TensorStorage外层struct、SharedStorage或名叫SmemAlignmentA/B的伪成员。

依赖表达式共5处：Alignment一次，SmemAlignmentA/B各两次。普通mixed_input的277/279行、array版本261/262行从 `alignment_for_swizzle(SmemLayoutA/B{})`计算；284/265行的assert只给出>=128，**不能推出等于128**。array_aligned的类模板形参Alignment默认16，也不能因此把所有实例固定成16；本次Clang实际实例化Alignment=64验证了此区别。

## 3. 原header字段offset的独立Clang正反

本次实际使用Clang21.1.8，命令从stdin输入C++17源码，直接include固定snapshot原header：

```text
clang++ -std=c++17 -I <atlas>/snapshot/include
  -I /usr/local/cuda-13.0/targets/x86_64-linux/include
  -I /usr/local/cuda-13.0/targets/x86_64-linux/include/cccl
  -x c++ -fsyntax-only -
```

输入是继承 `cute::aligned_struct<128>` 的Probe，含char prefix，以及两个 `CUTE_ALIGNAS(128) cute::array<float,3>` 字段a/b。这是直接使用原header的缩减布局oracle，不是完整FP8 kernel实例布局或GPU测量。

| 检查 | 原字段属性存在 | 只删两个字段属性 |
| --- | --- | --- |
| alignof(Probe) | 128 | 128，仍不变 |
| offsetof(a) | 128 | 4 |
| offsetof(b) | 256 | 16 |
| sizeof(Probe) | 384 | 128 |

三次编译实际执行：原正例退出0；删除属性但保留原offset/size断言时退出非0，stderr包含独立的review_offset_a、review_offset_b、review_total_size标记；按4/16/128写出的反例实际值断言再编译退出0。没有把缺CUDA/CCCL头文件等基础设施失败当有效负例。**只检查外层alignof会漏掉真实字段对齐丢失。**

## 4. 局部宏生命周期及非法形状控制

独立Clang控制进一步证明：

- include原header后#undef，再定义`CUTE_ALIGNAS(n) alignas(2*(n))`，`struct CUTE_ALIGNAS(16) Local`的alignof为32，编译成功。不能继续套用原宏得到16。
- include后#undef，可以定义名为CUTE_ALIGNAS的普通constexpr函数，`CUTE_ALIGNAS(16)==17`断言编译成功。此后同名调用不是对齐属性。
- 在原宏绑定下，`CUTE_ALIGNAS(16,32)`、空实参`CUTE_ALIGNAS()`、return表达式位置的`CUTE_ALIGNAS(16)`均被Clang拒绝。这不等于所有局部重定义下的同形调用都非法；提取器必须按可证明的宏绑定及owner判断，不能只看名字。

实现接入时应拒绝把这些位置算作已核清原CUTE_ALIGNAS属性；无法解释局部#undef/重定义、错误arity、未知位置或括号结构时保留明确pending/非对齐绑定证据，不能通过抹掉宏调用消除错误，也不能把对齐属性加入可无条件擦除的annotation列表。

## 5. 当前测试检查点

实际命令：

```bash
.venv/bin/python -B -m unittest discover -s tests -p test_cute_alignment_review.py -v
```

11项测试通过，耗时约9.9秒。测试文件SHA256为 **`b46605637c6eba428fdd5b14a16032c5fe42a42ac24fcffc7c9b9eaa42bd05f2`**。它目前不import待修core；source边界和Clang反例先独立成立，避免生成器自证。

尚待root接口后的接入核对：16处全部挂到正确owner，原macro/expression/字段位置可回切；条件定义同时保留；依赖表达式不固化；局部宏变化及非法/未知位置不被盲用；两种位置的字段/类型属性不互相错挂。当前结果不宣称新适配器或全库覆盖已完成。

## 6. 接入与独立对账复审：固定16处通过

上节的接入计划已经执行。本检查点有17项source/Clang/adapter/core测试、6项reconciliation测试，共23项独立测试方法。最后一次6项对账测试还使用`-W error::ResourceWarning`实际通过；SQLite连接由test cleanup关闭，不再出现未关闭连接警告。

### 适配器与occurrence归属

对5文件实际调用analyze_headers，全部16处有精确macro span、expression_span、alignment_expression及type/field owner_hint。每个字节范围都回切原文；字段owner span只包括对应字段声明，不包整个TensorStorage。两个definition及其__CUDACC__相反条件、原展开拼写和到alignment.hpp的每一条include边均核对源码。

再调用当前Extractor：每个实际owner occurrence的alignment_specifiers指向自己的owner_occurrence_id/owner_entity_id；文件级alignment_attributes的复数owner IDs恰好包含全部对应实例。CUTLASS字段的默认/定制namespace两实例均保留，未把属性错挂到伪SmemAlignmentA/B成员或上层struct。类型声明、字段声明的kind和parse_status也逐项检查。

局部#undef产生明确non_alignment_spellings，局部重定义、条件#undef、__align__重定义不套用固定宏语义；缺include来源、错误/空arity、表达式位置与当前不支持的global-variable属性位置均保留明确pending，没有生成已核清owner。FP8 blockwise字段恢复后没有scope_closure_mismatch，但879行原有独立pending仍为blocking，未被对齐适配顺带清除。

### Reconciler的独立证明

新增alignment_mapping只拦截source级CUTE_ALIGNAS invocation，在普通macro fallback前执行；没有改变syntax_interval的一般声明对账规则，也没有修改470518条冻结候选。它不调用analyze_headers或Extractor来判真，而是使用独立SourceProof/lex核对：

1. 完整macro调用、原表达式、bytes/物理行与candidate原文hash；只接受本次实际出现的单identifier或整数表达式，遇到新表达式形式保持pending，不做通用C++常量求值。
2. 正确type/field种类、class-key/field-prefix位置、精确owner syntax span、名称锚点、单owner ID及owner本身的可验证状态。
3. alignment.hpp的两个真实definition、formal n、body、source hash、条件位置及展开实参替换；少任一分支不能通过。
4. 到固定provider的逐步、无条件、位于使用之前的literal include链；本文件或所引链上的相关局部define/undef会使固定合同证明失败。
5. occurrence/file两层属性元数据一致、全部owner ID存在、namespace binding alternatives完整。

通过的调用候选输出`classified_non_api`、obligation=`phase1_declaration_attribute`，并保留完整alignment_specifiers、owner occurrences及binding实例；含义是“该词法候选为已证明的声明属性”，**不是没有API价值、不是删除owner声明，更不是新生成了一类API**。不能完成上述证明的CUTE属性候选仍为pending。

### 篡改与SQLite回程测试

手写正向fixture独立于声明/headers提取器。对occurrence和file两份元数据同时作相同篡改，以免测试仅靠二者不一致轻易拒绝：改表达式、表达式范围/行号、属性行号、owner kind/range、少一条件分支、改条件轴、删除条件或条件范围、改展开式/body/定义行/hash、删除或改include边，全部得到pending。

另验证错误owner ID/entity ID、删owner、删file元数据、scope taint、局部macro变更均不能通过。实际5文件的16处属性全部通过；对6个CUTLASS字段逐一删去一个namespace owner实例，均因binding alternatives不全而pending。

stage_declarations的SQLite文件slim现在保留`syntax_analysis.header_analysis.alignment_attributes`及其复数owner IDs；occurrence slim保留单owner alignment_specifiers。独立roundtrip测试将手写完整声明JSON经SQLite暂存、读回，再执行同一证明，分类仍通过。不会出现内存小样通过、全库阶段却因slim删掉file属性而全部pending的脱节。

### 冻结指纹与交付边界

| 文件 | 本检查点SHA256 |
| --- | --- |
| reconcile_candidates.py | `9ea91ad5a2ebe41c8914b4418d3381131dd9baabeca2e05cb998730334b29a79` |
| tests/test_alignment_reconciliation.py | `595fdd61a9e8b8dc660ddf036a34301541aed1b4608d186f40f9999d2e88a9be` |
| tests/test_cute_alignment_review.py | `789a00aad0a64188f713edf8a1b58e3a43812d141642dbbc13ba8ab3442aeb9d` |
| declaration_headers.py（只读检查） | `9ef704a75c48342ab501b27e2b893c2d73b64ccccf914d5c26b39ffa233c0576` |
| extract_declarations.py（只读检查） | `da8d2007310e15cbb5abe6924e402e2d71789887419da94aab34d608395abff1` |

代码与测试已通知Root冻结，后续仅写本文；全库mapping由Root另行运行，本审查没有重写全量JSON。**本次固定16处CUTE_ALIGNAS的源语义、adapter/owner接入与候选属性分类定点通过。** 它不扩展为任意宏生命周期引擎、CUDA设备布局运行证明或全库所有pending清零；其它语法/宏义务照旧保留。
