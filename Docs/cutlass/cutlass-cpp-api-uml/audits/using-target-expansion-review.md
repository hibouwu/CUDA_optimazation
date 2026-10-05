# using 目标表达式：固定宏分支帮助层与独立反例复核

本轮新增 `scripts/using_target_expansion.py` 与 `tests/test_using_target_expansion.py`，没有修改 core、reconciler、全局数据、模块或站点。该层实现的是**原始 target 表达式的有来源宏拼写分支**，不执行 C++ lookup，不创建目标 entity，不将 import／directive／继承构造器合并成 alias。

设计依据为 `data/using-interface-model.md` 的目标层。实现前完整读取相关 namespace_bindings 和 macro_expansion 代码，并检查既有 include 证明方法；未直接调用 namespace resolver：它不接受合法空 namespace replacement，且不能直接证明这里需要的完整 include 前缀顺序。新层复用独立 scanner 的 SourceScan／lex，保留其已核对的 phase-2 续行与注释／字面量规则，不复制一个逐行 using 正则器。

## 1. 冻结接口

```python
expand_using_target(
    path: str,
    source: bytes,
    target_span: dict,  # start_byte/end_byte，物理半开区间；可附 path/行号/hash
    source_reader=None # Callable[[repository_relative_path], bytes | None]
) -> dict
```

未传 reader 时只读 snapshot。路径不得绝对化或包含 `..`；无效区间、截断词法 token、与原文不符的可选 path／行号／raw_sha256 会抛 ValueError，属于调用合同错误。宏来源尚不能证明则正常返回结构化 pending，不猜测展开结果。

顶层保留 `source_expression`、`source_range`（物理字节／行号／raw_sha256）、`source_preprocessor_conditions`、`root_qualified_in_source`、`variants`、`proof`、`pending`、`diagnostics` 与 `completeness`。每个 variant 包含：

```text
variant_id
conditions
expanded_spelling / expanded_root_qualified
validation_status / source_contract_status
macro_definition_chain / macro_bindings
source_mapping
missing_external_bindings
inactive / condition_satisfiability / pending_reasons
```

`source_mapping.virtual_start_byte/end_byte` 只表示 expanded_spelling 内的 UTF-8 字节位置；`source_range` 始终表示实际原文件。原文非宏部分按 exact_source_slice 保留，宏 token 的替换映回其真实 token 区间。空 replacement 有零长度 virtual 区间、非零长度物理宏 token 区间，不丢掉发生过的替换。未改动的空白／注释不做展示性改写，所以 CMATH 原文 `MACRO :: isnan` 可输出带前导空白的 ` :: isnan`，其词法结果与 root-qualified 判断均是 `::isnan`。

结果不含 target entity_id、target_kind 或虚构的 qualified_name。无已知宏的 `::src::f` 仅保留 source-proven 原表达式；这不证明它有合法声明目标。所有结果明确 `cpp_lookup_performed=false`。

## 2. 三族固定来源及 override 状态

| 宏 | 固定 provider 与源位置 | 当前表达式分支 |
|---|---|---|
| CUTE_STL_NAMESPACE | cute/config.hpp:106–111 | RTC 为 cuda::std；另一分支为 std |
| CUTLASS_CMATH_NAMESPACE | cutlass/detail/helper_macros.hpp:136–140 | 定义 __CUDA_ARCH__ 时为空；另一分支为 std |
| CUTLASS_STL_NAMESPACE | cutlass/platform/platform.h:151–157 | `!defined(CUTLASS_STL_NAMESPACE)` 下再分 cuda::std／std；另保留外部已定义分支 |

三个 provider 的 SHA256、真实 replacement token、object-like 类别、分支条件均核验。任意改动 provider 原文件、额外本地同名 define／undef、未知同名替换或函数宏，不借这个固定模型自动获得 source_proven。

外部 override 的源码合同与提取失败严格区分：

- 两个固定默认分支：`validation_status=source_proven`、`source_contract_status=source_proven`，有确切 expanded_spelling。
- 外部已绑定但未提供 replacement：`validation_status=pending`、`source_contract_status=parameterized_external_binding`；expanded_spelling=null，保留所需外部 tokens／use-point rescanning。顶层是 `source_proven_with_parameterized_external_binding`，`diagnostics=[]`，`all_source_branches_accounted_for=true`，但 `all_expansion_spellings_proven=false`。
- include、定义／undef 顺序、来源 hash、未知替换等证明失败：variant 和顶层是 `source_contract_pending`，同时提供 diagnostics。调用者无需从英文 reason 字符串猜测这是外部参数还是提取缺口。

外部分支的正条件由 provider 第 151 行负 guard 的补集派生，保留原 directive evidence，同时用独立 `derived-complement:` branch_id 和 `derived_from_branch_id` 区分，不能冒充原物理分支。

## 3. include 前缀与宏状态的证明边界

帮助层按真实 directive 顺序读取 using 之前的 root 前缀；被包含头文件读取完整内容，因此 provider 定义之后的 tail include 也必须经过检查。每个证据记录文件 hash、物理 directive 区间、条件和 include 链。定义在 using 后面的宏不倒灌到此前位置。

本库已经存在的 config→debug→config 回边按真实无条件 `#pragma once` 处理。只有**无条件进入**的头文件才可加入跨路径 once_seen；条件进入只在本次 active inclusion 中截断递归，不能宣布其他配置也已经访问过。未知 include cycle 或预算不足保留明确 pending。

只提供 conditional provider include 时暂不求完整配置状态，返回 conditional_provider_visibility pending；不把这条链包装成无条件可见。local／transitive undef、provider tail 中的重定义、replacement token 自身还需宏 rescan，都保持 pending，而不替用户执行通用预处理器。

`CUDA_STD_HEADER(...)` 只有在调用点前可见唯一、无条件、未经覆写的固定 cutlass/cutlass.h 定义，且参数为 header、body 为 `<cuda/std/header>`、来源 hash 正确时，才归为 external_generated_standard_header。它仍是外部边界，不伪造库内 include。未知生成 include、缺失库内源文件、source_reader I/O 失败均保留诊断；普通标准库／CUDA 外部头不是 fixed-scope missing_definition。

外部头和命令行宏环境没有被执行，这是明确的环境边界；本层只证明 supplied／fixed source 前缀中的状态。允许外部 override 的固定 guard 已结构化记录，不借边界说明隐藏它。

## 4. 条件关联不是无条件笛卡尔积

同一宏在一条 target 表达式中出现两次时只选择一次 replacement，所以 `Box<CUTE_STL_NAMESPACE::T,CUTE_STL_NAMESPACE::U>` 产生两个相关结果，不产生混用 std／cuda::std 的四个组合。

原始 using 的 source conditions 与 target definition 条件共同保留。无 source define／undef 改变的 `defined(__CUDACC_RTC__)`／`defined(__CUDA_ARCH__)` 原子冲突可以排除，并保留 excluded_condition_combinations。例如 source 已在 RTC 分支时不再输出 std 分支。

条件记录还带真实 `evaluation_point`：宏定义 guard 检查的是 directive 进入时的状态，不能一律当成 using 处的当前状态。若源码在 include 前后改变这些配置宏，或 source condition 又读取已被前缀改过的宏，本层明确 pending。特别是 provider 的 `!defined(CUTLASS_STL_NAMESPACE)` 会被其 body 的 define 改变；不能把它与使用处后来的 `#ifdef CUTLASS_STL_NAMESPACE` 直接当成矛盾而删除默认分支。

复杂预处理算术或条件的可满足性不由本层求解。原始 conjunction 继续保留，`condition_satisfiability` 明示只识别字面恒假与稳定 defined 原子的冲突；source_proven 指的是条件展开规则及来源，不宣称任意条件组合必可运行。namespace owner 与 target 的进一步条件连接由 core/context 层完成，不能只取 target 的第一分支。

## 5. 实际测试与固定源码规模

`tests/test_using_target_expansion.py` 的 **23 项实际通过，最后一次耗时约 1.85 秒**，包括：

- 三个真实 provider 的字节／条件／include 证明；合法空 replacement 与根限定；CUTLASS override 第三分支的结构化状态。
- using 前后 include／undef 顺序、transitive undef、pragma once、条件 include、provider tail 变更、本地 override、未知 generated include。
- phase-2 续行注释不制造假 undef；UTF-8 前缀与跨行宏 token 的物理字节映射；字符串内容不被替换；同名宏多次使用保持同一选择。
- return 数据被调用者修改不会污染后续 cached scan；错误物理 token 区间被拒绝。
- 全部六个实际相关文件中 120 处非 alias using target：75 个 CUTE_STL_NAMESPACE、40 个 CUTLASS_STL_NAMESPACE、5 个 CUTLASS_CMATH_NAMESPACE。每处保留两个可证明的默认 spellings；只有 40 个 CUTLASS_STL_NAMESPACE 各自再保留一个 parameterized_external_binding。没有其他 source-contract diagnostics。

六文件为 cute/util/type_traits.hpp、cute/numeric/int.hpp、cute/numeric/integer_sequence.hpp、cute/container/array.hpp、cutlass/platform/platform.h、cutlass/functional.h。这一 120 是当前三族宏的源码出现数，不是全部 using 接口、更不是全库 API 分母。

既有 `test_using_declaration_independent_review.py` 没有修改；其四项独立 Clang semantic oracle 另行实际复跑通过，继续证明 import／alias／directive、overload、宏目标 namespace、继承成员与构造器访问边界。没有把那些编译器结果换成本模块输出的自我比较。

## 6. 独立反例审查：两个首版错误接受及关闭证据

独立审查者只读新模块并用 stdin 内存 fixture 检查，未改实现、测试或数据；发现两个此前会错误给出 source_proven 的具体反例。

**T01：条件 include 的一次性访问状态污染其他配置。** consumer 先在 RTC 分支 include 一个 pragma-once mutator，其内部仅在非 RTC 时 undef CUTE_STL_NAMESPACE；之后又无条件 include 同文件。首版第一次因矛盾跳过 undef，却全局设置 once_seen，第二次也跳过，错误输出 std／cuda::std 两个 proven 结果。修复限定全局 once_seen 为无条件进入；独立重跑原反例 now 得到一次真实 undef 事件、source_contract_pending、expanded_spelling=null。该反例已加入测试。

**T02：只认 CUDA_STD_HEADER 名字，跳过明确本地 mutating include。** consumer 先包含 config，再本地定义 `CUDA_STD_HEADER(x) "hidden_mutator.hpp"`，然后包含它；reader 明确提供头文件内 `#undef CUTE_STL_NAMESPACE`。首版不检查宏定义，将它当外部标准库头，错误输出 proven 结果。修复后只有固定来源、定义、参数和先后均被证明时才允许外部分类；原本地覆写 now 得到 unresolved_generated_include / source_contract_pending，不输出错误展开。没有为此扩展通用 include 宏求值器。

独立审查者在修复后只重跑原两个反例，确认二者均关闭，顶层 diagnostics 正确、all_expansion_spellings_proven=false；没有拿原本 pending 的能力边界冒充新缺陷。

## 7. 冻结与仍未实现的部分

宏语义检查点 SHA256（随后第 8 节仅修订隔离导入，保留此历史指纹）：

```text
scripts/using_target_expansion.py
8063b06003d473880ad7d6c6d1fa2e3fa22aef097b507fd13e3eff11f624703d

tests/test_using_target_expansion.py
9b87674e8a822e6984b88d2a86a625f0a570e17b0fa3f8786aeb330462f04525
```

仍不做：一般宏递归／prescan／rescan、任意 include 宏求值、复杂条件全状态预处理、C++ scope 或 overload lookup、目标实体身份、继承构造器可调用性、using 旧错误 entity 的迁移。包括显式本地 namespace 宏替换、cutlass namespace customization wrapper 在内，若超出这里三族证明模型，会留下具体 pending，不借一个表面可读的 token 串宣称完成。

本轮没有重新生成全库 ledger 或 mapping，也没有修改候选分母；新帮助层供 Root 后续接入 source/context 层。帮助层完成不等于 using 全局身份修复已完成，更不等于 target resolution 或最终全库阶段通过。

## 8. 隔离加载接入修订与最终指纹

Root 使用 importlib 从相邻绝对路径加载帮助层时，首版 `from scan_candidates import ...` 仍依赖调用者设置 sys.path。这是模块导入合同错误，不是宏语义错误。现改成以唯一模块名 `_using_target_expansion_scan_candidates`，通过该帮助层相邻 scan_candidates.py 的绝对路径／spec 加载；没有修改全局 sys.path，也没有改 scanner。

新增真实隔离子进程测试：用当前 venv Python 的 `-I -B`，工作目录 `/tmp`，仅以绝对路径导入帮助层，然后实际展开 type_traits.hpp:92；确认 std／cuda::std 两分支均成立，并断言 scripts 目录不在子进程 sys.path。不是在已有测试进程预设 sys.path 后假称隔离导入通过。

全部 **24 项测试实际通过，最后一次 1.997 秒**。此次只改变导入机制，原 23 项宏／位置／include 顺序及 120 真实位置的检查继续通过。最终冻结：

```text
scripts/using_target_expansion.py
4faf35b9df9dfea4bf0301481ea6b52372c8593758c2e0064f28df8d78e0f9af

tests/test_using_target_expansion.py
ac05a0acdfc92f9504666792c6d0c5ff077997678fb691e0bf1f7913bf3d0b33
```

调用与状态 schema、外部 override 边界、T01/T02 的关闭结论均未改变。本文件只作为实现及反例审查检查点；未参与后续 core/context 接入、全库升级或站点验收。
