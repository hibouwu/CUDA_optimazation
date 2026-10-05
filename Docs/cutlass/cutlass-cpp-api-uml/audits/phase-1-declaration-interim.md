# 阶段 1 声明身份独立中期审查

审查日期：2026-09-08。固定源提交：`8f50b052e1099fb982392a622caab69b97b63128`。

## 结论：阶段 1 仍未通过

本轮发现四个仍可复现的声明身份/作用域问题。它们不是“没有生成足够多实体”，而是源码中的函数被记成成员变量、不同重载合并、同一函数的 friend 声明与定义被拆分、同一命名空间按源码写法被拆成不同身份，以及宏控制的命名空间被当成真实字面命名空间。最小反例均为零诊断，说明仅把 parser error 清到零不能关闭这些缺口。

本轮只写此审查文件；没有修改提取器、测试、候选账本、声明 JSON、快照或上游。为避免同时加载约 656 MiB 的声明 JSON 与约 271 MiB 的候选账本，直接用 `Extractor` 对固定文件及缩减反例重跑。**本文没有声称已核对整份 `data/declarations.json`，也没有把小样本修复通过写成已落盘全库数据修复通过。**

提取器在本轮期间继续迭代，因此以下明确区分修订：

- 初次固定源码探针：`extract_declarations.py` SHA-256 `1838afb7a23c5dbad892ad113148c9af639eead7cc9072caa40d98dd61ede9ea`。
- 引用返回、friend、命名空间四类缩减反例和 `array_subbyte.hpp` 详细记录复核：`f4c5e1d12400d8664286f32264ff9abd1ae76e87fcbd10b6e3114671ed39a5fb`。
- `mma_sm90.hpp` 修复后及两个真实文件命名空间身份复核：`fb53fc6ccce759041ae8ee2e3e52c7532bedd5556bf28bd847a3aafa16cbdab4`。
- 本文最后一次四类反例与 23 个宏 operator 复核：`e14a142ed23d503c63a0c2f3251f0ad857204be262c42548990b1a230c523136`。同次执行前后散列相同，四类反例仍失败。

## P1-D01：引用返回函数被当成数据成员，重载被静默合并

固定源码：`include/cute/container/array_subbyte.hpp:260–267,278–287,295–304`。

真实接口分别是返回 `subbyte_iterator&` 的 `operator+=(uint64_t)`、前置 `operator++()` 和前置 `operator--()`。提取结果却为 `kind=member_constant`，整个函数体进入 `initializer`，`parameters` 和 `return_type` 字段缺失。这三个出现位置都为 `parse_status=parsed`；该文件唯一诊断位于 560 行的 `CUDA_STD_HEADER(tuple)`，与这三个接口无关。

例如 260 行记录：

```text
qualified_name = cute::subbyte_iterator::operator+=
kind           = member_constant
entity_id      = ent_acf041be6a86e8fdd10545ad
parse_status   = parsed
parameters     = absent
return_type    = absent
initializer    = { k = sizeof_bits_v<value_type> * k + idx_; ... }
```

去掉 CUDA、模板和宏后仍可复现，因此不是依赖环境才可判定的歧义：

```cpp
struct S {
  S& f(int)    { return *this; }
  S& f(double) { return *this; }
};
```

最后复核结果：0 个诊断，两项均为 `member`，都使用 `ent_3e8acab7b68bcb5ea7d8f5de`。若添加 `constexpr`，只是改成 `member_constant`，重载合并问题不变。两个调用接口没有作为 callable 入账，是“实体记录存在但 API 身份丢失”，不能被候选已有映射掩盖。

原因定位：`function_declarator()` 只沿 `child_by_field_name("declarator")` 下降；引用返回 declarator 的节点结构没有按其假设暴露这个字段，导致落入变量分支。变量身份只含名字，所以进一步吞掉参数不同的重载。

关闭条件：按真实 declarator 结构辨认函数与函数指针对象；引用返回函数必须保存参数、返回类型、限定符、函数体区间，不把函数体当 initializer；上述 `f(int)` 与 `f(double)` 必须得到两个 callable entity。应同时检查前置/后置 operator 和普通返回引用方法，不能只修 `operator+=` 拼写。

## P1-D02：friend 的出现位置被误当成语义 owner

固定源码：`include/cute/container/array_subbyte.hpp:233–235,332–337`。233 行的 friend 模板声明与 332 行开始的 `raw_pointer_cast` 定义属于命名空间函数，不是类的方法。

实际记录：

```text
233: method   cute::subbyte_iterator::raw_pointer_cast
     ent_dee967b6339eb31f520402f1
332: function cute::raw_pointer_cast
     ent_03012c84fec979e872f6c81d
```

234 行 `recast_ptr`、235 行 `print` 同样记成类方法。`include/cute/pointer_sparse.hpp:54–59` 的 friend 比较运算符也被赋予 `cute::sparse_elem::operator...` 的类成员限定名。friend 标志虽然存在，但没有改变 owner 或恢复签名范围：其中 `raw_signature` 还会从 `bool` 开始，丢掉前面的 `CUTE_HOST_DEVICE constexpr friend`，`attributes` 也相应不完整。

缩减反例：

```cpp
namespace n {
  struct S { friend int f(S const&); };
  int f(S const& x) { return 1; }
}
```

最后复核为 0 个诊断，却生成 `method n::S::f` 和 `function n::f` 两个实体。这里必须分开“friend 声明写在类内的出现位置”和“函数属于命名空间的语义身份”；把同一个 `scope_chain` 同时用来表达两者不能满足契约。

关闭条件：非成员 friend 保留类内声明位置和授友关系，但函数身份绑定到正确的命名空间；模板参数中区分外围类参数与该 friend 自己的模板参数。应与后续命名空间声明/定义归并，并保留 Host/Device、constexpr、friend 等完整声明前缀。

## P1-D03：复合命名空间与逐层写法被拆成不同身份

两个真实文件均为 0 个诊断：

```text
include/cutlass/detail/dependent_false.hpp:35
  namespace cutlass::detail
  qualified_name = cutlass::detail
  entity_id      = ent_87236719fd6dc676e108d30f

include/cutlass/relatively_equal.h:40,50
  namespace cutlass { namespace detail { ... } }
  qualified_name = cutlass::detail
  entity_id      = ent_586da25d90f989528f7a63c1
```

复合写法仅记录一个 `name=cutlass::detail`、空父链的 namespace；逐层写法则记录 `cutlass` 下的 `detail`。两种等价语法最终进入不同身份键。更严重的下游效果可以用同名函数证明：

```cpp
namespace a::b { int f(); }
namespace a { namespace b { int f(); } }
```

最后复核为 0 个诊断，两项 `a::b::f` 分别得到 `ent_a03d1239ee95636d975eab6e` 和 `ent_3f37831da5b6e6fb20b5c66d`，没有归并为同一函数的两个出现位置。

关闭条件：复合 namespace 按语义层级建立或复用中间命名空间身份，原来的复合源码区间作为声明出现位置保留。不能仅把最后显示名规范化，因为当前两个显示名已经完全相同，错误位于 owner identity。

## P1-D04：宏控制命名空间被当成真实字面命名空间，没有未解析诊断

固定源码：`include/cute/config.hpp:106–111` 明确给出：

```cpp
#if defined(__CUDACC_RTC__)
#  define CUTE_STL_NAMESPACE cuda::std
#else
#  define CUTE_STL_NAMESPACE std
#endif
```

`include/cute/container/array_subbyte.hpp:594–613` 使用 `namespace CUTE_STL_NAMESPACE`，随后声明 `is_reference`、`tuple_size`、`tuple_element` 特化。当前结果为 `namespace CUTE_STL_NAMESPACE`、`CUTE_STL_NAMESPACE::tuple_size<...>` 等正常实体，`parse_status=parsed`；没有针对这个宏 owner 的未解析诊断。独立候选账本按已知对象式宏拼写保留了这类位置，因此不能把宏候选直接映射到一个字面 namespace 实体就算闭合。

同文件定义宏的缩减反例也为零诊断：

```cpp
#define API_NS std
namespace API_NS { struct X {}; }
```

最终记录仍是 `API_NS::X`。因此这个问题不仅是单文件提取时未读取 include 的局限；即使宏定义已在当前文件中也没有正确展开或显式标明缺口。

关闭条件：影响 namespace/type/name 身份的对象式宏必须纳入宏处理或产生明确阻断项。对固定源，应保留 `__CUDACC_RTC__` 条件下 `cuda::std` 与其他条件下 `std` 的 owner 变体及宏来源，不能把 `CUTE_STL_NAMESPACE` 冒充可解析的真实命名空间。

## 已修复的回归：mma_sm90.hpp 条件分支恢复曾污染全局身份

初次探针的 `1838afb7...` 版本对 `include/cute/arch/mma_sm90.hpp` 生成 942 个出现位置、53 个实体、1,858 个诊断：其中有 868 个 `constant`、32 个 `function`，包括多个以 `constexpr` 为名字的伪函数。`ss_op_selector` 被截成 354–682 行，后续 selector 失去 `cute::SM90::GMMA` 前缀。这属于“解析错误后的派生实体不可信”，不是可以继续用于关系提取的 API 清单。

`fb53fc6c...` 版本已经针对错误函数体内的预处理指令作等长修复，并用独立词法大括号检查函数范围。相同固定文件复测得到 46 个出现位置、45 个实体、0 个诊断；868 个伪常量及伪 `constexpr` 函数消失，四个 selector 的名字和范围为：

- `cute::SM90::GMMA::ss_op_selector`：354–2595。
- `cute::SM90::GMMA::ss_op_selector_sparse`：2597–4838。
- `cute::SM90::GMMA::rs_op_selector`：4840–7082。
- `cute::SM90::GMMA::rs_op_selector_sparse`：7084–9326。

本次针对该文件的这个问题关闭，应保留其反例，不再把旧版污染计入当前失败。没有证据表明该文件存在被误收的函数局部 `using`；它的 24 个 `using` 都是六个架构 struct 内的 Registers 别名。本轮不把任务提示中的怀疑写成已证实事实。

## 已核对的正向样本：23 个宏生成 operator

在 `e14a142e...` 版本，对 `include/cute/numeric/integral_constant.hpp:223–248` 独立筛选宏来源的出现位置，实际得到 23 个 operator、23 个不同 entity：5 个一元参数、18 个二元参数。

这 23 项均为 `cute` owner、`parse_status=parsed`，保留 `CUTE_HOST_DEVICE`，有虚拟展开位置、宏定义位置、原物理调用位置及展开序号。原物理范围切片均对应 `CUTE_...(...)` 调用；一元 `+` 的返回类型确为 `C<(+ t)>`，定义来源在 204–209 行，调用在 223 行。该样本的“宏调用只入候选而没有生成实体”问题没有重现。

以上只是这 23 个固定调用的结构/出处核对，不代表其他宏或 `integral_constant.hpp` 全文件已经完成，也不代表展开后的 CUDA 编译、数值或运行行为通过。

## 复现四类零诊断失败

在本项目根目录运行以下只读探针。它不写声明 JSON，只打印每项的种类、限定名与实体 ID；未来修复后输出应改变，而不是把这些输入删除。

```bash
.venv/bin/python - <<'PY'
import importlib.util
import sys
from pathlib import Path

script = Path('scripts/extract_declarations.py')
spec = importlib.util.spec_from_file_location('audit_extractor', script)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

cases = {
    'reference_overloads':
        'struct S { S& f(int) {return *this;} S& f(double) {return *this;} };',
    'friend_owner':
        'namespace n { struct S { friend int f(S const&); }; '
        'int f(S const& x) {return 1;} }',
    'namespace_identity':
        'namespace a::b { int f(); } '
        'namespace a { namespace b { int f(); } }',
    'macro_namespace':
        '#define API_NS std\nnamespace API_NS { struct X {}; }',
}
for name, source in cases.items():
    extractor = module.Extractor('review_fixture')
    extractor.extract('include/cute/review.hpp', source.encode())
    print(name, 'diagnostics=', len(extractor.diagnostics))
    for occurrence in extractor.occurrences:
        print(occurrence['kind'], occurrence['qualified_name'], occurrence['entity_id'])
PY
```

真正的关闭要求是四个语义反例修复、固定源重跑并重新对账，且原始候选映射不能把错误 kind/owner 的实体当成“已覆盖”依据。当前阶段 1 判定继续保持未通过。

## 第二轮独立复审：D01–D03 关闭，D04 分开记录误报与覆盖

复审日期仍为 2026-09-08。本节追加，不覆盖以上初审证据。复审时核心实现已锁定：

```text
extract_declarations.py
489fcfc5f53e93d7ced9d2074836510a6ac35167268c1f4db85d3d6d4573343a

macro_expansion.py
30bc274f61294046d95cc1091fca45a603b86992d14bd1fcb37e27fab58e3a31

declaration_projection.py
bb2464afa35a159d6adaf45f87d9ff26d9ffcd3c1c1631a2f15a3b02abe52997
```

复审直接重跑本文四个缩减反例及对应固定源文件，不以制作方测试通过作为独立结论。提取器同次执行前后散列相同。

### D01：关闭

`S& f(int)` 和 `S& f(double)` 现在均为 `method`，返回类型都是 `S &`，参数分别为 `int` 和 `double`，实体分别为 `ent_14da2df68b64ff338aa172bb` 和 `ent_34ab28b8ee4cca498b6d1c1b`。没有再按数据成员名字合并重载。

对固定的 `array_subbyte.hpp` 重跑：260 行 `operator+=` 为 operator，返回 `subbyte_iterator &`；278 行前置与 288 行后置 `operator++`、295 行前置与 306 行后置 `operator--` 均为不同 operator entity，分别返回引用和值。参数、返回值和函数体不再丢失或进入变量 initializer。原错误分类及由此引起的重载合并关闭。

### D02：关闭

缩减反例的 friend 声明与命名空间定义现在均为 `function n::f`，都连接 `ent_8d991246a4a8ed4d53fe7e45`。

固定文件 `array_subbyte.hpp` 的 233 行 friend 声明与 332 行定义也均为 `function cute::raw_pointer_cast`，连接同一实体 `ent_03012c84fec979e872f6c81d`。类内词法位置仍作为出现位置证据保留，没有靠挪走原始位置归并。

同时复核 `pointer_sparse.hpp:54`：限定名已为 `cute::operator==`；`raw_signature` 以 `CUTE_HOST_DEVICE constexpr friend bool operator==...` 开始，`attributes` 包含 `CUTE_HOST_DEVICE`，`prefix_specifiers` 包含 `constexpr` 和 `friend`。本轮连同之前指出的前缀丢失一起关闭。

### D03：关闭

缩减反例中复合与逐层写法的两个 `a::b::f` 现在都连接 `ent_3f37831da5b6e6fb20b5c66d`。

对 `dependent_false.hpp` 与 `relatively_equal.h` 在同一 Extractor 中重跑，两项 `cutlass::detail` 都连接 `ent_586da25d90f989528f7a63c1`。核对的是 owner identity 真正归一，而不只是显示字符串相同。

### D04：静默误导关闭，条件命名空间实体展开仍待完成

缩减反例现在产生阻断诊断 `namespace_macro_expansion_pending`，不再在零诊断状态下把宏 spelling 当成已经解析的 namespace。

固定的 `array_subbyte.hpp:594` 同样产生该诊断；`name_resolution.status=macro_dependent`，保存 `cuda::std` 候选及 `config.hpp:107` 定义和 `defined(__CUDACC_RTC__)` 条件，以及 `std` 候选、110 行定义和对应否定条件。`missing_binding` 明确要求确定宏定义条件并构造 namespace-owner 变体。

该文件现有两个诊断：594 行 namespace 宏绑定待处理、560 行其他宏候选待处理。保留原占位 owner 的实体仍可供候选对账，但必须通过其命名空间作用域链读取 `macro_dependent` 状态，不能显示为最终的 `std`/`cuda::std` 精确目标。

因此，**P1-D04 的“静默假 owner”子问题关闭；“两个条件 owner 及其子实体变体实际展开完成”没有关闭，仍是阶段 1 待办。** 增加 pending 诊断是恢复诚实的状态表达，不是 API 覆盖完成。

## 新发现：成功宏展开仍有三类可复现错误

以下只审查 `macro_expansion.py` 已输出 expansion，且 `Extractor` 已标记 `expanded_and_parsed` 的路径。不要求当前模块变成完整 C++ 预处理器，也没有把已经返回 pending 的能力重新列成缺陷。

所有下列输入都在本机用 `/usr/bin/clang++ -E -P -x c++ -` 独立预处理，返回码均为 0。对有扩展输出的失败样本，当前 Extractor 均为 **0 诊断**并生成相应错误实体/初始化值。它们不属于“已诚实记录的未支持能力”。

### P1-M01：宏参数预展开及 token paste 后重扫被遗漏，却宣称成功

反例一：

```cpp
#define NAME Actual
#define MAKE(X) struct X {};
MAKE(NAME)
```

Clang 输出 `struct Actual {};`；本模块输出 `struct NAME {};`，Extractor 生成名为 `NAME` 的 struct，并标记 `expanded_and_parsed`。

反例二：

```cpp
#define A_type Actual
#define MAKE(X) struct X ## _type {};
MAKE(A)
```

Clang 输出 `struct Actual {};`；本模块只得到 `struct A_type {};`，同样成功发布错误名字。

这两个对象宏都在当前文件内，不涉及外部 include、未知编译选项或无法取得的定义。当前 `nested_names` 检查只寻找已知声明宏的函数式调用，漏掉影响名字的对象式宏；`##` 拼接后的 token 也没有重扫。

关闭条件：正确完成这些已知宏的预展开/重扫，或在发现会影响结果的残留宏时明确 pending 并停止发布错误成功实体。测试接受两种诚实结果，不要求立刻实现全部宏递归能力。

### P1-M02：续行没有统一应用于注释和 undef，凭空生成 API

续行注释反例：

```cpp
#define MAKE(X) struct X {};
// disabled \
MAKE(Fake)
```

Clang 预处理没有输出声明，因为物理第三行仍属于续行后的 `//` 注释。本模块却输出 `struct Fake {};`，Extractor 接着生成 `Fake`，没有诊断。

续行指令反例：

```cpp
#define MAKE(X) struct X {};
#un\
def MAKE
MAKE(Fake)
```

Clang 输出未展开的 `MAKE(Fake)`，说明 `#undef` 已生效；本模块仍使用旧定义生成 `struct Fake {};`，没有诊断。

这里的错误不是“不支持完整条件预处理”，而是翻译阶段的反斜杠换行拼接只应用于部分宏体，没有统一应用于注释判定和指令生命周期。独立候选扫描器已经有此类词法拼接保护，但宏模块没有等价约束。

关闭条件：在保留物理源码映射的前提下统一处理逻辑行，或显式拒绝目前无法可靠解释的拼接区域；注释中的调用不得生成实体，失效定义不得继续成功展开。

### P1-M03：字符串化没有把连续空白序列折叠为一个空格

```cpp
#define MAKE(X) struct Y { const char* s = #X; };
MAKE(a /**/ b)
```

Clang 输出成员初始化值 `"a b"`；本模块输出 `"a   b"`。Extractor 保存的 `Y::s.initializer` 也是 `"a   b"`，但整次展开仍为 `expanded_and_parsed` 且零诊断。

目前逻辑分别把参数的原空白 token、注释 token、后续空白 token 都替换为一个空格，最终累积为三个空格。它应把注释替换后的整个连续空白序列归一，同时保留参数中字符串字面量内部的空白；不能对最后的字面量做全局空格压缩。

关闭条件：输出与独立预处理结果一致，或明确拒绝该未正确实现的字符串化路径，不能成功记录错误初始值。

### 控制案例与新增门禁

没有把以下已正确处理的行为列为新缺陷：普通 `#undef MAKE` 后的调用没有 expansion，并明确返回 `macro_definition_lifetime_unresolved`；替换体字符串字面量中的 `" ## "` 与 Clang 一致，没有被错误当成 token paste 操作。

本轮只新增独立测试 `tests/test_macro_review_oracle.py`，核心代码未改。该文件包含 7 项 Clang CPP oracle 测试，实际结果为 **2 项通过、5 项失败**：M01 两项、M02 两项、M03 一项。之前的 47 项测试不包含这些反例，不能用“原有 47 项通过”覆盖新增失败。

复现入口：

```bash
.venv/bin/python -m unittest discover \
  -s tests -p test_macro_review_oracle.py -v
```

其中 M01/M03 测试允许“没有错误 expansion 且给出明确 diagnostics”的保守实现通过，意图是拒绝假成功，不是强制扩张支持范围。M02 明确要求不生成被注释或已失效宏的实体。

本轮没有在固定 824 文件中证明上述五个缩减样本的具体调用形态当前实际出现；不能把它们擅自换算成固定数据的错误实体数。它们证明当前生成器的成功状态不足以保证展开语义，与需要全库候选对账的待办共同维持阶段 1 未通过结论。

## 第三轮独立复审：M01–M03 的错误成功状态关闭，未展开能力仍待办

复审日期仍为 2026-09-08。本节保留上一节的实际失败结果，仅追加修复后的状态。提取器仍为 `489fcfc5...`；宏模块已锁定为：

```text
9e3e0f2b3debda4ba2cb1f59c14aa0ca3b45f1f29064035ae909bd63af6e9237
```

本轮实际重新执行了独立测试文件和全项目测试，得到新增 7/7、全项目 54/54 通过。另逐个调用 `expand_file` 与 `Extractor`，核查测试是否仅靠隐藏输出通过；同次执行前后宏模块散列相同。

- **M01 错误成功状态关闭，预展开/重扫支持没有完成。** `MAKE(NAME)` 当前无 expansion、无生成实体，明确诊断 `macro_argument_prescan_pending`；`MAKE(A)` 的拼接结果需要重扫时同样不生成实体，并诊断 `macro_result_rescan_pending`。没有再发布错误的 `NAME` 或 `A_type` struct。这属于将结果从假成功改为可追踪待处理，不代表正确的 `Actual` 实体已经生成，相关候选仍不能记为最终展开覆盖完成。
- **M02 关闭。** 续行注释样本没有 expansion、没有生成实体，且不需要诊断，因为它确实不是代码。续行 `#undef` 样本没有 expansion、没有生成实体，并明确诊断 `macro_definition_lifetime_unresolved`。不会再从注释中或失效定义产生 `Fake`。
- **M03 关闭。** 字符串化样本现在生成 `struct Y { const char* s = "a b"; };`，Extractor 中 `Y::s.initializer` 也是 `"a b"`，与 Clang 一致；该样本可保持 `expanded_and_parsed`。

两个原控制案例仍通过：普通 `#undef` 的 pending 行为和字符串字面量内的 `" ## "` 均保持原样。再次核对固定源 `integral_constant.hpp:223–248`，仍有 23 个宏生成 operator、23 个实体，5 个一元、18 个二元参数，没有因这轮宏保守处理丢失该正向样本。

本轮关闭的是本文给出的具体错误结果和假成功状态；D04 的条件 namespace-owner 实体展开、M01 的实际预展开/重扫以及全库候选逐项映射仍未完成。全项目 54 个测试通过不代表阶段 1 已完整，也不证明尚未重跑的数据文件已经同步最新宏模块；全量生成结果及其 provenance 应由制作方重跑后另外核对。
