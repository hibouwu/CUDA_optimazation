# using 形式、名称导入与目标身份：独立源码审查

本稿只读固定提交 `8f50b052e1099fb982392a622caab69b97b63128` 的 824 文件 snapshot，并使用独立词法扫描、代表源码回看和 Host Clang oracle；没有读取其他制作者的总结作证，没有修改 core、全局声明数据或站点。目的不是实现通用 C++ 名称查找，而是确定当前快照的真实语法类别及不能伪造的身份边界。

## 1. 物理源码库存，不是 API 分母

按 scope.json 逐个读取并核对 824 个 SHA256；词法器先处理续行，忽略注释与字符串内容，再从真正的 `using` token 保留到分号的物理区间。独立测试只复用 scanner 的词法器，不使用声明提取器输出来定义库存。

| 物理写法 | 出现数 | 计数的确切含义 |
|---|---:|---|
| `using name [attributes] = type;` | 59,136 | alias declaration 的源码出现；包括模板 alias、局部 alias，不等于新类型或独立库 API 数 |
| `using qualified-name;` / `using typename ...;` | 499 | 单名导入候选，包括 namespace、class 和函数内写法；目标可能为多个重载 |
| `using namespace ...;` | 184 | using-directive 出现，不是 184 个新 namespace 或类型 |
| `using enum ...;` | 0 | 固定源码未出现；不能据此宣称分析器支持该形式 |

总计 59,819 个 written using；其中 683 个非 alias 出现分布在 162 文件。所有分支均保留，不先执行预处理选择。扫描中没有 using token 位于预处理 directive / 宏定义正文；但 using 的目标限定符本身可以含宏。

184 个 directive 的原始目标为 `cute` 172 次、`detail` 11 次、`SM90` 1 次。499 个 import 中有 6 个 `typename` 前缀，全部在 `include/cutlass/gemm/kernel/gemm_grouped_per_group_scale.h:73–78`。按限定目标末名重复得到 83 个继承构造器**拼写候选**；这只是有界分组，不能用末名重复代替对基类、alias 和模板参数的绑定。

特别保留一个会破坏“using 后第二 token 必须是 =”的实际反例：`include/cutlass/functional.h:989–990` 是 alias template，原文为 `using red [[deprecated("use atomic_add instead")]] = atomic_add<T>;`。它不能落入普通 import。反过来，合理负例 `using Base::operator=;` 中虽有 `=` token，也不是 alias；本库没有这项实际写法，测试仅防止库存规则误推广。注释中的 `using ::isnan;` / `using std::isnan;`（helper_macros.hpp:129/131）均未计入。

`namespace TMEM = cute::TMEM;` 是另一种语法，不在上述 using 数内。固定范围的这一实际 namespace alias 在 `include/cute/atom/mma_traits_sm100.hpp:425–428` 的 `cute::UMMA` 中，local namespace alias 是 `cute::UMMA::TMEM`，目标是 `cute::TMEM`；不可与 directive 混记。

原始 Tree-sitter 仅用作有界交叉探针，不作真值：例如 sm90_visitor_tma_warpspecialized.hpp:702 的 directive 被原始恢复树塞进 class；sm90_tile_scheduler.hpp:45 的继承构造器被归在错误的 function 祖先下。回看原文分别是 detail namespace 关闭后（700）的导入，以及 40–45 行 class 的 public 成员。本稿因此不把 raw AST 的 namespace/class/function 自动数量提升成完整语义分类证明。

## 2. 当前快照里应分别建模的对象

### U01：普通 namespace 导入，local lookup 名称与 target 分开

`include/cute/util/type_traits.hpp:58–94` 位于 namespace cute。第 92 行 `using CUTE_STL_NAMESPACE::remove_cv_t;` 使 `cute::remove_cv_t` 成为可查找名称；原始 target 则是 `CUTE_STL_NAMESPACE::remove_cv_t`。不能构造 `cute::CUTE_STL_NAMESPACE::remove_cv_t`。

该文件有 64 个 CUTE_STL_NAMESPACE import；另有 int.hpp 的 8 个、integer_sequence.hpp 的 2 个、array.hpp 函数内 swap 的 1 个，共 75 个。platform/platform.h 有 40 个 CUTLASS_STL_NAMESPACE import。它们不全是类型：type_traits 的 `_v` 项、platform:954–958 的枚举值与函数模板 `declval`（540）也使用相同 using 语法，不能一律记录 `target_type` 并宣称目标已确认为类型。

条件来自真实宏定义：cute/config.hpp:106–111 的 RTC / 非 RTC 分支选择 cuda::std / std；platform/platform.h:151–157 另有 `!defined(CUTLASS_STL_NAMESPACE)` 外层 guard。必须保留定义来源、条件和外部 override 边界。目标在标准库时属于外部定义，不是 824 范围中的 missing_definition。

不含宏也会发生同类错误：`cute/numeric/complex.hpp:40–45` 在 cute 中导入 `cutlass::complex/is_complex/RealType/real/imag/conj`，local lookup 是 `cute::complex` 等，而不是 `cute::cutlass::complex`。其中既有类型模板，也有函数重载。`cutlass/gemm/gemm.h:59–75` 同时导入 `cutlass::detail::TagToStrideA`（detail/layout.hpp:54 的类型模板及其特化）和 `TagToStrideA_t`（185 的 alias template），不能因它们都来自 using 就把 target 声明类别抹平。

`cute/atom/copy_traits_sm100.hpp` 有 162 条 `SM100::TMEM` 目标导入。第 464 行导入 `SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp256b1x`，紧随的 Copy_Traits 特化 467 使用短名。目标原定义在 `cute/arch/copy_sm100.hpp:618`；这是导入路径加原实体，不是再定义一个同布局的硬件 copy 类型。

### U02：using-directive 是有词法位置的名称查找关系

P90 的 `cutlass/pipeline/sm90_pipeline.hpp:48–52` 在 cutlass 中 `using namespace cute;`，并没有声明一个名为 `cutlass::cute` 的库实体。`cute/atom/mma_traits_sm90_gmma.hpp:467–469` 在关闭 SM90::GMMA 后导入 SM90；`cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp:700–702` 在关闭 detail 后导入 detail。目标由所在 scope 查找，不是所有 `detail` 都是全库同一 namespace。

directive 应记录独立物理 occurrence、词法 owner、被提名的 namespace、有效条件及声明点。它不是把目标 namespace 的全部成员复制到当前 owner 的一次快照；后续声明、遮蔽和块范围仍然影响 lookup。不能用一个伪实体或自动新建若干本地副本替代该关系。

### U03：class 成员导入不创建新存储或新方法实现

`sm90_visitor_tma_warpspecialized.hpp:587–602` 的 Sm90VisitorImpl 继承 Sm90VisitorImplBase，589 定义 alias Impl，602 `using Impl::ops;` 引入基类对象成员；原始存储是 583 的 `tuple<Ops...> ops`。732 的 `using CallbacksImpl::callbacks_tuple;` 也位于继承 CallbacksImpl 的 ConsumerStoreCallbacks 中（726–730）。它们不是局部 alias type，也不能凭导入声明创建第二份字段资源。

`cutlass/gemm/threadblock/mma_multistage_blockwise.h:163–168` 的六个 Base 方法导入，以及 `conv/threadblock/threadblock_swizzle.h:122/163` 的 Base::get_tiled_shape，要保留基类来源和可能的重载集合，而非制造一个新函数定义位置。

`gemm_grouped_per_group_scale.h:66–78` 的六项 `using typename Base::...` 明确是 dependent type import。`typename` 是消歧义关键字，不是名称文本的一部分；可查找名字是 Mma、Epilogue、EpilogueOutputOp、ThreadblockSwizzle、Params、SharedStorage。Base 在 70 行绑定到真正的基类模板，不是一个新 namespace。没有模板实例绑定时保留这条 dependent target，不虚构已解析定义。

### U04：继承构造器不是普通导入的末名拼接

真实代表包括 `cute/pointer.hpp:87–88/150–151/222–223`、pointer_swizzle.hpp:76 的 iter_adaptor 继承构造器；`gemm/kernel/gemm_universal_with_visitor.h:60–61` 的 Base::Base；`gemm/kernel/sm90_tile_scheduler.hpp:40–45` 的 injected base template 名称；P90:160–165 的 ArrivalToken::ArrivalToken。83 个拼写候选主要分布在各类 FusionCallbacks 与 Visitor 适配器，也包括多行、带完整模板实参的 target（例如 epilogue/collective/sm100_epilogue_array_nosmem.hpp:966 起）。

应保留 `inherited_constructor_import` 或等价类别、derived owner、base 名称及其 alias/template 绑定、原始完整 target、实际可见性和声明位置。具体构造器候选及可调用性另从基类与语言规则建立，不能把 using 自身假装成一条拥有任意签名的构造函数定义。不要把 copy/move shadow 的存在自动解释成全都按相同规则继承可调用。

尤其 P90 第 161/165 行在 class 默认 private 区域，但基类 ArrivalToken(BarrierStatus) 在 public 区域（119–121），默认构造器在 124 删除。独立 Clang 证明 private using **不阻止**从外部调用继承的 public 带参构造器，default 仍被删除；反过来 public using 不会将 protected 基类构造器开放。必须分开 using 的词法 access 与最终构造器可访问性。

### U05：函数内 using 为 lookup / ADL 服务，不升格 namespace API，也不能丢掉其关系作用

`cute/arch/mma.hpp:52–60` 的函数体先 `using cute::fma;` 再未限定调用 fma；`cute/container/array.hpp:186–190` 的 swap 类似；tuple.hpp:643 的 print、cutlass/complex.h:467 的 sqrt 也是局部导入。它们不是再次定义 free function，更不能因不输出独立 API 实体，就认为后续调用目标不需要解释。

`cutlass/functional.h:411/454/480/625/662` 使用 `CUTLASS_CMATH_NAMESPACE :: isnan`。helper_macros.hpp:136–140 在 __CUDA_ARCH__ 下把宏定义为空，所以 target 是全局 `::isnan`；另一分支为 `std::isnan`。不能假定宏展开结果总是一个非空 namespace 名称。`relatively_equal.h:61/63、228/230、252/254` 的 cuda::std / std 条件导入同样要保持分支。

## 3. 当前提取器的可复现错误，不因 parsed 标签而关闭

本节以当前源码直接调用 Extractor 单文件取得结果，没有将巨大 declarations.json 全量装入内存。检查指纹：`scripts/extract_declarations.py` SHA256 **`da8d2007310e15cbb5abe6924e402e2d71789887419da94aab34d608395abff1`**。

| 原始声明位置 | 当前错误的 qualified_name | 当前状态 / 正确边界 |
|---|---|---|
| cute/util/type_traits.hpp:92 | `cute::CUTE_STL_NAMESPACE::remove_cv_t` | parsed；local lookup 与条件 target 必须分开 |
| cute/numeric/complex.hpp:40/45 | `cute::cutlass::complex` / `cute::cutlass::conj` | parsed；类型模板及函数重导出均受影响 |
| cutlass/fast_math.h:58 | `cutlass::::cuda::std::swap` | parsed；原文 `using ::cuda::std::swap;` 的根限定不能接到当前 owner 后 |
| cutlass/pipeline/sm90_pipeline.hpp:52 | `cutlass::cute` | parsed；实为 namespace directive，无此新实体 |
| cute/pointer.hpp:88 | `cute::gmem_ptr::iter_adaptor<P, gmem_ptr<P>>::iter_adaptor` | parsed；继承构造器导入不能普通 QName 化 |
| cutlass/pipeline/sm90_pipeline.hpp:161/165 | `cutlass::ProducerToken::ArrivalToken::ArrivalToken` / ConsumerToken 同型 | parsed；derived/base/source spelling 应分别保存 |
| sm90_visitor_tma_warpspecialized.hpp:602 | `cutlass::epilogue::fusion::detail::Sm90VisitorImpl::Impl::ops` | parsed；目标是基类成员，不是这个复合路径下新字段 |

CUTLASS_NAMESPACE 自定义 namespace 变体还保留同型错误，并非默认 cutlass 路径的一个显示问题。GemmGroupedPerGroupScale 的六个 typename import 当前则是 `contains_parse_error`，name 里还包含 `typename Base::...`；这是已经显式报告的解析缺口，与上表 parsed 的静默身份错误分别记账。

array.hpp:188 的函数局部 using 不在 API occurrence 输出中，本身不足以判定漏 API；后续 calls / lookup 记录仍需保存其物理出处与作用域。这是阶段 2 义务，不能靠新造 namespace 实体解决。

## 4. 独立 Clang oracle 与可组合身份合同

使用实际 Clang 21.1.8，stdin、自包含 C++17 / C++20 缩减源码，`-fsyntax-only -Xclang -ast-dump=json`。没有 GPU 执行，没有将单一 Host 配置说成 RTC 全分支编译通过。持久化的 7 项测试采用 C++17；额外独立探针也实际运行了对应 C++20 输入。

1. `namespace src { void f(int); void f(double); struct T {}; } namespace dst { using src::f; using src::T; using X=src::T; }`：Clang 给出一个 src::f UsingDecl 和两个不同 target id 的 UsingShadowDecl，分别为 void(int)/void(double)；T 的 import target 是 CXXRecordDecl，X 则是 TypeAliasDecl。两个 `__is_same` 与两个函数指针等值断言通过。一个 source import 不能限制为一个 overload target；alias 声明也不表示产生新底层类型。
2. `namespace owner { using TARGET::Thing; }`：TARGET 分别定义为 impl_a / impl_b 时，owner::Thing 与对应 target 类型相同；`owner::TARGET::Thing` 在两分支都因 owner 没有该 namespace 成员被拒绝。它直接否定“owner + targetQName”模型。
3. directive 的 AST 为 UsingDirectiveDecl，具有 nominatedNamespace，没有 name 字段，也没有成员复制的 UsingShadowDecl。`dst` 导入 src 后先使用 x，再声明自己的 x，前后 DeclRefExpr 分别指向不同目标；函数块内 directive 不泄漏到另一个函数。
4. `using Base::member` 的 `decltype(&Derived::member)` 仍为 `int Base::*`，不是 `int Derived::*`。dependent typename import 在模板定义处为 UnresolvedUsingTypenameDecl，实例化后才有已绑定 target。
5. Clang 21 对 `using Base0::Base0;` 的 UsingDecl.name 甚至显示 `Base0::Derived0`，ConstructorUsingShadowDecl.target.name 才是 Base0；模板 primary 的字段又不同。故编译器 AST name 也不能直接当 source spelling 或标准化 localQName。P90 private using / public base constructor 的正例、deleted default 与 protected base constructor 的两个负例均实际成立。

后续模型至少需要分开四层：**物理导入 occurrence；词法 owner 和有效条件；local lookup 名称或无名 directive / constructor 类别；原始 target 表达式及零到多个已证明 target 实体。** Alias declaration 有自己的声明名，引用底层类型但不新建底层类型；ordinary using 的 import 记录也不等于其 target 实体。函数内记录再绑定词法范围，继承构造器另受基类绑定和可调用性规则约束。

这里的 dependent target 可以在保存 Base<T>、模板参数和来源后保持未实例化；而上节已明确错误的伪 QName 应标识为 extraction_error，不能改名 symbolic 或仅靠 parsed 白名单放行。仅把末名截出来可缓解普通导入显示，但不能解决 overload、directive、继承构造器及跨配置身份这些问题。

## 5. 冻结结果与后续边界

新增 `tests/test_using_declaration_independent_review.py`，7 项实际通过，最后一次运行耗时约 6.3 秒。测试 SHA256：`6ff863482913c0866b52b42b9ef01ee997626aa37fd53bd96d724a3604adafa7`。三个测试检查固定物理库存和分类负例，四个测试使用独立 Clang 正反语义 oracle；不导入或修改核心 extractor。

本稿固定了错误与有限语法合同，未宣称 499 个 import 的所有 overload / 模板实例 / 条件目标均已解析，也未把 83 个拼写候选一概当成完整可调用构造器集合。没有写回全库 mapping 或修复全局身份；应由后续有界实现、全量对账和模块端点复核关闭对应缺口。
