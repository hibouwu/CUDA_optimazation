# array canonical 接入：独立破坏审查

本审查只涉及固定 `include/cute/container/array.hpp` 及其接入所需的 config.hpp / type_traits.hpp。测试直接从这三个固定原文件构造 labelled in-memory canonical fixture，运行真实 `stage_cute_array.main()`、真实 source checker 和 enrich_nodes；只替换全局 ledger 的迭代输入，捕获两次 dump，不写模块产物或全局数据。它不是一次读取或重建 824 文件 canonical ledger。

源码语义依据来自此前独立完整文件审查，见 `cute-array-independent-source-review.md`；本轮才交叉读取作者 draft，不能倒用 draft 总结证明原文。using 的身份错误仍保持独立 open，不在本轮修复范围。没有浏览器 / HTML 使用验收。

## 1. 初始检查点：正例被拒绝，负例不能假通过

初始指纹：

- stage_cute_array.py：`0809a36ef1082c1f4a7a522c542c5e3060d4f54ff5fa7f253a05df8f2caaabb3`。
- build_atlas.py：`5bac4f17e0a17c08ffd904db83234d550b1ab1cd8221f75add1e1694c0aaa7d8`。
- draft relations.json：`3cf7b7676690571e1b2b0ba48f842198375ae3131d997c3501d440a1631084e1`。
- draft check_source.py：`279cf8cf5cf1e5f0864f689088f9a6bdd8e576d121ced0b919ce55dc28fd0066`。

### J01：合法返回 cv 与成员限定符混合，初始 stage 正例不通过

直接对未篡改 fixture 运行 stage，在 `array.api.fn_100:qualifiers` 抛 ValueError。原文 99–100 为 `CUTE_HOST_DEVICE constexpr T const* data() const`：返回类型有一个 const，成员函数有一个 const。作者记录的成员限定符为 constexpr / const；当前 canonical 的 qualifiers 是 `['const','constexpr','const']`，把返回限定混了进去。不能以增加一项错误的成员 const 去修改作者源码合同。

同一声明 canonical return_type 是 `const T *`，作者原文拼写是 `T const*`，仅比较 token 原顺序也会误报。两个 spelling 在这里表示相同的返回类型；但比较时仍必须区分 `T const*`、`T* const` 以及成员函数后的 const，不能随意将全部 const 排序或集合化。

代码检查还发现：std tuple_size 合并节点保留 433 与 464 两处 template_parameters，共四个物理形参出现；逐 bound occurrence 的 canonical 环境只有对应位置的两个形参。不能以节点级聚合列表直接对比每个 occurrence 的列表而误报，也不能为通过而删除第二位置模板证据。此项在初始正例因 fn100 更早阻断，尚未到达完整 stage 对应分支。

`tests/test_array_stage_review.py` 因此在 setUpClass 先要求未篡改 stage 正例成立。初始冻结脚本上实际结果为 **0 tests executed / 1 setup error**，而非“所有 mutation 都被拒绝所以通过”；这样避免无关早期错误给负例制造假阳性。

### J02：隐式 begin / end 的 target 互换仍被 source checker 接受

只在内存交换两条 implicit_calls 的 target，保留 target 集合、所有源码区间和其他元数据：

- `__range.begin()` 错接 `array.api.fn_130`（end）；
- `__range.end()` 错接 `array.api.fn_106`（begin）。

初始 check_source.verify 实际接受，仍返回 `implicit_range_for_calls=2`。原因是只验证两个目标组成的集合，没有逐个 synthetic expression 核对。原文 172–174 的 primary array::fill 范围 for 使用非 const `*this`，正确配对是 begin→106、end→130，两个独立调用不能通过交换目标而仍算正确。

此时完整 stage 仍因 J01 先失败，因此本条先严格记为 **source checker 的独立接受反例**，不冒称已跑通完整错误 stage 输出。测试已加入端到端拒绝反例，待正例恢复后重新运行。

## 2. 新增 specializes 的独立源码回切

重新回看原文 446–473：外层条件是 `defined(CUTE_STL_NAMESPACE_IS_CUDA_STD)`；450 的 compiler major>=13 分支 include 外部 structured_bindings header；454 的 else 中，455 再在 RTC 条件下写两个本文件 primary 前置声明（456–457 的 tuple_size、459–460 的 tuple_element）。随后 464–473 是本文件 std 桥接特化。

已核对以下两组实际边：

- 465 的 `tuple_size<cute::array<T,N>>` → `array.type.decl_457`，目标完整 QName 为 std::tuple_size，signature 保留 `template<class... _Tp>`，definition=false。
- 470 的 `tuple_element<I, cute::array<T,N>>` → `array.type.decl_460`，目标为 std::tuple_element，signature 保留 `template<size_t _Ip,class... _Tp>`，definition=false。

每组还保留一个显式 external primary 分支，未借外部调用点冒充库内声明。取 A=外层 marker、B=compiler major>=13、R=RTC，则 local 条件是 `A && !B && R`，external 条件为 `A && (B || !R)`。枚举三个布尔量的全部组合证明两者互斥，且并集正好为 A。两组 evidence 的原始字节、465/470 行、source_expression 与 snapshot 一致。该结论是**这两个已写特化位置的条件来源分类正确**，不等于已编译任意外部宏环境或声明外部 header 内容均已提取。

## 3. 已准备的有界拒绝矩阵与待复核项

独立测试包含 QName / template-specialization 错配、const 重载借用非 const signature、形参 type/name/index/default/raw/span 改坏、模板形参删除、return/qualifiers/attributes/body-range 改坏、条件表达式/来源文件/来源行/缺失分支、遗漏/重复/替换 occurrence、同实体两位置拆开、std/cuda::std 合并、不同 operator[] 重载合并、第一位置条件覆盖 entity availability、any_of 改 all_of、输出位置删除、作者 body 原文篡改，以及 J02 的隐式调用错目标。

两项需特别核对，不能只依赖 canonical 的 entity_id 标签：同一个 std 实体的 433/464 要保持不同 occurrence；std 与 cuda::std，以及 const/非 const operator[]，则必须保持不同实体。当前初始 verify_join 仅比较 enriched node.entity_id 与被选 canonical.entity_id，两者若一同来源于同一个坏 canonical ID，这个比较本身不能证明没有跨节点误合并。已加入实际 array fixture mutation，正例恢复后再判定是否存在完整端到端漏检。

尚未有通过的无修改正例时，本审查不报告完整拒绝矩阵通过，不将源码计数 87/92 当成正确性证明。后续复验应追加关闭记录，保留上述初始反例与版本指纹。

## 4. 第二检查点：原矩阵闭合；拆分类型元数据发现新的漏检

Root 完成修订后，独立重新运行原 13 项审查矩阵：**13 / 13 实际通过**，约 6.6 秒。无修改正例现在先成功，随后所有负例才开始执行，所以此结果不依赖初始 J01 的无关阻断。

本次指纹为 extractor `72386e53a5c8e23c4fc71ef21e4a6d7ebffcee8dc00e705f7143d9c2ead69624`；stage `93abe89f16fedb9bd475802253648b47f0feea8954614bfe517922b96308fcb6`；build_atlas `34aa9b695eab2574d4de60260393f67c8097912bbd6b80108acea2f54e54c7c2`；checker `237b947ac317c69c79ed36f6222cffb9e292f2ede69cf8559e76b33850c9a6e7`；draft 仍为 `3cf7b7676690571e1b2b0ba48f842198375ae3131d997c3501d440a1631084e1`。

- **J01 在源码 fixture 检查点关闭。** 返回 const 与成员 const 已拆开，fn100 / fn255 的 qualifiers 为 const、constexpr，return_cv_qualifiers 为单独 const；合法 return_type 拼写 `T const*` 与 `const T*` 可等价，而 `T* const` 和 `T*` 两个破坏值均被完整 stage 拒绝。433 / 464 聚合节点的模板参数按各物理 occurrence 的词法模板环境独立匹配；正例保留两位置。
- **J02 在源码 fixture 检查点关闭。** 互换 implicit begin/end target 的完整 stage mutation 现在被 checker 逐 synthetic expression 拒绝。
- 原矩阵的跨 namespace entity 合并、const/非 const operator[] entity 合并、参数 index 改 17、条件来源/表达式破坏、重复/遗漏位置、any_of 改 all_of、第一位置 conditions 覆盖存在性等，均在合法 baseline 成功后被拒绝。

上述只复验当前源码得到的内存 fixture。Root 此时尚未重生成新版全库 declarations.json；不据这些结果声称实际全局 canonical 账本已经修复或发布。

### J03：完整数组类型已验证，但保留的 base / declarator 字段仍可自相矛盾

实际原文第 194 行是 `element_type __elems_[N];`。无修改输出正确分开保存：declared_type=`element_type`、declarator=`__elems_[N]`、declared_type_spelling=`element_type [N]`，并有说明 base specifier 语义的 declared_type_role。新字段没有在正常样本中直接覆盖 base 的旧语义。

但只在 canonical fixture 改动下列任一字段，保持 raw_signature、signature_range 与完整 declared_type_spelling 原样，完整 stage 仍成功，并把错误字段写入捕获产物：

- declared_type 改为 `float`；
- declarator 改为 `__elems_[N+1]`；
- declared_type_role 改为 `complete_type_only`。

同一检查点把完整 declared_type_spelling 改为 `element_type[N+1]` 会被拒绝。故问题不是完整数组形状没有参与比对，而是增加完整拼写后，其余保留字段尚未和原始声明作一致性验证。源合同正确、完整拼写正确，不能给其他自相矛盾字段提供通过证明。

### J04：return_cv_qualifiers 可以与已验证的 return_type 矛盾

只将 fn100 的 canonical return_cv_qualifiers 从 `['const']` 改为空列表，保留 return_type=`const T *` 以及正确的成员 qualifiers，完整 stage 仍接受并将空列表写入输出。当前对象选择和完整返回类型未改变，但新暴露的拆分元数据已经错误；应按**类型元数据校验未闭合**记录，不能把返回类型等价检查当作它已被核验的证据。

新增三个有界方法，测试总数现为 16。实际复跑结果是 16 方法运行、4 个失败 subcase，全部对应 J03 的三个保留字段及 J04；原 13 项矩阵仍通过。没有增加其他语法族、GPU 执行或写回产物。待修复后仅需重跑此定点矩阵并追加结果。

## 5. 第三检查点：J03 / J04 定点复验关闭

保持测试文件原样，重新执行 `test_array_stage_review.py` 的原 16 项：**16 / 16 实际通过，8.604 秒**。setUpClass 的无修改完整 stage 仍先通过，之后才执行各项破坏检查；未删除此前失败负例，也未改用标签断言替代 stage 的拒绝。

闭合依据不仅是作者报告或测试总数。本轮重新读取校验实现，确认它对固定 array 源码只等长屏蔽 CUTE_HOST_DEVICE，要求物理 AST 无 error，并用物理声明的 start/end/type 定位实际子节点：

- **J03 关闭。** field_declaration 的 `type` 子节点给出 base=`element_type`，`declarator` 子节点给出 `__elems_[N]`；保留的两个 canonical 字段分别与它们比较，role 现在必须是固定枚举 `declaration_type_specifier`。完整 `declared_type_spelling` 仍独立保留并验证 `[N]`，没有用它替换旧 base 语义。原先可放行的 float base、N+1 declarator、错误 role 三项全部被完整 stage 拒绝。
- **J04 关闭。** 每个真实 function_definition 的直接 type_qualifier 子节点给出 return_cv_qualifiers，独立于函数 declarator 后的成员 cv；fn100 丢掉 return const 的原负例现在被拒绝。合法 `T const*` / `const T*` 等价仍通过，`T* const` / `T*` 仍被拒绝，成员 qualifiers 仍只有一个 const 与 constexpr。
- **J01 / J02 保持关闭。** 原始正例、逐 occurrence 模板环境、begin/end 配对、跨实体错误合并、参数及条件破坏、位置重复/遗漏、any_of 存在性等原矩阵均继续通过。

最终源码 fixture 检查指纹：

- extractor：`2ff1b7be010b090f0ef57d093bfa062d6a113b04049abae5b1e2ca7d74b712c7`。
- stage_cute_array：`d9da88e7f9ca0d2daee86d52cc0c3461eca7a964b74c7155288d3c93edc2cee7`。
- build_atlas：`34aa9b695eab2574d4de60260393f67c8097912bbd6b80108acea2f54e54c7c2`。
- source checker：`237b947ac317c69c79ed36f6222cffb9e292f2ede69cf8559e76b33850c9a6e7`。
- 独立测试：`d37020571fbe482175d7baab9286e7be0163cbc48838b4177c9e3fc8cfacfd5d`，与第二检查点相同。

本检查点的结论为 **J01–J04 在固定 array 的源码内存 fixture 接入矩阵中全部关闭**。审查者未启动、读取或替换 `data/declarations-array-types-next.json` 的全量生成结果；Root 表示该新文件仍在生成、尚未替换，不能将本节等同于实际全库 canonical 发布验收。using 身份问题仍独立 open，SVG 静态展示和真实浏览器使用验收也未由本轮测试覆盖。后续只需针对实际发布的输入指纹与展示结果作下一步核对，不扩大通用语法分析器。
