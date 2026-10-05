# array 六张高风险详细图：独立静态展示复核

本轮只做静态 SVG 检查：使用本机 `/usr/bin/rsvg-convert --background-color white` 转成原始尺寸 PNG，再以 `view_image(detail="original")` 逐张目视。没有浏览器、没有 HTTP 或 HTML 预览，没有模拟点击，也没有将像素宽度或 renderer 名称当成可用性证明。

原始源码依据为固定 snapshot 的 array.hpp 及 type_traits.hpp；图数据为 `data/modules/cute_array/relations.json`，SHA256 `f201dc4a101f1d129cd12f5b65cb271703879790bc5ef6c1e163d79108d6f5bd`。本轮没有重新写模块数据、站点或全库 ledger。

## 1. 目视检查的六个确切文件

所有 SVG 路径相对 `site/diagrams/`。PNG 保存在 `/tmp/cute-array-static-review-Rlma2Y/`，采用相同文件 stem。

| SVG | 原始尺寸 | SHA256 |
|---|---:|---|
| edge-array.edge.0176.specializes_cb4a0f5f.svg | 1021 × 535 | `e67aaf7f4c88cbc60ad04b769d0d9f320eaf7e9b56b2d574089d32dcbc9be22a` |
| edge-array.edge.0211.calls_b5ef317b.svg | 716 × 571 | `027f3faafe87336e8c5c2da8b7a79211c53cb75d8758a6c13c9608725499685c` |
| edge-array.edge.0220.calls_6f814ab9.svg | 733 × 615 | `ede9326f14784852df1a387d8bf304c1d763a833eb4404bd8016f7ff5421c57b` |
| edge-array.edge.0237.calls_14e91587.svg | 820 × 634 | `2103319f65f2442e1fd1b08de53883760882e7e8eaa587b002cea09d53643501` |
| edge-array.edge.0243.implicit_calls_ae312246.svg | 727 × 651 | `1b8bd2c8e4fc1cc9f8080e4b4a78fbdbe1ba6863ad57fbb936f8c31df553f33a` |
| edge-array.edge.0244.implicit_calls_3c2e84af.svg | 727 × 651 | `2c54507d2d70f9d26c5a0499a90bd8792b9b130f87c5fbd2dc51fe20201629b2` |

上述哈希在转图前和六张目视结束后分别读取，保持一致。每张图都只有本关系的两个端点和一条关系箭头；签名 note 与其对象的浅色连接线没有遮住关系箭头或代码文字。

## 2. 各图的源码与可读性判断

### V01：std 桥接特化 → 本地条件 primary，0176

图的 source 显示 `array.hpp:464`，是该关系真正的第二处 std 桥接模板声明，不是首次的 433。source note 完整显示 `template<class T,size_t N>`、`tuple_size<cute::array<T,N>>` 和 `CUTE_STL_NAMESPACE::integral_constant<size_t,N>` 基类；target 显示 template 开始位置 456，note 保留原文 `template<class... _Tp> struct tuple_size`，对应 457 的 struct 前置声明。不能按常见标准库模板形状改写原文参数。

465 的 source_expression 和完整旧 RTC 条件均可读。长条件跨两行，未切掉宏名、比较运算符或末尾 RTC 条件。虚线箭头从桥接特化对象指向 primary 对象，没有画成运行调用，也没有连到 cuda::std 对象。两端同名 std::tuple_size 依靠完整声明 note、不同位置及方向区分。

原始 1021 × 535 下没有发现文字重叠、签名截断或右侧裁字。本图较其余五张宽，原因是条件文本较长；此尺寸观察不等于实际网页视口验收。

### V02：free clear → 条件 fill 绑定，0211

source 显示 template 开始位置 351；完整签名是 `void clear(array<T,N>& a)`，不是 array 的 member clear。关系明确显示原文第 355 行的 `a.fill(T(0))`，因此没有把它改画成 `a.clear()`，也没有吞掉 T(0) 的构造要求。

target 以 binding 标识 `array<T,N>::fill`，位置 355 是依赖表达式的来源，不冒充实际 member API 定义位置。箭头和说明明确表示参数化 / 条件候选、非已确定实例目标；图内没有把 N=0 / 正 N 的 fill 合并成一个已解析 API 身份。此图的 binding note 只是入口说明，详细候选、实参关系和缺少的模板绑定仍在关联记录里；本轮没有借这张图的尺寸宣告依赖实例化已经完成。

716 × 571 下完整函数签名、T(0)、callsite、binding 标识和图例都可读，没有文字互盖或边界截断。

### V03：get&& → cute::move，0220

source 显示 array.hpp:420 的模板起点，完整保留 `T&& get(array<T,N>&& a)`，没有变成 const& 或按值返回。425 行 `cute::move(a[I])` 独立显示；target 是 type_traits.hpp:194 的 `cute::move`，不是外部 std::move。

target note 完整保留 `template<class T>`、`remove_reference_t<T>&& move(T&& t) noexcept` 及 CUTE_HOST_DEVICE constexpr。箭头实际落在 move 对象，source 的代码 note 不遮挡箭头。733 × 615 下所有模板参数、`&&` 与 noexcept 均清楚，没有截断。

### V04：同一个 get&& → 非 const operator[]，0237

这是一张独立的第二调用图，source 仍为同一个 get&&，但关系表达式只是 425 行的 `a[I]`。target 是 primary array 的非 const `reference operator[](size_type pos)`，声明起点 55、函数名行 56；不是 61–62 的 const overload，也不是零特化。

条件文字明确 I<N 在 get body 实例化时检查，合法 body 因而 N>0；没有把函数体 static_assert 宣称成签名 SFINAE。source 命名的 a 在表达式中是左值，先得到 reference，再由另一图中的 cute::move 改变值类别；两条调用没有合并。

820 × 634 下 source/target 的不同返回形式、const 缺省、public access、关系条件及箭头端点均可辨，没有文字重叠或签名裁切。

### V05 / V06：range-for 的两个隐式调用，0243 / 0244

两图 source 都是 primary member fill，声明前缀起点 171、函数名行 172。关系的物理来源保留 174 行真实 `for (auto& e : *this)`，并显著标明 `language_desugaring`、`implicit-call source origin` 和“非源码字面调用”。合成的 `__range.begin()` / `__range.end()` 不伪装为原文件实际出现的字面函数调用。

0243 的 target 是非 const `iterator begin()`，起点 105、函数名 106；0244 的 target 是非 const `iterator end()`，起点 129、函数名 130。两条蓝色虚线箭头分别落在 begin / end 对象，未互换 target，未合成一个 begin/end API。独立边 ID、完整签名及合成表达式在 727 × 651 的原始图中均可辨。

两图未发现文字重叠、边标签压住签名或箭头路由到错误对象。图例仍保留“operation 为独立 API 操作节点，不是新增 C++ 类”的区分。

## 3. 来源选择 helper 的反例与定点关闭

初始 helper 指纹为 `7c065d33f3669737073ec8024e6747bbb4a1046bf98ed00c5123f2d2b77996ed`。独立遍历当前 array 的实际多位置端点得到 66 个需选 occurrence 的端点，其中 62 个成功选择、4 个回退。后位置 `array.edge.0084.member_of` 和 `0085.member_of` 真实显示错误：分别回退到 433 / 438，而非 464 / 469。

精确原因是原始 member_of 证据包括结尾分号，canonical AST declaration_range 止于右花括号。例如 0084 的证据 `[8914,9037)`，canonical 第二声明范围 `[8914,9036)`；严格包含失败后返回了 merged node 首条声明。此前的 specializes 465/470 已能正确选择，不能用这两个成功例掩盖 member_of 反例。

修复后重新读取 helper，指纹 **`dfe607e7e4d698d61cccd9a1bbecad5d11fff05124c902cf4d568dccb3fe7cbd`**。它只允许尾部真实源码 `suffix.strip()==b';'` 的精确差异，不改成任意相交。独立重新运行实际 66 端点：**66 / 66 全部唯一选择**；0084 选 464，0085 选 469。把各自 evidence.end_byte 故意扩大到原文件 EOF 的两项负例均保持不选择，没有扩大到不相关声明。

此处关闭的是 helper 的来源选择反例。Root 正刷新 0084 / 0085 等受影响 SVG；本轮列出的六张图不受该分号补丁影响，且本审查没有目视尚未交付的新 0084 / 0085 图，因此不把 helper 通过写成那些 SVG 已目视通过。

## 4. 静态链接检查与结论边界

对六份实际 SVG 的 XML 只读检查，共 12 个对象链接都指向 `../index.html?api=<确切 node_id>`，每个 node_id 存在于当前模块数据，且与各图两端一致。没有通过启动页面、点击链接或运行 JavaScript 来测试这些链接；这只是 SVG 中链接目标的结构检查。

结论：**上述六张给定 SHA 的逐关系 SVG 静态目视检查通过**，没有发现需要修复的文字重叠、签名裁切或错端点路由。该结论不覆盖全部 798 张 array 图，不等于真实 1366/1920 浏览器视口、缩放、点击导航或页面使用任务通过，也不关闭 using 身份缺口或全库最终完成条件。
