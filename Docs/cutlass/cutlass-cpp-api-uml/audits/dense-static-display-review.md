# Dense 代表路径：两张逐边图的独立静态展示审查

本审查只读取已生成 SVG、对应 PlantUML 与 `site/assets/atlas-data.js`，再用本地 `rsvg-convert -b white` 渲染 PNG，以原尺寸查看。**没有使用浏览器、HTTP 预览或桌面交互，不是 1366/1920 浏览器可用性验收。** 也没有改动生成器、模块数据或 SVG。

## 固定抽样与结果

| 项目 | Host 调用详细图 | types 模板绑定详细图 |
| --- | --- | --- |
| 边 ID | `host.adapter.init_lower.328` | `types.edge.selected_specialization` |
| SVG | `site/diagrams/edge-host.adapter.init_lower.328_f76e0765.svg` | `site/diagrams/edge-types.edge.selected_specialization_65287851.svg` |
| SHA256 | `1409392a71cb5b0190a245375fb3191c6b1ba4a4d12b101a3a2dc8dd982a0c4b` | `d36fb72df3717050ff72fe3066584bbba31eedf28ef1f34d10881d9ff4990c2e` |
| SVG width×height / viewBox | 1071×667 / `0 0 1071 667` | 1371×1568 / `0 0 1371 1568` |
| 实际文本字号 | 12px、14px | 12px、14px |
| 原尺寸目视 | 字符可辨，无可见裁切、叠字、断开的箭头 | 字符可辨，无可见裁切、叠字；高度明显大于一屏 |
| 关系端点 | initialize → Kernel::to_underlying_arguments，单一实线关系 | Builder绑定实例 → Builder偏特化，单一虚线关系 |

所抽 SVG 文件时间为 19:28，对应 PUML 文件时间为 19:32。这里只将 SHA 作为版本依据：PUML 或模块 JSON 已更新不等于这些 SVG 已重生成。Root 正在进行的导航按 source API 拆图改动，不能由这两份旧逐边 SVG 来证明或否定。

两个 SVG 的节点链接均指向存在的 `../index.html?api=<id>`，所用 API ID在当时的 `atlas-data.js` 中存在。此项仅为路径和结构数据检查，没有实际点击验证页面行为。

## Host：完整签名可见，但调用标签锚错了同一证据段中的另一 API

原尺寸图中 initialize 的 `Status`、四个形参及三个默认值完整保留；目标 `static Params to_underlying_arguments(Arguments const& args, void* workspace)` 也完整保留。节点源码路径及声明位置分别为 adapter312 和 kernel255，和 snapshot 的实际声明区域吻合。长 owner名称被硬换行切开了 `enable_if_t`、`GemmKernel_` 等标识符，没有丢字符，但会议阅读时需要跨行拼接，属于可读性改进项。

确定的问题在边标签：图的边 ID为 `.328`，表达式是 `GemmKernel::to_underlying_arguments(args, workspace)`，底下却显示 `gemm_universal_adapter.h:323`。snapshot 的323实际上是 **另一个 API** `GemmKernel::initialize_workspace(...)` 的调用；待审调用发生于328。

结构数据解释了成因：唯一 evidence区间是323–328，源码 URL固定指向 `#L323`。这个区间可以证明“initialize_workspace成功后再构造Params”的前置顺序，但不能将区间起点当作当前调用点位置。期望将实际 callsite328与前置条件证据323–328分开显示，至少给边标签一个准确的 callsite328链接。图内确实只画了一条调用线，不代表这条线的源码落点已准确。

## types：没有裁掉已提供文本，但目标 API 的完整声明尚未进入旧产物

源端保留了 Builder偏特化的完整泛型声明及约束；目标 `types.builder_sm100` 的 note却只有“固定源码中实际命中的Dense TensorOp偏特化”，没有完整声明。对应 `atlas-data.js` 中，源 `types.builder_bound` 有 entity/signature字段，目标 `types.builder_sm100` 只有 declaration_status而没有 signature。因此这是**该版本数据补全/生成结果仍未更新**，不是 SVG在排版过程中截断了声明，也不能宣称这张图已经满足“两端完整签名”。

目标声明锚 `sm100_umma_builder.inl:169` 是正确偏特化；虚线没有误画成运行调用。边表达式保留 Sm100、half_t、KernelScheduleAuto、Alignment8等具体绑定。边下方184是偏特化条件证据的起点，模板绑定并非运行callsite，此处不能机械套用 Host调用行的判断。

可读性问题是重复信息：源类框已经展示二十余行泛型限定名/特化条件，下面 note再展示完整模板声明，使画布达到1371×1568。具体绑定实例的角色不够突出，容易让人把泛型模板重复文本当作“这一配置的类型”。建议源节点使用可辨识的实例名与具体参数绑定摘要，声明原文保留一次；偏特化 API端则提供其完整声明或明确的展开入口。不能通过删除长约束或省略完整签名来解决高度。

原尺寸时12–14px文字可读，是此次静态目视结论。若将整个1568px高图缩到590px高区域，线性比例约0.376，14px文字只剩约5.3px；缩到430px则约3.8px。这只是尺寸算术，**不是实际浏览器缩放测量**。当前旧CSS的 viewport为430px/大屏590px，JS默认原尺寸并提供滚动及适应宽度，不是适应整图高度；因此“整图自动缩小不可读”也不应误报为当前代码已经发生的事实。真正的滚动、缩放、翻页、焦点和点击体验仍需后续实际UI验收。

## 复核产物与边界

本地临时PNG保留在 `/tmp/dense-static-svg-review-4CyvcE/host-init-lower-white.png` 与 `types-specialization-white.png`。第一次未指定底色的rsvg渲染显示透明背景，查看器以黑底合成，黑色边标签因此难辨；加白底后正常。没有将查看器背景差异当成 SVG裁切或字体颜色错误。

本轮仅抽这两张详细边图，不代表所有详细图、导航panel、关系分页或整个站点通过。复验应使用新生成SVG的SHA，至少确认 types目标声明入图、Host实际callsite锚正确，然后再做同样的原尺寸目视。浏览器/HTTP预览的策略限制不是 UI通过证据；本审查没有尝试绕过。

## 新构建追加复验：四张逐边图

新构建读取的 `data/atlas.json` SHA256为 `173b2f6087344399bb3a54d465e0e5d3311b39360f7baa3bc79f413bee83345e`。本节再次逐图执行 `rsvg-convert -b white`，用 `view_image(detail="original")` 查看原尺寸PNG；同时解析SVG文本、关系分组、atlas节点与callsite，回切snapshot。**仍是独立静态展示复验，不是真实浏览器检查。** 本次只更新审查记录，没有改数据/站点/生成器。

| 边 ID | 新SVG尺寸 | 新SVG SHA256 |
| --- | --- | --- |
| `host.adapter.init_lower.328` | 1007×760 | `b605ba10ed2ec0936f24f0972436b257393aea36cffdffd47641af3b9478c888` |
| `types.edge.selected_specialization` | 1574×1740 | `8f046080228a3a125a02912da389fe64bdfcdf4dc924f0b7fbbead847336450f` |
| `contract.edge.kernel_load_rest` | 1026×1123 | `fd03b272293280995d2446d14e8f8fa51fdc6c599f526f5d5265ed8cf8b3e4ba` |
| `contract.edge.kernel_free` | 985×794 | `2de3cd62d1066544956da2a5c2205c74a62ac7ec44fc95fc34b5c8cc612aab2d` |

四张图实际文本字号均为12px/14px。原尺寸目视均未见签名裁切、文字互相覆盖或箭头落错节点。每张SVG恰有一组关系link，连接两个独立端点；note引线不是额外API调用。对每个source/target，去除纯排版空白后，atlas的完整signature（或绑定类型full_name）均能在SVG文本中完整找到。四图相应source URL文件与物理行anchor均存在；这不是实际点击测试。

### 原问题关闭

1. **Host调用锚点错误关闭。** 新图边标签明确为 `gemm_universal_adapter.h:328 [callsite]`；atlas的source_url也指向`#L328`。snapshot328是 `params_ = GemmKernel::to_underlying_arguments(args, workspace);`。323的initialize_workspace成功条件仍作为上下文，没有再冒充当前调用位置。两端initialize和to_underlying_arguments完整签名均显示。
2. **types目标声明缺失关闭。** 新图左端明确显示“本例CollectiveBuilder实例”，note包含Sm100、FP16、RowMajor、8对齐、Tile256×128×64、Cluster2×2×1、Carveout33792、Auto的完整绑定类型；右端已显示偏特化完整template声明和约束。snapshot155起是该template声明，169是struct名称，atlas源码链接与图上169标签均落在正确声明区域。虚线仍表示模板绑定，184–223明确标为evidence range，不冒充运行调用点。

### 两张新增抽样的源码对应

`kernel_load_rest` 图连接Kernel::operator()与CollectiveMma::load，source callsite为Kernel653。图中实际参数末尾是 `k_tile_iter_next, k_tile_count - k_tile_prologue`，条件明确是同一work tile的prologue之后、producer state已在646更新，剩余K可为0；snapshot653–659逐字对应，660接收第二次load返回状态。目标note保留MainloopPipeline/State、LoadParams、TileCoordMNKL、KTileIterator及int参数，匹配Mainloop583–594声明。没有误画成下一work tile调用。

`kernel_free` 图连接Kernel::operator()与 `cute::TMEM::Allocator2Sm::free`，callsite为Kernel803，两实参分别是tmem_base_ptr和Sm100TmemCapacityColumns；目标note完整保留 `__device__ void free(uint32_t tmem_ptr, int num_columns)`，对应allocator157–159声明。图中“accumulator tail与peer handshake之后”是caller顺序，不再有free内部等待的箭头。此次只查直接free调用图，不把它当成完整TMEM生命周期图。

### 仍需改进的可读性

- **types详细图过高且重复文本过多。** 它从旧版1371×1568变为1574×1740。右端类框已经展开泛型限定名和全部偏特化条件，下方note再重复完整声明；会议读者需要在两处重复长文本中寻找同一个API。建议类框以可辨识的偏特化名称和来源作为标题，把完整声明保留在note一次；不能删掉真实约束来换取小尺寸。
- **换行会切断标识符/路径。** Host source框中`enable_if_t`、`GemmKernel_`仍被拆行；types边标签将`include/...`拆为上一行`includ`与下一行`e/...`。字符未丢，但读者必须自行拼接。路径应作为独立一行或按斜杠边界换行，源码标识符优先按作用域/模板分隔符断行。
- **Kernel长owner压过操作名。** load_rest与free图的Kernel源端用十余行泛型owner，真正的`operator()`在末尾才出现。建议让操作名作为第一眼可见标题，完整owner继续在可读字段中保留。两图的原尺寸12–14px文字目前可辨；这个改进针对阅读顺序，不是字体裁切。

本节没有检验204个导航panel、14个view的实际翻页体验，没有宣称1366或1920屏幕下交互通过。四图的源码忠实度和上述旧缺陷可按本次SHA关闭；新的排版改进项仍开放。

本地白底PNG为 `/tmp/dense-static-svg-review-4CyvcE/new-host-init-lower-white.png`、`new-types-specialization-white.png`、`new-kernel-load-rest-white.png`、`new-kernel-free-white.png`；旧PNG未覆盖。
