# array 完整文件接入与对抗审查

固定提交仍为`8f50b052e1099fb982392a622caab69b97b63128`。本工作包覆盖`include/cute/container/array.hpp`完整476行，未修改上游或已有教学文档，也没有改变824个库文件的最终范围。

## 声明、关系与来源

原文的87个物理声明、92条条件绑定均已逐项关联正式canonical账本，不使用人工身份覆盖。物理清单有52个可调用定义、22个别名、8个struct声明、1个数据成员和4个namespace出现位置；92包含宏命名空间带来的配置绑定，不是另一个物理API分母。

工作包有54个实际写出的形参位置、536项源码构造义务、509条关系。31个普通显式调用、9个重载下标调用与2个range-for隐式调用分开。T(0)、元素赋值和比较保留依赖表达式，不一概当作内建操作或虚构已确定的用户API。

canonical接入除了比对签名字节，还核对限定名、独立实体、各出现位置、形参／模板环境、返回类型、限定符、函数体范围和预处理条件。不同std/cuda::std实体不合并；同一std实体两处声明不丢位置。整体预处理存在条件按各位置条件组析取，不再用首处条件代替。

## 审查推动的修复

独立的[source审查](cute-array-independent-source-review.md)与[接入审查](cute-array-stage-independent-review.md)保留了原始反例和定点关闭记录：

- 显式空selector曾被当成“没有selector”，经短名或manual路径放行；现在空值／无效值明确失败。
- 返回基础类型中的const曾混入函数限定符；现与成员函数const分开，合法cv拼写顺序也不抹掉指针层级。
- 合并std两处声明后，模板参数曾按同一连续列表比较；现依据该occurrence及其词法祖先的真实签名范围选择模板环境。
- checker曾只检查隐式begin/end目标集合，交换两条边仍可通过；现逐synthetic expression核对API端点。
- 基础类型、声明器、完整类型拼写及return cv曾可相互矛盾；现逐字段回切实际array AST子节点，完整stage拒绝原反例。
- 详细图曾显示合并实体的首处声明而非当前关系的来源。现在按关系的源码范围选择实际声明，只允许精确的尾部分号差异，不以任意范围相交放行。

16项接入审查在未修改正例先通过的前提下完成，避免用无关早期错误制造“所有负例都被拒绝”的假象。[core字段复审](array-core-field-independent-review.md)另有12项固定array及合理组合检查，明确保留未支持的相邻宏／条件函数头组合为pending。

## 全库元数据更新

本次公共字段修复已重新应用到全部824文件，并经机械核对及[身份保持检查](array-metadata-upgrade-invariants.json)后发布：

- 150883个暂定实体、155698个出现位置、609项阻断诊断均未改变；源身份、原始签名和其他原有字段保持。
- 28216条可调用记录分离返回声明说明符cv；其中241份旧限定符列表得到纠正。
- 29560条数据声明增加包含抽象声明器的类型拼写；函数形参有明确序号。
- 当前账本SHA为`d5d3c4e7be930915294639549b461081cf3f66fe0d52c743195cd523e78bd987`，旧版保留在`declarations-before-array-metadata.json`。

字段变化遍及725个文件，这是公共元数据升级，不是改动了725份上游源码。`return_cv_qualifiers`只表示直接声明说明符cv，不包括别名展开、所有指针层或尾置返回中的全部cv；字段合同见[data/declaration-field-semantics.md](../data/declaration-field-semantics.md)。

全量候选再次重核后仍为470518项：117415项声明映射、342615项有依据分类、10488项pending。当前映射SHA为`8ac3cdb062400a3d3f041bc6060bd6cbd97f3fccaa8ca9f82fd73efca0638032`。分类中仍含待后续完成的关系义务，阶段1没有因此通过。

## 离线图集与实际验证边界

array工作包生成798张详细／导航图，汇入现有站点。当前全站有606个已记录节点、977条关系、1684张当前图；这些是已记录产物的数量，不是全库API覆盖率。

[独立静态展示审查](cute-array-static-display-review.md)对六张给定SHA的新图完成原尺寸目视，包括std桥接、free clear、get&&的两条调用、range-for begin/end。未见签名裁切、错API箭头或文字遮挡；12个对象链接的结构正确。来源选择helper的66个实际端点已逐项核对，过宽到EOF的负例仍拒绝。

当前公共Python回归495项通过，DOM double检查22项通过。Host Clang证据包含独立source契约正反例及作者保存的14次syntax-only编译；这些不代表GPU运行、所有T/N实例、CUDA／RTC分支或性能已验证。

真实1366×768、1920×1080浏览器使用验收仍未完成，当前安全策略限制没有被绕过。六图目视不扩大为全部798张均已目视通过。

## 尚未关闭的依赖身份问题

`type_traits.hpp:92`的using记录仍有伪QName。array模块保留真实导入语句、lookup binding及两条条件目标，不将已知提取失败伪装成普通模板依赖。进一步审查还发现namespace directive与继承构造器导入有同类问题，见[独立using审查](using-declaration-independent-review.md)和[修复模型](../data/using-interface-model.md)。

这项跨模块身份修复、其他文件的声明／关系／结构契约以及最终会议使用验收仍在持久目标内。没有将本工作包或四个已接入工作包标为全库交付完成。
