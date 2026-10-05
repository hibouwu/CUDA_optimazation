# 阶段进度

原全库目标已重新开启，当前优先把网页主路径用于方案验证，原目标尚未完成。固定提交为 `8f50b052e1099fb982392a622caab69b97b63128`，库版本4.6.0，以3.x API体系为阅读主线。原范围仍为824个库文件，未设置token预算。

当前入口默认进入同一次 GEMM 的五步方案验证主线，Kernel 总览、Host与Scheduler交接过程、逐参数解释和逐关系证据保留为后续查阅层次。已从单纯展示整理转向主线依赖的源码交接核对：本轮关联13个Scheduler声明、增加17条关系，并将原fixup调用的目标关联到具体方法；声明提取器与原全库声明账本没有改写。主要路径可用仍不替代全库完成。[总图审查](audits/overview-presentation-review.md)、[主线检查点](audits/main-path-review.md)和[Scheduler交接记录](audits/scheduler-handoff-review.md)分别保留各轮依据。

## 最新主线检查点

已保存的 Dense 示例 M/N/K/L=256/256/128/1 贯穿五步：问题与范围、参数与资源、工作分配与计算、结果交接与完成、复用失败与证据。默认不展开内部图和长参数清单；关键条件和现场追问保持可见。其他场景只作为同事实际承诺时的差异追问，不当作已完成的支持能力。

当前17项主线检查、12项Scheduler交接检查、23项总图检查、12项接口参数检查通过；页面DOM-double回归现为42项，全站产物检查零问题。当前发布619节点、994关系，其中新增声明全部关联已有规范账本，824文件范围不变。以上不是浏览器或GPU运行验证，也未冒称新主线已经过独立使用审查；相应验收继续保持未完成。

主线现附三类条件性方案走查：布局表达、完成条件、重复调用与参数更新。明确区分有意范围取舍、缺少必要条件和违反已声明接口，避免把“与官方不同”直接判错。固定源码的 Host 布局接口已实际运行：同一256/256/128/1问题使用紧密与行尾填充8元素两种步长，对照131072个A/B坐标；忽略填充或误用B坐标次序都会得到不同地址。该结果只证明这些 Host 地址映射，未运行GEMM，也未认定GPU支持填充变体。[布局检查记录](verification/scheme-layout/README.md)及[方案走查记录](audits/scheme-contract-review.md)分别说明了执行证据与源码推理的边界。

Scheduler主线现区分完整K任务、CTA级坐标、五个独立fetch调用、两个Epilogue判定重载与空fixup。MMA/后处理完成current后才替换next；Epilogue输入则提前替换任务描述但保留旧坐标处理当前输入，不能共用一条更新时间线。下一项应核对CLC响应可读、消费者释放与下一槽复用的关键约束；不继续增加假设案例，也不把已列直接交接当成完整swizzle/CLC/GDC或调度唯一性证明。

## 当前推进方式

采用“全库统一建账、按模块贯通、最终全量汇合”。声明、关系和有依据的非API分类分别记账；已核实模块可以继续协议与图集工作，不等待其他模块声明缺口清零。与固定快照无关的新C++特性不自动扩展为必做范围。

阶段0已经核实且快照未变化，不重复建设。阶段1到4按模块及必要依赖推进；阶段5仍要求全库汇合与最终会议使用验收。模块通过不代表全库通过。

## Dense FP16 2SM设计路径

[离线入口](site/index.html)已生成。该路径用于验证数据模型、分图粒度与接口会议阅读方式，不作为全库覆盖证明。

本例固定为Sm100 C++配方、`sm_110a`目标、FP16 A/B、FP32 C/D及累加、RowMajor、Tile256×128×64、Cluster2×2×1、alpha=1、beta=0.5。

当前内容包括：

- Host生命周期、各个run重载、参数转换、Cluster launch、设备入口及应用的stream同步边界。
- Builder入口/偏特化、Auto配置选择、Mainloop/Traits/Atom，以及实际指令包装。
- 完整的Dense `gemm` 重载链：四参83 → rank-4五参275 → D&&转发142 → rank-1五参190 → Atom::call94 → friend `mma_unpack`2072。调用点88、298、148、197、104分别保留。
- 输入SMEM、累加器TMEM、输出SMEM、TMEM分配/释放四份资源契约。提交、完成、等待成功、调用者顺序与允许复用分别表达。
- 独立逐关系PlantUML/SVG保留作为证据层；Kernel 与 Host 三主题现以总图和语义过程图为入口，原每组至多两条出边的分组仅作辅助。完整签名仍保留在详细图和接口页。
- 全部824文件的离线源码与行号跳转，以及静态编译证据副本。

新NVCC13.0.88静态探针确认了Dispatch<8,2,4>、Epilogue carveout33792、CTA shape128×128×64、Atom shape256×128×16及2SM F16包装。唯一设备入口内的4条PTX f16 MMA和4条UTCHMMA.2CTA位置有证据。另用DWARF inline来源核对了上述重载父子链。只运行Host类型打印，没有运行GPU。

### 审查与未完成项

独立审查发现并推动修复了public/private跳层、错误重载端点、调用行冒充声明、重复调用点、等待责任归属与friend归属。原始反例及逐项回归保留在[源码审查](audits/dense-path-independent-source-review.md)；[静态展示审查](audits/dense-static-display-review.md)还记录了旧图的调用行标签和缺少完整签名问题。最新产物以[构建报告](data/atlas-build-report.json)与[产物检查](data/atlas-artifact-checks.json)的版本为准。

此前`pipeline_init_wait`及三个PipelineAsync方法使用过经原文字节核对的人工身份补充。现已从匿名指针默认参数的语法恢复根因修正整个P90文件，并重生成全库账本；图集这四个节点已重新关联干净的canonical声明，旧补充只保留为审查历史，没有通过改显示前缀冒充解析修复。

实际浏览器验收尚未完成：当前安全策略拒绝file页面，权限审核也拒绝HTTP预览。没有换浏览器或用间接方式绕过。SVG静态目视、代码单元测试及链接检查不能替代1366×768/1920×1080的真实浏览器检查；此项保持未完成，需要后续合规的用户参与验收。

外围Scheduler/fusion/其他配置等边界仍单列。代表路径的已核事实不推广为全库完整、runtime合法性、数值或性能结论。

专用的时序／状态分图尚未全部制作。当前资源事件索引与源码先后说明分别显示，互斥分支或并发事件不会因为列在同一清单中而被当作一条串行时间线。

### TMEM生命周期协议

TMEM生命周期v3已完成独立源码与有界模型闭合审查，现已接入协议页面。每CTA的角色、分配指针、NamedBarrier、最终deallocation barrier及配对累加器排空前提分别保存。90个事件、252条量词化偏序和64条历史事实推导有明确来源；2条配对参与判定规则要求双方必要记录共同满足，不能由单侧free替代。

审查关闭了分配写/读观察脱节、单侧操作冒充配对释放、线程族量词混用，以及本侧指令发出与返回缺少约束等问题。v1/v2反例与原数据未删除。源码注释的“non-blocking”与实际规范的潜在阻塞也明确区分。

协议数据生成86张关联分图；不同API和CTA分别保留，同一API的不同锚点有独立连接点。事件页可查完整API声明、调用位置、predicate、线程域和所有约束。最大图宽914像素；真实浏览器验收仍未完成。[实现与分图记录](audits/tmem-protocol-implementation.md)和[独立审查](audits/dense-tmem-protocol-independent-review.md)记录了具体边界。

三张高风险协议图完成了[独立静态展示复核](audits/dense-tmem-static-display-review.md)，包括源锚点和目标箭头的行框对照。公共Python回归350项、协议语义反例10项、DOM逻辑回归16项通过；当前769张生成图的机械产物检查为零错误。这些计数不作为全库完成依据。

P90的三个选中方法及`pipeline_init_wait`已不再依赖当前生效的人工身份补充；退役依据是新账本中完全一致的原始签名字节、限定名、种类与可信作用域。其他资源的逐generation协议、其余模块与全库汇合仍需继续。

## 全文件模块接入

[模块登记表](data/module-registry.json)已建立20个唯一主归属模块，824文件逐个入账，包括全部40个`.inl`。独立审查重新核对了完整路径集合、824份原文散列和归属规则，没有漏项、重复归属或按Dense可达性缩小范围。归属状态与API/关系覆盖状态分开。

`clear.hpp`、`fill.hpp`、`axpby.hpp`三个完整文件已接入同一图集，主归属仍为`cute_core`。工作包包含8个函数模板及4个namespace出现、局部对象和lambda、20个物理call-expression、91项源码义务及52项参数记录。其6个视图分别展示接口转发、fill条件分派、axpby分支、资源/顺序、必要类型依赖和宏展开。模板依赖目标保留表达式、条件和依赖参数，没有统一猜成某个实例。

审查推动修复了非依赖表达式被统一标为symbolic、宏仅有说明而无展开关系、同一行调用位置可能被合并，以及契约关系ID不可跳转等问题。[三文件审查](audits/cute-elementwise-root-review.md)与[模块/调用点独立审查](audits/module-and-callsite-review.md)保留反例和回归。该工作包未做C++实例编译或GPU运行，不能据此声称任意模板实例都已验证。

离线入口已增加全库模块目录、逐文件入口、跨模块关系及逐源码义务/参数页面。文件归属、已记录关系和最终验收状态均显式区分。最近公共Python回归335项、工作包回归11项、DOM double回归13项通过；实际浏览器验收仍未完成。

## 全库统一账本

- [固定文件清单](data/scope.json)：824文件、27,024,583字节，来自Git blob；[阶段0核对](data/phase-0-checks.json)及两路独立审查通过。
- `data/declarations.json`最近全量重生成：155698个出现位置、150883个暂定实体；配置绑定实例会重复物理位置，这不是冻结后的API总数。
- 最近全量声明仍有609项blocking诊断：函数体内语法269、函数体外132、其他208；已报告的作用域闭合损伤及相关scope review归零，但其他提取缺口仍在。[机械自洽核对](data/declaration-mechanical-checks.json)通过，不代表语义完整。
- 冻结原始候选仍为470518项。当前[全量映射](data/candidate-mapping-checks.json)为117415项声明映射、342615项有依据分类、10488项pending；输入与最新声明账本匹配，16处CUTE_ALIGNAS均由独立属性证明规则核准。分类中仍有转交后续关系阶段的义务，不代表关系全部完成。计数之和与原分母一致，不通过删候选制造完成。
- 位域已接入：固定源确认62项、18项无名且身份独立。12214个冒号候选全部有分类；原103项分别为76个构造初始化列表、26个asm分隔符、1个decltype基类，不改位域记录和原始错误事实。
- 条件化函数头和13项CUDA对齐声明已接入。共享函数体、模板前缀、访问权限及原始属性保留；组合投影的临时名污染已实际修复。
- 原有返回引用函数、重载、friend、namespace条件绑定、源码位置及宏生成接口修复保留。
- 旧版`static_assert`兼容typedef、部分局部初始化/类型查找、声明宏及其相关模块仍待完成。已经明确属于关系的候选进入后续关系义务，不重复当作声明缺口。

### P90作用域修复与防误归属

固定源码中只有P90:732这一处直接匿名指针默认函数参数需要新增语法适配。解析用临时名称通过零宽SourceProjection插入，API数据仍保留无名称、原类型和nullptr默认值。整个namespace恢复至1388，文件由370条出现记录恢复为508条（含两种namespace绑定），默认提取为0诊断。

作用域检查另外核对解析树与可靠物理闭合。明确损伤会阻止局部看似干净的tail声明被当作可信global API；条件不确定不谎报源码非法，但仍进入scope review。宏生成声明继承调用点的风险；不可信身份按原始来源隔离，不与真正的同名global声明误合并。

全库前后[语义差异对账](audits/scope-repair-delta.json)显示，声明语义只在目标P90文件变化。新的检查揭露了一个FP8 Blockwise Collective文件中原有的3项结构体闭合错误，104条记录明确要求scope review；它们没有被删除或改称symbolic。该问题的CUTE_ALIGNAS字段宏根因正在单独核对。

[CUTE_ALIGNAS根因核对](audits/cute-alignas-scope-root-cause.md)对应的修复已覆盖5个文件的16处真实属性使用（10处类型、6处字段）。每处保存原始宏与表达式范围、两条条件定义、include来源链及实际类型/字段owner；GNU拼写仅用于语法解析，不替代CUDA或Host的实际宏展开。真实header的编译对照表明，删除字段属性会改变字段偏移与sizeof，即使整体alignof没有变化，因此没有把它加入普通注解mask。

新的[全库差异核对](audits/alignment-repair-delta.json)只在这五个预期文件发现声明语义变化；scope review由104条降为0，FP8文件879行的独立局部声明pending仍保留。全量机械核对零问题后才替换canonical账本；原账本保存在`audits/declarations-before-alignment-repair.json`。初次接入检查点的独立原文/Clang/adapter检查17项、公共Python检查405项和属性机械反例均通过；随后补全候选重核和工作包检查。两个基础头文件的16项物理声明、10处属性owner均已对接新canonical，临时manual身份为0。

随后完成了独立候选全量重核与两文件工作包接入，当前公共Python回归429项、DOM double检查19项通过。两文件有100项源码义务、18项参数义务、72条关系和117张详细/导航图，已汇入现有站点；全站886张当前图的机械检查为零问题。独立接入审查先找出五个错误放行反例，修复后14项测试通过；另对四张新SVG目视核准，关闭非类成员public标签、宏展开需手工代入、双引号被改写等显示问题。[本次接入记录](audits/cute-alignment-root-integration.md)区分了所有验证层级。真实浏览器与全库验收仍未完成。

### array完整文件与公共声明字段

`include/cute/container/array.hpp`已接入正式图集：87个物理声明、92条条件绑定、52个可调用定义、54个实际形参位置、536项源码义务和509条关系。std/cuda::std分开，同一std实体的不同来源及条件逐项保留。range-for的两条隐式调用保留独立身份及真实for头锚点，不伪造源码字面调用。

接入审查修复了空selector回退、首条件冒充实体存在性、模板环境按错误位置聚合、隐式begin/end配对漏检，以及多来源图选错声明的问题。16项接入测试、独立原文/Clang审查和六张高风险图的静态复验通过；更多细节见[本轮接入记录](audits/cute-array-root-integration.md)。

公共字段也已重新生成全库：函数的返回声明说明符cv与callable限定符分开，241份限定符列表纠正；29560条数据声明增加抽象声明器类型拼写，形参增加序号。前后实体、出现位置及原始签名全部保持。当前全库仍为609项阻断诊断，候选待核10488项；using身份还有新确认的语义缺口，不因计数未增加而忽略。

最新公共Python回归495项、DOM double检查22项通过。四个工作包共1684张当前图已生成，文件归属仍是20个主模块、824文件。真实浏览器与全库会议验收未完成。下一项按[using模型](data/using-interface-model.md)修复普通导入、namespace directive、继承构造器与目标实体混淆的问题；array的相关依赖端点继续明确标为未完成。

[独立扫描](audits/unnamed-default-parameter-inventory.md)、[根因报告](audits/sm90-pipeline-namespace-root-cause.md)及[接入对抗审查](audits/scope-integrity-integration-review.md)保留了原反例和复验。旧完整账本保存在`audits/declarations-before-scope-repair.json`，新账本经全量机械核对后才替换，API分母仍未冻结。

### NVCC方言定性更正

已撤回“12处无逗号asm是CUDA源码错误”的旧判断。[NVCC复核](audits/phase-1-asm01-nvcc-dialect-recheck.md)证明：NVCC13.0.88在sm_100a和sm_110a实际调用12个原文包装时，PTX、CUBIN、PTXAS和SASS均通过；每个架构raw/补逗号对照逐字节一致。普通GNU前端拒绝不等于目标CUDA前端拒绝。快照未改，方言解析仍是相应模块的提取工作，未借此删除操作数或放弃覆盖。

## 重生成与继续执行

在本目录运行：

```bash
.venv/bin/python scripts/build_module_registry.py
.venv/bin/python scripts/build_atlas.py
.venv/bin/python scripts/refresh_presentation.py
.venv/bin/python scripts/audit_atlas.py
node tests/test_atlas_runtime.js
```

图与文字迭代可用`build_atlas.py --skip-sources`复用已经生成的固定源码页。此选项不减少824文件范围。JS测试使用DOM double，不启动浏览器，不冒充界面验收。

全库提取器、独立候选核对与回归入口继续保留：

```bash
.venv/bin/python scripts/extract_declarations.py
.venv/bin/python scripts/audit_declarations.py
.venv/bin/python scripts/reconcile_candidates.py
.venv/bin/python -m unittest discover -s tests -v
```

日常先检查受影响范围；模块完成、公共规则变更及最终汇合进行相应全量回归。不开新目标替代原目标，不修改上游或已有教学文档，不提交、推送、同步飞书或公开托管。
