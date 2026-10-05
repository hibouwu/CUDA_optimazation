# Dense TMEM lifetime：事件列表不能直接变成 sequence

版本说明：下面的原始说明保留自v1（SHA `3af156cc…`）；v1原始JSON、generator和本文原文另存 `history/v1-3af156cc/`。v2修订与D01/D02/Q01反例结果、最新v3的D03定点修订依次见文末，不能把历史版本的数据量或闭合状态当作当前版本结论。

本稿对应同目录 `tmem_lifetime.json`，只读固定snapshot并引用既有 `contracts.json` 的node/edge IDs；未修改既有契约、快照或root渲染器。它是当前Dense FP16、Sm100 recipe、sm_110a、Cluster2×2×1、非overlap、4个accumulator stage的源码语义草稿，不是实测trace或runtime正确性证明。

## 已拆出的对象与行为

CTA rank按 `K426–429` 的 `rank%2` 与 `rank^1` 形成两对：0/1、2/3。草稿显式创建每CTA独立的MMA和epilogue actor，共8个，不把两个CTA合成一个事件。MMA是warp0的32线程，epilogue是4个warp共128线程；依据分别为 `cutlass.h:96`、`K138–142/234–240` 与 `E127`。

每CTA的allocation、SMEM base pointer、allocation-result NamedBarrier、deallocation ClusterBarrier各自独立；另有每pair的leader侧accumulator.empty四槽排空视图。18个resource实例不是18个API。关键计数不可互换：CTA内分配发布是32+128=160；acc.empty是两CTA共256个参与epilogue线程/槽；最后deallocation握手是32个MMA线程/CTA，等待固定phase0。

现有contract的事件列表有以下不能机械序列化的反例：

1. **列表中allocate、arrive、arrive_and_wait的先后不代表调用开始次序。** epilogue可以先进入870等待；真正受约束的是它成功返回，之后871读取本CTA的SMEM pointer并在872调用set_tmem_offsets。MMA则在727分配、728独立执行syncwarp、729到达NamedBarrier，730/731读取并绑定自己的Tensor。NamedBarrier是CTA内发布，不是跨CTA握手。草稿的NamedBarrier状态允许MMA或epilogue任一类先到，不强行加入MMA先行的状态转换。
2. **同名arrive不是同一事件。** 793和795调用同一remote-arrive API但predicate相反；同一调用点在leader/peer也分别形成事件。793只有peer生效，目的地为leader的本地dealloc barrier；795只有leader生效，目的地为peer的本地barrier。predicate=false调用仍是执行过的C++调用，但贡献0次barrier arrival。
3. **`acc_tail`不是两个CTA共同直接执行的动作。** K788仅leader调用；它经P100220→P901129，用state副本循环4次producer_acquire并递增副本，最终到P901188等待对应empty barrier。不能缩成等最后一槽、等full，或调用者state持久推进4次。相关stage的epilogue release来自E887等待TMEM读完成之后的E888；它们按256个到达汇总，而不是第一个线程返回consumer_release即可排空。
4. **dealloc wait不是accumulator wait。** K794的成员wait转发B372→B410，指令B420读取本CTA自己的屏障；remote-arrive则是B383→B486，B489检查pred，B493映射peer地址，B494发送到达。leader在794直接等待的是peer MMA warp的到达；等待两个CTA epilogue释放accumulator的是前面的788。
5. **没有两侧free调用的全序。** peer完成793后可以停在794之前，leader完成788/794/795并先到803；也可以leader完成795后停在803之前，peer794返回后先到803。草稿对两种free.enter顺序各加一条假设边检查，二者都保持DAG。这里只是API调用到达的可行顺序，不是允许任一CTA独立完成2SM deallocation。
6. **allocation permit、stage empty、allocation deallocated与输出D完成是四个不同事实。** K783/A174的release_allocation_lock执行relinquish_alloc_permit，不释放现有TMEM；当前CTA之后不能再alloc。K803/A159的free只有dealloc指令，没有内部producer_tail或peer wait。E888后仍有fusion、R2S和TMA输出，K949还可能执行store_tail；本草稿不加入“输出D完成→TMEM可释放”的额外必要依赖，也不把TMEM free当成D对Host可见。

其中K=`include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp`，E=`include/cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp`，A=`include/cute/arch/tmem_allocator_sm100.hpp`，B=`include/cutlass/arch/barrier.h`，P100/P90=`include/cutlass/pipeline/sm100_pipeline.hpp` / `sm90_pipeline.hpp`。JSON顶层evidence保存了完整路径、原行号、字节范围、源hash和相应短源码段。

## 必须保留的规范/注释冲突

A125注释称allocate为non-blocking，但固定CUDA13.0.0的PTX ISA9.0明确：资源不足时alloc可阻塞；`.cta_group::2`要求peer CTA各一warp协作；`.sync`说明warp内会合，不能替代调用者的peer生存/释放握手。图不能根据该注释承诺立即返回或无限资源。A155的`@returns true`也不能覆盖A157–159的真实`void free(...)`声明。[PTX ISA9.0 §9.7.16.7.1](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#tcgen05-instructions-tcgen05-alloc-dealloc-relinquish-alloc-permit)

K799以及epilogue K936–942属于overlap路径，本配置不选择；草稿保留为inactive_branches，不拿它们给当前32计数deallocation握手提供到达。

## 数据和检查边界

草稿包含90个事件、240条明示偏序边、18个resource实例、64条状态转换、23个视图。每事件只表示一个独立API、builtin、源数据读或明确control boundary。未展开的工作循环不是虚构“复合API”；它作为有原位置的control boundary，相关Mainloop/commit仍引用既有Dense范围。

`implementation_paths`逐步区分NamedBarrier public→private、ClusterBarrier member→static、producer_tail→Impl→acquire等真实C++调用。每步给source、callee声明path/line/qualified_name和完整原始callsite；内联PTX明确为hardware_effect，不伪造C++函数端点。`mma_init`调用范围已精确回切为733–735，去掉`auto mma_inputs =`声明前缀。

事件的enter/return/effect是不同锚点。return只表示相应本地调用成功返回，不自动等于异步外部效果完成；每线程程序顺序与需要全部32/128/256个到达的join分别标注。图只允许使用partial_order列明的边，不由数组、actor排名或图上纵向位置补充时间关系。

生成器已检查：已有node/edge引用、物理源hash、callsite锚点、状态端点、所有order/transition至少进入一个view、偏序无环、两种free调用顺序都未被排除。两pair之间没有人为时间边；共同cluster初始化是前置条件，未宣称枚举scheduler/load等所有初始化actor。完整accumulator per-generation协议仍作为有引用的前置契约，不被这份lifetime排空投影替代。

重生成入口是同目录 `build_tmem_lifetime.py`。没有GPU运行、调度公平性/前进性证明、无效输入的安全证明、数值或性能验证。独立审查基准另见 `audits/dense-tmem-protocol-independent-review.md`；该标准的存在不是本草稿自动通过审查的证据。

## v2追加：D01、D02与Q01的模型修订

当前JSON保持结构schema_version=1，另标 `semantic_revision=2`。v1原始JSON的SHA仍精确为 `3af156ccb074cc3f0e4d819069d1281fcfded5e3635e9d217f56f8a7b1bbecb6`。既有独立audit未改；本次只修改draft目录。

**D01不是靠“API返回=效果完成”修复。** v2增加8条明示约束：每CTA的allocation-result store操作先于MMA在730的同址读取观察，也先于epilogue在871经CTA内发布后的同址读取观察。依据是PTX9.0 §9.7.16.6.1的allocation shared同址ordering与原始caller发布/等待链；store.effect仍没有连向allocate.return。将任一读观察强行排在对应write.effect之前都会使线程实例图产生环。[固定PTX9.0内存一致性条款](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#tcgen05-memory-consistency-model)

**D02分成本侧操作记录与配对有效性验收。** free.effect现在是 `local_dealloc_instruction_issued`，其anchor域为一个本侧warp协作操作，不是32次独立free，更不是已经发生的整个pair物理释放。allocation终态改为 `local_dealloc_recorded`。每pair新增独立 `validity_rules`，其 `all_of`明确要求：双方本侧warp操作记录、双方32个free.enter线程实例，以及均匀512列/相同逻辑warp/正确allocation/peer活跃等调用前提。结果是 `paired_deallocation_participation_validated`，明确 `explicitly_not_temporal=true`；没有物理完成时刻或跨CTA API返回顺序含义。

单侧free.effect仍可先于peer.free.enter出现在偏序扩展中，因为它已不再声称整个pair完成；但只提供单侧记录、只给两个operation记录却缺线程参与证据，或缺少必要调用前提，均不能通过配对有效性规则。没有添加peer.free.enter→local.free.return，两种free.enter先后继续都合法。

**Q01不再靠一段“thread-family”文字解释。** 每事件给出 `instance_domain`；enter/return是所选线程实例域，alloc/free的effect则是独立的本CTA `local_warp_operation`域，并列出其32线程参与映射。每条order的 `quantifier`明确为 `pointwise_same_thread`、`all_sources_to_each_target`、`all_sources_to_operation` 或 `operation_to_each_target`等。普通program-order只连接同一线程；barrier真正需要的32/128/256参与量另有all-to-each约束。init的elected-lane域只有1个实例，不固定声称lane0被选中。

每条状态转换同时有 `trigger_quantifier` 和可执行 `derivation`：`all_of`由历史from-state事实与量词化anchor观察组成，满足后产生to-state事实；不消费from事实，也不添加时序边。`derive_state_facts`以最小不动点求解；较早记录的trigger在其余前提稍后齐备时仍可使用。NamedBarrier的mma_only/epi_only等稳定ID现在表示正向历史证明事实，标签不再暗示另一族尚未到达或物理先后。

所有resource.state_semantics均明确为 `logical_obligation_projection`。这些是非互斥的契约判定事实，不是一个必须按物理时刻逐状态穿过的自动机；否则“所有线程完成caller_safe→首线程进入free”会凭空制造family barrier。唯一事件时序仍来自量词化partial_order；一次完整生命周期验收还须满足独立validity_rules，DAG无环本身不足以给出完整有效性结论。

新增 `protocol_semantics.py` 执行结构化规则并展开有界线程实例图；新增8项反例测试全部通过：

1. v1归档原hash与孤立allocate.effect反例保持可重现。
2. 四CTA的MMA/epilogue同址读先于allocation write被拒绝。
3. 不把allocation effect强行接到API return。
4. 单侧操作、缺少32线程参与记录或调用前提时，配对验收失败。
5. 不增加peer.enter→local.return，两种free.enter先后都保留。
6. `arrive795.return(thread1)`不能推出`free.enter(thread0)`；反向次序可以作为合法偏序扩展，证明没有隐性family barrier。
7. 31个MMA贡献加128个epilogue贡献不能形成NamedBarrier both事实；补齐第32个才可推导。
8. 状态规则不消费历史事实、不添加时序，且观察集合扩展只增添证明事实。

命令：

```bash
.venv/bin/python data/module-drafts/dense_protocols/build_tmem_lifetime.py
.venv/bin/python -m unittest discover -s data/module-drafts/dense_protocols -p test_protocol_semantics.py -v
```

当前为90事件、8 actor、248条偏序、18资源、64条状态推导、23视图。这个测试器使用一个对应的stage-generation作有界反例，不自动推断scheduler的所有动态generation域，也不是执行真实GPU程序的模拟器。事件域、量词和推导规则的源码可读，方便后续独立复核；没有因此宣布runtime、完整内存模型或整个Dense协议已经验收通过。

当前draft SHA-256：`253091a30d2367bb268b230831a195adb2a6bf8add419a61ea1841abee77e4e0`。

## v3追加：只修D03本侧指令发出与返回顺序

v2的完整JSON、generator、语义模块、测试和本文原文已保留在 `history/v2-253091a3/`；归档JSON SHA仍为 `253091a30d2367bb268b230831a195adb2a6bf8add419a61ea1841abee77e4e0`。

独立审查指出：v2将free.effect改成“本侧dealloc指令已发出”后，却仍允许同侧free.return排在该锚点之前。这里不涉及peer或物理配对完成：`Allocator2Sm::free` 的A159–170先经过本侧inline asm，再结束本侧函数，因此这个本侧先后约束有直接源码依据。

v3仅追加4条 `local_operation_before_local_return`，ID为 `tmem.order.249` 至 `.252`，每条都是本CTA的 `free.effect(operation) → free.return(每个本侧线程)`，量词为 `operation_to_each_target`。原248条order完整记录保持相同，事件、资源、状态推导和配对有效性规则未扩展；没有增加peer.enter→local.return，也没有更改free.effect为物理配对完成。

新增反例覆盖4个CTA的全部128个本侧return实例：逐个强加 `return(thread) → effect(operation)` 均产生环而被拒绝。归档v2的四个lane0反例仍可复现。原D01/D02/Q01测试继续通过，尤其两种free.enter先后都保留、同侧不同线程没有凭空产生arrive→free barrier。

当前10项测试全部通过。数据为90事件、252条量词化偏序、18资源、64条历史事实推导、23视图；此次没有扩展其它语义，也没有GPU运行或新硬件时序假设。

v3 draft SHA-256：`3cd91a827e83be05fd397cebbfdf1d76775313c93917a36fa6e0bfba223ab633`。
