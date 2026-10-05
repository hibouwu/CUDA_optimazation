# Dense 非overlap 2SM TMEM 生命周期：独立源码反例标准

本稿先从固定snapshot建立审查标准，不以正在制作的 `data/module-drafts/dense_protocols/tmem_lifetime.json` 为证明。源码提交为 `8f50b052e1099fb982392a622caab69b97b63128`，目标是当前Dense配置：Sm100 recipe、编译目标sm_110a、FP16 A/B、float C/D、Tile256×128×64、Cluster2×2×1、AccumulatorPipelineStageCount=4、IsOverlappingAccum=false。

**目前完成的是独立源码标准，不是draft数据验收。** 本轮未执行GPU、未浏览/预览HTML，没有修改模块数据或生成器。外部指令契约采用本任务提供的Root已核PTX ISA9.0结论：`tcgen05.alloc` 在资源不足时可能阻塞；本审查不将该外部结论标为自己重新访问文档后通过的核验。以下物理源码均独立重新读取。

路径简写：K=`include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp`；E=`include/cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp`；A=`include/cute/arch/tmem_allocator_sm100.hpp`；B=`include/cutlass/arch/barrier.h`；P100=`include/cutlass/pipeline/sm100_pipeline.hpp`；P90=`include/cutlass/pipeline/sm90_pipeline.hpp`。均相对snapshot。

## 1. 图中至少有四种不同资源，不能共用一个“TMEM barrier”身份

| 对象 | 作用域、计数或单位 | 必须保存的身份与依据 |
| --- | --- | --- |
| TMEM allocation及其base pointer | 一对协作CTA；alloc/free的num_columns实参为512；每CTA有自己的shared pointer槽 | A117–141、K727/730/871。512是原指令实参，不能自行改写成两CTA合计1024列 |
| allocation-result NamedBarrier | 每CTA本地；NumMMAThreads+NumEpilogueThreads=32+128=160；ReservedNamedBarriers::TmemAllocBarrier | K557，B198/210/222及285/305。没有由调用者传入的accumulator index/phase |
| accumulator full/empty ring | 每stage的full计数1；empty等待两CTA共2×128=256个epilogue arrival；4槽index/phase环 | K169、528–529、596–597；P100145–146。full/empty与最终dealloc barrier不是一个对象 |
| deallocation-result ClusterBarrier | 每CTA一个`shared_storage.pipelines.tmem_dealloc`；非overlap各自init计数32，等待phase0 | K559–564。不是4槽ring，没有按每个work tile翻phase |

数字来源不是角色名推测：`cutlass.h:96` 明确NumThreadsPerWarp=32，K138将NumMMAThreads设为该值，K141使用CollectiveEpilogue::ThreadCount，而E127明确ThreadCount=128。

Cluster2×2×1有四CTA，但MMA协作单位是Atom的两CTA。K426–429使用`cta_coord_v=rank%2`、leader=`cta_coord_v==0`、peer=`rank^1`，因此是rank0/1和2/3两对，不是一个四CTA TMEM协议。不应把“SM0”解释成全设备或全cluster的唯一CTA0；地址清peer bit与rank^1的配对范围必须保留。

## 2. 分配与CTA内发布：必须保留的约束

K234–239把MMA设置为逻辑warp0，epilogue从warp4开始；K448–453分别设is_participant.mma与is_participant.epilogue。两CTA各自的MMA warp都会走K725分支。**只有实际MMA计算/accumulator commit在K761加leader guard，不能把该guard向上扩展到allocate或向下扩展到free。**

K727–731的源顺序是：

1. `allocate(512, &shared_storage.tmem_base_ptr)`。
2. 独立的`__syncwarp()`728。
3. 本CTA NamedBarrier::arrive()729。
4. 本MMA路径读取shared pointer730，然后set_tmem_offsets731。

Epilogue走K868分支，先`arrive_and_wait()`870，之后才能在871读取pointer，并在872设置自己的TMEM tensor偏移。NamedBarrier成员210/222分别转发private285/305；实际指令分别是`bar.sync`287与`bar.arrive`308。图必须表示不同调用者和不同API，不应把arrive画成MMA路径也会等待epilogue返回。

A128–132列出allocate的前置条件：两CTA提供相同shared地址偏移、每CTA单个完整活跃warp、不能同时多个warp发起、重复分配使用同一warp、两CTA使用相同逻辑warp ID。实际wrapper在137将generic指针转换为shared地址整数，139发`tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32`。两个shared槽分别属于两个CTA；“相同地址”不是两个CTA共享一个普通C++对象或同一个Host指针。

**源码注释与契约冲突必须显式保留。** A125写“non-blocking”，但Root提供的PTX9.0核对结论是资源不足时可能阻塞。图不得据注释画成“立刻返回、无限资源、无需考虑资源等待”；也不得自行增加源码中不存在的allocation completion mbarrier。可以记录allocate调用/潜在等待/指针发布边界，并区分该指令契约与后面的CTA内NamedBarrier发布。A155还有`@returns true`注释，真实free声明A157–159返回void；图以声明为准，不产生布尔完成返回值。

## 3. accumulator stage安全不是整块TMEM已释放

本配置K751–759采用producer state.index()选stage，不采用overlap路径的phase^1。K772每work tile递增producer state。Mainloop `init_tmem_tensors` 的构造调用在 `sm100_mma_warpspecialized.hpp:475`，`detail/sm100_tmem_helper.hpp:70–73` 在非overlap分支使用实际AccumulatorPipelineStageCount=4。不要照抄Mainloop474的旧“ACC_PIPE=2”注释，将此配置画成双缓冲。

full表示MMA结果可读，empty表示此前读者已释放该stage。P100212→260→264与B848/854的commit链必须保留；不能将MMA发射、commit函数返回、硬件完成通知合并成一刻。empty释放为P100242→272→275→B907；B909清peer bit，B912向配对leader侧屏障arrival。这是每参与epilogue线程的arrival，不是一个elected thread为整个warp代表到达。

K911调用`store<IsOverlappingAccum>`，故本配置ReuseTmem=false。E823–825的do_acc_release为最后subtile；E882先发TMEM load，887执行fence_view_async_tmem_load（B923–927实际为`tcgen05.wait::ld.sync.aligned`），888再consumer_release，889递增consumer state。**等待由epilogue调用者承担，consumer_release自身不执行TMEM load等待。** 图还要保留issue_tmem_load条件与do_acc_release条件不同，不能将未发load的线程从256个arrival参与者中静默删去。

K788只有leader CTA的MMA路径调用accumulator producer_tail。其真实链是P100219→P901129，后者1130–1132使用参数state的本地副本循环4次producer_acquire并递增；acquire1110→1185→1188等相应empty barrier。它排空4槽，不是只等最后一槽，不是等待full，也不是对调用者state对象作四次持久更新。

这证明的是TMEM accumulator读者已退出相应stage。E888之后仍可做fragment运算、R2S与TMA store；K948的store_tail也在epilogue末尾独立条件中。不能画出“D全局写完→才能释放accumulator/TMEM”的额外必需依赖；也不能反过来把accumulator stage释放当作D写完成。

## 4. 最终两CTA握手是偏序，不是一条全局时间线

非overlap时，两CTA的MMA elected lane仅负责K564的barrier初始化；**793/794/795没有lane_predicate guard**，MMA warp的32个参与线程各自执行。对应的remote arrive成员是B383，转发static486；489仅在pred为真时发指令，493把本CTA内的相同barrier偏移映射至peer CTA，494执行remote arrival。wait成员371→static410，420等待的是本CTA barrier，而不是在peer地址上执行wait。

设L=配对leader CTA，F=配对follower CTA；DL/DF分别为两CTA自己的dealloc barrier：

| 原调用点 | L侧作用 | F侧作用 |
| --- | --- | --- |
| K788 producer_tail | 排空accumulator所有stage | guard为假，不调用tail |
| K793 arrive(peer, !leader) | pred=false，不贡献arrival | 每个MMA线程向DL到达，共32 |
| K794 wait(0) | 等DL收到F的32个到达 | 等DF收到L的32个到达 |
| K795 arrive(peer, leader) | 每个MMA线程向DF到达，共32 | pred=false，不贡献arrival |
| K803 free | 本侧到达free调用 | 本侧到达free调用 |

必须保存的依赖是F793的有效到达满足DL；L的各线程先经过788并从794返回，才在795向DF到达；DF收齐32个有效到达后F794才能返回。于是F通过回信继承leader已经排空accumulator的前提。L794直接等的是F的MMA线程到达，不能标成“等epilogue”；真正等epilogue stage释放的是L788。

线程调用、32个到达的聚合完成、barrier phase完成应分别定义。图若用一个“warp事件”代表32个线程，必须说明它是全部相应线程事件的聚合，不把第一线程到达当成整个warp完成。不要从聚合节点的排版额外推导所有跨CTA事件有全序。

以下两种到达顺序都不被这段源代码排除，因此是协议图必须接受的反例：

- **先到leader free：** F完成793的32个arrival后暂停在794之前；L完成788、794、795并到达803；F稍后进入/完成794，再到803。
- **先到follower free：** L完成795的32个arrival后暂停在803之前；F794返回，经过predicate为假的795，先到达803；L随后到达803。

所以不能画L803先于F803，也不能画F803先于L803；不能声称两侧794同时返回或F必须先开始wait后L才可返回。上述反例只讨论**到达C++ free调用/指令发射位置**，不是说某一侧可独立完成2SM deallocation；实际`tcgen05.dealloc.cta_group::2.sync.aligned`的协作条件另属指令契约，不能从C++调用到达顺序虚构硬件完成全序。

K783的release_allocation_lock发生在本MMA路径的最终drain/握手/free之前，实际A174–176为`relinquish_alloc_permit`。它归还的是分配许可，不是释放当前512列；允许后续调度推进的目的也不意味着当前TMEM可以马上被覆盖。free A159–170只有dealloc指令及配置失败分支，没有accumulator producer_tail或peer wait。调用者的释放责任不能下沉成free的实现行为。

## 5. draft出来后应逐项拒绝的错误

下列任一项出现，相关protocol图不能通过该样本的源码审查：

1. 将2SM两CTA配对扩大成四CTA cluster单一分配/握手，或把leader叫全cluster唯一CTA0。
2. 把allocate/free裁成仅leader或单个elected lane执行；忽略完整warp、相同逻辑warp ID及共享地址偏移约束。
3. 将A125注释直接升级为“alloc绝不阻塞”，或凭注释生成free的bool返回值。
4. 把K728的syncwarp合进NamedBarrier::arrive API，或把K870和K729当成同一个API重载。
5. 将160人的CTA内allocation发布、256人的acc empty、32人的deallocation握手共用一个计数/资源/phase字段。
6. 根据旧注释将当前accumulator画成2槽；将deallocation phase0误用accumulator环的phase或index。
7. 把accumulator commit提交当作MMA完成；把consumer_release API自身画成等待TMEM reads。
8. 将producer_tail画成只等一槽/full，或认为follower直接执行了K788。
9. 合并793与795的两个predicate arrive；将predicate=false调用也计入有效arrival；把remote arrive目的CTA画反。
10. 把794画成remote wait，或把L794直接解释为等待epilogue到达。
11. 为两CTA的free调用添加任一固定先后关系，或把wait返回画成同时发生。
12. 将release_allocation_lock等同free、将stage empty等同allocation结束，或声称free内部完成tail/peer wait。
13. 将当前非overlap流程混入K799或K936–942的overlap epilogue直接deallocation-barrier arrival；该分支本配置未选。
14. 在accumulator释放或free之前强加D写完成/Host同步前置，或者把TMEM释放当作全局D可见性证明。

与几何/指令能力有关的未决项必须保持可核查来源：例如512列的实际物理资源单位、alloc潜在阻塞、两CTA指令配对与完成条件，不能用一条泛化“GPU完成”边替代。固定源码能证明调用者的调用/参数/条件与资源移交路径；硬件时序、正确启动、数值及性能没有在本轮运行验证。

draft尚未在本审查读取时出现。本稿不提前给制作数据通过结论；后续应先记录draft SHA，再逐节点/边检查以上标准，并保留任何新反例和修复前后的版本。

## 冻结draft追加审查：3af156cc版本

本节独立读取 `data/module-drafts/dense_protocols/tmem_lifetime.json`，开始和检查结束时 SHA均为 **`3af156ccb074cc3f0e4d819069d1281fcfded5e3635e9d217f56f8a7b1bbecb6`**。90个事件、8个actor、18个资源、240条partial_order、64条状态转换、23个view只是该版本数据量，不是通过依据。没有修改draft或renderer；没有使用作者的validation布尔值代替核对。

**结论：角色、调用点与所列源码约束大部吻合；NamedBarrier五态未强加MMA/epilogue先后，两种free.enter先后也均被允许。但“效果—状态—可观察读/协作有效性”的模型尚未闭合，下面D01/D02需解决后才能给出这张生命周期模型的定点通过。** 这是模型闭合不足，不是宣称CUTLASS硬件允许非法单侧free或读到尚未生成的地址。

### 独立检查结果

- 90个事件callsite的表达式经纯排版空白归一后全部匹配snapshot所列物理行；98项库内evidence quote逐行完全匹配。实现路径将NamedBarrier、ClusterBarrier、accumulator tail/release的真正重载和PTX效果分开。
- 角色和数字满足标准：四CTA分成两对，每CTA32个MMA线程/128个epilogue线程；alloc NamedBarrier=160、acc empty=256/stage、dealloc barrier=32且phase0；非overlap/4槽正确。
- 两个pred=false的793/795事件显式为`no_barrier_arrival`，arrival_family_size=0，没有向wait贡献虚假计数。794仍等本CTA资源；799与939/941另存inactive branch。
- 独立重建partial_order邻接图并加入反向次序进行拓扑检查：L.free.enter→F.free.enter和F.free.enter→L.free.enter均保持无环；NamedBarrier的MMA贡献族先完成、epilogue贡献族先完成两种次序也均保持无环。不是复用counterexample_checks的结论。

### 14条标准的逐项判断

| 标准 | 本版本判断 |
| --- | --- |
| 1 两CTA配对不能扩大为四CTA一组 | 吻合；actor、资源均保留rank和pair |
| 2 完整warp/双CTA参与 | 角色与preconditions吻合；完整deallocation效果的双侧有效性还需D02闭合 |
| 3 non-blocking/返回类型注释不能覆盖契约 | 吻合；may_block=true，冲突单列，未生成bool free返回 |
| 4 syncwarp/NamedBarrier调用不可合并 | 吻合；独立事件及真实API重载 |
| 5 160/256/32与三类barrier身份不可混用 | 数字与资源吻合；线程族锚点和状态聚合量词见下文Q01 |
| 6 四槽和dealloc phase0不能混用 | 吻合；未用旧ACC_PIPE=2注释 |
| 7 提交不等于完成，release不内置read等待 | 吻合于此lifetime投影；read_fence→release单列，完整stage协议作为显式precondition引用而非本稿重证 |
| 8 tail须排空四槽且仅leader执行 | 吻合；本地state副本、已空stage不虚构release |
| 9 两次predicate arrive不可合并/颠倒 | 吻合；两侧四个事件均保留各自布尔作用 |
| 10 local wait不冒充remote/epilogue wait | 吻合；owner CTA资源明确 |
| 11 不得虚构两侧free到达/返回全序 | free.enter偏序反例检查通过；effect有效性另见D02，不能据此推导内存物理释放时序 |
| 12 permit、stage empty、allocation结束不可混同 | 名称和转换分开；allocated-address发布链缺D01，配对释放效果缺D02 |
| 13 overlap路径不可串入当前非overlap流程 | 吻合；未用epilogue939/941补32到达 |
| 14 TMEM释放不是D完成/Host可见性 | 吻合；store_tail没有被强加为free前置 |

### D01：地址写效果与随后观察到的指针缺少明确的模型连接

`tmem.cta0.allocate.effect` 被定义为`tmem_allocation_and_result_store`，同时触发allocation由absent变live和base_pointer由unset变written。但在partial_order中，此锚点只有自身allocate.enter前驱，**没有任何后继**。`allocate.return→syncwarp→alloc_arrive→alloc_wait.return→epi_read_base.return` 这一条程序/同步链不包含allocate.effect。

机械反例：向当前partial_order临时加入

```text
tmem.cta0.epi_read_base.return → tmem.cta0.allocate.effect
```

仍存在拓扑序。它说明单看显式偏序图，可以完成epilogue的指针读取，而分配结果写效果节点还未发生。state_transitions.006–.008另有written→published→observed链，能在严格状态执行语义下拒绝这个次序；问题是draft同时声明`implicit_partial_order=false`，没有明确“偏序和状态前置共同约束同一执行”的组合规则。不能只用DAG无环通过来覆盖二者间的缺口。

Root本轮补充核对了固定PTX9.0 §9.7.16.6.1：alloc/dealloc/relinquish属于Synchronous分类，allocation相关shared访问与non-tcgen同地址访问有序；§9.7.16.7.1仍说明alloc结果store为weak。这里可以据**同地址内存顺序**连接allocated-address写与后续同地址读的观察，或明确状态/偏序的组合语义，使违反written前提的读不可被当作合法模型执行。**不要仅为补齐图形而推断“每个线程allocate.return等于整个配对物理store完成”**；API返回、对应内存访问顺序、跨warp发布仍是不同命题。

上述PTX补充由Root本轮在线核对后提供，本审查未访问HTML。源码对应A135–141、K727–731和870–872已重新核对。

### D02：单侧free.effect的名字过强，双侧参与仅存说明性前置

`tmem.cta0.free.effect` 的kind是`tmem_deallocation`，触发transition.005把本CTA allocation标为`deallocated`。它仅有自己free.enter前驱；`requires_cooperative_peer_call=tmem.cta1.free` 虽然记录了意图，但不是一个已明确求值的配对参与条件/有效性规则。

机械反例：向partial_order加入

```text
tmem.cta0.free.effect → tmem.cta1.free.enter
```

仍存在拓扑序。这不是证明硬件允许单侧独立释放，而是表明显式偏序层并没有兑现effect名字和transition所要求的“paired deallocation takes effect”。free.enter先后都允许，本来就是正确的；**不能为了阻止这个模型反例而额外发明peer.free.enter→local.free.return或两个return的全序。**

Root核对的PTX9.0 §9.7.16.5只要求peer各一个warp共同执行配对操作；不能仅从`.sync`推导某侧返回一定晚于另一侧开始。修复可采用下列任一有明确语义的表示：

- 把单侧可证的操作/调用返回，与“两个peer都满足参与要求”的逻辑有效性/验收状态分开；不要将单侧节点命名成已完成的整个配对deallocation。
- 若保留完成状态，给它一个明确的配对参与成立条件，且说明这是逻辑成立条件而非新声称的物理时间全序；验证器和图的阅读规则要共同使用该条件。

state transition的自然语condition“paired deallocation operation takes effect”没有被本审查当作错误硬件断言；开放项是它尚未与事件family、guard及执行有效性形成无歧义的模型契约。单侧enter、配对有效、物理内存释放完成不能互相替代。

### Q01：线程族effect/return量词与NamedBarrier五态

NamedBarrier五态的构造本身合理：neither明确表示两种**完整贡献族**都尚未收齐，并不表示零个部分arrival；mma_only和epi_only都能进入both，故未暗中规定MMA先或epilogue先。transitions.010/.012的条件分别要求对应family完成，.011/.013处理另一族随后收齐，.014才表示wait观察。32/128的join也避免一个线程arrival提前放行所有epilogue。

需要精确保留的语义是：普通program_order的`return→enter`是同一线程/同一调用实例；资源flags的变化则通常在**整族贡献收齐**时发生，不能以第一个线程effect替代。当前语义字段称event是per-thread family，而state trigger只引用同一个`event_id.effect`，聚合通过condition自然语说明。此处未发现一个明确错误的MMA/epi先后反例，**不单独将五态图判错**；但后续渲染/验证不能抹掉family-completion条件，最好给trigger/join直接保留量词，避免把family节点当作一个不可分的全warp瞬时事件。

不能给所有event机械补同一种`enter→effect→return`：init的本地初始化、NamedBarrier本地贡献发出、relinquish的本地语义可以按对应调用及指令契约判断；remote-arrival effect若表示已应用到peer，就不应仅凭本侧C++返回强加effect-before-return；alloc的shared地址写和free的配对效果按D01/D02分别建模。API返回在这一层仅表示本地调用返回，不自动表示远程barrier完成或整个配对硬件完成。

### 本版交付边界

本版已经保留大量必要的正确约束，不能将D01/D02描述成“所有协议都错”。反之，90个source匹配、98个quote匹配、DAG无环和两种free.enter顺序可扩展，也不足以宣称效果/状态的组合协议闭合。D01/D02解决后应按新SHA定点复验；本审查没有要求补猜规范未定义的硬件时序，没有修改data，也没有进行图形或浏览器验收。

## v2追加复审：D01/D02/Q01关闭，新增D03

本节绑定draft SHA **`253091a30d2367bb268b230831a195adb2a6bf8add419a61ea1841abee77e4e0`**；`protocol_semantics.py` SHA为 `a5d3c2a8d6b08714330a7431623702f0da7302220a39c2fe9e9b77a64c4a7c80`，专门测试文件SHA为 `c85e895738f0e6f1c9dca230ec0ec1107e70567ea0203488fec84677e0bdad2c`。旧v1在history目录的原字节SHA仍为3af156cc版本，历史未覆盖。

本审查完整读取了语义实现和8项专门测试，但**没有只复跑作者测试作结论**。独立按event domains和每条quantifier重建线程/operation图，得到12436个锚点、115472条边，与作者expand_graph逐项相同；然后执行额外反例和独立状态推导oracle。

### D01：同址write→read观察——关闭

v2新增8条明确的内存顺序，分别连接每CTA本侧allocation-result write operation到32个MMA或128个epilogue同址read.return。独立检查全部4×(32+128)=640个读者均从对应write可达，因此任一反向read.return→write都会形成环。原v1的反向指针观察反例被拒绝。

同时没有把write operation连到allocate.return；v2保留了weak store、API返回和跨warp发布的区分。此次关闭依据是同地址内存观察约束，不是增加一个未经证明的配对分配完成时间点。

### D02：配对参与有效性与物理完成分离——关闭

单侧free.effect已改为`local_dealloc_instruction_issued`；资源不再有`.deallocated`状态，改为本侧操作已记录的历史事实。`validity_rules` 是可执行的logical all_of：双侧local operation事实、双侧全部32个free.enter事实，以及明确的源码参与前提，全部满足才接受该pair的参与有效性。接受状态显式不声称物理配对deallocation完成时间。

独立构造两对CTA的有效输入，逐一删除每个CTA的每个lane——共128个缺lane反例——均被rule_satisfied拒绝；删除任一侧全部事实、任一local operation，或移除/置假参与前提也均拒绝。没有只检查“单侧”这一种缺失。两个peer.free.enter先后仍自由，且没有新建peer.enter→local.return顺序。

该关闭只表示逻辑配对参与条件已明确并真正执行；它不把bool验收结果变成物理内存已释放的时刻，也不宣称predicate事实来源已由GPU观测。

### Q01：线程族量词和历史事实最小不动点——关闭于有界模型

事件现在有明确instance_domain：普通enter/return按线程实例，alloc/free effect为一个本侧warp operation，不能解释成32次独立资源操作，也不合并peer CTA。普通program_order为pointwise_same_thread；wait join才是all_sources_to_each_target。

独立检查每CTA全部32×31个不同线程对，共3968对：`arrive795.return(thread i)`不能推出同CTA `free.enter(thread j)`（i≠j）；同线程前后关系则保留。因此没有为了画一个family节点而给free.enter添加全warp屏障。

状态推导也不是说明性布尔标志：derive_state_facts从各resource初始fact开始，反复应用derivation all_of直到最小不动点；from事实不消耗，既有anchor历史可在较晚取得其他前提后继续参与推导，不修改partial_order。

为避免仅相信此实现，另写内存中的独立oracle，以相反transition顺序逐轮计算同一最小不动点。结果包括：

- 逐一移除4CTA的32+128个NamedBarrier贡献，共640个反例，均不能推出both，且两实现事实集完全一致。
- 逐一移除4CTA的128个wait.return，共512个反例，均不能推出整族observed。
- 只有MMA完整贡献、只有epilogue完整贡献、二者完整贡献三种历史均得到相符结果。
- 完整历史可同时保留neither/mma_only/epi_only/both/observed等非互斥事实；这是历史证明投影，不是物理当前状态快照。
- 将transition数组反转，事实结果不变；推导前后的partial_order序列也完全相同，没有暗中补时间边。

既有8项测试随后全部通过。上述证明边界是实现自己明确的once-only/单对应generation有界模型；没有把它推广为scheduler任意动态generation的运行验证，也没有把任意未满足时间前缀的anchor集合称为合法硬件trace。

### D03：v2改名后，本侧“指令已发出”仍可晚于本侧API返回——开放

v1不将完整配对物理deallocation效果硬连本侧返回，是必要的谨慎；但v2已将free.effect重新定义为 **local_dealloc_instruction_issued**。这个局部instruction-issued操作在A159–170的函数体inline asm中发生，正常离开该函数在其后；它不是稍后才采集的日志，也不是远程/配对完成事件。

当前v2图仍没有 `tmem.ctaC.free.effect(operation) → tmem.ctaC.free.return(thread i)`。四个CTA分别加入以下反向边都仍无环：

```text
tmem.ctaC.free.return(thread 0) → tmem.ctaC.free.effect(operation)
```

这允许同一侧API已返回，却尚未执行本侧已经命名为issued的指令锚点，属于新的局部模型缺口。Root独立确认此判断，并安排只补本侧operation→各本侧return：不扩展到weak allocation-result store，不扩展到peer参与或物理配对释放完成，也不增加两CTA返回全序。

因此v2可关闭原D01/D02和Q01，但本侧issue/return的D03仍待新SHA定点复验。审查者未修改数据、语义模块或测试；本节保留v2反例，不提前把计划修复记为通过。

## v3最终定点复审：D03关闭及状态汇总

本节绑定draft SHA **`3cd91a827e83be05fd397cebbfdf1d76775313c93917a36fa6e0bfba223ab633`**；语义脚本SHA `11bc4bb364c721039104aece139a1d40cda26793d91e2db4c19315cbc40a2000`；专门测试SHA `2a53c63230afabbbc74c2aef885e8aea939dc93b6aa5ce28306efced0530c217`。v2原数据在history目录仍严格匹配253091a3版本SHA。

独立JSON逐项比较确认：events、actors、resources、state_transitions、validity_rules完全未变，前248条partial_order也完全未变；时间关系仅新增249–252四条本CTA `free.effect(operation)→free.return(each local thread)`。A159–170的原函数体重新回切，新增关系约束的是本侧inline PTX已经发出之后才离开本侧函数，不是配对硬件完成，更不是日志采集的时序。

新增4条量化关系展开为128条线程边，总图现在有12436锚点、115600边。独立逐项检验：

- 四CTA各32个return，共128个反向 `return(thread i)→local issued(operation)` 均使图成环，v2的D03反例全部拒绝。
- 四CTA、peer任意32个enter与本侧任意32个return，共4096对均不存在新增的peer.enter→local.return可达性。
- 3968个不同线程的本侧arrive795.return→free.enter仍不可达；同线程程序序保留。全warp共同发出本侧指令可约束本侧return，不能反向添加free.enter之间的family屏障。
- 两对CTA的两种free.enter先后仍都可扩展为无环图；没有把allocation-result write连到allocate.return。

作者的10项专门测试也重新运行通过，且独立反例检查先于该测试结论完成。没有重跑CUDA、修改数据或补猜物理配对完成时刻。

| 审查项 | 最终状态 | 关闭依据与边界 |
| --- | --- | --- |
| D01 allocation-result write与read观察未连接 | **关闭** | v2新增8条同址/发布观察关系，全部640读者反向反例拒绝；v3未改这些关系，不等同API返回与weak store完成 |
| D02 单侧效果冒充配对deallocation | **关闭** | v2改成本侧operation记录，加显式可执行pair validity；单侧、任一缺lane、缺operation/前提均拒绝；不声称物理完成时间 |
| Q01 线程族与历史状态量词 | **关闭于声明的有界语义模型** | 显式pointwise/join/operation domain；独立固定点oracle、缺贡献/缺返回测试吻合；历史状态不消费旧事实、不添加时间边 |
| D03 本侧issued晚于本侧API返回 | **关闭** | v3仅加四CTA本侧operation→each return；128反向反例拒绝，无peer-return或alloc-return副作用 |

因此，**这一固定Dense非overlap 2SM生命周期draft的上述源码语义与模型闭合反例审查通过**。通过对象是3cd91a82版本、明确线程/operation量词、一次/单对应generation的有界执行语义及逻辑参与验收；不是任意输入history的硬件trace验证器，也没有扩展为完整accumulator流水线的动态generation证明、GPU运行/数值/性能或浏览器展示验收。v1/v2发现与关闭链均保留。

## 三个P90方法的模块人工身份补充复核

只读复核 `data/modules/dense_fp16/protocol_api_overrides.json`，SHA256 **`88b1329cc84e687b3862a68ba727c0e604d9b56cb7eaea1b5385d78df378ab68`**。结论：三条人工签名、实际访问级别及参数信息与snapshot吻合，可作为这三个模块API节点的定点身份补充。**没有修复或宣称修复全局declarations账本。**

独立使用注释/字面量感知词法token匹配物理括号：P90的`namespace cutlass`从48行至1388行；`template<int Stages_>`从1014开始，`class PipelineAsync`在1015打开，1240闭合。三条方法之前的外层scope栈都恰为该namespace和该class，没有误借其他同名类。访问级别按class直接深度的最后有效access label核对，避免内层Params等类型的public/private干扰。

| manual API ID | 原始方法签名及关键参数 | 字节范围/物理行 | 访问级别 |
| --- | --- | --- | --- |
| `tmem.api.8c30950e7a917504663d` | `void producer_tail(PipelineState state)`；state按值，无默认值 | `[39730,39787)` /1128–1129 | public |
| `tmem.api.d6fd9b3aa96f2181d2be` | `void producer_acquire(PipelineState state, ProducerToken barrier_token = {BarrierStatus::WaitAgain})`；state按值，token默认值完整保留 | `[39042,39160)` /1109–1110 | public |
| `tmem.api.5d1d398869b0beec826c` | `void producer_acquire(uint32_t stage, uint32_t phase, ProducerToken barrier_token)`；三个参数均无默认值 | `[41373,41473)` /1184–1185 | private |

三个signature_range均回切到`CUTLASS_DEVICE`及实际完整方法声明；range末尾包含函数体左括号前的一个空格，去除边缘空白后与raw_signature完全一致，起止行也吻合。结构化parameters没有漏掉引用/指针或默认值：源中state确为按值传递，private重载的stage、phase不能省略，只有public acquire的barrier_token有`{BarrierStatus::WaitAgain}`默认值。三个方法返回void，都有CUTLASS_DEVICE；没有源中存在却被遗漏的const/ref/noexcept方法限定，也没有方法自身模板参数。类模板Stages_仍由1014的owner作用域证据保留；这项补充不将源方法伪装为独立的`<4>`专用声明。

继续只读检查stage生成器及已生成`tmem_protocol.json`（本次读取SHA256为`f10705b1b16e1f57049efaaef530965efafdf758d7869787a838c6847498e7c2`）：三个节点的qualified_name均以`cutlass::PipelineAsync::`开头，override的manual_declaration和access已原样应用；节点没有entity_id/declaration_occurrence_id字段，三个旧错误ID仅出现在identity_correction.incorrect_global_entity_id诊断来源中，没有被当作人工节点的有效身份。阶段产物的staging_provenance也记录上述override SHA。此检查证明模块补充已接入该阶段数据，不代表全局owner、其他P90方法、最终网页或浏览器显示已全部验收。
