# Dense FP16 2SM 代表路径：独立源码审查

## 审查结论与版本

本轮结论：**类型路径的抽查链条可以解释这一个固定配置的编译期选择；所审 contracts 版本还不能按“每条 calls 对应一个直接调用点、每个 API 节点对应真实声明”验收。** 发现了确定的重载跳层、调用位置冒充声明位置，以及把调用者前置等待归给被调用 API 的问题。异步协议文字本身已区分提交与完成，不能据此推导调用图身份也正确。

审查对象是 `dense_fp16_2sm_sm110a` 设计样本：C++ recipe 为 Sm100，设备目标为 `sm_110a`，FP16 A/B、RowMajor、float C/D、Tile 256×128×64、Cluster 2×2×1、Auto。不是整个 Adapter、Collective、Pipeline 或 824 文件的关系覆盖证明。没有执行 GPU kernel，没有作数值/性能验收；也没有浏览器交互可用性验收。

数据制作期间发生过更新，以下结论绑定到读取并在落稿前再次核对的版本：

| 文件 | 初次读到的 SHA256 | 本报告具体反例再次核对的 SHA256 |
| --- | --- | --- |
| `type_path.json` | `ce402e1a6b0fbebab1eda2e6f5ce4e73bbac60b08a69b22d58dfa61ad91e064a` | `8bee0401b8c4ef8ade045544b868e70229329e2c9a8f56bc78435490d0c3f32f` |
| `contracts.json` | `936db285ff87c43f005ebefacf9bf12a8bc69d1ac0476e0c0d58189393475a8f` | `9059aeaa1bda585f45e2a9209192384cfcbdc7aced9d87d1da9aabce7c521802` |

参考源码统一来自固定 snapshot，提交 `8f50b052e1099fb982392a622caab69b97b63128`。以下路径均相对 snapshot，行号为物理行。审查没有修改上述 JSON。作者之后修复时应追加复审版本与关闭证据，而不是将本报告当作对新版数据的否定或背书。

## 十条高风险跨层边

“核对吻合”只表示本行所述性质经源码复核，不表示整条模块路径闭合。

| # | 边 ID | 独立源码核对 | 结论 |
| --- | --- | --- | --- |
| 1 | `types.edge.auto_chooses_two_sm` | `gemm/collective/builders/sm100_common.inl:456–464` 的 `if constexpr`，静态 Cluster M=2、Tile M=256 满足整除条件后选 2SM helper。边为 `template_binds`，没有冒充运行调度。 | 核对吻合 |
| 2 | `types.edge.computed_stages` | `sm100_umma_builder.inl:84–98` 计算 stage；298–299 用 ReducedSmemCapacityBytes、SmemTileShape 调该函数；320–325 分别传入 mainloop、scheduler、accumulator 的 stage 数。新 probe 的实际输出第 16–18 行是 8/2/4。 | 核对吻合；8 不能替代另外两个 stage 轴 |
| 3 | `types.edge.traits_uses_wrapper` | `cute/atom/mma_traits_sm100.hpp:2034–2057` 的特化与 FP16 wrapper 相符；K=256/16=16，ThrID=Layout<2>。 | 核对吻合；Atom K=16、Collective K=64、Cluster 四 CTA 没有混成一个数 |
| 4 | `types.edge.unpack_calls_fma` | 同文件 2083–2091 从 A/B[0] 取 descriptor、从 D 取 TMEM 地址，再直接调选定 wrapper 的 `fma`；目标声明为 `mma_sm100_umma.hpp:563`。 | 核对吻合；这是运行调用，与前面的类型工厂分开 |
| 5 | `contract.edge.mma_gemm` | `sm100_mma_warpspecialized.hpp:702–705` 有 4 个实参 `(mma,A,B,C)`；目标 `cute/algorithm/gemm.hpp:190–194` 却有 5 个无默认值形参 `(mma,D,A,B,C)`。 | **错误直接端点**；入口应保留 83 的四参重载，88 再转五参，后续 rank 链另记 |
| 6 | `contract.edge.input_release_commit` | public `consumer_release(PipelineState)` 在 `sm100_pipeline.hpp:725`，726 调 private `(stage,skip)`；真正 arch 调用在 private 740 的函数体 745。 | **错误直接 owner**，应 public725 → private740 → arch848 |
| 7 | `contract.edge.acc_commit_binding` | public `producer_commit(PipelineState)` 在 212，213 调 private `producer_commit(uint32_t)`260；arch 调用在264。 | **错误直接 owner**，应 public212 → private260 → arch848 |
| 8 | `contract.edge.acc_release_sm0` | public `consumer_release(PipelineState)`242 在245调 helper272；arch 调用在275。 | **错误直接 owner**，应 public242 → helper272 → arch907 |
| 9 | `contract.edge.tmem_fence_complete` | `cutlass/arch/barrier.h:923–930` 实际包含 `tcgen05.wait::ld.sync.aligned`。调用点在 epilogue887，位于 release888 之前。 | 等待 API 与位置吻合；不能把这个等待搬进 release API |
| 10 | `contract.edge.free_waits_stage_release` | Kernel785–803 先 leader accumulator tail，再 peer handshake，最后调用 free；`Allocator2Sm::free`159–170 只有 dealloc 指令及配置分支，不读取 accumulator pipeline。 | **错误责任归属**；应由 Kernel 的调用点顺序/前置条件表示，不是 free 自身 `waits_for(acc_empty)` |

第 5 项的确定证据是参数个数与直接调用链。**不能另以 `is_rmem<T>` 断言 TMEM 不适配**：`cute/pointer.hpp:228` 将它定义为非 gmem、非 smem 的补集；本次没有把这一名称直觉当作反例。`contract.edge.gemm_atom` 自己标为 symbolic，不能消除前一条 `mma_gemm` 已标 `source_proven` 的错误直接端点。

## 可复现问题与应有表示

### R1：public/state 重载、private/stage 重载不能折叠成一条 calls

上表第 6–8 项是同类问题，都是源码中可精确区分的 API，而不是缺少可选解释文字。会议使用者需要知道 public API 接受 PipelineState，在哪个转发位置只传了 `state.index()`，以及真正绑定 full/empty barrier 的 API 在哪里。直接跳到 arch helper 会隐藏这个接口边界。

同类补充样本：

- `contract.edge.store_commit_arrive`：`PipelineTmaStore::producer_commit(PipelineState)` 位于 `sm90_pipeline.hpp:681`，682 调 private705，再在706调用 `tma_store_arrive()`。
- `contract.edge.store_acquire_wait`：public675 在676调 private697，private699才调用 `tma_store_wait<UnacquiredStages>()`。
- `contract.edge.input_wait_empty`、`input_full_wait_api` 与 accumulator 对应边：表达式是 `barrier_ptr_[stage].wait(phase)`，应先到 `ClusterBarrier::wait(uint32_t) const`（`barrier.h:371`），它在372转发到 static410。当前 bound wait 节点直接锚到两个形参的 static410，同样跳过一层不同 API。

修复应保留每个调用点的独立边和实参绑定；不建议把所有这些边仅改成 symbolic 后继续指向错误直接端点。若需要会议概览，可另外提供明确标为传递/概览的关系，不能复用精确 `calls` 的语义。

### R2：API 声明身份与调用位置混用，且两个调用点被合并

在所审版本中，下列节点 `kind=api` 的 path/line 实际为调用点：

| 节点 | 当前锚点实际内容 | 应区分的声明/位置 |
| --- | --- | --- |
| `contract.api.init_wait` | Kernel614：`pipeline_init_wait(cluster_size)` | 函数声明是 `pipeline/sm90_pipeline.hpp:1364` |
| `contract.api.copy_a`、`copy_b` | Mainloop620、621 的两个 `copy(...)` 调用 | 可保留 A/B 独立 callsite；具体 cute::copy 目标未完成绑定时显式缺口，不能把调用行当声明 |
| `contract.api.r2s` | Epilogue922 的 `copy(...)` 调用 | 同上，调用事件与 Copy API 分开 |
| `contract.api.alloc_arrive`、`alloc_wait` | Kernel729、870 | `NamedBarrier::arrive() const` / `arrive_and_wait() const` 声明分别在 `barrier.h:222` / 210 |
| `contract.api.dealloc_arrive`、`dealloc_wait` | Kernel793、794 | 成员声明分别在 `barrier.h:383` / 371 |

另外，`contract.edge.kernel_dealloc_arrive` 和 `dealloc_arrive_res` 将 Kernel793 的 follower-predicate 调用及795的 leader-predicate 调用合并了。两者分别位于 wait794 前后，谓词相反，正是 2SM 握手协议的重要区别。应按793、795分别保留边；它们可以指向同一声明，不应共用一个调用点身份。

`contract.api.tma_store_lambda` 确实在 epilogue766 有局部 lambda 声明，但它不是可按普通成员函数限定名查找的 `CollectiveEpilogue::store::tma_store_fn` 库 API。可表示为父函数中的 local callable/callsite owner，并独立保存源码声明位置。

`contract.api.mma_unpack` 的显示名写成 `MMA_Traits<...>::mma_unpack` 也有成员归属误导：源码2070明确是 friend 定义，实际是 namespace `cute::mma_unpack`；Traits 是声明位置及关联查找/模板绑定背景，不是成员 owner。`type_path.json` 同位置已采用正确的 `qualified_name: cute::mma_unpack`。

### R3：前置条件不能画成被调用 API 自身执行等待

`contract.edge.acc_reads_before_release` 从 `PipelineUmmaAsync::consumer_release` 指向 TMEM-read-complete，关系为 `waits_for`。源码真正等待的是 epilogue887的 `fence_view_async_tmem_load()`，之后888才调用 release；release242→272→275只发 mbarrier arrival。若实现者照图重写 release，会错误理解这个 API 自带 TMEM load completion 等待。

`free_waits_stage_release` 的 condition 虽然明确写“不是 free 函数自身调用 pipeline”，但与边的主谓语相冲突；不应让图形关系与注释分别承担相反含义。应把等待归给 Kernel 控制序列或具名调用事件，再表达 free 调用的已满足前置条件。

## 两个资源闭环

### A：A/B SMEM stage 的生产、使用与再利用

源码链可核到：Mainloop604/610先 try/acquire empty；`PipelineTmaAsync`531–538在必要时等 empty，且只有 `params_.is_leader` 对 full登记期望事务；Kernel464将 leader限制为 elected lane、MMA leader CTA、main_load 参与者。Mainloop620/621在 elected lane提交 A/B copy。低层2SM TMA wrapper（`copy_sm100_tma.hpp:113–122`、290–300）明确由两 CTA 执行，并清 peer bit，将完成事务计入 CTA0 的 barrier。不能因为高层“只有 leader expect_tx”就误画成只有一个 CTA 发出所有 TMA。

MMA侧在 Mainloop684等 full，使用保存的 `read_stage`；708释放的是提前保存的 `curr_mainloop_pipe_consumer_state`，不是已经递增的下一状态。release 经 private740后发 `tcgen05.commit.cta_group::2`；它请求硬件在被跟踪 MMA 完成后通知 empty，**函数返回并不等于 SMEM 可覆盖**。生产者下一次真正观察到相应 empty generation 完成才可复用。

stage 身份是8槽环的 index加phase。`sm90_pipeline.hpp:175–177` 的默认 consumer为0/0/0；254–259的 producer起始phase为1；204–210每绕完整个环才翻phase。`sm100_pipeline.hpp:563–570` 给 full arrival=1，empty arrival为 `(2/2)+(2/1)-1=2`。它不是“四 CTA就计4”，也不是 TMEM accumulator 的 consumer线程计数。数据的这组 stage/phase/count 叙述与所查源码吻合；R1/R2 的错误 API 图需要修复后才能把此语义核对推广为可准确浏览的调用路径。

### B：TMEM accumulator stage 与整块分配寿命

本配置 accumulator是4槽、非 overlap；Kernel751–759区分了 overlap用phase选切片和本配置用index选stage。只在 MMA leader CTA 的761–770调用 Mainloop::mma并commit accumulator full。`PipelineUmmaAsync` 的 full绑定在262，mask通过153–159和95–109按 AtomThrShape2×1范围计算，所以2SM peer pair不是 Cluster2×2的全部四 CTA。

consumer侧 epilogue866等 acc full，882提交 TMEM→register读；最后需要释放时887等先前 TMEM读完成，888才发 empty arrival。Kernel528–529初始化 full计数1、empty计数 `size(AtomThrShapeMNK{}) * NumEpilogueThreads`，本配置为2倍，而不是input empty的2个MMA完成通知。读完 TMEM后 D 的 register→SMEM、TMA store 还可继续，因此 accumulator复用不要求全局 D 写完。

整块 TMEM分配与stage再利用不是同一个环。Kernel727 allocate后经 NamedBarrier729向epilogue870发布base pointer；最后783归还的是 allocation permit，不是释放存储。非 overlap路径785–795由leader先排空acc stage，再与peer做两次不同谓词arrival和一次wait，最后803调用free。Allocator2Sm135/159/174分别对应 alloc/dealloc/relinquish；三者不可合并。这里的源顺序有证据，但 R3 把等待责任贴到 free API自身，故图仍需修复。

异步完成语义另外核对了 [NVIDIA PTX ISA 9.0：tcgen05.commit](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#tcgen05-instructions-tcgen05-commit)：commit是异步的，系统在被跟踪操作完成后通知mbarrier。输出侧抽查亦确认源码 `copy_sm90_tma.hpp` 的 wait使用 `.read`；[cp.async.bulk.wait_group](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group) 的 `.read` 只承诺描述符/源读取完成，不等于目标D写入完成。因此 contracts 已有“不以store提交/read等待冒充D完成”的边界是必要且正确的，不应在修图时丢掉。

## 位置抽查、范围与后续验收

`type_path.json` 的25个节点已逐一打开对应源码声明上下文，未发现 callsite冒充声明。新probe保存的 `host_type_dump.stdout.txt:3–19` 可复核具体策略、Atom形状、CTA形状和8/2/4；本次只读取既存产物，没有重跑编译，更没有将静态 probe作为GPU执行证明。

按追加请求，对 `host.json`（SHA256 `de23553f2e0b21ab5bd0315c4c007fd5dda6f69d7205948fbbc9455de7dda1e0`）库内 API/type声明锚点也逐一核对。它们落在相应 template/返回类型/函数名的真实声明区域，未发现本报告R2这类调用点冒充声明的问题。由于 Host数据由本审查者制作，此项只算补充位置自检，不冒充独立内容审查。示例 main不是824范围库API，CUDA外部API的空声明位置也不是库内遗漏。

已有 copy/gemm完整重载链、scheduler内部、epilogue callback、数值化mask/事务数、其他配置等 scope boundary仍然存在；本报告没有把诚实标明的范围外能力重复记成新bug，也没有把 boundary当成最终全库目标的豁免。修复R1–R3及具体arity错误之后，还需按新版SHA复核，检查实体身份、调用点唯一性和界面能否让会议使用者走完这一条路径。源链接/SVG能生成，不自动等于实际浏览器交互验收通过。

## 追加定点复审：S01–S05 的已知反例关闭

本节追加，不覆盖上面的旧版反例。复审读取结束时数据 SHA仍为：

- `contracts.json`：`ee96d5ae60428523d390529fbf35bb8468f8a8849d2fd94329e5a9fc505b475a`。
- `gemm_dispatch.json`：`d4e3a410ea1d4f6137f3422b78e9c5627668c4d07005e45b458d082abe3624ed`。

为对应本次修复清单，将原报告问题归为五组：S01=R1与gemm错误直接重载；S02=R2的声明/调用锚点身份；S03=重复callsite独立性；S04=R3的caller前置顺序；S05=friend身份。**这五组下面明确列出的反例在该版本关闭；不是整个contracts或所有同类调用的覆盖通过。** 复审没有修改data、重新编译或执行GPU。

### S01：直接重载链——列明反例关闭

逐项回切了16条重载/转发边的 `source_expression` 与snapshot原文，并核对其source和target的声明位置：

| 源API声明行 | 直接目标API声明行 | 文件/用途 |
| --- | --- | --- |
| 430 → 531；485 → 618 | State→stage/phase/token | `sm90_pipeline.hpp` 的input acquire/wait |
| 1110 → 1185；1155 → 1228 | State→stage/phase/token | 同文件accumulator acquire/wait |
| 371 → 410；553 → 588 | 成员→带地址的static重载 | `barrier.h` 的wait与arrive_and_expect_tx |
| 725 → 740 → 848 | public State→private stage/skip→arch | input release；前两节点在sm100_pipeline，末节点在barrier.h |
| 212 → 260 → 848 | public State→private stage→arch | accumulator commit |
| 242 → 272 → 907 | public State→private helper→arch | accumulator release |
| 681 → 705 → 1225 | public State→private stage/count→TMA commit | PipelineTmaStore及copy_sm90_tma.hpp |
| 675 → 697 → 1248 | public State→private stage/count→TMA wait | 同上 |

原4个 `barrier_ptr_[stage].wait(phase)`端点现在是资源/调用绑定，`instance_of`连接成员371，再由真实372转发到static410；没有把绑定行当作函数定义，也没有直接跳过成员重载。

gemm入口已改为83的四参重载；详细part给出真实链：`83 -(88)-> 275 -(298)-> 142 -(148)-> 190 -(197)-> MMA_Atom::call94 -(104)-> cute::mma_unpack2072`。独立读取了原始 `readelf_debug_info.stdout.txt`，重建DWARF DIE层级，逐个解析 `DW_AT_abstract_origin`，取最近的inline/subprogram父结点，再比较其实际声明行、调用行、目标声明行与完整linkage_name。五组结果分别为 `(83,88,275)`、`(275,298,142)`、`(142,148,190)`、`(190,197,94)`、`(94,104,2072)`；不是只读取debug_index的结论字段。

源码275的rank条件、288–298的8字节/row分支、142的D&&形参以及148的命名D左值转发，与保留的Host类型日志一致。原先“83或275直接到190”的捷径已移除。运行既有 `test_dense_gemm_dispatch.py`，6项离线回归通过；其作用仍限于静态产物、原文、类型绑定与DWARF链，不证明GPU执行。

### S02：9个锚点与人工声明——列明反例关闭

| 原问题节点 | 新表示及源码核对 |
| --- | --- |
| init_wait | 真正函数声明sm90_pipeline1364，Kernel614单独保存为callsite |
| copy_a | binding在Mainloop620，instance_of为copy.hpp541临时目标入口 |
| copy_b | binding在Mainloop621，独立于A，入口仍为541 |
| r2s | binding在Epilogue922，入口为541；不是cute::copy声明 |
| alloc_arrive | NamedBarrier成员声明barrier.h222，Kernel729另作调用证据 |
| alloc_wait | NamedBarrier成员声明barrier.h210，Kernel870另作调用证据 |
| dealloc_arrive | ClusterBarrier成员声明barrier.h383；793/795分别绑定 |
| dealloc_wait | Kernel794是binding，instance_of为barrier.h371成员wait |
| tma_store_lambda | source_entity_kind/manual_declaration均为lambda，766–800定义与761外层lambda、store父API分开，没有编造类成员限定名 |

对 `pipeline_init_wait` 另作独立的物理范围验证：使用候选扫描器的注释/字面量感知词法token，而非声明提取器的owner，匹配到 `namespace cutlass` 在48行的左括号与1388行的右括号；1362声明之前唯一外层括号正是这个namespace。人工signature_range回切后与raw_signature一致，参数字节区间也匹配原文。该节点没有entity_id/declaration_occurrence_id，且 `identity_correction.reuse_ledger_entity_id=false`。因此这项人工补充可纠正此模块节点的错误global身份；**不等于全局声明账本中原错误已经修复。** lambda签名字节范围也已回切匹配。

### S03：4组重复callsite——指定组关闭

下列组的两个位置分别有独立边/绑定，源码表达式逐行匹配，且条件区别已保留：

- Mainloop `producer_try_acquire`：604的初次probe、617在递增state后的下一stage probe。
- Mainloop `consumer_try_wait`：671的初始token、696在递减K并重算skip_wait后的下一stage token。
- Kernel deallocation arrive：793的`not is_mma_leader_cta`、795的`is_mma_leader_cta`；wait794仍在中间。
- Kernel deallocation wait：794属于非overlap且有peer路径；799属于overlap的else路径，当前Dense未选。799没有被错误附加has_mma_peer_cta guard。

这4组核对不代表全部重复调用清零。**本轮另外观察到一个仍未闭合项**：`contract.edge.kernel_load` 仍用一条calls同时收录Kernel639–645的prologue load与653–659的剩余K load，condition也写“两次调用”。两个调用的末两实参分别为 `(k_tile_iter,k_tile_prologue)` 和 `(k_tile_iter_next,k_tile_count-k_tile_prologue)`，中间还更新了producer state；应分别建边及精确callsite。它不推翻上面4组的关闭，但阻止将这次关闭扩展成“contracts全部调用点独立”。已向Root报告，当前审查者未改data。

### S04：caller precedes——原等待归属错误关闭

`acc_reads_before_release` 已改为 TMEM读完成事件→release 的 `precedes`，并保存Epilogue887–888调用者位置；真正 `wait::ld` 仍由fence API承担。`free_waits_stage_release` 已改为acc_tail→free的 `precedes`，condition说明这是Kernel非overlap leader顺序，follower经peer握手继承前置条件。另有dealloc wait→free、wait→leader arrive等caller顺序边。

回切Epilogue885–889、Kernel785–803以及Allocator2Sm159–170后，原先release/free自身 `waits_for` 的反例关闭。保留的组合source_expression此时是明确的caller顺序证据，不再自称单处普通C++调用。提交/完成、stage复用/整块TMEM释放、allocation permit/deallocation仍分开；本节不把协议文字正确升级成运行正确性。

### S05：friend身份——关闭

`contract.api.mma_unpack` 现为 `qualified_name=cute::mma_unpack`，source_entity_kind为friend_function_template；Traits存入declared_in_traits，仅表示声明/模板绑定背景。snapshot2070明确含friend，2072是函数名；没有再把它当成Traits成员方法，与type_path身份一致。

本次不复验新SVG/导航翻页，不复用旧静态图报告为新版UI背书。浏览器交互仍未验收。剩余copy深层分派、scheduler、epilogue外围、硬件运行和数值/性能边界仍按各part的明确issues保留；上面的Kernel两次load合并是另外一个具体数据残留，不能只用宽泛scope说明代替它。

## 再追加复审：两次load与相邻同步调用拆分关闭

本节只针对 `contracts.json` SHA256 **`cad206532636d5ac74994e9d3ea03b568d067faa8666747e29df9d774e1bde7a`**。模块数据未改，旧版残留记录保留。

`contract.edge.kernel_load` 与新增 `contract.edge.kernel_load_rest` 现在各自对应一个源码调用点：

| 边 | 精确调用位置 | 末尾两个实参 | 对应state更新 |
| --- | --- | --- | --- |
| kernel_load | Kernel639–645 | `k_tile_iter, k_tile_prologue` | 646：`mainloop_pipe_producer_state = mainloop_producer_state_next;` |
| kernel_load_rest | Kernel653–659 | `k_tile_iter_next, k_tile_count - k_tile_prologue` | 660：`mainloop_pipe_producer_state = mainloop_producer_state_next_;` |

独立回看Kernel624–675：两次load都位于同一次work-tile循环体内，第二次使用第一次返回的iterator与646更新后的producer state；它们之间648–651的条件load-order交接不表示prologue只属于首个work tile。下一work tile的获取和赋值是在664–670，发生于两次load之后。第二次剩余K可为0，也不因此从调用点账本中删除。原“S03后另报”的两次load合并残留在此版本关闭。

两个相邻同步表达式也已从直接calls中剥离：

- `tma_lambda_fence` 的唯一调用为Epilogue768的 `cutlass::arch::fence_view_async_shared();`，769的 `synchronize();` 只留作调用者相邻上下文，不再合成同一调用表达式，也不声称是fence API内部调用。
- `kernel_alloc_arrive` 的唯一调用为Kernel729的 `tmem_allocation_result_barrier.arrive();`，728的 `__syncwarp();` 只作为前置上下文，不再合并成arrive调用。

四条边的callsite字节区间、起止物理行、raw_source_expression均回切到snapshot匹配；格式归一后的表达式与edge source_expression一致。两条state_update原文逐行匹配。上述断言通过，结论限于这四条边的准确定位与分离；不把相邻同步“未合并”自动宣称为其完整API关系链已经覆盖。

本次仍未看新SVG，等待生成后的独立静态展示复核。浏览器交互、GPU运行和全库覆盖结论保持未验收/未完成。
