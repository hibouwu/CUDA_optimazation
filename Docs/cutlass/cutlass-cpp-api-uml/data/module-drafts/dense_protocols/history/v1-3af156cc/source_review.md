# Dense TMEM lifetime：事件列表不能直接变成 sequence

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
