# S11 全家族短正确性接口草案

状态：待独立 A 审查。本文不授 GPU、B3 或性能资格；旧源码、旧合同、既有编译和归档保持原样。

## 原 13 配置的实际目标覆盖

以 `synchronization_lowering_v3.json` 的13个配置为有限集合，每配置两个独立profile进程、seed=3，不运行预热或pilot。现有通用policy每进程最多4次target，因此固定拆为measured三launch与paired auxiliary二launch，保持原计划65次GPU工作量且不改core。

| profile | launch index | 实际函数 | 传入轮数 | 完成阶段 | 核对问题 |
|---|---:|---|---:|---:|---|
| measured | 0 | `sync_measured` | 1 | 8 | 实际测量分支、第一轮展开 |
| measured | 1 | `sync_measured` | 2 | 16 | 目标外层回边 |
| measured | 2 | `sync_measured` | 33 | 264 | 重复phase/token复用 |
| paired auxiliary | 0 | `sync_correctness` | 2 | 2 | 非均匀token、邻线程消费、消费后再次同步 |
| paired auxiliary | 1 | `sync_arrival` | 1 | 1 | 逐线程到达/离开与固定工作偏斜 |

共13配置、26进程、65次target。measured的target_iterations=[1,2,33]、input_profiles仅一个 `measured_nonuniform_lcg`；aux的target_iterations=[2,1]、input_profiles仅一个 `paired_auxiliary_nonuniform_tokens`，避免两role×两长度的隐式笛卡尔积。aux角色由固定launch_index映射为correctness/arrival，适配器必须核对对应顺序/长度。

actual `target_launches` 只使用既有launch_index/iterations/input_profile/threads/blocks，不能新增kernel_role。profile里的kernel_roles_by_launch_index是受审静态映射，未新增运行行字段。原三个函数和device body保持；仅新增host validate-only分支，旧正式入口语义不变。

`resource_identity` 保持既有字段；primary为measured profile的measured函数，或aux profile的correctness函数。`resource_identity.extensions.role_resources` 明确绑定三角色分别的实际函数symbol/寄存器/static/dynamic/local/occupancy；`role_target_identities` 绑定对应完整B2指令及128bit编码。三个函数不假设相同资源，aux中的arrival不可借用correctness的资源。三角色所有实际B2/attribute均须在该case每个profile核实。role_resources每角色固定字段为kernel_symbol/registers_per_thread/static_smem_bytes/dynamic_smem_bytes/local_size_bytes/occupancy_limit_ctas_per_sm，dynamic=0、local=0；role_target_identities每角色固定kernel_symbol/instruction_count/target_identity_sha256，digest由完整PC/指令/predicate/operand/两个64bit编码的canonical JSON生成并绑定实际B2。profile描述资源schema，不增加core字段。

## 完整数值及完成证据

每 launch 保存profile内独立编号的 `sync_threads_<i>.u32le`，形状 `[threads,14]`：依次为 value/errors/timeout/smid 四个 uint32，completed/attempts/checked/arrival_cycle/departure_cycle 五个 uint64 的 low/high uint32。另存 `sync_stamp_<i>.u32le`，形状 `[9]`，顺序为 start_ns/stop_ns/start_cycle/stop_cycle 的 low/high，再跟 smid。全值文件均有 SHA256；记录线程索引由数组位置确定。每次 launch 对可预知字段使用期望值反相 poison；SMID用UINT32_MAX，未知wait尝试数和时间戳用UINT64_MAX并由后述严格范围拒绝sentinel。不可只把输出初始化为统一 0xff 后漏检部分字段。

每线程初值为 `(3+17*tid) mod 2^32`。固定工作选中线程每阶段执行 256 个依赖 MAD，CPU 用独立仿射复合计算最终 value；未选中线程保持初值。完成阶段数必须匹配上表。errors/timeout 为零，全部线程 SMID 与该 CTA stamp 相同，SMID 非 UINT32_MAX；不要求物理 SMID 连续或小于 SM 数。

`sync_correctness` 的 checked=2，其余 checked=0。mbarrier 的 attempts 至少为阶段数，correctness 因每阶段两次 arrive/wait，至少为 4，并严格小于UINT64_MAX；其余模式 attempts=0。Purpose0/1 的 arrival/departure 字段应为零；Purpose2 每线程 0<=arrival<=departure<UINT64_MAX。stamp 四个时间戳完整读取，四字段均小于UINT64_MAX，止点不早于起点；短窗口零时长不否决数值证据。不从 arrival 数组反推正式每阶段等待时延；固定工作不保证编译后一定出现预期偏斜。

producer/consumer 路径保存 errors/checked 以及完整线程结果；它没有保存每次消费的原值，故只能由被冻结 host 对完整消费检查的实现和零 errors 证明该路径，不能声称离线从原消费数组重算。测量分支的完整最终 value 可以离线逐线程重算。

fence 发出后参与者数据消费仍由显式 CTA barrier 完成，不将 fence 或零 errors 当作异步复制、TMA、WGMMA 完成证明。

## 编译和影响范围

先核对原真实编译的 39 个 measured/correctness/arrival 函数完整 SASS、资源和控制流，再允许短 GPU。原已审 measured baseline 可复用；尚未审的 auxiliary 函数必须补充 B2。新增 host 分支需在当前 CUDA12.9/sm90a 编译中证明 39 个 device 指令和资源未变，否则按实际变化范围重新审查。mbarrier measured 保留 18 个静态 globaltimer 读取位置；超时检查及共同退出不得为配合旧计时审计而删除。

原 `warp_t32_aligned`/`warp_t32_fixed_work_skew` 没有 WARPSYNC，继续作为 nonexportable compiled logical phase 对照。编译消除不等于硬件不支持，也不能把两个点导出为原生 warp barrier 服务。

## warp 候选的实际降低与最小修订

job730803、CUDA12.9/sm90a 的三个 compile-only 目标已留存于 `implementation/target-compile-warp-candidate-a`。`warp_mask_bare_v1` 和 `warp_shared_pair_v1` 均为0个WARPSYNC、仅首尾2个BAR.SYNC；`cta_shared_pair_v1` 有18个BAR.SYNC，包含循环展开的16个。原源码、完整SASS、资源和负面降低记录保留；runtime mask 与共同路径 shared 消费仍未得到原生warp服务证据，不能声称完成覆盖，也不能据此判断硬件不支持。[CUDA12.9.1 Binary Utilities 的 Hopper 指令表](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-binary-utilities/index.html#hopper-instruction-set)明确列有 WARPSYNC；这里的缺口是可审查且真正保留目标的探针形式。

新增 compile-only `warp_sync_divergent_pair_v2.cu`，只验证两个目标：warp跨分支共享生产/消费pair，与完成相同数据任务的CTA对照。warp版本使用一个明确的PTX分支块：lane0–15和lane16–31各自在自己的分支内写shared token、执行第一次full-mask warp同步、读取对方half的peer=tid XOR16、执行第二次full-mask同步以防下阶段覆写。每个参与者每pair严格执行两次同mask同步，不退出或缺席。mask来自kernel参数，host只允许UINT32_MAX和完整32线程CTA；分支未加`.uni`。

[token模式]：`((seed+17*tid+(tid>=16?324508639:0)) mod 2^32) XOR (((phase+1)*2246822519) mod 2^32)`，phase从0开始。不同half的生成内容和读取地址可被CPU区分。CTA对照在共同路径执行两个 `bar.sync 1`，不把`.aligned` CTA barrier放入不同lane分支；它与warp候选的控制流开销可能不同。

[PTX ISA 8.8 的 bar.warp.sync 语义](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-warp-sync)规定mask成员必须执行同mask同步，并提供参与线程间内存顺序；同指令位置和convergence限制明确针对sm_6x及以下。本文对sm90跨分支对应同步的合法性是据此作出的推断，要求独立A/B2核实，而不将C++分支源码本身当作实际WARPSYNC证据。实际SASS还须证明两个half均有完整生产→同步→消费→同步路径，并保留合法的线程调度/控制流。若继续被消除，停止该候选准入并保留负面证据，不推进数值/性能结论。

job731009的真实compile-only证据已收回 `implementation/target-compile-warp-divergent-a`。warp目标保留32个 `WARPSYNC.COLLECTIVE` 静态位置和1个循环结束后的 `WARPSYNC.ALL`。32个COLLECTIVE位于主体EXIT之后的outlined helper，由循环中32个 `BRA.DIV` 位置进入；24个helper返回调用后的下一操作，另8个直接回到同pair的BSYNC共同merge。PC位于主体EXIT之后不等于不可达；静态33位置也不等于每轮动态执行33次。每参与线程每pair仍是源定义的两个对应同步、每迭代八个pair，实际服务需随后完整短正确性和独立控制流复核。

warp目标22个寄存器，CTA对照25个寄存器，均136 B static SMEM、零stack/spill。CTA目标保留18个BAR.SYNC（循环16个、首尾2个）。不同寄存器和控制流资源必须作为比较条件；额外WARPSYNC.ALL属于实际排空/控制开销，不能隐藏或当作pair内裸服务。当前仅有编译事实，没有GPU数值或性能资格。

拟定最小补充矩阵为两个有限点：`warp_divergent_shared_pair`、`cta_cross_half_shared_pair`，均32线程、单CTA，seed=3，每迭代8个pair。先固定1/2/33短迭代，共2进程6次target。每线程保存初值、最后一次对半peer消费值、全部pair消费checksum和完成pair数，共128个完整uint32输出，另保存两个同CTA clock64端点的low/high，共4个word。全字段exact或完整时钟顺序检查，输出使用反相poison/无效时间sentinel。formal2048迭代只是后续候选，本次草案不授权。

单位为 `clock64_cycle/producer_consumer_pair/CTA`。每pair包含两个同步、必要共享访问、checksum及分支/循环；不能除二冒充单barrier裸时延，也不能以warp与CTA时间相减得出精确WARPSYNC成本。未同步共享读写不作为性能对照；只有合法且完成同一数据任务的实现可比较。

## A 接受条件

有限坐标/阶段/字段和 CPU 参考可以独立重建；实际39目标和原三个及新两个feasibility候选目标的 B2范围明确；对 compiled-away warp、fence 和到达偏斜的结论受限；旧文件/已签记录不改。A 通过后才实现 ABI、host 与审计器，B2和全部短正确性通过后再接正式运行器。
