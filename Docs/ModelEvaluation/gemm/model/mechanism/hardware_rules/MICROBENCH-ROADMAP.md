# GH200 模块级微基准与模型接入

更新：2026-10-18。范围为单 GH200 GPU、本地 HBM 分配及 SM90a 计算路径，首先服务 cfg_b S6/S4。本文维护研究方向、已有证据、待辨识关系、实验设计和模块接入；当前硬件假设与成绩仍在各路径页和实验记录维护。首版九阶段及组合已交付；九方向保留为本轮覆盖与分工，当前推进顺序见[PLAN §1](../../../PLAN.md#1-近期顺序)。免费ROMEO上的已登记GPU测量已获许可；阶段1 L2开发完成，六形式按单调反例收口，留出未采；阶段2驻留A开发及留出完成，有限长度外推通过，两解释仍重合；后续Q05每组完成滞后形式通过两wait冻结长度验证，固定截距形式未通过；B/C零额外控制具体形式停止，D/E完成开发数值与路径观测，尚无新可迁移价格。L1原开发及8留出完成，快复用限定规则通过；Q02行距与CPU源定位、Q07四组织开发窗口已完成，见§6。本轮首版有限交付已齐；原完整模型失败保持。执行范围见 [GOALS §5.4](../../../GOALS.md#current-scope)，当前登记见 [PLAN](../../../PLAN.md)。

## 1. 九个方向与一个组合层

以下九个方向与组合层是 2026-10-17 这一轮的实验分工：按可独立控制的服务路径分工，再用组合层检验它们共同执行的结果。研究导航统一按[16 个研究问题](../../../experiments/gh200_sm90/README.md#research-questions)组织。方向不对应九个物理硬件块。GPU L1 cache 与 SMEM 分别研究；HBM 是存储资源，global 是地址空间；本文的 cache 名称与[模型 L0–L4](../../README.md)分层分别使用。

| 方向 | 对象与核心问题 | 已有入口 | 首版执行阶段 |
|---|---|---|---|
| 1. HBM 外存服务 | 大工作集读、写及混合请求的整卡联合能力 | [EXP-13](../../../experiments/gh200_sm90/Q03-source-service/EXP-13-global-rw.md)、[HBM服务](HBM-SERVICE.md) | 阶段5；复用已有外存对照 |
| 2. L2 cache 与共享源服务 | 热数据吞吐、地址状态、共享／独立源、跨 SM 竞争 | [L2服务](L2-SERVICE.md)、[R10](../../../experiments/gh200_sm90/Q03-source-service/R10-layout-cache.md)、[B01](../../../experiments/gh200_sm90/Q03-source-service/B01-bandwidth-cache.md)、[TMA 共享服务](TMA-SHARED-SERVICE.md) | 阶段1；开发完成，六形式停止、留出未采 |
| 3. L1 cache 与普通 LSU | global/local→寄存器的缓存、请求形成与返回 | [EXP-13](../../../experiments/gh200_sm90/Q03-source-service/EXP-13-global-rw.md)、[EXP-18](../../../experiments/gh200_sm90/Q08-register-residency/EXP-18-occupancy-aux.md) | 阶段8 |
| 4. SMEM 访问服务 | LDS/STS、矩阵访存、广播与 lane 布局，及路径间竞争 | [EXP-09](../../../experiments/gh200_sm90/Q07-smem/EXP-09-smem.md)、[EXP-10](../../../experiments/gh200_sm90/Q07-smem/EXP-10-ldmatrix-shfl.md)、[SMEM服务](SMEM-SERVICE.md)、[m11](M11-SMEM.md) | 阶段6；保留m11反例 |
| 5. 寄存器与操作数依赖 | 源读取、reuse、结果就绪与返回竞争 | [寄存器服务](register_service.md)、[R01](../../../experiments/gh200_sm90/Q08-register-residency/R01-readiness.md) | 阶段7 |
| 6. 计算指令服务 | mma.sync／WGMMA 的依赖延迟、接收与持续吞吐 | [EXP-06](../../../experiments/gh200_sm90/Q11-other-compute/EXP-06-mma.md)、[EXP-07](../../../experiments/gh200_sm90/Q05-wgmma/EXP-07-wgmma.md)、[R00](../../../experiments/gh200_sm90/Q05-wgmma/R00-anchor-target.md) | 阶段2；A长度留出通过，B/C具体形式停止，D/E开发诊断完成 |
| 7. 异步搬运服务 | TMA／cp.async 的发出、源服务、目的投递与排空 | [EXP-12](../../../experiments/gh200_sm90/Q02-tma-request/EXP-12-cp-async.md)–[EXP-16](../../../experiments/gh200_sm90/Q02-tma-request/EXP-16-tma-pipeline.md)、[TMA 共享服务](TMA-SHARED-SERVICE.md) | 阶段4；先用cfg_b原生tensor请求 |
| 8. 同步、可见性与释放 | mbarrier、group wait、proxy fence、等待恢复及槽代次 | [EXP-11](../../../experiments/gh200_sm90/Q04-sync-slot/EXP-11-sync.md)、[R13](../../../experiments/gh200_sm90/Q02-tma-request/R13-async-retirement.md)、[R14](../../../experiments/gh200_sm90/Q06-handoff-output/R14-stage-handoff.md) | 阶段3；复用阶段2短链 |
| 9. SM 发射、驻留与调度 | ready warp 的推进、混合发射、配额、CTA 活动及启动分布 | [联合服务](joint_service.md)、[驻留与配额](residency.md)、[R19](../../../experiments/gh200_sm90/Q01-scheduling/R19-critical-cta-tail.md) | 阶段9；先选cfg_b相关竞争 |
| 组合层 | 搬运→就绪→计算→释放→输出的短闭环及完整 GEMM | [程序图](../program/cutlass_program_graph.md)、[组合](../composition.md) | 阶段10；各模块先保留可接入端点 |

频率、计时和观测扰动是各方向共同的实验条件，复用[频率模块](FREQUENCY-MECHANISM.md)与[测量协议](../../../experiments/gh200_sm90/access_rules/RESEARCH-GUIDE.md)，本地图不另开独立频率课题。cluster／DSM／多播留在方向 2、4、7 的扩展路径；cfg_b 的 cluster 为 1×1，首轮不增加这些条件。

原[微架构审阅](MICROARCH-REVIEW.md#4-机制覆盖与缺口)的 M01–M25 是历史覆盖索引，按本地图分工复用：

| 方向 | 对应旧索引 |
|---|---|
| HBM | M20；翻译对照引用 M18 |
| L2 cache | M17、M19 的 L2 互连部分；翻译对照引用 M18 |
| L1 cache／LSU | M10、M18；local／spill 请求引用 M08；global 原子按 M21 另选 |
| SMEM | M11；shared 原子按 M21 另选 |
| 寄存器 | M01–M04 |
| 计算 | M06、M15 |
| 异步搬运 | M12–M14；DSM／多播引用 M19 |
| 同步与释放 | M16，依赖就绪引用 M01 |
| SM 发射／驻留／调度 | M05、M07–M09、M24 的实际活动与启动部分 |
| 组合与共同协议 | M22、M23；程序工作列表引用 M24；频率与观察引用 M25 |

## 2. 从微基准结果到模型规则

沿一条实际路径组织需求与服务：

```text
程序图给出请求、操作、依赖与槽代次
  ├─ 普通 load/store → LSU → 缓存服务（按 cache policy）→ 必要时 HBM → 寄存器／目的内存
  └─ TMA 请求 → 源端 L2／HBM 服务 → 目的 SMEM 投递 → full 完成
                                                       ↓
                                               消费者等待返回
                                                       ↓
                                      WGMMA 提交／完成 → 输入槽释放
                                                       ↓
                                 输出准备／store／源复用／最终排空
```

操作字段、资源 ID 与事件定义沿用[共同接口](../../common/interfaces.md)，参数证据与研究判定沿用 [GOALS §3.3–3.5](../../../GOALS.md#33-硬件服务规则)。模块实验分别交付以下内容，而不只交付一项最大吞吐：

| 要交付的关系 | 对应实验 | 模型使用方式 |
|---|---|---|
| 依赖就绪与完成间隔 | 固定生产者／消费者的依赖链；异步指令包含对应完成等待 | 给完成或指定消费者可用事件定价 |
| 接收与持续服务 | 独立链／请求并发增加到吞吐平台，保留工作量和资源占用 | 给服务间隔、条件吞吐和并发限制提供依据 |
| 地址与状态响应 | 固定请求多重集合，改变共享、顺序、复访、布局或先前写入 | 给命中、合并、替换和目的服务选择提供规则 |
| 多流联合能力 | 匹配独立、串行和重叠窗口，改变比例与相位 | 决定共享约束、仲裁或可兑现的服务重叠 |
| 有限链建立与排空 | 短／长 Ktile 及同一程序的首段、后段、尾部 | 连接局部服务与完整阶段，避免用稳态率代替整个短链 |

每个结果绑定相应操作形状、布局、依赖／独立流、并发、资源分配、缓存初态、输入类型、计时边界和时钟口径；字段及采样方法分别链接[共同输入](../../common/inputs.md)与[RESEARCH-GUIDE](../../../experiments/gh200_sm90/access_rules/RESEARCH-GUIDE.md)。已有实验未覆盖的关系保持为工作假设，当前参数仍从原绑定文件读取。

## 3. 各方向的具体研究

### 3.1 HBM：整卡外存读写联合服务

**已有证据：** [EXP-13](../../../experiments/gh200_sm90/Q03-source-service/EXP-13-global-rw.md)已覆盖大工作集纯读、纯写、七种读写比例及依赖／独立复制。工作集按实际 grid 步长舍入；读含 checksum，写含 device fence，完整窗口由[探针](../../../../../../microbench/gh200_resource_campaign/probes/global_duplex.cu)定义。

**要辨识：** 请求并发达到什么条件后进入平台；同一读写比例下交错粒度、地址分布和并发流是否改变联合服务。已有七种比例直接复用。

**最小补测：** 先选纯读、纯写、1:1，逐步增加实际活跃 SM／独立请求至平台；固定平台附近组织，再比较一种逐段与一种交错读写。大工作集与 L2 驻留对照使用同一访问体；内存分配方式、输入内容和页跨度固定，地址翻译异常另作成对定位。

**交付与接入：** 整卡共享的方向联合服务曲线、条件往返窗口及读写计账。接入 `gpu.memory.external` 的 miss／写回服务，统一处理所有 CTA 的需求；源端已经包含的 HBM 等待随原服务一同替换。

### 3.2 L2 cache：饱和曲线、地址状态与共享范围

**已有证据：** [EXP-13](../../../experiments/gh200_sm90/Q03-source-service/EXP-13-global-rw.md)有小工作集 `.cg` 请求吞吐；[R08](../../../experiments/gh200_sm90/Q01-scheduling/R08-waves-l2-reuse.md)、[R10](../../../experiments/gh200_sm90/Q03-source-service/R10-layout-cache.md)、[B01](../../../experiments/gh200_sm90/Q03-source-service/B01-bandwidth-cache.md)有复用与布局对照。[TMA 共享服务](TMA-SHARED-SERVICE.md)还保留源次序、局部域、先前输出写入及独立／共享源反例。

**要辨识：** 热驻留数据的接收与返回服务如何随请求并发和活跃 SM 变化；共享地址是否合并，何时停止合并；写入和复访次序怎样改变后续命中；不同访问者是否竞争同一服务范围。

**最小补测：** 先用每 SM 有明确地址集合的 `.cg` 向量读扫请求并发和活跃 SM；在平台附近固定并发，做同源／独立源配对。纯写和1:1混合沿用方向1的访问体，记录加载消费者完成、store后device fence与CTA会合边界；TMA输出另区分源读取与完整写完成。随后只选一组 cfg_b 原生 TMA 请求，与普通 load 路径作匹配对照。地址状态候选必须回放既有源重排反例；footprint 相同仍保留复访与请求到达次序。

**交付与接入：** L2 条件服务曲线及命中、pending fill、订阅合并、替换、写入影响的状态规则。通过源服务接口提供每个请求的源就绪事件；共享范围用候选服务域表达。NCU 可用时核对所在层级的 hit／sector／DRAM 读写；没有计数器时，交付注明缓存准备的请求服务曲线，物理流量维持现有资格。

### 3.3 L1 cache 与 LSU：普通指令的请求和返回

**已有证据：** [EXP-13](../../../experiments/gh200_sm90/Q03-source-service/EXP-13-global-rw.md)的小工作集 `.ca/.cg` 差异、[EXP-18](../../../experiments/gh200_sm90/Q08-register-residency/EXP-18-occupancy-aux.md)的 local／spill 窗口；[原8留出](../../../../../../results/gh200_resource_campaign/access_rules/Q13-l1-lsu/L1-01-minimal/20261017-Q13-l1-minimal-002/EXECUTION.md)的快复用条件规则通过，实际物理cache容量未知。

**要辨识：** 单 SM 复访容量、独立加载数、warp 数、访问宽度／lane mask 怎样影响接收、返回和消费者就绪；carveout 改变缓存与驻留后的联合响应。

**最小补测：** 每 SM 私有地址，固定访问体做 `.ca/.cg` 配对；工作集跨过本 SM 可见容量，分别测依赖链和独立加载。基准组织确定后只选两档 carveout，核对实际寄存器、SMEM 和 CTA 活动；local／spill 与 global 的请求来源分别保留。原子和地址翻译只有在目标路径使用或出现对应反例时加成对测试。

**交付与接入：** 普通 LSU 请求需求、条件 L1 命中／失效与返回规则。用于普通 load/store、local 和控制数据访问；TMA 沿方向7的路径服务。spill 是该路径新增的程序需求，驻留变化由方向9处理。

### 3.4 SMEM：按访问路径辨识服务

**已有证据：** [EXP-09](../../../experiments/gh200_sm90/Q07-smem/EXP-09-smem.md)、[EXP-10](../../../experiments/gh200_sm90/Q07-smem/EXP-10-ldmatrix-shfl.md)、[R03](../../../experiments/gh200_sm90/Q07-smem/R03-access-demand.md)覆盖标量／向量、广播与矩阵访存；[m11](M11-SMEM.md)两次冻结留出均未通过，保留 lane／word 组织反例；[联合服务](joint_service.md)记录普通 STS、WGMMA 与 TMA 的干扰。

**要辨识：** 地址与 lane 需求如何决定服务分批、广播及依赖完成；LDS/STS、LDSM/STSM、WGMMA 取数、TMA 目的写入分别使用怎样的条件服务。

**最小补测：** 先用已有 m11 反例筛查需求模型，必要时固定宽度、warp 数和动态指令，选相同旧需求值但不同 lane／word 关系的一对布局；依赖读与独立流分开。目标程序使用矩阵访问时测其真实短序列；只有单路规则可用后，再做一组 WGMMA／TMA 与普通 STS 的匹配配对。

**交付与接入：** 普通访问需求转换、各路径条件服务与受支持的联合约束。分别接入 `sm.access.*`；WGMMA 的已有复合计算服务若已含 SMEM 取数，细化时替换被覆盖部分。

### 3.5 寄存器：操作数服务与结果可用

**已有证据：** [R01](../../../experiments/gh200_sm90/Q08-register-residency/R01-readiness.md)、[R02](../../../experiments/gh200_sm90/Q08-register-residency/R02-register-service.md)及[寄存器路径页](register_service.md)覆盖依赖、源轮换、固定编译帧与消费尾部；物理 RF bank／端口尚未辨识。

**要辨识：** 源位置和复用距离如何改变发出；结果何时可被指定消费者使用；多个返回流是否相互限制。寄存器容量与访问服务分别输出。

**最小补测：** 固定同一 opcode、指令数、依赖、编译寄存器数及消费者，选复用／轮换源配对；逐函数核对源字段、reuse 和排程。只有这些条件可保持时才改变源映射；否则交付该编译帧的服务等价类。结果保留数与消费尾部另配对；实际 epilogue 使用实际消费者。

**交付与接入：** 源模式相关接收规则、依赖就绪间隔、条件返回竞争和容量输入。接入计算／控制节点的 eligible、issue、visible；先以编译帧条件服务表达，不从一次 FADD 窗口换算裸 RF 字节带宽。

### 3.6 计算：指令吞吐与真实有限链

**已有证据：** [EXP-05](../../../experiments/gh200_sm90/Q11-other-compute/EXP-05-fma.md)–[EXP-07](../../../experiments/gh200_sm90/Q05-wgmma/EXP-07-wgmma.md)有 FMA、mma.sync、WGMMA 的依赖／独立流；[R00](../../../experiments/gh200_sm90/Q05-wgmma/R00-anchor-target.md)有目标 N 形状及 N256 的 K/MN 成对补充。历史测量资格引用[计算复审](../../../experiments/gh200_sm90/EXP-01-compute-audit.md)。

**要辨识：** cfg_b 实际 N128、A K／B MN、有限 Ktile 下的接收、提交和完成；多独立链的持续能力能在多少长度后兑现。mma.sync 与 WGMMA 保留各自指令组织；scalar FMA／IMAD／转换只补目标程序实际需要的控制或输出工作。

**最小补测：** 先匹配 §4 的真实布局和组结构，测驻留的4／13／32 Ktile；固定累加容量，分别测依赖与独立工作。长度、独立链和等待组织分步改变，保存首段、后段和末尾完成。已有孤立峰值用作参照，新增工作集中在有限链及迁移缺口。

**交付与接入：** 条件计算接收、复合服务、commit 与 complete 的连接及短链尾部。替换 `compute_ns`、`compute_phases_ns`、提交／完成服务；当前512 cycle/Ktile与256 cycle提交提前量的采用状态继续见[组合](../composition.md)，新增测量不直接覆盖冻结参数。

### 3.7 异步搬运：请求、源、目的和排空

**已有证据：** [EXP-12](../../../experiments/gh200_sm90/Q02-tma-request/EXP-12-cp-async.md)、[EXP-14](../../../experiments/gh200_sm90/Q02-tma-request/EXP-14-tma-1d.md)–[EXP-16](../../../experiments/gh200_sm90/Q02-tma-request/EXP-16-tma-pipeline.md)，以及当前[TMA 路径](TMA-SHARED-SERVICE.md)。旧 bulk／rank-2 tensor 与 cfg_b 三条3D加载分别保留。

**要辨识：** 发出间隔、源完成、独立目的投递、软件等待返回和消费者释放如何限制后续请求；源与目的信用分别在哪个事件归还。

**最小补测：** 首轮直接用 cfg_b 的 A16 KiB＋B0/B1各8 KiB、128B SMEM swizzle 和已查询的实际描述符。固定物理缓冲分配，分步改变活动槽数、请求并发和消费者释放延迟，再从单 CTA 扩到多 CTA；多 CTA 共享范围实验可独立于单 CTA 内部拆分推进。cp.async 只在实际方案使用时补匹配工作集与组控制，独立维护参数。

**交付与接入：** tensor 请求形成、源与目的服务、就绪／释放事件和共享规则。输出路径分别给出源读取完成与目的写完成；输入槽复用由最后消费者释放推进。具体完成与等待观测设计引用[TMA 路径页](TMA-SHARED-SERVICE.md)和[窗口最小证据缺口](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261016-cfg-b-window-001/STATUS.md)。

### 3.8 同步：条件满足与线程返回

**已有证据：** [EXP-11](../../../experiments/gh200_sm90/Q04-sync-slot/EXP-11-sync.md)的齐步／偏斜与 fence 窗口；[R13](../../../experiments/gh200_sm90/Q02-tma-request/R13-async-retirement.md)、[R14](../../../experiments/gh200_sm90/Q06-handoff-output/R14-stage-handoff.md)的释放／交接；[计算量干预](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/g3-stage-credit/20261015-g3-compute-credit-discrimination-003/RESULTS.md)支持有限槽复用的作用。

**要辨识：** full 条件满足、wait 返回、计算完成和 empty／release 的间隔；快速已满足路径与实际等待恢复路径的差异；有在途操作时 fence 的覆盖范围。

**最小补测：** 同一 barrier 代码与槽代次，分别设置消费者到达前已完成、消费者进入等待后再完成；控制到达偏斜，记录完成的可观测时间区间与 wait 返回。观察器开／关成对核对扰动。WGMMA group 完成与输入 mbarrier 各自保留；在 §4 同一个短链里检查它们如何约束释放。

**交付与接入：** 同步状态转换、观察／恢复服务及 release 前驱。替换当前未单独定价而取前驱最晚时刻的对应控制节点；具体参数只覆盖其测量的快／慢路径与参与者组织。

### 3.9 SM 发射、驻留与调度：多个操作流怎样推进

**已有证据：** [R06](../../../experiments/gh200_sm90/Q08-register-residency/R06-issue-residency.md)、[R11](../../../experiments/gh200_sm90/Q07-smem/R11-mixed-issue.md)、[R12](../../../experiments/gh200_sm90/Q07-smem/R12-smem-path-contention.md)、[R16](../../../experiments/gh200_sm90/Q08-register-residency/R16-residency-quota.md)、[R19](../../../experiments/gh200_sm90/Q01-scheduling/R19-critical-cta-tail.md)，状态分别由[联合服务](joint_service.md)与[驻留页](residency.md)维护。

**要辨识：** ready warp／warpgroup 的竞争、混合指令的共同接收限制、实际驻留与配额等待、CTA 起步和推进离散度。静态容量用于约束合法组织，推进规则由活动窗口确定。

**最小补测：** 先重用独立／串行／重叠配对，选与目标反例对应的一种指令配比和到达相位，核对 SMID 与实际重叠。cfg_b 的现有 SMEM 占用约215 KB、每 SM 只驻留1个CTA，因此首轮使用跨 SM 活动规模和同 CTA producer／consumer 对照；同 SM 多 CTA 留给资源允许的配置。前端足迹、分歧或动态 `setmaxnreg` 只在相应 SASS／等待反例出现时展开。

**交付与接入：** `sm.issue` 的条件联合服务、资源可用时刻、CTA 启动／活动描述；静态工作列表仍由程序图生成。预测最晚 CTA 时使用同次事件推进结果，不用 occupancy 倒数给整段时间缩放。

## 4. 首个组合切片：cfg_b 真实短计算链

### 程序与工作量

cfg_b 软件 tile 为128×128×64，FP16输入／FP32累加。构建使用 `MMA_64x128x16_F32F16F16_SS<Major::K,Major::MN>`，A/B 均为128B SMEM swizzle；每 Ktile 的两个 M64片段各执行四个 K16操作，共8条 WGMMA。4 Ktile共32条、8,388,608 FLOP；以4096 FLOP/SM cycle作名义持续服务参照，计算工作量为2048 cycle。

来源：[冻结构建日志](../../../../../../results/gh200_resource_campaign/access_rules/Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011/build/cfg_b_plain.log)、[SASS](../../../../../../results/gh200_resource_campaign/access_rules/Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011/build/cfg_b_plain.sass)及[程序图](../program/cutlass_program_graph.md)。SASS 的 PC15c0–1780含8条 `HGMMA.64x128x16.F32`，均带 `.tnspB`。普通循环采用 `K_PIPE_MMAS=1`，commit后wait1再释放较早槽，末尾wait0释放最后槽。旧 R00 的 N256 K/MN 对照复用为布局依据，真实短链的有限实测与资格见[COMPUTE-01](../../../experiments/gh200_sm90/Q14-composition/COMPUTE-01-resident-chain.md)。

### 逐层实验

| 版本 | 固定或增加的内容 | 主要观察与模型决定 |
|---|---|---|
| A 驻留计算 | 同一计算体、布局、累加器与commit／wait；计时前准备SMEM，不在窗口内搬运 | 计算发出、后段提交间隔、完成尾；确定有限链计算服务 |
| B 正常交接 | A上增加full／empty槽协议；首S槽在窗口前就绪，复用后producer等empty再发布full | 测首批已就绪路径及实际复用交接；检查当前零额外控制时间假设 |
| C 受控迟到交接 | 同B代码与存储，producer只在第一个复用组取得empty后延迟显式D周期 | 区分就绪前等待与完成后的恢复；输入为已声明延迟，不用测后端点回填 |
| D 原生TMA供给 | B的槽与消费者，输入改为实际A/B三次TMA，缓存准备明确 | 发出、可观测full区间、wait返回、计算和release；接入输入服务与同步规则 |
| E 真实输出交替 | 选定13 Ktile条件，至少两个输出tile，两个数学WG按pingpong交替 | 跨输出槽代次、计算与输出重叠及最后排空；接入完整程序组合 |

首批A–D只选 Ktile数4／13／32、单CTA、固定物理缓冲分配。4检验建立与尾部；13覆盖S4/S6的重复复用，S6在第13项进入第3代首槽；32用于观察后段接收间隔和长度增量。先按S6顺序定位新增等待，再在B–D中按需增加S4配对；A没有输入槽协议，不重复标成S4/S6。1／2／5／7 Ktile用于定位特定边界，32后段仍未稳定时才加64。E不与全部版本全排列。

A–C使用固定驻留环重复读取已准备数据，性能配对的D使用相同逻辑数值序列；13／32 Ktile沿环推进，不预存全部独立输入。另用可识别槽／代次的非均匀输入检查复用正确性。B/C冻结时选明普通arrival或手动`expect_tx/complete_tx`路径，复用[barrier_return](../../../../../../microbench/gh200_resource_campaign/access_rules/probes/barrier_return.cu)的模式区分；D记录真实TMA的事务完成。

B/C的普通arrival派生探针、冻结预测与[实测停止结果](../../../experiments/gh200_sm90/Q14-composition/COMPUTE-01-resident-chain.md#bc-measured)已保存；开发数值与资格通过，零额外控制具体形式未通过plain响应，留出未执行。

D的原生三请求、计算与释放短链已完成[有限开发实测](../../../experiments/gh200_sm90/Q14-composition/COMPUTE-01-resident-chain.md#d-measured)。全部TMA含初始供给都在主窗口内；完整ring／代次数值及descriptor源字段／opaque身份已核对，没有正式时间预测或留出，现有价格保持。

E的[真实pingpong薄封装与开发实测](../../../experiments/gh200_sm90/Q14-composition/COMPUTE-01-resident-chain.md#e-measured)已完成：单CTA、S6、每输出13 Ktile、三个输出覆盖WG0→WG1→WG0，完整global数值／guard与八端点已核对。A的长度留出支持匹配驻留体；B/C及D/E的观察扰动不满足端点向plain迁移的条件，独立控制／供给／输出价格仍未辨识。

S4第一次复用在第5 Ktile，S6在第7 Ktile；4项短链不能单独决定循环反压。SMEM预留、寄存器帧和实际驻留逐构建核对；源／目的完成、槽复用和barrier计数都使用同一代次身份。

多CTA扩展使用方向2／7已选定的独立／共享源对照，再扩大活跃CTA数。进入这一步的条件是单CTA端点可解释、观察扰动足以区分候选；针对共享范围的既有多CTA实验继续独立复用。

### 输出与效率

每次短链输出模型的首个issue／compute、每代commit／complete／release及程序排空。实测保留发出／提交附近打点、wait1／wait0返回、release和依赖消费者结束；wait返回提供相应组已完成的上界，用来约束模型complete。原窗口周期和ns同时保留。以总FLOP除以同SM窗口周期得到有限链有效吞吐，与名义持续服务作比较；加入搬运／同步后的比值称该程序的端到端计算效率。硬件忙碌比例有独立观测时另列。

首段、后段和尾部各自评分；从4到13到32的响应约束建立、循环和排空，版本间差值先作为组合响应保存，再通过控制实验决定归属。真实计算端点至少区分第二数学WG的 `mma` 返回与 `MAIN_END`，沿用程序图的 `mma_tail` 边界。

## 5. 接入现有组合的具体位置

| 方向产物 | 当前接入位置 | 替换的范围 |
|---|---|---|
| HBM／L2状态与源就绪 | [supply_model](../../../../../../microbench/gh200_resource_campaign/access_rules/supply_model.py) 的 `input_source_async.issue(..., demand_id, notify_ready)` | 替换源路径完成事件；[HBM共享后端](HBM-SERVICE.md)已可接入读miss与显式写回，真实dirty策略待测 |
| 目的SMEM投递 | 同实现的 `input_destination_service(...)` → `start_ns/service_end_ns/ready_ns` | 独立目的服务；实际完成时经 `destination_done(t,demand_id)`归还相应源信用 |
| 计算、提交与完成 | [cfg_b_request_model](../../../../../../microbench/gh200_resource_campaign/access_rules/cfg_b_request_model.py) 的计算阶段参数与事件引擎 | 当前每Ktile复合计算服务；依赖检查仍绑定程序图 |
| 同步快／慢路径与槽释放 | 程序图的 `consumer_wait`、`compute_complete`、`release`及相应控制连接 | 对应控制服务与完成到返回的连接；代次规则保持来源可核对 |
| 普通LSU／SMEM／寄存器／发射 | 目标程序对应控制、取数或输出服务子图；现有[条件路径页](README.md) | 具备实际操作需求和端点后替换相应复合服务；目前未有通用逐指令接入器 |
| 真实输出 | [组合](../composition.md)的输出模块及store／store_tail端点 | 覆盖的输出整理、服务与排空；保留未覆盖部分 |
| 频率与plain映射 | 现有阶段频率、V09 observation接口 | 沿用现有版本并分项报告，不在本轮重新搜索 |

模块资格继续沿用组合的五种定义，采用与留出验收按 GOALS 判定。框架可先使用明确的复合假设完成运行；替换时以相同外部端点覆盖原模块，并保留未通过的条件及旧版比较。

## 6. 推进次序与本轮交付

下表保留2026-10-17这一轮九方向最小验证的分工与证据入口；当前按[PLAN §1](../../../PLAN.md#1-近期顺序)的模块、局部连接、共享与竞争、完整验收依赖推进。单轮许可与上限只在[GOALS §5.4–5.6](../../../GOALS.md#current-scope)维护。

| 原顺序 | 方向 | 本轮核心判断与最小干预 | 当前入口 |
|---|---|---|---|
| 1 | 同步与释放 | 原索引／递增回绕×首次复用D0／4096，先plain响应 | [同步服务](SYNC-SERVICE.md)、[当前轮次](../../../experiments/gh200_sm90/README.md#research-questions) |
| 2 | 计算 | 驻留长度约束及wait0／wait1条件对照，检验提交与完成 | [计算服务](COMPUTE-SERVICE.md)；[Q05冻结长度验证](../../../../../../results/gh200_resource_campaign/access_rules/Q05-wgmma/local-compute-chain/20261018-Q05-compute-wait-002/EXECUTION.md)两wait通过原5%标准，固定截距形式wait0失败；已有Q04迁移响应仍未通过 |
| 3 | 异步搬运 | 固定G5分配的四种B行距，Q1／Q4端点配对 | [TMA共享服务](TMA-SHARED-SERVICE.md)；[行距开发结果](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/row-stride/20261017-Q02-tma-stride-002/EXECUTION.md)已完成，Q4的B32近零、B80/B112约增加4.3%–5.4%；[CPU源定位](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/row-stride/20261018-Q02-source-isolation-001/score.json)确认现有源就绪及最终响应均为0 |
| 4 | L2共享源 | 复用源重排及两域反例；共享／独立源与次序最小配对 | [L2服务](L2-SERVICE.md) |
| 5 | HBM | 固定工作集／活动规模的相同比例串行与交错读写 | [HBM服务](HBM-SERVICE.md) |
| 6 | SMEM与输出 | 实际STS.64输出链与TMA输入的独立／串行／并发四组织 | [SMEM服务](SMEM-SERVICE.md)；[Q07开发结果](../../../../../../results/gh200_resource_campaign/access_rules/Q07-smem/input-output-joint/20261018-Q07-input-output-001/EXECUTION.md)12进程合格，并发比串行少918–955 cycle、比单独窗口max多730–742 cycle；含计时内fragment准备及两阶段join，机制信息不足 |
| 7 | 寄存器源 | 固定编译帧、依赖和消费者的源复用／轮换 | [寄存器服务](register_service.md) |
| 8 | L1／LSU | 原12开发条件冻结规则，原8个留出已检验 | [L1服务](L1-LSU-SERVICE.md)；[24进程留出结果](../../../../../../results/gh200_resource_campaign/access_rules/Q13-l1-lsu/L1-01-minimal/20261017-Q13-l1-minimal-002/EXECUTION.md)支持声明普通LSU条件内的快复用规则 |
| 9 | SM发射／驻留 | 匹配cfg_b配额，改变释放前后工作放置，检验消费者推进 | [驻留服务](residency.md) |

以下为首版交付及其未解决项。免费ROMEO上的已登记GPU测量已获许可；阶段1 L2开发完成，六形式按预置单调反例停止、未采留出；阶段2驻留A长度留出通过，两解释未区分；Q05两wait的每组完成滞后形式通过冻结长度验证、固定截距形式未通过，Q04迁移响应仍未通过；B/C开发合格但冻结零额外控制具体形式停止，未采留出。阶段3旧证据回放与条件接口已完成，实际短链独立返回价格未辨识。阶段4源／目的接口与旧证据回放已完成，D原生TMA开发数值／路径通过，D/E均无正式时间预测或留出；阶段5已完成HBM旧18条件回放与显式共享外存后端，真实写回与TMA迁移待测；阶段6已接入完整SMEM需求与输出写块工作假设，真实四组织开发窗口已测，含计时内fragment准备；准备完成后的联合响应及GEMM迁移仍缺；阶段7已接入实际寄存器需求与逐warp配额，条件窗口已回放，申请时长待测；阶段8已核对实际普通LSU需求及117份旧记录，局部缓存探针和静态构建已准备；原开发实测及独立评分完成，条件响应已记录；原8留出24进程的快复用规则通过，实际分区容量未知。阶段9已核对cfg_b容量、267份旧记录与R12快照，完整合同及模型活动报告已接通；E真实三输出端点已观测，尚无可迁移相位价格。阶段10已交付同版离线组合、三条件精确回归和原12评分，真实短链迁移与独立C2/C3未完成。

| 首版阶段 | 对应方向 | 本阶段有限交付 | 接入或转段依据 |
|---|---|---|---|
| 1 L2 | 方向2 | 固定工作集的读并发曲线、写／1:1选点、局部供给与共享服务候选 | [L2服务](L2-SERVICE.md)、[L2-01](../../../experiments/gh200_sm90/Q03-source-service/L2-01-saturation.md)；开发及构建核对完成，六形式停止，无冻结、留出未采 |
| 2 计算 | 方向6 | cfg_b实际形状／布局的4／13／32 Ktile服务，复用mma.sync证据 | [计算服务](COMPUTE-SERVICE.md)、[COMPUTE-01](../../../experiments/gh200_sm90/Q14-composition/COMPUTE-01-resident-chain.md)；驻留A长度留出通过、两解释未区分；Q05两wait每组完成滞后形式通过冻结长度验证，固定截距形式未通过，固定项未迁移；B/C具体形式停止，D/E开发诊断完成；未分离部分保留复合计算节点 |
| 3 同步与释放 | 方向8 | 已就绪／受控迟到交接，full、wait返回、empty与再取得槽 | [同步服务](SYNC-SERVICE.md)、[SYNC-01](../../../experiments/gh200_sm90/Q04-sync-slot/SYNC-01-replay.md)；phase／计数及条件返回已接通，旧46条件重算一致；B/C具体零控制形式已停止，独立返回价格未辨识 |
| 4 异步搬运 | 方向7 | A16KiB、B0/B1各8KiB原生TMA服务，复用cp.async；源与目的分别推进 | [服务接口](TMA-SHARED-SERVICE.md#stage-4-service)、[TRANSFER-01](../../../experiments/gh200_sm90/Q02-tma-request/TRANSFER-01-replay.md)；源／目的与条件等待已接通，旧资格保留；D开发路径完成，无正式预测／留出，供给价格未辨识 |
| 5 HBM | 方向1 | 大工作集读／写／混合参照与共享外存接口 | [HBM服务](HBM-SERVICE.md)、[HBM-01](../../../experiments/gh200_sm90/Q03-source-service/HBM-01-replay.md)；旧窗口外推反例、显式方向服务率和真实写回缺口 |
| 6 SMEM | 方向4 | 完整lane／word需求、实际STS.64输出块及旧联合窗口 | [SMEM服务](SMEM-SERVICE.md)、[SMEM-01](../../../experiments/gh200_sm90/Q07-smem/SMEM-01-replay.md)；地址与接口已核对，保留m11反例；Q07四组织开发已测（含fragment准备），准备完成后的联合响应、真实写块价格与GEMM迁移待测 |
| 7 寄存器 | 方向5 | 固定资源和生成代码的源复用／依赖／消费窗口、实际启动配额 | [寄存器服务](register_service.md)、[REG-01](../../../experiments/gh200_sm90/Q08-register-residency/REG-01-replay.md)；条件窗口与需求已核对，逐warp CTA池可执行，实际申请时长待测 |
| 8 L1／LSU | 方向3 | `.ca/.cg`、工作集及独立加载的普通访存服务 | [L1／LSU服务](L1-LSU-SERVICE.md)、[L1-01](../../../experiments/gh200_sm90/Q13-l1-lsu/L1-01-replay.md)；实际零普通global／local需求与旧窗口核对；原开发实测及独立评分完成；原8留出的快复用规则通过5%时间／5个百分点响应标准，限于原普通LSU条件，实际物理分区未知 |
| 9 发射／驻留 | 方向9 | cfg_b相关混合流、资源配额与活动推进规则 | [联合服务](joint_service.md)、[驻留服务](residency.md)、[SM-01](../../../experiments/gh200_sm90/Q08-register-residency/SM-01-replay.md)；容量与活动报告已接通，E已取得三输出软件端点；可迁移联合价格及启动分布仍待证据 |
| 10 组合验证 | 组合层 | 同一cfg_b程序图、模块版本、短链及原12条件的时间线、端点和plain时间 | [COMPOSITION-01](../../../experiments/gh200_sm90/Q14-composition/COMPOSITION-01-replay.md)；三条件与原价格精确回归、原12评分复用，真实迁移与独立C2/C3待证据 |

每阶段按 [GOALS §5.6](../../../GOALS.md#56-一轮研究)先登记有限条件、重复、候选及成本，再执行。转段交付包含已完成证据、可执行规则或明确工作假设及接口、有限结论和剩余观测；规则被反证或信息不足也须保存结果后收口。准备完成、实测完成与独立验收分别记录；等待GPU只阻塞依赖该观测的部分，后续独立离线准备可以继续。

每阶段先复用已有结果，按PLAN依赖选择最小对照，不展开全参数组合。原C1/C2/C3标准、旧阈值和失败记录保留；频率／暖机、C1续跑、其他配置、前端和设备保持原暂停范围。本轮九阶段首版有限交付及当前状态统一见[交付包](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/cfg-b-composition/20261017-module-campaign-001/README.md)。

## 7. 资料与使用位置

- [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)：L1／SMEM容量关系、TMA路径、驻留与寄存器约束；具体设备参数引用[GH200硬件文档](../../../hardware/gh200_sm90/README.md)。
- [PTX ISA 8.8](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html)：与现有CUDA12.9构建对应的指令、proxy、mbarrier、group完成及寄存器配额语义。
- [Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis](https://arxiv.org/html/2501.12084v1)：参考依赖／吞吐、缓存工作集、访问宽度和TMA并发的实验方法；论文设备参数与本机参数分别记录。

原始样本继续按[实验证据入口](../../../experiments/README.md)保存；本文不保存另一份可修改的带宽或延迟参数表。
