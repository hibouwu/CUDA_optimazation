# a044 锁频重测：逐阶段说明

2026-10-23 开始。预约 `compteurgpu_R302026`，节点 `romeo-a044`，结束时间为巴黎时间 2026-10-27 22:10。管理员设置 GPU SM 目标频率 1980 MHz；四卡的短负载和 NCU 计数器访问已检查成功。

本轮先确认旧卡到新卡的参照差异，再重测已有可执行实验。旧数据与失败记录保留，新数据单独归档。实验清单以本轮启动时的实际源码、二进制、条件和调用协议为准。

## 1. 先理解这轮比较

V09参照旧卡是a057的GPU-43269fbc，新主卡是a044的GPU-7473147b。资源实验按各自原始设备身份比较，例如S16原来就在a044的GPU-07be8606上。新测量使用管理员设置的1980 MHz目标，目标负载实际频率另测。因此两批数据的差异要结合设备、实际频率和运行协议解释。

| 观察量 | 回答什么 | 怎样理解结果 |
|---|---|---|
| plain 时间（µs） | 未插事件探针的 GEMM 实际花多久 | 直接比较端到端 kernel 时间与重复波动 |
| dual 包络（ns） | 从最早 CTA 进入到最晚 CTA 完成有多长 | 与 plain 对照，检查计时边界和观察扰动 |
| 同 CTA 周期 | 工作需要多少 SM 时钟周期 | 周期相近而时间缩短时，频率可能解释主要差异 |
| 同窗口 cycle/ns | 目标执行期间的有效频率 | 检查请求的 1.98 GHz 是否在实际 GEMM 中保持 |
| NCU 计数器 | 请求、缓存和执行管线实际发生了什么 | 选对应 kernel 单独采集，并注明缓存和重放协议 |

## 2. 执行顺序与交付

2026-10-23 16:05（巴黎时间），TMA tensor2D 首块作业 780806 把节点 `/tmp` 写满，a044 因 `SlurmdSpoolDir is full` 进入 DRAIN，重测中断，见[存储中断记录](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/storage-incident-20261023/README.md)。管理员恢复节点后，2026-10-26 起作业改在 scratch 上运行：提交前检查 GPFS 配额，运行时预算超限即终止，回传核对 SHA 后清理，因此再次写满的风险已有防护。780806 的 152 个完整调用在集群上逐包核验通过。当天下行带宽只有 4–100 KB/s；回收旧文件后 home 恢复可写，scratch 可写余量约 1.4 GiB；大输出阶段的排期受这两项限制，见[接续与排期](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/NEXT.md)。

| 阶段 | 测什么 | 为什么测 | 结果交付与状态 |
|---|---|---|---|
| A：旧协议参照 | cfg_a/b/c，M=2304、N=3072，K=1024/4096；plain/dual 各 10 独立进程，共 120 个槽 | 复用旧二进制与原调用协议，先量化新旧条件差异 | 已完成120/120槽，六条件二进制/SASS/输入匹配；见下方结果 |
| B：资源与局部机制 | 现有资源家族和 Q01–Q14 中已执行的局部机制条件 | 在本次环境下检查原服务成本、响应方向、反例是否保持 | 17资源家族754发布条件及补齐的能力查询共9222调用已准备；前11家族3951调用通过原审计，378个原发布条件满足原统计门槛；局部机制已尝试1201次，713次原协议修复待测；频率626次中494次已通过原数值与窗口复算 |
| C：完整 GEMM 回归 | 已有 V 系列、cfg_b 组合条件，及当前 graph 协议参照 | 检查局部变化如何反映到完整 kernel | graph参照54槽已完成；V09已回传720次，其中700次通过原数值/trace，20次暖机失败；剩余720槽及失败槽恢复待节点可用，旧模型预测作为迁移对照 |
| D：八候选开发矩阵 | 当前 C1 的 360 条件×8 配置，plain 7/dual 2，30 块并含参照 | 比较本次环境下的时间和配置排序 | 10-27 完成（作业 786088–786501）：30,780 槽全部有终态，30,769 成功、11 暖机耗尽；原回放 359/360 条件可用、356 完整。与旧卡共同完整的 304 条件中，242 个最快候选相同，沿用旧选择的 regret 中位 0、最大 8.1%（d310，候选本身 CV≈4%）；见[结果](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261023-Q16-a044-development-001/RESULTS.md) |
| E：NCU 定向采样与总结 | 各家族代表点及重测中的差异点 | 为时间变化增加硬件观测 | 与正式计时分开。global 读写、L2 来源、TMA AB/A/B、计算链、cfg_b 两个 K、单 pass 复核及 Q14 输入计数已完成，见第 7 节 |

C1 尚未执行的独立留出属于新验证阶段，须在新开发集完成后冻结模型与预测再采集；本轮先完整交付重测与开发比较。旧冻结模型和曾经留出的条件重跑后仍是回归证据。

## 3. 怎样跟随每个阶段

每个阶段完成后，在这里给出：观测事实、能解释的变化、仍分不开的因素、下一阶段为何继续。具体实验页保留其原问题与历史结论，新轮次通过链接加入。完整覆盖清单与执行记录放在 [本轮记录](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/)。

## 4. 阶段 A 结果：参照、时钟与跨作业变化

作业780658完成全部120槽；完整原始包回传SHA256核对通过，六条件输入、静态资源、二进制和SASS均与旧卡匹配。完整表见[六参照结果](../../../../../../results/gh200_resource_campaign/access_rules/Q15-full-time/V09-component-validation/20261023-Q15-a044-reference-001/RESULTS.md)。

- 首次桥接的plain时间比旧卡高1.85%–5.09%；最大CTA周期变化为−0.18%–+1.37%。
- GEMM的dual周期/时间比为1.678–1.763 cycle/ns。独立NCU在cfg_b、K4096上的SM周期派生频率为1.67 GHz，两种观测都低于1.98。
- 管理员的1980 MHz请求仍作为环境事实登记；完整测量以实际频率为准，不把本轮全部称为恒定1.98 GHz实验。
- 当前C1 graph协议的六参照也已完成（54/54槽，作业780670），有效频率仍为1.648–1.772 GHz；见[graph桥接](../../../../../../results/gh200_resource_campaign/access_rules/Q10-time-mapping/c1-timing-protocol/20261023-Q10-a044-graph-reference-001/README.md)。

NCU采样前后的`nvidia-smi`快照分别为345和1980 MHz；访存、FMA采样期间也查询到了1980 MHz。NVML查询瞬间的SM时钟与内核窗口的平均周期/时间比边界不同，目前不能据此判断管理员锁频失效。后续对齐这两种观测的时间窗口再解释差异。

补充作业780694在同一NCU pass中测得SM为1.687848 GHz、GPC为1.687591 GHz，两者相差0.0153%；GPU窗口92.48 µs，原stdout确认8次暖机后采到正式目标。所有已报告限频累计时长增量为0。50 ms的NVML采样比目标窗口长约541倍，因此目前仍不足以确定二者差异的原因。详细定义与复算见[时钟核对](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/clock-counter-check/analysis.md)。

同卡两套plain二进制的GPU代码已另行核对：cfg_a/b/c的完整SASS逐字节相同，六条件的公共输入与setup字段相同。graph参照时间比旧协议低1.598%～5.116%，因此应继续检查计时、预热、主机执行与作业漂移；这两批数据尚未单独隔离这些因素。见[两种协议的代码与输入核对](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/plain-kernel-comparison/README.md)。

约2.1小时后的V09六参照又完成240次四变体采样，全部通过原数值和trace重放；其中plain/dual的二进制、完整参数和真实GPU UUID与首次桥接相同。plain中位时间比首次下降1.48%～5.49%，dual有效周期率变化较小。该轮采用四变体原序列和1秒NVML监测，首次只测plain/dual并取前后快照，因此首次的跨卡时间差混合了作业状态与协议影响。完整对照见[同卡参照变化](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/v09-reference-drift.md)；另按首次桥接的[原协议重复120槽](../../../../../../results/gh200_resource_campaign/access_rules/Q15-full-time/V09-component-validation/20261023-Q15-a044-reference-002/)，继续区分这些因素。

第一轮回传遇到项目磁盘配额已满和Slurm stdout归档截断。已通过可信主机公钥核验，改为计算节点先完整传到登录节点临时盘、核SHA后释放作业，再传本地。新大结果放`/data/jianyeshi/gh200-a044-20261023/`，仓库保留入口；旧数据未移动。

## 5. 阶段 B 第一批：内存访问基线

复用原 memory_baseline 已执行的171个调用，包括正式重复、短校验和空窗口。目标程序、迭代次数、输入seed保持原样。

| 条件 | 源与目的 | 要比较什么 |
|---|---|---|
| SMEM read，stride=1/2/4/8/16/32 | 同CTA共享内存 → 线程寄存器，输出校验另行写回 | 随访问步长变化的周期成本；按实际逻辑字节算B/cycle |
| SMEM write，stride=1 | 线程寄存器 → 同CTA共享内存，随后消费校验 | 共享内存写入服务与读入服务的差别 |
| global read，8 MiB或256 MiB，ca/cg | GPU全局地址所指数据 → 寄存器；实际由缓存或HBM供给待计数器核对 | 工作集与缓存指令变化后逻辑请求速率是否保持 |
| global write / duplex | 寄存器 → GPU全局输出，或全局输入 → 寄存器 → 全局输出 | 写入与读写混合窗口的吞吐 |
| one CTA / all GPU 空窗口 | 无目标工作，只保留对应计时/控制结构 | 判断短窗口的测量底噪 |

作业780679在设备身份检查阶段发现旧辅助函数仅允许一个已分配GPU，而本轮取得整节点四GPU、只让一个UUID可见。此次没有执行目标探针，失败包保留；新运行包已按本次实际分配方式修正检查。测量次数和目标代码不变。后续作业780680已完成采集和完整归档回传。

171次调用全部通过原冻结数值、工作量和计时审计，暖机失败0次；原发布的11个吞吐条件全部通过原统计门槛。SMEM的B/cycle中位数变化最大0.00213%，五个global条件的GB/s变化为−0.805%～+0.531%，没有出现需要先修正的访存吞吐差异。详细表和复算命令见[访存阶段报告](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/resource-memory-report.md)。

原来因波动未发布的SMEM stride1读，本次30进程的CV从9.85%降至0.000389%，中位数几乎不变；这是稳定性的改善。全GPU空窗口仍在96/128 ns间波动，CV为9.30%，仍不作为稳定时间参数。1秒遥测的95个点中，起始点为345 MHz，其余94点为1980 MHz；正式内核的cycle和ns观测另行保留。

读写组合的作业780721也已完成：180次调用、18个原吞吐条件全部通过原审计。小工作集的请求载荷吞吐变化为−0.734%～+2.714%，大工作集为−0.313%～+1.061%；原来随读写比例变化的服务曲线保留。这里统计的是读与写的逻辑载荷之和，原数组经工作量对齐后分别为16.5 MiB和241.3125 MiB。逐模式表见[global读写组合报告](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/resource-global-duplex-report.md)。

## 6. 阶段 B 第二批：计算与局部服务

依次复跑原FMA、mma.sync和WGMMA条件，分别为930、550和850个调用。这三组先回答计算指令在给定并发、数据类型与依赖链下需要多少周期，再与GEMM中的计算服务对照。

- FMA测寄存器中的标量乘加，改变独立累加链等原参数，观察依赖延迟与并行吞吐。
- mma.sync测Tensor Core矩阵操作，输入片段和累加值由原探针准备；使用原输出oracle核对数值。
- WGMMA测warpgroup矩阵操作，输入位置与布局沿用各原条件，保留异步完成和结果消费边界。

比较时同时列周期、实际时间和有效频率。若周期相近而时间变化，先检查频率；若周期响应也改变，再看并发、资源使用和指令路径。完整GEMM是否受益由阶段C检验。

FMA作业780684已完成930次调用，全部通过原冻结数值与观测审计，暖机失败0次。93条件各有10个稳定样本，其中90个是原发布吞吐条件，3个为零工作控制。63个单CTA条件的FLOP/cycle中位数与旧值逐位相同；27个全GPU条件的GFLOP/s变化为−0.6095%～−0.0509%，最大CV为0.1187%。三个控制的294、218、296 cycles也与旧值一致。见[FMA阶段报告](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/resource-fma-report.md)。

mma.sync作业780695已完成550次调用，全部通过原审计，暖机失败0次。55条件各有10个稳定样本，其中54个是原吞吐条件，1个为空窗口控制。39个单CTA条件的FLOP/cycle中位数与旧值逐位相同；15个全GPU条件的GFLOP/s变化为+0.000016%～+2.09772%，最大CV为1.257%。空窗口为687 cycles，与旧值相同。见[mma.sync阶段报告](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/resource-mma-report.md)。

WGMMA作业780704完成850次调用，全部通过原审计，85条件均满足原稳定性门槛。59个单CTA条件中，58项FLOP/cycle中位数逐位相同，另一项变化−0.000000745%；23个全GPU条件的GFLOP/s变化为−0.307%～+4.006%。1秒遥测实际捕捉到1785～1965 MHz的运行样本，内核窗口的cycle/ns为1.8014～1.8308 GHz。每周期服务基本重现，而按秒吞吐和实际时钟需要分别保留。见[WGMMA阶段报告](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/resource-wgmma-report.md)。

后续cp.async作业780712完成240次调用，全部通过原审计。它测的是8 MiB全局数组经cp.async搬入SMEM，再由相邻线程消费，计时包含等待与消费。24个原条件全部稳定，单CTA每周期载荷变化−0.0112%～+0.0135%，全GPU载荷吞吐变化−0.0042%～+0.6382%；stage1→2→4的响应重现。见[cp.async阶段报告](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/resource-async-copy-report.md)。

其余五个基础家族也已完成原审计。它们补齐辅助算术、不同搬运路径与同步边界，检查计算主循环以外的条件成本能否继续使用。

| 家族与调用数 | 测什么、为什么测 | 结果说明什么 |
|---|---|---|
| [辅助服务 270 次](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/resource-auxiliary-report.md) | 寄存器算术、local 访问、global/SMEM 原子操作及资源压力；比较完整循环的条件成本 | 27 原发布条件稳定，周期/迭代变化 −0.000309%～+0.102725%；同址与独立地址原子的差别保留 |
| [TMA bulk 240 次](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/resource-tma-bulk-report.md) | global↔SMEM 的线性搬运，1～64 KiB、单 CTA/整卡；比较方向和载荷大小响应 | 24 原发布条件稳定，单 CTA B/cycle 变化 −0.4132%～+0.1382%，整卡 GB/s 变化 −0.0658%～+1.8830% |
| [SMEM 访问 170 次](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/resource-shared-memory-report.md) | SMEM↔寄存器的读、写、双向访问；改变宽度、步长和广播，检查地址组织的影响 | 15 原发布条件稳定，B/cycle 变化 −0.005459%～+0.536854%；stride 2/4/8/16/32 的约 64/32/16/8/4 B/cycle 响应保留 |
| [矩阵搬运与 warp 交换 220 次](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/resource-matrix-exchange-report.md) | SMEM↔矩阵寄存器片段的 load/store/往返及 warp 内寄存器交换；检查布局与独立流响应 | 22 原发布条件稳定，新旧中位数逐位相同 |
| [同步与 fence 130 次](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/resource-synchronization-report.md) | CTA/mbarrier 的参与线程数和到达偏斜、无在途工作时的 fence；检查线程交接的完整阶段成本 | 11 原发布条件稳定，中位数最大变化 +0.00041719%；2 个原 warp 对照继续保留，原 SASS 的 WARPSYNC 数为 0 |

前11家族合计3951次调用全部通过原数值与观测检查，暖机失败0次；378个原发布条件全部满足本轮原统计门槛。局部服务的每周期响应总体复现，而完整GEMM参照时间已出现差异，接下来用组合与完整回归检查频率、供给、输出和时间映射如何共同作用。

驻留计算的wait0／wait1配对另有18次完整输出已通过原校验，共294,912个数值。它将输入预先放在同CTA的SMEM，只改变计算提交后的等待组织，用4、13个Ktile的原规则预测32个Ktile。a044的K32中位周期为20,010／17,153，旧冻结预测误差为0.01749%／0.02040%，两者均通过原5%门槛。该结果支持继续使用这组驻留计算窗口规则，见[计算等待配对复核](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/q05-wait-returned-report.md)。

提交和软件返回端点另有39次完整数值检查通过，机制资格则各有结果：提交前后延迟仍因实际剂量差29 cycle而未过门槛；首组449 cycle再次排除原476／351两候选；K2返回分支的周期预测全部匹配，但一对trace窗口多64 ns，超过原5%扰动门槛，因此这批资格未通过。见[提交与返回端点报告](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/q05-returned-branches-report.md)。

Q04的48次不同进程完成原规则复核。索引方式改变使无注入窗口从10778降至7572cycle，同样4096cycle注入的响应从908变为2067cycle；旧卡中位值相同。原局部时间线虽然能预测K长度增长，仍以15.346%的误差低估注入响应；稀疏打点的六个配对全部超过5%扰动门槛。阻塞改为轮询后，行距响应从5373降到3266cycle，原“全部保留”与“全部消失”两个候选均不匹配。各自测量路径、原判据与下一步含义见[同步与控制阶段报告](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/q04-returned-stage-report.md)。

源服务、共享内存与L1另完成9来源173次调用的原评分。L1原冻结`reuse_at_128`仍唯一通过8/8时间及12/12响应，最大时间误差1.9435%；L2旧卡的微小稳定反向响应在新卡区间重叠，未复现原反例。共享内存prepared-joint的`max`连接通过6/6检查，但bank-layout与joint-count的部分三进程稳定性失败。结果说明服务关系的迁移能力不同，应保留具体准备条件和失败判据；见[内存服务阶段报告](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/mechanism-storage-stage-report.md)。

TMA请求的42个性能来源也完成了原判据复核：共414个不同计时进程。原Q4长度外推和零搬运长度规则继续通过；单B、延后槽等候选仍有区间无法区分。操作数分拆原桥接在新卡出现5.0785%cycle、6.6667%ns偏差，超过5%门槛，因而该轮保留“信息不足”。这些变化逐项列在[TMA请求阶段报告](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/q02-returned-report.md)；同来源另9个CPU程序查询单列，不计硬件测量。

COMPUTE-01–04的75次调用完成逐层复核，其中72个计时进程全部通过原数值与暖机资格，另3次是CPU查询。驻留计算K32的原冻结预测为17153cycle，新卡中位也为17153；加入槽交接后的三条响应预测却继续失败，说明当前组合缺口已落到具体交接形式。接回真实TMA和pingpong后，数值与程序端点通过；真实pingpong的三组完整kernel观测扰动均在原5%内，plain中位20.288µs。各层的路径、预测与失败界限见[驻留计算到真实pingpong阶段报告](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/resident-compute-stage-report.md)。

机制参数包作业780769另完成1201次尝试，1137次成功采集、64次失败。40次因原程序硬编码旧参考卡身份退出；24次是本轮打包误取了旧档中的失败构建。随后核对发现，20个Q14来源还通过`GH_*`环境变量切换输入来源、活跃CTA、输出数量等条件，而本轮包只保留了命令行参数；部分原交错顺序也被改为集中重复。对应数据保留作准备问题的证据，原机制判断等待正确二进制、环境和顺序的补测。身份修正只改本轮副本的期望UUID，CUDA fatbin与完整SASS均保持逐字节相同，见[身份修正证据](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/host-gpu-gate-patch/proof.json)与[原环境逐条映射](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/mechanism-env-mapping.json)。

## 7. NCU怎样补充这些测量

正式时间来自原未被profiler观察的运行。NCU另跑相同二进制和参数，回答逻辑工作量对应了多少物理访问，以及哪些执行管线参与了工作。当前先查询a044实际支持的指标，再执行以下代表点。

| 对象 | 物理问题 | 与已有测量怎样对应 |
|---|---|---|
| global `.cg`读小/大工作集，写大工作集 | 经过L2的sector数与L2↔HBM的字节数各是多少 | 原访存吞吐统计的是程序请求量；计数器补齐实际服务层次 |
| TMA的AB、仅A、仅B配对 | 同一组输入请求分开和组合时，TMA指令与缓存流量怎样变化 | 每轮A为16 KiB、B为两次8 KiB；32轮AB逻辑载荷为1 MiB |
| 驻留WGMMA计算链 | 给定32个Ktile、256条程序WGMMA操作时，Tensor管线的计数和活动如何 | 先测计算服务，再与包含TMA和槽交接的程序比较 |
| cfg_b，K=1024/4096 | 局部服务放进完整GEMM后，实际供给、计算与输出有什么变化 | 对照独立plain、dual及原冻结预测的分项残差 |

原名为8 MiB、256 MiB的访存条件，按CTA工作量取整后每数组实际为8,650,752 B、270,336,000 B；计数分析采用实际分配与原逻辑请求量。自适应暖机的程序按本次stdout中的暖机次数选择正式目标，固定暖机的TMA按原启动顺序选择。指标、启动选择和原数据链接见[NCU准备说明](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/archive/ncu-targeted-package/README.md)。

首个小工作集`.cg`读已采到正式目标：逻辑读取553,648,128 B，L2读sector对应554,016,320 B，DRAM读152,320 B，L2命中率99.92%。这支持已预热的小工作集主要由L2供给。该组指标需要3个replay pass，已安排分别只采DRAM读、L2读的对照，核实精确流量受重放状态影响的程度。正式目标绑定、单位与原报告见[首点NCU核对](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/ncu-first-point-report.md)。

大工作集的L2总读计数约为逻辑读量的1.5倍，按来源拆分后得到直接证据：作业780740的两点均为单pass，SM/TEX来源分别为17,301,504和135,168,000个sector，乘32 B后精确等于各自逻辑读量；LTCFABRIC来源分别为20和67,584,024个sector。大工作集额外约半份逻辑读量的L2查询主要来自跨分区转发，小工作集预热后几乎没有这种转发。模型应分别计量程序请求、L2内部查询和HBM传输，各路径的服务时间再用对应实验约束。现象、候选解释、判别实验与完整结果见[L2来源分析](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/ncu-256m-interpretation.md)。

TMA作业780752将同一程序的AB、仅A、仅B三种请求分别采集。这样既能核对程序请求数，也能检查单独测得的A/B服务怎样组合。32轮中的每轮A加载一次16 KiB，B加载两次8 KiB，三种条件的结果如下。

| 条件 | TMA加载次数 | NCU的TMA指令数 | TMA管线活动周期合计 |
|---|---:|---:|---:|
| AB | 96 | 98 | 194 |
| 仅A | 32 | 34 | 66 |
| 仅B | 64 | 66 | 130 |

三点均多出两条指令，与源码开头的两次描述符预取、SASS中的两条`UTMACCTL.PF`吻合。活动周期也符合“每次加载2周期、每次预取1周期”的计数关系；这里核对的是管线活动计数，搬运完成时间仍由请求到达与完成事件测量。源码还在局部计时开始前设置了100 µs起跑等待，因此NCU的120～130 µs完整kernel时间包含这段等待。程序边界、正式调用绑定和逐条计数见[独立核对记录](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/archive/ncu-tma-independent-review.json)。

这组指标用了3个replay pass，其中仅A的派生命中率报告为125.7%。原值已保留；接下来分别只采DRAM读量和L2读sector，检查单pass下的流量，再判断A/B缓存服务如何组合。采样本身成功，工具日志插入探针JSON造成的后处理错误已按原字节恢复，原归档保持不变，见[输出恢复与复算](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/ncu-tma-ab-a-b/cpu-recovery/README.md)。

计算链的正式NCU目标也已核对：程序执行256次warpgroup矩阵操作，NCU记录1,024条GMMA类warp指令，恰好对应每组4个warp。完整kernel为50.016 µs，探针局部计算窗口为9.472 µs；源码中局部起点之前还有196,608 B的SMEM准备，局部终点之后还有65,536 B的全局导出。组合模型的计算服务要接入这个局部窗口，完整kernel计时则用于核对包含准备和导出的整体过程。计数范围、原数值检查及周期速率差异见[计算链NCU核对](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/ncu-compute-report.md)。

完整cfg_b的两个K条件也与程序图对齐：432个输出tile、132个CTA，每个Ktile三次A/B加载；每个输出tile有八次TMA store和八次flush，每个CTA开始时有三次描述符预取。按这些程序事实计算，K=1024/4096时TMA类指令分别为28,044/90,252，均精确匹配NCU；GMMA类warp指令也分别匹配221,184/884,736。这个核对落实了程序图的工作量，服务完成时间仍由对应窗口实验确定。公式、SASS位置与原报告见[cfg_b完整kernel的NCU核对](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/ncu-cfg-b-report.md)。

2026-10-26 补完了待测的单指标复核和 Q14 输入计数。作业改在 scratch 运行，带配额检查。

- 改为单 pass 后，8 MiB 读和 TMA AB/A/B 的 DRAM 读量都接近 0。此前 3-pass 测到的 HBM 读量来自 replay 之间的缓存状态。
- TMA AB 的 L2 读量比 A+B 多 13%，两种协议之间也相差 17%，仍需按来源拆分并重复采样。
- Q14 的 A16/base L2 读 sector 比值：合格的 2 个 trial 为 1.391 和 1.498，不合格的 trial 0 为 1.512。按原判据为信息不足，但三个 trial 都与小幅读工作带不相容。增加的读取由 L2 供给，DRAM 读量反而下降约 6%。

见[补测报告](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/ncu-pending-20261026-report.md)。

10-27 继续拆分 TMA AB/A/B 的 L2 读来源，判据都在运行前冻结。
- 单 pass 重复 3 次（作业 785962）：AB 比 A+B 多 4.9k–5.7k sector，每次都为正，但三次的分散为 17%。把总量和 4 个来源指标放进同一个 profile 需要 4 pass，此时总量高约 34%，不能用于归因。见[结果](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/ncu-followup-20261027/RESULTS.md)。
- 逐指标单 pass，5 轮×5 指标（作业 786546）：25 次运行都是 1 pass，总量与此前单 pass 相差不超过 1.2%，所以测量状态一致。但 tex/ltcfabric/gcc/l1 四个来源之和比总量少 1.8%（AB）到 7.4%（B），多出量的闭合误差为 37%。按冻结判据，来源归因信息不足，本轮收口。
  - 描述：四个来源指标逐次完全相同；tex 精确等于 payload+4；gcc 在 AB 上为 7,360，在 A/B 上为 320。
  - AB 在每次运行中都是第一个被 profile 的 launch，所以分不清“AB 请求本身多读”和“首个被 profile 的 launch 多读”。
  
  见[结果](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/ncu-tma-single-20261027/RESULTS.md)。

## 8. 频率线：局部周期怎样换成时间

这一阶段测同一CTA、同一消费者的配对窗口，用周期差除以纳秒差得到有效频率；完整kernel时间另由plain进程测量。改变输入、参与SM数、K长度和目标前空闲，检查运行条件怎样影响这个换算关系。已完成前8组494次采样的原数值与窗口复算，核对了2,023,424个抽样输出值；原模型和门槛保持冻结，没有用新数据拟合。

在cfg_b开发组的8个条件中，a044的有效频率为1.5224～1.8293 GHz。同一K和SM数组合下，zero输入仍高于dyadic输入。例如K=16384、132个SM时，两者分别为1.7430和1.5224 GHz。因此，1980 MHz目标设置下，模型使用的窗口周期/时间比仍需结合输入和工作组织测量。

| 冻结主张 | a044结果 | 对模型的含义 |
|---|---|---|
| 纯有效频率预测，8个留出条件，逐项误差≤5% | 8/8通过，最大误差3.7966% | 该预测器在这组新卡条件上保持原精度 |
| 第一组cfg_b完整时间预测，8个形状 | 时间7/8通过，最大误差5.2356%；频率子标准7/8通过 | 保留超标形状，继续检查频率与其他分项的误差 |
| 第二组cfg_b完整时间预测，另8个形状 | 时间7/8通过，最大误差6.6015%；频率子标准6/8通过 | 这组迁移尚未满足原整体验收 |
| S6/S4阶段有效频率预测，10个阶段端点 | 6/10通过，最大误差10.7073% | 整体CTA窗口预测与分阶段预测须分别验收 |

目标前空闲的干预也保留了一个明确响应：`d044_cfg_b_s4`空闲1 ms后，所测阶段有效频率提高45.69%，完整plain时间下降10.95%，且通过原配对与重复范围门槛。阶段频率的变化与完整时间的变化幅度不同，后者还取决于该阶段在整体执行中的位置和占比。

每个条件的旧实测、新实测、冻结预测、原门槛，以及时钟包围、空闲历史和S6/S4的结果，见[频率阶段报告](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/frequency-stage-report.md)。剩余组按原窗口和判据接续；采样表之外的12次原预检另行补测。
