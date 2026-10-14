# GEMM TMA 供给：硬件候选与反例

状态：已有候选的开发诊断；未形成通过独立干预留出的组合规则。新实验顺序和预算只在[多 CTA 主线](TMA-SHARED-SERVICE.md)维护；S16 控制成本只在[S16](TMA-S16-STAGE-COST.md)维护。软件依赖见[程序图](../program/cutlass_program_graph.md)。

## 1. 现象

cfg_b 的 stage、生产者起步、预填与其他 CTA 输出干预同时影响首段等待、后段主循环和 CTA 离散度。主要证据为[R13](../../../experiments/gh200_sm90/access_rules/R13-async-retirement.md)、[R10](../../../experiments/gh200_sm90/access_rules/R10-layout-cache.md)、[R09](../../../experiments/gh200_sm90/access_rules/R09-inkernel-clock-stages.md)、[R15](../../../experiments/gh200_sm90/access_rules/R15-output-service.md)。源为本 GPU global 分配，目标为本地 SMEM；多播/输出配对按对应记录区分，逻辑覆盖量不作物理流量。

## 2. 假设

| 候选 | 当前判定 | 重新研究或保留的条件 |
|---|---|---|
| 固定边成本、只平移入口的图 | 不能解释延迟放大及已有响应 | 仅在图/观测边界改变且重写可检验预测时重新立项 |
| 同质 FIFO 服务＋统一入口间隙 | 受保序反例排除，不继续调统一 gap | 不把排除推广为所有排队模型被否定 |
| 有限槽＋源共享＋独立目的投递 | 保留结构候选；已测两套参数均失败 | 同一组参数须同时解释干预方向、幅度和 CTA 跨度 |
| 仅改变源订阅截止 | 已有有界拟合不能同时解释方向与跨度 | 当前不扩展截止扫描；不宣称全部函数族被排除 |
| 本地 epilogue 排他 | 只能解释部分响应，默认关闭 | 若实际依赖使两者相交，再用独立干预确认 |
| 非同质局部推进与跨 CTA 服务共同作用 | 当前待辨识方向，尚无可冻结参数 | 多 CTA／请求并发干预与局部 ready 端点共同约束 |

## 3. 实验验证

下面记录已完成的开发干预及其可排除范围；新实验依[多 CTA 主线](TMA-SHARED-SERVICE.md)执行。测过的干预已属开发证据，不能重命名为新候选的独立留出。

### 有限槽、交接与局部端点

[入口条件检验](../../../experiments/gh200_sm90/access_rules/R13-async-retirement.md#stage-conditioned)用后段slot关系排除了旧ℓ=2257.5用于本批。[两stage共同开发](../../../experiments/gh200_sm90/access_rules/R13-async-retirement.md#stage-reuse)得到后段复用组合Z约1755～1773 cycle；S−1来自软件释放依赖，不是物理队列深度。完整继承子类仍缺stage4早段约2079 cycle，不能作为完整预测交付，也不能给early另加专属常数。任意单一producer边界间隙仍不能突破该子类早段界；更宽分支的[联合端点约束](../../../experiments/gh200_sm90/access_rules/R13-async-retirement.md#stage-joint)只排除部分逐调用解释，不证明中央代表轨迹不可行。[同调用整卡上下文](../../../experiments/gh200_sm90/access_rules/R15-output-service.md#stage-output-context)还显示重叠面积本身不能定价，需保留stage输入复用与入口相位。诊断参数与实测入口均未进入正式预测。 739717改变本CTA输出顺序后仍有早段缺口；739740进一步固定本CTA处理，仅延后其它CTA输出，stage4早段又10/10缩短，配对中位−1008.5 cycle，但后段变长、plain未改善。局部供给必须允许全卡组织的影响；该处理也改变其它CTA输入请求，未识别唯一物理资源，未加入并发价格。延后的E0进入后段，故只能用选定无E0覆盖窗口约束有效c/T，不能将整段flag2作为清洁基线，见[R13条件迁移](../../../experiments/gh200_sm90/access_rules/R13-async-retirement.md#stage-background-baseline)。见[R15同调用背景配对](../../../experiments/gh200_sm90/access_rules/R15-output-service.md#stage-output-background)。

后续候选必须保留时间状态。[同位置起步干预](../../../experiments/gh200_sm90/access_rules/R10-layout-cache.md#producer-delay)在工作/SM不变时改变了行距响应，而且B0在扣除实际producer后移量后，首输出仍多耗816 ns、1156.5 cycle。固定事件前驱、固定同域边成本且其它根时刻不变的max-plus图，不能把单根延迟放大；仅平移旧时间线或增加固定位置罚项不充分。该约束不排除带可变仲裁、等待返回与初态的有限stage模型。producer/FIRST软件名次也不是物理请求到达，不直接拟合为服务价格。 [首次供给端点](../../../experiments/gh200_sm90/access_rules/R13-async-retirement.md#first-supply)把额外量定位在copy发出标记到FIRST之间，首次发射前成本与consumer进入等待时刻未明显增长；仍不能把此段当裸TMA延迟。[释放前端点约束的ready观察](../../../experiments/gh200_sm90/access_rules/R13-async-retirement.md#ready-observer)在新批40/40次绑定首代事务：延迟干预的issue→ready增量下界在两种行距各十次均正，ready→FIRST增量区间全部含零，排除本协议下仅consumer返回变慢的解释。请求排队、数据服务与完成可见性仍不可分；B32的L缩短也大多兼容等待移入S，不能直接提高稳态供给价。

[第二WG联合初态对照](../../../experiments/gh200_sm90/access_rules/R13-async-retirement.md#second-prefill)进一步支持保留交接状态：同时预等S槽、排空前WG并改变相位，使stage4 early十对均缩短，stage6基本不变，完整调用未稳定改善。候选须由槽ready与未完成计算依赖预测该响应；全预填后，恒定reduced复用关系的无参数窗口界仍在stage4十次全部失败，故仅修初态不够。此界不覆盖宽提交/退休分支；后续off-only宽拟合虽能使部分窗口RMS降至0.28%，仍失配局部端点和early干预，不采用。内部量只能辨识u=ℓ+a，交接δ也存在训练量不变方向；不能把分开的拟合系数解释为硬件价格。从预等开始核算，stage4到C16/M均未稳定缩短，stage6十次均变晚；early收缩包含等待重分配，不采用early专属价格。FIRST在barrier返回后、首HGMMA前；统一TC起动偏移在q≥v时只重参数化q/λ，不能修复当前响应。尚不能分离三个作用。stage6后段与更多E0重叠而周期基本不变，也排除了直接按重叠数收费的解释。

<a id="finite-source"></a>

### 共享请求与源身份

[supply_model.shared_stage_events](../../../../../../microbench/gh200_resource_campaign/access_rules/supply_model.py)给当前cfg_b两输出组织实现一个纯事件候选。对软件前驱允许发出的请求j，FIFO服务结束为`F_j=max(a_j,F_(j−1))+τv_j`，就绪/源复用为`R_j=F_j+ℓ_type`。每Ktile的A16KiB取v=1、B0/B1各8KiB取v=1/2，输出8KiB取v=η/2；这只是逻辑服务，不命名为物理L2/HBM。输入S槽、普通/尾释放、输出8子tile和双缓冲都显式保留，E0相位由图生成。

当前同质请求及无本地TC竞争的版本不再作为待调参候选：[R13](../../../experiments/gh200_sm90/access_rules/R13-async-retirement.md#stage-shared-service)的联合窗口最大残差虽只有3.32%，但全卡M跨度过窄、本CTA输出干预效应不足。两低误差校准解也不唯一。它不参与V09预测；测后E0、入口或完成排序不进入参数。末端仅覆盖源读取尾，不是全局D可见或完整CUDA时间。[请求形成](../../../experiments/gh200_sm90/access_rules/R13-async-retirement.md#stage-request-formation)已纠正B的请求粒度；[分阶段计时](../../../experiments/gh200_sm90/access_rules/R09-inkernel-clock-stages.md#stage-phase-clock)显示首/次输出换算率不同，当前统一ns服务价仍是待检验假设。观测PROD仅在单列诊断中作输入；两个未标定的有限gap示例消除了微小起步偏移的全卡分布跳变，说明零间隔并列处理影响旧结果。现有61cycle组外窗口没有分离A→B0、B0→B1和跨组控制工作，尚不能给它们定价。更强的[保序反例](../../../experiments/gh200_sm90/access_rules/R13-async-retirement.md#stage-order-invariant)表明：即使允许任意producer起步偏移，同质FIFO图也不能产生实测的FIRST0→MAIN1严格逆序。因此停止用统一gap或入口调参修复这张图；下一候选必须解释非同质推进，不能据此宣称识别某种硬件仲裁。[分量替换](../../../experiments/gh200_sm90/access_rules/R09-inkernel-clock-stages.md#stage-order-decomposition)只用于诊断：stage4保留较强周期成本差，stage6排序更依赖观测周期率；没有将这些观测值或比例接入预测。

在途源身份扩展仍是开发候选，未接入正式预测：[两套具体参数复放](../../../experiments/gh200_sm90/access_rules/R13-async-retirement.md#source-candidate-replay)中，共享版本能给出stage4预等early缩短约745ns、stage6不变的方向，但四条件的全CTA mma返回跨度都低于全部实测进程；不共享参数同时失配stage4累计响应。保留“源服务可共享、目的投递独立”的候选结构，拒绝将这两套参数当作可组合规则。三种同时请求插入顺序未修复跨度；下一步用既有逐CTA阶段与时钟记录约束非同质推进，不拟合stage专属价，不用观测入口/频率修正正式预测。

[源组分解](../../../experiments/gh200_sm90/access_rules/R13-async-retirement.md#source-spatial)进一步限定缺口：739095的stage4首/后输出中，A行组均值解释周期离散的94.6%/96.5%，组均值跨度约2356/2463cycle；stage6周期差更小而ns成组性仍强。A组、CTA与SM集合在这20进程完全绑定，不能定为A源服务价格。当前源键相等关系模型对键的一一重命名严格不变；这一不变量可用于设计地址/位置对照，不加入组编号罚项。

同一共享参数的[20进程producer切口](../../../experiments/gh200_sm90/access_rules/R13-async-retirement.md#source-offsets)也失败：原幅起步偏移下stage4/6的第二MAIN跨度中位仅129.9/318.3ns，实测2624/1136ns。trial0内部记录显示每个A/B源generation都被完整同组订阅，共同ready强烈同步了请求；这不是硬件复用率。有限gap下微小起步扰动未复现旧并列事件跳变，停止为这套参数调入口。非同质推进仍需由前驱、目的服务或源状态解释，不能转成组标签价格。

[A源单独轮换](../../../experiments/gh200_sm90/access_rules/R13-async-retirement.md#a-source-rotation-preparation)740033的首窗口10/10配对通过同SM/无E0覆盖门槛，但固定位置对应仅8/10优于跟随地址，两个方向残差仍大；同档重复的组曲线也不稳定。不支持将空间差直接定成A地址组价格，也未识别唯一位置机制。后窗口10/10不能替代主窗口；完整时间变化范围跨零，不能当成严格零响应验证。保留共享状态与局部推进的候选边界，不追加地址矩阵。

[源订阅截止检验](../../../experiments/gh200_sm90/access_rules/R13-async-retirement.md#source-join-boundary)未提供可采用的新参数：服务结束/开始截止虽减弱同步，但有界两坐标重拟后仍将stage4预等early预测为增长311/319ns，与十次实测缩短相反；ready截止保留方向而空间跨度仅123ns。默认模型保持原ready边界，其余选择仅作诊断。局部拟合退化、分支不光滑及单起点范围均保留，不宣称全族排除；下一步约束局部发射/消费与共享服务的组合，不再扩大截止时间扫描。

[本地E0排他](../../../experiments/gh200_sm90/access_rules/R13-async-retirement.md#local-epilogue-exclusion)沿用既有prepare/write时长，不新增数值参数；stage4输出延期响应最多扩大到约−0.86%，仍不足以解释旧批约−6.82%的周期响应。defer2下本CTA全部输入与MMA起动先于E0，局部排他严格不改变图，不能靠放大E0价格修复剩余stage4缺口。该候选保留为开发诊断，默认关闭；下一专业供给轮次由B负责。

## 4. 结论

当前采用“需要时间状态、局部推进与跨 CTA 条件”的工作方向，不采用上述失败参数作为硬件价格。备选与反证范围见上表；只有能够同时预测现有反例并对未测干预给出不同响应的候选，才进入下一次冻结。半经验 B 源响应式仍在[MODEL-CANDIDATES](../../semi_empirical/MODEL-CANDIDATES.md#source-sharing-dev)维护，两者不相互授予通过资格。
