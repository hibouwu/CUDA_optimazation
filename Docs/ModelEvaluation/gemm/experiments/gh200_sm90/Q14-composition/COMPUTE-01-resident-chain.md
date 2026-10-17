# COMPUTE-01：cfg_b 驻留计算短链

更新：2026-10-17。A的开发及留出完成，驻留窗口长度外推通过；B/C零额外控制具体形式在开发响应失败后停止；D原生TMA及E真实三输出完成有限开发数值／路径观测。各版资格与结果分别见下文。原[CPU准备包](../../../../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-01-preparation-001/README.md)保留；当前假设及接入位置见[计算服务](../../../model/mechanism/hardware_rules/COMPUTE-SERVICE.md)，本轮预算见[PLAN](../../../PLAN.md#stage-2-compute-a)。

## 1. 条件与生成程序

单CTA、单128线程数学WG，FP16输入／FP32累加，软件tile128×128×64。A K-major、B MN-major，SW128；固定六个预准备的输入槽，每槽32768 B，共196608 B。每Ktile沿`tile%6`取槽，四个不同K16偏移分别计算两个M64子块，共8条`m64n128k16`。首Ktile的各累加片段第一次使用ScaleOut::Zero，其余累加；首组commit，后续各组commit→wait1，最后wait0。

| Ktile | 分组 | WGMMA数 | FLOP | plain／trace |
|---:|---|---:|---:|---|
| 4 | 开发 | 32 | 8,388,608 | 各3个独立进程 |
| 13 | 开发 | 104 | 27,262,976 | 各3个独立进程 |
| 32 | 留出 | 256 | 67,108,864 | 各3个独立进程 |

seed固定3／22／41。共6条件、18进程；先导复用4 Ktile plain的一项，不另增条件。4／13开发冻结两个[窗口解释](../../../model/mechanism/hardware_rules/COMPUTE-SERVICE.md#2-两个有限解释与观测对应)的参数和32预测后，才采留出。plain取3个合格进程的clock64窗口中位数，32周期预测绝对相对误差≤5%为本轮窗口相容标准；竞争解释都通过时保留信息不足。trace另作扰动配对，窗口差超过5%时只约束trace程序。首轮GPU分配上限10分钟；不占用GPU等待CPU拟合。CPU准备限一版派生源、初次编译及最多两次定向修复，不展开B–E和其他指令组织。

A的活动计算帧为128线程；cfg_b完整kernel为384线程、215040 B SMEM，较A多出的18432 B包含17408 B epilogue张量及其他存储，见[构建查询与程序图](../../../model/mechanism/program/cutlass_program_graph.md)。A保留计算体和六槽输入布局，后续配对再核对新增参与者及资源。

## 2. 输入与完整结果检查

令`s=tile%6`、`p=seed%17`，输入精确存储为FP16：

```text
A[s,row,k] = ((7·row + 13·k + 3·s + 3·p) % 17 - 8) / 32
B[s,col,k] = ((11·col + 5·k + 7·s + 5·p) % 17 - 8) / 32
C0 = 0
reference[row,col] = Σ_tile Σ_k A[tile%6,row,k]·B[tile%6,col,k]
```

每个乘积是整数／1024；32×64项的绝对部分和上界131072／1024，FP32可精确表示。全部16384个输出按整数点积检查，正式输出保存为`output.f32`；Python另以六槽整数矩阵乘和使用次数独立复算。这个见证区分M64子块、K16偏移和槽选择；驻留环在窗口内不改写。

输出顺序为`((thread*2+m64)*64+j)`，thread=0..127、m64=0..1、j=0..63：

```text
row = 64·m64 + 16·(thread//32) + (thread%32)//4 + 8·((j//2)%2)
col = 2·(thread%4) + j%2 + 8·(j//4)
```

每进程先做1／2 Ktile完整非均匀检查，再按本条件暖机8–60次；末5个globaltimer窗口的样本CV≤2%时结束，随后一次正式调用。未收敛仍保存输出，标记performance_eligible=false。

## 3. 计时与trace

窗口前完成SMEM初始化、proxy发布、累加器初始化及编译器寄存器屏障。线程0读取globaltimer／clock64后CTA会合，循环内形成描述符、执行wgmma.fence与计算，最后wait0、累加器消费及CTA会合后取结束戳。完整输出导出与host校验在窗口后。报告同SM周期及globaltimer ns；CUDA event覆盖整个kernel，包含SMEM准备，单独保存。

plain不含循环内打点。trace只在最后Ktile记录`issue_before`、`commit_after`、`wait1_after`和`tail_wait0_after`，先留在寄存器，排空后写出；两种特化分别保存资源与SASS。本地CUDA13的`issue_before`时钟位于`WARPGROUP.ARRIVE`之前，覆盖fence及8条MMA发出。逐长度报告trace相对plain的窗口变化，再使用末段约束。等待返回对模型完成的约束见[计算服务](../../../model/mechanism/hardware_rules/COMPUTE-SERVICE.md#2-两个有限解释与观测对应)。

## 4. 准备、运行与交付

源码入口：[探针](../../../../../../microbench/gh200_resource_campaign/access_rules/probes/cfg_b_compute_chain.cu)、[准备与采样](../../../../../../microbench/gh200_resource_campaign/access_rules/cfg_b_compute_chain.py)、[CPU检查](../../../../../../microbench/gh200_resource_campaign/access_rules/test_cfg_b_compute_chain.py)。准备包固定源、CUTLASS 3.9.2头文件压缩包、条件和已有服务假设；编译时在临时目录解压头文件。

```bash
python microbench/gh200_resource_campaign/access_rules/cfg_b_compute_chain.py prepare \
  --output /tmp/cfg-b-compute-preparation \
  --cutlass-root results/gh200_resource_campaign/access_rules/20261006-r00-job734996/formal-v1/source/cutlass
python /tmp/cfg-b-compute-preparation/source/cfg_b_compute_chain.py compile \
  --output /tmp/cfg-b-compute-preparation --nvcc nvcc
python microbench/gh200_resource_campaign/access_rules/test_cfg_b_compute_chain.py -v
```

本轮已获免费ROMEO执行许可；在分配的GH200计算节点运行保存runner的入口如下：

```bash
python /tmp/cfg-b-compute-preparation/source/cfg_b_compute_chain.py run \
  --output /tmp/cfg-b-compute-preparation --split dev
python /tmp/cfg-b-compute-preparation/source/cfg_b_compute_chain.py run \
  --output /tmp/cfg-b-compute-preparation --split heldout --dev-freeze /path/to/dev-freeze.json
```

冻结文件schema为`gh200-cfg-b-compute-chain-freeze/1`，绑定`plan_sha256`、`build_identity_sha256`、`dev_summary_sha256`。`candidate_parameters`按`nominal`／`conditional`保存`a_cycle`、`q_cycle`，`predictions_cycle`保存对应的`k32_plain=a+32q`；`criterion`固定`unit=clock64_cycle/CTA`、`max_abs_relative_error=0.05`、`statistic=median_of_3_eligible_processes`。runner复用L2的命令收据、分配身份与单卡锁；首个留出进程前写入冻结身份，恢复时核对同一身份。旧成功采样只重放校验，不重新执行。

准备包中的源码直接通过9项CPU检查。本地CUDA13诊断编译及[静态审查](../../../../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-01-preparation-001/local-static-review/static-review.json)通过：plain／trace分别154／162寄存器，512 B静态SMEM，无stack／spill；每版本8条HGMMA均带`.tnspB`，保留首组跳过wait1及末尾wait0。构建采用临时glibc／libstdc++兼容头，完整命令、两次源码定向修复与环境错误记录已保存。实际目标CUDA12.9.41构建、数值与窗口核对已完成，结果如下。

### A实测与长度留出

[开发12进程](../../../../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-01-execution-002/dev/summary.json)与[留出6进程](../../../../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-01-execution-002/heldout/summary.json)使用同一GH200，全部性能资格通过。每进程的16384个输出由独立整数参考核对；活动SM保持，plain／trace分别154／162寄存器、196608 B动态SMEM、512 B静态SMEM、零local。下表周期与ns分别取各条件三个合格进程的中位数；trace扰动按同seed配对的周期相对变化取中位数。

| Ktile | 用途 | plain周期 | plain ns | trace周期 | trace扰动 |
|---:|---|---:|---:|---:|---:|
| 4 | 开发 | 2817 | 1664 | 2861 | +44 cycle／1.562% |
| 13 | 开发 | 7425 | 4192 | 7469 | +44 cycle／0.593% |
| 32 | 留出 | 17159 | 9536 | 17197 | +38 cycle／0.221% |

仅用4／13 plain中位窗口，[开发分析](../../../../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-01-fit-001/fit.json)及[冻结](../../../../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-01-fit-001/dev-freeze.json)得到两解释共同的`a=769、q=512 cycle`，32 plain预测均为17153 cycle。[留出评分](../../../../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-01-heldout-001/score.json)只读取该冻结，未重拟合或重选；相对实测中位17159 cycle的误差为−6 cycle／−0.034967%，满足原绝对相对误差≤5%。这支持该驻留窗口的4／13→32长度外推；两个解释预测重合，留出不能区分它们。

三种长度最后Ktile的trace软件段中位数均为：issue_before→commit_after 429 cycle、commit_after→wait1_after 28 cycle、wait1_after→tail_wait0_after 448 cycle、tail_wait0_after→窗口结束579 cycle。trace配对均在原5%扰动限内；这些区间包含打点及对应软件过程，wait返回只给计算完成上界，不作为裸MMA延迟或直接物理完成时间。

512 cycle/Ktile作为匹配驻留计算体的条件循环量保存；769 cycle的A复合固定项不接入GEMM，256 cycle提交提前量仍是原工作假设。原组合价格保持；独立长度外推通过不代表全GEMM C2/C3通过。B–E的实测与迁移继续按其登记推进。

## 5. B/C匹配交接准备

本轮[有限登记](../../../PLAN.md#compute-bc-preparation)接续A计算体，使用256线程、一个producer WG与一个数学WG；固定物理六槽196608 B，逻辑S4／S6。full采用普通arrival、到达数1，empty到达数128。首`min(stages,ktiles)`组在窗口前发布；后续producer等待对应empty后写入绝对代次并发布full。消费者完成full等待后核对代次，执行8条WGMMA、commit和wait1，128名消费者各贡献一次较早组的empty arrival，tail wait0后释放最后组。循环内无CTA会合。

B正常推进；同一二进制的C只在`tile=stages`第一次取得复用槽后延迟D个clock64周期。首批槽有预就绪依据，复用后的B与C分别由实际信号和等待区间判断路径。驻留操作数保持不变，输入数值按§2公式的`tile%stages`取槽；代次sideband检查控制交接。每进程另核1／2／`stages+1` Ktile的完整输出及代次。

trace保存逐组producer acquire／full信号前后、consumer线程0的wait／发出／commit／wait1及release，并记录四个warp的first-try成功位图。各128名消费者的arrival前后戳分别取最大值，最后必要信号位于`[max(before),max(after)]`；再取得槽的返回应不早于该区间下界。`try_wait`可能自行等待，其成功位保留为调用结果；线程0的进入状态由full信号与wait入口的括号判断，重叠记为`overlap`，语义见[PTX ISA 8.8](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait)。打点在共享存储中暂存，窗口结束后导出；plain不含循环打点，数值和代次检查保留。

入口：[派生探针](../../../../../../microbench/gh200_resource_campaign/access_rules/probes/cfg_b_handoff_chain.cu)、[准备／构建／离线校验](../../../../../../microbench/gh200_resource_campaign/access_rules/cfg_b_handoff_chain.py)。[B/C准备包](../../../../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-02-handoff-preparation-001/README.md)保存14个条件形状、42个未执行命令、无硬件时间价格的程序与计数见证及本地静态证据；D=2048为CPU示例。当时尚未冻结正式延迟和预测；实测轮已固定D=4096 cycle，并在开发前保存[完整预测](../../../../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-02-prediction-001/prediction.json)。

本地CUDA13诊断构建的plain／trace分别167／220寄存器、1152／12800 B静态SMEM，无stack／spill。首版因角色分歧后的累加器定义触发C7515，将8条WGMMA逐条排空；一次定向修复把初始化移到共同路径，恢复每组8条HGMMA、末条group提交、普通wait1与尾wait0。失败构建与修复版本均保存。B/C的线程数及资源与A分别登记，差值先作为程序组合响应；准备轮尚无GPU数值与窗口观测；后续实测结果如下，当前组合的计算与同步价格保持原版。

<a id="bc-measured"></a>

### B/C实测与停止

[执行包](../../../../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-02-execution-001/README.md)的30个开发进程与A同卡，目标构建、完整输出、代次和性能资格全部通过；[独立评分](../../../../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-02-score-001/score.json)重新核对491520个输出元素及90份短长度检查。正式D=8q=4096 cycle只作用于首次复用，冻结形式使用q512、commit_lead256及零额外控制价格，不迁移A的769 cycle固定项。

| plain响应 | 实测cycle | 冻结预测cycle | 原5%响应判别 |
|---|---:|---:|---|
| S6／13的C−B | 908 | 1792 | 未通过 |
| S4／13的C−B | 2373 | 2816 | 未通过 |
| 上两项的跨stage差 | 1465 | 1024 | 未通过 |

该具体形式按原停止规则收口，32 Ktile的12个留出未执行，没有重新拟合或选择候选。15个同seed trace配对扰动为29.001%–42.281%，全部超过5%，端点仅解释trace程序。代表trace能核对首次延迟、full信号与wait入口区间，以及128名消费者最后必要arrival的释放区间；first-try成功不能倒推进入时已就绪。A→B同时增加producer、full／empty协议、代次检查和资源帧，B−A及长度复合差不拆为单barrier或每Ktile价格。

## 6. D原生TMA短链准备

D沿用B/C的256线程与消费者计算／释放顺序，固定六个物理槽196608 B、逻辑S4／S6。producer等待empty后执行一次`arrive_and_expect_tx(32768)`，再发A16 KiB和B0/B1各8 KiB；full到达数1，empty到达数128。采用CUTLASS 3.9.2的StageCount6工厂，CPU逐一比较应用pointer swizzle后的98304个half地址，与B/C布局零差异、无重复地址。每槽A覆盖8192 half，B0/B1各4096且无重叠；请求形状与地址见[准备包](../../../../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-03-tma-preparation-001/README.md)。

唯一主窗口覆盖首个acquire／事务登记／TMA到tail wait0、末槽释放、累加器消费与CTA会合；所有Ktile搬运均在窗口内，4 Ktile含12次请求。性能输入按`tile%stages`沿用B/C的环；独立正确性调用用`source_tile=tile`及逐代增加的正dyadic值，分别核1／2／S+1／正式长度的完整16384输出和128人代次。缓存协议为同一环的重复调用暖机；本轮未增加P0，D与B/C之差按完整程序响应保存。

trace保存事务登记、A发出、B两笔共同发出区间、consumer wait返回、8条WGMMA／commit／wait1与全部128人释放。每组full完成区间取B pair发出前戳到consumer wait返回，producer的pair后戳可晚于返回；first-try保存调用结果，进入状态只报告早于B pair发出或尚无法区分。首等待返回／首计算前缀直接由同次时间戳给出。

入口：[派生探针](../../../../../../microbench/gh200_resource_campaign/access_rules/probes/cfg_b_tma_chain.cu)、[准备／构建／记录校验](../../../../../../microbench/gh200_resource_campaign/access_rules/cfg_b_tma_chain.py)。有限形状与预算见[PLAN](../../../PLAN.md#compute-d-preparation)；[D准备包](../../../../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-03-tma-preparation-001/README.md)保存未执行命令、请求／程序见证、两种独立整数参考、15项CPU检查与完整构建记录。本地CUDA13诊断构建plain／trace分别167／246寄存器、1152／14848 B静态SMEM，无stack／spill；每组8条HGMMA，普通wait1及尾wait0，producer循环体为3条UTMALDG.3D。首次B分区断言失败及两次定向修复均已保存。

CPU准备轮GPU执行0次。后续实测仅执行原清单开发部分，结果如下；计算、同步与输入价格未替换。

<a id="d-measured"></a>

### D开发实测

[执行包](../../../../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-03-execution-001/README.md)的18个开发进程与A／B/C同卡，目标构建、布局、数值与性能资格通过；[独立评分](../../../../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-03-score-001/score.json)核对ring与独立source代次输入共589824个完整输出元素、72份代次正确性报告。保存每进程descriptor编码输入与目标driver生成的128 B opaque hex；本轮核对源字段与实例身份，没有在CPU上重新编码或解码TensorMap位。

| 条件 | plain中位cycle | plain中位ns | 相同B条件的D−B cycle |
|---|---:|---:|---:|
| S6／4 | 4554 | 2528 | 796 |
| S6／13 | 11489 | 6336 | 711 |
| S4／13 | 11552 | 6368 | 629 |

S4−S6的13 Ktile plain响应为+63 cycle／32 ns。D主窗口含初始及循环全部TMA，B首S槽预填，差值只表示同卡完整程序响应。9个trace配对扰动38.933%–47.823%，全部超过5%；B pair括号是两笔请求的发出区间，consumer wait返回只给聚合完成上界，不能把pair后戳当作full就绪。重复调用暖机不证明纯L2命中。本轮`model_prediction=null`，未设新的预测通过阈值、未拟合，32 Ktile的6个留出未执行。

## 7. E真实pingpong输出准备

E薄封装真实CUTLASS 3.9.2 cfg_b工厂，固定S6、M384×N128×K832、调度SM数1，单CTA处理三个输出。实际工厂为384线程、128线程数学组、215040 B动态SMEM，mainloop／epilogue张量分别196608／17408 B；epilogue的StagesC4、StagesD2、ReuseSmemC=false、DelayTmaStore=true。数学交接复用`OrderedSequenceBarrier<2,2>`、每组128 arrival；epilogue沿原64×32 FP32工厂与两源槽推进。

直接复用[程序图实现](../../../../../../microbench/gh200_resource_campaign/access_rules/cfg_b_program.py)展开三输出：39输入组、117逻辑TMA请求、312条WGMMA、24输出子块。输入绝对组号`g=13j+k`，第二／三个输出首组为g13／g26，对应slot1／generation2与slot2／generation4；输出store计数则从0／8／16开始，两源槽按子块复用。输入为已有period17非均匀dyadic，完整49152个输出由独立整数点积核对，分配首尾另保留32个sentinel。

plain使用原工厂；trace复用冻结kernel／collective overlay，仅加四处调用边界，保存FIRST、mma返回、数学arrival后、MAIN_END、epilogue许可、store返回、store_tail返回及输出arrival后。下一WG可在上一WG的MAIN_END之前开始，leader的arrival后戳也可晚于对方推进；校验只采用真实必要前驱。`store_tail`的`.read`等待约束输出源复用，全局D在stream完成后完整检查。性能为整个kernel的CUDA event窗口，另报cycle/ns软件端点；E包含真实启动、调度和13组global源，与D分别按各自输入解释。

入口：[薄封装探针](../../../../../../microbench/gh200_resource_campaign/access_rules/probes/cfg_b_pingpong_chain.cu)、[准备／构建／记录校验](../../../../../../microbench/gh200_resource_campaign/access_rules/cfg_b_pingpong_chain.py)。有限条件见[PLAN](../../../PLAN.md#compute-e-preparation)，[E准备包](../../../../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-04-pingpong-preparation-001/README.md)保存源码、原overlay及四处插入、完整参考和14项CPU检查。本地CUDA13诊断的plain／trace均168寄存器、无stack／spill，两个CPU工厂查询一致；初编失败、两次定向修复及独立包回放缺工作列表依赖的失败均已保留，补齐后回放通过。

CPU准备轮GPU执行0次；后续实测沿原两形状执行，结果如下。

<a id="e-measured"></a>

### E开发实测

[执行包](../../../../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-04-execution-001/README.md)的6个开发进程与A–D同卡；[独立评分](../../../../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-04-score-001/score.json)核对真实工厂、overlay重建、实际grid／资源、完整294912个输出元素及192个guard值，数值与性能资格全部通过。冻结工作列表为39输入组、117逻辑输入请求、312条WGMMA、24输出子块；这些是由程序和输出／工作列表核对的动态需求，不是硬件动态计数器读数。

完整kernel的CUDA event中位时间为plain **22.495999932 µs**、trace **23.104000837 µs**。三个同seed配对中2个扰动超过原5%界限，因此八端点只解释trace程序，不能用两个中位数的约2.70%差替代配对门槛。三个seed中，output1的FIRST都早于output0的MAIN_END 47 cycle／32 ns，同时晚于其mma_return 421–430 cycle／224 ns；实际观察支持区分数学交接与尾部完成边界。`MAIN_END`是mma_tail后的软件完成上界，`store_tail_return`约束输出源读取／复用，stream完成后才核完整global输出；leader的arrival后戳不是最后128次arrival的完成戳。

E包含真实调度、连续global输入和输出整理，D使用六源环；E−D不能直接定价输出。本轮无正式完整时间预测、拟合或独立留出，未替换组合价格。A的q512仍仅支持匹配驻留计算体，B/C具体形式的失败及D/E诊断不能改写为全GEMM独立C2/C3通过。
