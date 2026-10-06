# GH200 访问规则：共享实验状态

主对话维护总览、R00、R02和公共接口；执行对话只更新自己章节。通过`update_status.py --section <ID> --body <文件>`取得短锁、重新读取并替换本节，不覆写整份文件。静态分工见[总计划](README.md#并行实现分工)。

<!-- BEGIN:OVERVIEW -->
## 总览（主对话维护）

- 主目标：FP16输入、FP32累加和输出；CUDA12.9、sm_90a。BF16/FP8按计划扩展。
- R00：30点/216个正式进程已完成，15个最小验证点已完成。本地回收、离线重算与独立B审查通过，按有限条件接受。
- 本次设备：romeo-a041，GPU-ec947ba1-3e15-9860-163d-330d6c66d1b4，GH200 120GB。不同UUID数据分别保存。
- 分配734996已到期回收；正式运行在此前正常退出（formal_exit=0）。执行对话不得进入该作业。
- 本地归档：`results/gh200_resource_campaign/access_rules/20261006-r00-job734996/formal-v1/`，844份源码和3个二进制身份已核对。
- 远端归档：`/gpfs/scratch/hibouwu/gh200_access_rules_20261006/{formal-v1,smoke-v9}`；早期失败目录smoke-v1至v8保留。
- 下一步：A实现R04/V01，B实现R01/R06，C实现R03/R05；主对话完成R00离线复核，继续集成和验收。

执行对话：B [GH200 依赖与资源实验](codex://threads/01a11139-05bf-7a51-8580-b611621a7704)；C [GH200 片上访问与异步运输实验](codex://threads/01a11139-0d1f-7382-9324-05697a099eef)；A [GH200 联合服务与组合验证实验](codex://threads/01a11139-126f-73d0-a66f-5090328303a4)。

验收：R00/R01/R03/R05/R06条件观测通过；R04的16个LDS家族点因动态计量外提/合并返修，WGMMA16点按实际等待限制接受；V01观测可接受，0项数值预测，预测验收未完成。详细证据见[ACCEPTANCE](ACCEPTANCE.md)。
<!-- END:OVERVIEW -->

<!-- BEGIN:R00 -->
## R00 — 主对话

状态：30点实测及本地离线复核通过，独立B审查已通过，按记录的有限条件接受。

30点/216正式进程，最大工作率CV约2.98%；单CTA点只有3次的初步重复性证据。代码为`run_r00.py`、`analyze.py`及`probes/r00_*`。

已修复：sm90a生成参数、FP8属性宽度、CUDA库路径、描述符K范围、CUTLASS无C输入资源。当前CUTLASS为128×256×64、cluster2×1×1、384线程、4stage、ElementC=void、epilogue128×32，SMEM231424B；旧有C构造264192B超过上限232448B，失败记录保留。

离线复核：844份源码、3个二进制哈希、216正式样本及全部WGMMA正式输出已核对，30点统计复算一致。独立B审查通过。公开分析器局部变量覆盖已修并加回归；原数据不受影响。K/K与CUTLASS K/MN、寄存器重分配及非相邻缓存采样边界已补文档。执行对话不改已测R00源码或原始归档。
<!-- END:R00 -->

<!-- BEGIN:R01 -->
## R01 — B：依赖与资源

状态：37点/146正式进程完成；主对话条件观测验收通过，未验证完整预测。

- 已实现：`run_r01.py`、`analyze_r01.py`、`probes/r01.cu`与`r01_support.hpp`。1/4流有限矩阵、逐步WGMMA commit/wait0、ldmatrix地址依赖与空/消费者控制齐全。
- 编译/数值：作业735060，romeo-a043，GPU-201f9d2a-b55f-2a3b-b355-3abf04c2bc88，CUDA12.9、CuTe v3.9.2、sm90a。38短进程/64384输出；正式146进程，另12控制，合计196进程/402080输出CPU重算通过。774份源码与二进制hash核对通过。
- 采样：最大正式CV0.78%，全部正式预热达到末5CV≤2%。FFMA/add独立流近噪声共同补10，大global512步按CV补10；全部保留，无失稳删除。
- 结果：WGMMA N=64/128/256每步增量105/137/201cycle，ldmatrix58，shared30，global小/大287.49/627.17。均含循环/地址转换，WGMMA含commit/wait0。FFMA/add均29且1/4流窗口近似，循环推进影响强，不能作为裸依赖延迟。
- SASS：19个计时specialization循环体/消费者检查通过，窗口无LDL/STL；三种N均一HGMMA+DEPBAR0；LDSM及load输出确实馈入下一地址。add实际VIADD/IADD3混合，不指定执行部件归因。
- 本地：`results/gh200_resource_campaign/access_rules/20261006-b-job735060/formal-r01-v1/`，完整报告/图/规则与CPU复核代码已保存。远端原始scratch归档同名；持久副本 `/gpfs/projet/r260073/gh200_access_b_20261006/formal-r01-v1/`。
- 缺项/下一步：单CTA3次重复仅初步重复性证据；本GPU与R00不同UUID，不合并拟合。V01与完整kernel预测另按A/主对话的匹配证据验收；不宣称完整预测通过，无公共接口修改需求。

- 分配735060：完整本地/持久远端源码、二进制、结果hash核对后已释放；不进入或续写原作业。

- 验收来源：主对话2026-10-06反馈，完整输出重算、源码/二进制身份、工作量、时钟包络与计数均通过；只接受本组有限条件的序列/资源观测，无需重测有效样本。
<!-- END:R01 -->

<!-- BEGIN:R02 -->
## R02 — 主对话：条件扩展

状态：默认不执行。其他对话记录异常证据，不自行启动完整RF扫描。
<!-- END:R02 -->

<!-- BEGIN:R03 -->
## R03 — C：片上访问与异步运输

状态：18点/54正式进程完成，源码、编译/SASS、短检查、样本和CPU报告已回收；主对话条件观测验收通过；预测适用性未验证。

- 实现：`run_r03.py`、`analyze_r03.py`、`probes/r03.cu`；每轮8次访问、warp独立8槽轮转。1/2/9轮非均匀输入18点通过。
- 编译：CUDA12.9，原生sm_90a；SASS每轮8条LDS/LDS.128、STS/STS.128、LDSM.16.M88.4或STSM.16.M88.4，未把初始化和读回指令混计。
- 采样：每点3独立进程，共54；共同44693轮，最大CV约0.253%，末5次预热全收敛。CPU重算124800个输出通过。
- 分配：735059，romeo-a041，GPU-ec947ba1-3e15-9860-163d-330d6c66d1b4；R03结束后同分配串行运行R05。
- 证据：本地`results/gh200_resource_campaign/access_rules/20261006-c-job735059/{r03-smoke-v1,r03-formal-v1}/`；远端`/gpfs/scratch/hibouwu/gh200_access_c_20261006/`。报告、rules.json、cases.csv和plots/service.svg齐全；原始归档源码哈希保留，当前分析入口新增图。
- R04代表点：128线程、连续lane16B、每轮8次、8槽轮转，load101.132、store92.564、ldmatrix115.066、stmatrix87.613请求B/cycle/CTA。load窗口包含4寄存器checksum与CTA会合，不能与无消费窗口直接拼接；全部正式值可读rules.json。
- 适用域：1/4/8warp的条件查表；8warp仍有增量，不宣称已经严格饱和。主对话已复算输出/工作量/时间并核对来源与二进制；后续用于匹配条件，预测适用性未验证。
<!-- END:R03 -->

<!-- BEGIN:R04 -->
## R04 — A：联合服务与验证

状态：动态工作量阻断已修正，选择性采样和CPU复核完成；待主对话验收。旧LDS/CVT条件表无效，不再使用旧“慢5.36%/9.82%”结论。

- 旧数据：`20261006-A-job735062/r04-formal-v3`的16个LDS家族暂停为`invalid_dynamic_work`。原固定地址LDS的n_A内层被消掉，恒定输入CVT被外提。原始源码、SASS、样本和数值结果保留；`r04-final-review/dynamic-work-qualification.json`与报告首段明确撤回。CPU正确与静态窗口内出现不证明动态次数。
- 修订LDS：`ld.volatile.shared.v4.b32`，串行/A-only每单元末次读真实消费并CTA同步后结束A；这些成本包含在窗口。CVT四独立输入逐action精确增加1/256，RNE FP16位模式逐次进入16bit校验和。B是FADD准备+CVT+checksum，不是裸转换服务。
- CFG：`build/dynamic-work-cfg.json`保存每kernel action/repeat回边、counter更新/比较、目标PC和实际目标顺序。串行/基线A loop为1 LDS×n_A；B loop为4 FFMA/CVT×n_B/4。交错1:4为1A+4B×32；1:1为4A+4B×20；4:1为16A+4B×8。8个编译kernel均通过目标位于双层循环且分支计数一致的门禁；时间戳uniform LDS读取不计为输入需求。源级AB可能被调度重排，按实际SASS序列解释。
- 跨WG：先尝试八条单块async PTX仍有C7520，失败编译证据保留。最终保留真实串行条件，每条显式fence→MMA→commit/wait0，消除C7520；不声称恢复了理想wait1。实际等待和工作量由SASS/CFG解释。
- 新采样：15个受影响LDS/CVT点×3=45正式进程；7个受影响跨WG点×3=21。短检查、8197单元代表长检查、完整输出CPU重算全部通过，误差0；LDS最大CV约0.000223%，跨WG约0。新SASS与旧版的同WG4 kernel及FFMA-only kernel逐字相同，没有重跑这些未受影响点或R00。
- 新环境：735138，romeo-a041，GPU-7c184a2e-41ea-3d2b-df5b-1699c95fc1fe，CUDA12.9/sm90a。与旧099dda…和R00 ec947…分开，不合并拟合。新点缺匹配B-only时max/sum保持missing，不搬旧UUID基线补空。
- 本地：`results/gh200_resource_campaign/access_rules/20261006-A-fix-job735138/{r04-volatile-formal-v1,r04-serial-formal-v1,r04-volatile-review,r04-serial-review}/`。源码、精确CUTLASS头文件、命令/编译日志、PTX、SASS、二进制、样本/输出、CFG与CPU重算器身份已保存。
- R03最小代表关系：本次已重测R04匹配的128线程/连续lane16B/固定地址/volatile/末次消费A-only代表点。同UUID本组联合点可使用此条件校准，不套用R03的8槽轮转/每次checksum/另一UUID的101.132 B/cycle；仍不宣称裸物理SMEM带宽或次数线性缩放。
- 保留范围：新数据是条件观测；真实FP8提升、理想异步跨WG和跨环境完整联合模型尚未取得。下一步主对话复核CFG与计数边界。
- 分配回收：735138的GPU采样已结束，全部必要证据完成本地回收及CPU/SHA/CFG核对后释放；squeue已无该作业。
<!-- END:R04 -->

<!-- BEGIN:R05 -->
## R05 — C：片上访问与异步运输

状态：40默认点+16控制完成实现、真实sm90a编译、检查、有效采样和本地完整复核；主对话条件观测验收通过；预测适用性未验证。

- 有效正式进程364=302目标+62控制：修正版A62目标+62控制，B/C/D/E240目标。A的WG0/32配对及控制补至10；B/C全部补至10；D/E各点10。共同2048轮。最大窗口CV1.988%，末5次预热全部收敛。
- 原A48目标+48控制（96正式进程）计量失效：加法被合并、WG等待提前；数值通过不能替代SASS合法计量。这些样本完整保留，原报告/规则已标`invalid_sass_schedule`，不与修正版拟合。
- 实现：`run_r05.py`、`analyze_r05.py`、`probes/r05.cu`、`probes/r05_wg_inline.cuh`。实际WG发出至wait之间0/32/128/512条LCG IMAD已按归档SASS逐实例核对；源读完成、消费/复用和最终输出完成分开。
- 检查：1/2序列短输出（A/B完整输入trace、C旧代/新代源、D最终完整SW128 tile、E全GMEM有效区/padding），全部正式/pilot数值、源码/二进制/数值文件哈希本地复核通过。旧A只作数值与失败计量证据。CPU新入口及图单独保存在analysis-revision，不覆写冻结源码。
- D：无计算消费者条件的供给观测，不是所有组织策略的严格上限。当前目标4stage输入192KiB；stage2/4固定预留此物理容量，活动在途量96/192KiB，保持1CTA资源条件。驻留上限1而非预设实际驻留；10进程均覆盖132SM。shared stage2/4中位55.454/55.475 B/cycle/CTA；independent15.264/15.261，需求48。实际访问15728640/253034496B，复访320/39Ktile；不证明物理L2命中，也不代替联合WGMMA/cluster。
- E：128/144/160/256B窗口均值950714/1314624/966899/898909ns；144B约慢38.3%，160B接近128B。仅16-bit运输条件代表，不能当FP32 epilogue服务。
- 本地完整证据：`results/gh200_resource_campaign/access_rules/20261006-c-job735059/{r05-smoke-v1,r05-formal-v2,r05-a-formal-v5}/`，报告、rules.json、cases.csv、图、source/build和全部原始数值均齐全。压缩总包52MiB及SHA256保留。
- 失败：GPFS r05-formal-v1复制头文件时quota不足，无GPU采样；失败日志保留。正式原件曾在节点/tmp，已全部回收。job735059已退出squeue，sacct为CANCELLED（0:0），分配已释放。
- 缺项/下一步：主对话条件观测验收通过，预测适用性未验证；A对话用于R04/V01匹配。没有FP32输出精确运输、联合竞争、cluster或完整GEMM预测检查。观测不升级为预测通过。
<!-- END:R05 -->

<!-- BEGIN:R06 -->
## R06 — B：依赖与资源

状态：6点/39正式进程完成；主对话条件观测验收通过，未验证完整预测。

- 已实现：`run_r06.py`、`analyze_r06.py`、`probes/r06.cu`。128线程8链，每lane16384FFMA，额外值计时前准备/后消费，整卡528CTA。
- 编译/数值：同作业735060/romeo-a043/GPU-201f9d2a-b55f-2a3b-b355-3abf04c2bc88，CUDA12.9、sm90a。12短进程/3656448输出；39正式，合计51进程/21914496输出CPU重算通过。7份源码和二进制hash核对通过。
- 采样：最大正式CV0.26%，全部正式预热窗口达到收敛；每档单CTA3进程、整卡10进程。
- 实值：62/94/128寄存器/thread，0spill；CTA/SM API上限8/5/4。单CTA61671/59639/59639cycle；整卡55017.6/52310.4/52537.6ns，即40.252/42.336/42.153有效TFLOP/s，仅128线程FFMA探针。
- SASS：三个资源specialization计时内各8条FFMA、循环opcode序列相同、寄存器编号不同，无LDL/STL；消费者/barrier在窗口内。所有正式整卡点每SM均观测4个计时区间重叠，未测到并发减少；62档单CTA已更慢，不能将整卡差全部归因于驻留。
- 落盘失败保留：scratch用户20GiB硬配额使94档整卡trial-04输出不完整，未接受。有效样本hash核对后复制到同节点/tmp，保留相同二进制并只补缺失进程；有效retry-1及resume源码/hash均保存。运行入口与CPU检查增加长度/完整数量校验，空/截断见证负检查通过。
- 本地：`results/gh200_resource_campaign/access_rules/20261006-b-job735060/formal-r06-v1/`；持久远端 `/gpfs/projet/r260073/gh200_access_b_20261006/formal-r06-v1/`。scratch旧formal-r06-v1仅为失败/部分归档，不当完整结果。
- 缺项/下一步：128线程FFMA不能替代384线程目标TC组合；当前grid未压满低寄存器档资源上限，无驻留下降惩罚规则。不启用R06-A/setmaxnreg扩展，无公共接口修改需求。

- 分配735060：完整本地/持久远端源码、二进制、结果hash核对后已释放；不进入或续写原作业。

- 验收来源：主对话2026-10-06反馈，完整输出重算、源码/二进制身份、工作量、时钟包络与计数均通过；只接受本组有限条件的序列/资源观测，无需重测有效样本。
<!-- END:R06 -->

<!-- BEGIN:V01 -->
## V01 — A：联合服务与验证

状态：至少一个新未观测单CTA长度的数值预测已验证；全部20点的通用组合/整卡预测仍未完成。

- 旧20点/137进程仍只属于观测，原预测20项null，没有反填7/19/47。主对话补充的CUTLASS布局/资源边界保留：固定CUTLASS A=K/B=MN、tnspB、动态producer40/consumer232；受控TC是K/K、统一154 registers，不能以同N迁移。
- 修订TC：单块async四描述符试验仍有C7520/C7517；最终使用真实显式串行fence→每条K16 MMA→commit/wait0。编译无C7520/C7517，目标154 registers/thread、小TC58，无spill；最终没有在途group。384线程/4输入stage/192KiB输入区及末尾两块64KiB复用输出保持，完整输出bulk wait0结束。
- 分段规则：同CTA clock64连续预填/主循环最终排空/完整输出；三段之和严格等于完整窗口。主循环整段包含TMA ready、WGMMA、槽与CTA控制，不重复叠加其他规则。
- 新校准6/14/30各3进程，不使用原7/19/47拟合。预填979 cycle、K6主循环9052、每额外4 tile主循环增量6270.5、完整输出8959；校准残差最大约0.105%。适用域：128×256×64、384线程、4输入stage、K/K、repeat-prepared dyadic输入、显式串行、单CTA、Ktile=2(mod4)的整段生命周期长度外推。不是独立原子服务任意组合模型。
- 新留出38：预测文件先只创建保存，预测69154 cycle/CTA；运行前核对UUID、probe源码、目标完整SASS，3个记录的预测SHA256一致。实测69119–69228，中位69122，CV0.0898%；误差+0.0463%，完整输出CPU误差0。38从未用于校准，也不是旧已知检查点。
- 选择性已知检查：只重跑受影响的small/target TC ×7/19/47×单CTA/整卡共12点/78正式进程，误差0，最大CV约1.507%。这些明确作为已知检查，不称留出。未重跑LDS组合、完整CUTLASS或R00。
- 环境：735138，romeo-a041，GPU-7c184a2e-41ea-3d2b-df5b-1699c95fc1fe，CUDA12.9/sm90a；所有GPU工作串行，未改频率/功率或重试NCU。旧099dda…、R00 ec947…不参与本次拟合。
- 本地：`results/gh200_resource_campaign/access_rules/20261006-A-fix-job735138/prediction-validation.md`、`prediction-k38-v1.json`；原校准`v01-calibration-v1/`、新留出`v01-heldout-k38-v1/`、CPU review与已知TC检查`v01-tc-checkpoints-v1/`。每次实际源码/精确头文件/二进制/编译命令/日志/PTX/SASS/完整输出已保存。
- 未闭合：理想异步TC、独立基础服务的广泛组合迁移、整卡共享供给/波次/缓存、CUTLASS MN布局/动态寄存器/cluster2×1×1及L3调度。新预测只支持当前明确域，不凑完整kernel预测。下一步主对话验收该窄域新长度预测与真实串行计账。
- 分配回收：735138的GPU采样已结束，全部必要证据完成本地回收及CPU/SHA/CFG核对后释放；squeue已无该作业。
<!-- END:V01 -->
