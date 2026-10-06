# S16 短控制器与完整保全接口草案

状态：controller A草案与CPU检查，尚未实施可执行run/storage B；run主动拒绝，不查询设备或启动目标。已审S16 probe、profile、actual source-B和r5短core保持各自资格；本草案不能单独放行GPU，也不借S15旧controller A3门禁。

## 有限执行和终态

合同36名义case，共用唯一profile `stage_short_1_2_5_33_v1`、topseed3，进程内部固定四次I1/2/5/33，不改probe/source/profile、不重新编译设备。实际device query的必要容量分类必须为32可进入短诊断坐标和4个S4R4容量拒绝坐标；不同能力布局需重新审查，不替换S/R/Q或静默删点。

四个拒绝点保存独立controller terminal `resource_reject_before_launch`，绑定actual device/query/source-B、所需dynamicSMEM与实际optin上限、target_launches=0。它们不调用case binary、不初始化diagnostic run目录，也不创建成功数值/B3证明；再次出现只重核同条件拒绝证据。剩余32进程合计128目标launch，全部合法短仍需独立B3。

执行顺序先取最大完整raw坐标 `gmem_to_smem_16kib_s1_r4_all_gpu`，再沿原合同顺序执行其余35点。index为本controller固定0..35顺序，contract_index保留原索引，第一点原索引9。每个Slurm slice默认最多1–4进程，半开区间start/stop有界；通过已审case只audit，不多运行一套“max测试”。第一进程本身就是原计划坐标，完成后继续剩余范围。

## 查询、身份与锁

首次与每个新分配使用实际S16已独审binary的只读device入口。wrapper须先validate_gate当前S16 source-B和自己的controller/storage B，再核binary SHA为该门禁唯一actual probe，检查有效单GPU Slurm、当前UUID/driver/compiler/tools/uid；30秒bounded、GPU UUID锁与注册/清理收据必需。query保存完整stdout、device JSON、binary/source-review SHA、实际job和process receipt，不增加target。不能拿历史132SM例子代替此次query或用CLI headroom数字替代quota。

family首次冻结environment/device身份和全部probe/contract/profile/adapter/SASS/controller/storage文件SHA。新分配同条件才能继续，source/controller/UUID变化不能续写原run。各分配query/Slurm记录分别保存；相同device事实不混合改变driver/compiler/uid的环境。family/controller锁防两个slice冲突，原诊断core继续suite/output锁和GPU UUID锁；文件锁不声称系统级独占。实际r5 core引用须用当前独审门禁，不复用其他family旧路由。

A代码只验证独立wrapper未来提供的query binding，没有执行query helper；上述锁/实际查询/启动行为必须在source B实施并测试后才准入。

## 每进程完整raw上界

令B为实际grid的必要容量上界，S/R与Q=16384固定，J=4，ΣI=41。使用真实device查询和实际编译寄存器32/40、static/local0，oneCTA B=1；cluster无关；allGPU B_upper=sms×min(4,floor(SMEM_per_SM/dynamic),floor(registers_per_SM/(regs×128)),16)。这是必要容量上界，不是实际occupancy或驻留。actual API grid仍由原host按相同条件记录。

| 工件 | 字节数 |
|---|---:|
| trace四次总和 | B×41×R×Q |
| final共享槽四次总和 | 4×B×S×R×Q |
| global完整环及guards四次 | 4×(B×32×R×Q+32) |
| lifecycle全部word，包括opaque token | B×41×8×2×4 |
| counts四次 | 4×B×12×2×4 |
| stamps四次 | 4×B×5×2×4 |

opaque token只是domain-only计量，仍全部占raw空间，不能从存储预算扣除。有效运输字节是B×41×R×Q，不叠加reset/capture/export或四次global保存。

以明确标注的历史132SM/232448B-optin快照为CPU例子：最大B_upper=396，raw=4,490,994,944 B（约4.18GiB）；32合法case总上界32,795,974,400 B。S15默认4GiB窗口不足容纳第一案例，因此S16草案独立采用6GiB resident窗口；不改变任何S15常量。部署/source/snapshot和64MiB每新进程metadata余量另计，完整seal副本按raw+1%+64MiB保守留量，不假设压缩比。

## 最早预算验证和配额

storage/source B首先在目标ARM环境用同一实际artifact writer，对实际device上界生成的最大完整工件形状测CPU写出/哈希/元数据时间；保留全部输入规则、字节/进程收据。它没有target kernel，不替代GPU数值。若仅writer下界已超过固定30秒诊断预算，先停并报告 `budget_preflight_not_passed`，不消费第一GPU目标、不擅改profile/超时、不写成GH200不支持。通过这个必要预算检查后，最大合法进程在30秒原诊断预算内进行第一次真实完整短验证；其GPU/CPU比较/归档时间仍可能失败，先保留失败调查，不继续扩大case。

每个新target前新query实际scratch quota，receipt≤60秒、同uid、filesystem覆盖本部署、原stdout SHA绑定，预留≥512MiB日志/输出余量。预算同时包含当前实际used（部署/binary/snapshot/住留raw）、下一进程完整raw+metadata和最坏新增seal副本。超窗口、头寸不足或receipt过期，在下一target初始化之前checkpoint；未执行case保持未执行，不错误转为失败数值或成功。source B要用真实mmlsquota语法验证wrapper，而不是接受任意CLI数字。

## 恢复与完整保全

resident passed只audit；pending/checkpoint且无attempt可resume；已有attempt、failed、started_unknown或初始化残缺先离线处理，不盲跑。已passed run缺展开目录不允许新target。

S16保全使用已审generic PackedEvidence/完整transaction机制，namespace为suite/tma_stage_request/short-v1-CASE--PROFILE；新S16 ledger、pack路径、controller锁与gate名称各自独立。S15的运输语义可复用，但它的固定family/68×3/profile及旧门禁不移植。

CPU草案的archive判据要求完整regular-file manifest闭包、原spec/raw/final state、固定case/profile/seed/四target元数据、archive/index/closure身份，remote+offhost两位置逐member完整解码SHA收据；resident缺失仍archive_audit，不启动GPU。CPU fixture只测试这些拒绝边界，不声明发生过真实remote/offhost运输，不授B3。

source B后续实施有限命令query-device/query-quota/seal/collect/verify/显式retire；每次封包前后精确成员与SHA一致，禁止links/额外/缺失/损坏工件。只有remote和offhost完整验证、真实origin身份和ledger发布后才能显式retire本次scratch目录。retire必须检查无active/未清理进程与精确run namespace，不删除canonical历史。原run原文保留直到双位置核实；无法保全则停止增加住留raw。packed过程只保全字节，完整数值仍由独立S16 B3重放。

## A接受与后续B

当前CPU11方法验证36/32/4、128launch计划、max-first、完整byte公式、6GiB与最坏seal预算、query的binary/UUID/job/receipt绑定负例、fresh quota、source/controller/UUID漂移、capacity零target、missing resident不重跑、完整archive解码和run草案拒绝。A没有实际query、ARM writer、quota或GPU证据。

独立A通过后才实施run和S16 storage有限wrapper。source B必须核当前r5 CLI/gate闭包、完整身份与两锁、fresh quota/budget检查、首最大case预算、非0停止、真实small pack/offhost逐member解码/retire负例与复现；最后才允许固定短GPU。完整32合法case+4拒绝终态及独立B3后才讨论正式采样。

## A r2 的归档限定

归档只接纳单个首次成功 `attempt_00`，拒绝额外attempt、failure summary/manifest、failed或未清理的登记。原snapshot contract/profile/policy、spec诊断角色、原environment/device及initial allocation一起核对；raw必须恰为device和validation两行，四次target严格ordered I1/2/5/33、固定input profile、128threads、同一合法blocks与实际device/resource匹配。原binary SHA、固定process argv、30秒成功receipt、stdout/stderr SHA、历史active PID/PGID/registry身份、闭合cleanup receipt和summary evidence必须一致。归档内同UUID、同host/boot的已记录GPU登记不得时间重叠；此处不宣称看到未记录的系统进程。

完整成员逐块解码/hash，metadata缓存仍受PackedEvidence的16MiB单对象／64MiB总缓存限制，4.49GB原始数组不整体装入RAM。数组只核保存字节/shape/SHA，不在controller中重做numerical B3。

新增持久小pack由实际CPU CUDA-shim host的四次执行输出及完整uint32工件产生；process为真实CPU子进程，GPU registry与remote/offhost收据是显式标注的协议fixture，没有真实GPU登记或双位置运输。初审17工件与receipt精确保存在initial-r1，两个旧review记录从root闭包修订的initial镜像按原SHA恢复；r2绑定当前3eb21e48 source-B与1c682f r5记录，原源码未改。
