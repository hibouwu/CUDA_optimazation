# GH200 分工与当前状态

问题见 [PLAN](PLAN.md)，当前目标见 [GOALS §5.4](../../../GOALS.md#current-scope)，顺序与依赖见[项目 PLAN](../../../PLAN.md#1-近期顺序)；测量与评分规则只在 [RESEARCH-GUIDE](RESEARCH-GUIDE.md) 维护。各轮登记、结果与结论在对应实验记录中，入口为[按问题索引](../README.md#research-questions)；原始轮次见[结果目录索引](../../../../../../results/gh200_resource_campaign/access_rules/README.md)。

## 分工

| 写入者 | 归属 | 相关的现行任务 |
|---|---|---|
| 主集成 | PLAN/RULES/EXECUTION、公共 probe/runner、v09_model、集成与冻结评分；公共文件串行维护 | 项目与执行 PLAN、冻结与评分 |
| A：时间输入 | R09、clock_model、r09 专属分析与测试；使用 B 定义的源工作口径 | [有效频率](../../../model/mechanism/hardware_rules/FREQUENCY-MECHANISM.md) |
| B：供给请求 | R10/R13、R14 输入生命周期、supply_model；接入 C 提出的跨 rank 前驱 | [多 CTA TMA](../../../model/mechanism/hardware_rules/TMA-SHARED-SERVICE.md)、[S16](../../../model/mechanism/hardware_rules/TMA-S16-STAGE-COST.md)、[GEMM 候选与反例](../../../model/mechanism/hardware_rules/TMA-GEMM-HYPOTHESES.md)；`tma_input_events.py`、`analyze_tma_s16_stage_cost.py` |
| C：输出边界 | R15/R18/R19、output_model；m11 专属 probe/模型/runner 与 R03/EXP-09 | m11 已停止，见[重启条件](../../../model/mechanism/hardware_rules/M11-SMEM.md#4-结论与重启条件) |
| C1-01 | `results/.../20261013-C1-design`、`-C1-models`、`-C1-dev`、`-C1-baselines` | [C1-01](../Q16-config-selection/C1-01-selection.md) |

用户已授权主集成与上述专业组之间的有界协调。具体对话身份保留在已有授权与执行记录中。

<a id="research-chat"></a>

2026-10-16 对话分工：主集成在本对话维护 cfg_b 程序、组合、针对已知反例的有限 CPU 建模、模块接入与冻结评分；[GH200 可选机制与待批实验](thread://01a14190-8d4a-7e03-8c58-f1929dc6d19a)承接可选逆向、证据缺口评估和待批执行方案。A/B/C 继续表示文件责任分工，执行许可统一见 [GOALS §5.4](../../../GOALS.md#current-scope)。

<a id="status"></a>

## 当前状态

[full完成边界筛查](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-full-boundary-screen-001/README.md)已完成1320条原始记录核对，模型／拟合／GPU0。将逻辑issue与B1后戳分开，以FIRST参考评分；上界705条、下界69条违例。原参数继续停止，模型完成精度未改善；下一规则使用已核对端点，跨试次固定年龄解释仍缺实际地址与短窗口双时钟。

[cfg_b行距与full等待方式](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-stride-wait-policy-001/README.md)780627已完成24进程、22有效，1GPU分配15秒、0构建／模型／拟合。两输出均排除Q04的0.622比例迁移；第二输出的行距周期增量在轮询下保持，第一输出信息不足。两暖机失败及一trace/plain超限保留，无补跑；现有服务参数与原12成绩保持。

[请求返回同地址比较](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-address-matched-return-001/README.md)已完成55次CPU调用、108.582秒，拟合／GPU0：更正trace0预测地址与trace1评分进程错配；同输入时间线保持，实际地址匹配后unique Q1／Q4仍失败、global1 Q4通过。当前参数版本继续停止，后续使用更正后的约束。

[请求返回与查询对齐](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-query-return-connection-001/README.md)已完成18调用：基础两种B1参考年龄数值接近而full状态相反，A16查询已偏晚；单独后移query的修复停止。`full_query`已接入，既有时间保持；[g43分请求完成集合](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-operand-completion-001/README.md)完成18进程、17有效，只有一个trial／输出共同合格，信息不足收口；该观察形式不补跑。完整目标未完成，原12及独立验收保持。

[请求形成前置](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-formation-before-cache-001/README.md)已收口：432调用、35.33分钟、GPU0。原源四L最大误差9.338%，但plain最大误差14.428%、full状态及纯TMA Q1失败；私有源plain接近而阶段端点仍失败。当前参数版本停止；后续返回、full就绪与查询位置约束见上方新轮次，原12及独立C2/C3保持。

[逐CTA等值源副本](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-private-source-001/README.md)经780301修正host符号绑定、780343完成24/24有效进程；累计GPU分配25秒，首次24初始化失败保留。四个L的副本／原源周期比2.3842–4.6979，全部超过冻结1.20下界，支持大的源工作集响应；行距增长保持比仍信息不足。[同服务接回](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-private-source-transfer-001/README.md)已完成68调用，私有源plain误差−0.765%／−0.033%，阶段端点及原源行距仍失败；[片段共同服务](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-gather-joint-service-001/README.md)含恢复51调用，部分行距增长但full仍仅1/126未完成，固定版本停止。原12及独立C2/C3保持。

[读取受理后查询缓存](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-port-ordered-cache-001/README.md)完成34次调用，首次中断计入后总调用上界51；查询顺序改变，持续行距增长仍未产生，固定结构停止。[真实输入路径计数定位](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-input-counter-localization-001/README.md)的780060／780099均已结束，四plain合格，NCU分别因无效参数／kernel过滤错误未采到计数；信息不足，权限尚未检验，本轮执行修复收口。

[本地输入／计算共同服务](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-local-joint-service-001/README.md)已完成34调用、56项相关测试；零拟合的双向争用版产生部分行距增长，A16端点仍失败，停止固定结构。默认版、原12及独立验收保持。

[请求应答网络](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-request-reply-network-001/README.md)已作为显式profile接入，默认版本与冻结旧版逐值保持，26项CPU检查通过。一次有限联合拟合及850次引擎调用已完成；A16 plain误差由−27.273%变为+8.686%，第一L周期误差10.801%未过门，本参数版本停止，六验证条件未求解。原12及独立C2/C3状态保持，GPU0。

[地址路由的共享读取](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-address-routed-input-001/README.md)已接入显式profile；30项CPU检查通过，306次引擎调用完成。两开发plain误差−3.410%／−3.432%，第一L周期误差11.160%未过门，联合标定未收敛，六验证条件未求解。已保存纳秒／周期及频率换算误差分解；本参数点停止，GPU0，原12和独立验收保持。

当前主交付为输入供给服务整体重建，顺序见[项目PLAN](../../../PLAN.md#1-近期顺序)。[132CTA源／目的判别](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/row-stride/20261022-Q02-shared-swizzle-001/README.md)779181构建成功、779198十二进程全部有效，3321888768值零错；Q1/Q4的旧差异在全卡未稳定保留，两项预测均未分开，信息不足收口，不补测。下一交付直接做输入服务整体替代；旧固定价格的缓存／队列微调与单CTA swizzle补测收口。

- [四组展开间隔775199](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/row-stride/20261022-Q02-prefill-gap-unrolled-001/README.md)完成48有效臂，三个seed的377→0cycle全过冻结资格，支持本版间隔隐藏；最大扰动0.136426%。一构建、零修复、GPU5秒，三作业均结束；量化服务位置与原12反例仍未解决。

更新：2026-10-22。目标active，完整目标未完成。773099及cycle-only版773128的S6观察与桥接通过，但区间分别宽576/496cycle，两控制解释仍相容；两轮均未测S4，见[源组织](../Q02-tma-request/source-organization.md)。输入/输出同批检验774013完成12开发进程，暖机/稳定性未过，未冻结或采留出，见[联合窗口](../Q07-smem/input-output-joint.md)。原12与独立C2/C3状态保持。132源g11观察774105九进程数值／暖机通过，窗口与consumer戳扰动通过，结束跨度扰动及向R桥接未过，信息不足，见[本轮收口](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261021-Q02-late-full-span-001/README.md)。等待路径身份四次CPU筛查预测相同，未补测。程序准备接线已完成八默认报告逐字回归及47项测试，价格保持；两控制解释对名义640cycle干预的跨stage响应带可分，见[程序准备](../Q02-tma-request/native-d-chain.md)。准备延迟轮774214在两有效进程后因checkpoint保存退出，修复后774221继续七个，合计九进程资格／桥接通过；响应6618–6699cycle超冻结校准域，未测S4，见[结果](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/native-d-chain/20261021-Q02-preparation-delay-001/README.md)。随后256项CPU约束保留逐轮准备类，774347独立K4／K32十八进程资格／桥接通过，长度响应落在冻结门槛内、首段形式被排除；N0−R随长度增长仍未解，见[长度结果](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/native-d-chain/20261021-Q02-preparation-length-001/README.md)。G3／A16两档准备价格的CPU迁移完成，替换旧软件gap、固定源／计算／输出／频率后，行距响应仍约0.0931%（实测28.9371%），两L增长均0，见[迁移结果](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261021-Q14-producer-preparation-transfer-001/README.md)。两参数版本停止。[计算间隔与行距对照](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261021-Q14-stride-compute-discrimination-001/README.md)774436完成单遍十二进程且资格通过，但两段周期增量105–133／587–687.5低于1000cycle下限，三遍未测、两类假设未判定；[单遍路径修复](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261021-Q14-direct-repeat-path-001/README.md)已完成774450零GPU构建；两binary均REG168、无spill，三TMA与原commit/wait保留。导出因home及项目配额失败，节点归档已直接取回，未重编译；774459完成24进程，原R两段行距增量重现，新N1基础慢约22%且响应塌缩，桥接未过，N3未采。源对照发现PipelineState仍使用运行时活动槽，下一项[恢复原固定S6推进](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261021-Q14-static-stage-repeat-001/README.md)，保留额外两遍开关，先核同批桥接。774523构建及774550测量均完成，24进程合格但R桥接仍未过、N3未采；恢复固定S6未恢复反例。138文件SHA／回放通过，完整构建也已取回。SASS增加ARRIVE与特殊HGMMA。[循环外静态选择](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-static-repeat-dispatch-001/README.md)774570资源解析失败、修复1的774581构建成功；774591完成36合格进程，R／N1桥接通过，两段A16增量比落入冻结可隐藏带，固定不可隐藏加项被排除。198文件SHA、二进制哈希及本地回放通过；源服务／等待返回／反压位置未区分。下一项CPU同时约束单遍与三遍响应，不扩展干预构建；原12成绩及独立C2/C3状态保持。

- 2026-10-22 [发布／查询CPU核对](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-publication-time-alignment-001/README.md)：18调用逐位复现旧预测；全部CTA的47/111发布进度与合格观测吻合，A16 full仍提前639–881cycle。原12成绩保持。后续775058构建与775061测量已完成：24有效进程、五合格输出530次未完成年龄下界均超旧完成上界；第六项桥接失败保留，见[结果](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-published-request-age-001/README.md)。
- 发布后年龄的空间诊断完成1452项坐标核对，A16的A组跨进程误差改善8%/18%，与逻辑CTA组仍绑定。Raster转置对照的两预测带已分开，见[CPU预测](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-raster-contrast-prediction-001/README.md)。775148构建保持GPU指令；775150完成12有效进程，时间检查全过、最大扰动2.1304%，原raster六项控制未重复固定空间形状，along_m按计划未测；三作业均结束，见[有限执行](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-raster-age-discrimination-001/README.md)。原12成绩及独立C2/C3状态保持。
- [旧A行距证据复用](../Q02-tma-request/row-stride.md)取消相同单CTA补测；连续旧PS四调用及入队容量3/6八调用均未过Q4预填。复用752276的[间隔执行](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/row-stride/20261022-Q02-prefill-gap-execution-001/README.md)775192有48有效臂、实际扰动0.284%，但gap0响应59低于冻结69.45cycle；信息不足，循环版收口，下一项恢复已有四组展开组织。零构建，GPU4秒，相关作业均结束，组合成绩保持。
- 2026-10-18 至 10-20 白天共 179 轮，其中约 70 轮有 GPU 作业，单次节点分配多为 4–30 秒；10-20 晚另有 15 轮，其中 2 轮有 GPU 作业。各轮已按合并实验归档，结论见各实验记录。
- 原12条件默认仍为 10-16 前缀版：plain 中位 10.225%、最大 25.422%，独立 C2/C3 未通过；显式局部联合规则为开发选项，见下表 cfg_b 组合行。
- [A／双B操作的探针推进规则](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/probe-control/20261020-Q02-operand-control-model-001/README.md)保持停止：13 项控制检查基本是校准输入的自洽检查，原生 A2 窗口仍偏 12.492%。
- [66／132源batch对照](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261020-Q02-source-count-build-002/README.md)（763587）的冻结源状态响应获限定支持，零响应被反证；绝对窗口与 CTA 结束跨度仍未通过，随后的 CPU 固定接线均未修复，见[源组织](../Q02-tma-request/source-organization.md)。
- 2026-10-20 结果目录按“问题／合并实验”重排，旧平铺路径对照见 [MOVED-PATHS-20261020.tsv](../../../../../../results/gh200_resource_campaign/access_rules/MOVED-PATHS-20261020.tsv)。

[同行CTA定位](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-active-peer-001/README.md)：G3/S6原grid及CTA0两输出保持，新132/1两臂固定16次全132前缀。775564构建、775581测量、775596回收均完成；36/36有效、零失败、三trial全SM124，桥接最大4.020351%、观察扰动最大4.049213%。新132两L增量12536–15401／11740–11943cycle，单CTA为8–117／1206–1362，两比值区间均入冻结同行依赖带。支持当前目标调用的同行工作影响大部分A16增长，源／前端／输出位置未分。节点exclusive保留4GPU，测量step用1GPU、分配16秒，原始回放逐值一致；开发定位，模型／拟合0，原12与独立C2/C3状态保持。共享投递固定上界FIFO轮已在22调用必要门失败后停止，未扩三遍。

[同行输出TMA定位](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-peer-output-001/README.md)：775646未生效gate由SASS发现；定向修复1的775647各8处gate生效，64条STS.64及commit／wait保持，REG168、无溢出。775662为COMPLETED，分配17秒，节点exclusive保留4GPU、测量step用1GPU；35/36有效，Nstore1基础plain的trial1原暖机CV超2%，保留不补。全写桥接三trial通过、最大3.791480%，单写有效trial0／2的两L响应比[0.9601194284,1.052762746]／[0.8954191188,1.118705036]均入冻结保留带。同行全局输出TMA不是当前大部分增长必要条件，132CTA输入／计算仍能维持；物理服务未定价，模型／拟合0，原12及独立C2/C3状态保持，原始回放逐值一致。

[等值A源折叠定位](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-equal-value-source-002/README.md)：源等值及既有第二L重叠核对已完成；775812零GPU构建通过，实际SASS折叠开关已核，REG168、无溢出、无修复。[001](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-equal-value-source-001/README.md)的775816未启动即取消、测量0进程；原零GPU回收775818因接收器退出而取消。002复用原构建，改用ROMEO分配的132SM GH200，UUID执行前绑定，三臂同批同卡；原区间、5%桥接与其余门槛保持，上限一次5分钟／36进程，776745在romeo-a021完成14秒；24/24进程数值、暖机及观察合格，参考两L增量15027–16051／11557–12202cycle。桥接仅trial0通过，trial1的A16 L0 cycle偏差+5.893461%，trial2的base L0 ns为−5.043228%、A16 L0 cycle为+5.043219%，未满足至少两项共同trial，Nfold未采，信息不足收口。模型／拟合0，原12及独立C2/C3状态保持。 [003邻接交换顺序](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-equal-value-source-003/README.md)按原最多48进程／5分钟GPU、0构建执行，24进程桥接后收口。 003的776879同卡24/24有效但0组过桥，折叠未采；[同binary负对照](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-bridge-null-001/README.md)777026在romeo-a056完成21秒，47/48进程有效；N1 A16 plain trial0暖机CV超2%保留，未补跑。R同二进制重复3/3、N完整配对2/2通过原门；旧、新二进制等价桥接仍未通过，同二进制对照可用。

[004同二进制源折叠](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-equal-value-source-004/README.md)：777113在romeo-a033完成13秒，24/24有效、零失败，最大观察扰动3.849898%。第一输出响应比[0.912199733,1.016899078]属于增长保持；第二输出[1.324059076,1.484525922]超出两个冻结类别，整体信息不足。源地址重用增加后第二输出行距增长反增，物理归因未定；模型／拟合0，跨二进制桥接失败与独立C2/C3状态保持，回放逐值一致。

[固定逐副本服务必要下界](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-copy-work-bound-001/README.md)：最忙GPC4608份16KiB，固定τ对应56.713846µs服务；按原F／κ映射的plain下界不计P0为59.470203µs、计P0为61.091820µs，均超原54.313600µs上限。同价格的逐副本单服务器即使工作守恒也停止；引擎／GPU／拟合0，未实现调度器或扫描价格，模型成绩保持。

各工作流状态（10-16 起未变的行保持原文）：

[在途源行订阅CPU筛查](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-line-subscription-screen-001/README.md)：8合成例、44引擎调用完成，两粒度源工作量有变化而持续行距增长仍0；单遍必要门失败后停止，未扩三遍或GPU，默认组合保持。

[共同SMEM服务必要下界](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-joint-smem-bound-001/README.md)：读取D三个长度、A及R12旧窗口，排除全部逻辑引用共同128B/cycle形式，保留B复用共同服务与独立路径。D已覆盖真实请求配比，下一项回到跨CTA源共享／次序的持续状态；模型引擎／构建／拟合／GPU0，组合与独立验收状态保持。

[full进入状态位图](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-full-entry-mask-001/README.md)：实现1的774800完成24个时间合格进程，但观察补丁被overlay覆盖，机制采样0、信息不足；[修复2](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-full-entry-mask-002/plan.json)774821构建成功，目标SASS的开关、查询、累积与导出已核过；774830完成24有效进程，桥接／扰动通过；A16后续full未完成率91.823%对基础4.099%，反证当前全ready预测，见[原始证据与回放](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-full-entry-mask-002/README.md)。下一项约束持续供给及full可见连接，价格／原12成绩保持。

[g43发布位置观察](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-producer-publication-001/README.md)：774863的18进程数值／暖机、三组时间对照和行距保持通过；N0基础trial2第二输出未完成17/132，超冻结10%，N1未采，两位置仍未区分。停止稀疏g43形式，保留有效条件；F同批保持原反例。CPU模型／拟合0，原12成绩不变。

[发布位置定位](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-producer-publication-003/README.md)：全局原子775011改变g43状态，停止；本地计数775039先遇对齐故障，775047修复后775049有23/24有效。三合格输出306次未完成均已发布并领先三组，发布前全域主导类被反证；六条件完整主张未验。下一项CPU核发布／查询／完成连接，不补齐同批GPU；原12成绩及独立C2/C3保持。

| 工作 | 当前结论 | 下一步 |
|---|---|---|
| S16 探针控制成本 | 历史离线诊断完成；token 控制（H1）与组状态阻塞（H2）对现有窗口预测相同 | 暂停；保留历史诊断，重启按当前阶段限制 |
| 多 CTA TMA 共享服务 | 每 GPC 两槽服务冻结留出通过（3.64%）；工作集、先前写入、暖机次序与两存储域获支持；共享／独立源窗口 5.09／17.89 µs，CTA 结束分布未解释 | 见[GPC 共享通路](../Q02-tma-request/multi-cta-gpc-sharing.md)、[源状态](../Q03-source-service/source-state-candidates.md)、[源组织](../Q02-tma-request/source-organization.md)、[共享源固定形式](../Q02-tma-request/shared-source-forms.md) |
| cfg_b 组合 | 默认版保持原12条件10.225%／25.422%；显式局部联合规则已完成同12条件8.706%／25.334%，stage响应6.738%对实测10.467%；A16响应0.0934%、B80/B112为0，关键CTA1/120，独立验收未通过 | 见[cfg_b 组合](../Q14-composition/cfg-b-composition.md)、[局部规则接回 GEMM](../Q14-composition/gemm-connection.md) |
| 有效频率 | 746447 完成 8 条件、80 个采样槽位；746491 的 8 条运行前频率预测全部通过 5% 标准，最大误差 4.52% | 746523 首轮 C3 仅 4/8 通过；饱和响应式在新 746534 留出 8/8 通过，最大时间误差 2.96%；交付冻结版；746814阶段频率10项通过8项，S4三输出前两段未过；活动比例联立候选修正了频率方向，但三输出首段L仍低估9.09%，先修正供给服务；747202空闲干预支持d044首阶段的前序状态效应，d114重复性不足；固定驱动及CTA阶段活动驱动方程均未覆盖阶段轨迹，独立频率与前序状态研究暂停，组合沿用已交付预测器；[频率记录](../../../model/mechanism/hardware_rules/FREQUENCY-MECHANISM.md#4-结论) |
| C1-01 | 暂停剩余测量、模型搜索与独立留出续跑；已有合格结果作为组合对照，恢复由用户决定。构建 743011、冒烟 743016、遥测对照 743051 完成；开发作业 743973–743976 按用户要求在运行前取消 | 746805/746820共318槽成功、2槽暖机耗尽；长条件graph满足原降噪判据，短条件保留缺失；746848/746850八候选对照334成功、2耗尽；固定graph暖机/目标、plain7/dual2与参照漂移规则，30块输入已准备；746873完成540槽后遇数值超限，746893确认八候选三个K均超原容差；按用户选择采用逐K16误差界，746908的144项原生与本地回放通过；正式开发v2首块746920的1026槽全部成功、96条主体记录独立回放通过；第二块746937独立回放通过（1025成功、1暖机耗尽），前两块共24条件、192条记录可用；第三块746945独立回放通过，前三块共36条件、288条记录可用；第四块746956回放通过11条件，d226因cfg_a暖机奇偶交替而不齐备；前四块47条件、376条记录可用，第五块746980回放通过，前五块59条件、472条记录可用；第六块746988回放通过，前六块71条件、568条记录可用；第七块746999独立回放通过，前七块83条件、664条记录可用；第八块747016独立回放通过，前八块95条件、760条记录可用；第九块747026独立回放通过，前九块107条件、856条记录可用；第十块747037独立回放通过11条件，d037因cfg_a暖机波动不齐备；前十块118条件、944条记录可用；第十一块747052独立回放通过，前十一块130条件、1040条记录可用；第十二至十七块747064、747074、747086、747099、747117、747124独立回放通过，前十七块202条件、1616条记录可用；128容量trace经747102验证通过，b15的新包已完成正式回放；b17因节点空间耗尽失败，保留失败包，747173按原输入整块重跑并通过，747180/747192/747200的b18/b19/b20也独立回放通过，b21/b22（747214/747379）也通过独立回放，b23（747440）也完成独立回放，b24（747469）回放通过11条件，d058因cfg_c暖机奇偶交替不齐备；b25（747498）回放通过12条件，前二十六块309条件、2472条记录可用；[恢复条件](../Q16-config-selection/C1-01-selection.md#恢复前要定的事) |
| m11 | 741153 完成 48 条件×3 进程；旧需求与新的最忙 bank 标量均有反例，首个新静态汇总形式得到离线反例。随后 v1（741915）、v2（742266）两次冻结留出均未通过（v2 中位 0.44%、最大 11.60%） | 不新增单 warp 任务；剩余留出预算不转为开发；[重启条件](../../../model/mechanism/hardware_rules/M11-SMEM.md#4-结论与重启条件)、[m11推导](../Q07-smem/R03-access-demand.md#m11-bank-vector) |
| 长 K 主线 | 校准 741159 在旧参照门槛处停止；120 个参照采样完成，540 个开发槽位未执行；新提交 1/1，无恢复。失败修正未接入公共模型。cfg_b G6 的频率迁移转入有效频率任务 | 暂停；相关残差进入组合未解决报告，恢复按当前阶段限制；[主线](../../../model/semi_empirical/MAINLINE-LONG-K.md) |
| 供给与其他机制 | 供给部分转为上面两项 TMA 任务；寄存器、联合服务、驻留等其他机制暂停 | 不自动恢复；达到[项目 PLAN](../../../PLAN.md#3-后续任务的进入条件)的进入条件再立项 |
| V10 与跨卡 | 未冻结、未采样，联合验证未完成 | 当前暂缓；[验证顺序](RESEARCH-GUIDE.md#order) |

旧 worktree、分支、准备包和结果目录暂时保留，本次没有删除。原清理审计留在[结果记录](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261012-repository-consolidation/)，需要清理时按当时 Git 与文件状态决定，不维护过期清单。

<a id="counter-access"></a>

计数器：[已有诊断](../../../../../../microbench/gh200_l0/results/20260930-counter-access/README.md)记录 NCU 权限受限；2026-10-20 零 GPU 作业 764001 复查同节点，`RmProfilingAdminOnly=1`，仍限管理员（[查询记录](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261020-Q02-source-base-replay-001/ncu-access.txt)），待用户联系 ROMEO 管理员。两项测量的作业与身份见[交付记录](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261012-authorized-measurement-integration/authorization.json)。

2026-10-22 [程序干预接入](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-program-intervention-001/README.md)完成：原前缀与正式调用分开，最后只执行CTA0的16条输出TMA，输入实际地址进入程序和共享身份，STS/控制保持。四条件固定价格回放仍预测两L行距增量0；接口修复已完成，下一项为供给服务整组替代，原12和独立C2/C3状态保持。

2026-10-22 整组供给替代的首批三种源片段／目的行流水参数版本均未过局部必要门；1864次局部调用、三次有限拟合，另8次保存参数评分回放，GPU0。后两版八完整窗过门但配对增长失败，未接入GEMM，见[行距轮次](../Q02-tma-request/row-stride.md)。已有完整组合误差和独立验收状态保持。

2026-10-22 [同覆盖A16／A80](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/row-stride/20261022-Q02-row-phase-invariance-001/README.md)的778301完成六进程、4718592值零错；双单位增长比0.889266/0.887850，采前不变性被反证。零构建／拟合；两个执行前故障零探针进程，累计分配124秒、三作业已终止。下一供给服务保留实际源片段和状态；原12成绩及独立C2/C3保持。

2026-10-22 实际字节流水374次局部调用、一次拟合，11窗全过但六项配对等失败，固定版本停止。778441的A80同集合次序扩展六进程有效、18874368值零错，三个窗比保持，A80/A16仍约0.889；A0/A16重复752416，后续复用而不追加排列扫描。见[行距轮次](../Q02-tma-request/row-stride.md)。

2026-10-22 移除整atom core的全局／两域真实源行服务各完成G3/base+A16的34调用，CPU120.883／142.951秒、GPU/拟合0；两L增长均未过原>1000cycle必要门，两固定版本停止。两域版源获取工作仅增0.5214%，下一供给结构区分源获取与逐请求地址处理／重组。见[组合回放](../Q14-composition/gemm-connection.md)。

[有限同址sector合并](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-bounded-sector-merge-001/README.md)34调用完成，两L增量0，固定版本停止；[显式分区与组相联源缓存](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261022-Q14-banked-source-cache-001/README.md)也完成三检查、34调用、CPU808.098秒，两L增长0，固定版本停止。两轮共68调用、GPU/拟合0，进程均结束；464个后续组仍提前就绪，原子服务到返回投递的共享占用仍未约束。原12及独立验收状态保持。

[独立TMA工作量约束](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-independent-service-bounds-001/README.md)完成40个既有进程分析：原读取价在global1-Q4的5进程均超过声明频率下的完整窗口上界，应答价未超界。[在途同址读取共享版](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-read-sharing-network-001/README.md)完成同服务纯tensor／GEMM的56调用，G3增长0，固定参数点停止；四干预未求解。下一项先约束纯tensor共享／独立源响应，模型默认与原12／独立C2/C3保持，GPU0。

[纯TMA一价标定](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-source-only-calibration-001/README.md)、[公开折叠转写修复](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-public-fold-correction-001/README.md)、[有限读取订阅](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261023-Q14-bounded-read-fanout-001/README.md)分别完成155／56／56次引擎调用。G3两L均未形成持续增长，三个参数版本停止；后续折叠／三遍条件未求解。51项CPU检查通过，一次执行器变量名修复且复用已保存unique；GPU0，默认版、原12及独立验收保持。下一项为本地输入／计算联合结构，预算在执行前另登。
