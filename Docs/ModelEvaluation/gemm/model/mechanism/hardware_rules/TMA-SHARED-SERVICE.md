# 多 CTA 的 TMA 共享服务

更新：2026-10-17。阶段4离线接口与证据回放见下节；真实计算／TMA短链迁移待测。既有研究状态：tensor 探针已通过原生正确性；首批与源分区候选留出均未通过；工作集状态的冻结干预获得支持，量化服务规则开发中；GPC 局部服务已获得新条件支持，cfg_b 匹配源迁移仍失败，近期源状态与两域工作映射已获独立干预支持，随事件更新的缓存服务尚未完成。验收与通用证据原则见[GOALS](../../../GOALS.md)。

## 1. 现象

研究对象是GEMM中多CTA供给受限、stage响应和CTA推进不一致。已有V09的stage差异、R05-D的共享/独立源对照、R13的多CTA与局部端点记录可作为开发证据，索引见[TMA历史盘点](TMA-HISTORY-REUSE.md)。S16单CTA结果用于检查探针控制成本，不承担跨SM共享范围的结论。

### 存储资源与搬运边界

原总PLAN的搬运边界在本任务维护，首轮限定如下：

| 对象 | 本轮具体条件 | 记录或对照 |
|---|---|---|
| 源分配 | 目标GH200本地 `cudaMalloc`，由GPU HBM支撑，指令源为 `.global` | GPU UUID、分配大小、每CTA地址集合与整卡地址并集 |
| 缓存准备 | 分别使用预热复用与弱复用协议 | 工作集、复访距离、预热序列、可用L2/HBM观测；无流量证据时按准备协议命名结果 |
| 目标搬运 | 首批直接使用 cfg_b 的 tensor global→本地 shared 单目的请求 | 每输入组为 A 一次 128×64（16 KiB）、B 沿 N 分两次 64×64（各 8 KiB），共 32 KiB；组内共同登记 full barrier 的 32768 B，TMA SMEM swizzle 为 128B。按 cfg_b 实际描述符和三请求顺序实现，见[程序请求](../program/cutlass_program_graph.md#2-对象操作与依赖)；描述符其余编码字段以构建查询为准，不混用参数 |
| 目的存储 | 发起CTA驻留SM上的本地SMEM，固定预留四个 32 KiB 输入组槽及 barrier 区域 | 活动组数改变时物理分配和 barrier 区域固定；资源配置保持每 SM 最多一个 CTA，并核对寄存器、驻留上限和实际 SMID/重叠 |
| 控制与消费 | 完成等待、数据可读、固定消费者及槽释放 | 使用核对过的控制实现，分开记录等待与释放端点 |
| 排除的路径 | SMEM→global输出、DSM、多播、Grace CPU LPDDR、host/managed、peer GPU | 不进入本轮参数与条件矩阵 |

资源依据见[GH200存储与搬运](../../../hardware/gh200_sm90/L2.md)和[片上资源](../../../hardware/gh200_sm90/L1.md)。共享小源与独立大源同时改变地址并集和CTA间复用，不能只凭两者差值归因为HBM服务；在每种源协议内比较并发响应，并保留源集合变化。

复用 [EXP-15 探针](../../../../../../microbench/gh200_resource_campaign/probes/tma_tensor_2d.cu)的分配、正确性检查、描述符导出和计时框架。其现有实现为 rank-2 描述符与 `tensor.2d`；cfg_b 冻结 SASS 为三条 `UTMALDG.3D`，加载部分须按实际实例实现。EXP-14/16 与 S16 的 bulk 数据只作历史对照。

<a id="stage-4-service"></a>

### 阶段4服务接口与资格

本阶段以真实cfg_b的S6／S4程序图为前端；上表四物理槽属于旧tensor探针。有限实现与回放登记见[PLAN](../../../PLAN.md#stage-4-transfer)，本轮证据见[TRANSFER-01](../../../experiments/gh200_sm90/Q02-tma-request/TRANSFER-01-replay.md)。

实际三请求接消费者的[D短链](../../../experiments/gh200_sm90/Q14-composition/COMPUTE-01-resident-chain.md#6-d原生tma短链准备)已完成CPU与本地静态准备：六物理槽、S4／S6，全部TMA在计时窗口内；新服务价格与迁移仍待实机。

[cfg_b_request_model.simulate](../../../../../../microbench/gh200_resource_campaign/access_rules/cfg_b_request_model.py)继续使用已有事件引擎，两种输入接法分别保留：

| 接法 | 输入与事件 | 当前资格 |
|---|---|---|
| `tensor_source` | `CfgBTensorSource`从共同程序取得实际A/B源box，调用已有`TensorSourceService.submit`；返回源开始、服务结束和就绪 | 既有工作假设；价格与失败范围仍按原候选记录 |
| `input_source_async`＋`input_source_bind_events` | 源接收`issue(t,cta,group,atom,weight,demand_id,notify_ready)`，可排队；通过回调给出`source_id/source_key/merged/source_busy_ns/service_start/service_end`及源就绪时间，定时事件接入同一队列 | 接口已核对；本轮两种信用归还仅用现成合成服务检查，调用方按索引与`request_inputs`映射实际源身份 |
| `input_destination_service` | 在目的准入后返回`start_ns/service_end_ns/ready_ns`；未指定时保持原每CTA目的FIFO | 显式工作假设；实际独立目的服务价格待辨识 |

各服务对象作为独立参数传入，价格JSON保持可序列化。`demand_id`对应一个逻辑目的请求，`source_id`对应一次源服务代次；多个目的可订阅同源，最后目的就绪按该源所有delivery取最大值。源信用的释放时刻由具体服务记录，软件槽释放由[同步服务](SYNC-SERVICE.md)和计算依赖推进。异步`issue`先接收软件请求，服务内缺信用时排队；本接口的producer仍可继续发其他请求，未引入UTMALDG发出阻塞的新规则。

独立路径前端在逻辑请求发出时预留，三类源统一执行：

```text
destination_arrival = max(source_ready + destination_latency,
                          input_path_end + input_min_latency)
    → destination service → request destination_ready
A、B0、B1 均完成 → full 条件满足
    → 条件wait返回 → 计算／wait1或wait0 → 输入槽释放
```

`input_sources`输出各源代次的请求到达、服务开始／结束、源就绪与`last_destination_ready_ns`；逐请求记录保留`demand_id`、地址和目的窗口，条件等待另由同步接口输出。共同入口的P0前缀只做一次时间原点换算。源完成、目的就绪和消费者释放分别定价；L2阶段当前只交付固定逻辑字节的整窗`read_window`，尚无可直接用于16／8 KiB请求的价格。

完成语义按[PTX ISA 8.8 §9.7.9.25](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html)：TMA global→shared在整个copy完成后贡献mbarrier事务字节，成功等待对应phase后读取结果；普通cp.async的commit／wait_group按发起线程分组。旧cp.async复制／消费窗口保留其CTA发布与完成边界，参数独立维护。输出bulk的`.read`等待与完整操作等待分别对应源读取和整个操作完成，原输出服务保持。

本轮修复了外接源跳过三项独立路径约束的具体连接：同一零价源加100 ns最小延迟，目的就绪现在从0变为100 ns。原零价路径的六个小图和两种异步轨迹保持原值，非零约束及通知乱序回归通过。本轮不拟合、不新增硬件价格或容量，不修改原冻结成绩；完整组合C2/C3仍未通过。

## 2. 假设

| 候选 | 状态与响应预测 | 区分所需证据 |
|---|---|---|
| 每CTA局部准入限制占主导 | 增加每 CTA 未完成输入组数先改善局部等待再平台化；低全局负载时改变其他CTA数量影响较小 | 低 CTA 数下的输入组并发扫描，以及加入背景 CTA 的响应 |
| 跨SM共享供给占主导 | 增加实际活动SM/CTA后整卡吞吐趋于饱和、单CTA等待增加；源复用协议改变饱和位置或程度 | 多CTA曲线、SMID及活动重叠、各CTA推进分布、cycle/ns双口径 |

[冻结 setup](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011/static_setup.json)中，cfg_b S6/S4 的 SMEM 分别为 215040/149504 B（210/146 KiB），`max_active_ctas_per_sm` 均为 1。同 SM 多 CTA 竞争留待多 CTA 驻留配置研究，本轮保留单 CTA 内局部限制与跨 SM 服务。

[Hopper 微基准 §IV-A](https://arxiv.org/html/2501.12084v1#S4.SS1)报告 H800 在半容量和全容量附近的 L2 转折；[NVIDIA 的存储局部域说明](https://developer.nvidia.com/blog/accelerating-data-processing-with-nvidia-multi-instance-gpu-and-numa-node-localization/)解释了数据中心 GPU 通过一致缓存保留本地域副本。结合本机 30–36 MiB 的转折，保留“分区驻留与相位相关服务”作为下一候选；分区到 SM 的映射及 TMA 请求在其中的状态转换仍待本机辨识。

这些候选可以共存，首轮目标是确定需要哪些共享状态，而非为整条路径选一个唯一瓶颈。容量回收先比较ready相关与消费释放相关规则，只在两者预测可分时增加消费者延迟对照。

## 3. 实验验证

主轴为 **实际并发 CTA 数 × 每 CTA 软件允许未完成的输入组数**。设计 CTA 数取 1、4、16、64、132，输入组数取 1、2、4；每组固定为上述 A/B0/B1 三请求，分别允许最多 3/6/12 个逻辑请求未完成。实际在途数量由执行过程产生。

1. 先将R05-D、R13和已有多CTA记录映射到这两个轴，保留设备、源协议、指令和资源条件，能回答的问题不重测。
2. 新探针每 CTA 固定输入组总数、控制实现及物理槽，保持上述单 CTA 驻留条件。记录 SMID 和活动窗口；启动数或 `--sm-count` 不能直接填入“实际并发”列。
3. 先做稀疏矩阵；消费者释放干预、背景相位、拐点加点与独立留出按候选分歧选择。每组明确固定每 CTA 工作量还是固定总工作量，并分别计账。
4. 每个源协议内记录总地址并集、每CTA复访和跨CTA共享关系；避免让并发数与缓存服务情形无记录地同时变化。频率线提供诊断口径，cycle与ns同时保留。
5. 输出整卡逻辑吞吐、每CTA周期/等待、推进离散度及支持域。将候选接入目标GEMM图，检查其是否能解释供给、stage及CTA间差异；没有改变这些判断的微小残差不自动扩展研究。

### 近期判别顺序

当前问题是：**局部请求/释放状态与跨 CTA 服务，怎样共同产生局部等待变化和全卡完成跨度？** 候选须满足的历史约束与端点在[联合约束入口](TMA-GEMM-HYPOTHESES.md#下一候选的联合约束入口)维护。下面从既定矩阵选择先做的部分，首轮条件安排见下一表。

<a id="bounded-supply-round"></a>

**2026-10-15 本轮安排。** 用户允许新增一组有限候选，固定为[按请求大小分道的两个版本](TMA-GEMM-HYPOTHESES.md#bounded-supply-candidates)。两版均在首项`prior-q1-skew8`违反原ns/cycle门槛；[本轮结束](../../../../../../results/gh200_resource_campaign/access_rules/20261015-cfg-b-size-lanes-service-001/RESULTS.md)，完成2次CPU回放，其余53项停止。新增GPU作业为0，原阻塞项判别仍为1/2轮。下表保留固定清单与停止规则，一版本一条件只回放一次。

| 次序 | 固定条件与上限 | 决定 |
|---|---|---|
| 1，保住局部约束 | [原18条件清单](../../../../../../results/gh200_resource_campaign/access_rules/20261015-cfg-b-dual-granule-service-001/cases.json)中的旧六项；先`prior-q1-skew8`、`prior-q2-skew32`，再`prior-q2-skew0`、`prior-q2-skew16`、`prior-q1-skew24`、`prior-q1-skew48`；最多2×6=12次 | 任一项不满足[原ns/cycle门槛](../../../../../../results/gh200_resource_campaign/access_rules/20261015-tma-gpc-heldout-002/plan.json)，即停止该版本。先用对齐、无复用且信用不约束的合成请求核对两通道、原每CTA目的FIFO和信用事件；按大小分道与原最早空闲仲裁分别记录 |
| 2，已有共享与重排反例 | 幸存版本先跑`permutation-grouped-q1`/`permutation-mixed-q1`，再补清单其余10项；累计最多2×18=36次 | 先比较GPC4主响应与原容差，失败即停止该版本。746797逐GPC的early/late/release与跨度全部报告；747595保留实际Q4响应及原“信息不足”结论。新候选的开发相容性与原冻结成绩分列 |
| 3，接回关键GEMM | 每个幸存版本跑`G3_base`、`G3_stage4`、`G3_A16`、`G5_base`、`G5_B32`、`G5_B80`、`G5_B112`，最多2×7=14次 | 同时报告主循环、stage/行距响应、空间跨度、六分项和plain；输出与[当前同版报告](../../../../../../results/gh200_resource_campaign/access_rules/20261015-cfg-b-full-matrix-001/comparison-final.md)逐项对照。旧六项的门槛只用于旧六项；组合按[GOALS验收](../../../GOALS.md#34-组合)另行冻结 |
| 4，完整矩阵收口 | 仅选一个工作版本补`G1_base`、`G2_base`、`G4_base`、`G6_base`、`G5_random_seed29`，最多5次；G6最后 | 按stage/A16/B80/B112四项响应方向的符合数降序，再按七项MAIN跨度最大绝对相对误差、plain最大与中位绝对相对误差升序选择；两版相同时暂用`size_ready`、归档另一版及未区分项。整组12条件的失败全部保留；没有可保留版本就交付未解决报告 |

本轮累计最多55次CPU条件回放，GEMM每项沿原10次暖机加1次目标；失败的版本停止后续，不补价格、容量或第三版本。结束时交付两个版本的筛查决定、幸存工作版本的整组报告，或具体反例与缺少的辨识条件。GPU第二轮只有在新候选给出可区分的冻结预测后才进入设计，通用限制见[GOALS当前范围](../../../GOALS.md#current-scope)。下表保留首轮已执行的设计顺序。

| 顺序 | 做什么 | 哪个结果决定下一步 |
|---|---|---|
| 1，CPU 建模与预测冻结 | 用已有局部窗口、累计窗口、`mma` 返回跨度与 `MAIN_END` 排序筛选候选。将 tensor 探针的发出、共同完成、消费与释放程序接入请求级框架，或扩展 `tma_input_events` 支持三请求组与多 CTA 共享服务；探针与 GEMM 使用各自程序前端，共用服务规则 | 所有保留候选对第 2 步拟议条件给出可观测端点的响应方向、幅度与参数范围下的预测；先检查区分能力，再冻结候选、参数、条件、预测及判定标准，执行要求按 [C2](../../../GOALS.md#35-c2-机制主张) |
| 2，首批 tensor 实机条件 | 原矩阵为 CTA 数 1/16/132 × 未完成输入组数 1/4 × 两种源协议。正式清单删除与共享源物理条件相同的两个单 CTA 独立源行，保留 10 行；另加 132 CTA、Q1、G128、128 槽共享源环的相位 0/1 两行，仍共 12 条件。相位对照的每 CTA 地址集合与整卡 4 MiB 地址并集相同；所有条件固定四组物理槽并记录实际 SMID、重叠与 ready/释放端点 | 同一源协议内，低 CTA 数的组并发响应约束局部推进；固定局部组织增加重叠 CTA，约束跨 CTA 服务。若整批对待区分候选的预测相同，或差异均小于测量分辨率与不确定性，则在冻结前替换条件，仍以 12 条件为首批上限；需要定位转折时再从原矩阵补 4/64 CTA 或组并发 2 |
| 3，按分歧选干预并保留独立留出 | 额外条件优先用于消费者释放延迟与独立留出。释放相关候选可分时，固定首批请求并延迟消费者释放，分别观察首批 ready 与后续 acquire/补发；背景状态候选可分时，固定目标 CTA 并改变背景请求相位 | 释放干预若只改变后续 acquire/补发，先由软件槽依赖解释；首批 ready 也变化时，保留服务耦合解释。背景相位改变目标 ready 时约束跨 CTA 时间状态。测后修改规则或参数的版本另经独立留出检验 |
| 4，接回完整 cfg_b | 将 tensor 服务接回包含计算、输出和槽释放的 GEMM 程序，回放 S4/S6 的局部与空间约束 | 检查探针到 GEMM 的程序耦合；开发约束通过后冻结，进入一次独立干预检验，失败按首个分歧端点收口 |

第 2 步沿用本页的 GPU global→本地 SMEM 路径；共享小源和独立源分别保留工作集、地址并集与复访条件。R05-D、R13 的请求组织与消费者随原记录保留，作为历史约束。首轮第 3、4 步的额外区分与留出从既定 8 条件中分配；后续目标模式轮次分别登记条件和用途。探针程序与 12 条件的候选预测已交付，当前冻结记录见下文。

CPU 草案入口为 [tma_tensor_multicta.py](../../../../../../microbench/gh200_resource_campaign/access_rules/tma_tensor_multicta.py)。当前定义每 CTA 64 个输入组、32 个源环槽、固定四个目的组槽；先填活动窗口，再等待/消费/释放最老组并补发，最后排空。共享源协议让所有 CTA 访问同一组 A/B 源，独立源协议按 CTA 分区；每 CTA 访问量相同，整卡地址并集分别记录。GPU 使用 cooperative grid 同步、共同 globaltimer 起点和 CTA 同步；正确性作业中，132 CTA 的计时起点跨度为 0–32 ns。

2026-10-14 初版 CPU 检查：12 个条件分别运行局部 FIFO、跨 CTA FIFO、在途同源合并三个结构对照，在两组人工时间比值下检查响应，共 72 次模拟。8 项新测试通过，覆盖三请求就绪、槽代次/释放、源代次失效、独立投递和已知等价条件；四组预填时的逐请求发出、源就绪与目的就绪还与既有 GEMM 服务引擎逐项一致。单 CTA 且源复访不重叠时局部/共享 FIFO 等价；独立源且无在途自复访时源合并没有收益；完成延迟占主导时局部/共享差异可缩小。因此正式区分仍需候选参数范围与测量不确定性。这些结构对照使用归一化时间单位，未成为 GH200 参数或冻结预测；原 GEMM 失败判定保持不变。

```bash
python -B microbench/gh200_resource_campaign/access_rules/tma_tensor_multicta.py \
  --output /tmp/tma-tensor-condition-screening.json
PYTHONPATH=microbench/gh200_resource_campaign/access_rules \
  python -B -m unittest test_tma_tensor_multicta test_tma_input_events -v
```

2026-10-14 原生检查与校准见[开发记录](../../../../../../results/gh200_resource_campaign/access_rules/20261014-tma-tensor-development/README.md)。最终相位版作业 746421 通过 28 组检查，共 966,983,680 个 FP16 元素、59,020 组事件，错误为零；普通及相位 memcheck 均通过。资源为 40 个寄存器、132096 B SMEM、零 local bytes，每 SM 最多一个 CTA。各样本导出 CuTe Mainloop 工厂生成的 A/B 描述符编码；SASS 每组 A/B0/B1 三请求已核对。

开发中发现不参与消费的线程跑空循环，在结束时间戳前引入重汇合尾段；性能模式提前退出这些线程后，同条件尾段从 16913 降至 185 cycle。旧二进制、反例与复测均保留。最终二进制另完成 28 个 Q2 校准条件、每条件 7 个进程，源环相位保持 0。共享源 1 MiB 与独立源 64 MiB 的差异同时符合在途合并和工作集驻留解释，因此加入“预热工作集在 60 MiB 内用驻留价格，否则用流式价格”的竞争候选；该候选不合并在途请求。

[首批冻结](../../../../../../results/gh200_resource_campaign/access_rules/20261014-tma-tensor-discrimination-001/binding.json)绑定二进制、CPU 模型、校准输入、[预测](../../../../../../results/gh200_resource_campaign/access_rules/20261014-tma-tensor-discrimination-001/predictions.json)与[执行判据](../../../../../../results/gh200_resource_campaign/access_rules/20261014-tma-tensor-discrimination-001/plan.json)。离散价格筛选保留 26 组在途合并参数和 78 组驻留参数；相位对照分别预测增加 135.3%–252.0% 与 0%。每条件测 trace 开/关，各 7 个进程，共 168 个进程。主端点为 CTA 开始至最后一组释放；要求同一冻结价格组对全部 12 条件的误差均不超过 5%，相位响应误差不超过 5 个百分点，并通过未记录模式的方向对照。这是冻结价格集合的检验，连续参数范围未作完备辨识。

首批作业 746425 完成 168 个进程，槽复用与实际活动重叠均通过核对。[评分](../../../../../../results/gh200_resource_campaign/access_rules/20261014-tma-tensor-discrimination-001/score.json)中，相位对照 trace 开/关分别增加 9.8446%/10.3439%；两个候选的通过参数组均为零。最佳联合最大误差分别为 119.22% 与 15.28%。残差还包括单 CTA Q1 相对 Q4 的完成等待，以及 132 CTA 独立源在 Q1/Q4 下的吞吐变化。下一候选把驻留/流式的服务率和完成延迟分开，并保留在途共享；沿用原软件槽释放程序，先检查全部开发端点，再冻结新的消费者延迟与独立条件。

后续先检验 B 行距解释：[746430 配对](../../../../../../results/gh200_resource_campaign/access_rules/20261014-tma-pitch-discrimination-001/README.md)中，单 CTA 的 ldb 128→2048 只使主窗口增加 0.0984%，未记录模式响应为零；未支持预测约 11.66% 的独立行距延迟，不加入该参数。

将驻留与流式完成延迟分开后，开发条件最大误差仍为 5.52%；再加入按源地址分区的候选，P16 的开发最大误差为 4.68%，P32 为 5.57%。[第二轮留出](../../../../../../results/gh200_resource_campaign/access_rules/20261014-tma-tensor-heldout-002/README.md)完成 8 条件：N48 独立源相位 1/2、N32/96 的 Q1/Q4、N132 的 512/2048 ns 释放延迟。746459 的相位响应为 −0.1873%，相对预测 +8.9650% 构成反例；局部时间最大误差 41.45%，P16 不再作为采用候选。两个释放延迟条件的局部误差均低于 0.7%。

探针与 GEMM 已共用源服务实现；[首次迁移](../../../../../../results/gh200_resource_campaign/access_rules/20261014-tma-tensor-development/gemm-transfer-001/README.md)保留原计算/输出价格，只换 TMA 后，S6/S4 的消费者跨度同为 39.180 µs，stage 响应仍为零。下一步检查计算完成/槽释放及计算与搬运的服务耦合；[同 CTA WGMMA 背景开发对照](../../../../../../results/gh200_resource_campaign/access_rules/20261015-tma-compute-development-001/README.md)746470 已通过原生正确性、memcheck/synccheck 并完成 112 个采样进程。未记录模式下，WGMMA 背景使主窗口增加约 17.6%–31.2%；Q1 的记录分解显示新增时间主要位于发出、补发和进入等待之前。

[局部干扰留出 746720](../../../../../../results/gh200_resource_campaign/access_rules/20261015-tma-local-interference-heldout-001/README.md)固定其他131个SM的WGMMA，只开关目标SM的计算。新源环Q1/Q4的未记录窗口增加12.10%/20.50%周期，有效频率仅变−0.125%/−0.031%；两个端点均通过冻结的局部干扰预测，排除仅频率解释。发出、补发区间仍是主要增量，下一服务候选区分本地指令发出、SMEM访问与队列反压。 [SS/RS初次对照746721](../../../../../../results/gh200_resource_campaign/access_rules/20261015-tma-local-rs-development-001/README.md)存在A寄存器重复物化；[修正后746723](../../../../../../results/gh200_resource_campaign/access_rules/20261015-tma-local-rs-development-002/README.md)的背景推进率相差不到0.7%，未记录模式的额外周期仍接近。将A从SMEM改到寄存器未消除局部干扰。[746785位置留出](../../../../../../results/gh200_resource_campaign/access_rules/20261015-tma-issue-placement-heldout-002/README.md)中，较慢位置随issuer移动，但2.25%的未记录周期差低于首选候选的冻结范围，三种定量候选均未通过；保留位置关系假设，不采用固定干扰百分比。

[工作集判别 746486](../../../../../../results/gh200_resource_campaign/access_rules/20261015-tma-footprint-discrimination-001/README.md)保持 48 CTA、48 MiB 分配和描述符，只改变预热实际访问集合。24/48 MiB 的共同前缀请求窗口为 416/832 ns，周期响应 −49.37%，符合冻结的 −50%±12 个百分点；频率变化 −0.19%，未记录模式每组时间下降 44.14%。源工作集状态候选在该范围内获得支持；30/36 MiB 的开发端点为 416/528 ns，[量化服务候选](../../../../../../results/gh200_resource_campaign/access_rules/20261014-tma-tensor-development/working-set-001/README.md)对 34 个开发条件的最大误差为 8.45%，尚未达到局部时间 5% 标准；继续核对并发残差及输出存储对状态的影响。

[先前 global 写入判别 746495](../../../../../../results/gh200_resource_campaign/access_rules/20261015-tma-output-state-discrimination-001/README.md)完成 56 个采样进程。相同 30 MiB 输入前写入 16 MiB，使共同前缀请求窗口增加 101.92%、周期增加 106.42%，均符合冻结响应范围；未记录模式增加 95.71%，频率变化 +0.12%。输入服务须保留先前存储活动。将 A/B/D 共同计入状态后，G3 的开发 stage 响应为 9.66%，但原有空间跨度与预等约束仍失败，见[联合回放](../../../../../../results/gh200_resource_campaign/access_rules/20261014-tma-tensor-development/working-set-001/README.md)。下一步先区分地址与执行位置或请求时间状态，再冻结完整组合。

[地址轮换 746504](../../../../../../results/gh200_resource_campaign/access_rules/20261015-tma-position-discrimination-001/README.md)完成 28 个采样进程。固定执行位置、交换两个源组后，快慢组窗口差的配对变化仅 −2.86 cycle；地址解释预测 −475.86±100 cycle，位置或推进状态解释预测 0±100 cycle。后者通过，未记录模式和 SMID 映射核对一致。下一项改变起步次序，继续分开位置与时间状态。

[起步次序 746516](../../../../../../results/gh200_resource_campaign/access_rules/20261015-tma-start-order-discrimination-001/README.md)完成 42 个采样进程。正反次序均成功改变，但错开起步后的组间差仅约 11–15 cycle，低于冻结的可区分门槛，C2 记信息不足；同二进制共同时刻起步时为 200.43 cycle。下一候选须同时保留驻留状态与请求相位，不能直接采用恒定 SM 位置价格。

[暖机/本次时序 746541](../../../../../../results/gh200_resource_campaign/access_rules/20261015-tma-warmup-phase-discrimination-001/README.md)将两次起步独立控制，共 56 个进程。只改暖机时插桩组差变化 +0.07 cycle，只改正式调用时为 −239.57 cycle；本次请求时序候选通过该端点，但未插桩组差的降幅超出冻结幅度，两种定量候选均未通过。下一步用 cluster 共驻留核对快慢 SM 的同 GPC 关系，再限定共享服务的分组与相位规则。方法参考[GPU 拓扑研究 §5.1](https://arxiv.org/html/2609.24270v1#S5.SS1)与[CUDA cluster 共驻留语义](https://docs.nvidia.com/cuda/archive/12.9.2/cuda-c-programming-guide/index.html#thread-block-clusters)。

### 首轮条件安排与迭代终点

2026-10-14 用户授权目标模式持续迭代至完整模型交付，不设总资源上限。下表的 30＋8 保留为首轮条件设计，后续轮次按新的候选分歧与独立留出需要安排；每轮分别冻结预测、记录资源与结论。

| 项 | 规划上限或收口条件 |
|---|---|
| 首轮开发条件 | 最多两种源协议×5个CTA数×3个输入组并发数，共30条件；首批最多12条件，历史证据可替代的条件不新增 |
| 区分与留出 | 最多8个额外条件，优先分给消费者释放延迟与独立留出，其余按候选分歧用于背景相位等区分；同 SM 多 CTA 放置对照留待其他配置 |
| 重复与运行预算 | 重复进程数、先导测量规模、GPU分钟上限和编译修复次数在冻结的执行清单中一次写定；最终校准 28 条件×7 进程已完成；正式判别为 12 条件×2 记录模式×7 进程，单作业 15 分钟 |
| 终点 | 一轮开发加一次冻结干预检验后，交付共享范围约束、保留/排除的规则、对GEMM判断的影响；若现有观测不能辨识则列缺失条件并结束本轮 |

## 4. 结论

原生正确性、开发校准与首批 C2 判别已完成；两套冻结价格集合均未通过。统一价格的局部 FIFO 与共享 FIFO 对相同 N64/Q2 的共享/独立源预测相同，与校准不符；单一价格的在途合并和单一完成延迟的驻留解释未通过首批。两种规则的组合转入新一轮开发，单 CTA 探针继续用于控制与端点检查。

[同 GPC 关系核对 746548](../../../../../../results/gh200_resource_campaign/access_rules/20261015-gpc-topology-001/README.md)完成 21 次 cluster launch、6720 个 CTA 记录。SM0/16/32 确认同组；124–131 只确定四个内部配对。[带宽对照 746555](../../../../../../results/gh200_resource_campaign/access_rules/20261015-gpc-contention-002/README.md)中，四对均使原六个 SM 的窗口增加 21.8%–22.1%，与已知同 GPC 对照一致，已知跨组对照接近零；暂采用同组映射。下一步检验按该组管理的请求服务；前两次加载被移出循环的失败记录保留。

按该组建立的[有限并行服务开发](../../../../../../results/gh200_resource_campaign/access_rules/20261015-tma-gpc-development-001/README.md)能产生相位响应。随后[746563 独立相位留出](../../../../../../results/gh200_resource_campaign/access_rules/20261015-tma-gpc-heldout-001/README.md)实测0/16/32 ns间隔的窗口为528/512/416 ns。8槽候选提前下降、低估21.12%；2/4槽虽更接近，最大误差仍为5.29%/5.47%，三组参数均未通过。下一轮只保留2/4槽结构，先结合新增端点重新校准，再对未测相位或释放条件冻结验证。

[746570 新相位与 Q2 留出](../../../../../../results/gh200_resource_campaign/access_rules/20261015-tma-gpc-heldout-002/README.md)完成84个进程。2槽版本通过6条新条件，ns/cycle最大误差3.64%/2.32%；4槽在24 ns相位低估约10%，8槽约16%。暂采用2槽结构进入GEMM组合诊断，并保留它前轮同步起步条件5.29%的失败；不以本轮覆盖前轮。G3 S6/S4的20个历史dual进程分别只有一套SM映射，且两种stage之间相同，见[放置回放](../../../../../../results/gh200_resource_campaign/access_rules/20261015-tma-gpc-development-001/gemm-placement.json)。


将该服务直接接回 G3 后，不合并源请求会高估跨度，按组内在途同源合并又使 S4/S6 响应为零；加入全卡工作集状态也未改善，见[GEMM 迁移记录](../../../../../../results/gh200_resource_campaign/access_rules/20261015-tma-gpc-development-001/README.md)。下一轮固定已核对的 cfg_b 工作坐标，直接测 A/B 部分复用，检验源合并与独立目的路径的组合。


[匹配源判别 746586](../../../../../../results/gh200_resource_campaign/access_rules/20261015-tma-gemm-source-discrimination-001/README.md)使用 cfg_b 两轮实际工作坐标暖机，共 34 MiB A/B。四个源订阅版本均未通过，最大 ns/cycle 误差 45.14%/44.47%；无先前写入时 Q1 存在较大 GPC 差异，先写入 16.5 MiB 后窗口普遍变长，Q4 隐藏了大部分等待。下一项只交换暖机的两轮访问次序，正式请求保持不变，检验近期源状态。


[暖机次序 746588](../../../../../../results/gh200_resource_campaign/access_rules/20261015-tma-gemm-recency-discrimination-001/README.md)保持正式 cfg_b 请求相同，交换十次暖机的两轮次序。GPC2/3 的早窗口分别缩短41.30%/39.53%，未插桩时间缩短35.07%/32.65%；cycle/ns、放置与频率门槛均通过。近期源状态获得支持，固定位置价格被反证。接下来补齐存储局部域的工作映射，使模型按访问序列更新源状态。


[局部域验证 746596](../../../../../../results/gh200_resource_campaign/access_rules/20261015-numa-affinity-heldout-001/README.md)在三次新分配的 192 对新页上支持 GPC{0,4,5,6}/{1,2,3,7} 的两域映射，每域66个SM。新模型按此映射维护缓存与未完成填充，令命中、同源等待和每CTA推进共同决定就绪时刻；先回放暖机次序及匹配源条件，再接回GEMM。

[动态缓存开发](../../../../../../results/gh200_resource_campaign/access_rules/20261015-tma-gemm-source-development-001/README.md)的完全相联及 8/16/32 路候选尚未重现实测 GPC 分布；独立校准的 tensor 慢路径价格在 Q2 上仍有 16.33% 残差。746619 已确认 A/B 相距 22 MiB，746632 确认十次暖机的 CTA→SM 放置保持相同。随后[按域重排工作判别 746641](../../../../../../results/gh200_resource_campaign/access_rules/20261015-tma-domain-layout-discrimination-001/README.md)保持 264 个输出坐标与 34 MiB A/B，把候选每域副本需求从 30 MiB 降至 28.5 MiB：GPC2/3 的暖机次序响应从 −37.50%/−38.10% 降至 −9.38%/0，未插桩响应降至约 ±1.3%。局部域候选通过这两个预定端点，全局有效容量候选被反证。新布局的 GPC5 仍有较大响应，完整缓存规则继续辨识；输入状态与 D 写入接回 G3 后，stage 与空间响应仍未通过；[同卡复测](../../../../../../results/gh200_resource_campaign/access_rules/20261015-g3-phase-reproduction-001/README.md)确认目标现象可复现。已核实[原程序 scratch 清零](../../../../../../results/gh200_resource_campaign/access_rules/20261015-g3-allocation-query-001/README.md)的准备访问，按归属域写分配加入后仍未修复反例；释放与缓存替换候选的结果在开发入口维护。下一步按746720辨识本地计算与请求推进的耦合。
