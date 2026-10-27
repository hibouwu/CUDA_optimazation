# C1-01：八个 CUTLASS 候选的选配置与完整时间

更新：2026-10-15。按 [GOALS](../../../GOALS.md) 先登记，再测量。

## 登记

| 项 | 内容 |
|---|---|
| 服务的能力 | C1 选配置；同时按 C3 报告完整时间误差。本轮不检验 C2 机制主张 |
| 预期改变的决定 | 选哪一类模型作为 C1/C3 的主模型；下一轮补什么数据 |
| 终止条件 | 冻结后的留出评分完成即结束本轮；结论按通过、被反证、信息不足记录 |

## 条件与候选

FP16→FP32，α=1、β=0，A/B 行主序，lda=K、ldb=N、ldd=N，swizzle=1，无填零。M、N∈[256,16384]，K∈[256,32768]，均为 64 的倍数、log 均匀；输入 zero/dyadic/random 等概率，seed=17。开发集 360 个条件用拉丁超立方，留出 120 个条件按同一分布独立抽样（[设计脚本](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261013-C1-design/design.py)，条件清单 SHA256 `0197166c…`）。

| 候选 | tile | 调度 | cluster | stage |
|---|---|---|---|---:|
| cfg_a | 128×128×64 | cooperative | 2×1 | 6 |
| cfg_a_s4 | 128×128×64 | cooperative | 2×1 | 4 |
| cfg_a1 | 128×128×64 | cooperative | 1×1 | 6 |
| cfg_b | 128×128×64 | pingpong | 1×1 | 6 |
| cfg_b_s4 | 128×128×64 | pingpong | 1×1 | 4 |
| cfg_c | 256×128×64 | cooperative | 1×2 | 4 |
| cfg_c_s3 | 256×128×64 | cooperative | 1×2 | 3 |
| cfg_c1 | 256×128×64 | cooperative | 1×1 | 4 |

恢复测量采用同卡单kernel graph暖机与目标，每个case为plain 7个进程、dual 2个进程；原取消设计为3/1。360开发条件拆成30块，每块12条件，前中后各测原六参照，记录200ms遥测。协议选择依据见[八候选对照](../../../../../../results/gh200_resource_campaign/access_rules/Q10-time-mapping/c1-timing-protocol/20261015-c1-graph-protocol-expansion-002/RESULTS.md)。

## 模型与选择规则

全部在开发集上用按条件分组的 10 折交叉验证（重复 3 次）比较；三次预测分别计入指标，C1 按重复编号和条件分组，C3 汇总三次误差。并检查 K 最大十分位、规模最大十分位和单一输入留出的外推。冻结时所有模型一起冻结、全部在留出上报告。除规格基线外都是半经验或经验模型：structural 的参数是有效值，名称不表示机制已验证，也不构成 C2 证据。

| 模型 | 形式 |
|---|---|
| spec_baseline | 规格基线：max(2MNK/P, (A+B+D 字节)/B_HBM)，P=132×4096 FLOP/cycle×1.98 GHz，B_HBM=4.0 TB/s；暖 L2 下不是严格下界，只报告 C3 差距 |
| fixed | 训练集上平均 regret 最小的单一候选 |
| wave_roofline | 带假设的时间估计：max(波数×Kt×q/f_max, 波内面板 HBM 字节/带宽) + 固定 4 µs，无拟合 |
| tilesight | 公开模型（commit 75428c4，H200 microbench profile 改 HBM 与 L2 容量），不用本项目实测；kernel_body + 其默认 2 µs 启动；不能表达 pingpong 与 stage 差别 |
| wave_bilinear | WaveTune 形式，每候选 T=a·W·L+b·W+c·L+d |
| gbdt / gbdt_on_bilinear | LightGBM 学 log 时间，或学 wave_bilinear 的残差 |
| relative | 条件内相对 log 时间单独建模，水平取 gbdt_on_bilinear |
| decomposed | log 周期与 log 有效频率各一个 GBDT，plain = F + κ·周期/频率 |
| structural | 带物理形式的半经验模型：周期（tensor、L2 供给份额、stage 在途约束、尾段、启动）与功率上限下的占空比频率联立，参数全局有效值 |
| structural_gbdt / relative_structural | structural 加 GBDT 残差，及其相对版本 |

主模型在冻结前按交叉验证机械选定：C1 取平均 regret 最小者（相同时比较最大 regret），C3 取最大误差最小者（相同时比较中位）。

## 判定标准

| 能力 | 标准 |
|---|---|
| C1 | 留出每个条件 regret ≤5%；同时报告平均、p90、top-1 和以单进程选择的测量噪声下限 |
| C3 | 留出全部 960 个 case 的完整时间误差中位 ≤5%、最大 ≤10%，无拒绝项；dual 的有效频率与包络误差单列为分项诊断 |

## 执行记录

正式开发首块b00的746873完成前六条件、540个槽后，在d333_cfg_a1的原数值容差检查处停止；48条主体case已独立回放。746893的八候选、三个K共24项复核均超出原容差。用户已选择另行制定并验证长K误差标准，见[数值诊断](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261015-C1-numeric-development-002/RESULTS.md)；旧协议失败保留，模型尚未冻结，也没有留出成绩。原冒烟部分都在参考卡 GPU-43269fbc（romeo-a057）上，[本地归档](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261013-C1-dev/)含驱动、构建、冒烟样本、解析与传输清单；远端目录为 `/gpfs/projet/r260073/hibouwu/gh200-c1-dev-20261013T0731Z`。原分块记录见[开发v1](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261015-C1-development-001/RESULTS.md)；新数值协议的采集见[开发v2](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261015-C1-development-002/README.md)，首块746920的1026槽全部成功，96条主体记录独立回放通过；第二块746937独立回放通过（1025成功、1暖机耗尽），前两块共24条件、192条记录可用；第三块746945独立回放通过，前三块共36条件、288条记录可用；第四块746956回放通过11条件，d226因cfg_a暖机奇偶交替而不齐备；前四块47条件、376条记录可用，第五块746980回放通过，前五块59条件、472条记录可用；第六块746988回放通过，前六块71条件、568条记录可用；第七块746999独立回放通过，前七块83条件、664条记录可用；第八块747016独立回放通过，前八块95条件、760条记录可用；第九块747026独立回放通过，前九块107条件、856条记录可用；第十块747037独立回放通过11条件，d037因cfg_a暖机波动不齐备；前十块118条件、944条记录可用；第十一块747052独立回放通过，前十一块130条件、1040条记录可用；第十二至十七块747064、747074、747086、747099、747117、747124独立回放通过，前十七块202条件、1616条记录可用；128容量trace经747102验证通过，b15的新包已完成正式回放；b17因节点空间耗尽失败，保留失败包，747173按原输入整块重跑并通过，747180/747192/747200的b18/b19/b20也独立回放通过，b21/b22（747214/747379）也通过独立回放，b23（747440）也完成独立回放，b24（747469）回放通过11条件，d058因cfg_c暖机奇偶交替不齐备；b25（747498）回放通过12条件，前二十六块309条件、2472条记录可用，见[开发结果](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261015-C1-development-002/RESULTS.md)。

a044（GPU-7473147b，1980 MHz）在 10-27 重测了全部 30 块开发矩阵（作业 786088–786501）：30,769/30,780 槽成功，原回放 359 个条件可用、356 个完整，是第一次得到 30 块齐全的开发数据。与旧卡共同完整的 304 个条件相比，plain 中位快 0.73%，242 个条件的最快候选不变，沿用旧选择的最大 regret 为 8.1%（d310）。见[a044 开发结果](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261023-Q16-a044-development-001/RESULTS.md)。模型比较、冻结和留出尚未执行。

| 作业 | 内容 | 结果 |
|---|---|---|
| 743011 | 复用 V09 的 cfg_a/b/b_s4/c 二进制，新编 cfg_a_s4、cfg_a1、cfg_c_s3、cfg_c1 的 plain/dual | 完成 |
| 743016 | 冒烟：3 个条件×8 候选，加 V09 的 6 条参照（前、中、后各一块） | 24 个 case 全部解析出 plain、有效频率与包络；最大条件 10944×5568×32000 有 2 次“预热后 5 次 CV>2%”失败尝试，重试后完成，其中 1 个 dual 缺频率 |
| 743051 | 遥测轮询开/关交替 6 次，参照 6 条与 d005 三候选 | 配对差中位：参照 −0.38%～+1.10%，8 µs 小 kernel −1.17%～+1.95%，无一致方向；正式作业可以开遥测 |
| 743973–743976 | 正式开发作业（四段串联） | 用户要求停止，排队中取消，未开始运行 |

模型代码已在历史数据和24条冒烟记录上试跑，并在[前七块正式开发数据](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261015-C1-development-002/analysis/first-seven-blocks/README.md)上完成12类模型的分组交叉验证与区域外推比较；[前十八块比较](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261015-C1-development-002/analysis/first-eighteen-blocks/README.md)扩展到214条件，C1首选的平均/最大regret为0.29%/6.47%，C3首选的中位/最大误差为1.92%/34.08%；正式选择与冻结等待全部开发数据。新[开发版本](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261015-C1-models/README.md)将无校准依据的50MiB缓存常量改为同卡实测60MiB，并修复缺失候选仍能给出C1通过的评分反例。[TileSight 基线](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261013-C1-baselines/tilesight/README.md)已对 480 条件×8 候选给出静态预测；前七块的实测比较已包含该基线，全部开发集及独立留出评分仍待完成。

## 恢复前要定的事

冒烟与遥测对照显示，计时噪声与作业间差异已经接近判定标准：

| 观测 | 743016（冒烟） | 743051（40 分钟后） |
|---|---|---|
| 参照 plain 中位相对 V09 冻结值 | +0.4%～+6.6% | −0.2%～+1.6% |
| 参照 event−包络 | 比 V09 多 0.85～3.2 µs | 3.73～4.72 µs，V09 为 3.86～3.98 µs |
| 同一 case 的进程间极差 | 参照最大 6.4% | 参照 2.2%～4.1%；8 µs kernel 10.8%～15.0% |

2026-10-15 [单次调用计时对照](../../../../../../results/gh200_resource_campaign/access_rules/Q10-time-mapping/c1-timing-protocol/20261015-c1-timing-protocol-development-002/RESULTS.md)已完成：746805/746820共318个槽位成功、2个暖机耗尽。cfg_b长条件的单kernel graph将plain和额外时间分布宽度都减半以上，设备包络、周期、频率变化均小于3%；短条件方向相同，但stream/plain与graph/plain各缺一个合格槽位。随后746848/746850完成原八候选和三个冒烟形状的334个成功槽、2个耗尽槽；graph暖机首尝试失败从13次降到3次，4个齐备case的设备变化超过3%。完整等价主张未通过，正式开发固定graph暖机和graph目标。

恢复测量的两项协议决定如下：

1. **样本数。** 统一plain 7、dual 2，依据旧同作业短case连续窗口与八候选间距；具体数字在上述对照结果中维护。每case至少5个合格plain、1个dual才可用，实际计数和完整7/2标记分别保留；成功与暖机耗尽槽不补测。评分保留单进程选择的噪声下限，原C1/C3门槛不变。
2. **作业间漂移。** 每块前中后六参照按同协议测量，以三阶段合并后各参照plain中位对开发基准±3%判断；开发基准在留出前冻结，单阶段变化另报。超出时C3记为信息不足，C1继续按同作业内候选的相对时间评分。

[留出静态容量核对](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261015-C1-heldout-static-audit-001/README.md)覆盖原120条件×8候选：每消费角色最多53个输出，原64槽trace足够；最大K=31168。留出测量继续等待模型与预测冻结。

[留出执行入口](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261015-C1-heldout-runner-001/README.md)已完成CPU验证，沿用原采样实现；正式包须绑定已选主模型的960条冻结预测及全部30开发块的参照基线。尚未提交留出作业。

[模型交付包验证](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261015-C1-bundle-qualification-001/README.md)已完成：12 类已训练模型在新目录、新进程中对 1712 条开发输入逐位重放，完整冻结命令生成的源码、输入和训练状态均已校验。正式留出冻结仍等待全部开发数据及最终参照基准。

## 相关轮次

以下各轮原登记在访问规则索引，按日期排列；全部结果目录见[Q16-config-selection/C1-01-selection](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| <a id="round-c1-development-comparison"></a>既定模型能否通过C1／C3开发比较（开发采样与CPU交叉验证） | 固定八个CUTLASS候选、原360开发条件与分组交叉验证规则，使用已验证数值协议和graph计时，按同作业内结果比较模型 | 已回放26块309条件；前十八块比较中C1首选最大regret6.473%、C3首选最大误差34.075%，均有超标条件，正式冻结与独立留出未完成 | [开发输入与分块结果](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261015-C1-development-002/RESULTS.md)、[前十八块模型比较与逐折诊断](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261015-C1-development-002/analysis/first-eighteen-blocks/README.md)、[模型／评分实现及反例修复](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261015-C1-models)、[全部开发比较准备](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261015-C1-development-002/analysis/full-comparison/README.md)；[开发v1数值失败](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261015-C1-development-001)、[b17存储耗尽后整块恢复](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261015-C1-storage-recovery-001)为执行历史 |
| <a id="round-c1-numeric"></a>长K旧容差失败是否只属于一个候选（746893，开发） | 固定random seed17及相同抽查索引，比较八候选在K=7360／14656／29248的输出与FP64参考 | 同K八候选输出逐位相同且三个K都超原容差；转查共同累加路径并制定独立新版数值标准，旧失败保留 | [诊断结果与重算](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261015-C1-numeric-development-002/RESULTS.md)、[逐候选分析](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261015-C1-numeric-development-002/analysis.json)；[001未提交原因](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261015-C1-numeric-development-001)为同轮准备历史 |
| 逐K16误差界是否可用于现有候选（746908，新数值协议验证） | 固定原GEMM设备SASS与资源，只换host数值检查，覆盖旧失败、新seed／K与dyadic精确输入 | 144目标及独立回放通过，新界接入开发v2；原数值协议失败保持原判定 | [数值合同与适用域](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261015-C1-accuracy-v2-001/README.md)、[验证结果](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261015-C1-accuracy-v2-001/RESULTS.md) |
| <a id="round-c1-trace-capacity"></a>trace扩到128槽是否完整且兼容（747102，资格验证） | 固定三个cooperative候选的plain及采样协议，只扩dual容量并比较旧溢出见证、超容量条件和64／128匹配控制 | 完整记录通过，时间最大变化1.655%、频率1.220%，满足原3%／2%门槛；三种dual获得后续超容量条件资格 | [通过结果](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261015-C1-trace-capacity-003/RESULTS.md)、[独立评分](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261015-C1-trace-capacity-003/score.json)；[001编译失败](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261015-C1-trace-capacity-001)、[002解析失败](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261015-C1-trace-capacity-002)为执行历史，[四块容量接续](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261015-C1-trace-continuation-001)为接入产物 |
| 模型推理是否只使用运行前输入（CPU资格核对） | 固定开发训练状态和1712条候选输入，分别保留全记录、仅静态字段、将实测时间／频率等替换为极端值 | 12类模型及分项预测逐位一致，当前推理输入依赖通过核对 | [输入依赖核对](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261015-C1-prediction-input-audit-001/README.md)、[逐模型结果](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261015-C1-prediction-input-audit-001/report.json) |
| 已训练模型包迁移后能否直接重放（CPU交付资格） | 固定开发训练状态、源码与依赖，将包复制到新目录并由新进程加载，另验证完整冻结命令 | 全部预测及周期／频率／包络分项逐位一致；开发包可重放，正式预测冻结仍等待全部开发与参照基线 | [迁移与冻结命令核对](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261015-C1-bundle-qualification-001/README.md)、[逐项结果](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261015-C1-bundle-qualification-001/report.json)；[留出静态容量](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261015-C1-heldout-static-audit-001)及[留出执行入口](../../../../../../results/gh200_resource_campaign/access_rules/Q16-config-selection/C1-01-selection/20261015-C1-heldout-runner-001)为未测留出的准备产物 |
