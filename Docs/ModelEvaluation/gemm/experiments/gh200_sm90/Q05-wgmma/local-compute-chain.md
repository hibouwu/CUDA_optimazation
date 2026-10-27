# 计算完成与提交

11 个目录（10-18、10-19）：驻留 A 计算链上的 wait0/wait1、提交提前量、首组路径与软件返回分支，以及独立计算模块的抽取。

## 结论

**冻结留出获支持**
- 完成滞后 d95：wait1 下每 Ktile 512 cycle，wait0 下 607 cycle。32 Ktile 冻结留出预测 20006.5／17156.5、实测 20010／17153 cycle，误差约 0.02%（751957）。固定截距形式的 wait0 误差 11.2%，被反证。
- 驻留 plain 窗口 769+512·K 与软件返回 448+28·[K>1]：K2 冻结留出误差 0.33%／0（756100）。

**被反证或不可辨识**
- 提交提前量 293 cycle 不在 256、381 的 5% 带内，两个完整局部形式停止（756036）。
- 提交提前量与调用控制的分配不能靠增加延迟点区分。
- K1 的提交到 tail 实测 449 cycle，476 与 351 两个模板均被反证（756090）。
- 独立满速 WG 形式被官方容量排除，保留共享进度。

**模型现状与迁移**
- 独立计算入口 `cfg_b_compute_chain.compute_timeline` 与完整请求引擎共用同一转换。
- d95 接入完整 GEMM 后，S6/S4 plain 只变化约 +0.06 µs，stage 响应 3.903%→3.899%。规则见 [COMPUTE-SERVICE](../../../model/mechanism/hardware_rules/COMPUTE-SERVICE.md)。

## 轮次

按日期排列；全部结果目录见[Q05-wgmma/local-compute-chain](../../../../../../results/gh200_resource_campaign/access_rules/Q05-wgmma/local-compute-chain/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| Q05计算等待组织开发（751950） | 同驻留8条WGMMA/Ktile，只改wait0/1；4／13 Ktile四条件12进程，布局／资源／消费者固定 | wait1长度斜率512、wait0为607；预列每组完成滞后得到d95与窗口固定项a677.5，两臂固定截距形式开发失败 | [开发与冻结](../../../../../../results/gh200_resource_campaign/access_rules/Q05-wgmma/local-compute-chain/20261018-Q05-compute-wait-001/EXECUTION.md) |
| Q05冻结长度验证（751957） | 同构建及开发冻结参数，32 Ktile两wait、六进程；不重拟合 | 两臂冻结窗口误差0.017491%／0.020405%；固定截距wait0误差11.156922%，匹配长度连接获支持，GEMM迁移待检 | [留出](../../../../../../results/gh200_resource_campaign/access_rules/Q05-wgmma/local-compute-chain/20261018-Q05-compute-wait-002/EXECUTION.md) |
| Q05计算模块能否独立运行并保留组合行为（CPU） | 提取共同单组转换；固定价格，核四请求组织、六反馈链与两wait长度公式 | 全部字段逐位保持，九独立时间线与七项检查通过；64调用、修复／拟合／GPU0，原硬件资格保持 | [输入、回归与复现](../../../../../../results/gh200_resource_campaign/access_rules/Q05-wgmma/local-compute-chain/20261018-Q05-local-compute-service-001/EXECUTION.md) |
| 跨WG组合是否重复分配单SM计算能力（关联问题5，CPU必要界） | 固定q512／lead256／d95和四组既有许可，按官方单SM容量核累计完成工作 | 独立满速形式在S6／S4／K2提前交接违反容量161cycle；共享进度全部满足，不新增GPU | [规格、逐端点及收口](../../../../../../results/gh200_resource_campaign/access_rules/Q05-wgmma/local-compute-chain/20261019-Q05-compute-capacity-bound-001/EXECUTION.md) |
| Q05首次提交后的提前量能否从局部干预辨识（755945，关联问题5） | 固定驻留A的K13／六槽／正常wait1，前置或首commit后一次延迟；旧版与两新零臂先桥接 | 旧六窗口等价；15进程零桥接及数值通过，响应差267cycle但实际剂量差29超12.8，信息不足、留出未采；两修复，原始／异目录评分一致 | [候选、构建与有限清单](../../../../../../results/gh200_resource_campaign/access_rules/Q05-wgmma/local-compute-chain/20261019-Q05-commit-lead-001/README.md) |
| Q05匹配活动延迟能否保留两个提前量形式（756036，关联问题5） | 两位置调用同一物理函数，保留计算／等待／六槽；测前将主端点改为匹配正剂量的完整窗口差 | 15进程、18项资格全部通过，活动跨度差−3cycle；主端点293超256／381的原5%带，两个完整局部形式停止，留出未采；39输入及两地原始评分保持 | [反例、条件与收口](../../../../../../results/gh200_resource_campaign/access_rules/Q05-wgmma/local-compute-chain/20261019-Q05-commit-lead-002/README.md) |
| Q05已有窗口能否区分lead与调用控制（CPU） | 固定756036同批数据及两个lead，构造共同调用控制见证，以独立有理方程复核 | 两组见证各过12窗口；秩4／5的等价变换对更多延迟长度保持，两地96核对一致，停止同边界延迟扩展；未采用见证价格 | [证明与缺失端点](../../../../../../results/gh200_resource_campaign/access_rules/Q05-wgmma/local-compute-chain/20261019-Q05-call-context-identifiability-001/README.md) |
| Q05已有trace是否提供首次物理完成边界（CPU） | 核实际ARM源码／SASS和18原始记录，以实际窗口构造异步观察反例 | 九配对只有最后组软件429／28／476cycle端点；整体38／44cycle增量仍容许局部125cycle，首次及局部误差界未辨识 | [复核与结论](../../../../../../results/gh200_resource_campaign/access_rules/Q05-wgmma/local-compute-chain/20261019-Q05-observer-boundary-001/README.md)；[K1／K2准备](../../../../../../results/gh200_resource_campaign/access_rules/Q05-wgmma/local-compute-chain/20261019-Q05-first-group-001/README.md) |
| Q05最后组软件返回窗口能否迁移首组（756090，关联问题5） | host开放K1／K2，设备完整编码与旧A保持；K4同批桥接后比较K1 plain／trace，原K2冻结留出只在唯一完整形式过门后执行 | 12进程、24项资格全过；K1完整窗口通过，提交到tail449cycle反证476／351两模板，留出未采；23输入及两地原始评分保持 | [条件、反例与收口](../../../../../../results/gh200_resource_campaign/access_rules/Q05-wgmma/local-compute-chain/20261019-Q05-first-group-001/README.md) |
| Q05按真实正常wait分支能否组合软件返回端点（CPU，关联问题5） | 固定旧28／448cycle片段、原完整窗口及实际设备路径；只按首组跳过正常wait连接，复核30条既有raw | 10项开发检查全过，44输入及两地报告保持，60契约调用；交付软件端点经验替代，冻结未采K2，物理提前量仍未辨识 | [契约、来源与开发检查](../../../../../../results/gh200_resource_campaign/access_rules/Q05-wgmma/local-compute-chain/20261019-Q05-return-branch-contract-001/README.md) |
| Q05软件分支契约能否外推K2（756100，关联问题5） | 同一binary，K4 plain／trace桥接后执行K2两未采留出；三seed，预测及原5%门测前固定 | 12进程、24项资格全过；K2完整窗口1799对1793、软件返回476对476通过，保留局部外推；22输入及两地原始评分保持 | [冻结与验证](../../../../../../results/gh200_resource_campaign/access_rules/Q05-wgmma/local-compute-chain/20261019-Q05-return-branch-validation-001/README.md) |
| a044 跨卡复核（mechanism-r3，780769） | wait0/1 共 18 次、commit-lead 15 次、first-group 12 次、return-branch 12 次，原评分器 | K32 冻结预测误差 0.0175%/0.0204%，通过；commit-lead 剂量差 29 cycle，仍未过等剂量资格；K1 commit→tail 为 449 cycle，再次排除 476/351；return-branch 周期端点全部匹配，但 K2 seed3 的 plain/trace 差 64 ns（6.06%），本轮资格未过 | [等待复核](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/q05-wait-returned-report.md)、[分支复核](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/q05-returned-branches-report.md) |
| a044 保序修复（785518–785575） | commit-lead-002 15 次，原顺序与环境，原评分器 | zero/dev/heldout 资格通过，无保留形式，与旧卡相同 | [修复报告](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/mechanism-repair-r2-report.md#q02q04q05-原评分器重评) |
