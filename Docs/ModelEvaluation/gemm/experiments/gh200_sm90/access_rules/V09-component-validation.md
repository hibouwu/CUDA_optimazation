# V09：组件组合的新留出（冻结待确认）

当前版本为 **job738972-r2**，已冻结，尚未执行任何留出 GEMM。30个条件全部保留：29个有数值预测，cfg_c 的512×3968×1024因首轮B填零供给不可辨识而明确拒绝。完整时间仍以全组中位误差≤5%、最大≤10%为目标；存在拒绝或失败条件时，不能把可评分子集称为整组通过。

## 冻结身份

- 参考GPU：`GPU-43269fbc-449d-3e0f-908a-9c81229546d3`，romeo-a057，NVIDIA GH200 120GB，driver 590.48.01。
- 校准作业738972：COMPLETED 0，耗时6分31秒；6个已见参照的数值预检查与240个正式进程全部通过。没有留出样本目录，留出仅执行setup取得静态资源。
- 修订冻结时间：2026-10-09T19:55:09+02:00（Europe/Paris）。CPU修订源码提交：`929cb63c7a30c0b408f6aa35a2bf167321e3121c`。
- 预测SHA256：`f4df39e00cf1d2ab61947c911e699bd5bea4e533609d93f726de43077515d81e`。
- 清单SHA256：`09da85d124eb697a95c4695bd05281b07c37877890d0eca18afbdd2cf213f2ab`。
- [预测文件](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-freeze-job738972-r2/frozen/predictions.json)、[冻结清单](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-freeze-job738972-r2/frozen/manifest.json)、[模型参数](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-freeze-job738972-r2/frozen/calibration.json)、[完整条件与容量设置](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-freeze-job738972-r2/cases.json)。三份冻结文件均为只读，SHA已从回传字节复核。

原作业自动完成了校准、拟合与冻结。回传后发现，ARM上的等价供给参数解使局部Jacobian判定错误地接受了cfg_c G2；训练证据没有增强。r2只改支持判定：用已有条件线性规划检查等价参数能给出的供给范围，拒绝非唯一窗口。原冻结包保留，未获准采样；r2逐字节复用原ARM参数、GPU二进制、SASS和六参照数据，没有重新测GPU、改训练点或拟合参数。[修订来源](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-freeze-job738972-r2/freeze-revision.json)记录原SHA、测量源码与本地CPU冻结环境。

## 完整留出表

以下时间是冻结预测，单位µs，尚无实测误差。a/b/c分别为cfg_a/b/c；默认dyadic seed17、swizzle1，stage数为a/b的6与c的4。行距增量以字节计；未注明的行距等于逻辑尺寸。表中不把输入、行距或stage变化重复计作新形状。

| 条件ID | 配置 | M×N | K | 变化 | 预测µs |
|---|---|---|---:|---|---:|
| cfg_a_G1_base | a | 768×768 | 256 | 基线 | 8.6919 |
| cfg_b_G1_base | b | 768×768 | 256 | 基线 | 8.3335 |
| cfg_c_G1_base | c | 768×768 | 256 | 基线 | 10.3427 |
| cfg_a_G2_base | a | 512×3968 | 1024 | 基线 | 12.1825 |
| cfg_b_G2_base | b | 512×3968 | 1024 | 基线 | 11.8796 |
| cfg_c_G2_base | c | 512×3968 | 1024 | 基线 | 不支持 |
| cfg_a_G3_base | a | 2816×1536 | 4096 | 基线 | 50.5345 |
| cfg_b_G3_base | b | 2816×1536 | 4096 | 基线 | 50.7910 |
| cfg_c_G3_base | c | 2816×1536 | 4096 | 基线 | 49.7004 |
| cfg_a_G4_base | a | 3328×768 | 512 | 基线 | 14.2269 |
| cfg_b_G4_base | b | 3328×768 | 512 | 基线 | 13.0038 |
| cfg_c_G4_base | c | 3328×768 | 512 | 基线 | 12.6352 |
| cfg_a_G5_base | a | 2304×2304 | 4096 | 基线 | 72.1375 |
| cfg_b_G5_base | b | 2304×2304 | 4096 | 基线 | 72.3968 |
| cfg_c_G5_base | c | 2304×2304 | 4096 | 基线 | 89.7430 |
| cfg_a_G6_base | a | 6144×8192 | 32768 | sw8 | 6098.0419 |
| cfg_b_G6_base | b | 6144×8192 | 32768 | sw8 | 7071.9351 |
| cfg_c_G6_base | c | 6144×8192 | 32768 | sw8 | 5439.4747 |
| cfg_a_G5_random_seed29 | a | 2304×2304 | 4096 | random seed29 | 79.9828 |
| cfg_b_G5_random_seed29 | b | 2304×2304 | 4096 | random seed29 | 82.6248 |
| cfg_c_G5_random_seed29 | c | 2304×2304 | 4096 | random seed29 | 96.6857 |
| cfg_a_G3_A16 | a | 2816×1536 | 4096 | A+16B | 52.8384 |
| cfg_b_G3_A16 | b | 2816×1536 | 4096 | A+16B | 51.5014 |
| cfg_c_G3_A16 | c | 2816×1536 | 4096 | A+16B | 49.7004 |
| cfg_b_G5_B32 | b | 2304×2304 | 4096 | B+32B | 77.1255 |
| cfg_b_G5_B80 | b | 2304×2304 | 4096 | B+80B | 86.7875 |
| cfg_b_G5_B112 | b | 2304×2304 | 4096 | B+112B | 86.7875 |
| cfg_b_G3_stage4 | b | 2816×1536 | 4096 | stage4 | 50.7910 |
| cfg_a_P1_partial_first_and_later | a | 960×2816 | 4096 | 部分边界 | 47.5454 |
| cfg_c_P2_partial_later_sw8 | c | 4096×2240 | 4096 | sw8；部分边界 | 188.0030 |

共8个不同M×N，其中6个基础形状各覆盖三配置；cfg_a/b/c分别9/12/9条件。与97份含实际样本的历史条件文件核对，8个M×N均未出现，[扫描依据](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-freeze-job738972-r2/inputs/novelty.json)。

## 模型与限制

供给使用R13 first/later同形式的条件max规则，事件递推沿用V06/V08；输出使用R15的single/multi与有效/越界分类；频率使用R09的计算/源请求活动候选。plain时间映射只由实测dual包络与plain event重新定值，不用组件误差作拟合目标。原校准行及来源均绑定进清单。

拒绝项的两个CTA（62/63）在预测频率约1.86 GHz下可能转入供给受限，现有数据没有唯一确定对应价格。r2的固定预测频率下，等价拟合参数给出首轮L约9.066–9.409 µs，超过既有1e-7相对辨识容差；[测后离线复核](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-freeze-job738972-r2/reanalysis/support-check/report.json)保存参数SHA、频率与条件区间。这只是固定频率、固定已拟合分支的范围，不是置信区间，也未重新联立时钟，因而没有用范围中点冒充受支持的冻结预测。详见[R13首轮补测](R13-async-retirement.md)与[支持定位](../../../../../../results/gh200_resource_campaign/access_rules/20261009-R13-R15-R18-composition-job738496/reanalysis/A-20261009-first-fill-merged-v1/support-diagnosis.json)。

短/长K、小grid、stage4、partial与新random seed是本轮待验证的迁移；代数支持不等于测量通过。部分输出插值、cfg_c普通多轮middle代理与零工作比例时钟混合仍是候选。局部慢窗口、首段供给和关键CTA定位的误差分别报告，不被总时间成绩覆盖。尚未覆盖跨卡、A+32/64/96B、基址偏移、rotation、K尾部、cfg_b partial以及A16+B非零组合。

## 采样条件

取得用户对本页30条件及上述两个SHA的确认后，才提交作业二。重新申请同一参考卡，先核验源码/二进制/清单身份并复测预列六参照；plain时间和dual包络允许±5%，有效频率±3%，同调用event与包络差允许±0.5 µs。超限停止，不现场调参。738972相对旧参照的最大变化分别为0.605%、0.487%、0.649%，绝对差检查也通过，见[参照检查](../../../../../../results/gh200_resource_campaign/access_rules/20261009-V09-freeze-job738972-r2/reference-check.json)。

作业二需要一份不含既有samples的执行副本，保留frozen文件的只读属性；两项期望SHA由用户确认版本传入heldout.sh。通过参照后才执行30条件的原容差数值检查与计时。失败条件和拒绝项均保留，分项报告首段、预填、first/later主循环、末次输出、尾段及关键CTA并列情况。本版本没有自动获得任何留出采样许可。
