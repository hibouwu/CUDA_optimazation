# R09：仅用前序event的共同／输出序号换算候选

前序event加固定校准系数，可以在本批开发数据中给出不读取目标cycle/ns的局部时间候选。一个共同系数的全窗口误差中位绝对值3.5410%、最大6.4464%；分首／次输出的两个系数为0.7980%／3.4367%。两者都是已揭示数据上的开发诊断，原局部观察失败仍在，不能升级成冻结留出成绩或公共时间规则。

## 输入与固定方法

校准只取740091的10个flag0进程，未读取flag1的数值窗口；评分使用740092全部20个flag0进程、两臂、4320个共同计算窗口。两批binary SHA均9fdbf10afe162a92f44a889a1fe3c87b89f57a0e73a5ec385abde32035742b55；固定GPU-ef8692f3-8bd7-80a2-608b-10731c9ec114、cfg_c 2304×3072×4096、tile256×128×64、S4、cluster1×2、compact行距、dyadic/seed17/sw1、16次同buffer同dual的observer-off前序，目标为首个调用且observer-off。其它组织、输入、前序、观察flag与设备不在该接口范围。

E_p是该进程末次前序的CUDA event，统一换算为ns。校准进程j中，分别取输出序号o的所有CTA共同窗口C/N中位，然后以进程等权方式校准：

```text
kappa_j,o = E_p,j × median_CTA(C_j,o / N_j,o)
kappa_o   = median_10process(kappa_j,o)
kappa_common = median_20process-ordinal(kappa_j,o)
rate_hat_o = kappa_o / E_p,target
N_hat_o = C_static / rate_hat_o
```

共同基线使用同一个kappa_common预测两个序号。两个方案和中位规则都在本次评分前写入contract和实现；没有搜索前序下标、滤波器、CTA系数、损失函数或异常子集。kappa_common≈155671.491839cycle；两个kappa为161241.282476／149151.884595cycle。它们是所声明观测协议的条件比例系数，不能解释为物理时钟或裸服务价格。

C_static仅取740095预定1024组独立MMA参考，按静态工作量得到1049379/16cycle。没有使用评分目标cycle修正它，没有减803cycle或拟合截距。740095晚于评分批次，且两个批次的结果此前已揭示；本次freeze只证明校准与此评分实现分离，不是前瞻采集或新的独立留出。

## 不读取当前目标的评分

预测函数只接收完整静态setup、一个末次前序event和冻结校准。输出序号来自已知排程，不读取目标CTA/SM分配、配对臂、当前event、cycle、ns或残差。评分脚本先计算两个序号的预测，再解码目标trace；源码证明timed()在event同步、读取并返回后，才完成前序向量插入及目标启动。现有程序没有接入本Python预测函数，但所需event已在目标前位于主机。

误差为预测/记录−1；表中均为全部相应窗口的中位绝对／最大绝对误差，单位%。

| 换算候选 | 序号0，2640窗口 | 序号1，1680窗口 | 全4320窗口 |
|---|---:|---:|---:|
| 一个共同系数 | 3.0529 / 5.9695 | 4.6471 / 6.4464 | 3.5410 / 6.4464 |
| 两个序号系数 | 0.7470 / 2.9884 | 0.8727 / 3.4367 | 0.7980 / 3.4367 |

因此当前证据不支持“为了5%/10%的相对误差尺度，必须加入输出序号状态”；较简单的共同候选也值得在新冻结局部留出中检验。两系数减小开发偏差，没有因此取得必要性或更强资格。全SM仅是描述性迁移诊断，SM124的40窗口另列于report.json；不把单SM指令参考自动扩成全SM价格支持。

逐窗口保留e_C=C_static/C−1、条件e_f=(C/N)/rate_hat−1、自由e_N=N_hat/N−1，8640条两方法误差精确满足(1+e_C)(1+e_f)=1+e_N。e_f显式用目标C作诊断；自由预测本身没有使用目标C。全部80个方法／进程／输出序号分组以及同SM分层均保留，各分项中位不相乘。单独的末次前序event→目标完整event基线中位绝对0.4621%、最大2.0212%，不由局部窗口组合得到。

## 身份、复验与边界

freeze SHA为0446ece5a1318ba5740b21aba9cb762abc960e773dede7712c1ffbad14a30661，绑定contract、校准、预测模块和完整评分实现。167项输入SHA包含两批冻结链、40份raw的身份、独立参考的已核验输入链与真实源码；本轮校准／评分只解析其中30份raw，另外10份仅沿原迁移报告绑定哈希。沿用已独立复核的原数值证据；这里检查check标记及文件身份，不重复完整数值参考计算。

verify.py不导入分析器，独立从raw重建2160校准窗口、20个进程序号中心、三个系数、4320评分窗口、8640项有标签分项及80组统计；从三份长参考raw重新推导静态周期。另调用预测模块，以限制读取的setup和不可用目标字段测试接口，检验前序时间翻倍时预测率减半，拒绝8种不支持setup、4种非法前序，并拒绝仍乘法闭合的分项互换负例。

直接复验：

```sh
python /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-off-off/reanalysis/A-prefix-ordinal-clock-v1/verify.py
```

新目录复放：

```sh
r09_prefix_output=$(mktemp -d /tmp/r09-prefix-clock-XXXXXX)
cp /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261011-cfg-c-producer-tail-off-off/reanalysis/A-prefix-ordinal-clock-v1/{contract.json,analyze.py,verify.py,r09_prefix_ordinal_clock.py} "$r09_prefix_output/"
python "$r09_prefix_output/analyze.py" calibrate --base /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules
python "$r09_prefix_output/analyze.py" evaluate --base /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules
python "$r09_prefix_output/verify.py"
```

校准末次前序event范围92.3520–94.1120µs，评分为92.5760–95.2960µs，越出校准时长范围的记录也全部保留。模型在数学上对正时长给出比例外推，除本批开发记录外没有范围外有效性证据。不得由这里外推其它K、布局、cfg_a/b、swizzle/源切换、连续或镜像历史；跨卡至少需要重新取得匹配的原生周期参考和该观测协议的转换校准，并另验其迁移，不能只换UUID。计时端点的物理准确度、局部观察、新冻结留出、完整模型组合与跨卡边界均未完成，所有qualification保持false。

没有GPU操作，没有修改B的源口径/价格、公共时钟、组合或runner。
