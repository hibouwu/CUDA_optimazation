# EXP-20：GH200 BF16 WGMMA 固定 tile 组合实测

实际数据与报告C已通过，105个条件工作率可进入发布桥接。发布复核已完成，条件观测已纳入公共索引。固定一个CTA、128线程、64×64×64 BF16输入和FP32累加；每Ktile为四条M64N64K16 WGMMA。

## 五个对照

compute在主窗口前预取输入，主窗口仍有片上供给和MMA；transport只搬运及消费校验；serial依次搬运和计算；overlap预取后续tile并在消费者离开后复用槽；output在overlap上加显式输出处理。它们用同一measured kernel和共同资源请求。

四个物理slot占65536B，四个mbarrier另占32B；stage是活动槽数，不是不同的物理分配。实际static SMEM、寄存器与occupancy查询附在条件里，不能当实际驻留证明。

仅90个K≤32的pilot共同选择N=733；全部105个对照和检查点用同一N。K64未参与长度选择或拟合。短检查包含两个独立映射witness；正式trace=null，保留全部最终值和生命周期。

## 工作量与输出边界

令K为K-tile数，每个完整序列的重复次数为N。每Ktile的四条M64N64K16 WGMMA共计524288 FLOP；下表按一个完整序列计量，整个样本再乘N。

| 模式 | 计入主指标的FLOP | 主窗口GMEM输入 | 主窗口GMEM输出 |
|---|---:|---:|---:|
| compute | 524288K | 0 B（预取在窗口前） | 0 B |
| transport | 0 | 16384K B | 0 B |
| serial / overlap | 524288K | 16384K B | 0 B |
| output | 524288K+8192 | 16384K B | 16384 B |

output对64×64个FP32累加结果分别执行一次 `0.5*C+bias` FMA。每个输出元素计2 FLOP，额外工作为8192N FLOP，并写回16384N B。output的FLOP/cycle分子是MMA与这项SIMT输出FMA之和，不是纯WGMMA工作率；传输字节另外列出。

输出变换、写回和threadfence均在主clock64窗口内。其他模式最终C及诊断导出位于停止stamp之后；计时后导出量不加入上述GMEM输入/输出量。每个完整K序列开始时C清零，输入就绪、消费者离开、缓冲复用和最终排空按原事件链核对。

## stage=2、K=8的真实比较

| 模式 | cycles/sequence中位数 | 主工作率中位数 | 单位 | CV |
|---|---:|---:|---|---:|
| compute | 5052.314461 | 830.174771 | FLOP/clock64_cycle/CTA | 0.001% |
| transport | 11058.086630 | 11.8530451 | B/clock64_cycle/CTA | 0.010% |
| serial | 10773.497271 | 389.316848 | FLOP/clock64_cycle/CTA | 1.170% |
| overlap | 8357.082538 | 501.886152 | FLOP/clock64_cycle/CTA | 0.012% |
| output | 10887.878581 | 385.979324 | FLOP/clock64_cycle/CTA | 0.031% |

transport单位是B/cycle，计算对照是FLOP/cycle，二者主工作率不直接比大小。统一比较完整序列的cycles/sequence，本CTA周期只除以重复次数N，不除以N×K。

真实overlap_s2_k8样本：MMA=524288×8×733=3074424832 FLOP，输入=16384×8×733=96075776B。用6127674个本CTA周期作分母，得到501.727871 FLOP/cycle；除以N后是8359.71896 cycles/sequence。raw SHA `c567e73a0efbfd53ab066ce1ef523dd629dc3b4fafbcbb3921a01a425336645a`。

## 曲线

![overlap：完整K序列周期及stage对照](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s20-main-execution/actual-report-v1/overlap.png)

![output：输出处理后的完整K序列周期](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s20-main-execution/actual-report-v1/output.png)

误差条为保留样本min/max，不是置信区间。原图的pending C是生成时的历史标签；当前实际资格见独立C。其余三图见[完整报告](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s20-main-execution/actual-report-v1)。

## 完成事件与拟合

TMA expect_tx/arrive和acquire建立输入就绪，proxy/CTA发布后进入WGMMA fence/commit/wait0。消费者离开才允许下一tile复用，最终全部排空。output写及threadfence在主窗口内，其他最终C和诊断导出在停止stamp后。活跃mbarrier token只保全原值及uint64域，不要求等于固定数字。

| 模式 | stage | 截距cycles | 斜率cycles/Ktile | 最大拟合相对误差 | K64相对误差 |
|---|---:|---:|---:|---:|---:|
| compute | 1 | 204.415969 | 605.997728 | 0.003% | -0.000% |
| compute | 2 | 204.373569 | 605.997532 | 0.008% | 0.000% |
| compute | 4 | 204.425696 | 605.999714 | 0.007% | 0.000% |
| transport | 1 | 309.138072 | 1471.777903 | 0.289% | -0.067% |
| transport | 2 | 369.161189 | 1338.758399 | 3.765% | -0.068% |
| transport | 4 | 272.041732 | 1369.996870 | 7.473% | -0.212% |
| serial | 1 | 163.655162 | 1330.436293 | 1.962% | 0.054% |
| serial | 2 | 86.739033 | 1349.326409 | 5.790% | 0.093% |
| serial | 4 | 132.183184 | 1357.719978 | 2.259% | 0.109% |
| overlap | 1 | 285.179546 | 1122.529112 | 1.594% | -0.193% |
| overlap | 2 | 363.537962 | 1001.746834 | 4.557% | -0.088% |
| overlap | 4 | 261.502922 | 1032.656254 | 9.529% | -0.296% |
| output | 1 | 2767.804355 | 1127.307974 | 1.442% | 0.138% |
| output | 2 | 2849.277813 | 1003.940107 | 0.406% | 0.048% |
| output | 4 | 2750.881615 | 1034.748396 | 1.518% | -0.162% |

直线只拟合K=1/2/4/8/16/32，K64只检误差。截距不自动等同裸指令启动延迟，斜率依赖布局、资源、同步和输出边界。失配保留残差，不强行导出系数。逻辑运输不证明物理HBM字节，固定tile序列不等同完整GEMM。

[实际候选报告](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s20-main-execution/actual-report-v1/RESULTS.zh.md)、[全部样本](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s20-main-execution/actual-report-v1/samples.csv)、[配置统计](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s20-main-execution/actual-report-v1/results.csv)、[拟合与检查点](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s20-main-execution/actual-report-v1/sequence-fits.json)、[来源](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s20-main-execution/actual-report-v1/sources.json)。原档在local-collection-job734626/actual，原环境回执在native-replay-v1/complete734633。


独立实际C签录见 [review](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s20-main-execution/review/actual-data-and-report-C-review.json)。overlap、stage=4最大拟合误差9.53%，15组OLS均未取得物理服务系数资格。

## 脚本与复现

冻结源码、采样和审查入口见[S20脚本说明](../../../../../microbench/gh200_resource_campaign/families/s20/README.md)。原GPU734626及CPU734633均正常完成。原文件、输入输出、metadata、守卫和进程记录见[完整运行包](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s20-main-execution/local-collection-job734626/actual)，原环境回执见[receipt](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s20-main-execution/native-replay-v1/complete734633/native-replay-v1/receipt.json)。

```bash
python3 -B /完整运行目录/repo/audit_s20.py --run /完整运行目录
```

统计严格重算使用原Python3.9.21算法；r3离线入口已获准；本家族完整换目录重算和报告已与原native/已审报告精确一致，S22最终独立验收仍待完成。新分配必须采用自己的pilot，N=733只是原分配结果。105条条件观测已通过发布复核并纳入公共754条索引。

[最终发布状态](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s20-main-execution/published-v1/publication-final-state.json)绑定准确参数、实际C、发布复核及manifest；原prepared包的bridge_pending标签保留，不改原raw或候选。
