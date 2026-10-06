# EXP-19：GH200 FP32 固定 tile 组合实测

实际数据与报告 C 已通过：105 个条件工作率、1050 个正式样本完成独立核验。发布桥接已通过，条件观测已纳入公共索引。实验比较固定 32×32×32 tile 在一个 CTA、128 线程下的五种执行组织。

## 五种对照测什么

| 对照 | 主窗口内的工作 | 用途 |
|---|---|---|
| compute | 输入预取在窗口前；窗口中保留SMEM消费与FMA | 观察给定片上供给下的计算序列服务 |
| transport | 复制、等待和消费校验；不做矩阵FMA | 观察同一循环组织的运输服务 |
| serial | 每tile复制完成后计算 | 建立顺序执行对照 |
| overlap | 预取后续tile并在完成当前消费后复用槽 | 观察供给和计算的组合 |
| output | overlap再加显式输出处理 | 观察输出处理带来的完成时间变化 |

所有模式预留相同四物理槽，动态SMEM请求32768B；stage表示活动缓冲数，不代表每个模式实际分配不同容量。资源上限由实际API查询记录，不能从stage数推定驻留。

## 公共重复次数和检查点

90个K≤32配置的pilot共同选择N=305。全部105个正式对照采用相同N，K=64没有参与长度选择或拟合。每个进程的C在每次完整K序列开头清零；transport校验和则跨重复按uint32累加。

## 工作量与输出边界

令K为K-tile数，每个完整序列的重复次数为N。下表是每个序列的主窗口工作量；整个样本再乘N。

| 模式 | 计入主指标的FLOP | 主窗口GMEM输入 | 主窗口GMEM输出 |
|---|---:|---:|---:|
| compute | 65536K | 0 B（预取在窗口前） | 0 B |
| transport | 0 | 8192K B | 0 B |
| serial / overlap | 65536K | 8192K B | 0 B |
| output | 65536K+2048 | 8192K B | 4096 B |

output对32×32个累加结果分别执行一次 `0.5*C+bias` FMA，bias由列号给出。每个输出元素计2 FLOP，因此额外工作为2048N FLOP，并在主窗口写回4096N B。表中的output工作率分子是主FMA与这项输出FMA之和，不是只统计GEMM主乘加。

output的转换、写回及threadfence在clock64停止stamp之前；其他模式的最终C及诊断导出在停止stamp之后，不加入主工作量。输入与SMEM缓冲初始化、资源预留和正确性回读位于窗口外；每个完整K序列开始时的累加器清零处于主窗口内。

## 一个真实的K=8、stage=2比较

| 模式 | cycles/sequence中位数 | 主工作率中位数 | 单位 | CV |
|---|---:|---:|---|---:|
| compute | 28534.836066 | 18.3736118 | FLOP/clock64_cycle/CTA | 0.000% |
| transport | 6620.052459 | 9.89961944 | B/clock64_cycle/CTA | 0.040% |
| serial | 32331.004918 | 16.216261 | FLOP/clock64_cycle/CTA | 0.062% |
| overlap | 29785.140984 | 17.602334 | FLOP/clock64_cycle/CTA | 0.000% |
| output | 30583.931148 | 17.2095601 | FLOP/clock64_cycle/CTA | 0.007% |

比较五种对照时先看cycles/sequence：一个sequence是完整Ktile序列，分母是N，不是N×K。transport的主值为B/cycle，计算模式为FLOP/cycle，因此它们的工作率不能直接作大小比较。

真实单样本 overlap_s2_k8：FMA工作量=65536×8×305=159907840 FLOP；输入逻辑量=8192×8×305=19988480B。除以本CTA的9084453个clock64周期，得17.6023631 FLOP/cycle；周期除以305，得29785.0918 cycles/sequence。raw SHA `c1d6801fb9428373ae54735c116e2bcf0745e02f0e4e067389fd304640b709c4`。

## 曲线与拟合

![overlap：完整K序列周期及stage对照](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s19-main-execution/actual-report-v1/overlap.png)

![transport：完整K序列周期及stage对照](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s19-main-execution/actual-report-v1/transport.png)

误差条为保留样本min/max，不是置信区间；K64是检查点。原图的pending C标题是报告生成时的历史标签，当前数据资格见独立C签录。compute、serial、output图见[完整报告目录](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s19-main-execution/actual-report-v1)。

## 怎样使用拟合

各mode/stage分别用K=1/2/4/8/16/32的cycles/sequence中位数作直线拟合，K64只计算预测误差。截距是这组序列的经验拟合项，不自动等同某一条指令的启动延迟。斜率同样依赖输入、资源、同步和完成边界。

| 模式 | stage | 截距cycles | 斜率cycles/Ktile | 最大拟合相对误差 | K64相对误差 |
|---|---:|---:|---:|---:|---:|
| compute | 1 | 150.840111 | 3547.999823 | 0.000% | -0.000% |
| compute | 2 | 150.840111 | 3547.999823 | 0.000% | -0.000% |
| compute | 4 | 150.840111 | 3547.999823 | 0.000% | -0.000% |
| transport | 1 | 191.311272 | 926.343449 | 0.558% | -0.131% |
| transport | 2 | 286.976397 | 792.259053 | 2.404% | -0.020% |
| transport | 4 | 446.898181 | 760.260699 | 8.603% | 0.455% |
| serial | 1 | 149.314061 | 4022.327336 | 0.073% | -0.000% |
| serial | 2 | 151.447207 | 4022.044737 | 0.139% | -0.007% |
| serial | 4 | 148.153275 | 4022.187953 | 0.062% | -0.005% |
| overlap | 1 | 186.015586 | 3769.172989 | 0.031% | -0.035% |
| overlap | 2 | 310.561675 | 3682.806720 | 1.157% | 0.014% |
| overlap | 4 | 455.529231 | 3647.044861 | 3.739% | 0.093% |
| output | 1 | 974.536571 | 3771.420743 | 0.946% | 0.025% |
| output | 2 | 1072.093859 | 3684.392935 | 1.169% | 0.042% |
| output | 4 | 1214.581005 | 3649.079118 | 3.486% | 0.130% |

没有可靠线性关系时应保留曲线和残差，不用一个系数掩盖变化。候选文件仍qualified=false，拟合项也禁止自动导出为服务参数。

## 原始结果与复现入口

[真实候选报告](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s19-main-execution/actual-report-v1/RESULTS.zh.md)、[逐进程样本](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s19-main-execution/actual-report-v1/samples.csv)、[全部配置统计](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s19-main-execution/actual-report-v1/results.csv)、[拟合与K64误差](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s19-main-execution/actual-report-v1/sequence-fits.json)、[来源](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s19-main-execution/actual-report-v1/sources.json)。冻结源码与原始归档见[完整运行包](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s19-main-execution/local-collection-job734622/actual)，原环境回执见[原环境重算](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s19-main-execution/native-replay-v1/complete734627)。实际 C 签录见 [独立审查](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s19-main-execution/review/actual-data-and-report-C-review.json)。参数发布桥接已通过，原候选不改写。


## 脚本与复现

冻结源码、采样及审查入口见[S19脚本说明](../../../../../microbench/gh200_resource_campaign/families/s19/README.md)。原GPU734622及原环境CPU734627均正常结束。对完整原归档使用：

```bash
python3 -B /完整运行目录/repo/audit_s19.py --run /完整运行目录
```

原统计使用Python3.9.21；r3离线入口已获准；本家族完整换目录重算和报告已与原native/已审报告精确一致，S22最终独立验收仍待完成。正式采样必须采用新的合法分配和自己的pilot，不能把上述原N=305当作其他设备的校准结果。105条条件观测已通过发布复核并纳入公共754条索引。

[最终发布状态](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s19-main-execution/published-v1/publication-final-state.json)绑定准确参数、实际C、发布复核及manifest；原prepared包的bridge_pending标签保留，不改原raw或候选。
