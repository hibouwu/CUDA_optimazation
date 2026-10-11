# guarded读数的联合误差边界

只分析740071已存的528个FIRST/MAIN三元戳，继承前一轮完整raw/源/二进制/模块/数值/SM身份复核。没有新增采样、拟合或修改原gate。这里量化读取精度假设与全卡共同周期演化的相容边界，仍不识别物理原因。

## 两个不同的容差变量

对同一SM三元戳(Cb,N,Ca)，允许真实周期位于[Cb-kappa,Ca+kappa]，取样时刻位于[N-epsilon,N+epsilon]；检验offset_sm+F(t)是否落入周期区间。每个SM原点独立未知，全卡共用任意连续非负F；上限分支另沿用旧r2声明1.98 cycle/ns。

epsilon单位ns，是额外单端点timer位移；kappa单位原生cycle，向每个已记录CLOCK区间两侧各扩张。它们不是测得的硬件误差、读开销或校正参数，也不是原128ns总窗口预算。不能未经独立换算把kappa改写成ns。

## 完整528戳的精确边界

固定epsilon后，全部图边界对kappa均为仿射函数。负闭环提供必要的有理数下界，在该下界重建图直至可行；最后一条下界与可行势共同证明精确最小值。时间节点此时固定，最小kappa可以取到，包括无速率上限分支；这与改变epsilon时可能不可取的无上限下确界不同。

| 额外timer单端点位移 epsilon (ns) | 无上限共同F：最小额外kappa (cycle) | 共同F≤1.98：最小额外kappa (cycle) |
|---:|---:|---:|
| 0 | 106 | 106 |
| 16 | 105 | 105 |
| 32 | 105 | 105 |
| 64 | 98.5 | 98.5 |
| 128 | 90.25 | 90.25 |
| 160 | 40.5 | 40.5 |
| 176 | 16 | 16 |
| 177 | 0 | 14.515 |
| 186 | 0 | 1.15 |
| 187 | 0 | 0 |

全部20个边界点保留精确分数、分离负环及可行势；46份证书、10560次原三元戳反演通过。verify.py重新构图，只校验证书和最优下界，不调用求解器。5个独立解析/平移/拒绝测试通过，相关R09测试共88项通过。

## 四端点就能核对epsilon=32的105cycle下界

使用SM39与SM115各自WG1_FIRST→WG2_MAIN的同SM计数区间。它跨原WG切点，仅用于计数器一致性，不能称为一个角色L、服务时间或价格。SM39的起点比SM115晚96ns，终点比SM115早224ns。因此在每端点±32ns下，SM39区间仍完全包含于SM115区间。共同非负F必须使内部区间的周期增量不超过外部区间。

原生CLOCK括号给出：

```text
SM39 内部区间：周期增量至少 66787 - 2*kappa
SM115外部区间：周期增量至多 66367 + 2*kappa
共同单调F要求：66787 - 2*kappa <= 66367 + 2*kappa
因此 kappa >= 105 cycles
```

完整528戳在kappa=105处又有可行证书，所以这一四端点下界在epsilon=32处是精确的。这里只比较各SM内部相减所得的周期增量界，没有跨SM相减原始clock。

还完整枚举了798个同SM计数区间，包含跨WG切点及同timer值的有效方向；找到2996个包含关系矛盾。诊断前沿在epsilon=0/16/32/64/128/160分别保存最强四端点证据。选择使用了已有观测，不能当未来关键CTA或留出选择；单个包含反例消失不代表整组可行，完整图边界另行保留。

## 仍不能确定的部分

现有读数至少与两类条件解释相容：各SM独立曲线在原括号、零额外timer位移下可行；全卡共同曲线在例如epsilon=0、kappa=106时也可行。没有独立的读取误差界，不能由数据选择SM物理时钟不同还是读取包围存在偏差，更不能把所需容差当实际误差测量。

要把条件反例升级为物理结论，缺的是独立建立的epsilon与kappa联合界，并证明它落在上述不可行区域；不能用目标端点拟合出的边界充当该证明。观察扰动和新独立留出要求也未解除。observation_qualified=false，原现象判读仍inconclusive。

```sh
python /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261011-R09-guarded-record-check/reanalysis/A-guarded-robustness-v1/analyze.py
python /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261011-R09-guarded-record-check/reanalysis/A-guarded-robustness-v1/verify.py
```

这项联合读取界用于时钟物理归因。经验时间预测继续按既定的自由预测、观察扰动、分项、整调用、新留出与跨卡要求评估。
