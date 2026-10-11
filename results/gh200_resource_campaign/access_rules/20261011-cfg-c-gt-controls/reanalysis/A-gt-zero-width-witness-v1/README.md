# 零宽GT同坐标的联合双点反例

740160开启记录与740161全部10个开启记录，共9504个GT三元组。完整枚举29040组同SM、不同CLOCK值的有序双点，发现55组反例：740160有9组，740161各进程依pair00至09分别有3、7、2、3、4、7、6、4、5、5组，共46组。全部是两个零宽GT盒落在同一个离散GT坐标；54组CLOCK差8 cycles，一组差12 cycles。没有`GT_before_earlier > GT_after_later`的严格反向盒。

这里只检查以下联合假设：

1. 每次中央CLOCK的真实采样时刻，落在以记录值为精确时刻的闭盒`[GT_before,GT_after]`内，额外GT误差e=0。
2. 同一GPU/SM内不回绕的CLOCK值，是该采样时间坐标上同一个单值非递减函数F的值。

对两个CLOCK值`C_a<C_b`，非递减且单值的F要求`t_a<t_b`。若盒同时给出`GT_a_before>=GT_b_after`，则`t_a>=GT_a_before>=GT_b_after>=t_b`，矛盾。这个证明允许计数器有平段和跳跃，不要求常数率、频率上限或可微性。

55组见证拒绝上述假设的合取。它们不判断GT对连续物理时间的实际误差、不识别读取乱序或CLOCK故障，也不提供一个必要的正误差预算。原e=0边际区间公式仍是条件代数；不能据此将离散记录的零宽GT盒直接当作可联合使用的精确物理时刻盒。既有接口的`joint_feasibility_proved=False`保持正确，无需修改公共接口或价格。

来源GPU为`GPU-ef8692f3-8bd7-80a2-608b-10731c9ec114`，binary为`c5476d9b00679859eef64bff34e87532e4936ea73f7f432c476597b8c36a6480`。固定cfg_c 2304×3072×4096、S4、cluster1×2、dyadic/seed17/sw1、16次同buffer关闭观察的前序。每个开启记录均完整覆盖132个不同SM，每个SM恰有一个记录CTA；枚举仅在同CTA/SM内，没有跨SM减CLOCK。

一个完整原始槽位见证来自[740160 flag1 raw](../../../20261011-cfg-c-gt-bracket-smoke-v2/returned/raw/flag1.jsonl.gz)，压缩文件SHA为`beac021143a115d8f7c3163df35b5a1d329f02d8cf0df295605028298a6c7af9`。CTA11、SM14、输出序号0、FIRST；下表索引均为原始call数组中从0开始的全局word索引。

| 中央CLOCK顺序 | WG | GT_before（shadow索引） | CLOCK（trace索引） | GT_after（trace索引） |
|---|---:|---|---|---|
| a | 1 | 1791725328316007232（103） | 1453442399967139（14912） | 1791725328316007232（14918） |
| b | 0 | 1791725328316007232（99） | 1453442399967147（14272） | 1791725328316007232（14278） |

两个盒都强制`t_a=t_b=1791725328316007232`，而CLOCK相差8；F在同一参数处不能同时取两个不同的值。其余54组的完整标签、索引、数值及raw SHA均保存在`analysis/witnesses.jsonl`，包括那组12-cycle见证，不挑选或替换记录。

`inputs.json`复用并核对前两批已绑定的raw、binary、构建审计与数值来源，共88项输入SHA；构建/数值证明沿用[740160审计](../../../20261011-cfg-c-gt-bracket-smoke-v2/reanalysis/A-gt-window-audit-v1/README.md)和[740161全部开启进程审计](../A-output-rate-interval-v1/README.md)，没有重新构建或重算全D。独立verifier直接从raw重新取点，以另一种有序枚举核对全部55组见证和逐进程计数。不同SM、不同GPU、相等CLOCK、有可用严格时间次序的盒四种负例均不构成该反例。两份完整分析文件和独立复验输出在新临时目录重放后逐字节相同。

```sh
task_base=/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules
task_audit="$task_base/20261011-cfg-c-gt-controls/reanalysis/A-gt-zero-width-witness-v1"
task_replay=$(mktemp -d)
python "$task_audit/r09_gt_order_witness.py" --base "$task_base" --inputs "$task_audit/inputs.json" --output "$task_replay/analysis"
python "$task_audit/verify.py" --base "$task_base" --inputs "$task_audit/inputs.json" --analysis "$task_replay/analysis"
```

本目录不扩展既有率诊断、不拟合e或价格、不重新评分、不新增GPU。740161观察与空对照失败、硬件包围与物理时间误差未知、新独立留出及跨卡未完成的状态保持；`output-sha256.json`绑定本目录产物。
