# R09：固定首切换倍率的跨批诊断与同批反例

上一轮在20个series/single首sw8目标上事后得到的共同倍率保持为`7115963128213099743/9097591205021478403`，即0.7821810155951385。这里不重拟合，检验它乘紧前已完成event的结果。只对已知sw1→sw8切换乘该倍率；sw1→sw1及sw8→sw8保留紧前event。当前目标的W/f及角色checkpoint不进入预测。

旧1160个目标全部保留，并加入更早739552共同/adaptive前序批次的全部40目标。数据此前均已用于开发诊断，不能称为新独立冻结留出。该历史规则仍只是一项失败的开发候选，没有接入公共模型。

| 739552条件，各10目标 | 中位绝对误差，% | 最大绝对误差，% | RMS，% |
|---|---:|---:|---:|
| adaptive sw1，倍率1 | 2.2916 | 4.5379 | 2.7220 |
| adaptive sw8，倍率1 | 0.5549 | 3.4938 | 1.6336 |
| common16 sw1，倍率1 | 1.0589 | 3.0582 | 1.4540 |
| common16 sw8，固定倍率0.782181 | 4.9682 | 11.0101 | 6.8392 |

不能只保留改善的首切换：旧same/mirror和sw1等全部1160项也按同一规则保存，原有失败没有被删除。角色分项不在本轮重新解释为物理服务或时钟率。

更强的同批反例不依赖固定倍率来自哪里。对common16 sw8的10个进程，令q=最后前序event/目标event。

| 见证 | 最后前序event，µs | 目标event，µs | q |
|---|---:|---:|---:|
| trial9，最小q | 6352.064133 | 5528.223991 | 1.1490243781 |
| trial5，最大q | 7295.519829 | 5140.448093 | 1.4192381086 |

预测r×前序event时，10%要求同时满足`r≥0.9/q_min=0.7832731987…`和`r≤1.1/q_max=0.7750637425…`，无可行r。该10进程的最小最大相对误差严格为

`(q_max−q_min)/(q_max+q_min) = 1279801964400078195/12163953950764594318 = 0.10521266107881255`。

最优r约0.7787366013，只是用于证明下界的oracle，不替代原固定倍率。全部30个首切换的极值恰好也由这两个739552进程给出；先前20个拟合目标的下界仍为9.990275%。这是本批观测上的精确经验界，不是总体置信界、硬件噪声极限或对所有历史模型的不可能性证明。它仅排除一个共同正倍率乘末次前序event的齐次规则。

两批均为GPU-ef8692f3、cfg_b、6144×8192×32768、stage6、132CTA、dyadic seed17。common批次每个目标之前执行16次同buffer、sw1调用。完整device SASS文件的实际SHA均为`cd2061d651dba08e52d633c902726aae3972638f2264b399d84e0af186d1c25e`；主机binary不同，不将跨批残差归因于唯一机制。同批10目标反例不需要假定两批主机协议等价。

`audit.py`核验40份原raw SHA和对应host记录、逐次warmup/target event一致性、原4096值检查通过标记；本轮不重新计算4096值数值参考。源码证明timed返回event后才执行后续目标。完整旧1160评分的来源与上一轮签名包绑定，不重跑此前已完成的raw数值复核。`transfer.json`保存96项输入SHA、全部case分层与失败身份，`event-scores.json.gz`保存1200项有理数评分。`bound.py`单独生成三组精确下界。`verify.py`不导入audit或公共模型，独立复算96项输入哈希、1200项评分和三组上下界及极值见证，全部通过。

新目录复放（把原脚本放在同一新目录后再执行bound和verify）：

```sh
r09_transfer_output=$(mktemp -d /tmp/r09-switch-transfer-XXXXXX)
cp /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261010-R09-common-history/reanalysis/A-switch-multiplier-transfer-v1/{audit,bound,verify}.py "$r09_transfer_output/"
python "$r09_transfer_output/audit.py" --base /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules --output "$r09_transfer_output"
python "$r09_transfer_output/bound.py"
python "$r09_transfer_output/verify.py"
```

直接复验归档：

```sh
python /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261010-R09-common-history/reanalysis/A-switch-multiplier-transfer-v1/verify.py
```

输入契约仍保留精确kernel/观测版本、前序、设备与event边界。未知前序、cfg_a/c、plain转换和跨卡均未由此建立支持。没有GPU操作、公共代码修改、源工作或价格变化、新留出资格。
