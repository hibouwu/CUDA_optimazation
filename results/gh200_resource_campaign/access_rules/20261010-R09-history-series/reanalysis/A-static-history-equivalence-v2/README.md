# R09：静态归一化的历史预测为何在本批退化

`E_previous × T0(current)/T0(previous)`在本批严格等于已有`previous_call`，因为批准的V09 r2递推与时钟消费的静态输入逐项相同。没有调用`predict_components`或`supply_predict`，也没有改写GPU UUID来绕过跨卡拒绝。

输入为原history-series与mirror-history的80进程：60个完整35调用series、20个single目标；single的16个prefix event从既有A-history-series读取并绑定同raw SHA。验证了80份原raw、2120个有目标/历史记录的调用、554400个series角色工作计数；1160个目标预测与已有CSV中的previous_call逐值相同。

- 批次配置：cfg_b，6144×8192×32768，stage6，tile128×128×64，cluster1×1，132CTA；dyadic seed17，aligned输入。
- sw1/sw8的3072个tile坐标有3056个不同，每CTA tile数量保持：36CTA×24、96CTA×23。
- 10个case/swizzle签名完全相同，含每CTA每输出的供给X、j/T/Kt/compute、有效输出比例、首轮输出字节、零乘积比例、schedule与Q/S；没有只比较Q/S就跳过递推。
- X分布（冻结定义）：2904项[4224,0,0,0,0,0,0]；132项[4188,0,0,0,0,0,0]；36项[1138.5,0,0,0,0,0,0]。不重定义源量、不使用物理带宽解释。
- 消费的静态输入相同、参数相同，故相同设备支持条件下的T0相同。这里没有声称访问顺序不影响真实硬件成本。
- actual history为GPU-ef8692f3，冻结供给为GPU-43269fbc，8个case的card_supported均false。结构相等不使跨卡自由预测变为受支持。
- 负例ldb+16和K减半均改变签名。

首sw8切换single/series各10个，原末值误差最大39.0855%/40.6200%。在全部20个首切换上事后选择共同倍率r，令q=previous_event/actual_event，则最小化max|r*q−1|的解为`r=2/(q_min+q_max)`，误差下界为`(q_max−q_min)/(q_max+q_min)`。本批r=0.7821810155951385，最大9.990275334161314%，中位绝对4.856491673644221%，RMS5.621922687365551%。这些是读取目标后的开发oracle；略低于10%/5%，不是“不可能通过”证明，也不是已验证系数。当前静态模型无法自行产生该倍率。

批准预测SHA为f4df39e00cf1d2ab61947c911e699bd5bea4e533609d93f726de43077515d81e；批准清单SHA为09da85d124eb697a95c4695bd05281b07c37877890d0eca18afbdd2cf213f2ab。清单通过source_hashes.json间接绑定源码，calibration.json另在批准清单中绑定。实际导入的冻结Python源逐一核对；不把当前共享模块当成旧源码。原始数值正确性沿用已核验回放，本轮不重新声称2120次调用都有逐调用数值检查。

`equivalence.json`保存182项输入SHA、全部分层event指标、签名身份、支持范围、精确有理数下界与负例。`static-signature.json.gz`保存完整静态消费输入，`event-identity.json.gz`保存全部1160预测。`verify.py`只检查原文件哈希、逐值预测与有理数边界，不导入公共模型或调用提取函数；182项SHA、1160项预测与20项下界计算均通过。静态消费集合的完整性仍依赖对冻结v09_model.py `predict_components`的源码检查。

复放（新输出目录）：

```sh
r09_static_output=$(mktemp -d /tmp/r09-static-history-XXXXXX)
python /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261010-R09-history-series/reanalysis/A-static-history-equivalence-v2/audit.py --base /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules --output "$r09_static_output"
```

直接复验归档：

```sh
python /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261010-R09-history-series/reanalysis/A-static-history-equivalence-v2/verify.py
```

v1原始文件保留，v2补充批准清单的间接源码绑定与全部预测对旧CSV的精确一致性。没有GPU操作、公共代码修改、新拟合、时间候选采纳、独立留出或跨卡资格。
