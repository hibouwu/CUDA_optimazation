# guarded单次记录：共同可变周期演化的条件一致性

只复核740071已存的guarded调用；继承A独立raw/身份/数值/原head复核。没有新采样。原常数率检验失败后，这里进一步放宽为全卡共同的任意连续非负演化F，各SM计数原点仍各自自由。

假设GLOBALTIMER硬件取样落在CLOCK-before/after内，则每个三元戳(Cb,N,Ca)要求某个t位于[N-epsilon,N+epsilon]，并有Cb≤offset_sm+F(t)≤Ca。周期括号已完整使用，epsilon是另外允许的单端点timer平移；它不是测得的误差，也不是128ns整个窗口预算。

令Y_sm为未知负原点。端点变为F(N-epsilon)-Y_sm≤Ca，Y_sm-F(N+epsilon)≤-Cb。与既有精确点图相比，只将上侧cycle约束从Cb放宽到Ca；保留单调性，并分别有/无旧r2声明1.98上限。所有原点先以整数去除，不跨SM相减clock64。

| 同一guarded调用，528个原FIRST/MAIN三元戳 | 最小可行整数额外单端点容差 | 实数边界 |
|---|---:|---|
| 全卡共同F，0≤F′≤1.98 | 187 ns | 可取最小值在(186,187] |
| 全卡共同非负F，无上限 | 177 ns | 下确界在[176,177]，可取性未证 |
| 每SM独立F，0≤F′≤1.98 | 全132个SM均0 ns可行 | 条件存在见证 |

共同模型的epsilon=0及相邻整数下界均有严格负闭环；最小整数有可行势。保存6份共同图证书、132份独立局部图证书，共138份；1584个原括号的可行性验证通过。verify.py独立重建图，只检查证书与连续曲线交集，不调用求解器。零宽括号严格退化为旧精确点图；另用解析50ns/放宽括号后0ns的算例验证端点边变更。

该结果说明：在同一读取包围假设下，只把常数率改成全卡任意可变F仍不足以无额外容差解释这些记录；每SM独立的观测条件见证可以做到0。不能据此认定SM物理时钟独立，也不能把177/187ns叫timer精度或归因于读指令。读取包围假设、SM局部计数与观察/历史影响尚未区分。局部曲线由目标读数构造，不是预测参数或SM服务价。

仍只有每臂一次冒烟，扰动分布与空间资格未取得；original_phenomenon_inference=inconclusive，observation_qualified=false。原128ns窗口预算、cfg_a控制失败及新留出要求不变。

```sh
python /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261011-R09-guarded-record-check/reanalysis/A-guarded-common-curve-v1/audit.py
python /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261011-R09-guarded-record-check/reanalysis/A-guarded-common-curve-v1/verify.py
```
