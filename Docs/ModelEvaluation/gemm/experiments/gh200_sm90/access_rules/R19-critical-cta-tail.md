# R19：cfg_b实际工作分配与关键CTA尾部

8个条件、160个通过数值与稳定性检查的进程已采样并独立复算，655,360个输出值正确；另保留4个30次预热后仍未收敛的尝试，不计入正式参数。每条件10组plain/trace，所有plain CV≤2.8%，trace扰动≤4.9%。同R18的job737122、GPU-099dda56，CUDA12.9.41、CUTLASS3.9.2、sm_90a、NDEBUG。

## 配置与计时

cfg_b固定128×128×64 pingpong、cluster1×1、6 stage，FP16输入、FP32累加和输出。两个consumer warpgroup交替接手输出tile；SASS为16条静态HGMMA、无spill和C7510。只改变四个形状与swizzle=1/8，不改寄存器配额、stage或epilogue。

完整时间为预热后CUDA event包围的一次GEMM；分配、填充、结果检查在外。trace逐CTA保存入口globaltimer/clock64、首末SM、每个实际tile坐标、主循环、输出许可、输出完成和最终退出。跨SM比较入口/退出用globaltimer，同一CTA的阶段长度用clock64。

## 完整服务与实际工作

| 形状M×N×K | swizzle | plain μs | 实际每CTA tile数 | 最大CTA周期中位数 | 中位CTA周期中位数 |
|---|---:|---:|---|---:|---:|
| 1408×1408×1024 | 1 | 12.416 | 1 tile×121 CTA | 16290.0 | 15407.0 |
| 1408×1408×1024 | 8 | 19.424 | 2 tile×124 CTA, 1 tile×8 CTA | 28253.0 | 26066.2 |
| 1408×1408×8192 | 1 | 47.456 | 1 tile×121 CTA | 74041.5 | 73503.0 |
| 1408×1408×8192 | 8 | 115.520 | 2 tile×124 CTA, 1 tile×8 CTA | 188566.5 | 161986.0 |
| 4096×4096×1024 | 1 | 51.008 | 8 tile×100 CTA, 7 tile×32 CTA | 81289.5 | 79757.0 |
| 4096×4096×1024 | 8 | 51.120 | 8 tile×100 CTA, 7 tile×32 CTA | 80830.0 | 78814.8 |
| 4096×4096×8192 | 1 | 364.624 | 8 tile×100 CTA, 7 tile×32 CTA | 570085.0 | 543078.5 |
| 4096×4096×8192 | 8 | 344.912 | 8 tile×100 CTA, 7 tile×32 CTA | 536073.0 | 534733.2 |

近单波形状1408²只有11×11=121个逻辑tile。swizzle1实际121个CTA各做1个；swizzle8补到16×16=256个物理tile，132个CTA中124个做2个、8个做1个。手算`124×2+8×1=256`。所以长K从47.456变成115.520 μs，不能先归因为“swizzle影响缓存命中”；工作量和波次已经改变。

多波4096²为32×32=1024个tile，两种swizzle都分成100个CTA做8个、32个做7个，`100×8+32×7=1024`。短K为51.008/51.120 μs，差0.22%，小于各自约1%的进程波动；没有稳定收益。长K swizzle8比1快`1−344.912/364.624=5.41%`，此时才是在相同tile数分布下比较条件服务。

![完整时间、CTA时长和入口分散](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R19-job737122-v1/reanalysis/local-independent/r19-service.png)

误差条为独立进程标准差。三个面板的时间边界与单位不同，不直接相减。

## 尾部怎样进入模型

多波长K、swizzle1中，最大CTA周期中位数570,085，中位CTA周期中位数543,078.5，相差约5.0%；swizzle8对应536,073/534,733.25，差约0.25%。不能仅用中位CTA服务估完整时间，也不能给所有规模加同一个5%尾部。

近单波长K、swizzle8的最大/中位CTA时长比约1.164，远高于swizzle1的1.007；边界补齐、两个tile的分工与服务共同变化，没有证据把差异全归为入口偏斜。所有条件的入口分散中位数只有96–128 ns，且最终最晚CTA的入口偏移中位数32–64 ns。完整包络取`max(end_ns)−min(entry_ns)`；最长clock64窗口与最晚退出CTA分别保存，二者不能强行视为同一CTA。

实际阶段序列为：consumer交替的FIRST_MMA→MAIN_END、EPI_PERMIT→EPI_DONE；后一tile主循环可与前一tile输出重叠。`tails.json`逐进程列关键CTA工作列表、每tile主循环/输出/许可窗口、交接、SM分布和入口偏斜。相邻主循环间隔可能含重叠，不能把负的差分改成零后再声称测得物理交接延迟。

当前可交付的是绑定规模、swizzle和真实tile分布的尾部条件。V07使用新形状验证事件递推和关键CTA预测；本组没有验证适用于任意尺寸的统一尾部系数，也不把不同卡或调用后频率拼成周期规则。

## 证据与复现

[归档](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R19-job737122-v1/)、[条件与检查结果](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R19-job737122-v1/reanalysis/local-independent/rules.json)、[逐进程关键CTA](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R19-job737122-v1/reanalysis/local-independent/tails.json)。四个预热失败的原始stdout和记录全部保留，分析中的`rejected_processes`说明预热未收敛的尝试为何未完成数值检查；有效慢样本未删除。

```bash
python3 microbench/gh200_resource_campaign/access_rules/run_r19.py prepare --output <新目录> --cutlass-root <CUTLASS3.9.2>
python3 <目录>/source/run_r19.py build --output <目录>
python3 <目录>/source/run_r19.py setup --output <目录>
python3 <目录>/source/run_r19.py sample --procs 10 --output <目录>
python3 microbench/gh200_resource_campaign/access_rules/analyze_r18.py --input <归档> --output <新分析目录>
```

历史v1的setup误用了R18同grid检查，修正运行器在`reanalysis/`；编译源码和二进制未改。新入口只给R18的普通/整块越界/有效零配对检查相同grid，R19允许swizzle改变补齐工作。预热失败最多重试两次，按独立文件保存，不覆盖旧尝试。

<a id="other-critical-cta"></a>

## 其他实验中的关键 CTA 与入口数据

- [V07 事后诊断](V07-rule-validation.md#事后诊断总时间为何接近离线不改判定)：cfg_a 边界条件中，最慢 CTA 不是边界 tile 所在的 CTA。
- [V08](V08-wider-validation.md#留出结果)：冻结所选关键 CTA 与实测最慢 CTA 的周期差，36 例绝对误差中位 1.11%、最大 20.11%。
- V06、V07、V08 与 R18 的打点逐 CTA 记录入口 globaltimer 与 SMID（`probes/r18_trace.hpp`），包括 cluster 2×1 的 cfg_a 和 cluster 1×2 的 cfg_c。本页只测 cfg_b、cluster 1×1，其入口数据可用于排查 CTA 派发，不足以识别多 CTA cluster 的放置规律；SMID 的数值区间也不能直接当作 GPC 编号。
