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

## V08 cfg_a/c 入口与最后完成者的离线复核（2026-10-09）

**最长本地周期 CTA 经常不属于最晚退出集合；入口偏斜不足以解释所有差异。** 对 V08 的 46 个 cfg_a 和 32 个 cfg_c 条件，各读取十次 stamped、十次 ends；原始记录通过现有数值检查。cooperative 的最终端点采用 role2/thread256 在 post-loop `store_tail()` 后记录的 `final_cycle/final_ns`，不把 `EPI_DONE` 当作最终退出，也不把最终退出称为所有 global 写完成。

分别求 `argmax(final_cycle−entry_cycle)` 与 `argmax(final_ns)` 的**并列集合**，只有两个集合无交集才记为不一致。下表是所有成功调用的直接端点描述，包含原 trace 扰动超限条件，不用它们拟合周期参数。stamped 与 ends 独立统计，不能按 trial 编号拼成同次执行。

| 配置 / 记录方式 | 调用数 | 两个集合无交集 | 入口分散中位数 / 最大值 | 最长周期集合中最晚者，距实际最晚退出的最大差 |
|---|---:|---:|---:|---:|
| cfg_a / stamped | 460 | 151 | 128 / 160 ns | 3744 ns |
| cfg_c / stamped | 320 | 153 | 128 / 160 ns | 6816 ns |
| cfg_a / ends | 460 | 153 | 128 / 160 ns | 5088 ns |
| cfg_c / ends | 320 | 164 | 128 / 160 ns | 4064 ns |

只看 V08 原 heldout 的 stamped 调用，cfg_a/c 分别为 48/120、68/120 次不一致。这是新定义下的事后 CTA 身份检查，不替换 V08 原“冻结所选 CTA 与最大本地周期 CTA”的评分。所有这些调用中，每个活动 CTA 都在不同 SM 上，入口与退出 SMID 一致；这不能推导 GPC 的编号或 cluster 放置规律。

一个可直接手算的例子是 `cfg_c_h06/stamped-02`。globaltimer 均减去本次调用的最早入口：

| CTA / SM | 入口偏移 ns | 本地周期长度 | 退出偏移 ns |
|---|---:|---:|---:|
| 6 / 4，最长本地周期 | 32 | 1,010,139 | 619,040 |
| 12 / 16，最晚退出 | 32 | 1,009,845 | 625,856 |
| 78 / 17，最晚退出并列 | 64 | 1,009,817 | 625,856 |

前两者入口相同，而退出差 `625856−619040=6816 ns`。其完整窗口平均 cycles/ns 为 1.631867/1.613625；单一中位频率无法精确转换每个 CTA 的内部事件。cfg_a_c5_k1024/stamped-02 也有同入口 +96 ns 的反例：最长 CTA76 在 +119296 ns 退出，CTA112/113 在 +123040 ns 退出，差 3744 ns。记录能证明两种排序不同，不能分辨其硬件时钟、执行状态或计时行为的来源。

因此调度归因至少保留每 CTA 的入口、SMID、本地周期长度和 globaltimer 退出，内核包络直接取 `max(final_ns)−min(entry_ns)`。已观测端点足够判定最后完成者，无需为这一结论补测；预测中的每 CTA 周期仍需适当的周期到时间关系才能确定最后完成者，不能仅给最大周期加统一入口修正。

输出并发另见 [R15 的同调用估计及其限制](R15-output-service.md#v08-输出窗口离线复核2026-10-09)，补齐路径与工作量分开见 [R18](R18-cluster-boundary.md#按轮次与-cta-工作量重算2026-10-09)。这三项事后分析没有改变模型公共接口或 V08 预测。

[逐进程端点及并列集合](../../../../../../results/gh200_resource_campaign/access_rules/20261008-V08-job737322-v1/reanalysis/B-20261009-output-issuer-v2/critical.json)还保存每 CTA 相对入口/退出与本地周期，复核时无需从进程间中位数反推一个不存在的调用：

```bash
python3 microbench/gh200_resource_campaign/access_rules/analyze_r15.py --v08-output \
  --input /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261008-V08-job737322-v1 \
  --output <该run下新的reanalysis目录>
```
