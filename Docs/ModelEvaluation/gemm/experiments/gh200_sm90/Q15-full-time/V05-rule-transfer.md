# V05：规则迁移验证

18个留出案例已完成四组观测，共1440个独立进程。冻结预测的整体迁移**未通过**：合格观测支持主循环和关键CTA周期的近似，但首tile供给等待存在系统性失配，另有输出和交接失配。实验实现、保存数值和计量解析均通过独立检查。

## 验证条件

FP16输入，FP32累加与输出，α=1、β=0；GH200、CUDA12.9、`sm_90a`、CUTLASS3.9.2、`-DNDEBUG`。cfg_a/b/c各验证以下六个尺寸：

| M×N×K | 检验内容 |
|---|---|
| 1280×1536×1536 | 不满一波 |
| 2304×4096×256 | 短K、多波 |
| 4096×5120×1024 | 中等K、多波 |
| 5120×7168×3072 | 较大输出 |
| 1024×3584×12288 | 长K外推 |
| 2696×3336×1544 | M/N/K尾部；lda=1600、ldb=3392、ldd=3360 |

历史检查排除了97个已使用尺寸，以上18个组合没有冲突。每次进程检查4096个输出位置，覆盖边界和尾块，并核对物理padding。实际grid和stage在预测前由编译后的CUTLASS实例查询；cfg_a/b为6 stage，cfg_c为4 stage。全部测量使用GPU-54896349、romeo-a053，作业735985。

## 如何组合规则

| 参数 | 同次观测边界 | 用于预测的条件 |
|---|---|---|
| S，初始供给等待 | 首个producer取得工作→该tile最早首MMA | R14同配置的首tile观察；包括程序路径和等待 |
| L，主循环 | 首MMA→最终MMA等待返回 | 合格K点间插值；校准域外显式外推 |
| G，交接 | 本tile主循环结束→下一tile首MMA | 可为负，表示主循环重叠 |
| W，末尾许可等待 | 最后输出tile主循环结束→epilogue许可 | 只取producer调度序号最大的tile |
| E，末尾输出 | 最后tile epilogue许可→最终源尾等待返回 | 源可复用端点；不包含此后完整全局写回 |
| C，关键CTA工作区间 | 首producer取得工作→最终输出源释放 | 比较各CTA本地跨度，不跨SM相减clock64 |

同一末tile先合并消费组：main_end和epilogue许可都取最大时间戳；`W=max(epi_permit)−max(main_end)`，`E=唯一issuer的source_release−max(epi_permit)`。不把单个消费组较早的许可当作整个CTA的许可。

冻结的候选公式为：

`C = S + T×L + (T−1)×G + W + E`

T由实际静态scheduler的最大CTA工作量确定。公式采用典型tile周期和交接间隙，包含同质tile、末块主导和4CTA校准向整卡迁移的假设。中间tile的cooperative源释放可能到下一tile acquire才被观察，不能把那段延迟再作为末尾E相加。

cfg_a的E/W只有Ktile=64合格，使用明确的单点常量假设；其他K标为外推。cfg_c的主循环使用R10对齐条件的Ktile=32锚点和基准斜率，其他服务采用基准代理。相同输入/输出字节数没有证明配置等价。R13的供给/退役和R15的输出组合提供条件对照，整段成本没有重复加进公式。

## 一个具体计算

cfg_a、4096×5120×1024共有1280个输出tile，grid为132个CTA，最长CTA处理T=10个tile；每tile有16个Ktile。

R14的主循环点为L(8)=4392、L(64)=33065，因此：

`L(16) = 4392 + (16−8)×(33065−4392)/(64−8) = 8488.143 cycle`

同样得到S=1250.571、G=2899；末尾单点假设给W=108、E=3052。于是：

`C = 1250.571 + 10×8488.143 + 9×2899 + 108 + 3052 = 115383 cycle`

合格留出观测的C为120259.5 cycle，误差4.05%；但S实测3112.5 cycle，误差59.82%。总区间接近不能替代供给阶段正确。冻结的微秒预测为70.240 µs，独立plain观测中位数77.264 µs；微秒单列评分。

## 独立检查与预测结果

72个“案例×观测组”中51个观测满足打点扰动与稳定性要求，可用于定量比较。50对trace/plain扰动超过5%，对应的21个观测不作定量判定；全部进程的预热、CV、原始身份、数值和schema检查通过。百分比阶段要求误差中位数≤10%、最大≤20%；小于512 cycle的阶段按事件顺序及重叠符号判读。

| 阶段 | 可定量比较案例 | 可用数据中位误差 | 最大误差 | 未判定 | 结论 |
|---|---:|---:|---:|---:|---|
| 主循环L | 13 | 0.86% | 5.51% | 5 | 可用数据子集符合，完整阶段未判定 |
| 交接G | 11 | 2.76%¹ | 27.83%¹ | 3 | 已证实失配；另4例无交接，N/A |
| 末尾输出E | 12 | 5.04% | 21.07% | 6 | 已证实失配 |
| 末尾许可W | 12 | 小阶段关系判读 | — | 6 | 可用数据子集事件关系符合 |
| 初始供给S | 14 | 48.62% | 61.38% | 4 | 14例均超20%，系统性失配 |
| 关键CTA区间C | 12 | 2.94% | 12.72% | 6 | 可用数据子集符合，完整阶段未判定 |

¹交接百分比只统计7个绝对值≥512 cycle的案例；其余4个小阶段检查重叠符号。

- 供给：4CTA校准的初始供给等待不能直接迁移到整卡启动条件，14个合格案例均失配。此区间包含程序路径和等待，尚不能把差值全归为纯TMA延迟。
- 交接：cfg_b的中等K、较大输出分别预测正间隙1045.64/160.21 cycle，实测为−328/−281.75 cycle，重叠关系预测错误；cfg_c尾部案例误差27.83%。
- 输出：cfg_a长K误差21.07%，cfg_c中等K误差20.20%；常量输出或基准代理不能视为已证明的通用规则。
- 时钟：较大输出案例的有效SM时钟低于校准值。全18例微秒误差中位10.73%、最大31.86%；测后时钟只用于诊断，没有回填预测。本轮未采powercap遥测，不能断言降频原因。

![各阶段误差与可用观测范围](../../../../../../results/gh200_resource_campaign/access_rules/20261007-V05-job735985-v3/analysis-C-v1/cycle-errors.png)

完整逐例判定、时钟图和CSV见[独立报告](../../../../../../results/gh200_resource_campaign/access_rules/20261007-V05-job735985-v3/analysis-C-v1/report.md)。

## 事后诊断：微秒误差来自频率换算

冻结的周期不变，只把换算频率从冻结值（cfg_a/b/c 为 1764/1741/1812 MHz）换成每例实测的有效频率，18 例微秒误差从中位数 10.7%、最大 31.9% 降到 4.1%、12.0%。三个大输出案例的实测频率只有 1178–1473 MHz，误差从 −20%~−32% 变为 −2%~0%。

这是测后诊断，不是预测，不改变 V05 的判定。它说明 V05 的微秒误差主要来自换算时用了单一校准频率，没有使用 [V03](V03-clock-rule.md) 按负载变化的频率规则；[V06](V06-revised-transfer.md) 按此修正后，频率误差 ≤4.6%。数据：[`posthoc-measured-clock.json`](../../../../../../results/gh200_resource_campaign/access_rules/20261007-V05-job735985-v3/posthoc-measured-clock.json)。

## 冻结与复现

[只读预测](../../../../../../results/gh200_resource_campaign/access_rules/20261007-V05-job735985-v3/predictions.json)的SHA256为`e1ff0d15b91a7ce7deb74d6d40b81014a3355c0673ec5403026e890cc73d0f6d`。冻结时间早于首次采样，来源、源码、二进制、实际工作分配和参数条件都绑定哈希；当时由独立复核者完成的冻结前检查记录在[reviews](../../../../../../results/gh200_resource_campaign/access_rules/20261007-V05-job735985-v3/reviews/C-prediction-admission.md)。所有预测保持原值。

入口为`microbench/gh200_resource_campaign/access_rules/`中的`run_v05.py`、`v05_calibrate.py`、`v05_predict.py`、`analyze_v05.py`。prepare/build/setup完成资源与工作分配查询；当时的calibrate只生成候选参数，经冻结前检查后由freeze生成只读预测。sample对main/output/supply/critical四组分别运行，每组每例10个独立plain/trace配对。输入预测放在单独的只读文件路径，运行目录保存其相同SHA的副本。

在项目根目录执行，历史重算直接使用本地归档，不需要GPU：

```bash
V05_RUN="results/gh200_resource_campaign/access_rules/20261007-V05-job735985-v3"
python "$V05_RUN/source/analyze_v05.py" --input "$V05_RUN" --output "$V05_RUN/new-offline-review"
python "$V05_RUN/reviews/C-analysis-source-v1/analyze_observer.py" --run "$V05_RUN" \
  --frozen-summary "$V05_RUN/analysis-frozen-v1/summary.json" \
  --output "$V05_RUN/new-independent-review"
```

输出目录必须是新目录。独立分析保留原冻结评分，再按各观测组的打点检查结果，分别报告可用数据的误差与未判定案例。

当时提出的整卡初始供给模型与显式交接递推，后续已在[V06](V06-revised-transfer.md)实现并重新验证。若再次校准，需要新的留出验证；本批V05不再用于回拟合后宣布通过。
