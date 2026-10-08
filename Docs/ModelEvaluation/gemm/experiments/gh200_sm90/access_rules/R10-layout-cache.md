# R10：B 行距

9点完整GEMM时间已测得；cfg_a/c使用v2全网格trace，cfg_b使用v4局部trace。本轮条件观测已完成核对；当时计划的预测迁移检验已记录在[V05](V05-rule-transfer.md)。本页只讨论固定行距对照。

## 结果

固定M×N×K=2600×3000×2000，FP16输入、FP32累加与输出，α=1、β=0。只改变B的ldb；A的lda=2000，D的ldd=3000。设备为romeo-a053的GPU-54896349-d69d-9358-b526-433454c04733，job735876，CUDA12.9、sm_90a、CUTLASS3.9.2。

| 配置 | tile | schedule | cluster | stage |
|---|---|---|---|---:|
| cfg_a | 128×128×64 | cooperative | 2×1 | 6（沿用V04自动选择） |
| cfg_b | 128×128×64 | pingpong | 1×1 | 6（沿用V04自动选择） |
| cfg_c | 256×128×64 | cooperative | 1×2 | 4 |

下表完整时间均来自[v2全9点plain](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v2/analysis-formal/summary.json)，每点10个独立进程；三个行距分别为6000/6016/6144B。

| 配置 | ldb | 完整时间中位µs | CV | 同档全网格trace周期/Ktile | trace判定 |
|---|---:|---:|---:|---:|---|
| cfg_a | 3000 | 66.400 | 0.588% | 630.234 | 通过 |
| cfg_a | 3008 | 62.944 | 0.559% | 606.500 | 通过 |
| cfg_a | 3072 | 63.328 | 0.777% | 615.906 | 通过 |
| cfg_b | 3000 | 76.272 | 0.601% | 816.562 | 通过 |
| cfg_b | 3008 | 58.272 | 1.223% | 不采用 | 超5% |
| cfg_b | 3072 | 57.136 | 1.748% | 不采用 | 超5% |
| cfg_c | 3000 | 54.592 | 0.820% | 1031.594 | 通过 |
| cfg_c | 3008 | 54.544 | 0.888% | 1031.500 | 通过 |
| cfg_c | 3072 | 54.016 | 1.049% | 1031.500 | 通过 |

cfg_b的6000B行距明显较慢，6016/6144B恢复大部分差异；cfg_a差异较小，cfg_c主循环周期基本相同。因此规则应按配置分别使用：这组结果支持cfg_b的行起点对齐解释，不能推广为所有配置统一需要128B行距对齐，也不提供物理cache/TMA/HBM流量归因。

cfg_b的最终合格周期来自[v4局部补测](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/analysis-formal/summary.json)及[CSV](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/analysis-formal/cases.csv)。只采线性CTA0–3、consumer0的last tile，每行距10对进程；下表两个周期列分别保留池化与按进程聚合口径。

| ldb | plain中位µs | plain CV | trace中位µs | 最大绝对配对扰动 | 局部池化中位cycle/Ktile | 先按进程中位再汇总cycle/Ktile |
|---:|---:|---:|---:|---:|---:|---:|
| 3000 | 76.160 | 0.401% | 76.448 | 2.068% | 854.531 | 869.906 |
| 3008 | 57.904 | 0.977% | 58.160 | 2.913% | 671.359 | 671.492 |
| 3072 | 57.456 | 0.895% | 57.568 | 3.775% | 659.047 | 662.812 |

v4三行距均通过每对绝对5%门槛，局部主循环和同档完整时间方向一致。v2完整时间与v4完整时间分别比较，局部周期不与旧全网格周期池化；v4不能替代整个grid的分布。

## 最小算例

B的物理地址为`B[k*ldb+n]`；ldb3008增加8个FP16 padding元素，使行距从6000B变为6016B。需要检查的B padding为`2000×8=16000`个元素；ldb3072为`2000×72=144000`，ldb3000明确为0。

例如v2 cfg_b在同一测量协议内从ldb3000的76.272µs变为ldb3008的58.272µs，完整时间减少18.000µs；这个差值不单独命名为TMA等待。Ktile数为ceil(2000/64)=32，末个Ktile只有16个有效K元素；局部周期指标将实际trace窗口除以32，保留循环与排空成本。单一K长度不能识别固定开销与每Ktile斜率。

## 时间、抽样与数值边界

完整服务时间取plain CUDA event，覆盖一次warm GEMM及输出完成。trace起点位于首个full-barrier等待返回后、首个MMA前，终点为mma_tail返回后，最终WGMMA等待和输入stage释放已执行。它包含此后输入等待、循环和排空，排除了起点之前的首次输入等待；不减旧模型c0，不作为裸MMA周期。clock64只在同CTA/同SM内相减。

v2 trace覆盖全grid两消费者的last tile；cooperative两组共同处理同tile，tile计数只用group0，pingpong两组交替处理不同tile。v4仅保留CTA0–3、consumer0，每进程4个last-tile观测，word4是该消费者实际完成tile数。未采CTA/WG为缺观测，不能当作0cycle或推断整个grid完成tile数。

v2 plain没有trace scratch准备；v3/v4的plain/trace都分配65536B scratch，初始化及预热后按相同顺序memset、device同步，再执行计时调用。三个版本分别归档和统计。每进程预热8–30次、末5次CV≤2%，trace/plain相邻配对、顺序随机；负向扰动超过5%同样不合格。

初始化与CUTLASS descriptor均使用真实stride，B视图为(N,K,L)，StrideB=(1,ldb,0)。所有B padding填65504（FP16位模式0x7bff），计时外检查完整padding；本矩阵A/D无padding。每进程保存4096个period17 dyadic见证的数值坐标与实际FP32值，覆盖首尾和64/128/256边界。v2/v3/v4共300进程、1228800个保存值通过独立逐K整数点积重算，误差0；每个GEMM工作量31200000000 FLOP，预热、padding和统计也已复核。该见证不声明任意输入误差标准。

## 来源与检查索引

测量源码、二进制和原始数值均保留。v3/v4 CUTLASS依赖按各自冻结SHA256匹配后由v2副本补齐，各838份；不是替换测量源码。资源与SASS保存在各build中。C为本组实现者，以下公式复算使用作者的另一种参考实现，不等同于由未参与实现者完成的复核；作者复算及对应hash见[v4 reviews](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reviews/independent-C.md)与[hash清单](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reviews/independent-C-hashes.json)。

未实施R10的ROOT已另行完成[独立复核](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reviews/ROOT-independent.md)与[核对证据清单](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/reviews/ROOT-independent.json)：300个正式进程、1228800个保存值直接逐K参考通过，FLOP、padding、预热、HGMMA计数及零spill核对通过。本文的结果判定依据这份未参与实现者的复核。

| 来源 | 范围与检查结果 | 保留原因 |
|---|---|---|
| [v1代表](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v1/analysis-representatives/summary.json) | cfg_b的ldb3000/3072，40进程 | 行距差可辨，决定扩大9点；cfg_b整段16个静态HGMMA为prologue8+steady8，与V04一致 |
| [v2正式](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v2/analysis-formal/summary.json) | 9点180进程，plain全部保留；7点trace满足扰动要求 | cfg_b3008的两对−5.927%/−6.108%、3072的一对−5.341%超门槛，不采用这两点trace |
| [v3对称准备](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v3/analysis-formal/summary.json) | cfg_b三行距60进程 | plain中位76.256/57.840/57.584µs；最大trace扰动1.607%/4.538%/6.202%，3072仍失败 |
| [v4局部trace](../../../../../../results/gh200_resource_campaign/access_rules/20261007-R10-job735876-v4/analysis-formal/summary.json) | cfg_b三行距60进程，三组通过 | 对称准备，原事件位置不变，仅限制CTA/WG；最终周期只在此抽样域使用 |
| 本地编译诊断`/tmp/gh200-gaps-local/R10/` | 未运行GPU | GCC16超出CUDA13支持范围，override标准库语法失败，GCC15缺cc1plus；正式测量由CUDA12.9完成 |
| 首tile/不同起点候选 | 未进入GPU构建，不作为规则 | 后续恢复原事件边界并先修正准备对称性；最终采用v4局部last-tile方案 |

对齐结论限于本页配置、2600×3000×2000和三种实际行距。未扩做缓存准备、其他补齐矩阵；v4不提供全卡周期分布。这些条件观测在[V05](V05-rule-transfer.md)中接受了迁移检验；分项误差独立评分，不能从完整时间调整。

## 复现

在主对话调度的GH200作业内加载CUDA12.9，新RUN目录必须不存在。当前入口对cfg_b使用v4局部trace；旧档必须使用各自source中的冻结分析脚本。

```bash
python3 microbench/gh200_resource_campaign/access_rules/run_r10.py cpu-check
python3 microbench/gh200_resource_campaign/access_rules/run_r10.py list
python3 microbench/gh200_resource_campaign/access_rules/run_r10.py build \
  --cutlass-root /PATH/TO/CUTLASS3.9.2 --output RUN
python3 microbench/gh200_resource_campaign/access_rules/run_r10.py sample --output RUN
```

sample结束自动写新`analysis-formal/`；不要在同一已有分析目录重复写入。若只复现cfg_b三行距，build加`--config cfg_b`后仍用sample，不加仅含两个行距的`--representatives`。CPU准备检查不运行GPU；已保存测量的独立复核证据见上表reviews。
