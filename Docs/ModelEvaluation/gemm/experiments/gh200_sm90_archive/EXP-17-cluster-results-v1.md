# EXP-17：GH200 DSM、cluster同步与TMA多播

固定128线程/CTA，比较cluster大小2/4/8、单cluster和API查询得到的cluster grid，共42个条件、420次正式独立进程。实际数据与报告C、发布桥接均已通过；42条条件观测已纳入公共754条索引。

## 测量对象与完成边界

| 对照 | 主计工作量 | 完成边界与解释 |
|---|---|---|
| local_read / dsm_read | CTA数×N×4096 B | 本地或远端SMEM访问，包含消费和循环控制 |
| local_write / dsm_write | CTA数×N×4096 B写请求 | 窗口内含必要读回；读回量单独列，不加入主写请求量 |
| cluster_sync | cluster数×N个phase | 所有参与者完成约定阶段；不是裸barrier延迟 |
| bulk_single_target / bulk_all_targets | cluster数×N×16384 B源请求 | 单目标或所有目标输入完成，接收总量另列 |

两种执行范围均以globaltimer的完成包络计时，不跨SM相减clock64。所有42个配置首批稳定，每项10个独立进程，最大CV约0.88%。各配置采用自身pilot选择的N，引用结果时保留完整条件。

## 本地访问与DSM对照

以下为单cluster的主请求工作率中位数，单位GB_transport/s/one_cluster。

| 形式 | C=2 | C=4 | C=8 |
|---|---:|---:|---:|
| local_read | 29.803716 | 58.981536 | 117.111235 |
| dsm_read | 15.407832 | 30.075264 | 60.260930 |
| local_write | 17.863388 | 35.548793 | 70.216109 |
| dsm_write | 14.092054 | 27.194745 | 53.986853 |

本地和远端对照都包含相应消费、读回或控制。不能把它们解释成某个端口的裸带宽，也不能把写实验与不含读回的纯store实验直接相比。

## TMA单目标与多播

下表使用同一源请求口径，仍为单cluster的GB_transport/s。所有目标多播时接收总量为源请求量×C，因此源请求速率与接收速率回答不同问题。

| 形式 | C=2 | C=4 | C=8 |
|---|---:|---:|---:|
| bulk_single_target | 15.401719 | 15.355635 | 15.454269 |
| bulk_all_targets | 13.728059 | 13.727366 | 13.698318 |

## cluster同步

主指标为cluster_phase/ns；它是完整同步阶段的吞吐。cluster_grid的分子汇总了多个cluster，不能取其倒数就称为单个barrier延迟。

| C | 单cluster：phase/ns | cluster_grid：phase/ns | grid中的cluster数 |
|---:|---:|---:|---:|
| 2 | 0.005217320 | 2.314466578 | 528 |
| 4 | 0.005162877 | 1.071490170 | 248 |
| 8 | 0.005125223 | 0.512479755 | 124 |

## 图与全部配置

![单cluster；中位数与保留样本min/max](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s17-main-execution/actual-report-v1/one_cluster.png)

![查询所得cluster grid；中位数与保留样本min/max](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s17-main-execution/actual-report-v1/cluster_grid.png)

图的误差条是样本min/max，不是置信区间。全部42个条件、样本数、CV与资源条件见[结果表](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s17-main-execution/actual-report-v1/results.csv)和[原候选](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s17-main-execution/actual-report-v1/parameter-candidates.json)。原候选的qualified=false保留；发布桥接已通过，原候选仍保留冻结标签；当前实际数据资格见[C签录](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s17-main-execution/review/actual-data-and-report-C-review.json)。

## 用一个真实样本手算

`bulk_all_targets_c2_cluster_grid`的一次正式样本有528个cluster，N=5173，每次源请求16384 B：

- 源请求量：528×5173×16384 = 44750340096 B。
- 全部接收方总量：44750340096×2 = 89500680192 B。
- globaltimer完成包络：17663840 ns。
- 主工作率：44750340096÷17663840 = 2533.44347 B/ns，数值等于GB/s。

raw SHA为 `39bae86745b8cc466d2dd0438b4f8781bbfdba35154a66d8d9a8c99adffc60d0`。分子使用源请求量，主计时后导出17301504 B不计入这项运输工作量。

## 适用范围和复现

cluster grid来自实际能力及occupancy查询，是配置条件，不是全部SM覆盖或实际驻留证明。CTA必须按约定存活到远端访问完成；cluster大小、布局、循环、完成与缓冲生命周期均随值引用。计数器权限阻塞，不根据这些逻辑字节推断物理HBM流量或硬件峰值。

GPU作业734618和原环境CPU重算734623均正常结束。原始归档见[完整运行包](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s17-main-execution/local-collection-job734618/actual)，原环境结果见[重算回执](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s17-main-execution/native-replay-v1/complete734623/native-replay-v1/receipt.json)。脚本与完整运行包说明见[S17源码入口](../../../../../microbench/gh200_resource_campaign/families/s17/README.md)。

离线重算使用完整运行包内的冻结版本：

```bash
python3 -B /完整运行目录/repo/audit_s17.py --run /完整运行目录
```

原统计使用Python3.9.21；新版解释器可能存在浮点末位差异。统一换目录重放入口已准备，完整执行和S22验收仍待完成。报告和两张图可以从已审CSV重建，入口见发布桥接包中的render_published_plots.py。

[最终发布状态](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s17-main-execution/published-v1/publication-final-state.json)绑定准确参数、实际C、发布复核及manifest；原prepared包的bridge_pending标签保留，不改原raw或候选。
