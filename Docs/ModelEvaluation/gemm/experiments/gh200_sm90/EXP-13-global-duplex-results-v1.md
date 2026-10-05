# S13 全局读写联合服务实测

状态：正式采样和独立于runner工作量函数的原始字段重算已完成，待独立C审查。18个配置各10次独立进程，共180个正式样本，全部运行器终态为stable；本文不授参数导出资格。

## 实验问题与参数

在同一GH200上，改变逻辑读写需求比例、读指令修饰符和相对L2容量的工作集，观察全GPU的请求服务速率。单独读写、依赖复制和独立读写是不同数据流；后续组合模型需要条件匹配的服务参数。

主指标为逻辑读字节与写字节之和，除以全GPU globaltimer包络，单位GB/s。B/ns在十进制下数值等于GB/s。它包含目标循环的地址推进、校验累加及约定的排空，不是单条load/store裸服务速率，也不是物理HBM流量。

## 一个最小例子

原始路径：`batches/independent_r2_w1_large/batch_00/trial_00/attempt_00/raw.jsonl`。

每 active array=253034496 B；16轮，读写比2:1。

读请求=16×253034496×2=8097103872 B；写请求=16×253034496×1=4048551936 B。

包络时间=1790880456298326496−1790880456294944320=3382176 ns。

总Q=读请求+写请求=8097103872+4048551936=12145655808 B。

总逻辑请求速率=12145655808/3382176=3591.07740342 GB/s（约分值126517248/35231）。采用指定配置的第一条记录，不挑选最快样本。逻辑请求量不能作为物理HBM流量。

## 配置矩阵与条件

固定256线程、16轮、单GPU；实际启动528个CTA并覆盖132个SM。线程读取非均匀地址函数，独立写使用另一地址函数；依赖复制将读取值写回目标。每次启动前按冻结host规则初始化数据和poison。

设备L2容量为62914560 B。small每个active array请求L2/4，large请求4×L2，再向blocks×threads×16 B对齐。实际每array分别为17,301,504 B和253,034,496 B；同时读写使用两份数组。small的两数组总量仍小于L2容量，large的每份数组均大于4×L2。大小关系只定义条件，不能证明命中、冷缓存或物理HBM流量。

读/写比采用1:0、0:1、1:1、2:1、4:1、1:2、1:4。读单测包括.ca与.cg；1:1同时保留依赖复制与独立读写。没有展开全部比例、线程数、循环长度或工作集的笛卡尔积。

GPU UUID为`GPU-43269fbc-449d-3e0f-908a-9c81229546d3`，CUDA12.9.41、sm_90a，实际device和工具链完整身份保存在run归档。正式作业731011，完成状态0:0。

## 计时、计量与正确性

设对齐后的每active array字节数为A，固定循环长度I=16，读写比例为r:w，则Q_read=I×A×r，Q_write=I×A×w。复制计读和写两侧请求；运输量的单次计数约定不能替代此联合请求量约定。

时间为max(CTA stop_ns)−min(CTA start_ns)，禁止跨SM相减clock64。CUDA event用于完成与包络交叉检查，主速率使用globaltimer。局部clock64字段保留在raw，不转换成全GPU参考周期或FLOP/cycle。

读验证检查每个非空线程checksum，写验证检查完整目标数组。纯写的读checksum不作为有效数值检查。长循环正确性记录是host的error_count_only证据；正式raw没有保存全读值/全写值，不能据它替换数值参考后重新逐值验证。全部短验证和真实SASS证据在B3归档，条件、源码和目标身份已冻结。

180个进程均预热收敛，各记录预热窗口范围8–10次；这不证明长期热稳态。正式统计使用全部10个样本，没有筛选最快批次。各配置只需第一批，最大正式CV为1.601%，低于5%接受阈值。报告用有理数重新计算原始Q/时间及CV²阈值，代码没有调用runner的工作量计算函数。

## 实测与图表

| 模式 | small 均值 GB/s | small CV | large 均值 GB/s | large CV |
|---|---:|---:|---:|---:|
| read .ca | 13465.119 | 1.601% | 3522.298 | 0.214% |
| read .cg | 8293.456 | 0.442% | 3524.314 | 0.168% |
| write .wb | 4530.698 | 0.457% | 3790.410 | 0.290% |
| dependent copy 1:1 | 7787.326 | 0.492% | 3434.130 | 0.075% |
| independent 1:1 | 7747.593 | 0.876% | 3427.757 | 0.082% |
| independent 2:1 | 10786.547 | 0.275% | 3591.401 | 0.211% |
| independent 4:1 | 9032.781 | 0.398% | 3671.015 | 0.045% |
| independent 1:2 | 6399.459 | 0.218% | 3386.995 | 0.136% |
| independent 1:4 | 5520.822 | 0.240% | 3327.379 | 0.278% |

均值和CV按配置分别计算；图表误差条是独立进程样本标准差，不是置信区间。表中stable描述原协议的统计判定，不等于已经通过C审查。

![small逻辑请求速率](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/global-duplex-formal-b3-v1-r2/small.svg)

![large逻辑请求速率](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/global-duplex-formal-b3-v1-r2/large.svg)

small的.ca读与.cg读结果有明显差别；large两者接近。缺少计数器证据时，这个对照只能说明已记录条件下的服务差异，不能直接量化缓存命中率。依赖复制与独立1:1是各自的结果，不能按名称合并。独立读写在比例变化时的联合速率也不同，不能用读单测和写单测速率简单相加。

## 来源、适用范围与复现

原始run：`results/gh200_resource_campaign/20261001-resource-suite-v2/global_duplex/formal-b3-v1/`。结构化逐进程表、逐批表、配置表、手算字段和输入哈希在[分析目录](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/global-duplex-formal-b3-v1-r2/provenance.json)。

完整采样归档正在传回；为避免重复传输冻结依赖导致等待，已经用原提交包的精确文件加955份新run文件恢复本地run。全部10629个snapshot文件和11573个measurement清单文件逐SHA匹配，见[收集记录](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s13-formal-exact-delta/collection.json)。这个过程不重新生成raw、计时或统计；完整原包仍会另行交叉核对。

本地Python与原ROMEO Python版本不同；原严格重放结论保留，已审有界浮点重放通过。最终C与封存仍需独立审查和原运行环境严格finalize。NCU权限缺口保留，本实验不导出物理HBM字节、缓存命中率或动态指令计数。

```bash
python3 -B microbench/gh200_resource_campaign/audit_replay.py results/gh200_resource_campaign/20261001-resource-suite-v2/global_duplex/formal-b3-v1
python3 -B microbench/gh200_resource_campaign/report_global_duplex_v1.py results/gh200_resource_campaign/20261001-resource-suite-v2/global_duplex/formal-b3-v1 --replay results/gh200_resource_campaign/20261001-resource-suite-v2/post_audits/bounded-float-replay-v1-r2/global_duplex-formal-b3-v1.json --out /tmp/gh200-global-duplex-replay-new
```

输出目录必须不存在且在run之外。报告只读归档，不能用于启动GPU。正式测量仍通过已审run_suite和Slurm入口执行；恢复必须使用原run ID及冻结条件。
