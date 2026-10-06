# GH200 TMA bulk 双向服务：24 个正式配置

24 个配置各完成10次独立进程采样，独立C、原ARM封存审查及换目录严格重放均通过。下面记录的速率用于本实验条件下的搬运服务建模；全精度参数、原始来源和离线复现见文末。

## 测什么

采用无tensor-map的1D bulk，分别测GMEM→SMEM和SMEM→GMEM。每CTA有128线程、一个stage，每次循环发出一个payload并等待完成；payload为1/4/8/16/32/64KiB，分别测单CTA和全GPU。

GMEM→SMEM的完成边界为mbarrier acquire与CTA消费门槛；SMEM→GMEM为bulk wait-group 0与CTA门槛。窗口包含请求、等待和CTA控制，不能解释成裸TMA引擎服务率。

| 范围 | 分子与分母 | 单位 |
|---|---|---|
| 单CTA | Q/C；C为同一物理SM的clock64周期差 | B_transport/clock64_cycle/CTA |
| 全GPU | Q/T；T为所有CTA的globaltimer完成包络 | GB_transport/s/GPU |

Q=B×N×q，B为实际blocks、N为冻结循环次数、q为payload字节；每请求计一次有效运输量，不再乘128线程。字节/ns在数值上等于十进制GB/s。单CTA结果不能乘SM数替代全GPU测量。

## 配置与接受条件

GH200/CC9.0、132SM，job733392/romeo-a041，UUID `GPU-ec947ba1-3e15-9860-163d-330d6c66d1b4`，CUDA12.9/sm_90a。动态SMEM为payload+32B控制区；每CTA循环使用32个全局槽。全GPU网格为132×min(4,occupancy)，occupancy只是API上界。实际寄存器、资源、工作集及每点N保留在参数文件。

本分配的24pilot与24smoke确定N。正式每点10个独立进程，8–30窗口有界预热，末5窗口CV≤2%，正式CV≤5%，最多3批且保留全部样本。本次24点均第一批稳定，最大CV为1.228465%，没有挑选最好批次。

## 用原始记录手算

单CTA的 `gmem_to_smem_64kib_one_cta`，batch0/trial0：B=1、N=15396、q=65536B，所以Q=1,008,992,256B。C=31,960,436cycle，因此Q/C=31.5700404087 B_transport/cycle/CTA。它的raw SHA为 `9994cfd93a477c9a9ea5b7357304c0bbfb0137eb2fb35c4bf7bceefebceb1b25`。

全GPU的 `gmem_to_smem_16kib_all_gpu`，batch0/trial0：B=528、N=7062、q=16384B，Q=61,091,610,624B，包络T=15,928,640ns，所以Q/T=3835.33124134 GB_transport/s/GPU。raw SHA为 `21e402ed2ebfec6d3d926460ec7649ecc595e9931bb3d5e8228202f6964ad89f`。

这两个数是单次样本，下面列十次样本的中位数。

## 结果表与图

|方向|payload KiB|范围|实际N|blocks|中位数|样本标准差|CV %|
|---|---:|---|---:|---:|---:|---:|---:|
|GMEM→SMEM|1|单CTA|42643|1|1.8129909|0.00384812|0.212487|
|GMEM→SMEM|1|全GPU|38910|528|1770.3772|2.28754|0.129163|
|GMEM→SMEM|4|单CTA|33057|1|6.2684564|0.00624759|0.099629|
|GMEM→SMEM|4|全GPU|20833|528|2941.4827|2.29664|0.0780741|
|GMEM→SMEM|8|单CTA|35714|1|10.772746|0.0255722|0.237557|
|GMEM→SMEM|8|全GPU|12738|528|3766.1441|2.47344|0.065673|
|GMEM→SMEM|16|单CTA|27586|1|17.387621|0.0230062|0.132315|
|GMEM→SMEM|16|全GPU|7062|528|3831.006|4.95342|0.129314|
|GMEM→SMEM|32|单CTA|22471|1|25.502097|0.0988379|0.388113|
|GMEM→SMEM|32|全GPU|3852|528|3796.7939|3.87777|0.102132|
|GMEM→SMEM|64|单CTA|15396|1|31.564865|0.14894|0.471844|
|GMEM→SMEM|64|全GPU|2563|396|3770.7924|7.69159|0.203952|
|SMEM→GMEM|1|单CTA|64000|1|3.778574|0|0|
|SMEM→GMEM|1|全GPU|47169|528|2535.6828|8.97892|0.353745|
|SMEM→GMEM|4|单CTA|60422|1|9.3302578|0|0|
|SMEM→GMEM|4|全GPU|17985|528|2607.4928|26.8064|1.02486|
|SMEM→GMEM|8|单CTA|45454|1|12.355913|0|0|
|SMEM→GMEM|8|全GPU|10090|528|2708.3838|33.3288|1.22847|
|SMEM→GMEM|16|单CTA|29112|1|14.747025|0|0|
|SMEM→GMEM|16|全GPU|5313|528|2674.1858|29.0564|1.09007|
|SMEM→GMEM|32|单CTA|17543|1|16.326805|0|0|
|SMEM→GMEM|32|全GPU|2816|528|2689.8941|14.3954|0.534313|
|SMEM→GMEM|64|单CTA|9191|1|17.250789|2.329e-07|1.35008e-06|
|SMEM→GMEM|64|全GPU|1858|396|2738.6571|16.755|0.612734|

单CTA中位数与标准差的单位为B_transport/clock64_cycle/CTA；全GPU为GB_transport/s/GPU。CV为样本标准差除以均值。误差条使用全部样本的标准差（n−1），不表示置信区间。少数点的CV=0仅说明有限记录具有相同速率。

![GMEM到SMEM单CTA](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/parallel/tma_cluster/analysis/s14-formal-733392-qualified/one_cta-gmem_to_smem.png)

![SMEM到GMEM单CTA](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/parallel/tma_cluster/analysis/s14-formal-733392-qualified/one_cta-smem_to_gmem.png)

![GMEM到SMEM全GPU](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/parallel/tma_cluster/analysis/s14-formal-733392-qualified/all_gpu-gmem_to_smem.png)

![SMEM到GMEM全GPU](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/parallel/tma_cluster/analysis/s14-formal-733392-qualified/all_gpu-smem_to_gmem.png)

GMEM→SMEM单CTA中位数随payload从1KiB到64KiB由约1.813增至31.565B/cycle。全GPU在8–64KiB约3.77–3.83TB_transport/s；SMEM→GMEM在4–64KiB约2.61–2.74TB_transport/s。payload同时改变工作集和部分资源上界，不能把方向差异归因于单一瓶颈。

## 适用范围与复现

参数只适用于记录中的设备、工具链、payload、N、网格、资源、32-slot布局和完成边界。逻辑运输量不是物理HBM流量；本实验没有证明冷缓存、缓存命中、HBM物理峰值、张量映射TMA、多stage并发或完整GEMM。其他路径由各自实验测量。

- [24条合格条件参数](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/parallel/tma_cluster/analysis/s14-formal-733392-qualified/parameter-candidates.json)与[完整CSV](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/parallel/tma_cluster/analysis/s14-formal-733392-qualified/cases.csv)：原文件名保留，参数内部资格为 `qualified_conditional_services_C_verified`，不再作为未审候选使用。
- [准确复现命令](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/parallel/tma_cluster/analysis/s14-formal-733392-qualified/REPRODUCE.md)：包含完整运行归档和小型报告源码附件的解包及离线重放入口，不需要聊天记录。
- [交付来源](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/parallel/tma_cluster/analysis/s14-formal-733392-qualified/delivery-manifest.json)及[完整独立后审](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/parallel/tma_cluster/reviews/C-733392/S14-qualified-delivery-post-review-r2.json)。

原raw、summary、measurement manifest、C与COMPLETE未改。本地派生浮点重算采用既有≤2ULP白名单与精确Fraction门槛，保留严格失败记录；原ARM严格审查实际通过。
