# EXP-16：GH200 TMA 并发与缓冲实测

36个名义坐标，32个合法条件、320次正式独立进程已通过实际C r2；4个S4/R4容量超限项保留零target launch。发布桥接已通过，32条条件观测已纳入公共754条。

工作量为 B×N×R×16384 B；S 改变并发和缓冲资源，不直接乘入字节数。单 CTA 用自己的 clock64 差，全 GPU 用 globaltimer 完成包络。

## one_cta

| 方向 | S | R | 状态 | 中位数 | 单位 | CV | 样本 |
|---|---:|---:|---|---:|---|---:|---:|
| gmem_to_smem | 1 | 1 | stable | 14.7486825 | B_transport/clock64_cycle/CTA | 0.067% | 10 |
| smem_to_gmem | 1 | 1 | stable | 9.21344063 | B_transport/clock64_cycle/CTA | 0.117% | 10 |
| gmem_to_smem | 1 | 2 | stable | 21.3837068 | B_transport/clock64_cycle/CTA | 0.321% | 10 |
| smem_to_gmem | 1 | 2 | stable | 12.5229582 | B_transport/clock64_cycle/CTA | 0.041% | 10 |
| gmem_to_smem | 1 | 4 | stable | 28.5124106 | B_transport/clock64_cycle/CTA | 0.366% | 10 |
| smem_to_gmem | 1 | 4 | stable | 14.6549502 | B_transport/clock64_cycle/CTA | 0.550% | 10 |
| gmem_to_smem | 2 | 1 | stable | 21.0857626 | B_transport/clock64_cycle/CTA | 0.000% | 10 |
| smem_to_gmem | 2 | 1 | stable | 14.7621049 | B_transport/clock64_cycle/CTA | 0.267% | 10 |
| gmem_to_smem | 2 | 2 | stable | 39.8122962 | B_transport/clock64_cycle/CTA | 0.000% | 10 |
| smem_to_gmem | 2 | 2 | stable | 18.1630513 | B_transport/clock64_cycle/CTA | 0.000% | 10 |
| gmem_to_smem | 2 | 4 | stable | 42.334426 | B_transport/clock64_cycle/CTA | 0.000% | 10 |
| smem_to_gmem | 2 | 4 | stable | 18.223883 | B_transport/clock64_cycle/CTA | 0.000% | 10 |
| gmem_to_smem | 4 | 1 | stable | 19.6974002 | B_transport/clock64_cycle/CTA | 0.001% | 10 |
| smem_to_gmem | 4 | 1 | stable | 14.913393 | B_transport/clock64_cycle/CTA | 2.453% | 10 |
| gmem_to_smem | 4 | 2 | stable | 37.3740148 | B_transport/clock64_cycle/CTA | 0.000% | 10 |
| smem_to_gmem | 4 | 2 | stable | 18.1528167 | B_transport/clock64_cycle/CTA | 0.000% | 10 |
| gmem_to_smem | 4 | 4 | capacity_reject | 无有效统计 | B_transport/clock64_cycle/CTA | — | 0 |
| smem_to_gmem | 4 | 4 | capacity_reject | 无有效统计 | B_transport/clock64_cycle/CTA | — | 0 |

真实算例 `smem_to_gmem_16kib_s1_r1_one_cta`：1×16096×1×16384=263716864 B；除以 28623131 得到 9.21341778 B_transport/clock64_cycle/CTA。计时后导出的 16384 B 不计入主量。raw SHA `9cfa273e76fe7cf3f63e86bc7e76e39a744b7e1e48e5b883e73864aa5ac1d0c1`。

## all_gpu

| 方向 | S | R | 状态 | 中位数 | 单位 | CV | 样本 |
|---|---:|---:|---|---:|---|---:|---:|
| gmem_to_smem | 1 | 1 | stable | 3835.32244 | GB_transport/s/GPU | 0.068% | 10 |
| smem_to_gmem | 1 | 1 | stable | 2629.2228 | GB_transport/s/GPU | 0.176% | 10 |
| gmem_to_smem | 1 | 2 | stable | 3706.00485 | GB_transport/s/GPU | 0.662% | 10 |
| smem_to_gmem | 1 | 2 | stable | 2685.4482 | GB_transport/s/GPU | 0.149% | 10 |
| gmem_to_smem | 1 | 4 | stable | 3695.39129 | GB_transport/s/GPU | 0.413% | 10 |
| smem_to_gmem | 1 | 4 | stable | 2705.32898 | GB_transport/s/GPU | 0.170% | 10 |
| gmem_to_smem | 2 | 1 | stable | 3686.82546 | GB_transport/s/GPU | 0.475% | 10 |
| smem_to_gmem | 2 | 1 | stable | 2634.85149 | GB_transport/s/GPU | 0.721% | 10 |
| gmem_to_smem | 2 | 2 | stable | 3663.57513 | GB_transport/s/GPU | 0.958% | 10 |
| smem_to_gmem | 2 | 2 | stable | 2638.64676 | GB_transport/s/GPU | 0.571% | 10 |
| gmem_to_smem | 2 | 4 | stable | 3764.79275 | GB_transport/s/GPU | 0.407% | 10 |
| smem_to_gmem | 2 | 4 | stable | 2765.28007 | GB_transport/s/GPU | 0.306% | 10 |
| gmem_to_smem | 4 | 1 | stable | 3601.02349 | GB_transport/s/GPU | 0.462% | 10 |
| smem_to_gmem | 4 | 1 | stable | 2636.26304 | GB_transport/s/GPU | 0.352% | 10 |
| gmem_to_smem | 4 | 2 | stable | 3512.58937 | GB_transport/s/GPU | 1.133% | 10 |
| smem_to_gmem | 4 | 2 | stable | 2708.61603 | GB_transport/s/GPU | 0.257% | 10 |
| gmem_to_smem | 4 | 4 | capacity_reject | 无有效统计 | GB_transport/s/GPU | — | 0 |
| smem_to_gmem | 4 | 4 | capacity_reject | 无有效统计 | GB_transport/s/GPU | — | 0 |

真实算例 `gmem_to_smem_16kib_s1_r1_all_gpu`：528×7461×1×16384=64543260672 B；除以 16852032 得到 3829.9987 GB_transport/s/GPU。计时后导出的 8650752 B 不计入主量。raw SHA `83c76bdec05b4999a7c10c082118360fe638dd4d3fb4d385da054fceacecc478`。

软件 stage、每 item 的请求数和硬件队列深度是不同对象；本实验只报告给定循环和资源的完成服务。不同 S/R 的比较还需匹配 CTA 数、占用上限和 N。


## 图表

![单CTA条件运输率中位数](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s16-main-execution/actual-report-v2/one_cta.png)

![全GPU grid的条件运输率中位数](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s16-main-execution/actual-report-v2/all_gpu.png)

颜色与格内数字为中位数；没有误差条。完整样本、min/max和CV见[CSV](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s16-main-execution/actual-report-v2/results.csv)。白色reject格为容量排除，不是零速率。原图pending C标签保留为生成时状态，当前数据资格见[C r2](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s16-main-execution/review/actual-data-and-report-C-review-r2.json)。

## 完成、保全与复现

GMEM→SMEM由issuer对对应mbarrier执行acquire等待，确认本item输入运输完成，再通过CTA同步发布就绪。槽复用要等该item退役及消费者离开后的slot-release gate；不能只以提交请求作为输入就绪。短检查capture=true会逐item读回数据，正式capture=false不做逐item读回，保留退役和槽释放同步；这是运输与同步服务，不包含实际GEMM消费者。

SMEM→GMEM的源缓冲可复用与目标完整写完成是两个条件。本探针退役使用完整bulk-group等待，确认相应旧组完成后再发布槽释放；不会仅凭源侧read等待来结束输出计时。最后执行 `cp.async.bulk.wait_group 0`，排空所有输出组，并完成剩余item退役及CTA同步后才写停止stamp。因此主窗口包含完整输出完成的final drain。最终SMEM导出与GMEM→SMEM的mbarrier失效在停止stamp之后，不加入主运输量或主窗口。


原GPU734543完成全部采样，只因归档ACK等待超时为FAILED75:0。原终态保留；CPU734612保全原960成员，CPU734616在原Python3.9.21环境完整重算。恢复没有重新采样。原裁切图和report-v1保留，v2只修绘图布局，五份数值/正文文件逐字节相同。

软件stage数影响缓冲、资源与流水组织，不直接乘入字节数。请求数R与stage S分别计量；API资源上限不证明实际驻留，软件扫描不证明物理队列深度。计数器权限不足，不把逻辑运输量当物理HBM流量或峰值。

[冻结源码与入口](../../../../../microbench/gh200_resource_campaign/families/s16/README.md)；[完整原始包](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s16-main-execution/storage-v2/collection-job734543-r2-CPU734612/actual)；[原环境重算](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s16-main-execution/storage-v2/native-replay-v1/complete734616-r2/native-replay-v1/replay.json)。

```bash
python3 -B /完整运行目录/repo/audit_s16.py --run /完整运行目录
```

原N、CTA数与输入条件必须随值引用；重新采样使用新的合法分配、自身校准和新输出身份。r3入口已获独立准入，完整换目录重放正在执行；S22最终验收尚未完成。

[最终发布状态](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s16-main-execution/published-v1/publication-final-state.json)绑定准确参数、C r2及发布签录；原prepared包的bridge_pending标签保留。
