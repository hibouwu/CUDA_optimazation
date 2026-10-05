# EXP-10：GH200 矩阵片段搬运与 warp 交换实测

这组实验回答两个问题：SMEM 与寄存器之间的矩阵片段搬运，在给定布局下每周期能服务多少逻辑字节；warp 内寄存器交换，在给定依赖链与并发流下每周期能执行多少条 warp 指令。22 配置共 220 个正式进程样本，两个问题采用不同单位，图表分别展示。

## 探针与计时

矩阵探针固定单 CTA、32 线程、m8n8.b16。比较 load、store、load→store，x1/x2/x4，正常与 `.trans`；源与目标每方向4096 B，八个槽各512 B，动态 SMEM 共8192 B。x2 表示两块矩阵，由完整 warp 共同访问，不能再乘32线程。

shuffle 探针固定单 CTA，32或128线程，1或4条独立流；每条流按 `(lane+1)%32` 读取前一状态，自身有依赖链。每轮八个操作位置，所有配置固定8192轮。

输入初始化在计时前；线程0记录同 SM 的 clock64，再经 CTA barrier 进入循环。窗口内包含地址计算、矩阵结果消费或 store 值生成、必要同步及结果排空；load→store 的 PTX 源码每对指令请求 `bar.warp.sync`，但本次编译的六个往返目标在对应位置降为 NOP，SASS 中没有 `WARPSYNC`，不能把它另算成实际同步成本。host回读与核验在窗口外。因此结果代表本完整循环的服务，不能直接当成裸指令延迟或物理端口带宽。

## 矩阵搬运结果

单位为 **逻辑 B/clock64 cycle/CTA**；往返配置的工作量为读取与写入之和。

| 配置 | 收敛样本 | 中位数 | 最小值–最大值 | CV | 状态 |
|---|---:|---:|---:|---:|---|
| load_x1_normal | 10 | 16.510148 | 16.510148–16.510148 | 0.000000% | stable |
| load_x1_trans | 10 | 16.510148 | 16.510148–16.510148 | 0.000000% | stable |
| load_x2_normal | 10 | 28.433791 | 28.433791–28.433791 | 0.000000% | stable |
| load_x2_trans | 10 | 28.433791 | 28.433791–28.433791 | 0.000000% | stable |
| load_x4_normal | 10 | 35.302172 | 35.302172–35.302172 | 0.000000% | stable |
| load_x4_trans | 10 | 35.302172 | 35.302172–35.302172 | 0.000000% | stable |
| store_x1_normal | 10 | 18.952809 | 18.952809–18.952809 | 0.000000% | stable |
| store_x1_trans | 10 | 18.953794 | 18.953794–18.953794 | 0.000000% | stable |
| store_x2_normal | 10 | 33.012694 | 33.012694–33.012694 | 0.000000% | stable |
| store_x2_trans | 10 | 33.015097 | 33.015097–33.015097 | 0.000000% | stable |
| store_x4_normal | 10 | 41.775728 | 41.775728–41.775728 | 0.000000% | stable |
| store_x4_trans | 10 | 41.778537 | 41.778537–41.778537 | 0.000000% | stable |
| roundtrip_x1_normal | 10 | 8.826736 | 8.826736–8.826736 | 0.000000% | stable |
| roundtrip_x1_trans | 10 | 8.826736 | 8.826736–8.826736 | 0.000000% | stable |
| roundtrip_x2_normal | 10 | 14.839103 | 14.839103–14.839103 | 0.000000% | stable |
| roundtrip_x2_trans | 10 | 14.839103 | 14.839103–14.839103 | 0.000000% | stable |
| roundtrip_x4_normal | 10 | 22.138926 | 22.138926–22.138926 | 0.000000% | stable |
| roundtrip_x4_trans | 10 | 22.138926 | 22.138926–22.138926 | 0.000000% | stable |

![矩阵搬运全部样本、中位数和范围](/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s10-main-execution/runtime-r2/analysis-v2/matrix.png)

## warp 交换结果

单位为 **warp指令/clock64 cycle/CTA**。一条完整 warp 的 `shfl.sync` 计为一条指令；128线程包含4个warp，不能把 lane 数当成指令数，也不计为 FLOP。

| 配置 | 收敛样本 | 中位数 | 最小值–最大值 | CV | 状态 |
|---|---:|---:|---:|---:|---|
| shuffle_t32_streams1 | 10 | 0.038458 | 0.038458–0.038458 | 0.000000% | stable |
| shuffle_t32_streams4 | 10 | 0.153829 | 0.153829–0.153829 | 0.000000% | stable |
| shuffle_t128_streams1 | 10 | 0.153830 | 0.153830–0.153830 | 0.000000% | stable |
| shuffle_t128_streams4 | 10 | 0.615314 | 0.615314–0.615315 | 0.000049% | stable |

![warp交换全部样本、中位数和范围](/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s10-main-execution/runtime-r2/analysis-v2/shuffle.png)

图中点为预热收敛的独立进程，菱形为中位数，横线为全部合并样本的最小值到最大值，不是置信区间。本次差异很小，许多样本点和误差条重叠；完整精度保留在参数和CSV中。所有批次和未收敛记录均保存在归档中；只导出状态稳定且允许导出的配置。

## 从原始字段手算

`load_x2_normal` 每位置共同加载 `2×8×8×2=256 B`，总工作量 `8192×8×256=16,777,216 B`。第一条 raw 的同 SM clock64 差为 **590,045 cycle**，所以结果是 `16777216/590045=28.433790643 B/cycle/CTA`。见[原始记录](/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s10-main-execution/runtime-r2/deployment/repo/results/gh200_resource_campaign/20261001-resource-suite-v2/matrix_exchange/formal-v1/batches/load_x2_normal/batch_00/trial_00/attempt_00/raw.jsonl)。

`shuffle_t128_streams4` 工作量为 `8192轮×8位置×4条流×4个warp=1,048,576 条warp指令`。第一条 raw 周期差 **1,704,132 cycle**，得到 `1048576/1704132=0.615313837 warp指令/cycle/CTA`。见[原始记录](/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s10-main-execution/runtime-r2/deployment/repo/results/gh200_resource_campaign/20261001-resource-suite-v2/matrix_exchange/formal-v1/batches/shuffle_t128_streams4/batch_00/trial_00/attempt_00/raw.jsonl)。

load→store 的字节速率包含两个方向和它们之间的依赖、同步，不可把完成时间除二声称单向延迟。比较多流或多warp时，也要同时考虑寄存器、循环和CTA同步的变化。

## 正确性与适用范围

job733613 在 romeo-a041 上执行本次自己的22点检查。每个进程用非均匀输入验证矩阵1/2轮或shuffle 1/3/33步，长循环仍检查全部片段、校验和、输出及padding。SASS审查确认相关指令留在循环内。实际完整输出数组没有持久化，归档保存的是host核验结果和原始样本，不能宣称事后逐元素独立重算了长循环输出。

使用[条件参数](/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s10-main-execution/runtime-r2/published-v1/qualified-parameters.json)时匹配线程、warp、片段数、布局、独立流数、动态SMEM和完成边界。本组没有测Tensor Core直接消费这些片段、多个CTA竞争或全GPU服务，也没有测完整GEMM；物理端口峰值和bare latency不在结论范围内。本次已通过[独立C](/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s10-main-execution/runtime-r2/deployment/repo/results/gh200_resource_campaign/20261001-resource-suite-v2/matrix_exchange/formal-v1/reviews/C/review.json)、原ARM严格封存和同leaf换目录重放，见[完成与重放收据](/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s10-main-execution/runtime-r2/ARM-finalize/receipt.json)。条件参数只在上述范围内使用。

## 复现入口

[原始run](/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s10-main-execution/runtime-r2/deployment/repo/results/gh200_resource_campaign/20261001-resource-suite-v2/matrix_exchange/formal-v1)、[逐样本CSV](/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s10-main-execution/runtime-r2/analysis-v2/samples.csv)、[手算字段](/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s10-main-execution/runtime-r2/analysis-v2/worked_examples.json)。离线审查使用归档冻结入口：

```sh
python3 -B /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s10-main-execution/runtime-r2/deployment/repo/results/gh200_resource_campaign/20261001-resource-suite-v2/matrix_exchange/formal-v1/snapshot/repo/microbench/gh200_resource_campaign/run_suite.py audit /home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s10-main-execution/runtime-r2/deployment/repo/results/gh200_resource_campaign/20261001-resource-suite-v2/matrix_exchange/formal-v1
```

图表和说明可用 `report_matrix_exchange_v1.py --run <run> --output <新目录> --document <新文档>` 重生。重新测量使用该run的冻结合同和源码，在有效单GPU Slurm分配内先完成自己的preflight与独立B，再正式运行；不能直接把历史B当成新设备的测量准入。
