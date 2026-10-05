# EXP-11：GH200 同步与完成事件实测

这组实验比较齐步到达与固定整数工作导致的到达偏斜，测量一个CTA完整同步阶段的周期；另外测没有未完成内存请求时的fence指令序列。13配置共130个正式进程样本。原两个warp点没有编译出目标WARPSYNC，仍展示逻辑阶段观察，但不导出原生warp barrier参数。

## 探针和计时

同步点固定2048轮、每轮8阶段；fence点固定8192轮、每轮8个warp指令位置。CTA与mbarrier分别有128/256线程，warp与fence有32线程。偏斜条件是最后一个warp，或warp点的后16个lane，被选中的每个线程每阶段执行256条有依赖的整数MAD，其余参与者先到达。

计时窗口包含循环、固定工作、被测同步或fence、mbarrier等待与计数、超时保护及最终结果排空。输入初始化在窗口前，host回读在窗口外。起止时间由同一CTA的线程0用同SM的clock64记录。每个进程另做2阶段consumer正确性检查和1阶段到达诊断，两个额外launch都不贡献正式周期；不能把不同launch的clock64相减。

## CTA与mbarrier

单位：`clock64 cycle / collective phase / CTA`。

| 配置 | 收敛样本数 | 中位数 | 最小值–最大值 | CV | 状态 |
|---|---:|---:|---:|---:|---|
| cta_t128_aligned | 10 | 21.259216 | 21.259094–21.259216 | 0.000182% | stable |
| cta_t128_fixed_work_skew | 10 | 1080.008545 | 1080.008545–1080.008850 | 0.000009% | stable |
| cta_t256_aligned | 10 | 29.260315 | 29.260193–29.260437 | 0.000440% | stable |
| cta_t256_fixed_work_skew | 10 | 1082.384399 | 1082.384338–1082.384460 | 0.000006% | stable |
| mbarrier_t128_aligned | 10 | 217.889282 | 217.889282–217.889282 | 0.000000% | stable |
| mbarrier_t128_fixed_work_skew | 10 | 1293.507996 | 1293.507996–1293.507996 | 0.000000% | stable |
| mbarrier_t256_aligned | 10 | 225.889771 | 225.889771–225.889771 | 0.000000% | stable |
| mbarrier_t256_fixed_work_skew | 10 | 1294.774231 | 1294.645630–1294.774231 | 0.003141% | stable |

![CTA与mbarrier](/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s11-main-execution/analysis-v2/collective.png)
## 无未完成请求的fence

单位：`clock64 cycle / warp instruction / CTA`。

| 配置 | 收敛样本数 | 中位数 | 最小值–最大值 | CV | 状态 |
|---|---:|---:|---:|---:|---|
| fence_cta_no_outstanding_work | 10 | 10.877899 | 10.877899–10.877899 | 0.000000% | stable |
| fence_gpu_no_outstanding_work | 10 | 267.431091 | 267.431091–267.431091 | 0.000000% | stable |
| proxy_async_no_outstanding_work | 10 | 10.877106 | 10.877106–10.877106 | 0.000000% | stable |

![无未完成请求的fence](/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s11-main-execution/analysis-v2/fence.png)
## 不可导出的逻辑warp阶段

单位：`clock64 cycle / logical phase / CTA`。

| 配置 | 收敛样本数 | 中位数 | 最小值–最大值 | CV | 状态 |
|---|---:|---:|---:|---:|---|
| warp_t32_aligned | 10 | 3.633606 | 3.633606–3.633606 | 0.000000% | stable |
| warp_t32_fixed_work_skew | 10 | 1086.132874 | 1086.132874–1086.132874 | 0.000000% | stable |

![不可导出的逻辑warp阶段](/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s11-main-execution/analysis-v2/logical_warp.png)

图中的点为预热收敛的独立进程，菱形为中位数，横线为全部合并样本的最小值到最大值，不是置信区间。所有批次与未收敛记录保留；稳定且允许导出的配置才进入参数文件。

## 从原始字段算一条结果

`mbarrier_t128_aligned` 的整个CTA共同完成 `2048×8=16384` 个阶段，不能再乘128线程。第一条raw周期差为 **3,569,898 cycle**，结果是 `3569898/16384=217.889282227 cycle/phase/CTA`。见[原始记录](/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s11-main-execution/deployment/repo/results/gh200_resource_campaign/20261001-resource-suite-v2/synchronization/formal-v1/batches/mbarrier_t128_aligned/batch_00/trial_00/attempt_00/raw.jsonl)。

该条记录的等待尝试总数为2,097,152，除以阶段数和128线程，得到每线程每阶段平均1.000000次。该数量包括成功与重复轮询，不能当成mbarrier操作数或物理队列深度。

## 如何用于模型

CTA/mbarrier的阶段时间包括参与者就绪、同步服务、等待和本循环处理。偏斜点还明确包含依赖整数工作；不能将其与齐步点的差全部解释为同步指令裸延迟。单阶段的到达时间差只来自另外的诊断launch，作为就绪条件的观测，不能直接从正式循环时间中扣除。

三个fence点均没有未完成内存请求，其结果只代表这段空请求条件下的序列。fence提供相应顺序关系，不是TMA、cp.async或WGMMA统一完成事件。两个warp点被编译为无目标WARPSYNC的逻辑循环，不授native warp barrier吞吐或延迟。

匹配参与线程、同步类型、偏斜工作、等待与超时协议、静态SMEM和完成边界后，可使用[条件参数](/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s11-main-execution/published-v1/qualified-parameters.json)。这里仅测单CTA，不外推多CTA竞争、完整GEMM或硬件峰值。

## 正确性、来源与复现

job733638在romeo-a041上完成本次自己的13点检查，正式源的39个measured/correctness/arrival角色逐机器码和编译资源核对。三个角色的最终逐线程值、阶段计数、timeout/errors、wait_attempts、consumer检查和诊断时间均持久化，可离线按uint32递推独立复算。中间phase tokens未全部保存，逐phaseconsumer零错误来自执行检查，不能宣称所有中间状态全值重放。

[run归档](/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s11-main-execution/deployment/repo/results/gh200_resource_campaign/20261001-resource-suite-v2/synchronization/formal-v1)、[逐样本CSV](/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s11-main-execution/analysis-v2/samples.csv)和[手算字段](/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s11-main-execution/analysis-v2/worked_examples.json)。本次已通过[独立C](/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s11-main-execution/deployment/repo/results/gh200_resource_campaign/20261001-resource-suite-v2/synchronization/formal-v1/reviews/C/review.json)、原ARM严格封存与同leaf换目录重放，见[完成与重放收据](/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s11-main-execution/ARM-finalize/receipt.json)。

严格summary重放需要原ARM Python3.9的path/SHA/version条件。本机Python3.14执行冻结审计仍退出2，仅5处派生CV相差一个浮点末位；[差异记录](/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s11-main-execution/local-derived-differences.json)保留，未改原统计或放宽接受门槛。

原运行环境中的冻结入口重算：

```sh
# 替换为原ARM环境中实际复制的归档路径，保留末级目录名formal-v1。
RUN=/path/to/copied/formal-v1
python3 -B "$RUN/snapshot/repo/microbench/gh200_resource_campaign/run_suite.py" audit "$RUN" --require-complete
```

在有效单GPU Slurm分配内、没有并行测量时，可以用归档中的SM90a binary复现一个检查点：

```sh
"$RUN/binary/probe" mbarrier_t128_aligned 2048 3
```

它输出本次设备信息与一条观测，包含正确性、到达诊断、预热和完整循环结果；单进程观测不自动获得十进程参数资格。

图表和说明用 `report_synchronization_v1.py --run <run> --output <新目录> --document <新文档>` 重生。重新采样在有效单GPU Slurm分配内完成自己的preflight与独立B后运行原合同；原始参数与旧设备B不能自动授权未来运行。
