# EXP-09：GH200 SMEM 访问实测

本次固定单 CTA、256 线程、两个 32 KiB SMEM 数组，比较连续/跨步、向量、广播及独立读写。15 配置共 170 个正式进程样本；结果单位为 **B/clock64 cycle/CTA**。窗口包含访问循环、加载 checksum 和 CTA 同步，初始化、回读与输出核验在窗口之外。

## 结果

| 配置 | 合并样本数 | 中位数 | 最小值–最大值 | CV |
|---|---:|---:|---:|---:|
| read_w4_stride1 | 10 | 87.130832 | 87.130493–87.131172 | 0.0004% |
| read_w4_stride2 | 10 | 63.992188 | 63.991822–63.995362 | 0.0016% |
| read_w4_stride4 | 10 | 31.998901 | 31.998779–31.998947 | 0.0002% |
| read_w4_stride8 | 10 | 16.000233 | 16.000202–16.000244 | 0.0001% |
| read_w4_stride16 | 10 | 8.000092 | 8.000081–8.000566 | 0.0019% |
| read_w4_stride32 | 10 | 4.000092 | 4.000089–4.000332 | 0.0019% |
| read_w8_stride1 | 10 | 127.974675 | 127.973760–127.974858 | 0.0003% |
| read_w16_stride1 | 10 | 127.987611 | 127.986757–127.987794 | 0.0003% |
| write_w4_stride1 | 10 | 127.946556 | 127.945824–127.946556 | 0.0003% |
| write_w8_stride1 | 10 | 127.973394 | 127.973150–127.980350 | 0.0027% |
| write_w16_stride1 | 10 | 127.990998 | 127.983889–127.991334 | 0.0029% |
| read_w4_broadcast | 30 | 88.780904 | 88.519991–110.687365 | 4.7972% |
| duplex_w4_stride1 | 10 | 106.372157 | 106.372157–106.372663 | 0.0002% |
| duplex_w8_stride1 | 10 | 127.986116 | 127.985536–127.987794 | 0.0005% |
| duplex_w16_stride1 | 10 | 127.992157 | 127.991761–127.993836 | 0.0006% |

![15配置的全部样本、中位数及最小最大范围](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s09-main-execution/device-revalidation-v2/analysis-v1/services.png)

图中点为预热收敛的独立进程，菱形为中位数，横线为最小值到最大值，不是置信区间。广播配置触发三批，全部30个样本参与合并，CV约4.8%，接近5%的接受界限；应同时参考范围和原始分布。

![广播三批样本](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s09-main-execution/device-revalidation-v2/analysis-v1/broadcast.png)

## 从原始字段算一条结果

`duplex_w16_stride1` 每线程每轮8次16 B读取和8次16 B写入，8192轮，逻辑请求为 `256×8×16×8192×2 = 536,870,912 B`，读写各268,435,456 B。第一条原始记录的同 SM clock64差为 **4,194,563 cycle**，所以 `536870912/4194563 = 127.992096435 B/cycle/CTA`。见[原始记录](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s09-main-execution/device-revalidation-v2/repo/results/gh200_resource_campaign/20261001-resource-suite-v2/shared_memory/device-formal-v2/batches/duplex_w16_stride1/batch_00/trial_00/attempt_00/raw.jsonl)。

广播每轮仍有 `256×8×4 = 8192 B` 逻辑请求；一个warp共享地址，整个CTA每轮只有 `8个warp×8个地址×4 B = 256 B` 不同地址。请求量与地址量含义不同，不能用请求速率推断物理端口流量。

## 如何用于模型

标量stride 2/4/8/16/32的请求速率约64/32/16/8/4 B/cycle，与地址映射相符；这组实验没有用硬件计数器证明bank事务量。连续8 B和16 B向量访问、向量写入及向量独立读写约128 B/cycle，但这是本循环的服务观测。4 B读取的checksum依赖与循环控制也占用计时窗口，不能把不同访问宽度之间的差异全部归因于SMEM裸带宽。

匹配线程数、访问宽度、地址、读写组织、64 KiB分配和完成边界后，可引用[条件参数](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s09-main-execution/device-revalidation-v2/published-v1/qualified-parameters.json)。每条使用中位数，同时保留样本数、CV和范围。这里只测单CTA；ldmatrix、Tensor Core取数、TMA写SMEM、多CTA竞争及完整GEMM需要各自实验。

## 正确性、来源与复现

GPU job733542在romeo-a057正常结束。原b9f81编译/SASS产物保持不变；新GPU UUID采用独立审查的设备重验证分支，重新执行自身15点检查，同一分配再正式采样。原job733522在UUID准入检查时退出，无性能样本，继续保留。

每个加载lane参与host checksum参考，写入在计时后逐word核对；所有正式raw的errors=0、N=8192。未持久化全输出数组，数值证据限于原host检查。广播三批及全部失败/预热记录保留。本次已通过独立结果审查，原ARM CPU解释器完成严格封存及同leaf换目录重放；[C审查](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s09-main-execution/device-revalidation-v2/repo/results/gh200_resource_campaign/20261001-resource-suite-v2/reviews/S09-C-device-formal-733542-review.json)、[COMPLETE](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s09-main-execution/device-revalidation-v2/repo/results/gh200_resource_campaign/20261001-resource-suite-v2/shared_memory/device-formal-v2/COMPLETE)和[封存与重放收据](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s09-main-execution/device-revalidation-v2/ARM-finalize/receipt.json)共同提供资格。当前未采集本组物理流量计数器。

原始归档：[run](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s09-main-execution/device-revalidation-v2/repo/results/gh200_resource_campaign/20261001-resource-suite-v2/shared_memory/device-formal-v2)；逐样本数据：[samples.csv](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s09-main-execution/device-revalidation-v2/analysis-v1/samples.csv)；手算字段：[worked_examples.json](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s09-main-execution/device-revalidation-v2/analysis-v1/worked_examples.json)。

离线重算入口使用归档自己的冻结版本：

```sh
RUN=/path/to/device-formal-v2
python3 -B "$RUN/snapshot/repo/microbench/gh200_resource_campaign/run_suite.py" audit "$RUN"
```

统计严格相等依赖原Python运行环境；本机与原ARM解释器的浮点末位差异单独记录，不能修改raw以消除差异。图表与表格可用 `report_shared_memory_v1.py --run <run> --output <新目录> --document <新文档>` 重生。
