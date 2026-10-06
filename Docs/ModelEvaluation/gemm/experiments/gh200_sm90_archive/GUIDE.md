# GH200 微基准的阅读与复现

先读[结果总览](RESULTS.md)，选择需要的资源或组合实验。当前17个家族已发布754条条件观测；最后四个家族的采样与原环境重算已完成，实际审查、发布接入及最终离线重放仍在进行。具体参数资格以结果总览链接的独立审查和来源为准。

## 从一条正式样本算工作率

选择已发布的 [S09 SMEM 访问实验](EXP-09-shared-memory-results-v1.md)中的 `duplex_w16_stride1`。它是单CTA、256线程，每线程每轮8次16 B读取和8次16 B独立写入，共8192轮。读写各268435456 B，总逻辑请求量为：

\[
Q=256\times8\times16\times8192\times2=536\,870\,912\ \mathrm{B}.
\]

第一条正式记录的 `read_payload_bytes` 和 `write_payload_bytes` 都是268435456。同一CTA记录的 `stop_cycle-start_cycle=4194563`，所以：

\[
Q/\Delta c=536\,870\,912/4\,194\,563
\approx127.992096435\ \mathrm{B/cycle/CTA}.
\]

[这条 raw](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s09-main-execution/device-revalidation-v2/repo/results/gh200_resource_campaign/20261001-resource-suite-v2/shared_memory/device-formal-v2/batches/duplex_w16_stride1/batch_00/trial_00/attempt_00/raw.jsonl)的 SHA256 是 `843509c0d7ebe8c61f3c6f13bb72144e072ad7cbbbe2af0886b41ed936d70b59`。正式统计使用10次独立进程的中位数127.992157 B/cycle/CTA，而不是只引用这一条样本。

窗口包含访问、加载checksum及CTA同步，初始化和输出核验在窗口外。这里的字节是逻辑请求，不是物理SMEM端口事务。广播实验中，线程请求量与不同地址量也分别记录。使用这个值时，必须匹配线程数、访问宽度、地址映射、读写组织、64 KiB SMEM分配及完成边界。

## 从单项服务走向组合服务

先匹配组件条件，再检查组合的事件链：输入何时就绪、计算何时完成、缓冲何时可复用、输出何时完整。只知道计算FLOP/cycle和搬运B/cycle，不能直接得到完整流水线的周期；地址、校验、同步以及共享资源可能随组织方式改变。

S19已通过实际数据与报告C及独立发布桥接。其固定32×32×32 FP32 tile、单CTA128线程、stage=2、K=8的对照为：

| 对照 | 完整K序列的周期中位数 | 主工作率中位数 |
|---|---:|---:|
| compute | 28534.836066 | 18.3736118 FLOP/cycle/CTA |
| transport | 6620.052459 | 9.89961944 B/cycle/CTA |
| serial | 32331.004918 | 16.2162610 FLOP/cycle/CTA |
| overlap | 29785.140984 | 17.6023340 FLOP/cycle/CTA |
| output | 30583.931148 | 17.2095601 FLOP/cycle/CTA |

output每个完整序列额外执行2048 FLOP并写回4096 B，输出FMA、写回和threadfence在主窗口内；其工作率分子为65536K+2048 FLOP，包含输出处理。

同一组条件下，overlap比serial的完成周期少约7.87%。这是一组固定实现的组合观测；不能把compute与transport的周期简单相加，或只取两者最大值后宣称预测已经成立。transport的校验、compute窗口前的预取、共享同步和output的额外工作都有各自边界。完整工作量、原样本及拟合残差见 [S19结果说明](EXP-19-fp32-combination-results-v1.md)。

每个sequence是完整K序列，周期除以重复次数N，而不是N×K。FLOP/cycle和B/cycle也不能直接比大小。15组直线拟合只保留为诊断；transport、stage=4的最大样本内相对误差8.60%，没有导出为物理服务系数。

## 复现既有结果

源码在 `microbench/gh200_resource_campaign/`，原始归档在 `results/gh200_resource_campaign/20261001-resource-suite-v2/`。原始结果受忽略规则管理，单独拿到Git源码不等于拿到了完整实验包。

| 操作 | 应使用的材料 |
|---|---|
| 阅读一个值 | 实验说明、参数文件、对应独立审查及完整条件 |
| 重算既有结果 | 完整运行归档中的冻结审查器与原始记录 |
| 重新采样 | 当前合法GH200分配、新输出身份、完整运行包及自己的校准 |
| 重建图表 | 对应实验的报告脚本、原数据或已审结果表、新输出目录 |

S09既有重算命令使用运行自己的冻结版本：

```bash
RUN=/path/to/device-formal-v2
python3 -B "$RUN/snapshot/repo/microbench/gh200_resource_campaign/run_suite.py" audit "$RUN"
```

严格统计重算依赖原Python环境。原ARM重算与换目录重放已经通过；本机解释器导致的统计末位差异保留在实验记录中，不能改raw或放宽相等检查来消除差异。

最后四个家族的源码入口分别为 [S16](../../../../../microbench/gh200_resource_campaign/families/s16/README.md)、[S17](../../../../../microbench/gh200_resource_campaign/families/s17/README.md)、[S19](../../../../../microbench/gh200_resource_campaign/families/s19/README.md)、[S20](../../../../../microbench/gh200_resource_campaign/families/s20/README.md)。[换目录重放包](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s22-relocation-last-four-v1/README.md)已保全准确文件及统计源码；r3入口已通过独立增量审查，完整重放正在执行；S22最终验收尚未通过。

## 使用结论的边界

样本稳定、数值正确、计时完整和计数器可用分别判断。预热收敛不证明长期热稳态，occupancy上限不证明实际驻留，工作集较大也不单独证明物理HBM流量。完整完成时间需要约定的等待和排空；fence不能替代所有异步操作的完成事件。

本轮给出组件及固定组合的条件证据，没有实施完整GEMM预测器、最优tile搜索或多GPU执行。后续模型应使用这些明确的输入条件和事件边界，并保留未取得证据的参数。
