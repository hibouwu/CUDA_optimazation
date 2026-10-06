# EXP-08：一次 FP8 WGMMA 长累加差异观察

固定 `wgmma_e4m3_g1_one_cta` 的 8192 轮诊断得到 8192 个完整输出，全部为 `64.0`，FP32 位模式均为 `0x42800000`。数学参考为 `16384`。独立结果审查通过的是观察与归档的有效性；本记录不授予数值正确性、性能或整个 FP8/INT8 家族的 B3 资格。

## 测了什么

沿用 [长累加诊断约定](EXP-08-long-accumulation-diagnostic.md)：单 CTA、128 线程、E4M3×E4M3、`.f32` 累加器/输出类型，`m64n64k32`，两条独立累加链，每轮每链 16 条 WGMMA（warpgroup 每轮共 32 条），8192 轮，A/B 均为 `1/16`，初始累加为零。seed 为 3；此固定均匀输入不随 seed 改变。

ROMEO 作业 `730368` 成功退出。运行器先查询设备，然后只启动一次目标诊断，没有预热、pilot 或正式性能采样。完整输出包含索引、十进制值与 FP32 bits，因而可以离线重算差异和频数。它与仅保存一个错误计数的旧失败记录具有不同的证据范围。

## 用原始字段手算

每个输出元素每条 WGMMA 累加 32 个乘积，因此数学增量为：

`32 × (1/16) × (1/16) = 1/8`。

每个输出属于一条独立累加链。每轮沿该链执行 16 条指令，8192 轮的数学总量为：

`8192 × 16 × 1/8 = 16384`。

归档 `diagnostic_summary.json` 的 `evidence.result` 保存 `output_count=8192`、`finite_count=8192`、`difference_count=8192`，`minimum=maximum=64.0`。独立审查从 raw 的全部输出重新计算这些字段，未把数学参考作为通过阈值。由此能直接说明本次所有输出与数学参考不同，不能从一个终值反推出内部累加位宽、舍入机制或差异第一次出现的位置。

## 与已有短检查的关系

同一目标函数的短诊断在均匀/非均匀输入、1/2 轮下通过了原先的精确参考。新长诊断的 496 条目标 SASS 与短诊断及批准的目标编译结果相同，资源为 148 registers/thread、5136 B static SMEM，occupancy API 上限为 3 CTA/SM。实际诊断只启动 1 CTA。

短检查排除了其输入和长度下可见的错误；它不能代替长循环数值验证。长诊断保留了有限差异，也不能据此断言旧 preflight 在预热还是正式计时阶段失败。后续需要新的受审、有限坐标诊断区分原因，不能把容差调大后直接采性能。

本次计时字段只用于收据和完成边界核对，未导出 FLOP/cycle 或吞吐结论。诊断运行时间也不是完成 GEMM 的性能预测。

## 归档、审查与复现

- [完整 run](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/low_precision/long-uniform8192-v1/run_spec.json)：冻结源码、构建、设备、原始输出及执行收据的入口。
- [原始输出](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/low_precision/long-uniform8192-v1/attempts/attempt_00/raw.jsonl)：完整输出值与位模式；原 raw SHA256 为 `f3ee719fc06981e198d3ffef92ec7201ceb47bf022d22f288ffdff0644b12c06`。
- [结果审查](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/reviews/S08-long-accumulation-result-review.json)：374 项 manifest 文件、源码绑定、相同目标 SASS、资源、进程清理和搬目录重放均通过。该审查绑定原实验工件，不表示本说明已独立审阅。
- [作业入口](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/job-730368/driver.sh)与[收集收据](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/job-730368/collection.json)：记录提交脚本和传输哈希。

在仓库根目录执行以下命令，只做离线审计，不启动 GPU：

```bash
python -B microbench/gh200_resource_campaign/run_accumulation_diagnostic.py audit \
  results/gh200_resource_campaign/20261001-resource-suite-v2/low_precision/long-uniform8192-v1
```

入口会验证快照并转到归档内冻结脚本。可将完整 run 复制到新目录后再审计；不得只复制 summary 而省略 snapshot、baseline、raw 或收据。
