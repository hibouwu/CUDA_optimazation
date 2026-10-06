# EXP-08：FP8 长累加差异的四点定位

本次只运行 [边界诊断约定](EXP-08-accumulation-boundary-diagnostic.md) 的 31、32、33、64 轮四个点，每点一个独立进程、一次 target 启动，保存全部 8192 个输出。作业 `730432` 正常结束，冻结审计通过；结果证据已通过独立审查，本说明及图表待本次独立复核。

| 实际轮数 | 单元素数学参考 | 全部输出的值 | 与参考不同的元素数 |
|---:|---:|---:|---:|
| 31 | 62 | 62 | 0 / 8192 |
| 32 | 64 | 64 | 0 / 8192 |
| 33 | 66 | 64 | 8192 / 8192 |
| 64 | 128 | 64 | 8192 / 8192 |

条件与此前保持一致：E4M3×E4M3、`.f32` 累加器/输出类型、M64N64K32、128 线程、单 CTA、两条独立链。A=B=1/16，每条链每轮执行16条WGMMA，整个warpgroup每轮32条。新 host 入口不调用原正式测量 main，没有预热、pilot 或性能采样，也没有重复此前8192轮实验。实际目标496条SASS及资源与短baseline一致。

## 一条原始输出怎么算

目标写回的物理索引为 `index=(thread×2+chain)×32+fragment`。因此8192个元素对应128线程×2链×32个累加片段。索引0是thread0、chain0、fragment0，不在这里把物理位置冒充逻辑矩阵坐标。

单个输出沿一条链，每轮数学增量为 `16×32×(1/16)²=2`。例如33轮的参考为66，而raw索引0记录64，FP32 bits为`0x42800000`。31→32、32→33、33→64三个相邻区间的实际差分分别为2、0、0；对应数学增量为2、2、62。

[boundary_aggregate.json](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/low_precision/boundary31-64-v1/boundary_aggregate.json) 对每个位置都保存了四个值、bits、线程/warp/lane/chain/fragment和三个精确有理数差分。各长度中相同thread/fragment的两条链没有位模式差异，所有位置均呈现表中模式。这里的8192个元素是一次启动内的不同输出，不是8192次独立实验。

## 图与可解释范围

![四个长度与31–33轮局部视图](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/boundary31-64-v1-r4/boundary.png)

橙色范围是同一启动内输出的min/max，本次二者相同。它不是重复采样误差条或置信区间；蓝色虚线只连接数学参考，不代表测得的增长速度。

结合此前8192轮全为64的观察，本次把当前均匀输入的差异定位到了32/33轮之间。仅靠这些点仍不能区分固定轮数截断与数据相关的增量丢失，也不能反推确切内部位宽或舍入机制。每个完整输出均有限这一事实不授予数值正确性、全家族B3或性能资格。

后续若改变输入、累加方式或长度集合，须建立新约定及独立审查；原失败记录与这四点保留。当前没有通过放宽容差进入正式采样。

## 复现和来源

- [run_spec.json](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/low_precision/boundary31-64-v1/run_spec.json)：固定四点、完整快照及审查身份。
- [diagnostic_summary.json](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/low_precision/boundary31-64-v1/diagnostic_summary.json)：各点raw/receipt哈希和数值频数。
- [结果审查](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/reviews/S08-accumulation-boundary-result-review.json)：四点完整输出、逐位置差分、链比较和搬移重放通过；不授性能资格。
- [source-B审查](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/reviews/S08-accumulation-boundary-source-B-review.json)：执行前源码、运行器、实际SASS及恢复检查；不替代结果审查。
- [分析manifest](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/boundary31-64-v1-r4/manifest.json)：原始输入身份、CSV、PNG/SVG及冻结绘图代码。
- [作业入口](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/job-730432/driver.sh)：四点有界执行命令。

从仓库根目录只读复算完整原始记录：

```bash
python -B microbench/gh200_resource_campaign/run_accumulation_boundary.py audit \
  results/gh200_resource_campaign/20261001-resource-suite-v2/low_precision/boundary31-64-v1
```

在新目录重生成本版图表：

```bash
python -B results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/boundary31-64-v1-r4/report_boundary.py \
  results/gh200_resource_campaign/20261001-resource-suite-v2/low_precision/boundary31-64-v1 \
  --output /tmp/gh200-boundary-analysis-reproduced
```

绘图入口先调用归档内冻结审计，再生成数据和图；它不会启动GPU。图像精确字节还受matplotlib版本及字体环境影响，原数据和工件哈希可以单独核对。
