# 同 SM 原生 L 周期一致性

740161 的局部差异在原生 cycle 域也存在。off/off 的单条配对相对周期差最大为 2.484611%，不能将原 ns 控制失败全部解释为周期到时间的换算问题。这里不分离执行变化、打点影响和读数误差，也不据该数值设新门槛。

输入为原 40 个进程，每进程 432 个 L 窗口；off/on、off/off 各 10 对。45 个输入 SHA 已核对，8640 个原生周期差及 GT 差逐条匹配原分析，所有配对的 SM 相同。完整坐标数据见 [coordinate-pairs.json.gz](data/coordinate-pairs.json.gz)，分 WG/ordinal 统计见 [report.json](data/report.json)，复核及 V09 来源身份见 [verification.json](verification.json)。

## 统计对象

固定坐标为 `(CTA, WG, ordinal, tile_m, tile_n)`。每个坐标先计算 10 对的差 `arm1 - arm0`；相对差除以各自该对的 arm0 窗口。表中“典型配对绝对差”是先在每坐标取 10 对绝对差的中位数，再对 432 个坐标取中位数；“最大配对绝对差”是全部 4320 对的最大值。它们都不是两批均值或中位数之差。

| 批次 | 典型配对绝对差 / cycles | 最大配对绝对差 / cycles | 典型配对绝对差 / 相对 cycles | 最大配对绝对差 / 相对 cycles | 典型配对绝对差 / ns | 最大配对绝对差 / ns |
|---|---:|---:|---:|---:|---:|---:|
| off/on | 144.75 | 1373 | 0.221028% | 2.094648% | 224 | 1056 |
| off/off | 92 | 1667 | 0.140091% | 2.484611% | 240 | 960 |

原控制对每坐标要求配对绝对 ns 差的中位数不超过 64、最大不超过 256；两批仍各有 432/432 个 L 坐标失败。此分析未换算 GHz、未扣除共同变化、未相减两批中心、未拟合或评分新预测。off/on 与 off/off 是先后两批，不能从表中两行相减得到 observer 的因果效应。

## 不能互换的检验

脚本调用原冻结 `metric` 执行四个合成例子；以下是数学反例，不是 GPU 实测。

| 构造 | 原点差门槛 | 能说明的边界 |
|---|---|---|
| observer 真影响和读取误差均为 0，执行配对差为交替 ±512 ns | 失败 | 失败本身不定位 observer 偏差 |
| observer 真影响 +512 ns，读取误差差值 −512 ns，观测差为 0 | 通过 | 通过不能排除抵消或认证时间戳准确度 |
| off 全为 10000；on 六个 9400、四个 10900 | 失败；均值相同、中位数差 −600 | 均值等价不能替代中位数等价 |
| off 全为 10000；on 六个 10000、四个 11000 | 失败；中位数相同 | 中位数等价不能验证尾部或关键 CTA 排序 |

当前 [V09 模型](../../../../../../microbench/gh200_resource_campaign/access_rules/v09_model.py)以十个独立进程的 elapsed_us 中位数评分完整时间；[V09 验证记录](../../../../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/V09-component-validation.md)采用同一口径。完整时间中位数与上述逐次局部控制是不同检验对象。未来若明确验证中位数及不确定度，必须单独定义新协议，保留本轮失败；均值或中位数检验均不能替代时间戳准确度、扰动和逐次关键 CTA 排序的证据。

NVIDIA 将 clock64 定义为周期计数器，将 globaltimer 定义为纳秒计时器，同时注明后者行为依目标而定；该条目未提供 GH200 的数值物理误差界。[PTX ISA 8.8：clock64](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#special-registers-clock64)、[globaltimer](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#special-registers-globaltimer-globaltimer-lo-globaltimer-hi)。重复性也不能完整描述测量变化来源，参见 [NIST：Variability](https://www.itl.nist.gov/div898/handbook/mpc/section1/mpc114.htm)。这些定义不能直接替本实验标定误差。

## 复现

在仓库根目录执行，输出目录须不存在：

```sh
python results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-controls/reanalysis/main-native-L-consistency-v1/analyze.py \
  --run results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-controls \
  --output /tmp/gh200-native-L-replay
```

`observation_qualified=false`、`prediction_qualified=false`。未运行新 GPU 实验，未修改模型或留出标准。
