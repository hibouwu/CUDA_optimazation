# 规格基线计算

范围：稠密 Single GEMM，FP16/BF16 输入、FP32 累加与输出，α=1、β=0。给定 M、N、K 和规格，不使用目标样本、拟合利用率、波次折扣或启动常数。

\[
F=2MNK,\quad Q=2MK+2KN+4MN\ \mathrm{B},\qquad
T_{compute}=F/P,\quad T_{memory}=Q/B,\quad T_{ideal}=\max(T_{compute},T_{memory}).
\]

Q 假定 A/B 各从 HBM 读一次、D 写一次，计算与搬运完全重叠。实际源分配为本 GPU HBM 支撑的 global 内存；缓存可由 L2 服务，Q 不是观测到的 HBM 流量。暖 L2、输出完成口径或数据驻留不符合假定时，结果只作为条件理想参照，不能称严格下界。重复加载、补齐、调度与同步由另外两类模型解释。

## 规格与口径

计算器读取[硬件规格配置](../../hardware/gh200_sm90/spec_baseline_profiles.json)，每个配置携带来源：

| 配置 | 计算吞吐 P | HBM 带宽 B | 用途 |
|---|---|---|---|
| `gh200_product`（默认） | 稠密 FP16/BF16 产品值 990 TFLOP/s | 4.0 TB/s | 产品规格参照，出处见[计算规格](../../hardware/gh200_sm90/L0.md#官方单-gpu-计算规格)与[存储规格](../../hardware/gh200_sm90/L2.md) |
| `c1_reference` | 132×4096×1.98 GHz = 1070.53056 TFLOP/s | 4.0 TB/s | 严格复现[C1-01](../../experiments/gh200_sm90/access_rules/C1-01-selection.md)的参考频率口径；不改其冻结设计 |

两者数值不同，必须随成绩保存配置名。1.98 GHz 是参考频率，不是当前 kernel 的有效频率。规格来源与参考条件在硬件文档维护。

## 运行与输出

从仓库根目录运行[计算器](../../../../../microbench/gh200_resource_campaign/access_rules/spec_baseline.py)：

```sh
python microbench/gh200_resource_campaign/access_rules/spec_baseline.py --m 4096 --n 4096 --k 4096
python microbench/gh200_resource_campaign/access_rules/spec_baseline.py --m 4096 --n 4096 --k 4096 --profile c1_reference
```

JSON 输出有效 FLOP、A/B 与 D 字节、计算时间、搬运时间、理想时间、主导项、配置和假定。4096³ 的 F=137438953472，Q=134217728 B；产品口径为计算约138.827 µs、搬运约33.554 µs，取前者。

计算器可独立复算规格参照，尚未替换 C1-01 已冻结的执行代码。各 tile 配置在这一简化基线下可并列；它不负责根据实测补上配置差异。`wave_roofline`、`structural` 的结构假设与参数仍归[半经验候选](../semi_empirical/MODEL-CANDIDATES.md)。
