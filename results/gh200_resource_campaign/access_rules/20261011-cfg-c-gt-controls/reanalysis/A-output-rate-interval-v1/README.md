# 740161全部开启观察进程的首／次输出区间

全部10个开启观察的新进程，840个同CTA共同窗口配对及1680个同WG配对，首／次输出的条件平均计数率区间均不相交。共同配对的`first.rate_low/second.rate_high−1`最小／中位／最大为7.499877%、9.528519%、11.193204%；每个进程内均复现分离。

这是已返回记录的条件诊断。显式e=0仅在记录GT坐标上做算术，中央CLOCK真实采样被两GT读取包围仍是前提，GT对连续物理时间的误差界未知。结果只拒绝同CTA首／次输出共享一个精确平均率，不唯一识别物理频率、不自动排除有容限的时间预测，不取得观察或预测资格。

来源为740161、`GPU-ef8692f3-8bd7-80a2-608b-10731c9ec114`、binary `c5476d9b00679859eef64bff34e87532e4936ea73f7f432c476597b8c36a6480`。固定cfg_c 2304×3072×4096、S4、cluster1×2、dyadic/seed17/sw1及16次同buffer关闭观察的前序，全部40原始记录身份保留；GT_before只在10个开启目标内存在，没有把关闭记录重解成GT三元组。复用740160已审计的`stamp`／`window`原语，按同SM原生cycle域选最早FIRST、最晚MAIN，GT随端点标签，相等时选WG0。

| enabled进程的pair序号 | 共同首／次配对数 | 首输出率下界／次输出率上界−1最小，% | 中位，% | 最大，% |
|---|---:|---:|---:|---:|
| 00 | 84 | 8.674241 | 9.199005 | 9.910926 |
| 01 | 84 | 8.080269 | 8.833815 | 9.428732 |
| 02 | 84 | 9.025531 | 9.745743 | 10.389320 |
| 03 | 84 | 7.499877 | 7.987618 | 8.749756 |
| 04 | 84 | 9.910742 | 10.610029 | 11.193204 |
| 05 | 84 | 9.705919 | 10.332647 | 10.966026 |
| 06 | 84 | 9.149666 | 9.713045 | 10.487031 |
| 07 | 84 | 9.009641 | 9.543665 | 10.290972 |
| 08 | 84 | 8.977739 | 9.823226 | 10.546729 |
| 09 | 84 | 8.119409 | 8.908267 | 9.446139 |

逐进程全部648个窗口保留，共4320个WG窗口、2160个共同窗口、8640个GT三元组。WG区间与该批冻结reader逐值相同。配对只在同一进程、CTA、SM及WG/共同窗口内进行，不跨进程相减，也不拟合额外GT误差预算、规则、周期或时间费用。

`inputs.json`绑定67项输入SHA，涵盖全部40份raw、主线数值/控制报告、构建审计与binary；该批16项冻结文件及输入/输出archive的17/105文件与落地副本核对相同。构建source的858项核对复用[740160审计](../../../20261011-cfg-c-gt-bracket-smoke-v2/reanalysis/A-gt-window-audit-v1/README.md)，没有重做构建或全D reference。主线全D及40×4096数值验证作为绑定证据复用。独立整数／有理数复算全部6480窗口、2520配对及逐进程汇总，错误标签、边界、跨进程身份和错误相容判定四项负例均拒绝。临时目录重新运行的三份分析文件与独立复验输出逐字节相同。

[主线观察/空对照](../../README.md)的off/on 4912/6372、off/off 4729/6372点序列失败，L各432/432失败。上述区间结果不更改这些门槛与资格，不复用旧binary校准、不评分预测、不替代主线跨进程同位置的区间交集分析，也不建立完整event、新冻结留出或跨卡支持。

```sh
task_run=/home/jianyeshi/Note/CUDA/CUDA_optimazation/results/gh200_resource_campaign/access_rules/20261011-cfg-c-gt-controls
task_audit="$task_run/reanalysis/A-output-rate-interval-v1"
task_replay=$(mktemp -d)
python "$task_audit/r09_gt_output_rates.py" --root "$task_run" --inputs "$task_audit/inputs.json" --output "$task_replay/analysis"
python "$task_audit/verify.py" --root "$task_run" --inputs "$task_audit/inputs.json" --analysis "$task_replay/analysis"
```

`analysis/windows.jsonl`保留process/CTA/SM/work/ordinal/WG、原始槽位与精确区间；`phase-pairs.jsonl`保留相交判定；`audit.json`保留逐进程及全批汇总。`verification.json`、`replay-verification.json`和`output-sha256.json`记录独立复验、重放及产物SHA。仅CPU离线，无新GPU提交。
