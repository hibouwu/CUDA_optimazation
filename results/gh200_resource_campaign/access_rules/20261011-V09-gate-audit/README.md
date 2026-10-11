# V09 冻结观测的联合验收缺口审计

本记录只复用 V09 开发数据，不修改历史评分，不设事后通过阈值，不构成新的独立验证。复放 30 条件的 300 个 dual 进程，29 条件有冻结预测；cfg_c G2 的不支持状态保留。原始文件经 replay 的哈希、数值检查和布局校验，预测及 manifest 与原批准 binding 的 SHA 一致。

## 分项聚合

旧指标先在每进程内取各 CTA 窗口的有符号误差中位数，再跨进程取中位数并取绝对值。新诊断先对每个窗口取绝对值，再按进程和条件聚合。两者以下均报告跨条件中位数；最坏窗口单列，不用总时间抵消。

| 分项 | 有预测且有窗口的条件 | 旧口径 % | 先取绝对值 % | 最坏单窗口 % |
|---|---:|---:|---:|---:|
| prefix | 29 | 11.39 | 11.39 | 71.57 |
| prefill | 29 | 15.34 | 15.34 | 148.33 |
| first_L | 29 | 4.97 | 5.34 | 73.17 |
| later_L | 21 | 7.62 | 7.73 | 55.86 |
| last_E | 29 | 3.73 | 6.74 | 89.84 |
| tail | 29 | 13.58 | 13.58 | 76.39 |

本批可比较窗口中非正观测时长计数为 0。later_L 缺少后续 tile 的条件不产生窗口；这不是误差为零，也不从完整条件集移除。最坏单窗口仍受打点端点、量化与扰动限制，不能直接解释为局部硬件服务误差。

## 关键 CTA

采用共同 globaltimer 终点差 `max(end_ns) - end_ns[chosen]`，归一化分母为 `max(end_ns) - min(entry_ns)`。旧评分将两个 entry 不同的 CTA 持续时长相除，不能直接度量最终完成时刻的差距。观测完全相同终点的 CTA 全部保留，冻结预测集合保持原样；不据此设定硬件时间分辨率或扩大候选。

- cfg_a G3 A16：代表 CTA 终点差的进程中位数占观测包络 14.64%；冻结并列集合为 132 个，集合命中 10/10，代表 CTA 命中 0/10。
- cfg_c P2 partial/sw8：冻结集合 7 个，集合命中 0/10；代表 CTA 终点差的进程中位数占包络 5.08%。

## 下一次冻结前的缺口

当前 run_v09.py 的 complete_time_passed 只检查总时间。新一轮联合通过还必须在采样前固定：逐项观测端点及适用窗口、无效/缺失观测处理、先绝对值后聚合的层级和阈值、关键 CTA 的可辨识精度与候选限制、观察器扰动标准。仅总时间通过或命中宽并列集合不能代替这些条件。此处不选择数值阈值，具体值需以独立的观测有效性证据支持后冻结。

## 复放

在仓库根目录执行：

```sh
python -m unittest discover -s microbench/gh200_resource_campaign/access_rules -p test_v09_gate_audit.py
python microbench/gh200_resource_campaign/access_rules/analyze_v09_gate_audit.py --run results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011 --output /tmp/v09-gate-audit.json
```

[完整审计](audit.json)保存逐进程分项、关键 CTA、原始文件哈希和冻结策略。没有新 GPU 作业。
