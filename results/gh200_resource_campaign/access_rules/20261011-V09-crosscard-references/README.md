# V09 共同参照的跨卡复放

本记录是既有数据的回顾性对照，不是新的独立跨卡留出，不重新标定参数，不给出迁移通过判定。声明范围仍包含 cfg_a/b/c；本批只有 cfg_a/b 四个共同参照，cfg_c 和新留出缺口保持未完成。

源批次为 20261009-V09-heldout-job739011，GPU-43269fbc-449d-3e0f-908a-9c81229546d3；目标批次为 20261010-R09-R13-R18-failure-controls，GPU-ef8692f3-8bd7-80a2-608b-10731c9ec114。四个共同条件的输入字段一致，static setup 除 GPU UUID 外逐字段一致；cfg_a/b 的 plain/dual 二进制和 SASS 均逐字节一致。每个原始进程再次核对 GPU UUID、命令所用二进制名，并经 replay 验证哈希、数值和暖机规则。

共复放 160 个成功进程：4 条件×2 变体×2 卡×10。失败尝试保留在完整 JSON。以下为各自十进程中位数的目标/源变化百分比，正值代表目标卡批次更大。

| 条件 | plain 时间 | dual 时间 | dual 终点包络 | 最大 CTA 周期数 | CTA 有效 GHz 中位数 |
|---|---:|---:|---:|---:|---:|
| cfg_a_r13_comp_k1024_aligned | +1.39% | +1.06% | +0.88% | +0.03% | -1.00% |
| cfg_a_r13_comp_k4096_aligned | +1.04% | +1.15% | +1.59% | +0.55% | -0.30% |
| cfg_b_r13_comp_k1024_aligned | +0.23% | -1.41% | -0.62% | -0.64% | -0.64% |
| cfg_b_r13_comp_k4096_aligned | +1.25% | +1.06% | +1.16% | +2.95% | +2.69% |

有效 GHz 为各 CTA 自身 entry→end 的 cycle/ns，再在进程内取中位数。最大周期数与中位有效频率可能对应不同 CTA，不能直接相减或相乘重建总时间。包络使用共同 globaltimer 的 max(end)−min(entry)。

源批次 cfg_a K1024 的暖机次数为 8–10，cfg_b K1024 为 8–12；目标批次均为 8。其余共同参照两卡均为 8，但不据此推断运行前硬件状态完全相同。两批均为 adaptive 暖机；GPU 身份与调用历史/环境一同变化，不能把所有差异因果归于卡本身。

## 迁移结论与缺口

- 四个普通对齐参照的 plain 中位时间变化为 +0.23%～+1.39%；这只说明这些观测接近，不证明更长 K、布局、边界或 cfg_c 的迁移。
- cfg_b K4096 的最大 CTA 周期数和有效频率中位数分别增加 2.95% 与 2.69%，即使总时间接近，仍需分开检验周期规则和时钟换算。不能只校准一个完整时间缩放系数。
- 尚无新冻结模型在第二卡上、覆盖完整 cfg_a/b/c 声明范围的独立留出；这一要求未完成。未来应先保留不重校准的迁移结果，再只使用预先声明的参照集重校准，并在独立条件评分。留出不能参与重校准，也不能因当前四个参照接近而缩小范围。
- 实测最小/最大组合产生的范围保存在 JSON，仅为描述性范围，不是置信区间或新验收阈值。

## 复放

```sh
python -m unittest discover -s microbench/gh200_resource_campaign/access_rules -p test_v09_crosscard.py
python microbench/gh200_resource_campaign/access_rules/analyze_v09_crosscard.py --source results/gh200_resource_campaign/access_rules/20261009-V09-heldout-job739011 --target results/gh200_resource_campaign/access_rules/20261010-R09-R13-R18-failure-controls --output /tmp/v09-crosscard.json
```

[完整结果与来源哈希](audit.json)。本次未启动 GPU 作业。
