# EXP-04：访存基线与计时空窗口（已并入其他页面）

[结果总览](README.md)

本页内容于 2026-10-08 按测量对象迁出，原始 run `memory_baseline/formal-v3-a` 与探针 `probes/memory_baseline.cu` 不变。

| 原内容 | 现位置 |
|---|---|
| SMEM stride 读写，含 stride=1 不稳定的记录 | [EXP-09 前序测量](EXP-09-smem.md#exp-04) |
| global 读、写与依赖复制 | [EXP-13 前序测量](EXP-13-global-rw.md#exp-04) |
| 单 CTA 与整卡计时空窗口 | [R09 计时空窗口](access_rules/R09-inkernel-clock-stages.md#empty-window) |

迁出前的全文见 Git 历史。
