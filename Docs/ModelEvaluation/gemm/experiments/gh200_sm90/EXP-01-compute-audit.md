# EXP-01：GH200 历史计算测量复审

## 范围与结果

新审查器不导入旧分析器，按 PTX 形状和线程参与数重算工作量，再核对归档源码、目标函数 SASS、周期、样本和 summary。

| 批次 | 配置 | 正式记录 | 结论 |
|---|---:|---:|---|
| 20260930-initial | 32 | 672 | 完整性通过，保留单 CTA 条件化斜率 |
| 20260930-sustained | 42 × 2 个范围 | 420 | 完整性通过，实际周期与整卡参考周期分开 |
| 20260930-audit | 7 × 2 个范围 × 3 个长度 | 504 | 完整性通过，另核对 84 次空窗口与预热 |

三份原始二进制已从 ROMEO 取回，均匹配旧 `SHA256SUMS`。初始批次没有归档独立 cases.json；使用原 summary 配置与归档源码交叉核对，并保留缺口。

## 核对项目

FMA 按线程和打包 lane 计 FLOP；MMA/WGMMA 按矩阵形状、warp/warpgroup 数计量。循环、链数、batch、CTA 数分别核对，所有记录检查数值错误字段。

初始结果重新拟合三个长度的中位周期；持续结果重算网格完成窗口；复测仅在同一 SM 内组合 clock64 覆盖区间。核对中位数、最小/最大值及 WGMMA 配对比值，不跨 SM 相减时钟。三批共 81 个目标函数进一步检查计时边界内的每个向后分支，均包含对应计算指令；没有发现本轮 SMEM 读取那样的循环外提问题。此静态检查仍不能代替 NCU 动态指令数。

## 保留的资格缺口

- 三批均未在当时冻结统一 run spec、进度/状态及完成哈希。
- 7、5、12 次记录来自同一进程，未满足至少 10 次外部独立进程重复。
- 首批只有一条遥测；全程遥测也不自动证明每种配置热稳态。
- WGMMA wait 3 的预热未达阈值，已由原始 event 时长重新核对。
- 无可用 NCU 动态计数器；SASS presence 不替代动态计数或缓存流量。
- 固定简单输入及必要控制/同步属于适用条件；非均匀正确性补测不扩展性能分布。

三批均为 `integrity_pass=true`、`qualified_like_thor=false`。这些检查支持有条件的经验解释，不支持硬件峰值、纯指令延迟或完整 GEMM 预测。

## 来源

- [独立审查器](../../../../../microbench/gh200_resource_campaign/audit_legacy.py)
- [完整报告](../../../../../results/gh200_resource_campaign/20261001-legacy-audit-v2/REPORT.md)
- [机器可读审查](../../../../../results/gh200_resource_campaign/20261001-legacy-audit-v2/audit.json)
- [二进制恢复记录](../../../../../results/gh200_resource_campaign/20261001-legacy-audit-v2/archive_recovery.json)
- [旧实验入口](../../../../../microbench/gh200_l0/README.md)

原始数据和原报告未改写；本次只增加独立审查结果和找回的二进制。
