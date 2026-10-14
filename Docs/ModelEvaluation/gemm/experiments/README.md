# 实机验证

更新：2026-10-13。规格基线、半经验模型与机制模型共用这里的实机证据；目标与验收见 [GOALS §4](../GOALS.md#4-实机验证)，模型怎样使用证据见[模型入口](../model/README.md)。

## 证据的组织

实测按设备与实验保存，不分别搬进三类模型：同一个实验常同时约束半经验参数与机制候选，例如 R09 既支撑半经验时钟，也支撑频率机制研究。各模型页链接证据并写明自己的使用方式；实测索引的“类别”与“供谁使用”两列记录这种关系。

| 位置 | 内容 |
|---|---|
| [gh200_sm90](gh200_sm90/README.md) | GH200 资源微基准（EXP），按资源查实测证据 |
| [gh200_sm90/access_rules](gh200_sm90/access_rules/README.md) | 访问规则实验（R、B01）、验证轮次（V 系列）与 C1-01 比较，按实验与验证轮次查证据 |
| [gh200_sm90_archive](gh200_sm90_archive/ARCHIVE.md) | 历史合同、旧记录与兼容副本 |
| 原始数据 | `results/gh200_resource_campaign/` 按 run-id 保存源码、构建、环境、样本与评分，由各实验页链接 |

## 协议入口

| 内容 | 位置 |
|---|---|
| GH200 的预测边界、按用途划分的测量检查、验证顺序与留出要求 | [RESEARCH-GUIDE](gh200_sm90/access_rules/RESEARCH-GUIDE.md) |
| 当前测量状态、开放问题与作业安排 | [执行计划](gh200_sm90/access_rules/PLAN.md)、[EXECUTION](gh200_sm90/access_rules/EXECUTION.md#status) |
| case 字段、默认值与运行协议字段 | [共同输入](../model/common/inputs.md) |

开发、校准数据与冻结留出分开，规则在 GOALS §4 与各设备的 RESEARCH-GUIDE；本页不另写规则。
