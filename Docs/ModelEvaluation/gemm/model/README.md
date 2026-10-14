# 模型与证据入口

## 结构

```text
共同输入 common/inputs.md：问题、程序、输入数据、硬件、运行协议、预测要求
  │  共用接口 common/interfaces.md 与层级 L0–L4
  ├─→ 规格基线 spec/            → 理想时间与限制项
  ├─→ 半经验模型 semi_empirical/ → 条件化完整时间预测
  └─→ 机制模型 mechanism/
        程序图 program/ ＋ 硬件服务规则 hardware_rules/
          └─→ 组合 composition.md → 事件时间线与完整时间
                    │
实机验证 experiments/：同一份证据按冻结标准评分，C1/C2/C3 的定义见 GOALS
```

三类模型接收同一份问题描述，各自读取需要的字段；目标与验收按模型分节写在 [GOALS](../GOALS.md)，推进顺序见 [PLAN](../PLAN.md)。

三类模型共用实机证据，机制模型内部再分程序图、硬件规则和组合。类别表示文档的主要用途；“供谁使用”表示依赖关系；通过与否由各次冻结评分决定。V 系列主要保存验证记录，归入实测，所验证的模型仍是半经验模型。

## 目录

模型文档按模型分文件夹；实测证据同时约束多个模型，留在 [experiments](../experiments/README.md) 按设备与实验组织。

| 文件夹 | 内容 |
|---|---|
| `common/` | [共同输入](common/inputs.md)、[interfaces](common/interfaces.md) 与 [L0](common/L0.md)、[L1](common/L1.md)、[L2](common/L2.md)、[L3](common/L3.md)、[L4](common/L4.md)：各模型共用的输入字段、操作、资源、事件与层级接口 |
| `spec/` | [规格基线计算](spec/spec_baseline.md) |
| `semi_empirical/` | [现行半经验候选与参数绑定](semi_empirical/MODEL-CANDIDATES.md)、[长 K 残差](semi_empirical/MAINLINE-LONG-K.md) |
| `mechanism/program/` | [CUTLASS 程序图](mechanism/program/cutlass_program_graph.md) |
| `mechanism/hardware_rules/` | [硬件路径目录](mechanism/hardware_rules/README.md)及各路径研究页：TMA、有效频率、m11、储备路径与微架构证据审阅 |
| `mechanism/` | [组合首版](mechanism/composition.md)：cfg_b 的模块来源与资格、连接方式、输出与完成判据 |

## 按模型阅读

| 类别 | 目标与验收 | 维护入口 | 输入与产物 | 当前资格 |
|---|---|---|---|---|
| 规格 | [GOALS §1](../GOALS.md#1-规格基线) | [规格基线计算](spec/spec_baseline.md) | 形状、数据类型、规格 → 理想计算/搬运时间 | 可运行；产品规格与 C1 参考口径分别输出，不拟合 |
| 半经验 | [GOALS §2](../GOALS.md#2-半经验模型) | [MODEL-CANDIDATES](semi_empirical/MODEL-CANDIDATES.md)、[长 K 残差](semi_empirical/MAINLINE-LONG-K.md) | 静态配置、协议、冻结参数 → 条件时间预测 | V09 仍为冻结比较基线；资格见 V09 评分；开发版本另列 |
| 机制-程序 | [GOALS §3.2](../GOALS.md#32-程序图) | [CUTLASS 程序图](mechanism/program/cutlass_program_graph.md) | 源码、配置、调度 → 操作、依赖、槽代次 | 已整理源码与递推对应关系；尚无独立图导出器和完整 SASS 验收 |
| 机制-硬件 | [GOALS §3.3](../GOALS.md#33-硬件服务规则)、[C2](../GOALS.md#35-c2-机制主张) | [硬件路径目录](mechanism/hardware_rules/README.md) | 现象、竞争假设、干预 → 支持/排除/未区分的规则 | TMA、频率为主线；m11 两次留出失败；其他路径为储备 |
| 机制组合 | [GOALS §3.4](../GOALS.md#34-组合) | [组合首版](mechanism/composition.md)、[组合接口](common/interfaces.md#7-程序图的组织与组合)、[推进依赖](../PLAN.md#2-三类模型与程序图的接入依赖) | 程序图＋版本化服务规则 → 事件与完整时间 | 尚未形成可独立验收的组合模型；现有半经验递推不更名 |
| 实测 | [GOALS §4](../GOALS.md#4-实机验证) | [实机验证入口](../experiments/README.md)、[资源索引](../experiments/gh200_sm90/README.md)、[访问与验证索引](../experiments/gh200_sm90/access_rules/README.md) | 原始样本、测量协议、冻结预测 → 观测与评分 | 每项保留其采样范围、失败和历史判定 |

## 唯一维护位置

| 内容 | 在哪里修改 | 其他页面如何引用 |
|---|---|---|
| 各模型的问题、子问题与验收，共用的有效性、C1/C2/C3、选题与停止规则 | [GOALS](../GOALS.md)，按模型分节 | 链接对应章节；PLAN 只写顺序和依赖 |
| case 字段、默认值、来源与读取方 | [共同输入](common/inputs.md)；解析实现为 [case_inputs.py](../../../../microbench/gh200_resource_campaign/access_rules/case_inputs.py) | 模型与任务只引用字段名，不另写默认值 |
| 官方规格及参考口径 | [硬件文档](../hardware/gh200_sm90/README.md)、[规格配置](../hardware/gh200_sm90/spec_baseline_profiles.json) | 计算入口读取配置；不用实测吞吐覆盖产品值 |
| 当前半经验公式、版本、参数绑定 | [MODEL-CANDIDATES](semi_empirical/MODEL-CANDIDATES.md) | 数值由被绑定的冻结 JSON 维护；R/V 页保留当时版本的分析 |
| 程序操作、槽复用和调度依赖 | [程序图](mechanism/program/cutlass_program_graph.md) | 模型和硬件研究引用图；硬件延迟另由规则提供 |
| 当前硬件假设、备选和重启条件 | 各[硬件路径页](mechanism/hardware_rules/README.md) | R 页只链接当前解释；原观察、历史推导及反证仍在原页 |
| 实测值、冻结身份和逐例评分 | EXP/R/V 页链接的原始 run、manifest、score | 索引仅写摘要并指向证据；不再抄一套可修改参数表 |
| 采样与评分方法 | [RESEARCH-GUIDE](../experiments/gh200_sm90/access_rules/RESEARCH-GUIDE.md) | 具体任务冻结自己的条件清单；旧 INTERFACE 合同只作历史证据 |

新增实验时，在相应实测索引登记类别、供谁使用和状态，再从对应模型或硬件路径页链接证据。只有数值拟合改善时更新半经验候选；获得区分性证据时更新硬件假设。文档职责与链接可用 `python microbench/gh200_resource_campaign/access_rules/check_model_docs.py` 检查。
