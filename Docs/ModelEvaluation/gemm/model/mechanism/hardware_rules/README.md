# 硬件规则与研究证据

本目录按物理服务问题导航。活动研究页与本页同在此目录，每条路径只保留一份状态；本页只维护入口。选题优先级与预算原则见[GOALS](../../../GOALS.md)，当前顺序见[PLAN](../../../PLAN.md)。

| 路径 | 当前假设与研究入口 | 原始观测入口 | 状态与使用方 |
|---|---|---|---|
| TMA global→本地 SMEM，跨 CTA 服务 | [多 CTA 主线](TMA-SHARED-SERVICE.md)、[GEMM 候选与反例](TMA-GEMM-HYPOTHESES.md) | R05/R10/R13/R18；[历史盘点](TMA-HISTORY-REUSE.md) | 主线设计；供未来机制组合与半经验供给开发 |
| TMA 单 CTA 探针控制与槽代次 | [S16](TMA-S16-STAGE-COST.md)、[输入事件候选](TMA-INPUT-EVENT-MODEL.md) | EXP-16、S14/S16 归档 | 限额收口；不能独自确定跨 SM 共享范围 |
| 有效频率及调用历史 | [频率主线](FREQUENCY-MECHANISM.md) | R09/V09、合格的 C1-01 开发数据 | 与 TMA 并行；半经验预测与 C2 主张分别验收 |
| 普通 LDS/STS、广播与 lane 组织 | [m11](M11-SMEM.md) | R03、两次冻结 score | 两次均未通过；当前不进入正式组合 |
| 寄存器源与结果消费 | [寄存器服务](register_service.md) | R02 | 机制储备，当前只可用条件窗口 |
| 发射及 SMEM 联合服务 | [联合服务](joint_service.md) | R11/R12、旧 R04 | 机制储备，未识别通用共享资源规则 |
| 驻留与配额 | [驻留/配额](residency.md) | R06/R16 | 机制储备，容量与释放依赖已有条件证据 |
| 其他计算、缓存、DSM/多播等 | [微架构证据审阅](MICROARCH-REVIEW.md) | EXP/B01 | 选题储备，达到 PLAN 进入条件再立项 |

程序依赖统一见[CUTLASS 程序图](../program/cutlass_program_graph.md)。一条路径页可以引用多个实验；每个实验也可以约束多个候选。仍未区分的解释保留在该路径页，后续发现异常时按页内条件重启，不改写旧冻结成绩。
