# 硬件规则与研究证据

本目录按物理服务问题导航。活动研究页与本页同在此目录，每条路径只保留一份状态；本页只维护入口。选题优先级与预算原则见[GOALS](../../../GOALS.md)，当前顺序见[PLAN](../../../PLAN.md)。

[实验导航](../../../experiments/gh200_sm90/README.md#research-questions)按 16 个研究问题组织；[微基准地图](MICROBENCH-ROADMAP.md)的九个方向与组合层是 2026-10-17 这一轮的实验分工，列明既有证据、补测问题、实验轴和模型接入位置。一个 TMA＋WGMMA GEMM 按机制的定价事件、证据类型、已辨识规则与缺口见[机制分解](../README.md#3-机制分解)。

| 路径 | 当前假设与研究入口 | 原始观测入口 | 状态与使用方 | 对应问题 |
|---|---|---|---|---|
| L2固定工作集的请求服务 | [L2服务与候选](L2-SERVICE.md) | EXP-13/R10/B01；[L2-01](../../../experiments/gh200_sm90/Q03-source-service/L2-01-saturation.md) | 阶段1准备包及11项CPU检查通过的记录保留；48个开发进程完成，六个形式停止，未导出新参数 | [3](../../../experiments/gh200_sm90/README.md#question-03) |
| HBM外存与读写竞争 | [外存服务](HBM-SERVICE.md) | EXP-13；[HBM-01](../../../experiments/gh200_sm90/Q03-source-service/HBM-01-replay.md) | 阶段5旧逻辑窗口重放与共享后端；真实miss／写回参数及迁移待测 | [3](../../../experiments/gh200_sm90/README.md#question-03) |
| cfg_b有限计算链 | [计算服务](COMPUTE-SERVICE.md) | EXP-07/R00；[COMPUTE-01](../../../experiments/gh200_sm90/Q14-composition/COMPUTE-01-resident-chain.md) | 阶段2 A准备包、9项CPU检查及本地静态审查记录保留；A长度留出通过，B/C具体形式停止，D/E开发观测完成 | [5](../../../experiments/gh200_sm90/README.md#question-05)、[14](../../../experiments/gh200_sm90/README.md#question-14) |
| cfg_b同步与槽释放 | [同步服务](SYNC-SERVICE.md) | R13/R14、三轮barrier_return；[SYNC-01](../../../experiments/gh200_sm90/Q04-sync-slot/SYNC-01-replay.md) | 阶段3计数／phase与条件接口已接通，旧46条件重算一致；真实短链迁移未测 | [4](../../../experiments/gh200_sm90/README.md#question-04) |
| TMA global→本地 SMEM，跨 CTA 服务 | [多 CTA 主线](TMA-SHARED-SERVICE.md)、[GEMM 候选与反例](TMA-GEMM-HYPOTHESES.md) | R05/R10/R13/R18；[历史盘点](TMA-HISTORY-REUSE.md)、[TRANSFER-01](../../../experiments/gh200_sm90/Q02-tma-request/TRANSFER-01-replay.md) | 阶段4源／目的接口与旧证据回放完成，真实短链价格迁移待测；旧候选失败保留 | [2](../../../experiments/gh200_sm90/README.md#question-02)、[3](../../../experiments/gh200_sm90/README.md#question-03) |
| TMA 单 CTA 探针控制与槽代次 | [S16](TMA-S16-STAGE-COST.md)、[输入事件候选](TMA-INPUT-EVENT-MODEL.md) | EXP-16、S14/S16 归档 | 限额收口；不能独自确定跨 SM 共享范围 | [2](../../../experiments/gh200_sm90/README.md#question-02)、[4](../../../experiments/gh200_sm90/README.md#question-04) |
| 有效频率及调用历史 | [频率主线](FREQUENCY-MECHANISM.md) | R09/V09、合格的 C1-01 开发数据 | 与 TMA 并行；半经验预测与 C2 主张分别验收 | [9](../../../experiments/gh200_sm90/README.md#question-09) |
| SMEM实际需求与输出写块 | [SMEM服务](SMEM-SERVICE.md) | R03/R12与m11两次冻结；[SMEM-01](../../../experiments/gh200_sm90/Q07-smem/SMEM-01-replay.md) | 阶段6实际STS.64地址、完整需求与离线接线；匹配服务价格待测 | [7](../../../experiments/gh200_sm90/README.md#question-07)、[6](../../../experiments/gh200_sm90/README.md#question-06) |
| 普通 LDS/STS、广播与 lane 组织 | [m11](M11-SMEM.md) | R03、两次冻结 score | 两次均未通过；当前不进入正式组合 | [7](../../../experiments/gh200_sm90/README.md#question-07) |
| 寄存器源与结果消费 | [寄存器服务](register_service.md) | R01/R02/R16；[REG-01](../../../experiments/gh200_sm90/Q08-register-residency/REG-01-replay.md) | 阶段7条件窗口回放、实际需求与逐warp CTA配额；申请时长待测 | [8](../../../experiments/gh200_sm90/README.md#question-08) |
| L1 cache与普通LSU | [普通LSU服务](L1-LSU-SERVICE.md) | R01、EXP-13/18；[L1-01](../../../experiments/gh200_sm90/Q13-l1-lsu/L1-01-replay.md) | 阶段8实际需求、旧条件回放与单CTA缓存探针已交付；新增36个开发进程及独立评分完成，8个留出条件未采样 | [13](../../../experiments/gh200_sm90/README.md#question-13) |
| 发射及 SMEM 联合服务 | [联合服务](joint_service.md) | R11/R12、旧 R04；[SM-01](../../../experiments/gh200_sm90/Q08-register-residency/SM-01-replay.md) | 阶段9完整合同可执行，目标cfg_b相位价格待测 | [7](../../../experiments/gh200_sm90/README.md#question-07)、[5](../../../experiments/gh200_sm90/README.md#question-05) |
| 驻留与配额 | [驻留/配额](residency.md) | R06/R16；[SM-01](../../../experiments/gh200_sm90/Q08-register-residency/SM-01-replay.md) | 阶段9实际容量及旧SMID窗口核对，模型活动报告保留原价格 | [8](../../../experiments/gh200_sm90/README.md#question-08)、[1](../../../experiments/gh200_sm90/README.md#question-01) |
| 其他计算、缓存、DSM/多播等 | [微架构证据审阅](MICROARCH-REVIEW.md) | EXP/B01 | 选题储备，达到 PLAN 进入条件再立项 | [11](../../../experiments/gh200_sm90/README.md#question-11)、[12](../../../experiments/gh200_sm90/README.md#question-12)、[13](../../../experiments/gh200_sm90/README.md#question-13) |

程序依赖统一见[CUTLASS 程序图](../program/cutlass_program_graph.md)。一条路径页可以引用多个实验；每个实验也可以约束多个候选。仍未区分的解释保留在该路径页，后续发现异常时按页内条件重启，不改写旧冻结成绩。
