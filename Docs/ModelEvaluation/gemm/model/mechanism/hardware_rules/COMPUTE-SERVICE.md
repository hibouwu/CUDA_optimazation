# cfg_b 有限计算链服务

更新：2026-10-20。阶段2驻留计算版本A的开发与留出18进程已完成并合格，4／13→32 Ktile窗口长度外推通过；首轮两个解释重合。Q05的wait0／wait1单因素对照及冻结长度验证支持匹配驻留窗口的每组完成滞后形式，固定截距形式未通过。实验条件、数值见证和操作入口见[COMPUTE-01](../../../experiments/gh200_sm90/Q14-composition/COMPUTE-01-resident-chain.md)，B–E的设计与依赖见[模块地图 §4](MICROBENCH-ROADMAP.md#4-首个组合切片cfg_b-真实短计算链)。

## 1. 当前证据与工作假设

| 已有证据 | 可复用部分 | 留给本阶段的缺口 |
|---|---|---|
| [EXP-07](../../../experiments/gh200_sm90/Q05-wgmma/EXP-07-wgmma.md) | N64指定长循环的4095.942 FLOP/cycle参照、完整排空与数值检查 | 实际cfg_b N128短链；`555+105n`的条件窗口未获独立留出，也未导出参数 |
| [R00-B](../../../experiments/gh200_sm90/Q05-wgmma/R00-anchor-target.md#r00-b目标-wgmma-形状15-点) | N128 K/K、1/2WG长循环分别4095.111/4095.548 FLOP/cycle；128个累加寄存器/线程 | 实际K/MN、两个不同M64子块、四个K16偏移及每组8条指令；旧每组32条 |
| 同页K/MN配对 | N256、2WG、1024轮的布局对照 | N128有限长度、末尾等待和两个输出WG的pingpong |
| [计算量干预](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/g3-stage-credit/20261015-g3-compute-credit-discrimination-003/RESULTS.md) | 增加两遍计算的增量接近名义工作量 | 原程序首段、提交提前量及完成后的返回价格仍未分离 |

当前请求级组合的计算递推在[supply_model](../../../../../../microbench/gh200_resource_campaign/access_rules/supply_model.py)维护，`cfg_b_request_model.predict()`按阶段频率换算512 cycle/Ktile与256 cycle提交提前量。历史不带该预测器的`simulate()`则读取原候选288 ns与144 ns；两者分别保留来源。

```text
M[i] = 本Ktile计算开始
E[i] = max(M[i], E[i-1]) + compute_duration
C[i] = max(M[i] + issue_floor, E[i] - commit_lead)
D[i] = E[i] + completion_lag
W[0] = C[0]
W[i] = max(C[i], D[i-1]) + wait_overhead       i > 0
release[i-1] = W[i]
mma_return = W[last] + return_overhead
main_end = max(mma_return, D[last])
release[last] = main_end
```

E为内部累计服务进度，C为提交，D为模型完成，W为软件等待返回。零额外开销、驻留输入就绪时，现有512/256假设输出：

| Ktile | 最后提交／mma_return | 最后D／main_end | 资格 |
|---:|---:|---:|---|
| 4 | 1792 cycle | 2048 cycle | 工作假设 |
| 13 | 6400 cycle | 6656 cycle | 工作假设 |
| 32 | 16128 cycle | 16384 cycle | 工作假设 |

离线入口[cfg_b_compute_chain.py](../../../../../../microbench/gh200_resource_campaign/access_rules/cfg_b_compute_chain.py)的`resident_hypothesis()`以零外部等待调用独立计算服务，按SM cycle保存上述事件。外部服务成本设0，探针的起步会合、打点和结果消费另有实际窗口，因此这些值作为服务假设保存。

`compute_timeline`由同步模块或调用方提供逐组外部前驱满足后的许可时刻，使用共同单组转换独立输出开始、提交、完成、正常wait0/1返回及末尾排空；`resident_hypothesis()`提供全部驻留就绪的特例，保留原字段与数值。跨输出调用通过`initial_service_end_cycles`传入同一服务的上一进度，缺省0；正常wait历史在本输出首组重新开始。输入／槽代次和释放由调用方连接，计算参数仍按本页证据选择。[独立模块交付](../../../../../../results/gh200_resource_campaign/access_rules/Q05-wgmma/local-compute-chain/20261018-Q05-local-compute-service-001/EXECUTION.md)核对了完整引擎与局部链逐位回归、两wait公式和异目录复现。[跨输出接口交付](../../../../../../results/gh200_resource_campaign/access_rules/Q14-composition/gemm-connection/20261019-Q14-compute-handoff-contract-001/EXECUTION.md)修复独立调用重置计算进度的反例，四组织762端点与既有完整引擎一致，默认报告保持。

[跨WG容量必要界](../../../../../../results/gh200_resource_campaign/access_rules/Q05-wgmma/local-compute-chain/20261019-Q05-compute-capacity-bound-001/README.md)按[硬件稠密计算容量](../../../hardware/gh200_sm90/L0.md#官方单-gpu-计算规格)核累计完成工作量，排除每WG各持一份q512满速服务、其他价格不变的具体形式；S6、S4及K2提前交接均有161cycle缺口，输入延后时相容。当前同SM共享计算进度保留为工作规则，内部仲裁仍未辨识。

[提交提前量的局部辨识](../../../../../../results/gh200_resource_campaign/access_rules/Q05-wgmma/local-compute-chain/20261019-Q05-commit-lead-001/README.md)核实：零额外等待、q512／d95且lead在0–417cycle内时，旧两wait完整窗口公式均不含lead，六窗口回放无法选择256或381cycle。在同活动调用足迹假设下，首次计算前／首次commit后的同剂量延迟给出两种不同预测；755945的15进程零桥接及数值通过，原始响应差267cycle，但实际剂量差29cycle超冻结12.8门槛，结论信息不足；原2048未采，两形式均未选择，价格保持。匹配与后续辨识结果见第3节。

## 2. 两个有限解释与观测对应

首轮固定两个完整窗口解释：`T(Kt)=a+512·Kt`，以及`T(Kt)=a+q·Kt`。a与q非负，后者是包含发出、commit／wait及操作数访问的条件循环服务。仅用4／13 Ktile的plain开发窗口确定参数和32 Ktile预测，冻结后采32；trace配对检查打点对窗口的影响以及末段的提交和等待。两段长度增量不相容时保存有限链反例，按[本轮登记](MICROBENCH-ROADMAP.md#6-推进次序与本轮交付)收口。

Q05保留首组无wait、末尾wait0，只改变中间各组的wait立即数，全部使用plain。预列两种形式：分别加固定截距的`T1=a1+512·Kt、T0=a0+512·Kt`；每组完成滞后的`T1=a+512·Kt+d、T0=a+512·Kt+(Kt−1)·d`。后者预测两臂差为`(Kt−2)·d`；4→13的每组长度增量分别为512和512+d。

| 探针观测 | 组合接口 | 约束范围 |
|---|---|---|
| 最后Ktile的`issue_before` | 对应`compute_batch`前 | 8条计算发出前；本地SASS将时钟放在WARPGROUP.ARRIVE之前 |
| `commit_after` | `commit`之后 | 提交返回边界及发出窗口 |
| `wait1_after` | `wait1`之后 | 较早group已经完成，约束较早D的上界 |
| `tail_wait0_after` | `mma_tail`等待后 | 全部计算已完成，约束最后D的上界 |
| 结果消费与CTA会合后结束 | 驻留探针完整窗口 | 有限链复合服务；与plain配对报告 |

等待及寄存器访问语义按[PTX ISA 8.8](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-multiply-accumulate-instructions-wgmma-wait-group)。真实程序的槽释放和两个WG交替依赖仍由[程序图](../program/cutlass_program_graph.md#3-槽代次与两种调度)提供；A没有full／empty协议，也不产生实测slot_release。

## 3. 接入与当前结论

- **条件循环量**：512 cycle/Ktile 有匹配六槽驻留计算体的证据（[COMPUTE-01](../../../experiments/gh200_sm90/Q14-composition/COMPUTE-01-resident-chain.md) A：a=769、q=512，长度留出通过）。769 cycle 是 A 完整窗口的复合固定项，不加到 GEMM 前缀或每 Ktile 服务。
- **完成滞后 d95**：wait1/wait0 的长度增量为 512/607 cycle/Ktile，32 Ktile 冻结验证误差约 0.02%；固定截距形式被反证。d 保留为匹配驻留计算组织的完成连接量。
- **软件端点**：驻留 plain 窗口 769+512·K 与软件返回 448+28·[K>1] 通过 K2 冻结验证。
- **提交提前量不可辨识**：完整窗口不含 lead；提交提前量与调用控制之间可以在 256／381 cycle 间等价变换，原假设 256 cycle 保持。
- **复合连接**：计算完成与控制的复合连接（消费者继续推进 133.4 cycle/组、empty 反馈 275 cycle）由[同步服务](SYNC-SERVICE.md)维护。
- **接入 GEMM**：可选 d95 默认不启用；启用后 stage 响应 3.899%（实测 10.467%），普通 wait1 由当前 commit 主导。
- **真实 GEMM wait0 干预（752070）**：不完整样本中 S6 +21.6%、S4 +15.5%，与 d0 的预测（S6 不变）明显不符，按资格门记为信息不足。
- **其他**：B/C 的零额外控制形式被反证；D/E 只有开发数值与路径核对。
- **未辨识**：等待返回与物理完成的关系、独立控制价格、完整供给／输出的价格迁移。

各轮记录见[计算完成与提交](../../../experiments/gh200_sm90/Q05-wgmma/local-compute-chain.md)、[局部同步链](../../../experiments/gh200_sm90/Q04-sync-slot/local-sync-chain.md)与[局部规则接回 GEMM](../../../experiments/gh200_sm90/Q14-composition/gemm-connection.md)。
