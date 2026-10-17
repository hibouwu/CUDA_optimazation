# cfg_b 有限计算链服务

更新：2026-10-17。阶段2驻留计算版本A的开发与留出18进程已完成并合格，4／13→32 Ktile窗口长度外推通过；两个解释重合，尚不能区分机制。实验条件、数值见证和操作入口见[COMPUTE-01](../../../experiments/gh200_sm90/Q14-composition/COMPUTE-01-resident-chain.md)，B–E的设计与依赖见[模块地图 §4](MICROBENCH-ROADMAP.md#4-首个组合切片cfg_b-真实短计算链)。

## 1. 当前证据与工作假设

| 已有证据 | 可复用部分 | 留给本阶段的缺口 |
|---|---|---|
| [EXP-07](../../../experiments/gh200_sm90/Q05-wgmma/EXP-07-wgmma.md) | N64指定长循环的4095.942 FLOP/cycle参照、完整排空与数值检查 | 实际cfg_b N128短链；`555+105n`的条件窗口未获独立留出，也未导出参数 |
| [R00-B](../../../experiments/gh200_sm90/Q05-wgmma/R00-anchor-target.md#r00-b目标-wgmma-形状15-点) | N128 K/K、1/2WG长循环分别4095.111/4095.548 FLOP/cycle；128个累加寄存器/线程 | 实际K/MN、两个不同M64子块、四个K16偏移及每组8条指令；旧每组32条 |
| 同页K/MN配对 | N256、2WG、1024轮的布局对照 | N128有限长度、末尾等待和两个输出WG的pingpong |
| [计算量干预](../../../../../../results/gh200_resource_campaign/access_rules/20261015-g3-compute-credit-discrimination-003/RESULTS.md) | 增加两遍计算的增量接近名义工作量 | 原程序首段、提交提前量及完成后的返回价格仍未分离 |

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

离线入口[cfg_b_compute_chain.py](../../../../../../microbench/gh200_resource_campaign/access_rules/cfg_b_compute_chain.py)的`resident_hypothesis()`直接调用现有引擎，以1 GHz数值坐标换成cycle保存上述事件。外部服务成本设0，探针的起步会合、打点和结果消费另有实际窗口，因此这些值作为服务假设保存。

## 2. 两个有限解释与观测对应

首轮固定两个完整窗口解释：`T(Kt)=a+512·Kt`，以及`T(Kt)=a+q·Kt`。a与q非负，后者是包含发出、commit／wait及操作数访问的条件循环服务。仅用4／13 Ktile的plain开发窗口确定参数和32 Ktile预测，冻结后采32；trace配对检查打点对窗口的影响以及末段的提交和等待。两段长度增量不相容时保存有限链反例，按[本轮登记](../../../PLAN.md#stage-2-compute-a)收口。

| 探针观测 | 组合接口 | 约束范围 |
|---|---|---|
| 最后Ktile的`issue_before` | 对应`compute_batch`前 | 8条计算发出前；本地SASS将时钟放在WARPGROUP.ARRIVE之前 |
| `commit_after` | `commit`之后 | 提交返回边界及发出窗口 |
| `wait1_after` | `wait1`之后 | 较早group已经完成，约束较早D的上界 |
| `tail_wait0_after` | `mma_tail`等待后 | 全部计算已完成，约束最后D的上界 |
| 结果消费与CTA会合后结束 | 驻留探针完整窗口 | 有限链复合服务；与plain配对报告 |

等待及寄存器访问语义按[PTX ISA 8.8](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-multiply-accumulate-instructions-wgmma-wait-group)。真实程序的槽释放和两个WG交替依赖仍由[程序图](../program/cutlass_program_graph.md#3-槽代次与两种调度)提供；A没有full／empty协议，也不产生实测slot_release。

## 3. 接入与当前结论

[准备包](../../../../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-01-preparation-001/README.md)的9项CPU检查与本地静态审查保留；[开发实测](../../../../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-01-execution-002/dev/summary.json)及[留出](../../../../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-01-execution-002/heldout/summary.json)已完成同卡数值、资源与窗口核对。[开发冻结](../../../../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-01-fit-001/dev-freeze.json)的名义与条件解释均得到`a=769、q=512 cycle`，两者预测完全重合；[独立长度评分](../../../../../../results/gh200_resource_campaign/access_rules/20261017-COMPUTE-01-heldout-001/score.json)通过本轮原5%标准，但不能排除竞争解释。

512 cycle/Ktile现在有匹配六槽驻留计算体的条件循环量证据，覆盖该程序的发出、描述符／fence、提交／等待与操作数访问。769 cycle是A完整窗口的复合固定项，包含会合、消费等未分离过程，不加到GEMM前缀或每Ktile服务；256 cycle提交提前量仍为原工作假设。末段观测只保存软件发出／提交／等待返回边界，不将其差值命名为物理指令延迟。当前cfg_b计算服务及其他价格保持原版。

[B/C实测](../../../experiments/gh200_sm90/Q14-composition/COMPUTE-01-resident-chain.md#bc-measured)的开发进程全部合格，但q512／lead256与零额外控制价格的具体形式被冻结plain响应反证，按原规则停止，未采留出；trace扰动超限，端点不迁移到plain。该失败没有单独辨识barrier或返回价格。

[D原生TMA](../../../experiments/gh200_sm90/Q14-composition/COMPUTE-01-resident-chain.md#d-measured)与[E真实pingpong输出](../../../experiments/gh200_sm90/Q14-composition/COMPUTE-01-resident-chain.md#e-measured)已完成有限开发数值／路径核对，没有正式时间预测或留出验收。E的trace实际区分了mma返回与MAIN_END；D/E观察扰动均不满足向plain迁移端点的条件。

当前限定结论是A驻留复合窗口的有限长度外推相容，B/C具体时间连接已停止，D/E提供匹配程序的数值与软件路径证据。等待返回与物理完成、独立控制价格及完整供给／输出的价格迁移仍未辨识；当前组合没有新增参数或全GEMM独立C2/C3通过资格。
