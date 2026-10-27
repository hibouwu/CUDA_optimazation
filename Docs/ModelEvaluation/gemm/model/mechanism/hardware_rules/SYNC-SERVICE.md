# cfg_b 同步与槽释放服务

更新：2026-10-20。阶段3已完成计数／phase规则、请求级组合接线、旧数据重算及B/C、D短链的实际路径观测；首复用稀疏补测也已完成，观察扰动仍未通过原门槛，未扩S4。B/C零额外控制具体形式已停止，独立返回价格尚未辨识。实验与复现入口见[SYNC-01](../../../experiments/gh200_sm90/Q04-sync-slot/SYNC-01-replay.md)，本轮范围见[PLAN](MICROBENCH-ROADMAP.md#6-推进次序与本轮交付)。

## 1. 程序事实与状态

固定 cfg_b cluster1×1。冻结[CUTLASS kernel](../../../../../../results/gh200_resource_campaign/access_rules/Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011/source/cutlass/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp)的每个数学WG有128个消费者；[PipelineTmaAsync](../../../../../../results/gh200_resource_campaign/access_rules/Q15-full-time/V09-component-validation/20261009-V09-heldout-job739011/source/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp)按该数初始化empty，cluster1分支使每个消费者贡献arrival。full由1次producer arrival及每Ktile共32768 B事务完成；实际请求为A16 KiB、B0/B1各8 KiB，见[程序图](../program/cutlass_program_graph.md#2-对象操作与依赖)。

对整个CTA持续编号的输入组g：

```text
slot = g % stages
generation = g // stages
consumer full_wait_phase = generation % 2
producer empty_wait_phase = (generation + 1) % 2
```

full与empty初始phase均为0；producer首次用phase1检查，空槽立即可取。消费组的普通wait1返回后释放较早组，末组在tail wait0后释放。槽状态跨输出tile保留，两个数学WG轮流消费；输出交接与末组完成的依赖继续由[程序图 §3](../program/cutlass_program_graph.md#3-槽代次与两种调度)维护。

[BarrierPhase](../../../../../../microbench/gh200_resource_campaign/access_rules/mbarrier_service.py)执行arrival与事务计数。只有pending arrival和tx-count同时为0才翻转phase并重置下一轮到达数；tx-count允许complete_tx先到而暂时为负，语义依据[PTX ISA 8.8 §9.7.13.15](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html)。绝对generation另外保留，以区分相隔两代的同一phase位。

## 2. 条件返回假设与组合接入

[ConditionSyncService](../../../../../../microbench/gh200_resource_campaign/access_rules/mbarrier_service.py)采用一个明确的条件假设：

```text
eligibility = max(wait_enter, input_ready)
path = ready       if wait_enter >= input_ready
       blocked     otherwise
wait_return = eligibility + r_path
```

`fast_return_ns`、`blocked_return_ns`由调用方显式给定。本轮没有从旧探针导出GEMM价格；默认组合保持原值。每个full_wait核对full(1、32768 B)，release按既有计算事件批量贡献empty的128个逻辑arrival，额外释放耗时沿用原假设。

接口接在[supply_model.shared_stage_events](../../../../../../microbench/gh200_resource_campaign/access_rules/supply_model.py)的consumer调度与普通／tail释放处。[cfg_b_request_model.simulate](../../../../../../microbench/gh200_resource_campaign/access_rules/cfg_b_request_model.py)通过独立参数`input_sync_service`接收服务对象；返回窗口包含进入、就绪、具备返回条件、实际返回，以及槽、绝对代次、两个phase位和资格。等待返回先影响计算，再经释放影响下一代搬运；`consumer_wait`绑定返回，`compute_batch`绑定实际计算开始。

启用`input_release_delay_ns`时，完整图将软件发布`release_emit`与empty完成`release`分别绑定；消费者后继及末组`main_end`依赖发布，producer槽复用及tail依赖完成。[接线修复](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/local-sync-chain/20261020-Q04-empty-completion-contract-001/README.md)通过23项测试、独立／完整入口104端点及四CTA跨输出依赖核对，八零默认报告逐位保持；释放价格仍由调用方给定。

P0存在、首次进入参数为0时，首个consumer从共同入口0等待；正`consumer_entry_delay_ns`按producer_first_work为原点换算，producer就绪端点同样带前缀，其余组按既有推进换算。[首次入口接线修复](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/local-sync-chain/20261020-Q04-wait-entry-prefix-contract-001/README.md)保留显式进入时刻，零价格与无钩子的对应端点一致。首轮前缀不会把尚未就绪的等待改判为ready。未启用接口时不增加报告字段；显式零价格保留原事件值。

普通arrival的B/C局部入口为[cfg_b_handoff_chain.timed_lifecycle](../../../../../../microbench/gh200_resource_campaign/access_rules/cfg_b_handoff_chain.py)：计时前预填，full仅一次普通arrival且事务字节为0；显式输出计算完成、full返回、wait1返回、empty发布／最后到达和producer再取得槽。生产者控制在empty检查前推进，声明延迟在第一次复用的empty返回后执行；消费者本地继续与empty最后到达分开。当前q512／lead256及d0／驻留d95的零额外控制版本通过已有长度增长、未通过延迟响应，见[局部模型交付](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/local-sync-chain/20261018-Q04-local-handoff-001/README.md)；原GEMM同步价格保持。

## 3. 旧观测约束与剩余缺口

三轮独立回放共46条件、138进程、4416轮，原始戳、每轮公式、每进程和每条件摘要均与原分析一致。源码、二进制、SASS及原始压缩样本身份保存在[回放结果](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/SYNC-01-02-control/20261017-SYNC-01-replay-001/replay/summary.json)。

下表单位为同SM clock64 cycle；每进程32轮取中位，再跨3进程取中位。两列使用各自的+512与−512条件，最后必要信号的位置保留为戳区间。

| 绑定程序与调用 | +512：等待调用窗口 | −512：最后必要信号→返回区间 |
|---|---:|---:|
| 001，手动complete_tx，1 arriver／128 waiters，try_wait后按需wait | 78 | [47,82] |
| 001，普通arrival，128 arrivers／128 waiters，同一等待形式 | 81 | [51,85] |
| 003，普通arrival，128 arrivers／1 waiter，直接wait＋后继事务登记 | 174 | [87,121] |
| 003，同一二进制改用try_wait＋后继事务登记 | 125 | [94,128] |

这些窗口包含相应调用组织及参与者推进，probe没有实际TMA或WGMMA。+512／−512的入口分别在全部信号戳之后／之前，gap0仍有重叠；实际分支结果未记录。观测支持保留调用形式与进入状态，GEMM中`r_path`及empty最后到达至producer返回的价格仍为信息不足。

CPU已检验128次arrival、三笔事务完成、负tx-count、三代phase、跨输出复用、前缀和计算排他下的独立返回端点；六个旧引擎结果逐项不变，21项同步／请求组合回归通过。[B/C实测](../../../experiments/gh200_sm90/Q14-composition/COMPUTE-01-resident-chain.md#bc-measured)已按实际信号与wait入口区间记录首复用路径和first-try调用结果；冻结的零额外控制具体形式未通过plain响应判别，按原规则停止、未采留出。[D实测](../../../experiments/gh200_sm90/Q14-composition/COMPUTE-01-resident-chain.md#d-measured)完成真实TMA事务与重新填槽数值核对。两者trace扰动超限，端点仅解释trace程序；full完成至返回、empty最后到达至producer返回的独立价格仍缺匹配证据。

[首复用稀疏补测](../../../experiments/gh200_sm90/Q04-sync-slot/SYNC-01-replay.md#3-首复用稀疏补测)的全部配对扰动仍超限，端点只解释 sparse 程序。

当前结论（2026-10-20）：

- **驻留局部链的工作规则**：消费者继续推进 133.4 cycle/组、empty 反馈 275 cycle、公共项 546 cycle，配合 q512 与 d95。S4/K13 与 S6/K32 的冻结窗口与响应误差都在 1.5% 以内。ready full 返回与 B 条件完成给出相同窗口，位置未辨识；参数与适用域见[局部同步链](../../../experiments/gh200_sm90/Q04-sync-slot/local-sync-chain.md)。
- **就绪与返回**：full 就绪与阻塞返回在现有窗口内是 773 cycle 的连续等价族。旁路观察给出相位完成到 consumer 返回不超过 77–98 cycle，但首次 TRYWAIT 总是成功，未能判别。
- **不迁移**：按 blocked 入口逐次收 275 cycle 时，S4 的 stage 响应过冲到 18.3%（实测 10.5%），已停止。
- **等待方式**：TMA 探针上 try_wait 改为 test_wait 轮询，B80 增量由 5411 变为 3366 cycle，两个极端形式均被反证；hint 由 10000000 改为 32 ns 无响应。
- **GEMM 同步服务保持原值**：full 完成到返回、empty 最后到达到 producer 返回的独立价格仍缺匹配证据。

其他轮次见[原生 D 链](../../../experiments/gh200_sm90/Q02-tma-request/native-d-chain.md)与[探针控制预算](../../../experiments/gh200_sm90/Q02-tma-request/probe-control.md)。
