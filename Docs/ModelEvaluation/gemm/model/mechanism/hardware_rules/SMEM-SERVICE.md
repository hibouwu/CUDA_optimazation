# SMEM需求与条件服务

更新：2026-10-17。阶段6的程序需求、整窗证据与接线范围。当前m11候选及原失败维护在[M11-SMEM](M11-SMEM.md)，本轮回放见[SMEM-01](../../../experiments/gh200_sm90/Q07-smem/SMEM-01-replay.md)，顺序与上限见[PLAN](MICROBENCH-ROADMAP.md#6-推进次序与本轮交付)。

## 1. 地址需求

程序的普通访存输入为按指令排列的完整lane起始word地址、每lane访问宽度和读写方向，消费／依赖组织随程序保留。[ordinary_accesses](../../../../../../microbench/gh200_resource_campaign/access_rules/smem_service.py)按指令→warp→lane展开全部word，输出每bank的word集合、同word的lane列表、逻辑请求字节及不同地址字节；不会将lane排列丢掉后只留下冲突标量。

bank标号采用`word % 32`，对应[CUDA文档的连续32-bit word映射](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)。同址列表对读表示可能的广播关系，对写保留别名；服务如何分批仍按所选假设决定。m11两次冻结需求规则仅在原证据回放中复算，不赋予完整地址描述新的时间模型资格。

`cfg_b_program.smem_demands(program)`给出四类需求及唯一覆盖位置，`epilogue_smem_addresses(subtile)`给出实际普通输出写的8×128个起始word地址。具体程序事实和核对来源见[程序图 §2](../program/cutlass_program_graph.md#2-对象操作与依赖)。

| 路径 | 每个Ktile或输出子块的逻辑需求 | 当前覆盖位置 |
|---|---|---|
| TMA输入目的投递 | 每Ktile A16 KiB、B0/B1各8 KiB | 输入完成路径；独立接入时为`input_destination_service` |
| WGMMA隐式操作数引用 | 每Ktile8条；每条A2 KiB、B4 KiB，共48 KiB引用 | `compute_ns`复合计算服务；引用量与物理SMEM读取分别记录 |
| 普通输出写 | 每子块128线程×8条STS.64×8 B，共8 KiB | `epilogue_write_ns`；每输出8子块 |
| TMA输出源读取 | 每子块8 KiB，共8次 | 输出服务的`source_done`与源槽复用 |

## 2. 可执行写块假设

首版保留“一个真实8 KiB STS写块使用共同条件周期q”的复合假设，由调用方声明q和对应GHz：

```python
selected = with_output_write_cost(parameters,
    cycles_per_subtile=q, frequency_ghz=f)
report = cfg_b_request_model.simulate(program, selected, detailed_ctas=range(4))
```

接口以`q/f`替换原`epilogue_write_ns`。输出准备、控制、TMA源读取、两源槽和排空继续由现有事件引擎推进，`write_begin/write_end/submitted/source_done`各自输出；其他路径仍沿表中覆盖关系。默认参数保持，q未从旧checksum或联合窗口率拆出，也没有新增按stage、case编号或行距补偿的项。

[独立输出链](../../../../../../microbench/gh200_resource_campaign/access_rules/q07_output_chain.py)的`simulate`接收每输出的上游许可时间、准备／写／控制／tail价格及`source_read_service(t,out,j)`返回的源读取完成。与完整引擎共用`pingpong_output_state/step`，输出八子块的准备、STS写入、提交、源完成、双槽代次／复用前驱与末尾read0端点。四组改前后全部字段和observer顺序保持，独立回放及异目录复现见[交付](../../../../../../results/gh200_resource_campaign/access_rules/Q07-smem/input-output-joint/20261018-Q07-local-output-chain-001/EXECUTION.md)；价格资格沿原假设。

普通STS、WGMMA取数与TMA目的服务分别保留；[R12联合服务](joint_service.md)按原角色和工作量引用整窗。TMA与WGMMA在原条件下接近重叠，因此本轮没有将所有SMEM路径改为共同串行FIFO。

## 3. 资格与缺口

S4/S6输出指令、完整lane地址和两源槽已完成静态核对；完整地址输入与替换接口通过CPU检查。m11两个原留出仍失败，原R03/R12窗口仍绑定其消费者、轮数、资源和二进制。上述q为显式工作假设，没有新增实机价格或独立C2/C3验收。

[四组织匹配窗口](../../../experiments/gh200_sm90/Q07-smem/SMEM-01-replay.md)已使用实际八块STS.64／双输出槽及输入三请求完成开发对照。简单max失配，但目标SASS中的fragment准备仍在计时内；准备、写入与TMA源读取价格尚未分开，保留q和现有复合服务。下一缺口为准备完成后的联合响应与GEMM迁移；矩阵LDSM/STSM继续使用原条件入口，不替代当前STS.64路径。

[准备完成后的联合窗口](../../../../../../results/gh200_resource_campaign/access_rules/Q07-smem/input-output-joint/20261018-Q07-prepared-joint-001/EXECUTION.md)752311已将fragment生成及见证写置于起戳前，数值与资源通过；输出单独cycle／ns范围6.324641%／6.225681%超原5%，结论信息不足，未选择时间连接。准备见证使用lane×128+i，源码及SASS 0x4470／0x4490–0x6380确认每条scalar STS的同warp地址相隔512 B，按[NVIDIA的bank映射](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#shared-memory-access-patterns)，32个不同word落在同一bank；窗口增长尚未归因。[bank排列修复](../../../../../../results/gh200_resource_campaign/access_rules/Q07-smem/input-output-joint/20261018-Q07-bank-layout-001/EXECUTION.md)已完成准备布局与计时依赖核对，四组织八项稳定性全部通过；max与serial的双单位误差均超原5%，两个简单窗形式停止，整体信息不足。逐seed配对的并发−max中位709cycle／352ns、串行−并发中位863cycle／448ns，保留为交互约束；正式窗口17处等价算术／调度等编码变化使跨版差值仍只作观察组织对照。未选硬件规则、不拆q、不重采，原q及组合价格保持；当前缺口为正式控制推进与输入投递／输出读取同时服务的区分。

[输入组数的最小判别设计](../../../../../../results/gh200_resource_campaign/access_rules/Q07-smem/input-output-joint/20261019-Q07-joint-count-design-001/README.md)只比较额外联合窗口固定或随输入组数缩放。所有seed共用旧配对中位J；原5%带在输入1组时可分，2组时重叠。[执行756133](../../../../../../results/gh200_resource_campaign/access_rules/Q07-smem/input-output-joint/20261019-Q07-joint-count-001/README.md)完成7进程的数值及资源核对，input1/seed22暖机CV3.827328%超原2%后停止；绝对预测未冻结、并发单组留出未采，两形式未判定。原q和联合服务资格保持，本轮不重采。


[同批条件化检验774013](../../../../../../results/gh200_resource_campaign/access_rules/Q07-smem/input-output-joint/20261021-Q07-joint-count-same-batch-001/README.md)复用已审查binary，改为由同批四组分支导出共同J，再冻结一组并发预测。12个开发进程数值983424值全部通过；input1/seed3暖机CV5.266563%超2%，input4周期范围6.591470%超5%，开发资格未过，未定J、未冻结或采留出。两形式仍信息不足，本轮收口；原q和GEMM输出服务保持。

[保存时间线的接入范围](../../../../../../results/gh200_resource_campaign/access_rules/Q07-smem/input-output-joint/20261018-Q07-output-phase-001/connection-audit.json)：G3 base/S4/A16各132个CTA的第一次输出均与下一tile输入相交，输入待完成区间的交集中位为1756.115／1742.031／1758.364ns；最后一次输出均无本CTA输入在途。固定入口状态时，本CTA输出后的联合规则只能影响后续事件，首次主循环缺口保留；跨CTA共享影响另按实际共享范围检验。[相位判别](../../../../../../results/gh200_resource_campaign/access_rules/Q07-smem/input-output-joint/20261018-Q07-output-phase-001/EXECUTION.md)753514完成6进程／491712值零错，但D0桥接失败，D4096未执行，信息不足并停止。新增phase控制同时改变了首WARPSYNC位置；R15输出干预已接入共同程序，输出共享候选的当前判定见[TMA假设](TMA-GEMM-HYPOTHESES.md#2-假设)；q及默认连接保持。
