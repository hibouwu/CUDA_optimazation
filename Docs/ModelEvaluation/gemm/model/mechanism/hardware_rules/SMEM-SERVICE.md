# SMEM需求与条件服务

更新：2026-10-17。阶段6的程序需求、整窗证据与接线范围。当前m11候选及原失败维护在[M11-SMEM](M11-SMEM.md)，本轮回放见[SMEM-01](../../../experiments/gh200_sm90/Q07-smem/SMEM-01-replay.md)，顺序与上限见[PLAN](../../../PLAN.md#stage-6-smem)。

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

普通STS、WGMMA取数与TMA目的服务分别保留；[R12联合服务](joint_service.md)按原角色和工作量引用整窗。TMA与WGMMA在原条件下接近重叠，因此本轮没有将所有SMEM路径改为共同串行FIFO。

## 3. 资格与缺口

S4/S6输出指令、完整lane地址和两源槽已完成静态核对；完整地址输入与替换接口通过CPU检查。m11两个原留出仍失败，原R03/R12窗口仍绑定其消费者、轮数、资源和二进制。上述q为显式工作假设，没有新增实机价格或独立C2/C3验收。

下一项匹配观测是cfg_b实际128线程STS.64短块：同一lane映射和源槽下，区分准备、写入与后续TMA源读取，并与计算／TMA做匹配联合对照。缺少该观测时保留q和现有复合服务；矩阵LDSM/STSM继续使用原条件入口，不替代当前STS.64路径。
