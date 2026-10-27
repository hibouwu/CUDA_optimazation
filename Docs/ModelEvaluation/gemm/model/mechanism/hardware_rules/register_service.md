# 寄存器源供给与结果消费

状态：阶段7离线接入。有限范围见[PLAN](MICROBENCH-ROADMAP.md#6-推进次序与本轮交付)，旧证据回放与剩余观测见[REG-01](../../../experiments/gh200_sm90/Q08-register-residency/REG-01-replay.md)；实测入口保留[R02](../../../experiments/gh200_sm90/Q08-register-residency/R02-register-service.md)。

## 1. 现象

R02 固定 40 reg/thread 后，轮换源仍比复用源慢 5.26%；改变保留目的数还改变串行消费尾部。固定编译资源已缩小旧差异，剩余差异尚未隔离到 RF 物理结构。

[R16](../../../experiments/gh200_sm90/Q08-register-residency/R16-residency-quota.md)在与cfg_b相同的初始／角色配额下，先释放的consumer申请窗口为144.5 cycle；将64／256次producer整数动作移到释放前，窗口变为2106.5／7866.5 cycle。该窗口提供释放前驱的条件证据。

## 2. 假设

| 候选 | 区分性预期 | 当前证据 |
|---|---|---|
| 编译资源数量是全部原因 | 固定帧后源模式差异消失 | 已被 R02 固定帧对照排除 |
| 源字段、复用标记或发射排程决定剩余差异 | 受控改变这些字段或排程时响应随之变化 | 尚未分离；当前工作解释保留这些条件 |
| RF 服务约束贡献剩余差异 | 固定动态指令、消费与排程后，受控源映射仍改变推进 | 尚不能与上一项区分；不命名 bank/端口数量 |
| CTA池内可用配额限制申请成功 | 延后producer释放将延后consumer取得所需寄存器 | R16支持这一条件依赖；具体等待时长仍绑定原探针 |

## 3. 实验验证

本阶段先按REG-01回放固定证据。下一次源服务匹配干预先找同一帧、相同指令数与依赖的可实现条件，核对 SASS 的源字段、reuse、寄存器数、spill 和消费者。分别量推进与最终可用，不以整窗口差给每次 RF 访问定价。如果编译器使两路排程同时改变，则报告不可区分，不强拟合端口模型。预算与数值预测在立项时冻结，本页不自动启动测量。

## 4. 结论与接入

| 模型接口 | 当前可用的内容 | 不成立的外推 |
|---|---|---|
| L0/L1 条件计算服务 | 精确源模式、链数、参与 warp 和 SASS 下的推进/完成周期、FLOP或OP计量 | 任意编译帧的源池倍率；一个 RF 读端口峰值 |
| 结果消费 | 指定串行 FP32 消费者 P=1/8/32 的窗口及额外尾部范围 | RF 物理写延迟；任意消费者、P=64/128 或实际 epilogue 的成本 |
| 混合服务 | 相同工作量和明确地址/角色下的完整时间与消费尾部 | 为每条 opcode 独立附加一个返回惩罚 |
| 保留容量 | 当前编译寄存器数、保留结果数、无 spill | 仅凭占用数推实际驻留或寄存器 bank 结构 |

固定帧诊断已排除“只因寄存器分配数量不同”这一完整解释，但两个分支仍有不同源字段、复用标记与编译排程，未隔离物理端口机制。物理 RF 布局和最小可用性间隔仍未知；合法依赖的零额外尾部不等于写服务时间为零。

新反例只有在影响目标配置的源模式/消费判断时触发重新研究；当前数值只从 R02 对应窗口读取，不维护通用 RF 单价。

实际cfg_b需求由`cfg_b_program.register_demands(program)`输出，程序事实见[程序图 §2](../program/cutlass_program_graph.md#2-对象操作与依赖)。[CfgBRegisterQuota](../../../../../../microbench/gh200_resource_campaign/access_rules/register_quota.py)按12个warp维护CTA配额池，`try_set(warp,target)`在可用量足够时完成调整；调用方决定尝试顺序。一个producer warp释放4096后即可支持两个consumer warp各取得2048，其余producer warp可以稍后释放。[PTX 8.8](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg)规定warp同步和CTA池，同WG各warp执行相同指令。

`expand_cta(...,register_quota=True)`展开12个启动完成节点；producer首次工作依赖warp0释放，各consumer首次计算依赖本WG四个warp取得配额。资源耦合由配额账本判断，软件图保留各warp本地先后。`simulate(...,register_quota=True)`输出这些未单独定价的节点，原时间服务保持；六图的原请求、端点和输出事件已核对相等。FIRST_WORK只确认producer warp0的释放在前缀内，其余warp释放和consumer申请时长待测。
