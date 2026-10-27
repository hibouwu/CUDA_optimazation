# 驻留容量与动态寄存器配额

状态：阶段9已交付cfg_b容量核对与模型活动报告；实际启动分布及联合服务迁移仍待测。实测入口[R06](../../../experiments/gh200_sm90/Q08-register-residency/R06-issue-residency.md)、[R16](../../../experiments/gh200_sm90/Q08-register-residency/R16-residency-quota.md)，本轮回放见[SM-01](../../../experiments/gh200_sm90/Q08-register-residency/SM-01-replay.md)。

## 1. 现象

同一 TC tile 增加可驻留 CTA，收益随源规模和 stage 改变；`setmaxnreg` 的申请窗口还随 producer 释放前工作显著变化。CUDA 查询容量、实际 CTA 活动重叠与时间收益是三种不同观测。

## 2. 假设

| 候选 | 预测 | 当前判定 |
|---|---|---|
| 时间与 occupancy 成固定反比 | 容量 1→2 在不同供给/stage 下获得同样比例收益 | R16 不支持 |
| 更多 CTA 主要覆盖等待 | 供给与 stage 改变时，额外 CTA 的收益改变 | 与已有对照相容，尚无通用时间函数 |
| 配额申请是固定成本 | 延后释放而保持后续申请相同时，申请窗口不变 | 已被释放顺序对照排除 |
| 配额由可满足的释放事件约束 | 将工作从释放后移到释放前会延后申请完成 | 当前工作假设；不等于识别池仲裁算法 |

## 3. 实验验证

优先复用 R16 的先释放/后释放配对和供给×stage 条件。若目标配置确实受配额或同 SM 驻留影响，再固定编译资源、检查 SMID/重叠窗口，干预释放时点或请求量。软件图提供释放前驱，硬件候选负责多个请求可满足时的仲裁；现有证据不能给后者定价。当前不增加 GPU 任务。

## 4. 结论与接入

- 容量输入：使用实测编译资源与CUDA查询上限；stage增加和CTA容量增加可分别覆盖等待，不能只按occupancy比例加速。
- 配额依赖：consumer的请求必须能从producer释放获得所需资源；把释放前工作保留在事件依赖中，不能给所有setmaxnreg附加同一固定成本。
- 后续消费：本组的192次依赖FADD与其输入/输出边界明确，仅作为活跃值和分配后执行见证，不替代CUTLASS实际consumer计算服务。
- 未得到物理RF bank、端口容量、池仲裁公式，也未验证其他配额比例或tile形状。

`sm_residency.cfg_b_capacity(program, device=GH200_DEVICE_CONDITIONS)`使用显式设备条件和实际构建需求。cfg_b S4/S6均为384线程、168 reg/thread，初始分配64512个32-bit寄存器；完整SMEM分别为149504／215040 B，已含epilogue。按GH200的65536寄存器、233472 B SMEM及保留量／分配粒度核算，两构建的register／SMEM／thread／block容量分别为1／1／5／32 CTA/SM，寄存器与SMEM均独立限制到1。总上限与旧CUDA查询一致；逐因素分解采用本地SDK的算法，来源及限定保存在[核对包](../../../../../../results/gh200_resource_campaign/access_rules/Q08-register-residency/REG-01-SM-01-replay/20261017-SM-01-replay-001/examples/capacity-audit/capacity-audit.json)。`setup.smem`优先，缺省采用冻结构建需求；launch bound中的`MinBlocksPerMultiprocessor=1`用于编译。

旧G3 S6/S4各一份dual调用的132个CTA分别落在132个SMID；记录的共同`ENTRY→consumer exit`区间为45152／48960 ns，窗口与完整CTA生命周期的边界见SM-01。请求级组合的`sm_activity=True`保存复合计算实际服务段、输出准备／写块、输入在途和它们的重叠；计算排队不计入服务占用。该报告读取已求解时间线，保留原价格与端点。当前CTA入口仍用共同模型起点，物理SM分配及启动离散未建模。
