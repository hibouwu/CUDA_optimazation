# 发射与 SMEM 多路径联合服务

状态：阶段9已交付完整合同的条件联合服务；R11/R12尚无经过独立干预留出的可迁移硬件规则。数据分别来自[R11](../../../experiments/gh200_sm90/Q07-smem/R11-mixed-issue.md)和[R12](../../../experiments/gh200_sm90/Q07-smem/R12-smem-path-contention.md)，本轮回放与接入见[SM-01](../../../experiments/gh200_sm90/Q08-register-residency/SM-01-replay.md)。

## 1. 现象

匹配消费者后，WGMMA＋独立 IMAD、TMA＋WGMMA 接近覆盖；FFMA＋IMAD、WGMMA＋STS、TMA＋STS 则有不同程度的额外时间。SMEM 区域不重叠不能保证服务独立。早期消费者被 CSE 消除的版本保留为失败控制，不用于价格。

## 2. 假设

| 候选 | 区分性预期 | 当前判定 |
|---|---|---|
| 所有独立流均可直接取 max | 相同工作与消费者下所有并发窗口接近较慢单路 | 已被部分配对排除 |
| 共同发射/控制或消费者贡献额外成本 | 调整发出角色、相位、消费者后，额外量随组织变化 | 保留，需固定容量和工作量 |
| 数据服务存在共享约束 | 在控制代码及消费匹配后，改变流量比例/布局仍系统改变响应 | 保留；未定位物理端口或仲裁范围 |

## 3. 实验验证

先复用已有串行、交错、跨组并发和阶段位置控制。需要进入目标 GEMM 组合时，再预先选一组各候选预测不同的流量比例或相位干预；共同保持缓冲预留、消费者和动态指令。将整个联合窗口交给组合，不能先把差值分配给某条 opcode。新预算和数值预测须另行冻结，当前不扩展扫描。

## 4. 结论与接入

### 发射配对

结构化接口为`joint_service(pair, organization, work, consumers, barriers, SASS)`条件表。在本TC帧中，独立IMAD可几乎完全覆盖；scalar两组应引用联合实测。不能由此推出物理执行管线或返回端口完全独立，也不能把动态组织差值除以动作数，变成每opcode固定惩罚。

CVT动作包含输入更新、转换和数值摘要，它的工作量和消费者与各基线一致，但不是裸CVT率。单CTA与实际GEMM的整卡竞争仍分开。资源数量、循环长度、源模式或同步变化后，需要验证迁移，而非默认沿用本倍率。

### SMEM 配对

- 本工作量、角色、缓冲和消费代码下，TMA与WGMMA可近似取匹配阶段的max，观测余量约0.3%。这不是任意payload/stage的通用重叠规则。
- WGMMA与普通STS、TMA与普通STS不能直接取max。按布局及具体组织引用联合窗口；不把差值拼成每条STS的固定惩罚。
- SW128没有使全部组合一致变快：TC+STS并发稍快，TMA+STS并发稍慢。布局收益取决于路径与工作量。
- 本实验使用单CTA、64 KiB反复读取源，属于可复用小工作集条件。没有HBM流量、物理队列深度或物理端口识别证据。

若目标 GEMM 的 payload、角色、布局或并发规模离开这些条件，重新检验迁移；数值权威仍在 R11/R12，不能从两个单路峰值推出联合服务参数。

可执行入口为`sm_joint_service.joint_service(contract, ready_times_ns, frequency_ghz)`。合同保留pair、organization、工作、角色、消费者、同步、SASS、原始身份及完整覆盖边界；外部开始取所有前驱的最大值，结束为`begin+cycles/frequency_ghz`，频率对应原SM clock64域。被覆盖的旧节点由调用方整体替换。36个正式合同及调用例子保存在[交付包](../../../../../../results/gh200_resource_campaign/access_rules/20261017-SM-01-replay-001/joint/contracts.json)，其中R11的12项逐值回放，R12的24项复用阶段6重算及已知CV差异。当前cfg_b没有匹配合同，计算与输出仍沿原独立连接。

最小迁移对照为cfg_b下一输出WGMMA与前一输出STS.64的相位配对，保持实际三条输入TMA、两源缓冲、两个math WG和寄存器帧；计算完成、wait返回与源槽释放端点沿[计算短链](COMPUTE-SERVICE.md)取得。已有R12使用四条2D TMA、64条WGMMA/unit及STS.128，完整窗口保留在原程序条件下。
