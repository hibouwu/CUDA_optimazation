# 静态历史归一化主线重算

重新执行A冻结源码提取审计，equivalence.json、完整static-signature.json.gz和1160预测event-identity.json.gz三个解析内容均与A逐字段相同。80raw/2120调用/554400角色工作计数、10签名只有1种、1160 previous_call精确一致；ldb+16与K减半负例均区分。

现有冻结模型所消费的价格输入相同，故同参数同设备支持条件下T0比值为1。swizzle坐标变化不进入这些价格输入，不代表实际硬件成本不受顺序影响。实际history卡与冻结校准卡不同，拒绝跨卡自由预测保持；只提取静态输入，未调用受拒绝的数值预测。

A的182输入SHA/1160预测/20项oracle算术验证也重跑通过。20目标事后倍率最大误差9.990275%只属于目标已知的oracle，不作为通过或采用。另一个同批十目标的更强失败见main-switch-multiplier-bound-v1。
