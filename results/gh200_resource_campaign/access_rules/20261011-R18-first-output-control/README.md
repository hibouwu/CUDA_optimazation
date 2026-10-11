# 首输出共同区间控制

仅map1024、同740066二进制、10对off/on。每进程16前序off；偶数trial off/on、奇数on/off，pair顺序固定seed20261012。指定ef卡必须属于实际Slurm分配，exclusive单流；完整AllocTRES保留。20样本全部保存，不补样、不改变门槛。

每个固定CTA/role分别要求FIRST→MAIN ns配对绝对差中位≤64ns、最大≤256ns，24角色全部通过才判共同L控制通过；不池化、不用有符号中位抵消。整调用绝对比例中位≤2%、最大≤5%，off整调用CV≤5%。此为已有约4μs现象的工程误差预算，256ns为其约1/16，非实测时钟精度。通过也不约束五个内部端点的再分配，不升级完整观察资格或生成服务价。旧dual与新off前序不同，未纳入本次因果对照。

源/二进制/分析/次序采样前冻结。解析每调用4096参考值、零填充、输入身份、24角色和off清零，逐对保留SM变化。只做common_L_control_passed判定，internal_endpoints_qualified与observation_qualified始终false。目标map960不采样。
