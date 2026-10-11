# 首输出共同区间控制

仅map1024、同740066二进制、10对off/on。每进程16前序off；偶数trial off/on、奇数on/off，pair顺序固定seed20261012。指定ef卡必须属于实际Slurm分配，exclusive单流；完整AllocTRES保留。20样本全部保存，不补样、不改变门槛。

每个固定CTA/role分别要求FIRST→MAIN ns配对绝对差中位≤64ns、最大≤256ns，24角色全部通过才判共同L控制通过；不池化、不用有符号中位抵消。整调用绝对比例中位≤2%、最大≤5%，off整调用CV≤5%。此为已有约4μs现象的工程误差预算，256ns为其约1/16，非实测时钟精度。通过也不约束五个内部端点的再分配，不升级完整观察资格或生成服务价。旧dual与新off前序不同，未纳入本次因果对照。

源/二进制/分析/次序采样前冻结。解析每调用4096参考值、零填充、输入身份、24角色和off清零，逐对保留SM变化。只做common_L_control_passed判定，internal_endpoints_qualified与observation_qualified始终false。目标map960不采样。


## 740068结果：共同L控制失败

20样本无缺失/错误，全部4096值逐调用参考复核及零填充检查通过；on各24角色，off新戳全零。24角色全部超过局部门槛：绝对差中位范围128..352ns、最大范围448..800ns，全部原配对SM相同。整调用绝对比例中位0.40043%、最大2.61609%通过，off CV0.97849%通过；不能抵消局部门槛失败。common_L_control_passed、internal_endpoints_qualified、observation_qualified均false。

逐trial在24角色上的有符号差中位为[16,-528,-96,320,-32,-112,0,-480,-368,-416]ns，只用于说明空间共同变化，不参与冻结评分或补偿。各角色有符号中位在-256..-16ns，并非一致正的读指令开销；尚未分离异步服务、历史、调度或观察器影响，不据此作因果归因。

作业a058/指定ef卡，exclusive实际AllocTRES为288CPU/4GPU/16GiB，19秒，只运行一个串行计时流；目标卡确认属于原始分配设备集合后才收窄可见域。完整输出包output.tar.gz含原始记录和分配证据，解到returned；analysis_snapshot.tar.gz解到本目录，执行python analyze.py可复放。冻结脚本的独立算术检查确认有符号抵消被拒绝、边界含等号、单个最大异常保留。此批失败后不补样、不扩大预算、不采map960。
