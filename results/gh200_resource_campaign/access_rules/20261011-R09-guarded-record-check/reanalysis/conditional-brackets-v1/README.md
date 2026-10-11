# guarded单次记录的条件常数周期率检验

只分析740071已有三臂记录。先完整重放verify.py身份/数值/packet链，再使用冻结snapshot中的common_rate_interval；主集成另以Fraction直接重算每对交集、精确最小总区间预算，并检验最小点可行，132对全部一致。两窗口均满足原3200ns时长下限。

假设GLOBALTIMER读取落在前后CLOCK括号内，允许每CTA独立选择正的常数周期率，原128ns总区间预算下仍为0/132可行。所需最小总区间预算min/median/max=1149.598/1212.218/1242.163ns。它是诊断量，不修改128ns预算，不作物理频率或服务价解释；没有跨SM相减clock64。

单次guarded-minus-matched同角色差：首段ns中位+96、cycle中位-761.5；后段ns中位+160、cycle中位+79.5。全部264配对SM相同。原多进程局部cycle中心门槛为512cycle，但此处只有单对，不能当作10进程评分或充分扰动证明。原现象判读仍inconclusive，observation_qualified=false。

此结果表明，在明确的读取包围假设和预算内，单一CTA内常数周期率无法同时解释两段观测。它不证明该假设成立，也不区分实际时钟演化、读取误差和观察/历史影响。不能据此启动原30进程研究、V10或迁移到cfg_a/c。

复放执行本run的conditional-brackets.py，需要本输出目录尚不存在；逐CTA原始括号、独立有理数交集和全部单对差保存在conditional-brackets.json。
