# 同ef卡逐线程release配对开发观测

采样前冻结：manifest.json保存40个study样本（stage4/6×off/on×10）、20个plain参照、每个实际二进制SHA、全部分析源码SHA和顺序。freeze.json绑定采集/分析脚本及清单。样本使用740056已做全D和SASS检查的二进制，不重新编译。目标仅GPU-ef8692f3-8bd7-80a2-608b-10731c9ec114；先在原Slurm CUDA可见域枚举UUID，只有该卡确属分配才收窄可见域，setup再核对。单个exclusive节点只运行一个串行计时流，AllocTRES据实记录。

新r13-release-threads-v2覆盖128个必要线程各自的原arrival前后戳。两臂同一二进制、common-sw1-16，16次前序均关闭额外观测，目标调用才改变flag。源码和协议不与旧四leader/adaptive样本拼接。每stage五对off→on、五对on→off，pair顺序用固定seed20261011打乱，失败不补样。另在开头/结尾各五个plain参照，每stage十次，CV≤5%才判稳定；前后漂移和off/plain增量完整报告，不以它们拟合修正。

原门槛不变：每线程bracket≤32cycle；local绝对配对差中位≤32/max≤128cycle；H16→wait1绝对配对差中位≤64/max≤192cycle；event绝对配对比例中位≤2%/max≤5%。先逐对abs再聚合，不用abs(median)替代。按同GPU/SM/work/group/slot/phase核对，所有40样本和无效原因保留。64cycle仅为诊断分辨带，不是硬件价格。

四次全D off/on和两次plain/stamped完整D比较在计时前执行；全部时间样本另验证4096个参考值。源/二进制/数值/身份/参照/局部与整调用门槛齐全才可能取得观察资格。这轮仅开发机制辨识，不是V10、新独立留出、跨卡通过或新服务价。

740059在CUDA Runtime按名称加载时失败，未开始数值或时间调用，records不存在。完整失败包归attempts/740059。revision2仅从当前nvcc安装前缀定位唯一libcudart.so.12，并保存库路径/SHA；manifest、条件、顺序、二进制和全部门槛不变。新freeze在重试前保存。
