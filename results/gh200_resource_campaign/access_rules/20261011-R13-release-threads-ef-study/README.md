# 同ef卡逐线程release配对开发观测

采样前冻结：manifest.json保存40个study样本（stage4/6×off/on×10）、20个plain参照、每个实际二进制SHA、全部分析源码SHA和顺序。freeze.json绑定采集/分析脚本及清单。样本使用740056已做全D和SASS检查的二进制，不重新编译。目标仅GPU-ef8692f3-8bd7-80a2-608b-10731c9ec114；先在原Slurm CUDA可见域枚举UUID，只有该卡确属分配才收窄可见域，setup再核对。单个exclusive节点只运行一个串行计时流，AllocTRES据实记录。

新r13-release-threads-v2覆盖128个必要线程各自的原arrival前后戳。两臂同一二进制、common-sw1-16，16次前序均关闭额外观测，目标调用才改变flag。源码和协议不与旧四leader/adaptive样本拼接。每stage五对off→on、五对on→off，pair顺序用固定seed20261011打乱，失败不补样。另在开头/结尾各五个plain参照，每stage十次，CV≤5%才判稳定；前后漂移和off/plain增量完整报告，不以它们拟合修正。

原门槛不变：每线程bracket≤32cycle；local绝对配对差中位≤32/max≤128cycle；H16→wait1绝对配对差中位≤64/max≤192cycle；event绝对配对比例中位≤2%/max≤5%。先逐对abs再聚合，不用abs(median)替代。按同GPU/SM/work/group/slot/phase核对，所有40样本和无效原因保留。64cycle仅为诊断分辨带，不是硬件价格。

四次全D off/on和两次plain/stamped完整D比较在计时前执行；全部时间样本另验证4096个参考值。源/二进制/数值/身份/参照/局部与整调用门槛齐全才可能取得观察资格。这轮仅开发机制辨识，不是V10、新独立留出、跨卡通过或新服务价。

740059在CUDA Runtime按名称加载时失败，未开始数值或时间调用，records不存在。完整失败包归attempts/740059。revision2仅从当前nvcc安装前缀定位唯一libcudart.so.12，并保存库路径/SHA；manifest、条件、顺序、二进制和全部门槛不变。新freeze在重试前保存。

740060在UUID符号查找时失败，也没有数值或时间样本；完整包归attempts/740060。revision3移除临时ctypes枚举，逐个原Slurm可见token调用现有plain二进制setup取得UUID，确认目标属于分配后才选中。仅启动层变化，manifest SHA继续不变；两次失败不计入样本或替换样本。


## 740061结果：观察资格未通过

a058上GPU-ef8692f3-8bd7-80a2-608b-10731c9ec114属于实际分配slot0。exclusive作业完成65秒，退出码0；AllocTRES为billing=288,cpu=288,gres/gpu=4,mem=16G,node=1，step为4CPU/4GPU，只运行一个串行计时流。40个配对样本和20个plain参照全部收齐；全D检查、4096值逐调用检查、身份与记录解码全部通过，无补样或删除。

| stage | local绝对差中位/最大（cycle） | 上游绝对差中位/最大（cycle） | event绝对比例中位/最大 | plain CV | 资格 |
|---|---|---|---|---|---|
| 4 | 2 / 43 | 64.5 / 848 | 1.3132% / 3.6866% | 0.9939% | 上游失败 |
| 6 | 6 / 64 | 8.5 / 1020 | 0.8521% / 2.1066% | 0.6383% | 上游失败 |

上游有符号中位数分别为0和3.5cycle，不能替代绝对差门槛。20个on样本的条件区间虽均与post-arrival路径相容，但整批观察资格失败，不能采用机制排他结论或服务价格。

事后分解保存在upstream-components.json，全部40条严格满足H16=full_wait+issue_commit+wait1。最大两项分别为stage4 trial8 off：1451=1338+107+6；stage6 trial0 on：1395=1282+107+6。相应producer issue_end分别早于consumer full_wait_begin 826和1945cycle。异常定位于发出之后的consumer full_wait，尚未区分异步完成、轮询与调度，也未证明由观察器导致。该分解不修改原门槛。

复放：在仓库根执行 `python results/gh200_resource_campaign/access_rules/20261011-R13-release-threads-ef-study/analyze-study.py`。analysis_snapshot保存采样前源码；manifest绑定其逐文件SHA，来自175ff7a。最终manifest SHA为482b6557fcbe00a8fc4659bfcca778773a82fe299acf0abbfd1f03b7e77ded74，freeze SHA为98188cb220665bb3bd4a8ca38be282b628ce36a506a154b430b866fb92d705e8。回传包SHA为38c782b9305e94b918d783ca2412949564a0ead959b2dc17f148174b872c7cc0，大小1374814bytes；receipt.json和returned保存资源与原始证据。两次启动失败740059/740060保留且均无时间样本。
