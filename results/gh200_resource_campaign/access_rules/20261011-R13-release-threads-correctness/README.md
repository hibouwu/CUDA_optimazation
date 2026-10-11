# R13 全部128个必要 arrival 的逐线程观测

作业740056在romeo-a057 / GPU-43269fbc-449d-3e0f-908a-9c81229546d3完成，134秒，4CPU/16GiB/1GPU。非独占，仅作数值与记录资格pilot；不是原ef卡的40进程研究，不是新独立留出。

协议改为r13-release-threads-v2、magic 0x52313308。CTA0第二输出global group80在local k17的consumer_release前后，每个必要线程256..383记录自己的clock64。未加arrival、同步或轮询。words48:303在工作循环结束后写回128对；原四leader words40:47仍保留，并与对应线程强制逐对一致。off只记录thread256。原v1数据不覆盖、不升级资格。

四leader不能无条件代替128个必要发出。仅补activemask仍不能保证后续指令保持相同执行集合，故本版本直接保留每线程自己的原arrival前后戳；见[NVIDIA说明](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives)。本版本仍需要SASS次序核验，不能把发出区间等同于barrier目的可见或裸硬件时延。

stage4/6 plain/stamped的4次off/on全D检查各覆盖4,325,376个FP32，different=0、nonfinite=0；两stage的plain/stamped-off完整D由compute节点cmp逐字节一致。四份D SHA相同，见returned/full-d-sha256.txt；完整D未回传，原命令、返回码和检查结果保留。

四份common-sw1-16 smoke全部通过严格解码和各4096值参考检查；nonfinite/padding/reference_error均0。开启时两个stage都128/128线程对完整，关闭时仅thread256非零。

| stage | on最大单线程区间cycle | local off→on cycle | H16→wait1 off→on cycle | event on/off−1 |
|---|---:|---:|---:|---:|
| 4 | 25 | 231→231 | 842→836 | −0.5414% |
| 6 | 22 | 231→232 | 356→370 | 0.0000% |

每条件仅一次，全部paired gate整体仍因not_ten_prelisted_trials_per_arm不通过；本批不能消除740054旧v1的368cycle扰动预警。128线程直接界得到post-last-emit到acquire后区间stage4 [206,231]、stage6 [210,232]cycle，仅为本次未合格pilot的条件诊断，不采用机制或价格。

构建SASS中，stage4前戳/arrival/后戳为0x2220/0x2280/0x2290，stage6为0x22b0/0x2310/0x2320，前后同谓词，中间原arrival无该谓词，区间内无分支或写回。结合每线程完整记录，不需假设四leader同步包围整个warp。plain/stamped各stage的静态HGMMA/UTMALDG/UTMASTG/arrival/wait指令数分别16/3/8/6/2；ptxas均报告168寄存器、2 barrier、无spill。这不是运行时物理流量或占用率测量。

build-audit.json、smoke-decoded.json、smoke-intervals.json和thread-diagnostics.json保留完整输入身份、128对、逐条件诊断及门槛结果；B旧四leader入口遇release_threads将拒绝，不静默降级。下一轮须冻结同ef卡、十对/每stage、明确前序与参照身份再判观察资格；原32/128、64/192cycle和2%/5%绝对配对门槛不放宽。
