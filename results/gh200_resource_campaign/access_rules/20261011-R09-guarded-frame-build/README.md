# R09实际CUTLASS guarded读取构建

仅cfg_b/stage6目标CUDA12.9 CPU构建，三个target和独立plain-prefix共享库；不执行setup、数值、pilot或留出。三target同1552word/CTA、1638912byte清零布局，共同模块前序。原tile0..9保留，FIRST/MAIN新after字段10/11；matched写零，guarded调用原候选。此阶段未完成reader及实际读取包围/扰动资格。


740070完成，a057，130秒，实际4CPU/16GiB、无GPU。三个target与prefix都为168寄存器/2barrier，零stack/spill，静态shared1024bytes（不等于总动态SMEM）。四kernel的HGMMA16/TMALD3/TMAST8/arrival6/wait2数量相同。三个原kernel/collective overlay与原归档逐字节相同。

实际guarded FIRST读PC为1a60/1af0/1ba0，MAIN为24e0/2570/2620；前两次读均由高低两半比较及真实BRA控制下一次读，重试上限2。全零退出和两处独立标签保留。完整SASS片段和二进制身份见build-audit.json。此结果只验证编译后的控制流，不证明硬件读取包围、实际精度、动态occupancy、数值或扰动有效；完整reader拒绝链仍未完成，不进入30进程研究或新留出。

output.tar.gz包含四二进制、SASS、资源/编译日志、冻结清单与资源记录，解到returned。source_hashes.json逐文件校验通过；派生源码可由冻结prepare入口及interface重建，不覆盖历史run。
