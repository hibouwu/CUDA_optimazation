# cfg_c独立MMA参考：源码与构建

05700b2冻结派生源码，740094在a057以4CPU/16GiB、无GPU分配编译。原r00_wgmma.cu保持不变；新prepare入口生成固定FP16 SS N128、B MN-major、两WG、每WG两累加链×4MMA/组，wait1及最终wait0。只允许单CTA、64或1024组，分别对应1024或16384条全CTA MMA；固定16次同探针前序，不按CV重试。目标设备UUID在执行前检查为参考ef。

841份源SHA及原输入/返回冻结链通过。nvcc编译与cuobjdump成功，主循环1650处ARRIVE、1670..16f0八条HGMMA.64x128x16.F32 tnspB（R88/R24各四）、1700 wait1、1710回跳、1720最终wait0。短坐标检查kernel每链一条，共两条，保留1/2轮坐标参考。两kernel均154寄存器/1barrier/1152B静态SMEM，动态32768B，零stack/spill。

这不是完整CUTLASS计算循环：操作数驻留SMEM并重复使用，两个累加链使用相同描述符；没有producer、TMA、stage环或cluster通信。编译器在循环头插入ARRIVE。计时从初始同步附近到最终wait0后的checksum/drain/CTA同步，边界比GEMM FIRST/MAIN宽，不减去猜测开销。因此64组只检查短窗口量级，1024组检查持续参考，不使用两点拟截距自动修补GEMM。

当前只有静态构建证据：未执行GPU短检查、全输出参考或计时。下一步固定两个长度各三个独立进程，参考ef独占执行、保存完整output.f32与raw/stamps；先独立数值复核，再报告单CTA native cycle/FLOP。不得扩大为整GEMM服务识别、跨卡转移或原观察资格通过。

复核：`python audit-build.py`。build-audit.json绑定binary/SASS。返回归档与冻结输入完整保存。
