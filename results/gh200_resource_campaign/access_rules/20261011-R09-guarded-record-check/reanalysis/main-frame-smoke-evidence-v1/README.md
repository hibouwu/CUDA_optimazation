# 实际编译帧、运行记录与当前reader的证据连接

主集成审查两个A审计程序后重跑，再运行当前reader的原raw/破坏副本回放。`replay.py`保存三个程序的SHA、输出与报告SHA，原冻结snapshot不修改。

实际CUDA12.9编译帧：860份源文件与四个binary/SASS身份通过；原三个overlay逐字节相同。四kernel均168寄存器、2barrier、零stack/spill；consumer/producer寄存器池232/40。FIRST和MAIN各自保留两半比较、真实BRA、最多两次读取、全零失败出口，thread128/256写者与字段偏移通过逐指令检查。静态shared为1024B。

740071运行记录单独给出动态shared215040B、epilogue18432B，API最大CTA/SM=1，设备与scheduler均132SM。12288个不同抽样位置重新精确验证，真实头字段与角色端点有效。API资源上界不等于实际执行占用率；本次没有全D验证。

A对原冻结reader的10个非法副本接受结果仍可复现；主线加严reader将相同10个副本全部拒绝，真实matched/guarded各264角色仍有效。这是两个明确版本的结果，不能互相覆盖。

不调用原区间helper的Fraction检查，与主集成132个CTA交集和最小预算逐项一致：原128ns总区间预算下0/132可行；所需最小预算min/median/max为1149.598185/1212.217542/1242.163148ns。这仍以硬件读取包围成立为条件。

编译路径和资源一致不能证明动态耗时一致；一调用/臂不能资格化局部或空间扰动。硬件读取包围、有效局部/关键CTA观测、新留出和跨卡验收均未完成。保持observation_qualified=false，不由这份证据连接自动启动30进程研究或V10。
