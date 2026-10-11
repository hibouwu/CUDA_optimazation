# 累计观察反馈的四模式探针

从固定累计等待prepare3源manifest `b57bbb98…` 派生；新协议`r13-wait-feedback-v1`，marker`0x5231330a`。固定cfg_b 2816×1536×4096、stage4/6、CTA0/thread256、CTA输出1。host模式0/1/2/3对应CUDA常量mask 0/0x1fffc/4/65536；循环按group位选择同一个H→try/wait→B→record路径。

原sum/count/实际mask/firstH/lastB/invalid保留，word40声明mask，41/42新增Q2/Q16。只有ALL15产生Q_early及R_early；ONLY2/ONLY16的sum只代表本组选中等待，完整Q/R为None，未选中单点也为None。解析器检查完整包含链，Q16用新H/B差值，不能拿旧progress的H16替代。所有值为同线程原生cycle，组件ns未识别。

用ALL15与ONLY16的Q16比较此前14次和0次观测bundle；用ALL15与ONLY2的Q2作没有前序累计操作的对照。bundle包含H/B、校验、累加和汇合，不是纯算术成本。不同stage运行各自binary的16次off前序；不称相同跨stage历史，也不把新binary与旧批次混池。

CPU-only job740367在CUDA12.9.41编译成功，4CPU/16GiB/无GPU；两个GEMM均REG168、零spill、16静态HGMMA。共核858源文件，只有声明的5个旧文件改变，kernel/pipeline保持。stage4的H/try/wait/B地址为1ed0/1f20/1f40/1f80；stage6为1dd0/1e20/1e40/1e80。快/慢成功均到同一个B，mask选择是统一路径；B后collector至汇合53条静态指令，无循环GMEM/local/shared store。静态条数不是周期成本，同地址不证明相同调度。

5项新CPU测试覆盖四模式×两stage×两个大整数clock原点、严格mask/包含/未观测拒绝、四包数值检查及实际C++位选择/累计代码。首次准备误把协议字符串出现次数3写为2，在编译前停止，根目录部分source保留；prepare2为实际构建。SASS检查脚本首次假设两stage的group均在UR17，实际stage6为UR21，脚本修正后通过，binary未改。

完整证据在[build-audit.json](build-audit.json)、[CPU检查](cpu-tests.json)与两个SASS摘录。源码准备快照中的compiled=false保持原样；编译完成由build-audit单列。尚无硬件正确性、观察或预测资格。
