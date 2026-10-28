# 共用 plain 前序模块

9 个目录（10-11）：为 R09 及后续实验构建共用的 plain 前序模块（common-sw1-16），完成 host 接入检查与 guarded 读取的构建和记录检查。由 R07-R09 移入；这些是测量协议的准备，不是频率测量。

## 结论

- plain 前序模块在 740042 构建，host 接入经五轮检查，最终作业 740048 完成；协议 `r09-plain-module-sw1-16/v1` 在 setup 与 call 中记录 ABI 与每次 trace 清零字节数。
- guarded 读取的三个 target 由 740070 构建，其记录检查只覆盖数值抽样、trace 协议与资源信息，不是计时或观察资格。

## 轮次

按日期排列；全部结果目录见[Q09-frequency/R09-plain-prefix](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/R09-plain-prefix/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| R09实际CUTLASS guarded读取构建 | 仅cfg_b/stage6目标CUDA12.9 CPU构建，三个target和独立plain-prefix共享库；不执行setup、数值、pilot或留出。三target同1552word/CTA、1638912byte清零布局，共同模块前序。原tile0..9保留，FIRST/MAIN新after | output.tar.gz包含四二进制、SASS、资源/编译日志、冻结清单与资源记录，解到returned。source_hashes.json逐文件校验通过；派生源码可由冻结prepare入口及interface重建，不覆盖历史run。 | [原记录](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/R09-plain-prefix/20261011-R09-guarded-frame-build/README.md) |
| guarded三臂记录检查 | 复用740070三个target和同一plain-prefix模块，每臂一进程、16次共同模块前序后一个目标调用。仅数值抽样、完整trace协议与runtime资源信息检查，不是全D、计时或观察资格。只使用Slurm分配的一张GPU，无跨卡预测结论，不做30进程采样。二进制/模块/源码清单/编译报告 | 复放将output.tar.gz解到returned、analysis_snapshot.tar.gz解到本目录，保留原740070 build目录（含source）后运行python verify.py。复放入口核对采样前manifest绑定的source manifest/build audit与每份source、 | [原记录](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/R09-plain-prefix/20261011-R09-guarded-record-check/README.md) |
| 共用 plain 前序模块：构建检查 | 作业740042在romeo-a057完成，退出0:0、用时69秒，AllocTRES为4CPU/16GiB/1节点，无GPU分配。CUDA12.9.41编译共享模块及原plain程序；未创建有效GEMM上下文或启动GPU kernel，不构成正确性、计时或观察器验收。 | 本地先前失败分别是CUDA13拒绝GCC16、GCC15缺cc1plus，以及绕过检查后的GCC16/glibc头文件错误；没有为通过构建修改系统头文件。受支持的远端构建解决了这一准备阻碍。 | [原记录](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/R09-plain-prefix/20261011-R09-plain-prefix-build/README.md) |
| 20261011-R09-prefix-host-check | — | 见原记录 | [原记录](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/R09-plain-prefix/20261011-R09-prefix-host-check/) |
| 20261011-R09-prefix-host-check-r2 | — | 见原记录 | [原记录](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/R09-plain-prefix/20261011-R09-prefix-host-check-r2/) |
| 20261011-R09-prefix-host-check-r3 | — | 见原记录 | [原记录](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/R09-plain-prefix/20261011-R09-prefix-host-check-r3/) |
| 20261011-R09-prefix-host-check-r4 | — | 见原记录 | [原记录](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/R09-plain-prefix/20261011-R09-prefix-host-check-r4/) |
| 共用 plain 前序 host 接入检查 | 最终作业740048在romeo-a057、GPU-43269fbc-449d-3e0f-908a-9c81229546d3上完成，退出0:0，用时98秒，分配4CPU/16GiB/1GPU。仅plain-host与dual-host各一个正确性检查进程，不是正式性能采样。 | 当前仅完成前序host基础接口。bracket布局/指令包围关系、空间扰动资格与正式观察方案尚未通过；A的30进程机制辨识仍deferred。整体cfg_a/b/c预测目标和V10授权边界不变。 | [原记录](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/R09-plain-prefix/20261011-R09-prefix-host-check-r5/README.md) |
| Plain 前序准备身份 | 协议`r09-plain-module-sw1-16/v1`在setup和call同时记录ABI、每次trace清零字节数以及`trace-memset-device-sync-per-call`规则。case必须显式声明协议和清零字节数，call必须保留16次前序、同buffer标识与16个有限非 | 740050在romeo-a057完成，退出0:0，67秒，仅4CPU/16GiB、无GPU分配。两个host重编成功，协议字符串在制品中；GEMM SASS分别与冻结V09 plain/dual逐条相同。3项新身份测试、18项既有历史测试通过；此轮没有验证新增字段的GPU运行输出或采集时间。 | [原记录](../../../../../../results/gh200_resource_campaign/access_rules/Q09-frequency/R09-plain-prefix/20261011-R09-prefix-identity-build/README.md) |
