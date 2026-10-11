# 共用 plain 前序 host 接入检查

最终作业740048在romeo-a057、GPU-43269fbc-449d-3e0f-908a-9c81229546d3上完成，退出0:0，用时98秒，分配4CPU/16GiB/1GPU。仅plain-host与dual-host各一个正确性检查进程，不是正式性能采样。

两个进程均用同一共享库，并向前序与目标传入各自同一组A/B/D地址。模块/workspace、trace及event handles在输入填充前创建；16次plain前序与1次目标按默认stream执行，保留16个调用间隔。实际grid为[1,132,1]、stage6、132CTA。二者4096个dyadic抽样值均精确正确、非有限值0、padding错误0；原解析器再次复放，dual解出132CTA。尚未做前序结束后单独的全D检查，不将此次抽样写作全输出正确性证明。

共享库和plain-host的GEMM SASS为1280条，dual-host为1536条，均16HGMMA。plain/dual分别与冻结V09对应kernel逐条一致；前序库与plain亦一致。该证据检查设备指令，不证明新host调用历史等同旧adaptive历史。

[运行与复放摘要](runtime-summary.json)、[制品哈希](artifact-sha256.json)、[构建检查脚本](run.sh)、[分配启动参数](command.json)、[源码输入哈希](input-sha256.json)、[回传身份](receipt.json)保留当前版本。原始输出的gzip副本在returned/plain.jsonl.gz和returned/dual.jsonl.gz。完整输入包、输出包、二进制和SASS保留在同目录，远端临时盘不是唯一副本。

## 保留的准备失败

- 740043，原prefix-host-check：dual的overlay include在CUTLASS原头之后，编译保护拒绝；未启动GEMM。
- 740044，prefix-host-check-r2：旧overlay trace与当前probe trace重复定义；未启动GEMM。
- 740046，prefix-host-check-r3：模块grid保护拒绝；未启动GEMM。
- 740047，prefix-host-check-r4：补显式Heuristic后仍拒绝；原始setup复核表明提案及初版保护把[1,132,1]写反为[132,1,1]，不是CTA总量问题；未启动GEMM。
- 740048，当前r5：统一trace来源、固定Heuristic、按原setup逐维核对prefix/target grid后通过。

各失败的输入/输出archive、receipt和returned日志分别留在相邻结果目录。失败不合并进成功版本，也不重写为有效样本。

当前仅完成前序host基础接口。bracket布局/指令包围关系、空间扰动资格与正式观察方案尚未通过；A的30进程机制辨识仍deferred。整体cfg_a/b/c预测目标和V10授权边界不变。
