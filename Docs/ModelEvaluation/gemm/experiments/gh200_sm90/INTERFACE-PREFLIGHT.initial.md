# S02-B 最小设备检查补充

本补充不修改 S02-A 的已审接口。新增 `run --preflight`，用于 family B 阶段取得编译、SASS 和最小 GPU 正确性证据。它读取 S02-A 及 family A 的独立门禁，编译被冻结依赖，每 case 运行一次独立进程并保留相同预热诊断。框架 S02-B/S03-B 与 family B 尚未通过时，正常 `run` 必须拒绝正式采样。

preflight 使用独立全新 run 目录，`run_spec.kind=preflight`，不创建正式 batch 或 COMPLETE；完成状态为 `preflight_pending_review`。它可以报告预热未收敛，但数值、计量或 SASS 错误必须失败。不能通过 resume/finalize 把 preflight 转为正式采样；正式运行另起目录并读取 B 门禁。

正常运行每次只推进一个 case-batch 轮次（每个当前待测点最多 10 次独立进程），符合单次 Slurm 10 分钟限制。未收集完或仍需重测时保留状态，由下一有效 Slurm 分配调用 resume；已收集的同协议 batch 和冻结 binary 继续使用。中断重启不覆盖已有尝试。终态、计数器检查和 C review 的接受规则仍按 INTERFACE.md。

补充审查只授权这一最小检查入口和分配间恢复，不提供任何 GH200 数值或性能资格。CPU fixture 不具备正式硬件资格。
