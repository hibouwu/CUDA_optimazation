# S02-B 最小设备检查补充

本补充不修改 S02-A 的已审接口。新增 `run --preflight`，用于 family B 阶段取得编译、SASS 和最小 GPU 正确性证据。它读取 S02-A、补充设计及 family A 的独立门禁；只豁免尚未存在的 family B 运行资格，ROMEO、有效单 GPU Slurm、GH200/SM90、CUDA12.9、双锁、冻结快照和进程组超时清理要求全部保留。无 GPU 时只能保存静态证据，不能运行设备检查或声称完整 preflight。它编译被冻结依赖，每 case 运行一次独立进程并保留相同预热诊断。框架 S02-B/S03-B 与 family B 尚未通过时，正常 `run` 必须拒绝正式采样。

preflight 使用独立全新 run 目录，`run_spec.kind=preflight`，不创建正式 batch 或 COMPLETE；完成状态为 `preflight_pending_review`。它可以报告预热未收敛，但数值、计量或 SASS 错误必须失败。不能通过 resume/finalize 把 preflight 转为正式采样；正式运行另起目录并读取 B 门禁。

10 分钟是整个 Slurm job 的总预算。runner 查询当前 job 的 EndTime/剩余时间，编译前预留 180+20 秒，设备进程前预留 120+20 秒（20 秒用于清理和 checkpoint）；预算不足立即保存 checkpoint 并释放锁，不启动该进程。resume 在新的有效分配继续同 case、同 batch 的缺失 trial，不增加统计批次数。是否完成一批由 10 个独立进程收据决定，不能把一次 Slurm 分配等同于一个统计批次。所有已采样、失败及中断证据保留，严禁覆盖或挑样。preflight 中断也允许用 `resume --preflight` 延续冻结 binary 和缺失 smoke case，但不能转正式采样。终态、计数器检查和 C review 的接受规则仍按 INTERFACE.md。

补充审查只授权这一最小检查入口和分配间恢复，不提供任何 GH200 数值或性能资格。CPU fixture 不具备正式硬件资格。


审查修复：独立 S02-preflight-A 的环境约束与 job 总预算两项发现已据此补充。初稿保存在 INTERFACE-PREFLIGHT.initial.md；原审查意见保留，须由原独立 reviewer 复核通过后使用新增入口。
