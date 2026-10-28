# issue-placement-interference（草稿）

主题：TMA 发出位置与本地计算背景的干扰。

7 个目录，其中 7 个由别处移入（原族列出）。“记录性质”按目录内是否有 Slurm 日志、receipt、submission、allocation 或回传目录判定；“卡”由文本中的 GPU UUID 识别，未注明为 —；“结论摘录”取自各轮正文，最终版按原记录改写。

| 轮次 | 原族 | 问题 | 记录性质 | 卡 | 结论摘录 |
|---|---|---|---|---|---|
| `20261015-tma-compute-development-001` | gemm-source-candidates | TMA 与同 CTA WGMMA 的并发开发对照 | 含 GPU 执行 | — | 8 条件×trace 开关×7 进程，共 112 个开发采样。原生构建后先做 12 个正确性条件、memcheck 与 synccheck；失败则停止该次作业。候选辨识用途与资源上限在 plan 中登记，正式 C2 预测将在开发之后另行冻结。 |
| `20261015-tma-issue-placement-development-001` | multi-cta-gpc-sharing | TMA 发出者与背景 warp 的位置对照 | 含 GPU 执行 | — | SASS把源码每轮64个加法折叠成四条`VIADD`（步长16、48、80、112）；实际背景包含这四条、轮数加法、globaltimer读取、比较与shuffle。16项数值预检覆盖四个issuer与四个背景warp，2,359,296个TMA元素和全部背景校验通过，memch |
| `20261015-tma-issue-placement-heldout-001` | multi-cta-gpc-sharing | 位置留出执行失败 | 含 GPU 执行 | — | 第二次执行保留完全相同的输入包、冻结时间、二进制、条件与阈值；只将本次环境快照写到独立的`execution-*.txt`。原失败输出和receipt保留。 |
| `20261015-tma-issue-placement-heldout-002` | multi-cta-gpc-sharing | 更换发出 warp 的独立留出 | 含 GPU 执行 | — | 第一次执行746731在采样前被环境文件覆盖的绑定校验中止，见失败记录。第二次的输入包、冻结时间、二进制与预测完全相同；只修正执行快照的位置。完整冻结判据与评分见plan、score。归档SHA256：`b9e0b770cc49709c62fbb52efb92eafbaf5982 |
| `20261015-tma-local-interference-heldout-001` | multi-cta-gpc-sharing | 固定远端计算背景，开关本地 WGMMA | 含 GPU 执行 | — | 四个计时外原生数值检查均通过，TMA及背景累加错误为零。采样前冻结文件与逐进程SHA均通过复核。第一次SSH启动早于上传结束，driver尚不存在，未启动Slurm或采样；完成上传后原样执行。 |
| `20261015-tma-local-rs-development-001` | multi-cta-gpc-sharing | 本地 WGMMA 的 SS/RS 开发对照 | 含 GPU 执行 | — | 八项计时外TMA数值检查及全部背景累加值检查通过，memcheck与synccheck均零错误。正确性明细见preflight-summary.json，资源和SASS见构建目录。原始结果、重复进程与控制分项见分析。归档SHA256：`9d8b43a8ffde0f8add1e97 |
| `20261015-tma-local-rs-development-002` | multi-cta-gpc-sharing | 复用 A 寄存器后的 SS/RS 对照 | 含 GPU 执行 | — | 八项计时外数值检查、全部背景累加值、memcheck和synccheck通过，见正确性明细。源代码、二进制、逐进程结果和分项保留在returned与分析。归档SHA256：`4dd386585d6e5f8905907dea4b2886ac688c6c91e0b3814c60ae0 |

## 族内分组（待定稿）

- （按假设或端点分小节，审阅时补）

