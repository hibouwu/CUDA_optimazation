# request-preparation-return（草稿）

主题：请求准备、发出受理与返回反馈。

14 个目录，其中 14 个由别处移入（原族列出）。“记录性质”按目录内是否有 Slurm 日志、receipt、submission、allocation 或回传目录判定；“卡”由文本中的 GPU UUID 识别，未注明为 —；“结论摘录”取自各轮正文，最终版按原记录改写。

| 轮次 | 原族 | 问题 | 记录性质 | 卡 | 结论摘录 |
|---|---|---|---|---|---|
| `20261018-Q02-async-admission-001` | probe-control | Q02 异步源准入返回接线 | CPU／已有数据 | — | 本轮修复执行连接，未替换源规则或改变默认预测。原12条件plain误差中位10.224758%、最大25.422005%保持。下一项将已保留字节／额外覆盖价格用于连续请求服务；探针控制分配与多CTA共享范围分别用已有证据约束。 |
| `20261018-Q02-issue-return-001` | probe-control | Q02：同步源服务的发出返回连接 | CPU／已有数据 | — | 第一条命令执行3次合成调用，第二条执行22次系统引擎调用，结果见输出的comparison.json；原回归报告和收口保留本轮实际执行计数。未修改冻结或历史记录，没有commit或push。 |
| `20261018-Q02-queue-return-001` | probe-control | Q02 队列受理返回与源开始返回 | CPU／已有数据 | — | 当前只保留“排队受理返回＋两槽在服PS＋共同覆盖价”的这一开发版本。下一项最小观测是同设备代码Q4基础/A16配对：冻结新响应后，用独立数据检验跨组行距迁移。多CTA共享范围及与域缓存分支的连接仍待检验。默认cfg_b及原12条件plain中位10.224758%、最大25.42 |
| `20261020-Q02-admission-feedback-replay-001` | source-organization | 状态化源受理返回对producer的反馈 | CPU／已有数据 | — | 两项均失败，按计划停止，未运行其余三组、S4或完整GEMM。原服务与当前完整版本保持。保存预测、末调用时间线及评分保留全部结果；没有新增独立C2/C3结论。 |
| `20261020-Q02-producer-control-contract-001` | native-d-chain | producer程序控制与源准入的接线 | CPU／已有数据 | — | 剩余缺口为软件控制、请求受理与就绪的量化分工；本实现停止。原12条件误差与全部失败保持，完整目标未完成。 |
| `20261021-Q02-control-placement-intervention-001` | native-d-chain | 控制放置的生产者节流判别 | CPU／已有数据 | — | 先以有限计划筛查：固定S6/K13和已有两个解释，仅把组间隔0改为1024cycle。八次CPU调用、拟合与GPU为0；响应区间不分则收口，不开展实机采样。 |
| `20261021-Q02-preparation-delay-001` | native-d-chain | empty 后准备延迟对照 | 含 GPU 执行 | — | **资格通过，冻结的校准形式失败；S4 未采，未选择两个控制解释。** 原始评分的信息不足保留，停止本具体校准形式。接下来先用已有源码和本轮响应，检验原有 producer 准备工作是逐轮推进还是只出现在启动段；输入受理／就绪与计算／释放固定，从 CPU 约束开始。原 12 条件 |
| `20261021-Q02-preparation-length-001` | native-d-chain | 逐轮准备的独立长度外推 | 含 GPU 执行 | — | 关闭臂 N0 相对 R 的差随长度增长：K4为22–58、K13为91–128、K32为296–398 cycle。当前固定J不能同时解释这三档基线差。独立长度证据支持逐轮复合响应，P与源残量的分配仍未单独辨识；完整组合C2/C3保持未通过。 |
| `20261021-Q02-program-preparation-001` | native-d-chain | 程序准备与请求受理接线 | CPU／已有数据 | — | 单元测试复核：`PYTHONPATH=microbench/gh200_resource_campaign/access_rules python -B -m unittest test_cfg_b_tma_chain test_cfg_b_request_model`。保存报告 |
| `20261021-Q02-recurrent-preparation-001` | native-d-chain | 原有程序准备的逐轮放置 | CPU／已有数据 | — | 只把自然准备放在首段时，在同一 J／δ 域内有 `Δ13≤12·129+13·641−6144=3737`，低于观测下界6617；该具体分配被排除。 |
| `20261021-Q14-producer-preparation-transfer-001` | Q14/gemm-connection | 逐轮准备接回G3与A16（对象：producer 请求准备/形成/返回；变量：准备价格、形成与缓存先后、返回反馈；同地址评分修正；端点：请求 release/早晚窗口，或接回 GEMM 的 L/plain 响应） | CPU／已有数据 | — | 正式四组共44引擎调用，102.027秒，GPU／拟合0。首次尝试在末次求解后因保存源码漏掉延迟导入的`sm_residency.py`而退出，消耗11调用；补齐两份原样依赖后按同一条件重试，见故障记录。累计55调用，其中11为实现失败；原失败输入保留在failed-import |
| `20261023-Q14-address-matched-return-001` | Q14/gemm-connection | 请求返回的同地址比较（对象：producer 请求准备/形成/返回；变量：准备价格、形成与缓存先后、返回反馈；同地址评分修正；端点：请求 release/早晚窗口，或接回 GEMM 的 L/plain 响应） | CPU／已有数据 | — | 本轮为开发证据更正。原12条件与独立 C2/C3 验收状态保持，旧形成轮结果原样保留。 |
| `20261023-Q14-formation-before-cache-001` | Q14/gemm-connection | 共享请求形成先于缓存查询（对象：producer 请求准备/形成/返回；变量：准备价格、形成与缓存先后、返回反馈；同地址评分修正；端点：请求 release/早晚窗口，或接回 GEMM 的 L/plain 响应） | CPU／已有数据 | — | 整轮CPU复现入口为`python -B "$study/run.py" --output /tmp/formation-round-replay`，同样执行有限标定。六份参与修改的模块在source保存；完整时间线、端点及输入位于output。相关161项CPU测试通过。 |
| `20261023-Q14-formation-return-001` | Q14/gemm-connection | 形成服务的返回反馈（对象：producer 请求准备/形成/返回；变量：准备价格、形成与缓存先后、返回反馈；同地址评分修正；端点：请求 release/早晚窗口，或接回 GEMM 的 L/plain 响应） | CPU／已有数据 | — | 结论：unique Q1 每CTA只有 1 组在途、源各自独立（合并片段 0），时间由逐组往返决定：实测 release／64 组约 675–750 ns，基线预测约 571–588 ns；单请求早／后段窗口实测 528–648 ns，预测 422–458 ns。缺口在低在途时的 |

## 族内分组（待定稿）

- （按假设或端点分小节，审阅时补）

