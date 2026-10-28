# request-shape-response（草稿）

主题：单 CTA 请求形状、长度与次序的实测响应。

20 个目录，其中 20 个由别处移入（原族列出）。“记录性质”按目录内是否有 Slurm 日志、receipt、submission、allocation 或回传目录判定；“卡”由文本中的 GPU UUID 识别，未注明为 —；“结论摘录”取自各轮正文，最终版按原记录改写。

| 轮次 | 原族 | 问题 | 记录性质 | 卡 | 结论摘录 |
|---|---|---|---|---|---|
| `20261015-cfg-descriptor-query-001` | Q14/program-graph-export | 实际 CUTLASS tensor-map 查询（对象：实际 launch 参数中的 tensor-map；变量：八候选、两种形状的 setup 查询；端点：rank/shape/stride/box/swizzle 与实际描述符绑定） | 含 GPU 执行 | — | 本地 `replay.py` 重新核对输入清单、返回归档 SHA、全部设备函数、CUDA 枚举、原始 JSONL、launch params 绑定、两种形状的尺寸和步长；结果为 8 个构建、16 次查询、32 个输入描述符全部通过。证据身份在 `replay.json`。本次运行 |
| `20261017-Q02-tma-stride-001` | row-stride | Q02：固定分配内的 cfg_b B 行距定位 | CPU／已有数据 | — | 执行 `python cpu_check.py`：28 组参数检查、固定长度初始化及逻辑坐标检查通过；两轮坐标在 4 种行距下的 1,056 项地址边界检查通过，最大 B 元素下标 9,666,503，小于分配长度 9,666,560。详见 cpu/report.json。 |
| `20261017-Q02-tma-stride-002` | row-stride | Q02：B 行距执行包 | 含 GPU 执行 | a057 | 完整评分、CPU重算、配对摘要、目标审核与传输核对保存。独立留出未启，原12条件完整模型的10.225%／25.422%及独立C2/C3未通过状态保持。 |
| `20261018-Q02-a-stride-001` | row-stride | Q02：A 真实行距的分支判别 | 含 GPU 执行 | — | 原A两窗口及B32增量控制最大误差0.599401%，四条件最大跨进程范围为cycle 0.130163%、ns 0.411523%，均通过原5%门槛。冻结判定量R_cycle=1860/2014=0.9235352532，距地址无关形式的1为0.0764647468，距整个A分 |
| `20261018-Q02-descriptor-factor-001` | row-stride | Q02：A描述符outer维度重分解 | 含 GPU 执行 | — | score.json保存全部原始配对与资格结果，SHA256为 `facaf371ef14b48085d9233b2ad1f7948e4da2af130109f45624569967332c74`；binding.json绑定70个输入及批准构建。cpu-check-r1.jso |
| `20261018-Q02-ktile-heterogeneity-001` | row-stride | Q1四角的Ktile异质性筛查 | CPU／已有数据 | — | `ProgramGemmSource.input()`使用真实line与home缓存服务；两分区负载定价位于`CfgBTensorSource`的分区分支。本次没有更改或调用这些模型。下一项先检验共用非负价格能否把静态特征变成满足四角约束的规则，再决定是否需要历史状态。Q4发出响 |
| `20261018-Q02-operand-split-001` | row-stride | Q02：A／B请求的独立响应分解 | 含 GPU 执行 | — | 缺口是能同时解释独立A/B响应、三请求共存差异、Q4发出反馈及原GEMM stage／行距响应的请求级服务连接。循环内高扰动观察、已停止的线性覆盖／信用／源后FIFO形式不恢复。 |
| `20261018-Q02-request-order-001` | row-stride | Q02：A/B请求顺序 | 含 GPU 执行 | — | score.json SHA256为`a18d4959c71d1b1ac4467c13eb69c87ada1bd6548260591248980d8f09205804`；binding.json绑定73份执行输入。原始输出、完整目标代码与两次失败构建均保存；收口状态见closur |
| `20261018-Q02-single-b-discrimination-001` | row-stride | Q02：单B／双B连接判别 | 含 GPU 执行 | — | 按原预测门重算，单B基础必由软件支主导的余量为−84.761905cycle，故`discriminating=false`，不采留出；两个物理候选均未判胜负。共享与独立的形式区间虽分开，该轮要求的软件支证明未成立。原评分完整保留。 |
| `20261018-Q02-source-permutation-001` | row-stride | Q02：同源集合的Ktile排列 | 含 GPU 执行 | — | 双单位落在原1±0.05带内，结论为本次排列与A16响应不变形式开发相容。未新增服务价格或替换默认组合。逐seed差值先配对再取中位，详见评分；保存输入迁移目录后评分字节一致。独立raw复核见运行后复核，原始执行见runs、作业记录、收口。最小缺证仍为请求准入返回及A/B共存的共 |
| `20261018-Q02-source-step-001` | row-stride | Q02：相邻Ktile源地址状态判别 | 含 GPU 执行 | — | CPU检查通过56项检查、10个反向拒绝、11个评分场景、8组地址枚举，旧prefill评分逐值相同。构建运输130项SHA、执行运输186项SHA全部一致。Slurm记录见作业状态：CPU构建19／22秒，GPU分配7秒。 |
| `20261018-Q02-tma-active-001` | row-stride | Q02：固定暖机的正式 Q1/Q4 判别 | 含 GPU 执行 | a057 | 行距响应依赖正式在途组织；Q4的约295cycle不能直接作为跨Q不变的一次成本或Q1每组价格。尚缺能同时解释Q4长度不变与Q1响应的量化连接，数据就绪和控制推进仍未分开。完整GEMM的C2/C3保持未通过，未替换组合价格。冻结说明及原预测保留在README。 |
| `20261018-Q02-tma-active-length-001` | row-stride | Q1 正式长度的冻结判别 | 含 GPU 执行 | — | 752088已分配参考卡，Popen前因复制二进制缺执行位而停止：`PermissionError: [Errno 13] Permission denied: build/probe`。没有启动probe、没有数值或正式臂；冻结128文件字节保持。按第一次定向实现修复，在新执行 |
| `20261018-Q02-tma-active-length-002` | row-stride | Q1 正式长度的冻结判别 | 含 GPU 执行 | — | 保存stdout的主线程与独立重放评分逐字相同，SHA为`70b1f205f29dbee0844cf1acbf4388fd78ebfd505b17b3013acf0856c0003eb9`；结果见评分，身份见回传核对。本轮条件清单收口，不追加长度、Q或调价。 |
| `20261018-Q02-tma-granule-001` | row-stride | Q02：Q1四行距的冻结响应判别 | 含 GPU 执行 | — | base/B80完整cycle/ns控制最大误差0.048605%，B80配对增量控制误差0.258684%；四条件最大跨进程范围为cycle 0.184609%、ns 0.205339%，均通过原5%门槛。B32对两候选的幅度误差为44.512195%／56.842818%，B |
| `20261018-Q02-tma-length-001` | row-stride | Q02：单 CTA 长度冻结判别 | 含 GPU 执行 | a057 | 156份核心、158份交付绑定与89份返回文件的SHA及长度均核对通过；独立冻结评分与目标评分逐字节相同，SHA为72cc007d636e3be6bd3c93acc2a94586596c0a8d8ba015447eb5c41fbff0ffdb。源码、构建、输入、配对命令、原始st |
| `20261018-Q02-tma-single-cta-001` | row-stride | Q02：单 CTA 源定位 | 含 GPU 执行 | a057 | 本轮无数值幅度冻结，保持开发定位，独立C2=false，按2条件／6进程收口。原始资格及评分见score，回传身份见capture-manifest，单CTA与132CTA逐配对比较见localization及重算脚本。下一项长度判别在PLAN单独冻结。 |
| `20261022-Q02-fullset-order-001` | row-stride | 完整请求集合的次序判别：A80扩展 | 含 GPU 执行 | — | 三布局均在采前1±5%内，保留本次次序不变性。step1／33的A80／A16增长比分别0.888523／0.888694（cycle），ns均0.887850；地址布局响应仍在。G64的step1 A16/A80增长中位13671/12147cycle，约为此前G32的6773 |
| `20261022-Q02-row-phase-invariance-001` | row-stride | Q02：同覆盖计数的A16／A80不变性 | 含 GPU 执行 | — | 采前登记、冻结预测与几何、冻结身份、执行汇总、原始评分、本地回放。复用775198二进制，同一卡、单CTA `[7,6]`、Q1/G32、A16KiB+B0/B1各8KiB、SW128。正式plain不增加打点。三个布局共用alloc_lda4136/alloc_ldb2360， |
| `20261022-Q02-shared-swizzle-001` | row-stride | 132 CTA 源地址与目的组织判别 | 含 GPU 执行 | — | 模型／拟合0，现行模型和原12成绩保持，独立C2及完整GEMM C2/C3仍未通过。后续直接交付输入服务整体替代，以真实GEMM现有反例检验结构与共同参数。 |

## 族内分组（待定稿）

- （按假设或端点分小节，审阅时补）

