# source-organization（草稿）

主题：D 多 CTA 源 batch 组织与完成跨度。

20 个目录，其中 0 个由别处移入（原族列出）。“记录性质”按目录内是否有 Slurm 日志、receipt、submission、allocation 或回传目录判定；“卡”由文本中的 GPU UUID 识别，未注明为 —；“结论摘录”取自各轮正文，最终版按原记录改写。

| 轮次 | 原族 | 问题 | 记录性质 | 卡 | 结论摘录 |
|---|---|---|---|---|---|
| `20261019-Q02-cta-group-entry-001` | source-organization | Q02 D多CTA独立入口 | CPU／已有数据 | — | 已交付 `cfg_b_tma_chain.timed_cta_group`：将逐CTA的源batch映射、三请求、计算、full返回和empty释放放入同一次既有事件引擎求解。计划、输入绑定、报告及执行统计保留。 |
| `20261019-Q02-endpoint-boundary-001` | source-organization | 源batch端点与输入核对 | CPU／已有数据 | — | 本轮收口：不增加服务候选或价格。下一项设计同一分配、固定源到SM绑定的S4/S6配对，先CPU核对程序及预测，再冻结有限执行；尾部内部观察在该控制成立后按需要设计。原评分及默认GEMM保持。 |
| `20261019-Q02-fixed-source-pair-001` | source-organization | 固定源到SM绑定的stage配对 | 含 GPU 执行 | — | 部分评分：SM绑定两组均超出冻结零带±254.4ns，seed22也超出正带1884.8–2956.8ns；两个冻结区间均未同时符合已有两组，缺第三组完整资格，正式结论为信息不足。固定分配／源到SM绑定后仍观测到stage跨度差。 |
| `20261019-Q02-local-cta-sharing-001` | source-organization | D局部链的CTA共享推进 | 含 GPU 执行 | — | 构建作业756678使用0GPU、43秒。原SSH流和SCP回传失败、共享存储配额失败均保留；经登录节点临时目录取回同一965019B归档，SHA与节点原件一致，未重新构建。 |
| `20261019-Q02-producer-control-input-001` | source-organization | 多CTA producer控制输入 | CPU／已有数据 | — | 本轮补齐独立模块与组合使用同一producer控制输入的能力。实际producer的组内／组间推进和共享受理仍缺局部约束；继续依据匹配程序与端点选择干预，既有同质FIFO的统一gap扫描保持停止。 |
| `20261019-Q02-sm-source-domain-001` | source-organization | SM绑定程序的源到服务域输入 | CPU／已有数据 | — | 报告、执行收口和异目录复放：41输入核验，主回放22、迁移22，共44/48调用，修复／拟合／GPU0；五份预测与时间线逐字一致。 |
| `20261019-Q02-source-batch-001` | source-organization | Q02 固定源batch的共享／独立对照 | 含 GPU 执行 | — | 本轮开发定位已收口，模型／拟合0、实现修复1；默认GEMM及原独立C2/C3保持。下一项先以这组绝对窗口、stage及CTA结束分散约束已有共享源服务，不追加CTA／源规模扫描；真实cfg_b的部分共享与双数学WG连接仍待局部检验。 |
| `20261019-Q02-source-batch-input-001` | source-organization | Q02 源batch输入合同 | CPU／已有数据 | — | 四默认报告逐位保持，23项相关测试通过；地址无关服务保留相同端点，地址相关合成服务在同步／异步接法均给出12／667 ns的首组就绪端点。报告及异目录复现逐字一致，48输入保持。累计62/64次引擎调用、两次夹具修复；第二次修复保留原审计预期漏算发出间隔的反例。GPU／拟合／硬件 |
| `20261019-Q02-source-batch-service-001` | source-organization | 固定源batch的现用服务迁移 | CPU／已有数据 | — | 本目录执行`PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python -B run.py`。异目录新进程重建四条件及完整时间线，报告和压缩时间线逐字一致；41输入保持。主回放44次、迁移复放44次，共88/96引擎调用，修复／ |
| `20261019-Q02-source-count-build-001` | source-organization | 66／132源batch同binary构建 | 含 GPU 执行 | — | 输入与返回的完整身份见构建输入、构建命令、编译错误、作业状态和执行记录。模型调用、拟合和GPU均为0；默认GEMM及完整独立C2/C3保持未通过。 |
| `20261019-Q02-source-count-prediction-001` | source-organization | 固定分配的66／132源batch预测 | CPU／已有数据 | — | 两顺序的窗口响应均下降，按两个窗口各5%误差传播，预测响应区间为−60.555888%至−50.681226%，与零响应的±5个百分点可分。两排序都保留。本轮没有目标实测；当前源规则的132源绝对窗口仍失败，现有结论是可供新局部干预检验的冻结预测。 |
| `20261019-Q02-source-footprint-screen-001` | source-organization | D、E与真实cfg_b的源工况覆盖 | CPU／已有数据 | — | 首版直接把冻结E的有效输入当成原始case，被当前解析器以`undocumented fields: gpu_uuid, grid, scheduler_sm_count`拒绝。一次定向修复按原E程序入口字段重建case，setup及其余22项原输入保持；原脚本和失败空输出保留。 |
| `20261019-Q02-zero-time-order-001` | source-organization | 零耗时控制的同刻排序核对 | CPU／已有数据 | — | **结论：同刻排序显著影响多CTA轨迹，硬件排序规则仍信息不足。** 已保存seed3/22观测的S6−S4结束跨度差为2592/3840ns，替代排序的响应方向仍相反。默认引擎保持；下一连接缺口是producer请求推进与跨CTA受理次序，现有结果尚未区分软件发出时间和共享准入 |
| `20261020-Q02-data-gated-service-001` | source-organization | 数据准备约束共享服务准入 | CPU／已有数据 | — | 停止该固定连接，未运行66源、另一顺序、S4及完整GEMM。该形式虽实际推迟了共享服务进入，仍未解释已登记的绝对窗口和空间推进；不改变原模型，不增加价格。没有新增独立C2／C3成绩，GPU／拟合0。 |
| `20261020-Q02-first-full-span-001` | source-organization | 首个full完成的多CTA跨度 | 含 GPU 执行 | — | 本轮排除的是当前观察实现的进入资格，未排除硬件顺序。剩余缺口是132／66源组织下同代首full完成的跨CTA区间；不得用模型预测替代这项观测。该观察器不继续扩展，后续先按现有模块与显式假设交付同版cfg_b组合，并保留未解释的绝对窗口、CTA分散和正式C2/C3缺口。 |
| `20261020-Q02-late-slot-observer-001` | source-organization | 后续槽代次旁路观察的CPU可行性 | 含 GPU 执行 | — | 当前有限构建按原一次远端上限收口，结论为实现未完成，见report.json。模型／GPU／拟合0；原输入服务、全部价格和完整组合保持。下一项缺失检查是修正后开启版的编译、循环与资源保持，尚无硬件观察成绩。 |
| `20261020-Q02-source-count-build-002` | source-organization | 66／132源batch：构建接续 | 含 GPU 执行 | — | 该固定源状态服务对这次源并集干预响应获得限定支持，零响应形式被反证。当前模型的绝对窗口和CTA结束分布仍有缺口，详见同条件比较：实际132／66源的结束跨度中位为4672／1696 ns。整体同移不能改变这一跨度，下一项需约束共享服务推进或完成边界，而不是添加共同前缀来修正分散。 |
| `20261021-Q02-late-full-span-001` | source-organization | 132源g11的跨CTA full完成跨度 | 含 GPU 执行 | — | 每CTA的full界宽最大160–192ns，无漏过失败test的CTA。上述观察区间均高于两种保存预测，但目标跨度的扰动／桥接资格未过，**结论信息不足**。停止本轮观察，不补采、不改比较带，未选择共享源排序。 |
| `20261021-Q02-late-slot-cycle-observer-001` | source-organization | g11 相位的 cycle-only 观察 | 含 GPU 执行 | — | 另从保存的132/66源时间线直接提取后续g11的跨CTA full完成跨度，模型求解0。132源两顺序分别为1197.122/2979.044ns，差1781.922ns；66源为1047.545/2095.089ns。见下一项证据筛查。这比本轮局部87.449cycle间距更 |
| `20261021-Q02-late-slot-switch-001` | source-organization | g11 同二进制观察：采前预测核对 | 含 GPU 执行 | — | 本轮未改变模型参数或默认版本，未重跑原12条件，独立组合 C2/C3 仍未通过。 |

## 族内分组（待定稿）

- （按假设或端点分小节，审阅时补）

