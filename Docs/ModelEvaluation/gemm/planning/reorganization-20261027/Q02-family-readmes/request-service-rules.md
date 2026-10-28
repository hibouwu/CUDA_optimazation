# request-service-rules（草稿）

主题：单 CTA 请求服务的规则形式筛查（CPU，用已测数据）。

25 个目录，其中 25 个由别处移入（原族列出）。“记录性质”按目录内是否有 Slurm 日志、receipt、submission、allocation 或回传目录判定；“卡”由文本中的 GPU UUID 识别，未注明为 —；“结论摘录”取自各轮正文，最终版按原记录改写。

| 轮次 | 原族 | 问题 | 记录性质 | 卡 | 结论摘录 |
|---|---|---|---|---|---|
| `20261018-Q02-cta-credit-bound-001` | row-stride | Q1本CTA请求信用的闭式筛查 | CPU／已有数据 | — | 原始记录、原plan/score/binding和native CU保存在inputs/prefill；旧sector-bound的score/binding保留同一误差口径来源。plan记录各副本的原路径与SHA，binding绑定本包core。score可从raw独立复算，全 |
| `20261018-Q02-dual-cover-admission-bound-001` | row-stride | 两覆盖占用价与本CTA信用的必要界 | CPU／已有数据 | — | plan先于复算写入，保存范围及全部副本的原路径与SHA；binding绑定14项core。replay仅用Fraction重算raw、覆盖和闭式界，不导入或运行模型，不选择β价格。开发结构排除，独立C2/C3与完整GEMM验收均false；模型调用0、GPU0、拟合0，CPU上 |
| `20261018-Q02-excess-coverage-001` | row-stride | Q02：字节基价与额外行覆盖 | CPU／已有数据 | — | 当前资格为开发约束通过的工作假设。尚需多请求连续执行和cfg_b迁移；未更改默认组合，原12条件plain误差10.224758%／25.422005%及独立C2/C3状态保持。下一项先接通异步源实际准入返回与producer后继，再接入该有限服务规则。 |
| `20261018-Q02-home-join-bound-001` | row-stride | 静态home价格的两分支必要界 | CPU／已有数据 | — | 两个具体静态定价连接停止。抽象的Ktile异质max仍保留聚合相容性；本界没有覆盖含A/B混合临界路径的其他连接。下一项先推导现有两slot源与目的FIFO的Q1路径表达，确认是否已有能改变联合响应的连接，再选择具体规则。 |
| `20261018-Q02-plain-control-allocation-001` | row-stride | Q02：同版 plain 控制分配 | CPU／已有数据 | — | 停止登记的共同控制/独立整请求slot/线性覆盖组合形式。默认cfg_b及原12开发成绩10.224758%/25.422005%保持，完整独立C2/C3未通过。下一项检验共享源工作量这一不同连接。 |
| `20261018-Q02-plain-release-001` | row-stride | Q02 当前源服务隔离回放 | CPU／已有数据 | — | 输入及源码身份见binding，逐次计时见execution，分项及同条件比较见comparison。独立核对确认每条件25344请求／8448组结构保持，全部release等于wait_return；脚本只读保存结果即可复核。冻结runner的summary.model.con |
| `20261018-Q02-postcore-extra-bound-001` | row-stride | Q02：源后共享补齐 FIFO 的必要筛查 | CPU／已有数据 | — | 只复用752146四角和752126的base/B80同 seed 配对记录；B32/B80增量各按 `abs(obs−pred)/pred≤5%` 给价格必要区间。A16增长及A16下B32残留也沿此误差分母，保留A轮原冻结R绝对偏差0.05。所有原始数值与两单位窗口保留在输入 |
| `20261018-Q02-resident-sector-001` | row-stride | Q02 驻留32B覆盖源服务候选 | CPU／已有数据 | — | base/B32复制的原summary保留consumer校准记录64.66924216458013ns；旧冻结worker实际调用consumer0。新评分分别保存 `effective_consumer_ns=0` 与原摘要记录，未改复用输出。原worker摘要的资格文字沿用 |
| `20261018-Q02-return-bound-001` | row-stride | Q02：固定完成尾部与返回边的必要界 | CPU／已有数据 | — | 首次复算错误读取旧raw不存在的request_mask／transaction_bytes，失败保存在attempt-001；一次定向修复只删除该行。原probe无条件发出A与两条B、expect32768，历史raw和计划未改。 |
| `20261018-Q02-sector-admission-001` | row-stride | Q02：32B/128B覆盖的共同源占用 | CPU／已有数据 | — | 含原运行、保存参数重放及独立／迁移核对的总调用见收口；输入身份与core身份保持。没有GPU，原12组合成绩及独立C2／C3状态不变。 |
| `20261018-Q02-sector-bound-001` | row-stride | Q1覆盖线性模型的结构筛查 | CPU／已有数据 | — | 准备阶段纠正两处：原重放路径多写`-plain`而找不到stdout；族定义原先只写总成本非负，允许负截距，反例`c(w)=w−250`在256／288时为6／38，不满足1.125界。最终明确`a_e,b_e≥0`。修改前脚本、plan、binding与score均保留，原始实 |
| `20261018-Q02-shared-throughput-bound-001` | row-stride | Q02：统一共享源吞吐必要界 | CPU／已有数据 | — | 原计划54分支未执行；线性求解、拟合、引擎、GPU、实现修复均0。默认cfg_b与原12条件10.224758%/25.422005%保持，独立C2/C3未通过。下一项先检查地址导出的部分共享与单请求形状响应，见结果与推导。 |
| `20261018-Q02-single-b-general-bound-001` | row-stride | Q02：单B一般增长界与两条件留出 | 含 GPU 执行 | — | 本轮模型／拟合／native修复均0。两个具体候选失败的独立证据保留；默认组合及原12条件plain中位10.224758%、最大25.422005%不变，完整GEMM的C2/C3未通过。下一项只用已有数据检验地址定价：区分对齐搬运基价与行切分新增服务，不预设两者同价；先筛单B、 |
| `20261018-Q02-single-edge-bound-001` | row-stride | Q02：A单边固定前段的必要界 | CPU／已有数据 | — | `verify.py`从保存评分的formal逐条件记录，先同seed相减再取中位，得到A-only／三请求组的6091／6754cycle、3072／3392ns；不使用窗口中位之差。按abs(obs−pred)/pred≤5%，预测区间是[obs/1.05,obs/.95]。 |
| `20261018-Q02-source-admission-001` | row-stride | 共同请求成本与源准入反馈的有限回放 | CPU／已有数据 | — | 初次执行607次引擎调用，其中拟合582次、缺省对照4次、七条件筛查21次，CPU约0.35秒；GPU0。默认cfg_b与原12开发成绩保持，独立C2/C3尚未通过。 |
| `20261018-Q02-source-isolation-001` | row-stride | Q02 当前源服务隔离回放 | CPU／已有数据 | — | 本轮维持已失败基线资格，没有提出或重新授予候选。它收窄了缺口：需要能解释全 local 条件下真实 tensor 行距响应的量化服务；本轮没有辨识其物理来源，也没有新增 C2/C3 验收。 |
| `20261018-Q02-tail-redistribution-001` | row-stride | Q02：公共完成尾部与源驻留重新分配 | CPU／已有数据 | — | 首次执行、保存参数与迁移复算见原收口，根复算见root-replay.json。37项core保持原SHA，追加执行凭据由closure-binding.json绑定。 |
| `20261018-Q02-tma-join-bound-001` | row-stride | Q02：两个非负max连接的开发结构界 | CPU／已有数据 | — | 全部七项沿原候选口径使用`／实测−预测／/预测≤5%`。128B的非负max形式必要下界为47389.473684cycle，超过B80允许上界38183.157895cycle，停止此形式。64B的闭式见证为b=1.658203125、A=31958cycle、C=0；四窗口误 |
| `20261018-Q02-two-stage-bound-001` | row-stride | 串行前段与并行尾部：64B线性覆盖形式被反证 | CPU／已有数据 | — | 因此该非负、无状态、两阶段均按64B覆盖线性计费的形式被反证。下一项需改变覆盖如何定价或状态连接；给此式追加非负串行价格不能修复反例。当前引擎已有共享前沿和三请求完成汇合，未新增同义资源、改默认模型或启动GPU。原12条件plain误差仍为10.224758%／25.422005 |
| `20261022-Q02-byteflow-gather-001` | row-stride | 实际源字节与目的32B片段流水 | CPU／已有数据 | — | 评分保留全部条件，参数保留唯一拟合。局部控制价格沿旧合同，未迁移成GEMM价格。 |
| `20261022-Q02-coverage-prefill-001` | row-stride | 额外覆盖规则：Q1到Q4首次预填迁移 | CPU／已有数据 | — | 评分：Q1初始增量预测0、实测0；Q4预测677.076311、实测513cycle，误差31.983686%，未过5%门槛。四次调用、无拟合/GPU。停止该固定参数的连续两slot形式，Q1的16项开发相容性保留；未接回完整GEMM。复算：`python -B score.py |
| `20261022-Q02-fragment-decoupled-001` | row-stride | Q02：源流与目的行解耦 | CPU／已有数据 | — | 共同控制c565与边界470沿旧动态mask版本迁移，为本次工作假设；失败结论为参数版本停止。Q4分支未执行，未给出跨版本预填判定。 |
| `20261022-Q02-fragment-dual-stage-001` | row-stride | Q02：片段读取与组装双级流水 | CPU／已有数据 | — | 局部共同控制价c565及边界470从旧动态mask版本迁移，是本次工作假设。参数版本的失败包含这项迁移假设，没有证明全部行级流水结构失败。 |
| `20261022-Q02-fragment-stream-001` | row-stride | Q02：源片段与目的行流水 | CPU／已有数据 | — | 共同控制c565与边界470沿旧动态mask版本迁移，为本次工作假设；失败结论为参数版本停止。Q4分支未执行，未给出跨版本预填判定。 |
| `20261022-Q02-publication-queue-001` | row-stride | 软件发布与源服务准入：有界入队返回 | CPU／已有数据 | — | 八次CPU评分完成：容量3/6保留两A行距的Q1源完成及零预填增量；Q4增量分别670.192259/382.686857，对实测513cycle误差30.641766%/25.402172%。停止这两个固定形式，不增加容量值；拟合/GPU0。复算：`python -B scor |

## 族内分组（待定稿）

- （按假设或端点分小节，审阅时补）

