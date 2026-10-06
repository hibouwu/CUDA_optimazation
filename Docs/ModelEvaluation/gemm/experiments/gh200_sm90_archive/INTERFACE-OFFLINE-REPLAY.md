# 有界浮点比较的只读后审接口

状态：A 草案，未实现。规则见 [offline_replay_v1.json](../../../../../microbench/gh200_resource_campaign/contracts/offline_replay_v1.json)。本补充遵守 [GOAL.md](GOAL.md)，不修改旧 raw、summary、snapshot、完成标记或历史门禁，也不发起 GPU 重测。

## 已发现的问题

ROMEO 作业 730217 的 memory/FMA/MMA/WGMMA 冻结 preflight 在本地重放时，分别有 1/6/3/2 个 `warmup_cv` 的 binary64 末位不同；10 项差 1 ULP、2 项差 2 ULP。病例数为 14/93/55/85，pilot 数为 0/36/24/24。其余比较字段通过不表示整个 B 自动通过；只在后审逐项重新验证后认可证据。原严格比较失败也作为事实保留。

## 比较规则

允许误差的字段必须同时满足：属于下表白名单、两端均为有限非负 binary64 浮点数、ULP 距离不超过 2。零只与精确的正零相等，拒绝负零、NaN、无穷及正数与零之间的近似接受。禁止用统一的相对/绝对容差递归放宽整个 JSON；未列字段按原语义和类型精确比较，整数不能以浮点数或 bool 替代。

| 比较对象 | 允许不超过 2 ULP 的派生字段 | 保持严格的字段 |
|---|---|---|
| `preflight_summary.json` 的 `cases[case_id]` | `value`、`warmup_cv` | case 集合、单位、覆盖数、`warmup_converged`、binary 哈希及非硬件资格标记 |
| 正式 `summary.json` 每 case 的 `batches[*].stats`、`merged` | `mean`、`median`、`min`、`max`、`cv` | 样本数、批次编号、样本次序、case 状态、导出资格 |
| 正式 summary 的 `batches[*].samples[*]` | `value`、`warmup_cv` | PID/PGID、主机时间、raw/receipt 路径和哈希、case/batch/trial、单位、SM 数、布尔标记 |

raw 的 `event_ms`、时间戳、操作数、工作量和资源字段不属于“与另一份 summary 近似相等”的范围，必须依原合同解释与验证。源/contract/设备/binary/SASS/收据哈希、pilot 参数、解析后的整数迭代数、任何状态与错误计数严格核对。JSON 字段集合按已识别 schema 验证；未知字段不能因不在白名单而被忽略。

阈值决策独立重算，不给阈值增加 2 ULP：

- 预热使用合同规定的真实整数窗口及样本 CV。令 `n` 为窗口数、`S=Σx`、`Q=Σx²`，`CV²=n(nQ−S²)/((n−1)S²)`；以整数交叉乘法精确判断 `CV≤1/50`，同时验证首个合法收敛窗口及 8–30 窗限制。
- 正式主指标由整数 raw 工作量/时间和合同十进制 scale 构造精确分数，独立计算同一 CV²，判断 `CV≤1/20`。不使用已记录的 rounded rate 或 CV 决定是否需要重测。
- 旧布尔/状态若与独立决策不符，后审拒绝并交独立 reviewer 判断边界原因。不得借派生字段容差翻转通过状态，也不自动覆盖旧状态或触发 GPU 重测。

该后审保留旧 binary64 派生数的生成公式，只改变白名单字段的比较；不另行量化、裁剪或重写记录。未来更换统计算法属于新的规则版本。2 ULP 不是任意架构、语言或统计库的普遍精度保证，超界即拒绝。

## 只读入口与收据

拟新增独立 `audit_replay.py RUN --policy POLICY`。默认只向 stdout 输出机器可读收据；可由调用方保存到原 run 外的新后审目录。工具不得在旧归档中写锁、缓存、字节码或任何修复文件。

入口先验证完整旧快照及身份，再在隔离的 Python 进程中加载受审冻结审查逻辑，重算原始证据；受审重放适配层负责取出尚未进行最终 float 相等比较的结果。它不得修改旧文件、猴子补丁原比较函数、伪造内存数据使旧工具返回 pass，或跳过原有数值、工作量、SASS、进程、覆盖及哈希检查。不支持的旧 schema/审查接口必须拒绝或先补受审适配。

输出 `replay_receipt.json` 的逻辑内容包括：规则 ID/规则文件哈希、后审代码及依赖哈希、重放环境、输入 run 的相对身份、完整输入工件清单与 SHA256、旧严格工具实际结果及日志哈希、逐字段差异（JSON pointer、记录值、重算值、ULP 距离）、阈值独立重算结果、未通过检查和限定结论。工件清单必须覆盖 run_spec、snapshot manifest 及全部快照文件、binary、SASS/build、device/environment、raw、receipts、待比较 summary、适用时的 measurement manifest/COMPLETE/C review；收据不能仅绑定一个可被重新生成的 summary。

成功结论只为 `pass_under_bounded_float_replay_v1`，不伪称“原冻结审查器严格比较已通过”，也不直接授予 family B/C。失败时保存 `replay_rejected` 与准确原因，禁止静默略过差异。

## 门禁与正式导入

本设计取得独立 `S02/offline-replay-A` 通过后方可实现；后审入口、规则和负例取得独立扩展 B 通过后方可用于认可旧证据。历史 S02/S03 B 记录和受审源保持不动。

family B 可直接绑定后审收据、该收据的完整输入清单及所依赖的审查证明；基础数值/计量/SASS等仍逐项审查。新版本正式校准导入显式选择 `evidence_mode=bounded_float_replay_v1`，同时核对独立扩展 B、family B 对收据的直接绑定、收据中的完整旧工件身份，以及复用 binary/pilot/resolved 的原严格身份约束。它必须记录原严格审查失败和采用后审规则的原因。

旧版正式导入仍执行原冻结规则；本补充不让它在后台自动绕过失败。需要后审模式的导入由新的受审扩展入口承接，其新收据引用原 binary/SASS/build 并明确没有重编译。整数迭代数、设备/工具链身份和编译依赖不因浮点比较政策改变而放宽。

## 接受真值表

A/B 检查必须覆盖：白名单字段 0/1/2 ULP 接受、3 ULP 拒绝；非白名单 float 与所有身份/整数/哈希/布尔变更拒绝；正零/负零/NaN/无穷；阈值两侧即使相距 1 ULP 也不能翻转旧状态；缺 raw、损坏收据、单位变化、少样本及未知 schema 拒绝；旧目录只读且搬移后结果可重放。现有 12 个真实差异是回归样本，不能代替负例。原 run 文件的前后哈希必须相同。
