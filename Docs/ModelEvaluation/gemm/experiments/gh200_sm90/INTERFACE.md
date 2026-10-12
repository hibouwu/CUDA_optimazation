# S02-A 总控与探针接口

状态：设计待独立审查。机器可读约定见 [interface.json](../../../../../microbench/gh200_resource_campaign/contracts/interface.json)，采样数与阈值以 [protocol.json](../../../../../microbench/gh200_resource_campaign/contracts/protocol.json) 为准。本文件冻结新 `run_suite.py` 的职责；不表示框架、探针或 GPU 运行已经完成。

## 入口与家族边界

| 命令 | 输入与行为 |
|---|---|
| `plan` | 读取 contracts，列出有限案例、源码依赖、审查依赖与未实现项；不编译、不接触 GPU。 |
| `run` | 指定 suite、family、全新 run 目录；检查门禁和 ROMEO 单 GPU Slurm 分配，冻结依赖并编译、运行、归档。 |
| `status` | 只读状态和缺失证据；不把文件存在解释为审查通过。 |
| `resume` | 指定已有 run 目录，核验冻结依赖、binary、环境身份及已有收据；只续跑缺失工作，不重编译。 |
| `audit` | 指定归档目录，从 raw 独立重算并检查哈希、计量、覆盖和完成绑定，结果输出到 stdout。 |
| `report` | 根据归档重新计算中文报告，输出到 stdout；不得改写输入归档。 |
| `finalize` | 离线核验 C 门禁，仅持 suite/output 锁，写最终状态与 COMPLETE；不调用 GPU、不重采。 |

所有新接口使用 `schema_version: 2`。v1 `.py/.cu/.sh` 保持字节不变；旧结果只读路由到旧 schema 的审查入口，不启动新 v1 测量，不重写旧 summary、raw 或 COMPLETE。新家族采用显式 adapter 注册，不按 raw 字段猜测家族或 schema。

每个 family contract 必须声明 `family`、`stage`、`adapter_id`、`source`、`dependencies`、`build`、`cases`、`review_dependencies` 和 `sass_contracts`。每个 case 声明唯一 `id`（对应 raw 的 `case_id`）、`iterations`、`scope`、`threads`、`launch`、`parameters`（具名模型参数和输入条件）、`capabilities`、`work_model`、`work_unit`、`metric` 和 `coverage_policy`。不满足能力条件须有设备查询或受控验证证据；未知能力不等于不支持。

S04 的 `adapter_id` 为 `memory_baseline_v2`。`build` 为 `{"compiler":"nvcc","flags":["-std=c++17","-O3","-lineinfo","-gencode","arch=compute_90a,code=sm_90a","-Xptxas=-v"],"include_dirs":[]}`；source/output/depfile 参数由 runner 根据冻结相对路径追加，不接受 shell 字符串。`review_dependencies` 的元素为 `{"stage":"S04","phase":"A","path":"reviews/S04-A-review.json"}`，path 相对 suite；B/C 同理。case 的 `capabilities` 例如 `{"cc":"9.0","device_name_contains":"GH200"}`；资源或 cluster 条件由家族扩展。实际执行前还检查 S02/S03 框架门禁，不能只列家族审查跳过基础约束。

adapter 负责验证 contract、返回编译参数、构造探针参数、验证 device/raw 扩展字段、独立重算工作量、核查 SASS 计时循环与提取指标。总控负责快照、锁、进程、采样、恢复和终态。adapter 不自行发起 GPU 子进程，全部正确性、测量、NCU 调用经过受控执行入口。每家族 A 审查先冻结具名工作量公式和适用输入；B 审查验证实现。未登记的 adapter/work_model 必须拒绝，不提供通用公式 `eval`。

## 探针 CLI 与原始记录

```text
BINARY device
BINARY CASE_ID ITERATIONS SEED
```

`device` 仅输出一行 device JSON。一次正式调用是一个独立进程，成功输出恰好两行 JSONL：device、trial；stderr 保存诊断，stdout 不混入说明文字。非零退出码、缺行、多行或数值错误不能成为有效样本。原始输出逐字保留，派生摘要引用相对路径和 SHA256。

device 必含 `schema_version=2`、`type="device"`、`uuid`、`name`、`cc`、`sms`、`driver_version`、`runtime_version`。资源、cluster 和其他能力字段可由家族明确扩展。设备查询本身也在 GPU UUID 锁内运行；取锁前使用分配信息和 NVML/`nvidia-smi` 解析物理 UUID，锁后再次核对探针 UUID。

trial 必含以下字段：

| 字段 | 约束 |
|---|---|
| `schema_version`, `type` | `2`、`"trial"`。 |
| `case_id`, `iterations`, `seed` | 与冻结 case、执行参数及收据一致。 |
| `threads`, `blocks`, `scope` | 与冻结启动条件一致；scope 初始只接受 `one_cta`、`all_gpu`。 |
| `errors`, `correctness` | errors 必须为 0；correctness 记录具名检查方法、检查元素数/位置及输入条件，与家族约定核对。 |
| `work_unit`, `work_count` | 单位为 `FLOP`、`OP`、`byte`、`operation`；工作量由 adapter 从冻结模型独立重算。 |
| `read_payload_bytes`, `write_payload_bytes` | 请求 payload 字节数，不能解释为物理 HBM/L2 流量。 |
| `start_ns`, `stop_ns`, `event_ms` | GPU globaltimer 边界和外层 CUDA event 交叉检查；数值有限，时间差为正。 |
| `blocks_detail` | 每 CTA 一项，含 `block_id`、`smid`、`start_ns`、`stop_ns`、`start_cycle`、`stop_cycle`；block_id 恰好覆盖 `[0, blocks)`。 |
| `warmup_samples_ns`, `warmup_converged` | 相同 kernel/配置/分配上的预热窗口数组与收敛标记；独立重算窗口数和尾部 CV，不能只相信布尔值。 |

顶层 globaltimer 边界应等于所有 CTA 的最早开始和最晚结束。`clock64` 仅在同一 CTA 内相减；`one_cta` 必须只有一个 CTA。`coverage_policy` 明确取 `one_sm`、`exact_sms` 或 `observed_subset`，由家族结合合法启动验证；是否要求每个 SM 被观察到由此明确冻结，不能从 scope 或启动 blocks 推断满覆盖。cluster scope 后续须显式扩展 interface 版本及家族约定，经独立 A 审查后使用，不自动归入 all_gpu。

额外字段允许存在，但不能替代必填字段。资源用量、地址布局、完成/排空边界、计数器或特定指令正确性必须由相应家族约定和审查器明确解释。

## 工作量与统计

`work_model` 是已注册的名字，case 的 `parameters` 是对应模型允许的参数。模型的公式及参数量纲必须在家族 A 文档和独立审查器中说明。审查器分别重算工作量、读字节和写字节，不借用探针实现中的求和函数。

`metric` 明确给出 `numerator`、`denominator`、`scale`、`unit`。分子和分母都必须来自有限 quantity 名称：`work_count`、`read_payload_bytes`、`write_payload_bytes`、`payload_bytes`、`elapsed_ns`、`cta_clock64_cycles`、`operation_count`、`one`。`payload_bytes` 是独立重算的读写之和；`elapsed_ns` 是最晚 CTA stop_ns 减最早 CTA start_ns；`operation_count` 由家族工作量模型独立重算；`one` 固定为 1。`cta_clock64_cycles` 仅允许 one_cta。指标等于分子/正分母×scale；单位与 scale 必须由家族审查器核对。空窗口用 `elapsed_ns/one` 得到 ns/window，同步用 `cta_clock64_cycles/operation_count` 得到 cycles/op，吞吐用 `work_count/elapsed_ns`。空窗口工作量可以为 0，但实际采用的分母不能为 0。不能让任意单位字符串使 FLOP 自动成为 OP，或把请求字节变为物理流量。

每点每批恰好 10 个独立进程；同一固定种子决定跨案例顺序，顺序列表归档。每进程预热 8–30 窗，至少第 8 窗才允许停止，最后 5 窗样本 CV≤2%。CV 使用样本标准差除以正均值。未收敛记录完整失败证据，不纳入有效正式样本，也不以无限替换凑满 10 个通过样本。

每个完整批次保存全部 10 个正式值，CV>5% 时按原协议完整重测，最多 3 批。分批统计和全部有效正式样本的合并统计同时保存，不挑最好批次；合并 CV≤5% 才可给稳定资格，否则达到上限后标记有界复测仍不稳定。预热未收敛同样消耗该 case 的一个批次尝试，最多 3 批，保留未完成批次的记录；重测和上限逐 case 管理，不稳定或预热不收敛不阻断同家族无关 case。代码/数值/计量错误直接为实现失败。更改 iterations、编译选项、输入或计时边界必须新 run。

## 门禁与归档

每个阶段 A/B/C 记录独立 reviewer、implementer、五项检查、发现/修复/复核、`gate_files` 和对应 SHA256。reviewer 必须不同于实现者，五项均 pass 且无未关闭的阻断发现才可放行。review 可保存 `reviewed_files_sha256` 作为审阅背景，但实际门禁仅检查明确 `gate_files`；可变 `implementation_status.json` 不得作为 gate file。旧的独立意见保留，不由总控覆写。

B 工作读取 A 门禁，C 正式采样读取 B 门禁及其 A 依赖；对外完成读取 C 门禁。框架自举时 S02-B 读取 S02-A；S02-B 产物通过独立审查后才授权其他家族使用。没有 family adapter、探针或通过门禁时 plan/status 明确显示未实现，不标成外部缺口。

新 run 归档为 `<suite>/<family>/<run>/`，包括 `run_spec.json`、`snapshot/`、`binary/`、`build/`、`environment/`、`reviews/`、`batches/`、`failures/`、`progress.jsonl` 及完成时的 summary/报告/完成绑定。source 和引用均以快照内相对路径保存，拒绝绝对路径、`..` 越界和逃逸符号链接。原工作树绝对路径只可作为非执行性 provenance，不能用于离线审查依赖解析。

快照包含本次执行的 Python import 依赖、C++ 本地 include 依赖、探针、launcher、contracts、审查器、测试、参考、绘图源，以及独立 review 和其 gate_files。编译 depfile 核对本地 include 闭包；工具链系统头文件/共享库记录版本、解析路径和 SHA256 清单，明确属于外部工具链，不把只存清单说成可离线重编译。每个项目依赖均复制并逐文件哈希，snapshot-only 编译/运行/审查不得回退读取工作树。记录 Git HEAD 和 dirty 状态只是背景，文件 SHA256 才标识本次源码。

初次编译后固定 binary 和 SASS 哈希。resume 根据该 run 的冻结 contracts/source/reviews/binary 解释已有收据，不加载后来扩展的工作树家族定义；设备 UUID、驱动、runtime/compiler 身份变化拒绝续写，使用新 run。未提交收据的原始文件保留为中断证据，记录后续 attempt；不得静默覆盖或当有效样本跳过。收据绑定命令相对表示、case/batch/trial、种子、binary/raw 哈希、主机起止时间、PID 和进程组。

audit/report/status 不建立锁文件、不写缓存或 `__pycache__`、不调用 GPU；可在复制后只读目录中运行。采样结束先进入 `collected_pending_review` 并释放 GPU 锁，写不可变 measurement manifest 和 summary，供独立 audit/C review；此时没有 COMPLETE。measurement manifest 绑定 raw、收据、环境、SASS、binary、快照、采样次序和 summary，排除自身、可变状态、COMPLETE 和稍后产生的 C review。C gate 只绑定这些不可变测量产物，不绑定 campaign_status.json、implementation_status.json 或 COMPLETE。C pass 后由离线 `finalize` 核验冻结证据及门禁，绑定 summary、measurement manifest、C review 的 SHA256 后写最终状态及 COMPLETE；finalize 不重采、不改 measurement manifest/summary。审查器从 raw 重算后比对 summary，不能只检查 COMPLETE 存在。

## 锁、超时和环境

suite/output 锁使用规范化 suite 路径派生的固定锁身份，覆盖会变更该 suite 的 run/resume；GPU 锁使用实际 UUID 在同一节点受控 runner 域的固定公共锁根下命名。锁根不随用户任意参数、suite、job 或输出目录改变，拒绝符号链接和不可信现有锁文件。始终先取得 suite/output 锁，再取得 GPU 锁，释放顺序相反。锁表示参与此协议的执行器互斥，不声称系统级独占。

正式运行验证 ROMEO、有效 Slurm job、单 GPU 可见性、GH200/SM90 和 CUDA 12.9，保存验证命令及结果；失败不降级为本地执行。所有 GPU 正确性、预热、计时和 NCU 共享 GPU 锁。

进程 120 秒、编译 180 秒，Slurm 每批 10 分钟。子进程采用独立进程组，超时先 TERM 后 KILL，只清理本任务进程组，并确认该组已消失；不能确认则保存失败日志和 UUID 隔离标记，后续受控运行拒绝该 UUID，直至有明确清理复核。正常退出也检查遗留子进程。失败收据保留 stdout/stderr、超时阶段、信号、PID/PGID 和清理结果。

NCU 与正式采样分离。设备 UUID、驱动、NCU 版本、执行身份与可观察权限配置组成权限环境指纹，在固定节点缓存中记录一次检查；指纹未变不逐点重试。明确计数器权限拒绝只阻断计数器依赖结论；工具缺失、未知失败和超时分别记录，不能伪装为权限拒绝。权限变化须有新的环境证据后才重新检查。缓存结论及原始证据复制入 run，可离线审查。

代码/数值、计时、稳定性、计数器、归档分别给状态。只有 verified unsupported、权限拒绝和有界复测不稳定属于 protocol 允许的缺口；实现错误不属于缺口。

## S02-B / S03 验证接受条件

CPU 测试覆盖 schema/公式拒绝、独立进程收据、两把锁竞争、快照闭包与篡改、门禁过期、冻结恢复、超时组清理、NCU 缓存、批次复测上限、完成哈希和换目录只读审查。假探针仅用于总控测试，明确标记 fixture，绝不能生成 GH200 完成资格。实际设备编译、SASS、最小正确性和采样由各家族 B/C 审查提供，不以 CPU 测试代替。
