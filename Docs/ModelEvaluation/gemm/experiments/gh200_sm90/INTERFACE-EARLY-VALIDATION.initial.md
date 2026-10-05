# 前置短正确性与证据复用接口

状态：A 草案，未实现。有限 schema 草案见 [early_validation_v1.json](../../../../../microbench/gh200_resource_campaign/contracts/early_validation_v1.json)。本补充落实 [GOAL.md](GOAL.md) 的 B1–B4 顺序。已签 S02/S03 B 的源码、旧 run 及历史 gate 均保持原样。

## 顺序与诊断选择

对每个家族，在任何新的 pilot、预热、长循环 smoke 或正式计时前，先完成该家族冻结矩阵的全配置短正确性覆盖及独立 B3 复核。多个就绪家族优先调度短检查；这不是等待全部 S04–S20 实现后才放行任何家族。无依赖家族可以继续，受控 GPU 执行仍按 UUID 串行。

B1 的源码/独立参考与 B2 的编译/SASS 合格后才能运行 B3。B3 合格只允许进入 B4；B4 的计量、pilot、长度冻结及完整独立 B 通过后才允许 C。B1–B4 仍是原 B 阶段的内部证据，不新增研究阶段。

允许 `--case CASE_ID` 的单配置短诊断，用于立即定位已知失败，尤其当前 `wgmma_e4m3_g1_one_cta`。它使用相同环境、快照、双锁、UUID 进程注册和超时保护，但记录 `selection_scope=case_diagnostic`；单点通过不能生成 family B3/B 或正式资格，也不能启动 pilot/预热。未选择的点明确为未检查。该入口可在扩展 A 通过、S08 源码/CPU及短入口实现经独立复核后先交付，不等待其他家族改造。

家族全矩阵模式的覆盖集合必须等于合同中的 case ID 集合；有限矩阵中的每点最终都要有新验证通过、受审旧证据复用，或独立确认的不支持能力终态。任何未解决的实现/数值/完成条件错误均阻止受影响家族进入 B4，不能改称外部缺口。

## 探针模式

```text
BINARY device
BINARY CASE_ID ITERATIONS SEED
BINARY validate-only CASE_ID PROFILE_ID SEED
```

旧测量命令的参数和行为保持不变。新命令是独立 host 分支，必须在进入既有 pilot、`gh::warmup` 和正式测量调用前返回；不能仅把现有“包含完整预热的 preflight”改名为 validate-only。

`PROFILE_ID` 对应 A 审查冻结的有限计划，不能自由传表达式或任意循环长度。通用上限为每 case 至多 4 次 target kernel launch、每次至多 64 个循环单元；显式初始化/验证辅助 kernel 上限由家族 profile 冻结，禁止隐藏循环重试。保留原 case 的 threads、参与组、chains、batch/wait、形状、布局、资源和完成协议，只缩短 profile 指定的重复长度。不能为使诊断成功而减少参与者、换指令或替换成另一参考 kernel。

S08 首个 profile 为 `short_uniform_nonuniform_1_2`：同一个目标特化在 uniform/nonuniform 输入各运行 1、2 个迭代，共 4 次 target launch；每次 fresh D/poison，完整输出对照独立参考。保留原精确比较及所有目标参与组。仅诊断收据，不计算可导出的速率。零工作控制和具有最小合法生命周期的异步家族须各自声明合法 profile，不能套用正工作量的 1/2 循环。

正确性检查要求 CUDA 调用成功、所有声明参与者和输出均被核对、异步完成/缓冲复用条件成立。短窗口可能无法分辨 globaltimer；validate-only 不以“正计时差/CV收敛”作为成功条件，也不把零时间当作性能值。设备完成与输出检查不能因此省略。`all_gpu` 保留合法整卡启动几何；短 kernel 不保证占遍每个物理 SM，B3 验证的是全部已启动 CTA 的参与及结果，原测量合同的 `exact_sms` 仍在 B4/C 严格检查，不能从 B3 推断整卡计时覆盖。

## 新原始记录

成功 stdout 恰好两行 JSONL：原有 device 行，以及 `type="validation"` 行。第二行使用 `schema_version=2`、`validation_schema_version=1`，不能进入旧 `trial` 统计路径。

validation 行必含：`case_id`、`profile_id`、`seed`、`scope`、`threads`、`blocks`、`errors`、`target_launches`、`checks`、`resource_identity`，以及固定 `performance_eligible=false`、`warmup_executed=false`、`pilot_executed=false`。每项 target launch 给出迭代长度、输入 profile、完整 CTA 参与计数、输出检查数、检查方法/参考版本、错误数和完成检查结果。adapter 从冻结 case/profile 独立核对这些字段与工作/输出规模，不相信汇总 errors=0 即通过。

检查覆盖所有声明输出。输出证据按 profile 明确分为 `full_values`、`full_declared_checksum_outputs` 或 `error_count_only`，存在的输出文件必须具有类型、形状、相对路径和 SHA256。是否持久保存所有值与是否检查全部输出分开记录；仅错误计数不能支持未来更换参考后的离线正确性重审。禁止未声明的截断或抽样冒充完整检查。

失败时尽量输出同类型的失败记录再非零退出；无法形成完整记录则保留 stdout/stderr 和失败收据，不构造 pass。stderr 可保存最多 8 条 stage/length/CTA/thread/chain/fragment/坐标/actual/expected 定位信息。崩溃、部分 JSON 和超时均不是 verified unsupported。

## 可恢复运行与身份

拟新增受审入口 `validate_suite.py run --contract CONTRACT --output NEW_VALIDATION_RUN [--case CASE_ID]`，及对应 `resume/audit/status`。`--case` 仅诊断；省略时要求完整家族覆盖。原 `run_suite.py` 及已冻结入口不改。

每 case 的短进程上限为 30 秒，启动前还须满足 Slurm 剩余时间大于该上限加 20 秒清理/归档余量；编译、10 分钟 job、双锁及 UUID 跨 run 登记沿用已审协议。真正的短执行由有限 profile、源码分支与命令收据共同证明，不由“运行未超时”单独证明。

归档包含冻结 contract/profile/adapter/probe/参考/审查依赖、编译与 SASS 身份、device、每点原始输出及进程收据、完成/失败/中断证据。`validation_spec.json` 在首次启动前冻结选择范围和顺序；append-only journal 与逐点收据支持跨 Slurm 恢复。已通过点的证据哈希和身份一致时跳过，不因恢复重测。中断点保留 attempt，依据既有 UUID 清理协议确认旧组结束后才重试；实现或数值失败不自动重试凑通过。

验证身份至少包含：case 的语义哈希（指令、类型、形状、参与者、布局、输入与完成协议）、profile 哈希、binary/SASS、编译依赖、参考代码/参数、GPU UUID及设备/工具链/关键环境。记录维度可分离，但完整依赖快照仍保留；不能只看文件名、case ID 或最近一次成功状态。

全配置证据收齐后生成不可变 validation manifest，状态为 `validation_collected_pending_B3_review`。独立复核在原 B 内生成限定 `phase="early-validation-B3"` 的增量审查，直接绑定 manifest 和证据，授权进入 pilot/warmup；它不能作为原 `phase="B"` 的完整门禁。新编排器在进入 B4 前检查该限定授权，不能靠可变状态文件判断。

## 已有成功证据

作业 730217 的三计算家族及 memory 已有成功 preflight 不按新步骤名称重复执行。先回收原工件，离线审查原正确性、编译/SASS和校准证据；浮点末位问题用 [只读后审接口](INTERFACE-OFFLINE-REPLAY.md) 处理，原严格失败与后审结果同时保留。S08 已成功的 16 个 MMA 点也保留，不因后续 WGMMA 失败整体清零。

复用必须有独立 `validation_reuse.json` 映射：原 run/case、原 raw/receipt/binary/SASS/参考身份、适用的已审正确性要求、新选择范围中的对应 case、覆盖证明及原/新身份之间的关系。结论为 `reused_prior_numeric_evidence`，不能伪称旧二进制执行过新 validate-only 命令。

流程顺序优化不自动增加旧合同未要求的数值要求。既有受审 full preflight 满足原家族正确性覆盖时，可以作为 B3 目的已达到的证据；仍须逐项说明实际输入、长度、参与者及检查范围，而非由“长运行通过”推断任意短 profile 已执行。不能证明所需语义覆盖的缺项，才补做相应短检查。

同一 binary/设备/参考及语义身份可直接审查复用。身份变化时，必须提供独立影响记录和实际证据说明哪些路径未受影响，例如对应特化的编译/SASS/资源与参考检查路径；不自动把同文件哈希变化解释为所有家族失效，也不自动把 kernel 名相同解释为不受影响。参考改变而原归档只有错误计数时，受影响配置必须重新验证。任何已有成功点如被发现与同一错误实现共享实际依赖，也应标记失效待复核，不能为了避免重测保留错误资格。

原 run 不导入新字段或重新封口。复用映射、影响记录和新审查保存在原 run 外的扩展归档，直接绑定原工件哈希。改变规则或测量身份时新开 run；历史结果仍代表原条件。

## 失败范围

- 单点数值/布局/完成条件错误：先保存定位证据，阻止其相关 kernel/参考/协议依赖进入 B4/C；未知影响范围时保守阻止该家族，先分类再扩大诊断。
- 同家族无依赖特化可继续短诊断以定位范围，但不得把未解决的整个家族标成 B3 通过。
- 公共运行时或参考问题：列出实际调用该路径的家族和 case；未受影响分支须有独立证据才能复用。
- 未确认进程清理：按 UUID 隔离，所有使用该 GPU 的受控命令暂停，不限于失败 run。
- 无依赖家族不撤销原有效证据；新的计时/稳定性问题仍按原协议逐 case 有界处理。

## 改动面与扩展门禁

| 组件 | 后续所需改动 |
|---|---|
| probe / 生成器 | 增加独立 validate-only 分支及有限 profile；复用目标 kernel/参考，跳过所有 pilot/预热/长测量；按实际受影响 family 实施。 |
| adapter | 新 ABI 1 导出 `validation_argv`、`validate_validation` 和受审的 prior-evidence 对照入口；现有 `validate_trial/audit_sass/resolve_iterations` 保持各自职责。 |
| 编排器 | 新版本加入可恢复 B3 阶段及全矩阵屏障、单点诊断选择、复用映射和影响记录；B4/C 入口必须验证 B3 授权。 |
| 审查器 | 独立验证短输出、全覆盖、身份及禁止进入性能统计；新增后审收据可以复核旧数据，不能更改旧结果。 |

本补充通过独立 `S02/early-validation-A` 后才实施。可以先交付受同等约束的 S08 单点诊断入口，取得代码/CPU与已有框架安全能力的独立扩展 B 复核后运行；这不授予 family B。完整家族 B3 调度与屏障再通过相应扩展 B 负例后启用。

采用一次性受审的版本化扩展入口/通用 adapter host，复用固定哈希的已审安全原语。旧文件不就地改名、补丁或覆盖；新逻辑和 ABI 写入新文件、使用新的扩展门禁。若确需更改安全原语，则新建明确版本并单独复审，旧 snapshot 继续使用旧文件。

每个家族单独提交 `adapter_manifest`：ABI版本、受限模块路径、导出能力、case/profile合同及模块/依赖 SHA256、家族 A/B1/B2 和适用 B 审查。通用 host 先验证这些显式门禁和哈希，再载入快照内 adapter；未知 ABI、未审模块或路径逃逸必须拒绝。B1/B2 的限定授权允许取得 B3 实证，完整 family B 仍控制正式资格，避免先有 B 才能取得 B 证据的循环。

核心扩展 B 只绑定 loader/ABI/流程/安全原语及通用负例，不绑定一个不断增长的家族目录或可变 registry 总表。新家族仅增加独立受审 manifest、adapter、contract/profile及依赖，不为加入名称改核心 Python 注册表；每次 run 冻结实际载入的 manifest 与依赖集合。家族新增不能通过无门禁动态 import 获得资格。

## 最小接受检查

独立测试必须拒绝：未完整 B3 就执行 pilot/预热；validate-only 触发长循环或 warmup；单点诊断被升级成 family B；复用错误 case/参考/binary/设备；中断恢复重复成功点或覆盖失败；仅错误计数支持更换参考；未审 adapter 与未知 ABI；把 validation 行当 trial；未确认 GPU 组继续执行。正例必须证明成功旧点无需 GPU 重跑、单点失败不清空其他已验证证据、复制归档后可离线核验。新增检查不改变 C 的十进程/三批/禁止择优协议。
