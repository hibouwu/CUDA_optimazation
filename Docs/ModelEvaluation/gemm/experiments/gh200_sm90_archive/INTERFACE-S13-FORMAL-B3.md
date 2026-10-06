# S13 全家族短正确性接入正式计量

新增 `global_duplex_formal_v1.json` 使用 `adapter_id=global_duplex_formal_v1`、`execution_revision=family-b3-v2`。它仅增加准入元数据，原 `global_duplex.json` 的18个case、16轮、256线程、全GPU scope、工作量、source、build、地址/缓冲/计时要求均逐字段保持相同。原合同、探针、数值auditor、参考、短profile与短manifest不修改。

本文是新增准入与兼容要求，尚不授权GPU。独立A通过后实现；独立接入B通过后可运行原18点的16轮B4 preflight；独立完整S13 B通过后才可正式每点十进程采样。已通过的1/2短B3不重复运行，除非后续证据漂移导致复用条件不成立。B3不是性能或完整B。

## 有限版本选择

保留 `family-b3-v1 / async_copy_formal_v1 / S12 / async_copy` 的既有接受域和拒绝域。新增且仅新增 `family-b3-v2 / global_duplex_formal_v1 / S13 / global_duplex` 组合。其他revision、adapter、stage、family组合以及缺revision但含B3元数据/新adapter均拒绝。

在 `common/family_b3.py` 内用两个明确分支选择有限policy，复用已有bundle/归档/构建/环境/恢复校验循环。不建设通用registry，不接受合同给出的Python模块或可执行回调，不复制runner循环。原S12 adapter导入的常量/API继续有效。

S12 v1的24点、profile/seed/48launch/65680640检查数、已审依赖规则和校验条件不变；本轮S13新增的逐目标完整指令等价检查只适用于v2，不反向施加给v1。旧run始终经冻结入口执行原核心；当前工作区的核心变化不改写旧归档、旧review或COMPLETE。新建运行必须通过当前版本核心B，不能持旧核心哈希授权新代码。

## S13 B3准入证据

新合同绑定原合同、固定profile、独立B3 review和coverage的仓库相对路径与SHA256。初始化在任何设备程序前验证review的stage/phase/status、独立性、完整gate_files及coverage直接绑定关系；复合implementer字符串不能隐藏自审。

必须核对以下确切集合和数量：

- 原18个case恰好各出现一次，无缺失、重复或额外配置；小/大工作集均保留9种请求条件。1:0与0:1端点仍引用同工作集pure read_cg/write，不新增重复测点；dependent copy仍独立。
- 每点profile=`short_global_duplex_1_2`、seed=3、两次target长度1和2，共36次成功launch。
- 总检查950501376元素，其中有效read checksum4325376、destination word946176000。pure write的零checksum不算读数据覆盖。
- 每launch的检查数独立重算为 `(reads>0 ? blocks*256 : 0) + (writes>0 ? array_bytes/4 : 0)`。实际device、occupancy、grid及对齐工作集用于重算，不直接相信coverage总数。
- 原probe/完整C++依赖、参考、profile、短manifest/ABI身份及各case冻结contract与绑定原件一致。所有18个manifest的精确文件清单、raw/receipt/cleanup/命令/seed/profile和进程GPU身份均核对。

每个case原`family_B3_eligible=false`及原collection pending标记保留；家族资格来自独立汇总B3签署和完整覆盖证明。原输出属于error_count_only：受审host/reference检查了全部声明值，但没有保存输出数组；这些收据不能用于换参考后的离线数值重算。

## 构建、资源和环境等价

B3传递证据文件与原始case归档全部进入新run快照。所有路径以受检冻结仓库为根，离线或搬目录审查不能回查工作区。initialize、构建后的load、第一次新pilot/预热前、resume和离线audit执行同一策略；已有binary、device或样本不能跳过。

正式构建必须通过原S13语义SASS auditor，并与所绑定B3构建逐个核对9个目标函数的完整规范化指令：保留PC、谓词、opcode、操作数及机器编码，排除工具标题/编译绝对路径等非指令元数据。18个case中同symbol的证据必须一致。禁止只凭symbol名、指令数量或原auditor的结构性pass认定等价。binary含路径/调试信息可不同，不强求whole-binary SHA相等，但每个binary/SASS仍分别hash绑定。

比较两侧实际ptxas资源：每个symbol寄存器数、barrier数、静态shared，且stack/spill均为0。当前S13为1024B静态shared、0动态shared、0local、1个barrier。结合完全相同的probe host代码、头文件、编译器和工具SHA、外部toolchain头文件清单及SHA、动态库清单及SHA、设备UUID/属性、driver/runtime环境，才可复用原B3实际occupancy。明确标记这是受审等价条件下复用，不声称新occupancy API查询。

仍保留原runner实际设备query与GPU锁。新进程resume也查询设备；第一次预热前逐文件核对现场外部toolchain/runtime依赖。若节点/UUID、driver/runtime/toolchain或编译依赖不符，拒绝采样；不能自动到别的GPU重测或修改B3指纹。

S13 wrapper要求每份正式raw的kernel、五项资源字段、blocks以及`requested_array_bytes`、`array_bytes`、`aggregate_array_bytes`和读写比例与B3条件一致，再调用原 `global_duplex.validate_trial`。实际工作集仍按原L2相对大小、grid和对齐规则重算。B3短检查只要求全部已启动CTA完成，不强求全SM；正式协议仍要求原exact_sms覆盖。

S13正式分子为 `Qread=16*array_bytes*reads`、`Qwrite=16*array_bytes*writes`、`Q=Qread+Qwrite`，分母为全grid globaltimer包络。该分子是两类请求之和，不套用S12单批运输Q的去重规则；不声称物理HBM字节或缓存命中。计时窗口包含原GPU级write fence、依赖结果shared drain和CTA完成门槛。

## 文件和审查依赖

新增薄wrapper `auditors/global_duplex_formal_v1.py`，委托原数值/资源/计时/SASS逻辑，只加B3资源身份约束。`auditors/suite.py`只显式注册新ID，原global_duplex_v2不作为绕过B3的正式入口。`suite_snapshot.required_reviews`按有限policy选择stage与接入source-B要求；不改 `suite_runner` 采样循环。

本增量A复核为 `reviews/S13-formal-b3-A-review-r2.json`，初稿审查保留。预定接入source-B为 `reviews/S13-formal-b3-source-B-review.json`，正式完整B仍为 `reviews/S13-B-review.json`。新合同预先指向这些路径，避免审后改合同。

共享核心新版使用 `reviews/S02-formal-b3-v2-B-review.json` 与 `reviews/S03-formal-b3-v2-B-review.json`；新建短case诊断使用 `reviews/S02-early-validation-B-review-r4.json`。旧review不覆盖。新建旧家族formal/preflight也须新版核心B；短case仍无需家族B3，但须新版early core B。所有调用required_reviews/freeze/load的派生入口必须检查依赖闭包影响；未另审的派生入口可以拒绝新建，不能借旧source-B放行新核心。既有冻结入口继续原版复现。

S12 v1继续引用原formal-B3 A；S13 v2在保留基础S02 A/preflight-A规则之外引用本补充A。S13 preflight要求B3+接入source-B，不提前要求完整S13 B以免B4循环；formal另要求完整S13 B。required_reviews在freeze/load/audit中必须得到精确相同集合。


新建S12 v1还有两项明确的当前核心兼容门禁：有限policy将原合同中的 `reviews/S12-formal-b3-source-B-review.json` 选择为 `reviews/S12-formal-b3-source-B-review-r2.json`，将formal阶段的 `reviews/S12-B-review.json` 选择为 `reviews/S12-B-review-r2.json`。原合同字节不改，旧冻结入口仍使用原路径；新freeze/load/audit对映射后集合精确一致。preflight仍不要求完整B。

这两份r2必须显式绑定旧review原文SHA、原B4/数值证据、原受审核心字节存档及本轮新核心，独立确认旧测量/参考/正确性资格没有改变、只增加当前核心兼容桥。不能跳过旧review来源或宣称其旧live gate仍有效，不能仅靠一个无证据的新pass替代原完整B；没有r2时新建S12应拒绝。兼容复核不重新签署原probe作者自身的工作，不要求重测已封存GPU数据。S12 v1测量/数值接受域与既有bundle校验条件不变，变化仅为当前核心版本需要对应的新审查证明。

## CPU验证和拒绝条件

增加独立正式host CPU检查，复用实际host分支但CUDA由shim替代：18点固定16轮、多个seed、每次预热和测量的重新poison、所有有效read checksum及目的word、完整stamp、正式JSON/Q与资源字段。shim只能证明host/协议，不是GPU正确性。原1/2短测试与全部SASS负例继续通过；不为测试修改原probe或旧auditor。

新增负例至少覆盖：

1. 缺失/损坏/非独立B3，错误stage/phase/purpose，coverage不在gate中；少点、重复点、额外点、不同profile/seed/长度、错误总数或pure-write虚假读覆盖。
2. 原contract case/source/build或参考/profile/manifest漂移，缺少或额外传递文件，错误raw/receipt/cleanup/设备身份，改动宿主头文件但只保留同名kernel。
3. 9目标缺失/重复、仅改操作数/谓词/编码仍保持opcode计数、不同寄存器/shared/barrier/spill、grid/occupancy/工作集漂移；工具版本串相同但工具/头文件/动态库SHA不同。
4. 已编译/已有device/已有样本的resume绕过；所有这类拒绝须发生在新pilot/预热前。移动完整run后只依赖快照；改工作区不得影响旧归档核验。
5. S12 v1既有正负例与24点实际bundle继续通过；旧六家族formal/preflight新版核心门禁、短上下文无B3、旧冻结入口dispatch回归；未知revision/adapter组合拒绝。

正式接入B仅认证CPU实现、SASS/资源等价策略和当前B3复用；后续18点16轮真实B4 preflight及完整B仍独立，不自动提交GPU、不签C或COMPLETE。
