# 大型短验证归档的压缩准入草案

本草案只处理 S14 的完整输出归档运输与冻结。原 84 组输出约 12 GB；将原文件同时展开为部署输入和运行快照，会超过当前 scratch 20 GiB 硬配额。原结果、独立 B3 审查、数值参考和计时约定保持原版本。本文未授权新正式运行，也未实现新的准入。

## 数据与身份

压缩包保存全部原文件字节，另存逐文件逻辑路径、长度和 SHA256 清单、压缩包长度与 SHA256。原审查和 coverage 位于包内，原来绑定的每个文件都必须存在且哈希相同。禁止只保留错误计数、汇总值或审查证书来替代完整输出。

新增独立桥接审查绑定压缩包、清单、原审查和 coverage 的内容身份，证明压缩运输没有改变原资格。当前 S12-v1/S13-v2 的准入不修改；新有限修订只接纳明确登记的 S14 包，不从合同指定任意可执行模块。

## 检查顺序

1. 部署或重放先核对包的外部身份，再流式读取全部成员；逐项重算长度、SHA256，核实成员集合与清单完全一致。
2. 拒绝重复成员、绝对路径、路径穿越、链接、设备文件、额外或缺失成员、损坏压缩流、超出预声明大小以及原审查绑定缺失。不得直接对未检查的包调用无约束 extractall。
3. 从包内读取原审查和 coverage，检查独立性、各维度终态、原合同/profile/source/参考身份、全部配置覆盖、数值与完成检查身份。原完整证据可以在独立离线审查时逐值重算；运输检查不授予新的数值正确性。
4. 冻结快照只复制经过完整核验的压缩包和清单，不重复展开其原始输出。运行中及换目录重放使用同一核验接口；缓存只能节省重复解码，不能省略对实际包字节身份的检查。
5. 编译所需源码、当前正式合同与实际编译工件仍按普通快照冻结。压缩的历史短证据不作为可执行代码的替代入口。新增 formal capture=false 短证据单独保留并准入，不能由原 capture=true 包推导正确性。

## 验收与影响

独立 A 审查后再实现有限的新准入。B 需要完整旧包离线核验、换目录重放、损坏包/清单/审查绑定/成员负例、旧 S12/S13 回归和实际冻结大小检查。旧已封存运行继续调用自己的原快照，不改历史清单或 COMPLETE。

包仅用于减少归档物理占用，不改变资源服务、GPU 身份、配置矩阵、采样次数、数值检查或接受阈值。只有归档运输和准入依赖受影响；不因此重跑已完整验证的原 84 组 GPU 输出。

## r2：解析接口与冻结边界

新增只读 `PackedEvidence` 接口，构造参数为包路径、外部SHA256及逻辑成员清单。`verify_all()` 流式核对整个包和全部成员；`read_bytes(name)`、`read_json(name)`、`file_sha256(name)`、`inventory(prefix)` 只解析经清单确认的安全逻辑路径，拒绝未知成员，缓存不得持有大输出数组。实现可以进行多次串行扫描；在 CPU 阶段记录耗时与峰值内存，不让归档扫描进入GPU计时窗口。内存上限采用固定读取块；大型payload只流式hash，JSON先按清单声明长度设置有界读取。

历史门禁通过独立 `validate_packed_gate(bundle, path, stage, phase)` 读取原审查JSON，并验证其独立性、维度、终态及所有gate_files在包内的完整SHA。这个函数只检查历史namespace，不将包内旧核心文件与当前源码比较；不调用当前 `validate_gate(path, live_repo, ...)` 来解释原门禁。当前修订的源码、核心、正式host准入由新的普通门禁在真实文件namespace内审查。旧S12/S13准入不路由到此接口。

新增S14有限policy以原coverage的 `records` 集合为输入（不是S13的 `cases` 字段）。必须证明24个原合同坐标分别有固定1/0、2/3、33/UINTMAX三配对，并有12个S2G source-release记录；总84个记录唯一且无缺失。核对每条record的原run/spec/manifest/raw/receipt/完整数组身份、18目标实际编码资源以及相同device/environment。原coverage保留原schema和字节，policy生成新的只读规范化索引，不改原审查。新增capture=false的24点72launch采用另一个包及另一个独立B3审查，不能合并成原84或声称原profile已覆盖它。

`validate_packed_bundle()` 返回两部分：`physical_names` 仅包含压缩包、外部成员清单、桥接审查及必要当前源码文件；`historical_qualification` 包含已核实原合同/profile/source/参考/coverage、目标和环境身份。`freeze()` 只把physical_names复制进快照并哈希，绝不把原gate_files的虚拟路径加入普通names。`check_snapshot()` 先照常核验物理快照，再调用对应有限policy检查包内历史资格；每个新run只接纳登记过的修订。包路径仍必须位于快照内，离线重放不依赖原目录、原部署或外部可变源码。

桥接审查的普通gate_files绑定新policy/resolver/负例、包及清单SHA；同时在显式历史字段中绑定包内原review/coverage SHA。桥接不得只写一个pass布尔值或自行重新授予原B3。包内全部原review绑定缺失或变更时必须拒绝。原review中重复引用同一内容可以在打包时使用一个逻辑成员；不同路径不得悄悄合并，清单要保留完整逻辑成员集合及每个成员的原字节。

B阶段补充验收：用实际原84包验证以上records规则；新capture=false包未取得B3时拒绝formal准入；snapshot体积不包含展开payload；搬走或删除源展开目录后仍能核验已冻结包。schema桥接、历史namespace与当前源码namespace混用、遗漏gate文件、替换原coverage、篡改manifest后重打包、匹配错误GPU、同名不同内容和跨包冲突都需要负例。保存原S12/S13真实bundle的离线回归结论。

## r3：成员清单、流式数值重放与配额

外部index固定字段为 `schema_version=1`、`namespace=repository_relative_v1`、`archive_sha256`、`archive_bytes`、`members`（逻辑路径到 `{bytes,sha256}`）、`roots`（原review、coverage、合同、profiles及84个run的逻辑路径）。逻辑根是仓库相对路径，不接纳不同根的自动猜测、basename匹配或路径重写。文件权限/时间不参与内容身份；成员必须为普通文件。

成员预期集合由独立已绑定的历史review与coverage作为起点，递归收集原gate_files、全部84个run manifest的完整文件列表及manifest/spec/state本身，以及这些spec指定的冻结依赖manifest、原合同/profile/reference、source/binary/SASS/compile/environment/raw/receipt和所有payload/control工件。先离线对原展开证据生成独立closure记录，再由桥接review绑定其SHA。打包index成员集合必须与这个固定closure完全一致，不允许以index自报的成员列表决定哪些原证据可以丢弃。原raw中引用的数组路径也必须在对应run manifest中且列入closure。历史门禁引用其它清单或包时同样递归列入；若存在不同路径的相同字节，两个逻辑路径仍分别保留。

`read_bytes/read_json` 仅允许不超过16 MiB的小对象；过大JSON应改用有界流式解析或在A增量中明确扩大需求与内存上限。新增 `iter_bytes(name, chunk_bytes=1048576)`，固定最大读取块1 MiB，支持payload完整逐word参考重放；数值checker必须按数组dtype/shape/byteorder维护跨chunk残余word，不得将完整payload转为list。若现有checker要求seek，可将单个member解码为独占临时文件；必须预先核对长度、空间上限和哈希，使用完删除自己创建的临时文件，不展开整个包。初版优先流式接口，单member临时方式只有独立审查过的调用点才准入。

所有准入缓存键包含实际archive/index/bridge/closure/checker源码SHA及修订号；使用缓存前重新计算实际archive字节SHA。缓存不扩大资格，也不使不同源码、namespace、设备或schema共用结果。最终独立C和S22重放至少各执行一次不使用缓存的完整核验。

配额验收记录实际部署压缩包+index、冻结快照包+index、当前run工件、最大并存临时文件及其他本任务占用的字节总和。新run初始化前检查scratch当前用户配额和所需峰值，必须小于20 GiB硬配额，并为输出与日志保留明确余量；不能仅比较一份包大小。实际演练冻结后移走源展开目录并离线重放，报告峰值磁盘与内存。若上限不足，先调整归档存放及流式实现，不能删减证据或重复GPU采样。
