# S15 短验证分批运行与完整归档草案

状态：待独立controller A/source-B。既有S15实验A、source-B保持；不启动GPU、不修改通用core。controller当前必须找到自己的两份独审门禁，否则run拒绝。packed机制只保全字节，不继承S14资格或自动授S15 B3。

## 有限任务与身份

原68个case按合同顺序，各固定三个profile：1轮/seed0、2轮/seed3、33轮/seedUINT32_MAX，共204独立进程、204target。`--start/--stop`为半开区间0≤start<stop≤204，plan输出index/case/profile/seed，可按同一固定序列恢复；不做seed/长度笛卡尔积。

同family第一次实际GH200有效Slurm分配，冻结实际UUID、设备query、driver/compiler/tools/uid及contract/profile/adapter/probe依赖哈希。之后不能混设备或改源码续写。S15第一次设备不要求等于历史S14 UUID。只读device query由已审S15 binary的device入口获取，保存stdout/receipt/binary/env身份，不增加target。设备JSON必须与当前分配UUID一致；footprint公式使用实际sms/SMEM/register容量，不能把历史132SM当此次实际查询。

family controller锁防同family两个slice冲突；原core的suite/output锁与GPU UUID锁继续生效。文件锁不宣称系统级GPU独占。

## 完整artifact空间模型

Q为有效tile bytes，H/W为box，P为global row stride，n为固定短轮数，B为实际grid；必要资源容量上界得到 `B_upper=sms*min(4,floor(SMEM/(Q+1056)),floor(registers/(40*128)),16)`，oneCTA为1。实际occupancy来自GPU API，不能把这个容量上界当实测驻留。

每进程完整文件字节为：

`logical=B*n*Q`；`physical=B*n*Q`；`global_ring+padding+guards=B*32*H*P+256`；completion与stamp各`B*5*2*4`；opaque descriptor为128 B。合计 `2*B*n*Q+B*32*H*P+256+80*B+128`。

两个运输方向都保存完整global数组；padding和guards计入存储，但不计有效transport payload。使用最大合法B估计，启动前再依据已绑定device query检查空间。不能靠压缩比或降低B/Q/轮数来缩小实验。

## 最小状态及不重测规则

仅一个family ledger保存204坐标及身份，底层run保持原core工件。已通过resident只audit；pending/checkpoint且无任何attempt可resume；失败、已启动不明或初始化残缺先离线审查，不能盲跑。

已外部封存坐标状态为`archived_verified`，绑定archive/index/closure及remote成员验证receipt、offhost完整成员验证receipt和原run manifest SHA。closure固定schema1/run_path/原validation_manifest SHA/members，run_path必须等于该case/profile的规范S15路径。包内validation_manifest与spec、raw、diagnostic_summary必须存在，manifest全部成员加manifest自身恰好等于包成员集合；通过完整解码后在受审只读事务核对spec和raw的case/profile/seed及原完成状态。remote/offhost receipt必须分别明确location、complete_decode_all_members_sha256终态、archive/index/closure SHA、原run/manifest SHA、完整成员表、真实成员数和解码字节数；布尔声明或placeholder拒绝。真实封包/receipt生成及retire仍属于后续source-B，CPU fixture只验证边界，不声称发生远端保全。展开目录缺失时仍先验证这些refs并执行archive audit，绝不因此run新GPU。ledger不是B3证书；source/数据终态仍由原raw、receipt、manifest和独审记录确定。

## 分批保全与配额

默认resident raw窗口最多4 GiB，每个Slurm slice原则上1–4进程，最多10分钟；大case可单进程批。累计达到窗口或不能满足下一进程quota时，在下个target初始化前安全停止并记录`quota_checkpoint_before_next_target`，该坐标仍未执行。controller保存完整住留目录的实际size，估计每个新进程再留64 MiB元数据上界；正式实施B需验证这个元数据预算足够。

配额预算至少包括：当前用户实际scratch占用、部署源码/binary/固定snapshots、当前raw、下一个进程完整artifacts+元数据、封包最坏额外空间、输出/日志余量。封包最坏预算按raw完整长度加tar/XZ开销，不能假设2%压缩比。20 GiB是此次平台已观察硬限制，实际启动前用当前quota receipt中的hard/used/headroom核对，而不允许一个随意CLI数字充当验证证据。controller只接纳60秒内、同用户、覆盖本部署filesystem的quota receipt，绑定原query stdout SHA；hard/used/reserve均为字节整数，reserve至少512MiB。source-B须包含生成并验证这种receipt的薄wrapper及真实query解析，不允许随意CLI headroom数值替代。每新target前重新检查freshness与raw+最坏seal预算，过期先刷新query而不启动target。

批次全部进程终态后，suite/family锁下只读固定regular-file清单，逐文件hash冻结；对比封包前后源hash和完整成员集合，拒绝目录mtime-only失败、链接、路径穿越、损坏/额外/缺失成员。采用无损tar.xz，uint16数据可选择delta2，控制数据仍保留所有字节。单批原文保留直到压缩包在remote完整解码核对，并转存到本地/另一个独立有持久空间的归档点后逐member再次验证。

只有remote+offhost两份完整字节验证成功且收据/pack身份写入ledger，才允许显式retire本任务scratch的该批resident目录以释放quota；controller不自动删除。retire先确认无active进程/未清理attempt，path在本部署scratch内且manifest完整，不删除本地canonical历史归档。若丢失offhost收据或包无法验证，禁止retire，停止后续占用增加。保全包和源双方均损坏不能伪装为通过，更不能按缺展开目录重测。

每批运送/封存仅完成运输层。完整S15全值重放、opaque descriptor输入/128 B身份、logical/physical/swizzle/padding、lifecycle及所有204坐标仍需另一实施者独立B3；本轮归档不会授性能。

## A/B接受条件

CPU枚举204唯一坐标和最高artifact/总容量；slice/ledger状态与archive缺展开目录不重复GPU负例；源/profile/UUID漂移拒绝；实际quota不足在target前退出。首次真实query后重新核footprint与必要容量，实际编译/sourceB复核controller/driver。运输B使用真实小批完整压缩和offhost重放，逐member不抽样；每批原始字节有唯一终态和来源，code只调用已审validate_suite，不扩core。
