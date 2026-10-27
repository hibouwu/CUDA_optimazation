# SYNC-01：同步观测回放与槽接口

更新：2026-10-17。阶段3完成CPU接线与旧证据重算，旧回放轮没有新增GPU采样；后续首复用稀疏补测已完成，见§3。服务规则与有限结论统一见[同步服务](../../../model/mechanism/hardware_rules/SYNC-SERVICE.md)；登记与预算见[PLAN](../../../model/mechanism/hardware_rules/MICROBENCH-ROADMAP.md#6-推进次序与本轮交付)。

## 1. 复用范围

| 原始归档 | 条件 | 独立进程 | 轮数 | 程序区别 |
|---|---:|---:|---:|---|
| [001](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/barrier-return/20261015-barrier-return-development-001/README.md) | 18 | 54 | 1728 | 普通arrival／手动complete_tx、1／128到达者、128等待者、三种gap |
| [002](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/barrier-return/20261015-barrier-return-development-002/README.md) | 16 | 48 | 1536 | try_wait形式，1／128等待者、可选后继事务登记、±512 gap |
| [003](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/barrier-return/20261015-barrier-return-development-003/README.md) | 12 | 36 | 1152 | 同一二进制三种wait形式，可选后继事务登记、±512 gap |

每次调用32轮；全部为单CTA、256线程、动态SMEM215040 B、驻留1CTA／SM，实际SMID124。原始程序每进程两次同条件暖机后采样，没有收敛检测。源码与正式构建各自绑定；三个归档中的同名条件不合并。

## 2. 重算与交付

[回放入口](../../../../../../microbench/gh200_resource_campaign/access_rules/replay_barrier_return.py)固定三份归档SHA，核对源绑定、命令、返回码、样本SHA、条件和线程戳，独立重算每轮窗口，再与原analysis比较。`wait_call_cycles`与`last_signal_to_return_cycles`分别保存；后者上下界不取中点。完整结果见[本轮包](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/SYNC-01-02-control/20261017-SYNC-01-replay-001/README.md)。

在仓库根目录执行，输出目录须尚不存在：

```bash
python -B microbench/gh200_resource_campaign/access_rules/replay_barrier_return.py \
  --output /tmp/sync-01-replay
PYTHONPATH=microbench/gh200_resource_campaign/access_rules \
  python -B -m unittest test_mbarrier_service test_cfg_b_request_model -v
```

回放46条件／138进程／4416轮全部通过；六个小图的默认引擎结果与接口加入前逐项相等，21项CPU回归通过。保存副本的源码和命令也已重跑；第一次副本运行的历史数据相对路径错误保留在包内，随后显式指定原证据目录后全部通过。

本轮交付的是可运行的phase／计数与条件返回假设，以及已核对的旧观测。当前没有新增校准价格接回GEMM，也未新增独立C2/C3成绩；真实短链的剩余观测见[服务页](../../../model/mechanism/hardware_rules/SYNC-SERVICE.md#3-旧观测约束与剩余缺口)。

## 3. 首复用稀疏补测

[SYNC-02执行包](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/SYNC-01-02-control/20261017-SYNC-02-sparse-004/README.md)固定S6、13 Ktile、256线程与六个SMEM驻留操作数槽（196608 B），full(1)／empty(128)均为普通arrival，无TMA或操作数重新填槽。首S组在窗口前就绪，D=0／8192 cycle只在首次复用取得empty后延迟一次。plain／sparse各3个seed，共12开发进程；[独立评分](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/SYNC-01-02-control/20261017-SYNC-02-sparse-004/score.json)核对完整196608个输出元素、槽代次、资源与暖机资格全部通过。本轮同卡配对绑定GPU-fd5317a4，不与旧卡窗口合并。

主窗口从准备完成后的起始CTA会合之前取戳，到tail wait0、末槽释放、累加器消费及末次CTA会合之后取戳，端点导出在窗口外。下表分别列三seed中位数及[min,max]；稀疏端点在循环中以clock64取戳，窗口ns来自globaltimer。

| D cycle | 版本 | cycle中位数[min,max] | ns中位数[min,max] |
|---:|---|---:|---:|
| 0 | plain | 10778 [10778,10778] | 5984 [5952,5984] |
| 0 | sparse | 13455 [13450,13455] | 7424 [7424,7424] |
| 8192 | plain | 15779 [15778,15781] | 8704 [8704,8768] |
| 8192 | sparse | 17739 [17739,17739] | 9824 [9792,9888] |

只保留full的g0／g6、empty取得g6／g7及释放g0／g1的端点；128名消费者各自的arrival前后戳留在寄存器，窗口后导出，再取`[max(before),max(after)]`界定最后必要arrival。以下观察仅属于sparse程序：g0预就绪的wait窗口178–183 cycle；g6在D0下进入前已就绪，wait为178 cycle，D8192下先进入再收到信号，wait为4128 cycle、full信号至返回位于[103,123] cycle。D8192下等待入口早于信号，first-try仍成功；两种入口状态按信号与入口的先后分类。g0的empty取得均先进入再释放，最后arrival至返回区间[110,234] cycle；g1在D0下同为等待路径，区间[261,372] cycle，D8192下则进入前已就绪、调用135 cycle，其[7786,7935] cycle信号至返回区间包含进入前时间，调用窗口单独保留。

D8192−D0的窗口响应为plain 5001 cycle／2720 ns、sparse 4284 cycle／2400 ns，分别保留为程序复合响应。D0三seed的sparse扰动为cycle +24.79%–24.84%、ns +24.06%–24.73%；D8192为cycle +12.41%–12.43%、ns +12.50%–12.87%。六个配对全部超过原5%门槛，按原规则停止S4扩展；端点不迁移到plain。未冻结绝对价格预测、未拟合或新增独立C2/C3，full完成至返回及empty最后arrival至取得槽的独立价格仍未辨识。

## 4. 索引控制开发判别

[新轮次](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/SYNC-01-02-control/20261017-Q04-index-control-001/README.md)保持驻留WGMMA与普通full/empty，比较原runtime槽索引和递增回绕；主判据使用plain完整窗口及受控延迟响应。[执行751879](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/SYNC-01-02-control/20261017-Q04-index-control-002/README.md)的12进程数值、代次、资源与暖机合格；D0由10778降至7572 cycle，D4096−D0由约908变为2067 cycle。原索引同批重现旧窗口；软件控制改变推进和复合隐藏余量。独立控制价格及跨stage／长度外推尚未验证；旧首复用观察失败保留。

首轮[CPU分析](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/SYNC-01-02-control/20261017-Q04-index-analysis-001/README.md)已排除只改consumer循环推进的形式；下一轮[consumer-only与长度配对](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/SYNC-01-02-control/20261017-Q04-control-localization-001/README.md)把producer／tail控制保持原样，独立定位剩余连接。

751902的[控制连接配对](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/SYNC-01-02-control/20261017-Q04-control-localization-001/README.md)12进程合格；incremental K4/K13增长为509.111 cycle/组，consumer-only K13 D0／D4096为7859／10003 cycle。只改消费者推进已消除原基线的大部分超额时间，生产者与tail原控制合计仍差287 cycle；硬件返回价格保持未辨识。

## 相关轮次

以下各轮原登记在访问规则索引，按日期排列；全部结果目录见[Q04-sync-slot/SYNC-01-02-control](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/SYNC-01-02-control/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| 索引控制能否解释短链推进失配（开发判别） | 固定S6/13、六物理槽、generation与计算／barrier；原索引／递增回绕×D0／4096 | 751879的12进程合格，递增回绕改变D0及延迟响应，支持显式软件推进；跨stage／长度规则未验 | [准备与失败](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/SYNC-01-02-control/20261017-Q04-index-control-001/README.md)、[执行与判定](../../../../../../results/gh200_resource_campaign/access_rules/Q04-sync-slot/SYNC-01-02-control/20261017-Q04-index-control-002/README.md) |
| a044 跨卡复核（mechanism-r3，780769） | 索引控制、控制定位、稀疏观测各 12 进程，原检查器与预测 | 无注入窗口 runtime/incremental 索引为 10778/7572 cycle，4096 cycle 注入的响应为 908/2067，与旧卡中位相同；局部组合两版的长度增长误差 0.567%，通过，但注入响应 1792 对 2067，误差 15.35%，未过；稀疏打点 6 对扰动 12–25%，全部超过门槛 | [Q04 报告](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/q04-returned-stage-report.md) |
