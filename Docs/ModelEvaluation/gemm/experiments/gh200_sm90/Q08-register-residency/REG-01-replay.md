# REG-01：寄存器条件证据与启动配额

更新：2026-10-17。阶段7离线交付，未新增GPU采样或拟合。有限登记见[PLAN](../../../model/mechanism/hardware_rules/MICROBENCH-ROADMAP.md#6-推进次序与本轮交付)，需求与配额规则见[寄存器服务](../../../model/mechanism/hardware_rules/register_service.md)。

## 1. 固定证据

| 证据 | 本轮范围 | 重放结果 |
|---|---|---|
| [R16配额](R16-residency-quota.md) | 六条件、66进程 | 3269376个值逐一核对；原条件、进程窗口、SASS与身份分析精确相等，六条件合格 |
| [R02 v8](R02-register-service.md) | 32服务条件、382记录 | 1379840个uint64见证字重算；与`with-carry-controls-v1`的rules和checks逐字节相同，保留28个推进／32个消费合格条件 |
| [R02固定帧](R02-register-service.md#固定帧诊断差异缩小但没有消失) | 82记录 | 514304个uint64见证字重算；rules和修正SASS分析与`local-independent`逐字节相同 |
| [R01正式FFMA链](R01-readiness.md) | 128／512／2048步各3进程，共9记录 | 288个输出重算，64步展开、HFMA2.MMA准备与原窗口核对；固定关系`T=265+4.265625N`及零残差精确复现 |

合计539份旧记录。R02原380记录报告保留，后来两条u64进位控制增加了12544个见证字，32个服务条件与价格保持。固定帧plain的两代码位置均为复用9880、轮换10400 cycle；position1／trace轮换为10399。旧22%与固定帧5.26%来自不同GPU。上述来源与逐记录身份保存在[交付包](../../../../../../results/gh200_resource_campaign/access_rules/Q08-register-residency/REG-01-SM-01-replay/20261017-REG-01-replay-001/README.md)，沿用原评分和资格。

## 2. 实际程序与配额

S4/S6的初始配额、角色目标、累加器与无spill事实见[程序图 §2](../../../model/mechanism/program/cutlass_program_graph.md#2-对象操作与依赖)。CTA池生命周期以warp为单位：初始申请受阻；一个producer warp释放后，两个consumer warp即可取得配额，第三个继续等待。全部操作结束时，初始与稳态总量同为64512。成功事件记录调用顺序，不赋予硬件仲裁或等待价格。

六项配额检查覆盖守恒、部分释放、三种申请次序、重复确认和CTA隔离；三项程序检查覆盖需求与软件依赖、一次启动，以及4／13／32 Ktile×S4／S6的默认／可选展开回归。新增节点之外，原请求、全部原节点时刻、输出服务与观察端点保持，当前时间预测能力保持。

FIRST_WORK来自thread0，前缀只确认覆盖producer warp0释放；其余producer warp与consumer的申请可能部分重叠，现有观测未给出分项时长。下一项最小匹配观测是实际cfg_b启动时各warp的释放／申请完成与首次计算；操作数源服务则需绑定实际输出消费者后再检验迁移。

## 3. 复现

在仓库根目录执行，输出写入新目录：

```bash
python -B microbench/gh200_resource_campaign/access_rules/replay_register_evidence.py \
  --results-root results --output /tmp/reg01-source-consumer
python -B microbench/gh200_resource_campaign/access_rules/analyze_r16_quota.py \
  --input results/gh200_resource_campaign/access_rules/Q08-register-residency/R16-residency-quota/20261008-R16-quota-job737122-v2 \
  --output /tmp/reg01-quota
PYTHONPATH=microbench/gh200_resource_campaign/access_rules \
  python -B -m unittest test_register_quota test_cfg_b_register_program test_cfg_b_request_model -v
```

[交付包](../../../../../../results/gh200_resource_campaign/access_rules/Q08-register-residency/REG-01-SM-01-replay/20261017-REG-01-replay-001/README.md)另存源码、SHA、实际资源核对、逐warp生命周期和六图接线例子。配额和程序需求已核对，源／消费窗口保持原条件服务，完整cfg_b独立C2/C3仍待验收。
