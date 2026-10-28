# TRANSFER-01：异步搬运接口与旧证据回放

更新：2026-10-17。阶段4完成源／目的接口、独立路径约束修复和有限旧证据回放，没有新增GPU采样。当前服务定义维护在[TMA路径](../../../model/mechanism/hardware_rules/TMA-SHARED-SERVICE.md#stage-4-service)，条件与上限见[PLAN](../../../model/mechanism/hardware_rules/MICROBENCH-ROADMAP.md#6-推进次序与本轮交付)。

## 1. 固定证据与完成边界

| 证据族 | 本轮读取与重算 | 保留的完成边界及资格 |
|---|---|---|
| [实际三请求tensor](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/TRANSFER-01-tensor-2d/20261014-tma-tensor-discrimination-001/README.md) | 12资源条件×trace开关×7进程，共168份raw；16项冻结绑定、502656组记录、1938090条cycle/ns软件许可及目的槽复用顺序边；旧评分逐对象相等 | issuer的发出、等待和软件released；性能模式无实际消费者。原生正确性沿用绑定记录，两组冻结价格集合继续失败 |
| [S16 bulk](../../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s16-main-execution/published-v1/) | G2S／S2G×(S,R)为(1,1)/(1,4)/(2,2)×单CTA／整网格，共12坐标、120个已发布样本；重算逻辑字节、分母和中位数，核对发布封套与原审查输入身份 | 含建立、循环、完整排空及必要CTA门槛的限定运输率；保留32条原发布资格和4项容量排除，本轮未重开全payload审查 |
| [cp.async](../../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/async_copy/formal-b3-v2/) | `.ca/.cg`、16B、stage1/2/4、单CTA／整网格，共12坐标、120份raw；核对COMPLETE/C、构建、样本与收据，独立重算工作量、实际周期／ns包络及原中位数 | 复制、相邻线程shared消费和CTA同步的完整窗口率；原C通过，summary的旧pending标签保留；物理L2/HBM流量未辨识 |

S16与cp.async的原设备、资源和计时口径分别保存在[本轮回放](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/TRANSFER-01-tensor-2d/20261017-TRANSFER-01-replay-001/replay/summary.json)中，限定参数留在各自证据族。

tensor共享ring32的四个原条件展示了Q与打点的影响。单位ns，均为对应进程统计的中位数：

| 条件 | trace=0 CTA窗口 | trace=1 CTA窗口 | trace=1末组release窗口 | trace=1稳态wait中位 |
|---|---:|---:|---:|---:|
| N1、Q1 | 29568 | 32608 | 32512 | 160 |
| N1、Q4 | 19584 | 24352 | 24256 | 32 |
| N132、Q1 | 30848 | 33920 | 33824 | 192 |
| N132、Q4 | 19680 | 24480 | 24384 | 32 |

原相位响应保持trace为+9.84456%、未记录为+10.34393%；两套冻结价格的最佳联合最大误差为119.218%／15.277%，通过集合均为空。局部wait与打点窗口各自保留，本轮没有重新拟合或挑选价格。

## 2. 接线与复现

[回放入口](../../../../../../microbench/gh200_resource_campaign/access_rules/replay_transfer_evidence.py)输出三个证据族及身份摘要；只运行已有冻结评分公式。新增请求组合检查复用现成合成源服务，分别展示源就绪归还与最后目的完成归还的轨迹。真实地址的同步源路径继续使用`CfgBTensorSource`；外部异步服务按CTA／group／atom索引定义自己的源身份，合成合并规则没有成为硬件结论。

在仓库根目录执行，输出目录须尚不存在：

```bash
python -B microbench/gh200_resource_campaign/access_rules/replay_transfer_evidence.py \
  --results-root results --output /tmp/transfer-01-replay
PYTHONPATH=microbench/gh200_resource_campaign/access_rules \
  python -B -m unittest test_tma_transfer_service test_r13_async_source \
  test_mbarrier_service test_cfg_b_request_model test_r13_input_path \
  test_r13_input_path_interval test_r13_input_latency test_r13_destination_latency -v
```

[交付包](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/TRANSFER-01-tensor-2d/20261017-TRANSFER-01-replay-001/README.md)保存40份Python源码、完整命令、63项通过检查、六个默认小图与两条异步轨迹的修复前后相等记录。另存100 ns最小路径延迟的[具体反例](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/TRANSFER-01-tensor-2d/20261017-TRANSFER-01-replay-001/defect-regression.json)：同步外接源目的就绪0→100 ns，异步源11→100 ns，内置源保持100 ns。合成接口例子的任取价格及信用策略单独标注，未接回GEMM校准。

下一项实机证据仍是实际计算／同步／TMA联合短链。目的就绪与观察返回的间隔、真实消费者释放和源信用反馈需在匹配程序中辨识；本轮保留现有失败范围，未新增独立C2/C3成绩。

[多CTA producer控制输入](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/source-organization/20261019-Q02-producer-control-input-001/README.md)已补齐独立入口已有的`producer_gap_ns`与`operand_gap_ns`透传；间隔从请求受理返回后计，四形状的独立／组合请求、端点和槽状态一致。默认0保持，实际硬件取值仍待局部证据约束。

## 5. 固定分配的B行距开发判别

[新准备轮次](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-shape-response/20261017-Q02-tma-stride-001/README.md)固定B分配与真实tensor请求，比较四行距及Q1／Q4端点。范围为首两轮源坐标，当前只做CPU准备。

目标执行包见[002](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-shape-response/20261017-Q02-tma-stride-002/README.md)：先核对原生12.9前后构建及18臂完整数值，再采同6条件的18臂plain；独立评分从原始NDJSON重算。

751907完成[目标执行与开发判定](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/request-shape-response/20261017-Q02-tma-stride-002/EXECUTION.md)：18数值臂零错误、18plain臂完成；Q4下B32近零，B80/B112中位cycle增加约4%–5%、ns包络增加约7%–11%，Q1的包络响应不稳定同向。固定分配下地址响应仍存在且受在途组织影响，先据此筛查源服务规则；未新增独立C2或组合价格。

## 相关轮次

以下各轮原登记在访问规则索引，按日期排列；全部结果目录见[Q02-tma-request/TRANSFER-01-tensor-2d](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/TRANSFER-01-tensor-2d/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| tensor 首批冻结判别（746425） | 固定三请求、二进制与物理四槽，改变 CTA 数、活动组数、源共享及相位；比较在途合并与工作集驻留 | 两套冻结价格集合均未通过；相位响应约 +9.84%，在途合并／驻留候选的最佳联合最大误差为 119.22%／15.28%；转入服务率与完成延迟分开建模 | [本轮记录](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/TRANSFER-01-tensor-2d/20261014-tma-tensor-discrimination-001/README.md)、[评分](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/TRANSFER-01-tensor-2d/20261014-tma-tensor-discrimination-001/score.json)、[预测与绑定](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/TRANSFER-01-tensor-2d/20261014-tma-tensor-discrimination-001/binding.json)；[原生检查、控制修正与 Q2 校准](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/TRANSFER-01-tensor-2d/20261014-tma-tensor-development/README.md)属于本轮准备 |
| tensor 第二次冻结留出（746459） | 固定 P16 候选，检验新相位、N32/96 的 Q1/Q4 和消费者释放延迟，共 8 条件 | 相位预测 +8.965%、实测 −0.187%，P16 被反证；局部时间最大误差 41.45%，不再采用该版本 | [本轮记录](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/TRANSFER-01-tensor-2d/20261014-tma-tensor-heldout-002/README.md)、[评分](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/TRANSFER-01-tensor-2d/20261014-tma-tensor-heldout-002/score.json)、[冻结预测](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/TRANSFER-01-tensor-2d/20261014-tma-tensor-heldout-002/predictions.json) |
| B 行跨度冻结对照（746430） | 固定单 CTA、请求、实际源字节并集与预热，只改 ldb 128→2048 | 实测窗口 +0.0984%、未打点为零，不支持 +11.66% 的独立行距延迟；不增加该参数 | [本轮记录](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/TRANSFER-01-tensor-2d/20261014-tma-pitch-discrimination-001/README.md)、[评分](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/TRANSFER-01-tensor-2d/20261014-tma-pitch-discrimination-001/score.json)、[冻结条件](../../../../../../results/gh200_resource_campaign/access_rules/Q02-tma-request/TRANSFER-01-tensor-2d/20261014-tma-pitch-discrimination-001/plan.json) |
