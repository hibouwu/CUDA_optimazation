# TRANSFER-01：异步搬运接口与旧证据回放

更新：2026-10-17。阶段4完成源／目的接口、独立路径约束修复和有限旧证据回放，没有新增GPU采样。当前服务定义维护在[TMA路径](../../../model/mechanism/hardware_rules/TMA-SHARED-SERVICE.md#stage-4-service)，条件与上限见[PLAN](../../../PLAN.md#stage-4-transfer)。

## 1. 固定证据与完成边界

| 证据族 | 本轮读取与重算 | 保留的完成边界及资格 |
|---|---|---|
| [实际三请求tensor](../../../../../../results/gh200_resource_campaign/access_rules/20261014-tma-tensor-discrimination-001/README.md) | 12资源条件×trace开关×7进程，共168份raw；16项冻结绑定、502656组记录、1938090条cycle/ns软件许可及目的槽复用顺序边；旧评分逐对象相等 | issuer的发出、等待和软件released；性能模式无实际消费者。原生正确性沿用绑定记录，两组冻结价格集合继续失败 |
| [S16 bulk](../../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s16-main-execution/published-v1/) | G2S／S2G×(S,R)为(1,1)/(1,4)/(2,2)×单CTA／整网格，共12坐标、120个已发布样本；重算逻辑字节、分母和中位数，核对发布封套与原审查输入身份 | 含建立、循环、完整排空及必要CTA门槛的限定运输率；保留32条原发布资格和4项容量排除，本轮未重开全payload审查 |
| [cp.async](../../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/async_copy/formal-b3-v2/) | `.ca/.cg`、16B、stage1/2/4、单CTA／整网格，共12坐标、120份raw；核对COMPLETE/C、构建、样本与收据，独立重算工作量、实际周期／ns包络及原中位数 | 复制、相邻线程shared消费和CTA同步的完整窗口率；原C通过，summary的旧pending标签保留；物理L2/HBM流量未辨识 |

S16与cp.async的原设备、资源和计时口径分别保存在[本轮回放](../../../../../../results/gh200_resource_campaign/access_rules/20261017-TRANSFER-01-replay-001/replay/summary.json)中，限定参数留在各自证据族。

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

[交付包](../../../../../../results/gh200_resource_campaign/access_rules/20261017-TRANSFER-01-replay-001/README.md)保存40份Python源码、完整命令、63项通过检查、六个默认小图与两条异步轨迹的修复前后相等记录。另存100 ns最小路径延迟的[具体反例](../../../../../../results/gh200_resource_campaign/access_rules/20261017-TRANSFER-01-replay-001/defect-regression.json)：同步外接源目的就绪0→100 ns，异步源11→100 ns，内置源保持100 ns。合成接口例子的任取价格及信用策略单独标注，未接回GEMM校准。

下一项实机证据仍是实际计算／同步／TMA联合短链。目的就绪与观察返回的间隔、真实消费者释放和源信用反馈需在匹配程序中辨识；本轮保留现有失败范围，未新增独立C2/C3成绩。
