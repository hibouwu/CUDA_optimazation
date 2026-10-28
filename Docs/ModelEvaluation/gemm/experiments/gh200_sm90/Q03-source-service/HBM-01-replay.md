# HBM-01：global读写旧证据回放

更新：2026-10-17。本页记录阶段5的固定证据、窗口重算与两种纯端点外推；共享资源定义见[HBM服务](../../../model/mechanism/hardware_rules/HBM-SERVICE.md)，有限登记见[PLAN](../../../model/mechanism/hardware_rules/MICROBENCH-ROADMAP.md#6-推进次序与本轮交付)。本轮没有新增GPU采样或价格拟合。

## 1. 条件、源码与原资格

回放[EXP-13](EXP-13-global-rw.md)的9个large及9个small条件，每条件10个独立进程，共180份raw。模式为`read_ca`、`read_cg`、`write`、`dependent_copy`与独立值读写1:1、2:1、4:1、1:2、1:4；固定528 CTA×256线程、16次迭代、16 B访问。large每数组253034496 B，small每数组17301504 B；混合条件分配两个数组，整卡地址并集与单数组大小分别保留。

证据来自[原run](../../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/global_duplex/formal-b3-v1/)，逐条核对raw／receipt SHA、零错误状态、声明检查量、暖机、CTA／SM覆盖、时间包络及原发布统计。身份承接原`COMPLETE`、[C审查](../../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/global_duplex/formal-b3-v1/reviews/C/review.json)的`pass`与strict finalize收据；本轮核验所读文件，不重新扫描整个旧manifest或重做完整输出检查。

源码出处固定为`run_spec.source_path`指向的[snapshot/global_duplex.cu](../../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/global_duplex/formal-b3-v1/snapshot/repo/microbench/gh200_resource_campaign/probes/global_duplex.cu)，并保留原二进制、[SASS](../../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/global_duplex/formal-b3-v1/build/sass.txt)及audit身份。读指令为`ld.global.ca/cg.v4.u32`，混合读使用`.cg`，写为`st.global.wb.v4.u32`；每线程每group先读并立即累加四word checksum，再执行独立值写。copy写实际加载值，协议与独立读写分别记录。

## 2. 计时窗口与两种外推

方向逻辑请求量为\(Q_r=16\,A\,r\)、\(Q_w=16\,A\,w\)，其中\(A\)为实际单数组字节数。原速率为\((Q_r+Q_w)/(\max CTA\ stop_{ns}-\min CTA\ start_{ns})\)。该globaltimer整grid包络含入口取时后的CTA gate、地址循环、checksum／shared排空，含写配置还包含设备fence与CTA排空；它没有给出物理HBM字节或裸指令服务率。

各条件先重算10个进程的请求速率，再沿用冻结helper的精确统计核对[发布CSV](../../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/global-duplex-formal-b3-v1-r2/cases.csv)。表中时间比较以\((Q_r+Q_w)/\overline{rate}\)为参考，是均值速率对应的等效窗口，不能称为原`elapsed_ns`的算术均值。

固定同工作集`read_cg`／`write`均值为\(B_r,B_w\)，不加混合参数，分别检验完整窗口外推\(T_{max}=\max(Q_r/B_r,Q_w/B_w)\)与\(T_{sum}=Q_r/B_r+Q_w/B_w\)。large纯读／纯写均值为3524.313522／3790.410230 GB/s；下表预测率为总逻辑字节除以对应预测时间，时间误差为\(T_{sum}/T_{ref}-1\)。

| large独立读写比例 | 原均值GB/s | `max`预测GB/s | `sum`预测GB/s | `sum`时间误差 |
|---|---:|---:|---:|---:|
| 1:1 | 3427.757376 | 7048.627044 | 3652.521812 | −6.1537% |
| 2:1 | 3591.400960 | 5286.470283 | 3608.761691 | −0.4811% |
| 4:1 | 3671.014939 | 4405.391903 | 3574.501391 | +2.7001% |
| 1:2 | 3386.994644 | 5685.615346 | 3697.356239 | −8.3941% |
| 1:4 | 3327.378882 | 4738.012788 | 3734.024127 | −10.8903% |

1:1的`max`预测时间误差为−51.3699%。纯读十进程最长1152064 ns、纯写最长1072736 ns，相加2224800 ns，仍短于1:1混合最短2359552 ns。两种无额外项的完整窗口外推均存在具体反例；纯／混合循环的动态指令和控制成本不同，结论不排除物理HBM独立方向或共享资源本身。small只作同协议工作集参照，不据其高请求率证明缓存驻留。

## 3. 复现、接线与剩余范围

[交付包](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/HBM-01-backend/20261017-HBM-01-replay-001/)保存源码身份、180份逐进程重算、两种外推及服务接线检查。在仓库根目录执行，输出目录须尚不存在：

```bash
python -B microbench/gh200_resource_campaign/access_rules/replay_hbm_evidence.py \
  --results-root results --include-small --output /tmp/hbm01-replay
PYTHONPATH=microbench/gh200_resource_campaign/access_rules \
  python -B -m unittest test_hbm_service test_tma_stateful_source \
  test_cfg_b_request_model -v
```

25项CPU检查通过，六个4／13／32 Ktile×S4／S6默认请求图新旧结果逐对象相同。服务例子使用调用方声明的价格和字节，未将上述复合窗口率接为裸HBM参数。剩余实测缺口是匹配程序的真实miss字节、写回时点与读写并发平台，以及global窗口到TMA源服务的迁移；本轮没有新增独立C2成绩。

后续[固定1:1两组织轮](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/HBM-01-backend/20261018-Q03-hbm-order-001/README.md)仅比较CTA内交错／读写分阶段窗口，保留实际控制及同步变化；不会把原纯端点外推重新称为已支持形式。

751916的[两组织结果](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/HBM-01-backend/20261018-Q03-hbm-order-001/EXECUTION.md)6进程通过资格，CTA分阶段窗口比交错慢4.144951%，落在冻结5%响应门槛内；保留无额外组织修正项的工作假设，机制判定信息不足。本轮收口，不拆方向或同步价格，原纯端点外推失败保持。

## 相关轮次

以下各轮原登记在访问规则索引，按日期排列；全部结果目录见[Q03-source-service/HBM-01-backend](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/HBM-01-backend/README.md)。

| 研究轮次与检验主张 | 固定与改变的条件 | 改变的判定 | 证据与相关产物 |
|---|---|---|---|
| 源状态是否已抹去stage推进差（CPU） | 固定原数据服务，独立回放两个保存请求流；核每域并集、miss及HBM忙段 | 10296就绪端点精确复现；现版本两stage均全miss持续忙，末组就绪相同；12模块回放、两地一致，GPU0 | [独立模块、定位与收口](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/HBM-01-backend/20261019-Q03-data-service-isolation-001/README.md) |
| 只改变外存共享范围能否修复局部S6（CPU） | 按home／请求域各设两个半速FIFO，固定总能力及缓存／价格／计算／释放；先S6必要检查 | 两固定接法均未过原5%窗口与结束跨度门，S4未运行；45输入及五输出两地一致，66调用、一处复现输出修复、拟合／GPU0 | [两候选、计账与收口](../../../../../../results/gh200_resource_campaign/access_rules/Q03-source-service/HBM-01-backend/20261019-Q03-backend-scope-001/README.md) |
| a044 跨卡复核（mechanism-r3，780769） | CTA 内交错/串行组织各 3 进程 | 串行相对交错 +4.8804%（旧卡 +4.1450%），仍在 5% 内，零组织响应未被反证；距门槛约 0.12 个百分点。节点 NCU 现已可用（RmProfilingAdminOnly:0） | [内存服务报告](../../../../../../results/gh200_resource_campaign/access_rules/non-experiment/records/20261023-a044-remeasurement/mechanism-storage-stage-report.md) |
