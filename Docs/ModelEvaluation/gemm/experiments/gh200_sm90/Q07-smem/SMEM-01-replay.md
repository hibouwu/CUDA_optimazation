# SMEM-01：完整需求与旧证据回放

更新：2026-10-17。阶段6离线交付，未新增GPU采样或拟合。有限登记见[PLAN](../../../PLAN.md#stage-6-smem)，程序需求和写块假设见[SMEM服务](../../../model/mechanism/hardware_rules/SMEM-SERVICE.md)。

## 1. 固定证据与原结论

| 证据族 | 固定读取范围 | 本轮复核与保留结果 |
|---|---|---|
| [m11 v1](../../../../../../results/gh200_resource_campaign/access_rules/20261012-m11-holdout/) | 60条件、180进程，50支持／10诊断 | 冻结需求、预测与支持标签逐float相等；原中位4.5524%、最大34.8208%，未通过 |
| [m11 v2](../../../../../../results/gh200_resource_campaign/access_rules/20261012-m11-holdout-v2/) | 66条件、198进程，47支持／19诊断 | 同样严格复现；原中位0.4400%、最大11.5972%，唯一超标仍为随机8 B／1 warp独立读 |
| [lane对照](../../../../../../results/gh200_resource_campaign/access_rules/20261012-m11-lane-contrast/run-20261012T110219Z/) | 48条件、144进程，14排列对、20桥接 | 全word集合、多重度和字节不变的排列反例保持，12/14对不相容；16 B主对照4194515／16777430 cycle，接近4倍 |
| [R03](../../../../../../results/gh200_resource_campaign/access_rules/20261006-c-job735059/r03-formal-v1/) | 18条件、72记录（18 pilot、54正式） | 独立重算124800个uint32；原rules与CSV数值相等，普通LDS/STS和矩阵访存各自保留 |
| [R12 v3](../../../../../../results/gh200_resource_campaign/access_rules/20261008-R12-job736989-v3/) | 24正式条件、18阶段／空控制，共174记录 | 独立检查14966784个值；42 kernel与六组阶段对照复现，保留TMA／WGMMA近重叠及与STS的条件干扰 |

合计768份旧进程／记录。留出评分按原支持标签和门槛重算，不执行旧评分脚本的写入主入口；已看过的留出保留为开发证据。两份留出的真实CUTLASS iterator实测条件数均为0，当前cfg_b地址核对另以静态证据登记。

有两处CV重算相差1 ULP：lane的`dense_contiguous_k4`和R12的`tma_wgmma_SW128_serial`。新旧数值与原门槛判断保存在回放结果中，其他评分／窗口字段严格相等，没有将对应JSON宣称为逐字节相同。R03旧SASS audit将`Li16`误解析成`width=1`，本轮保留旧文件，从完整符号及取时之间的主循环重新确认实际宽度和8条目标访问。

## 2. cfg_b程序与接口核对

冻结S6／S4 SASS均有64条STS.64、8个8 KiB输出子块，没有LDS/LDSM/STSM。模板名中的`SM90_U32x4_STSM_N`用于构造复制布局，FP32实际路径选择普通向量写，详细出处见[程序图](../../../model/mechanism/program/cutlass_program_graph.md#2-对象操作与依赖)。

CPU CuTe查询与SASS地址链逐项比较128线程×两个源槽×16个word，共4096项，差异为0。每槽2048个word各出现一次：slot0为shared字节`[0x800,0x2800)`，slot1为`[0x2800,0x4800)`；输出子块按奇偶复用两槽。完整lane／word需求和四类服务覆盖清单已可生成。

七项SMEM检查覆盖lane排列、宽同址关系、写别名、逻辑字节、两槽覆盖、价格替换和六个4／13／32 Ktile×S4／S6小程序。等价写块价格逐对象复现默认结果；改变写块价格只替换对应服务，输出提交与源槽复用顺序保持。价格为CPU人工输入，原完整GEMM失败和验收状态保留。

## 3. 复现与剩余观测

[交付包](../../../../../../results/gh200_resource_campaign/access_rules/20261017-SMEM-01-replay-001/README.md)保存源码、CuTe查询与构建证据、完整地址、服务例子和旧证据回放。在仓库根目录执行，输出目录须尚不存在：

```bash
python -B microbench/gh200_resource_campaign/access_rules/replay_smem_evidence.py \
  --results-root results --output /tmp/smem01-m11-replay
python -B microbench/gh200_resource_campaign/access_rules/replay_smem_protocols.py \
  --results-root results --output /tmp/smem01-protocol-replay
PYTHONPATH=microbench/gh200_resource_campaign/access_rules \
  python -B -m unittest test_smem_service test_cfg_b_request_model -v
```

剩余实机证据为该实际STS.64短块的条件周期、准备／写入／源读取端点，以及与计算和TMA共同执行的迁移。旧m11标量失败不恢复搜索；本轮未新增通用bank批次价格或独立C2/C3成绩。
