# L1-01：普通LSU证据与局部缓存探针

更新：2026-10-17。阶段8；有限登记见[PLAN](../../../PLAN.md#stage-8-l1-lsu)，服务与接入见[L1／LSU服务](../../../model/mechanism/hardware_rules/L1-LSU-SERVICE.md)。旧回放及CPU准备轮GPU作业与拟合为0；原开发清单的目标实测及独立评分已完成，8留出未启。

## 1. 固定证据与程序

| 证据 | 本轮范围 | 核对内容 |
|---|---|---|
| [R01](../Q08-register-residency/R01-readiness.md) global | 10条件／37进程 | 单lane.cg依赖链与4warp的1／4条链；原地址见证、窗口、SASS与评分 |
| [EXP-13](../Q03-source-service/EXP-13-global-rw.md) read.ca／read.cg | small／large四条件／40进程 | 同体cache对照、工作量、原始记录及完整窗口；写路径沿用[HBM-01](../Q03-source-service/HBM-01-replay.md) |
| [EXP-18](../Q08-register-residency/EXP-18-occupancy-aux.md) local／spill | 32／128 word四条件／40进程 | 实际LDL／STL及资源、原校验资格、完整cycle/loop窗口 |

实际cfg_b的普通访存需求与条件见[程序图 §2](../../../model/mechanism/program/cutlass_program_graph.md#2-对象操作与依赖)，构建核对、旧记录回放及源码保存在[交付包](../../../../../../results/gh200_resource_campaign/access_rules/20261017-L1-01-replay-001/README.md)。上述117份旧记录保持原条件与资格。

回放后，R01逐值校核4640个输出；EXP-13读取对照各184条SASS，只有PC0x410的cache语义与一个编码位不同。local／spill主指标与伴随指标精确相等，两处暖机CV的1 ULP差异逐项匹配原C封存诊断，原阈值资格保持。正式local／spill raw未保存输出数组，当前承接原host检查资格。保存源码的30项CPU检查通过，六图启用需求报告后原完整结果保持。

## 2. 新探针条件与观察量

固定单CTA×128线程、16 B向量节点；节点包含下一地址与三个payload word。J=1时每线程一条地址依赖链，J=4时四条分开的链；每条链仍由上次结果决定下一地址。各条件均完整遍历同一个CTA私有工作集W，并在目标kernel内部完成一次准备遍历后计时256次完整遍历；总逻辑读取量为`256×W`。

| 分组 | W | J | cache policy | preferred carveout | 条件数 |
|---|---|---|---|---|---|
| 开发 | 16／256 KiB | 1／4 | .ca／.cg | 100 | 8 |
| 开发 | 64 KiB | 4 | .ca／.cg | 0／100 | 4 |
| 留出 | 128 KiB | 1／4 | .ca／.cg | 0／100 | 8 |

原准备清单每条件3独立进程，合计60；现行轮仅采12开发条件的36进程，8留出不启。先导`w16_h100_cg_j1`／seed3复用开发点；600秒分配包含目标构建与开发，最多两次定向实现修复。观察量为同一CTA的clock64／globaltimer内窗、包含准备与诊断的CUDAevent、前后SMID、每线程准备与计时checksum、函数资源与请求的carveout。具体计时包含的循环、消费与会合边界以保存源码及SASS为准；carveout字段记录请求值。

两项假设方向筛查为工作集／carveout改变时的.ca／.cg差别，以及固定读取量下J1→4的响应。目标CUDA12.9先核对同体指令与资源，再在同A的GPU UUID执行并核对每进程768个checksum及单CTA资格。主clock64内窗、伴globaltimer，跨W按`cycle/(256W)`比较，event保持完整kernel边界；固定5%相对内窗变化仅作描述响应门槛，同点3进程范围一并报告。原“先导后冻结幅度再留出”是未执行预案；旧117份观测不足以冻结新链绝对幅度，本轮不从先导编造幅度／置信界，不拟合、不追加条件／候选，不称C2数值判别或独立验收。

本地CUDA13.0静态编译采用GCC16兼容头，J1／J4分别为22／32 reg/thread，均512 B SMEM、无stack／spill。.ca／.cg指令体只差缓存修饰；J4正式循环四条16 B加载均先于第一次消费。编译收据、SASS、兼容头及可重建审查脚本保存在交付包；目标CUDA12.9已核对同一条件，四特化静态审查通过，资源同为J1／J4的22／32 reg/thread、512 B SMEM、无spill。

[开发摘要](../../../../../../results/gh200_resource_campaign/access_rules/20261017-L1-01-execution-001/dev/summary.json)的36进程全部合格，与A同GPU UUID；[回传验证](../../../../../../results/gh200_resource_campaign/access_rules/20261017-L1-01-execution-001/transport-verified.json)核对228文件／1703550 B全SHA一致。[独立评分](../../../../../../results/gh200_resource_campaign/access_rules/20261017-L1-01-score-001/score.json)重算全部收据、目标静态审查及16项响应，与原摘要精确相同；逐节点与payload枚举27648个checksum，与每进程768个检查全部一致，未调用原闭式参考。

下表为clock64内窗的三进程中位数及[min,max]，单位cycle；最后一列按`.ca/.cg−1`计算。跨工作集比较另按`cycle/(256W)`归一化，globaltimer与完整event保留在[原响应](../../../../../../results/gh200_resource_campaign/access_rules/20261017-L1-01-execution-001/dev/response-screen.json)。

| W KiB | 请求carveout | J | .ca中位数[min,max] | .cg中位数[min,max] | .ca相对.cg |
|---:|---:|---:|---:|---:|---:|
| 16 | 100 | 1 | 160838 [160837,160841] | 703409 [703072,703863] | −77.134498% |
| 16 | 100 | 4 | 72099 [72097,72103] | 214069 [213855,214090] | −66.319738% |
| 256 | 100 | 1 | 11010615 [11009147,11011405] | 11008567 [11007237,11011889] | +0.018604% |
| 256 | 100 | 4 | 3250066 [3249737,3250159] | 3249054 [3248112,3249647] | +0.031148% |
| 64 | 0 | 4 | 250286 [250283,250304] | 824829 [823477,825700] | −69.656014% |
| 64 | 100 | 4 | 824396 [823277,825216] | 822748 [821512,824185] | +0.200304% |

固定W下，J1→4使内窗减少55.173%–70.486%，三进程范围均不重叠。W64／J4请求carveout由100改0，.ca窗口减少69.640%，.cg增加0.253%且两范围重叠。W256的两项cache差均小于预置5%描述门槛；J1范围重叠，J4虽只有小幅差异但范围不重叠，不能统称为噪声。观测支持小工作集／请求carveout下的条件cache响应和多链重叠响应；`actual_carveout=null`，不反推实际分区容量、L1驻留或物理流量。本轮未冻结绝对时间预测、未拟合，8留出未采，不新增通用load价格或独立C2/C3成绩。

## 3. 复现

在仓库根目录执行，输出写入新目录：

```bash
python -B microbench/gh200_resource_campaign/access_rules/replay_l1_lsu_evidence.py \
  --results-root results --output /tmp/l1-lsu-evidence
python -B microbench/gh200_resource_campaign/access_rules/audit_cfg_b_lsu.py \
  --repo . --output /tmp/l1-lsu-build
PYTHONPATH=microbench/gh200_resource_campaign/access_rules \
  python -B -m unittest test_cfg_b_lsu_program test_l1_lsu_probe -v
```

采样入口与本地静态编译证据见[交付包](../../../../../../results/gh200_resource_campaign/access_rules/20261017-L1-01-replay-001/README.md)。本阶段不扩大到原子、地址翻译或多SM放置。
