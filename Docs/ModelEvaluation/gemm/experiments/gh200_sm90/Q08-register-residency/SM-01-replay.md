# SM-01：发射联合窗口与驻留容量

更新：2026-10-17。阶段9；有限登记见[PLAN](../../../model/mechanism/hardware_rules/MICROBENCH-ROADMAP.md#6-推进次序与本轮交付)，模型入口见[联合服务](../../../model/mechanism/hardware_rules/joint_service.md)和[驻留服务](../../../model/mechanism/hardware_rules/residency.md)。本轮GPU作业与拟合为0。

## 1. 固定证据与边界

| 证据 | 本轮范围 | 核对内容 |
|---|---|---|
| [R06](R06-issue-residency.md) 展开版 | 6条件／39正式＋12短记录 | 单CTA与整卡FFMA；源码／机器码、21914496个值、SM窗口与原评分 |
| [R16](R16-residency-quota.md) 驻留 | 12条件／120正式＋24短记录 | stage1／2／4×源规模×容量1／2；467140608个值、窗口重叠与原评分；配额沿用REG-01 |
| [R11](../Q07-smem/R11-mixed-issue.md) v6 | 两组／12正式条件，36正式＋24短＋12控制记录 | FFMA＋IMAD.WIDE、WGMMA＋IMAD；1486080个u64、匹配B与空控制、原SASS和联合比较 |
| [R12](../Q07-smem/R12-smem-path-contention.md) | 复用阶段6的24正式／18控制条件、174记录 | 固定快照SHA、原始身份、控制与已知CV差异；本轮未重复解压 |

本轮新回放合计30正式条件、267份旧记录、490541184个值。R06保留原短见证的暖机不收敛；R16两项CV相邻1 ULP差异逐项保存，其他字段严格相等，原5%资格保持。R12沿用已登记的一项CV差异。见[交付包](../../../../../../results/gh200_resource_campaign/access_rules/Q08-register-residency/REG-01-SM-01-replay/20261017-SM-01-replay-001/README.md)。

R06计时是FFMA及其消费／会合窗口。R16驻留窗口包含bulk TMA、ready等待、计算、commit／wait0、释放、末尾归约与global输出；最后输入槽保存位于窗外。R11/R12提供保留消费者和同步的整CTA窗口，各自的循环、资源、工作与完成边界随合同保存。

## 2. 判别结果与目标程序

R06编译寄存器63／95／128对应CUDA容量8／5／4 CTA/SM，整卡均值为34028.8／34224／34553.6 ns，容量8→4只增加1.54%；实际正式窗口每SM仍有4个CTA。R16同stage只改动态SMEM预留，实测窗口峰值随容量1→2变为1→2：小源stage1耗时48112→34128 ns，stage4为32272→27792 ns；大独立源stage4为55024→55312 ns。固定occupancy倒数缩放被这些对照排除。

匹配消费者后，R11交错与跨组的FFMA＋IMAD窗口比对应较慢单路高约22.62%；WGMMA＋IMAD分别为+0.0678%／−0.1509%。R12 SW128并发相对匹配阶段max的余量为TMA＋WGMMA 0.2327%、WGMMA＋STS 9.7345%、TMA＋STS 6.2601%。统一取max被部分配对排除；完整条件窗口保留，物理争用位置尚未区分。

实际cfg_b的构建容量、历史API及G3两份dual的SMID核对见[驻留服务](../../../model/mechanism/hardware_rules/residency.md)。当前可选活动报告只读取同版求解结果；其计算服务、输出准备／写块、输入在途与重叠均保留原边界。36份联合服务合同覆盖原R11/R12程序，cfg_b的寄存器帧、请求形状和相位尚无匹配合同，原时间价格保持。

## 3. 复现与缺口

在仓库根目录执行，输出写入新目录：

```bash
python -B microbench/gh200_resource_campaign/access_rules/replay_sm_residency.py \
  --results-root results --output /tmp/sm-residency-evidence
python -B microbench/gh200_resource_campaign/access_rules/replay_sm_joint.py \
  --results-root results --output /tmp/sm-joint-evidence
python -B results/gh200_resource_campaign/access_rules/Q08-register-residency/REG-01-SM-01-replay/20261017-SM-01-replay-001/reproduce_examples.py \
  --repo . --output /tmp/sm-01-examples
```

容量、完整合同及六图接线的29项CPU检查通过。代码审查修复了开启活动报告时一次性CTA选择迭代器被重复消费的问题；新增反例回归通过，默认端点保持。

本轮释放时点问题复用[R16配额](R16-residency-quota.md)即可：旧探针与cfg_b同为384线程、初始168、producer40、两consumer232。每producer warp释放4096个配额，可供两个consumer warp各申请2048。固定256个producer动作，仅移至dec前／后，consumer inc中位窗口为7866.5／144.5 cycle，分配后FADD为787.0／788.0 cycle；每条件10正式进程，已由[REG-01](REG-01-replay.md#1-固定证据)核对。释放前驱进入申请窗口的规则获得旧干预支持，不重复相同测量。实际cfg_b逐warp端点仍未取得，不把该复合窗口加到前缀。

最小实机缺口为cfg_b实际WGMMA与前输出STS.64的相位配对、计算完成／槽释放及启动分布。下一阶段汇成同版cfg_b组合，复用原12条件与失败，实际迁移和独立C2/C3继续等待对应证据。
