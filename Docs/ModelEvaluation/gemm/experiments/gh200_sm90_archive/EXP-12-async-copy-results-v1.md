# EXP-12：cp.async 有限流水线实测

本次 [formal-b3-v2](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/async_copy/formal-b3-v2/summary.json) 包含24个配置、每点10个独立进程，共240个正式样本。全部在首批满足批内及累计CV≤5%，没有追加批次、中断或恢复；240个进程均在第8个预热窗口收敛。按实际kernel调用合计1920次预热与240次测量，不能把预热当作额外正式样本。作业730480用时3分40秒、exit0；进程结果覆盖区间约116.108秒，二者均不是下文kernel吞吐分母。

此前的 [formal-b3-v1失败记录](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/async_copy/formal-b3-v1/campaign_status.json) 保留：运行设备与B3身份不同，拒绝发生在正式样本启动前，batches中没有raw。v2使用新的run ID，并在冻结前另核对分配UUID；没有覆盖或恢复这份失败归档。

本报告从原始字段重新计算运输量和统计，等待未参与报告实现者独立C。原[完整B](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/reviews/S12-B-review.json)与[B3](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/reviews/S12-early-validation-B3-review.json)均保留。

## 计算与计时边界

每CTA有128个线程。每线程每轮发8次copy请求，单次请求为4/8/16B；`.ca`覆盖三种宽度，`.cg`只覆盖16B。stage取1/2/4，正式长度固定8192轮，无pilot或长度拟合。两种scope与12种请求条件组合为24点。

整个launch共享一份8MiB只读全局source，各CTA拥有独立shared环形槽。第step次请求的源偏移为：

`((block×128+thread+step×blocks×128)×request_bytes) % 8388608`。

每个请求独立commit。流水线先填入stage个槽，消费最老槽前按当前剩余group数等待，然后CTA会合；每线程读取相邻线程`(thread+1)%128`复制的全部uint32字，累加模2³² checksum。另一次CTA会合防止未读完就覆盖，再补入新请求。尾部逐步降低wait深度，最终wait0排空。

计时包括prefill、稳态循环、尾部排空、consumer读取及CTA同步；初始化和最后host读回不在计时内。它测的是这套复制与消费者协议的整体服务，不能当作裸copy指令延迟或独立异步复制引擎峰值。

主工作量为：

\[
Q=\mathrm{blocks}\times128\times8192\times8\times\mathrm{request\_bytes}.
\]

`work_count=Q`只计一次。`read_payload_bytes=Q`表示GMEM侧有效读取，`write_payload_bytes=Q`表示SMEM侧有效写入；两者是同一批运输数据的两端，不能相加成2Q。`consumer_read_bytes=Q`是随后由普通指令完成的shared读取，另列需求，也不加进运输分子。

单CTA分母为该CTA `stop_cycle-start_cycle`，单位为B_transport/clock64_cycle/CTA。整卡分母为`max(stop_ns)-min(start_ns)`，单位为B/ns，即十进制GB/s_transport/GPU。两种时钟域不互相换算，不跨CTA相减clock64；CUDA event只是辅助包围计时。

## 正确性与配置条件

短B3已经覆盖全24点的1/2轮，检查所有消费word及最终槽。正式模式每次启动重新poison输出，host检查每个线程的模checksum及全部最终槽，检查数为：

`blocks×128×(1+stages×request_bytes/4)`。

正式raw保留参考方法、checked count与错误数；它没有保存每一步的全部消费轨迹，因此不能从正式raw重新换一套数值参考。源码和独立参考的哈希绑定仍是长循环数值证据的一部分。

stage增加同时增加shared槽空间；宽度改变复制量和消费者读取量。`.ca/.cg`只是请求条件，不能据名称判断实际命中层级。8MiB source、实际L2容量、寄存器、静态/动态shared、local、occupancy和grid均须随结果列出；容量关系不能证明缓存命中，更不能把有效运输量当作物理HBM字节。

## 正式结果与图表

设备记录为NVIDIA GH200 120GB、SM90、132个SM；实际`global_memory_bytes=102005473280`、L2为62914560B（60MiB）。工具链为nvcc12.9.41、CUDA runtime12090、驱动590.48.01（CUDA driver API版本13010）。所有点静态shared和local均为0，occupancy上限均为16CTA/SM；整卡按`132×min(4,16)=528`个CTA启动，实际覆盖132个SM。单CTA只覆盖一个实际SM。

下表都是10个进程速率的中位数，保留三个stage，不挑最快配置代替矩阵。完整min/max、样本CV、每进程原始来源及资源列见CSV。



单CTA单位：**B_transport/clock64_cycle/CTA**。

| 请求条件 | stage1 | stage2 | stage4 |
|---|---:|---:|---:|
| .ca, 4B/thread | 1.285541 | 2.255461 | 2.818244 |
| .ca, 8B/thread | 2.316038 | 4.047356 | 5.298747 |
| .ca, 16B/thread | 3.961247 | 6.460462 | 7.529222 |
| .cg, 16B/thread | 4.078937 | 6.460473 | 7.529243 |

整卡单位：**GB/s_transport/GPU**（有效运输B/ns，不是物理HBM带宽）。

| 请求条件 | stage1 | stage2 | stage4 |
|---|---:|---:|---:|
| .ca, 4B/thread | 1266.900 | 1748.874 | 2014.852 |
| .ca, 8B/thread | 2236.738 | 3267.811 | 3936.776 |
| .ca, 16B/thread | 3688.129 | 4613.794 | 4734.195 |
| .cg, 16B/thread | 3849.708 | 5098.202 | 5303.622 |

stage4/stage1的单CTA中位数比值依次为2.192、2.288、1.901、1.846；整卡为1.590、1.760、1.284、1.378。对固定宽度和cache modifier，增加缓冲阶段在本次条件下提高有效运输率，但幅度依赖scope与请求宽度。

16B整卡从stage2到stage4的增幅较小：`.ca`约2.61%，`.cg`约4.03%。同为16B时，`.cg/.ca`的整卡中位数比在stage1/2/4分别为1.044、1.105、1.120；单CTA stage1约1.030，stage2/4在六位有效数字内几乎重合。这里没有计数器证据能把差别唯一归因于某一级缓存或队列。stage变化也伴随编译后寄存器数及shared大小变化，不能把整条曲线解释成一个输入无关的硬件常数。

| 请求条件 | stage1/2/4寄存器每线程 | stage1/2/4动态shared B |
|---|---|---|
| .ca 4B | 26 / 29 / 29 | 512 / 1024 / 2048 |
| .ca 8B | 27 / 27 / 26 | 1024 / 2048 / 4096 |
| .ca 16B | 32 / 29 / 30 | 2048 / 4096 / 8192 |
| .cg 16B | 32 / 29 / 30 | 2048 / 4096 / 8192 |

三个图使用全部240个通过预热的正式样本。横向微移仅为分开同stage的散点；线连接中位数，误差线是实际min/max，不是置信区间。所有case只有一批，样本CV最大为0.2828%，明显低于5%协议界限；这不保证另一设备、频率、输入工作集或同步结构也具有相同速率。

![单CTA各阶段](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/async-copy-formal-b3-v2/stages-one_cta.png)

![整卡各阶段](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/async-copy-formal-b3-v2/stages-all_gpu.png)

![所有case样本CV](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/async-copy-formal-b3-v2/process-cv.png)

NCU状态为`permission_denied`，且只属于权限能力记录。其receipt与此前MMA归档逐字节相同，是已有权限结果的复用，不是新的S12 profiling。因此`.ca/.cg`的实际命中率、实际HBM字节和物理内存带宽均未得到计数器证明。

本机Python3.14冻结严格审计报告浮点复算差异；[受审bounded重放](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/post_audits/bounded-float-replay-v1-r2/async_copy-formal-b3-v2.json)对720个允许派生浮点字段检查，34处存在差异，最大2ULP。264项阈值按原始整数时间和精确速率重新判定，没有容差或状态覆盖；整数、身份、raw及全部11399项输入文件仍严格绑定。原strict失败保留，不能写成“本机原严格审计已通过”。

## 原始字段手算与复现

报告入口为 `microbench/gh200_resource_campaign/report_async_copy_v1.py`，要求外部只读bounded replay先通过且全量输入hash完全相同，然后从raw重算Q、分母、统计与固定条件比较。在新分析目录冻结脚本和所有输出；不写入原运行，不执行GPU。

手算固定选取下列case的首个预热通过记录，本次均为batch0/trial0，未选择最快样本。完整四例及原始hash见[手算记录](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/async-copy-formal-b3-v2/manual-examples.md)。

`ca_w4_stages1_one_cta`：Q=`1×128×8192×8×4=33554432 B`。原始clock64为`18467920533509−18467894432170=26101339`，所以单样本速率为`33554432/26101339=1.28554446958 B/CTA-cycle`。这一个样本不等于表中的十样本中位数1.285541。

`ca_w16_stages4_all_gpu`：Q=`528×128×8192×8×16=70866960384 B`。原始globaltimer包络为`1790861693837858752−1790861693822891840=14966912 ns`，所以单样本速率为`70866960384/14966912=4734.90860266 GB/s`。GMEM读Q、SMEM写Q及consumer读Q分别记录，不能把分子改为2Q或3Q。该启动的全量检查数为`528×128×(1+4×4)=1148928`。240份正式输出合计检查60263680个checksum/最终槽元素；没有把这些元素数当作独立样本数。

`ca_w16_stages4_one_cta`与`cg_w16_stages4_all_gpu`另外两例分别为7.52922084989 B/CTA-cycle和5296.07901338 GB/s，原始字段、确切分数和checked count均在同一手算工件中。

## 工件与重生成

- [逐进程trials.csv](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/async-copy-formal-b3-v2/trials.csv)：240样本、Q四列、时钟域、原始分子/分母、seed、资源、receipt及raw哈希。
- [全部case统计](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/async-copy-formal-b3-v2/cases.csv)与[全部批次](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/async-copy-formal-b3-v2/batches.csv)：24配置/24批，不筛选最佳批。
- [固定条件比值](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/async-copy-formal-b3-v2/comparisons.csv)：24组stage比较与6组同16B cache条件比较，仍待C。
- [分析manifest](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/async-copy-formal-b3-v2/manifest.json)：全11399项输入身份、全部输出及重放receipt哈希。
- [正式作业入口](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/job-730480/driver-v2.sh)、[收集记录](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/job-730480/collection.json)与[UUID前置门禁](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/job-730480/allocation-gate-730480.json)：只作启动/收集来源，不替代原始审计。

从仓库根目录，在全新目录重生成；报告入口不改run，也不启动GPU：

```bash
python -B results/gh200_resource_campaign/20261001-resource-suite-v2/analysis/async-copy-formal-b3-v2/report_async_copy_v1.py \
  results/gh200_resource_campaign/20261001-resource-suite-v2/async_copy/formal-b3-v2 \
  --replay results/gh200_resource_campaign/20261001-resource-suite-v2/post_audits/bounded-float-replay-v1-r2/async_copy-formal-b3-v2.json \
  --output /tmp/gh200-s12-analysis-reproduced
```

上述命令针对分析manifest绑定的收集版本。以后若finalize增加C审查、COMPLETE或状态记录，旧receipt的完整文件清单会按设计拒绝新版归档；可从保留的收集包恢复该版本到另一目录复现，或对当前版本另做只读replay并输出到新分析目录，不能改旧receipt来通过检查。

同一matplotlib3.10.8和字体环境下，PNG/SVG及manifest采用确定性输出；图像跨环境字节不作保证。复算时不能修改原receipt、raw、summary或工作区里的冻结源来“消除”浮点差异。
