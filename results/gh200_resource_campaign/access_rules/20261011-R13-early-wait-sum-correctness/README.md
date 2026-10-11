# R13 累计等待四目标正确性检查

固定构建job740240的stage4/6两个binary，每stage的observer0/1各一次，四个独立进程顺序为s4/off、s4/on、s6/off、s6/on；每次目标前有16次同buffer、同stage binary的observer0前序。

只核对指定ef GPU、声明参数、各目标4096个唯一dyadic值、padding/nonfinite、完整同调用记录、15组累计mask/count及端点序；并不验证全D，也没有性能通过门槛。失败停止并保留，禁止替换样本。时长和Q只作为smoke诊断保存，不能估计stage效应、观测扰动或服务价格。全部观察、预测与留出资格仍为false。

manifest和freeze在提交前保存，二进制/脚本/解析器均固定SHA；a058排他分配计实际4GPU/288CPU，仅一个计时流。collect在Slurm设置的可见设备上先用setup核实际GPU，不覆盖CUDA_VISIBLE_DEVICES。命令与结果在本目录保存。

## 实际结果

原job740259仅完成stage4/off，binary返回0；collector因额外的`input_zero_storage`包拒绝而退出1。随后离线还查明原reader误把host输出的二维cluster写成三维。原manifest、freeze、返回包、失败状态保持；新reader只接受四个声明包、输入检查0/0及cluster=[1,1]，不丢弃任意额外包或宽接受多种形状。

修复后重新冻结的remaining3由job740276执行，只运行未执行的三个原目标；两批合计恰好原四目标，无重复或替换。两次均为a058排他原GPU-ef，观测线程均SM124。原stage4/off按新reader离线复核有效，但不追认740259冻结流程通过。

| 原目标 | C16−C1 / cycle | Q / cycle | 等待外 / cycle | 独立数值样本 |
|---|---:|---:|---:|---:|
| stage4/off | 11500 | 未观测 | 未观测 | 4096 |
| stage4/on | 10739 | 4633 | 6106 | 4096 |
| stage6/off | 7982 | 未观测 | 未观测 | 4096 |
| stage6/on | 8582 | 1301 | 7281 | 4096 |

两个on均count15、mask0x1fffc、invalid0，端点包含链通过；四目标合计16384个唯一位置的逐目标样本精确正确。这里“唯一”在各自调用内成立，不声称跨四次调用位置不重复。每条件仅一次、分属两job，不估阶段因果差、观察扰动分布、服务价或性能均值。未执行全D检查。

[verify.py](verify.py)从两份固定SHA回收包核对各11个冻结输入、原进程归属及binary返回码，加载remaining3固定reader重算四条记录；完整结果见[verification.json](verification.json)。复放命令为`python verify.py --output NEW_REPORT`。初次本地复放临时目录层级不足，旧模块的相对根路径解析失败；改为临时frozen/analysis_snapshot布局后通过，未改被验证的decoder或原数据。

C对唯一旧stage4/off另以289类坐标×完整K4096整数dot核对4096值，1,183,744乘积、零差异；[主线原样重放](reanalysis/main-C-first-record-review-v1/data/report.json)逐字同，仅第一条证据，不能扩大为全D。源码/SASS和CPU时间边界复审见[构建记录](../20261011-R13-early-wait-sum-build/README.md)。所有观测、预测、关键CTA及留出资格仍false。
