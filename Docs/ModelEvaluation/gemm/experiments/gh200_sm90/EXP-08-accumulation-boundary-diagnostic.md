# S08：长累加差异的最小边界诊断（草案）

作者：`/root/gate_reviewer`。当前只完成 CPU 推导和资料核对，待另一代理独立审 A；没有可执行 manifest、source-B 或 GPU 授权。新草案与既有32点合同、源码和8192轮归档分开保存。

## 已知事实与参考边界

原 `wgmma_e4m3_g1_one_cta` 在 uniform/nonuniform 各1/2轮短检查通过。一次独立8192轮诊断的8192输出全部为64，FP32 bits为`0x42800000`；目标496条SASS与短检查一致。数学每个输出每轮增加 `16×32×(1/16)²=2`，故8192轮参考为16384。链数2不再乘进单个输出值。完整证据见[结果审查](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/reviews/S08-long-accumulation-result-review.json)。

CUDA12.9.1归档的 **PTX ISA8.8** WGMMA条款原文是：“When `.dtype` is `.f32`, accumulation of the intermediate values is performed with at least single precision.” 同一段另说明累加次序、舍入和次正规数处理未指定，但没有FP8专门例外。[PTX8.8原文](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma)

当前官方 **PTX ISA9.4** 则对E4M3/E5M2输入、F32输出专门补充：“current implementation does accumulation at higher than half precision but lower than single precision.” 该澄清列于9.2变更记录。它不能冒称PTX8.8已有原句，也没有给出本设备精确尾数、分组或舍入方式。[当前WGMMA条款](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma)、[9.2变更记录](https://docs.nvidia.com/cuda/parallel-thread-execution/#changes-in-ptx-isa-version-9-2)

因此保留版本差异：IEEE FP32逐项或逐dot模型可得到数学参考，但并不足以把它当作当前FP8 WGMMA长累加的准确性判据。64的实测值本身也不足以证明某种精度模型合法或kernel有错。原严格参考不改、不加容差。

## 第一轮：四次原kernel诊断

只改变运行时iterations，保留原seed3、A=B=1/16、D0、shape64×64×32、两链、每链batch16、单CTA128threads和wait0。

| 新profile | 轮数 | 数学参考/元素 | 新target次数 |
|---|---:|---:|---:|
| original_uniform_31 | 31 | 62 | 1 |
| original_uniform_32 | 32 | 64 | 1 |
| original_uniform_33 | 33 | 66 | 1 |
| original_uniform_64 | 64 | 128 | 1 |

已观测64对应数学第32轮，因此四点检验31/32/33的边界并用64确认更远处是否继续增长。复用已有1/2/8192记录，不再测8192，不默认单调性，也不自动二分。每个profile独立进程、一次target，总计4次、32768个完整输出；无warmup/pilot/辅助kernel。长度虽不超过64，但这是单独诊断，不能自动算作现有early-validation或family B3。

新host wrapper必须直接包含冻结原设备源码。实际编译后完整目标SASS、148regs/5136B shared/0local/occupancy3需与原归档相符。每次保留8192个index/value/bits、stamp、CUDA完成状态、设备/资源和进程收据；先poison，再检查全部值有限、FP32往返一致、索引完整且不重复。数字与数学参考不同时保留差异，不将观察标为kernel数值合格。

继承有效单GPU Slurm、环境身份、套件/输出锁与GPU UUID锁、进程登记和清理。编译180秒、单进程120秒、清理预留20秒，预算不足checkpoint；续跑只允许有明确证据证明target尚未启动的profile。target一旦已启动或可能启动，即使缺少完整输出/receipt，也标为unknown/failed而非pending，不自动重跑；失败/中断同样消耗总计四次launch预算。需要重跑时另审修订，完成的有限差异记录不自动重跑。所有source/header/runner/auditor/reference及旧证据完整冻结。未来source-B必须明确CLI和字段，当前草案不伪造其哈希。

## 结果如何改变下一步

| 四点实测模式 | 能得出的结论 | 下一步 |
|---|---|---|
| 全元素62/64/64/64 | 在这些坐标观察到增长停止 | 保留原参考争议；考虑下面同dot、不同product的条件对照 |
| 全元素62/64/66/128 | 第32轮附近未停止 | 停止本轮；根据64与已有8192结果另审有限区间，不自动加点 |
| 不同thread/fragment/chain出现不同模式，或值下降 | 均匀停滞模型不足 | 先检查映射、寄存器依赖、循环和同步，不进入性能采样 |
| CUDA、poison、SASS/资源/快照身份失败 | 诊断无效 | 修实施或环境问题，不解释数值 |

即使出现62/64/64/64，**固定只执行32轮**与**数据相关的增量丢失**在这一组输入上仍可能完全相同。不能仅凭这四点反推硬件尾数位数。原SASS检查降低了静态循环错误的可能性，但不构成动态执行次数计数器。

## 条件输入对照：尚不实施或执行

若第一轮支持停止增长，另起独立合同并审A/source-B，最多再6次target。用一个新的诊断kernel从运行时tile载入数据，所有模式共用同一binary/symbol、WGMMA主体、batch、wait和排空；冻结性能源码保持不动。

- dense：A、B全1/16，D0，分别1/64轮，用来连接新tile载入路径与原kernel结果。
- sparse：A每行只有k=0、16为1/4，其余0；B全1/4，D0，分别1/64轮。每dot仍为`2×(1/4)²=1/8`，数学每轮仍加2，但贡献由32个小product变成2个大product。
- 原seed3 nonuniform：分别1/2轮，用既有独立逻辑参考检查新路径的映射和链初值。既有参考只支持1/2，不能默默扩为64轮。

先保存完整逻辑/打包tile，CPU独立验证pack→unpack与逐k求和。新kernel实际SASS必须证明issue数量、循环和wait保留；载入前缀变了，不能声称完整函数与旧kernel相同。dense1/64若不复现旧结果，或nonuniform短检查失败，停止因果解释，优先查载入/映射/初始化/同步。

若同一新kernel的dense64停止而sparse64继续增长，数据无关的固定32轮截断解释会被削弱，输入贡献粒度与累加行为成为更强解释；这仍不确定内部精度。如果两者均停止，dot级舍入和固定次数截断仍未区分，不能宣称诊断已经找到原因。若只有nonuniform失败，先定位映射和同步。是否需要带循环完成标记或改变wait频率的进一步对照，应按这些结果另审，不提前展开。

## 返回原32点任务

这份诊断只涉及一个E4M3 WGMMA坐标。原S08的32点、FP8/INT8、两个group和两个scope保持不变。其余31点应按现有已审短profile逐点获得证据，已有单点记录只在所有身份仍相符时复用；不能从E4M3结论外推E5M2、INT8或mma.sync降低路径。诊断不会授family B3。

INT8保留完整精确参考及溢出检查：uniform8192每元素为4194304，小于INT32_MAX；短非均匀输入另按有符号范围验证。FP8 mma降低点仍为logical计量且不导出原生FP8吞吐。若证据支持原FP8长数学参考不适合作为判据，应另审参考/正确性协议变更，把短完整参考与长服务观察分开；不能把64硬编码成新的正确答案。若改为周期性重置或FP32分段累加，必须承认kernel和工作量已变，另立合同、编译证据与计量边界；不得悄悄替换原32点。正式适配器与完整B通过前不做性能采样。
