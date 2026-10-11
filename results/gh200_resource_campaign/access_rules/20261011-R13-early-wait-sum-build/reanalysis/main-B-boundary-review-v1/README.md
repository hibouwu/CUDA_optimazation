本审查只核对固定 `prepare3` 的源码位置、循环覆盖、保存范围、前序开关及 cycle 解码接口。归档及候选的完整 source manifest 均逐项通过 SHA 核对，结果见 [JSON](B-q-early-boundary-review.json)。主线保留 prepare2 的双副本定义编译失败；prepare3 使用 canonical `probes/v06_trace.hpp`，overlay 只 forward。本次不重复主线的 5 项 CPU 测试。

`K_PIPE_MMAS=1`，K4096 对应 64 个 K64 组。prologue 消耗 local0，循环遍历 local1～63。thread256 属于 Consumer1，其首个自有输出（WG local j=0）是 CTA j=1，因此 `C1` 位于 global group65，`C16` 位于 group80；两者之间包含 local2～16（global66～80）的 15 次 full try/wait。stages4/6 的 slot/phase 逐项见 [覆盖表](group-coverage.csv)，count=15、mask=0x1fffc。

候选 H 放在当组 `consumer_try_wait` 前，B 放在 `consumer_wait` 返回后；token 在当前循环现场求值，没有提前为下一组生成 token。try 已返回 WaitDone 时，wait 跳过阻塞，仍须记录 H/B。新增 record 在 B 后，其成本属于 `D−Q`。本审查还删除候选的三个新增块并逐字比较，恢复后的 collective 与归档完全一致；kernel/pipeline 本身逐字一致。

`R13SparseConsumer r13_s` 在 WG scheduler while 外构造，整个 work loop 结束后才 flush。应保留这个范围，只有目标输出更新累计值，后续 WG 输出不重置或覆盖；循环内不写 GMEM。

同线程、同输出、同调用的时间顺序下，`D=C16−C1`，`Q=Σ(B_i−H_i)`，且 `D−Q=Σ[(H_i−C_(i−1))+(C_i−B_i)]`，i=2～16。先逐调用计算这些整数 cycle 差，再统计。Q 是取时后 full try/wait 路径的累计观测，包括软件与取时成本，不能直接当目的 ready、裸 TMA 服务或物理源价格。

最少新增 sum/count/mask/显式开关与唯一 protocol 身份；候选同时保留 H2、B16 和 invalid bits，便于拒绝坏记录。on 必须 count15、mask0x1fffc、invalid0、同 SM，且 `C1≤H2≤B16≤C16`、`0≤Q≤D`。off 字段为零表示未观测，Q 和 residual 应为 None。16 次前序调用始终 observer0，再设置目标 flag0/1；每个 stage 使用自身同一 binary/缓冲区，两个 stage 的执行历史不能据此视为相同。

当前 decoder 已核对上述字段、唯一 marker、SM/work/flag 身份和 on/off 分支；其 `Q≤B16−H2` 比 `Q≤D` 更强，且两个组件的 ns 均为 None。CPU helper 能证明记录算术与拒绝条件，不证明编译后的 GPU 计时顺序。完整采集身份和数值验收仍由主线负责。

同一 binary 的 off/on 只量化运行期开关的增量影响；新增长寿命累计寄存器可能使 off 也不同于旧 progress binary。record 虽在 B 后、归入残余，但会推迟后续等待到达，进而改变后续 Q。因此这是带观测路径的精确分段，不能直接恢复未加观测时的等待量。

源码边界审查通过；尚未核对编译后 SASS。需检查快/慢 token 路径上的 H→实际 barrier try/wait→B 顺序，以及寄存器、spill 和新增软件路径。B 本轮没有 GPU、SSH/Slurm、公共源码修改；CPU 静态通过不授予硬件正确性、观测或预测资格。
