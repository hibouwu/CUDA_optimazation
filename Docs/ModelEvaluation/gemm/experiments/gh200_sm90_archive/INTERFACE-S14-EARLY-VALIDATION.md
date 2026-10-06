# S14 固定短 profile 与完整工件接口

待独立 early-validation-A。原`contracts/tma_bulk.json`、EXP-14及feasibility字节保持不变；本页只落实既有配对、完整输出和固定ABI，不授予GPU执行资格。

每个正式case依次要求三个独立进程/收据：`bulk_short_1_seed0_v1`、`bulk_short_2_seed3_v1`、`bulk_short_33_seed4294967295_v1`，分别为1/seed0、2/seed3、33/UINT32_MAX。每个profile一次target、零辅助kernel。source-release仅适用于12个SMEM→GMEM坐标，独立`bulk_source_release_one_request_v1`、1轮/seed3、一次target，不可替换前三份。24×3+12=84份独立诊断，不是长度×seed笛卡尔积。

保留128threads、Q=1/4/8/16/32/64KiB、每CTA32global槽与原scope。实际target特化查询static+dynamic资源，dynamic为Q+32，设置所需cudaFuncAttributeMaxDynamicSharedMemorySize；按实际occupancy决定all_gpu网格，cudaMemGetInfo检查全部global/trace/control分配。不得减少32槽、缩小Q或延长30秒进程上限/20秒清理余量使其通过。

短kernel与正式kernel复用同一bulk复制原语、Q特化和资源布局。短模式每次成功完成并CTA会合后，全部线程把Q/4个word保存至trace，再经CTA会合才可下一请求复用；正式模式的计时循环没有逐word消费。GMEM→SMEM按真实opaque token等待，每phase至多1秒；timeout先记录失败状态，再fatal trap，绝不正常return、inval或复用仍有请求访问的shared。SMEM→GMEM必须同issuer fullwait再CTA。source-release使用另外的单请求kernel：readwait→release clock→CTA→逐word volatile补码覆盖→CTA→fullwait→completion clock→CTA，不能把readwait当目的完成。

S2G ring的所有slot统一初始化为`~shared_input_word`，不按visited/unvisited的最终`destination_word`再取反；未访问槽因此保留原source值的补码。必须被写出的trace、source_after等capture才用最终expected按位取反。33轮覆盖回绕。source-release另保存全部source_after word证明补码覆写，目的slot0仍须保留原始source值。global数组有效payload前后各16B guard，payload地址仍16B对齐；guard8个word固定`0xd15ea5e0+i`，逐项核对。mbarrier位于独立8B对齐控制区，不与payload重叠。

成功stdout只有device行与固定validation行；checks/launch/resource字段遵循ABI1，不加自由顶层字段。每profile唯一launch_index=0，seed固定在顶层。resource extensions仅为Q、32槽、global分配、completion_kind和validation_role。g2s/s2g使用普通target符号；source-release使用不同符号并记录自己的实际资源，不推断其计时端点与正式窗口相同。

输出工件均用显式little-endian uint32，固定相对文件名、shape与SHA256。现有core支持uint32而不支持uint64工件，因此生命周期计数和clock64以`[low32,high32]`成对无损编码，避免改公共schema或用double损失位数。

| 工件 | 普通短profile | source-release |
|---|---|---|
| `bulk_trace.u32le` | `[B,I,Q/4]`，每请求完整word | 不产生 |
| `bulk_ring.u32le` | S2G保存`[B,32,Q/4]`全部槽，含未访poison | 同左，仅slot0已写 |
| `bulk_guards.u32le` | `[2,4]`首尾guard | 同左 |
| `bulk_completion.u32le` | `[B,5,2]`，completed_requests、mbarrier_wait_attempts、timeout_flag、read_release_waits、full_bulk_waits | 同左 |
| `bulk_source_after.u32le` | 不产生 | `[B,Q/4]`全部source补码 |
| `bulk_release_clocks.u32le` | 不产生 | `[B,3,2]`同CTA的start/release/full cycle |

数据比较计数为普通短`B×I×Q/4 + (S2G ? B×32×Q/4 : 0) + 8`，source-release为`B×33×Q/4+8`。完成计数与时间戳按生命周期条件单独全量检查，不混入payload/guard数值元素计数。g2s要求completed=I、wait_attempts>=I、timeout=0、read_release/full_bulk_waits=0；普通s2g要求completed=I、mbarrier_wait_attempts=0、timeout=0、read_release=0、full_bulk_waits=I；source-release要求completed=1、read_release=1、full_bulk_waits=1、其余0。source-release每CTA三时钟均被覆写且单调不减，允许相等，不跨CTA相减或导出带宽。每CTA普通stamp全部字段必须被覆写，SMID不等于UINT32_MAX，不假定物理SMID连续。

`output_evidence_kind=full_values`。host逐word比较后写全部工件；ABI1纯函数只能收到metadata，现有core核对工件大小/hash，不接收word数组。因此S14另提供纯`audit_artifact_values(case,profile,seed,arrays)`：独立离线审查者读取完整工件后传入，重新检查每word/guard/计数/时间戳。它不替代或扩展三个固定导出，不创建另一个runner。失败保留partial stdout/stderr/receipt，缺项不能构造成功；定位最多8条。

现有validate_suite只提供case diagnostic，每份profile可以直接复用；它尚不自动收齐本家族84份。S14提供纯有限集合检查，要求原24case各三指定profile/seed及12份source-release，身份一致且无遗漏/替换/重复，返回待独立B3审查，不授pilot资格。此后仍须现有流程形成独立全家族B3门禁，才可pilot/预热。正式calibration/收据绑定未集成时不启用正式host入口，不用环境变量或未受审文件绕开门禁。

profile和manifest草案分别为`contracts/tma_bulk_validation_profiles_v1.json`、`contracts/tma_bulk_validation_adapter_v1.draft.json`（均在microbench/gh200_resource_campaign）。manifest草案不可执行，正式manifest将绑定实际代码、参考与编译依赖，并经独立source-B复审。当前参考保持独审通过版本，不擅自修改。完整uint32工件序列化与流式SHA256使用新增host-only `common/word_artifacts.hpp`，namespace `word_artifacts`；它尚非已审库，必须经独立hashlib标准向量/边界/实际文件核验及实际target编译后再审。
