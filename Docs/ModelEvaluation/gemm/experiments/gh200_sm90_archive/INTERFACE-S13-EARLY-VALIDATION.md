# S13 固定短检查与 issue 顺序

待独立 early-validation-A 审查。本补充只冻结原18点的短profile、完成条件与issue顺序，不授予GPU执行或整家族B3资格。

每个case执行同一target特化1、2轮，共两次target launch，零显式辅助kernel；threads=256、实际occupancy网格、L2决定的对齐分配、cache form和r:w均保持原合同。两档工作集复用同一操作特化，不能为了短检查缩小实际数组。输入源为`uint32(17*word_index+seed)`，独立输出为`uint32(29*word_index+seed)`，copy逐word等于源。

每轮每个group先发出r次读，把四个uint32 word全部累加；随后发出w次独立写，写值仅由目的地址和seed得到。copy固定r=w=1，读出的uint4直接写到独立目的数组的同一index。组索引绕回保持`index%T=thread_id`，与原公式等价。连续写同一地址仍采用volatile PTX，实际重复请求及其比例必须由目标SASS确认。

所有线程将自己的sum写入`volatile __shared__ uint32[256]`，作为读结果依赖排空；静态SMEM为1024B，计时bookkeeping不计payload。含写模式随后执行device threadfence，全CTA通过结束barrier，thread0再取终点。读模式也通过同一结束barrier。初始化/poison/host核验在计时外。该排空和fence必须在B2真实SASS验证，不凭源码自行认定降低正确。

每次launch在host逐项计算目的word和thread checksum的expected，将`expected XOR UINT32_MAX`通过cudaMemcpy填入对应输出；纯write的零sum以UINT32_MAX填充。每项正确结果都与其poison不同，因此必须被设备覆写。stamp全部字节为0xff。poison和校验共享参考的构造不替代独立reference测试；这些初始化全部在计时外且没有辅助kernel。host核对全部已启动CTA的stamp四个u64字段被覆盖，SMID不是UINT32_MAX；不以`smid<SMcount`假设物理SMID连续。完成要求包括CUDA launch/event同步/readback成功。短窗口允许零时间差，不要求短launch实际占满物理SM；正式仍遵守exact_sms。

有读路径核对全部`T=blocks*256`个thread的完整模2^32 checksum；有写路径核对目的数组所有`A/4`个word。纯write的每线程sum=0只检查完成bookkeeping，不把它算作读或数据覆盖。每launch计入`checked_elements/expected_elements`的数量固定为`(r>0?T:0)+(w>0?A/4:0)`；纯write仍必须完整核对全部目的word。与原独立闭式host参考比较，exact/null tolerance，不抽样。

成功stdout恰好原device行加固定ABI1 `type=validation`行。profile为`short_global_duplex_1_2`，input_profile为`address_seed_nonuniform`，两个launch索引0、1。检查字段、参考SHA、全部CTA IDs、资源七字段及三个false性能标记按公共接口。resource extensions仅允许requested_array_bytes、array_bytes、aggregate_array_bytes、read_requests_per_group、write_requests_per_group，adapter从设备L2、原case、occupancy和对齐公式独立验证。small聚合>=L2或可用GPU内存不足时拒绝，不改变条件凑通过。

仅保存错误计数，`output_evidence_kind=error_count_only`、output_artifacts为空；不能据此离线更换reference。失败优先保存partial validation并非零退出，缺少形成记录条件时保留原stdout/stderr/receipt。定位详情最多8条。ABI三个函数保持纯Python；没有独立批准的旧证据映射时返回insufficient，坏哈希必须抛ValueError。

profile位于`microbench/gh200_resource_campaign/contracts/global_duplex_validation_profiles_v1.json`。adapter manifest草案同目录`global_duplex_validation_adapter_v1.draft.json`明确不可执行，不伪造未来模块/审查哈希；实现后冻结实际传递依赖并由独立source-B复审。原case-only runner只能诊断单点，完整18点覆盖和独立B3仍是任何pilot/预热/正式样本之前的门槛。
