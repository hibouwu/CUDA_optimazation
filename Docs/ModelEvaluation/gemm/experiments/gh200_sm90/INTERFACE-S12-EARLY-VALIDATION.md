# S12 短正确性固定接口

状态：待独立 A 审查；本页与 profile、adapter manifest 草案不授予执行资格。

## 有限 profile

`short_copy_1_2` 对同一冻结 case 依次执行 1、2 iterations，共两次 target launch、零次显式辅助 kernel。输入为 `uint32(17*word_index+seed)`，地址相关且非均匀。保留 128 threads、ca/cg、请求宽度、stages、参与者与原 occupancy 启动策略。每个 iteration 含 8 copy steps，最短 8 steps 足以覆盖最大 4 stages 的 prefill、消费、复用和 tail。

令 `B=blocks`、`W=request_bytes_per_thread/4`、`S=stages`、`I=iterations`。每次完整核对 `B*128*I*8*W` 个 neighbor-consumption trace word、`B*128` 个模 2^32 sum、`B*128*S*W` 个最终 slot word，共 `B*128*(I*8*W+1+S*W)` 个 uint32 输出。最终 slot 对应 `slot+floor((I*8-1-slot)/S)*S`，consumer 对应 `(thread+1)%128`。sum 使用独立 floor_sum 闭式参考，trace 和 slot 使用解析地址参考。

每次启动前重置全部输出和 stamp。host 短分支将 stamp 全字节设为 `0xff`，逐 CTA 检查四个 u64 时间字段已覆盖、`smid` 在设备范围内；计数不把 stamp 当作数值输出元素。CUDA launch、event synchronize 和 readback 必须成功。零计时差不算失败，也不要求短 kernel 实际占遍物理 SM。保留原设备 kernel 的全部源码字节，不添加 SMEM 或指令。

## ABI 与记录

成功 stdout 恰好为原 device 行和固定 `type="validation"` 行；`schema_version=2`、`validation_schema_version=1`，`performance_eligible=false`、`warmup_executed=false`、`pilot_executed=false`。launch/check/resource 的字段集合完全沿用 `INTERFACE-EARLY-VALIDATION.md`，不得保留自定义 `s12_short_validation`。两个 launch 的 `input_profile` 均为 `address_seed_nonuniform`；exact 比较、null tolerance、完整 CTA IDs、冻结 reference model/hash、expected/checked 元素数逐项独立核对。resource extensions 为空。输出保存方式为 `error_count_only`，`output_artifacts=[]`；检查了全部值不等于保存了值，此证据不能支持离线更换参考。

异常优先生成同类型失败记录并非零退出；已完成与失败 launch 的 check 可保留，失败 launch 的 completed=false 或 errors>0。未执行 launch 不补造成功。分配、解析或设备查询等错误导致记录条件不足时保留 stdout/stderr 和失败收据。失败详情最多 8 条，至少定位 length/CTA/thread/stage 或 step/word/actual/expected。

adapter 只导出 ABI1 三函数，纯 Python，不读写文件、不开子进程、不访问 GPU。`validation_argv` 返回固定五参数命令。`validate_validation` 拒绝未知 profile、输入/长度/输出/参与者/参考/资源/完成条件不符及全部失败/残缺记录。`validate_prior_evidence` 校验已提供绑定哈希；目前没有独立批准的 S12 复用映射，合法但覆盖不足的旧证据返回 insufficient，损坏证据抛 ValueError，不自动批准复用。

独立家族审计模块提供 `validate_device` 与 `audit_sass`，供现有 case-only runner 接口使用，不改公共 core 或 S08。SASS 核对全部 12 特化的符号、请求宽度/cache、commit/wait、循环、CTA barrier、完整消费和最终排空；直接重放既有实际 CUDA 12.9 SASS及负例。设备核对 GH200/SM90 和资源容量。新 host 编译后须独立比较 device SASS及资源，不能用源码字节未变直接声称二进制未变。

## 工件与实现门禁

profile：`microbench/gh200_resource_campaign/contracts/async_copy_validation_profiles_v1.json`。

manifest 草案：`microbench/gh200_resource_campaign/contracts/async_copy_validation_adapter_v1.draft.json`。`planned_files` 只是预期依赖清单，`files_sha256` 为空，明确不可执行。独立 A 通过后才实施 host/adapter/auditor/test，随后生成正式 `async_copy_validation_adapter_v1.json`，冻结所有实际依赖哈希并接受独立 source-B 审查；不得自签。公共文件只引用，不修改。

负例覆盖漏 launch、换长度/输入、漏 CTA/trace/slot/sum、假 completed、参考/hash 不符、错误计数、隐藏性能工作、非法 occupancy/SMEM/符号、损坏旧证据，以及删除 copy/wait/consumer/release barrier 的真实 SASS 变异。CPU 参考测试继续验证跨 warp neighbor 与尾部 slot。

当前 runner 只有 case diagnostic；本次接口实施不声称完整 family 调度或 B3 通过。24 cases 全部短正确性证据与独立 B3 审查仍是进入 B4 的前提。本任务不提交 GPU。
