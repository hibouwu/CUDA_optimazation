# EXP-05：FMA 计算基线约定

当前为 S05-A 草案，尚未实现探针、适配器或 GPU 验证。机器约定见 [legacy_fma.json](../../../../../microbench/gh200_resource_campaign/contracts/legacy_fma.json)，旧点身份来源见 [EXP-03](EXP-03-compute-baseline.md)。A 独立审查通过后才实施 B。

## 实验问题与参数

测量指定类型、线程数、独立累加链数和循环长度下，FMA 循环及必要计时/排空工作的服务速率。保留 FP32、FP64、FP16、FP16x2、BF16、BF16x2 六种形式；打包形式每条线程指令包含两个独立标量 FMA。结果不能解释为裸指令延迟或物理峰值。

| 来源 | 配置与范围 | 条件数 |
|---|---|---:|
| initial | 6 类型 × chains=1/8；32 线程；单 CTA；128/512/2048 次循环 | 36 |
| sustained | 6 类型 × threads=32/128/256；chains=8；单 CTA/全 GPU；重新校准长度 | 36 |
| audit 正式 | FP32/FP64/FP16x2；256 线程、chains=8；两范围 × 8192/32768/65536 | 18 |
| audit 空窗口 | 上述三个配置；单 CTA、iterations=0 | 3 |

共 93 项，全部 batch=16。每项原封保留 `legacy.key`、旧 raw 行号及实际长度；新 `parameters` 明确类型、打包 lane 数、chains、batch、输入、结果排空和 v2 计时模型。不按基本组合删去不同来源和长度。

## 工作量与手算

具名模型 `fma_dense_issue_v1`：

```text
FLOP = blocks × iterations × batch × chains × threads × packed_lanes × 2
```

例如 FP16x2、128 线程、8 条链、batch=16、8192 次循环、单 CTA，工作量为 `1×8192×16×8×128×2×2 = 536870912 FLOP`。chains 表示独立累加器数量，不能再乘一次打包 lane。FMA 指令的 `shape=null`，标量并行度由 `packed_lanes=1/2` 明确表示。

`compute_empty_window_v1` 的工作量为 0，保留相同函数的计时、结果求和和 barrier；指标为 `elapsed_ns/1`，单位 ns/window，不从计算窗口中扣除空窗口。

`read_payload_bytes=write_payload_bytes=0` 仅表示计时循环没有外部矩阵输入/输出 payload；寄存器运算、结果排空的 SMEM 访问与计时元数据仍实际发生，不能据此声称物理流量为零。结果求和、转换和计时指令不计入名义 FMA FLOP。

## 输入与数值检查

正式输入保持旧条件：每条链从 0 开始，反复执行 `D=D×0.5+0.25`。参考值按输入/累加类型逐次舍入到最近偶数；已有正式长度收敛到精确的 0.5。空窗口参考值为 0。检查每个 CTA、线程、链和打包 lane 的全部输出，拒绝 NaN、Inf 和任意错误；所有预热和正式启动重新初始化 D，不能跨启动继续累加。

每个独立进程先在计时外执行非均匀检查，验证循环数取 1、2，仍用同一特化算术路径：

```text
D0 = ((thread + chain + packed_lane + seed) % 7) / 16
multiplier = 1/2
addend = (1 + (thread + 3*chain + packed_lane + seed) % 7) / 32
```

CPU 使用整数表示的二进制有理数执行融合运算并按类型舍入，不能以 GPU 的输出或探针内部求和函数充当参考。打包的两个 lane 分别检查，输出先写入 poison，检查覆盖数为 `blocks×threads×chains×packed_lanes`。非均匀输入只验证正确性，正式性能仍属于原均匀输入条件。

## v2 计时与动态长度

操作数准备之后先做 CTA 同步；thread0 保存 globaltimer 起点和 clock64 起点，**随后执行 CTA barrier**，所有计入 FLOP 的运算都必须在该 barrier 后。循环结束时，所有活跃累加器参与 volatile double SMEM 排空，CTA barrier 完成后 thread0 读取 clock64 终点和 globaltimer 终点。结果复制及主机检查在窗口之外。

新增起点 barrier、强制 globaltimer 记录和相关元数据属于 v2 窗口开销。旧窗口保留在 `legacy.key` 中，不声称新旧绝对周期可直接等同。单 CTA 指标为 FLOP/clock64 cycle/CTA；全 GPU 为 `FLOP/(max(stop_ns)-min(start_ns))`，数值单位 GFLOP/s。只在同一 CTA 相减 clock64；旧 audit 的 SM span 求和只保留历史语义。

全 GPU 网格为 `sms×min(4,实际该 kernel occupancy)`，要求观察到所有 SM；不足时报覆盖失败。资源用量由实际二进制查询，禁止以旧 occupancy 代替。

sustained 的 36 项在 B preflight 使用 8192 次循环取得 CUDA event 时长，按具名 `legacy_event_ms_truncate_then_clamp_v1` 解析：

```text
resolved_iterations = clamp(int(8192 × 100 / max(pilot_event_ms,0.01)),8192,1048576)
```

`cases.iterations=8192` 在这些项上仅是 pilot 参数。preflight 生成 `resolved_cases.json`，绑定 case、解析长度、策略、pilot raw 相对路径及 SHA256，并通过预检收据绑定 binary、contract 和设备。正式 run 必须显式 `--resolved-cases` 导入 family B 审查直接绑定的解析文件与 pilot。10 个进程、所有重试和 resume 都使用封存长度；缺少解析、适配器钩子或身份不一致时拒绝正式运行。旧实际长度另存，不能代替新校准。

## B 阶段审查要求

固定 CLI 为 `BINARY CASE_ID ITERATIONS SEED`，成功输出恰好 device、trial 两行 v2 JSONL；`BINARY device` 仅输出 device。遵守 [INTERFACE](INTERFACE.md) 的必填字段、独立进程、锁、预热与有界复测约定。iterations=0 只对空窗口合法。

SASS 逐特化函数检查计时范围、起点 barrier 对计算的支配关系和每个循环回边内的 FFMA/DFMA/HFMA2；不能仅在整个二进制搜索 token。PTX 工作量与 SASS 降低后的指令数量分别记录，不预设一条 PTX 恰好对应一条 SASS。拒绝 local/spill；静态证据不冒充动态指令计数。

CPU 接受计划包括：93 项与 S01 对应集合完全一致；错误 lane/chain/batch/线程数的工作量负例；丢失或重复旧映射；错误固定长度；未封存动态长度；错误单位和全 GPU clock64 指标；打包 lane 篡改；不完整输出检查。GPU B 证据尚未取得，实测图表与参数导出留待 C。

指令语义依据 [NVIDIA PTX ISA 8.8](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html) 的浮点、半精度与 BF16 FMA 章节；本文件的矩阵和采样条件由当前 contract 冻结。
