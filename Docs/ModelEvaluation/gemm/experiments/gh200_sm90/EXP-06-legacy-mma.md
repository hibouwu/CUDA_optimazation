# EXP-06：mma.sync 计算基线约定

当前为 S06-A 草案，尚未实现探针、适配器或 GPU 验证。机器约定见 [legacy_mma.json](../../../../../microbench/gh200_resource_campaign/contracts/legacy_mma.json)。A 独立审查通过后才实施 B。

## 实验问题与矩阵

测量以下稠密 warp 级矩阵指令在旧线程规模、独立链数和循环条件下的服务速率，包含必要循环与排空成本。所有形式都是 `.row.col`，A/B/D 位于寄存器；不包含矩阵搬运性能或完整 GEMM。

| 输入与累加 | PTX 形状 M×N×K | 每 warp 每条指令 FLOP |
|---|---|---:|
| FP16×FP16→FP32 | 16×8×16 | 4096 |
| BF16×BF16→FP32 | 16×8×16 | 4096 |
| TF32×TF32→FP32 | 16×8×8 | 2048 |
| FP64×FP64→FP64 | 8×8×4 | 512 |

initial 保留 4 种形式 × chains=1/8 × 128/512/2048 次循环，共 24 项，均为 32 线程、单 CTA。sustained 保留 4 种形式 × threads=32/128/256 × 两范围，共 24 项，chains=8，动态长度。audit 保留 FP16、256 线程、chains=8 的两范围与三个长度，共 6 项，以及一个单 CTA 空窗口。总计 55 项，batch 均为 16。

每项嵌入 S01 的完整 `legacy.key` 和来源行号；矩阵形状、输入/累加类型、寄存器片段模型、参与 warp 数、chains、batch、计时和原长度各自保留。完整 32 线程 warp 必须齐步执行同一形状；线程数必须是 32 的整数倍。

## 工作量与最小例子

具名模型 `mma_dense_issue_v1`：

```text
FLOP = blocks × iterations × batch × chains × (threads/32) × 2×M×N×K
```

例如 `m16n8k16`、128 线程、8 条链、batch=16、8192 次循环、单 CTA：`1×8192×16×8×4×4096 = 17179869184 FLOP`。这里的 4 是独立执行矩阵指令的 warp 数，不是每条指令再乘 128 个线程。

空窗口使用 `compute_empty_window_v1`，iterations=0、工作量 0，仍执行相同计时/排空路径，报告 ns/window，不做空窗口扣除。读写 payload 均为 0，仅指计时循环无外部矩阵 payload；寄存器服务和 SMEM 结果排空不作为物理 HBM 字节计量。

## 片段与数值检查

每线程的片段大小与坐标公式冻结在 contract 的 `fragments`：FP16/BF16 的 A/B/D 分别为 8/4/4 个标量元素，TF32 为 4/2/4，FP64 为 1/1/2。FP16/BF16 的两个元素按低/高 16 位打包。B 实现必须先在 CPU 验证每个坐标映射完整覆盖逻辑矩阵、无越界和重复，再使用对应指令装配；不能用均匀输入检查替代布局检查。这些片段规则来自 [NVIDIA PTX ISA 8.8 的 mma 片段章节](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float)。

正式输入为 A=B=1/16、D0=0。每个独立累加器的期望值为：

```text
expected = iterations × batch × K / 256
```

chains 和 warp 数不进入单元素期望值。全部输出按类型精确比较，拒绝 NaN/Inf；每次启动重新初始化 D。固定及校准上限保证原输入累加仍可精确表示，不能跨窗口累加。

每个独立进程还在正式计时之外，以验证循环数 1、2 和以下非均匀矩阵检查同一特化算术路径：

```text
A[m,k] = (1+m+2*k+seed%3)/256
B[k,n] = (1+n+3*k+seed%5)/256
D0[group,chain] = (1+group+chain)/64
```

CPU 按逻辑行、列、K 做整数分子乘加，乘积共同分母为 65536，再加 D0；输入都是精确二进制数，验证长度下 FP32/FP64 结果可精确表示。CPU 参考独立于设备片段装配代码。所有 CTA、warp、链和 D 元素必须比较，输出预置 poison；检验范围是 `blocks×threads×chains×(4 或 FP64 的 2)`，不能只检查一个 checksum。非均匀输入只取得正确性资格。

## v2 窗口、范围与校准

thread0 保存 globaltimer/clock64 起点后，执行新的 CTA barrier，所有计入 FLOP 的矩阵运算位于其后。结束前各线程把全部结果参与 volatile double SMEM 排空，CTA barrier 之后 thread0 读取 clock64/globaltimer 终点。准备、主机比较与最终输出复制在窗口之外。起点 barrier 和强制计时记录属于 v2 新开销，旧窗口完整留作来源，不宣称绝对时数等同。

单 CTA 报 FLOP/本地 clock64 cycle；全 GPU 报 `FLOP/(max(stop_ns)-min(start_ns))`，单位 GFLOP/s。clock64 只在同一 CTA 相减，旧 audit SM span 求和不作为 v2 指标。全 GPU 启动 `sms×min(4,实际 occupancy)` 个 CTA，独立核对所有 SM 都被观察到。

24 个 sustained 点先用 8192 次循环做 B preflight，按 CUDA event 校准：

```text
resolved_iterations = clamp(int(8192 × 100 / max(pilot_event_ms,0.01)),8192,65536)
```

具名模型为 `legacy_event_ms_truncate_then_clamp_v1`。动态项的 `cases.iterations=8192` 只是 pilot，不可直接当正式值。`resolved_cases.json` 保存解析值、case、策略、pilot 路径与 SHA256，并经预检收据绑定设备、binary 和 contract。正式 `--resolved-cases` 只能导入 B review 直接绑定的解析文件与 pilot，随后所有 10 个独立进程、重试及 resume 使用同一冻结长度。旧实际值另存；缺钩子/缺解析/身份不符必须拒绝正式采样。

## B 阶段审查与复现边界

探针固定接口为 `BINARY CASE_ID ITERATIONS SEED`，成功输出 device、trial 两行 v2 JSONL；设备查询为 `BINARY device`。输入、coverage、输出元素检查数、资源用量和数值错误字段必须与 [INTERFACE](INTERFACE.md) 一起验证。只有空窗口允许 iterations=0。

SASS 逐目标函数检查 start barrier 对全部算术路径的支配、计时循环每个回边内的 HMMA/DMMA、结果依赖和退出后的排空。记录每种 PTX 的 SASS 降低形式，不假定一条 PTX 对应一条 SASS；拒绝 local/spill，静态循环证据不等于动态计数器证据。

CPU 接受计划覆盖 55 个旧映射的精确集合、四种形状工作量、warp 数/chain/batch 负例、全部片段坐标双射、错误布局与 poison 检查、未解析动态长度拒绝、空窗口工作量及 v2 范围单位。尚未取得 GPU 正确性或采样证据，图表和参数导出留待 C。
