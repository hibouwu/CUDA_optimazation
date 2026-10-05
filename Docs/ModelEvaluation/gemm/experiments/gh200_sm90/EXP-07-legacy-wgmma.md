# EXP-07：WGMMA 计算基线与 BF16 来源对照约定

当前为 S07-A 草案，尚未实现探针、适配器或 GPU 验证。机器约定见 [legacy_wgmma.json](../../../../../microbench/gh200_resource_campaign/contracts/legacy_wgmma.json)。A 独立审查通过后才实施 B。

## 实验问题与矩阵

测量 FP16/BF16 的 `wgmma.mma_async.sync.aligned.m64n64k16.f32` 指令，在指定 warpgroup 数、独立累加链、batch 和 wait 条件下，连同循环、提交与完成等待的服务速率。每个 warpgroup 为 128 个线程；128/256 线程分别对应一个/两个参与组。旧点均为 A/B 来自 SMEM 的 SS 形式，FP32 累加。

| 来源 | 条件 | 项数 |
|---|---|---:|
| initial | FP16/BF16；chains=1/2、batch=1/4/16；128 线程、wait0；单 CTA × 128/512/2048 次循环 | 36 |
| sustained | FP16/BF16；128/256 线程、chains=2、batch=16、wait0/3/7；两范围、动态长度 | 24 |
| audit 正式 | FP16；128 线程、chains=2、batch=16、wait0/3/7；两范围 × 8192/32768/65536 | 18 |
| audit 空窗口 | 上述三个 wait 配置；单 CTA、iterations=0 | 3 |
| 新 BF16 来源对照 | SS/RS × 单 CTA/全 GPU；128 线程、chains=2、batch=16、wait0、8192 次循环 | 4 |

81 个旧项逐项保留 `legacy.key` 和来源行号，加四个明确的新代表点，共 85 项。新点不冒充旧覆盖：SS 使用两个 SMEM 描述符，RS 的 A 使用四个 b32 寄存器，B 仍来自 SMEM 描述符；两来源具有不同 case ID 和 `operand_source_form`，相同范围的 SS/RS 用 `pair_id` 配对。只覆盖这组 BF16 代表条件，不扩展为来源形式或形状的穷举。

## 工作量与完成策略

具名模型 `wgmma_dense_issue_v1`：

```text
FLOP = blocks × iterations × batch × chains × (threads/128) × 2×64×64×16
```

例如 128 线程、2 条链、batch=16、8192 次循环、单 CTA，为 `1×8192×16×2×1×131072 = 34359738368 FLOP`。batch 是每轮每链发出的 PTX 次数；commit/wait 不增加名义 FLOP，耗时仍在窗口中。

每轮完成 batch×chains 条发出后提交一个 group，再执行原 wait0/3/7。initial 只有每轮 wait0；sustained/audit 和新来源代表点还有循环后的 wait0，之后才读取累加器做结果排空。该差别直接写入 `parameters.drain`。所有组完成前禁止改写 A/B SMEM 或读取未完成的 D；所有 warpgroup 线程一致执行相同指令和 wait。

空窗口工作量为 0，iterations=0 绕过算术循环，但保留各自约定的显式最终 wait、结果排空和计时 barrier，报告 ns/window。无空窗口扣除。读写 payload 均为 0 只表示计时循环无外部矩阵输入/输出 payload，不否认 WGMMA 从 SMEM 读取矩阵、排空写 SMEM 或计时元数据访问。

## 布局、输入和数值参考

SMEM A/B 使用 128 字节对齐、无 swizzle 的 K-major 布局，描述符 leading/stride offset 分别为 1024/128 字节。以 16-bit 元素计：

```text
offset_A(m,k) = (m%8)*8 + (m//8)*64 + k%8 + (k//8)*512
offset_B(k,n) = (n%8)*8 + (n//8)*64 + k%8 + (k//8)*512
```

RS 的 A 每线程含八个 BF16 元素，按相邻两元素低/高 16 位打包为四个 b32 寄存器。令 thread 为 warpgroup 内线程号、e 为标量片段下标：

```text
row = (thread//32)*16 + (thread%32)//4 + ((e//2)%2)*8
col = (thread%4)*2 + e%2 + (e//4)*8
```

A 使用 e=0..7，D 使用 e=0..31，分别覆盖 64×16 和 64×64；两个 warpgroup 分别从自己的组内 thread=0 重新计算。RS 不提供 A 转置参数，B 转置为 0；scaleD/scaleA/scaleB 均为 1。寄存器片段与来源合法性依据 [NVIDIA PTX ISA 8.8 WGMMA 片段说明](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-fragment-wgmma-64n16)，B 阶段仍须通过实际 CUDA 12.9 编译、目标函数 PTX/SASS 和非均匀数值验证。

正式输入保持 A=B=1/16、D0=0，FP16 位型为 `0x2c00`、BF16 为 `0x3d80`；RS 正式 A 四个寄存器均为 `0x3d803d80`。每个输出期望值为 `iterations×batch×16/256`，与 chains/warpgroup 数无关。全部输出精确比较并拒绝 NaN/Inf；所有启动重新初始化 D。

每个独立进程在计时之外用验证循环数 1、2 检查非均匀矩阵：`A[m,k]=(1+m+2*k+seed%3)/256`、`B[k,n]=(1+n+3*k+seed%5)/256`、`D0=(1+group+chain)/64`。CPU 按逻辑矩阵用整数分子和共同分母 65536 独立乘加，再加 D0，验证长度下结果精确可表示。SS 与 RS 使用相同逻辑矩阵，但分别装配各自来源；所有 CTA、组、链和 32 个/线程 D 片段都比较，输出预置 poison。不能仅检查 checksum 或把非均匀正确性扩成性能结论。

## v2 计时与校准

SMEM 初始化、proxy fence、wgmma fence 和准备同步位于计时前。thread0 保存 globaltimer/clock64 起点后执行**新增 CTA barrier**，所有 WGMMA 发出都在 barrier 后。循环及完成等待之后，全部 D 寄存器参与 volatile double SMEM 排空，结束 barrier 之后 thread0 读取 clock64/globaltimer 终点。输出复制与主机检查在窗口之外。

起点 barrier 和强制计时记录属于 v2 开销。原计时边界保留在 `legacy.key`，不宣称新旧绝对周期等同。单 CTA 使用本地 CTA clock64 差；全 GPU 使用所有 CTA 的 globaltimer 包络，单位分别为 FLOP/clock64 cycle/CTA 和 GFLOP/s。旧 SM span 求和只保存历史含义，不新导出为 v2 指标。

全 GPU 网格为 `sms×min(4,实际 kernel occupancy)`，逐项验证每个 SM 被观察到。RS/SS 资源占用可能不同，必须分别保存寄存器、SMEM、occupancy 和实际 blocks；来源配对只固定请求策略，不声称驻留完全相同。

24 个 sustained 动态项在 B preflight 重新取得 8192 次循环的 CUDA event 时长，以 `legacy_event_ms_truncate_then_clamp_v1` 得到 `clamp(int(8192×100/max(pilot_event_ms,0.01)),8192,65536)`。`cases.iterations=8192` 仅为 pilot。解析结果和 pilot 路径/哈希写入 `resolved_cases.json`，由 B review 直接绑定，通过预检收据核对 binary/contract/device；正式 run 显式 `--resolved-cases` 导入并冻结。10 个正式进程、所有重试与 resume 均使用同一解析值，禁止进程内部重校准；旧实际值另存，不用作新正式值。

## B 阶段审查与复现边界

CLI 固定为 `BINARY CASE_ID ITERATIONS SEED`，成功输出 device、trial 两行 v2 JSONL，`BINARY device` 仅输出 device；其余约束遵守 [INTERFACE](INTERFACE.md)。只有三个旧空窗口允许 iterations=0。

SASS 逐特化目标函数确认起点 barrier 支配计时内计算、每个循环回边保留 HGMMA、commit/wait immediate 正确、累加器读取前完成等待、无 local/spill。SS/RS 需从 PTX 操作数和降低结果交叉确认，不能只看 HGMMA token。分别记录静态降低数量，NCU 动态计数器缺失时保留资格缺口。

CPU 接受计划包括 81 个旧项无遗漏/重复、四个新配对项彼此可区分、work_model 的 warpgroup/shape/batch/chain 负例、SMEM/RS/D 坐标全覆盖无重复、wait0/3/7 与最终 drain 差别、错误来源或 transpose 拒绝、未解析动态点拒绝、计时范围单位和 poison/输出覆盖。尚未取得 GPU B 证据；实测图表、来源差别的效果和参数导出留待 C。
