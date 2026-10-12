# EXP-10：矩阵片段搬运与 warp 交换

状态：A 约定草案，尚未通过独立审查，未实施探针或运行 GPU。

## 实验问题与参数

测量 SMEM 矩阵片段与寄存器之间的搬运，以及一个 warp 内的寄存器交换。结果对应给定参与线程、片段数、访问布局和依赖条件下的循环服务；计时包含必要的地址计算、结果消费和同步，不宣称裸指令延迟。

## 一个最小例子

一次 `ldmatrix...m8n8.x2...b16` 由完整 32 线程 warp 共同读取两块 8×8 的 16 位矩阵，共 `2×8×8×2=256 B`。每轮执行 8 次，共 8192 轮，逻辑读量为 `16,777,216 B`。除以同 CTA 的周期差，得到 B/clock64 cycle/CTA。x2 表示两块矩阵，不是两个线程，也不是每个线程再读两块完整矩阵。

本例尚无实测值；C 阶段需补一条原始记录的字段、哈希、周期差与手算结果。

## 有限配置矩阵

| 家族 | 配置 | 点数 |
|---|---|---:|
| 矩阵搬运 | load、store、load→store；x1/x2/x4；正常与 `.trans`；单 CTA、32 线程 | 18 |
| warp 交换 | `shfl.sync.idx.b32`；32/128 线程；1/4 条独立流；每流自身保留依赖链 | 4 |

矩阵形状固定 m8n8、元素为原始 16 位数据。每个方向预留 4096 B SMEM，均分八个 512 B 槽；总动态 SMEM 固定 8192 B。每槽最多容纳四块矩阵，行起点满足 16 B 对齐。线程 0–7 提供第一块的行地址，x2/x4 按每八个线程增加一块；不用的地址线程也填入有效行地址。

所有 case 固定 8192 轮、每轮八个操作位置，不进行线程、stride 或 bank 布局的额外扫描。完整字段见 [matrix_exchange.json](../../../../../microbench/gh200_resource_campaign/contracts/matrix_exchange.json)。

## 计时与计量

输入初始化在窗口前。CTA 起点同步后，由线程 0 保存时间，再经 CTA barrier 进入循环，防止其他线程提前执行被计操作。结果消费及完成同步后保存终点，host 回读在窗口外。

- load：每位置加载对应槽，所有结果寄存器加入逐 lane 的整数校验和。记录逻辑读取字节。
- store：值由逻辑坐标与当前轮次生成，不依赖同窗口读入。最终 CTA barrier 后才停止，窗口外检查整个有效输出。记录写入字节。
- load→store：源槽与目标槽独立，加载和存储使用相同转置限定符，使完整往返恢复源矩阵；每对指令后 warp barrier，消费所有加载寄存器。读写量分别保存，总量为两者之和。不会把往返时间除二当作单向延迟。
- shuffle：每一步读取 `(lane+1)%32` 的上一状态，membermask 为全部 32 个 lane，clamp 为 31。每条流有自身依赖，多条流之间独立。记录 warp 指令数 `轮数×8×流数×warp数`，不计为 FLOP 或 SMEM 字节。

多个 warp 仍属于同一个 CTA，主指标统一使用该 CTA 的局部周期；不会把它们的 `clock64` 相减，也不外推全 GPU 峰值。方向和指令计数必须由实际 SASS 循环确认，不能仅凭源代码的循环次数。

## 正确性与接受条件

矩阵输入按行、列、矩阵编号、槽和 seed 构造不同的 16 位值。CPU 独立枚举正常与转置片段，检查每个 lane 的全部寄存器。store 和往返另检查完整目标数组，保留非有效槽的 poison 检查。正式长循环之外，每进程运行 1/2 轮矩阵验证。

shuffle 各 lane、warp、流的初值不同，CPU 按同时更新的 lane 状态计算参考。额外运行 1、3、33 步，避免正式步数恰为 32 的倍数时，只检查回到初态而漏掉错误路由。所有流的输出都必须核对。

B 阶段需要 CPU 片段往返、错误坐标负例、目标编译、循环内实际矩阵/交换指令、完整 GPU 数值检查及独立审查。load 循环若被编译器外提，必须修复并重编，不能接受只剩校验和运算的计时。

正式 C 采用公共十进程、有界预热和最多三批协议。图表分别展示矩阵请求服务和 shuffle 指令服务，列出误差条与原始记录来源；未执行前不填性能结论。

## 来源与复现

合法形状、warp 参与和对齐要求依据 [PTX ISA 8.8 ldmatrix](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix)、[stmatrix](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#warp-level-matrix-instructions-stmatrix) 和 [shfl.sync](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-sync)。本轮仅选择 SM90a 的 m8n8.b16 形式。

实现完成后通过 `run_suite.py` 指定本合同启动新的 preflight 归档；当前源文件和适配器尚未实现，因此还没有可运行的复现命令或有效结果。
