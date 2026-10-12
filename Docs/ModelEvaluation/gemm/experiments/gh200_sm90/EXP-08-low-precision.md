# EXP-08：稠密 FP8 与 INT8 代表计算服务

状态：S08-A 草案待独立审查；尚无探针、CUDA编译、GPU数值或性能结论。

## 实验问题与参数

分别测FP8 E4M3/E5M2和INT8 S8/U8在两条原生矩阵指令路径上的有限配置服务。浮点累计工作量用FLOP，整数用OP；不能把相同输入位宽当作相同执行能力，也不覆盖混合编码、稀疏或饱和形式。

## 冻结候选矩阵与算例

- `mma.sync`：M16N8K32，输入在寄存器，1/2个warp，即32/64线程。
- WGMMA：M64N64K32，SS形式，A/B在SMEM，1/2个warpgroup，即128/256线程。
- 每路径4种同类型输入，分别FP32或S32累加；每配置2条独立链、每批每链16条指令、8192轮。
- 单CTA和全GPU两个范围；全GPU按实际occupancy取 `SM数×min(4,CTA/SM上限)`，并验证SM覆盖。
- 合计32点。WGMMA每批commit/wait0，结束再wait0；不在本组额外扫描等待深度。

一CTA、一个warp的FP8 MMA请求工作为 `2×16×8×32×2×16×8192=2,147,483,648 FLOP`。两个warp时是两倍；同形状INT8使用OP单位。全GPU用实际CTA数乘入分子，除以globaltimer网格包络得GFLOP/s或GOP/s；单CTA用本地clock64差。

## 正确性与范围上界

性能输入为FP8的1/16或INT8的1，D0=0，每次launch重置。每输出累计 `8192×16×32=4,194,304` 个乘积；FP8统一以1/256为分母，整数分子小于2^24，INT8结果也远小于2^31。链和参与组各自保有独立累加器，不把它们的工作量叠到单个输出上。

每个独立进程另执行1/2轮非均匀数值检查：FP8采用可精确表示的0、±1/8、±1/4、±3/8；S8含正负小整数，U8含0/1/2。A/B以行、列、K坐标和seed构造，host以整数点积独立计算参考；所有输出参与检查，初值poison，拒绝NaN/Inf。具体公式和片段坐标见 [low_precision.json](../../../../../microbench/gh200_resource_campaign/contracts/low_precision.json)。这不是完整FP8/INT8数值域的正确性证明。

## 计时与审查

保存起点后经CTA barrier再进入被计循环；计算完成且结果排空后结束。初始化、数值对照与host回读不计入设备窗口。按照公共有界预热、10外部进程、最多3批协议收集。

B阶段须逐类型检查合法PTX、实际SASS循环、类型/形状/带符号编码、全部输出及资源占用；没有通过A独审前不实施采样路径。无法证明的物理峰值、缓存流量与完整精度覆盖保持未证明。

## 来源与后续产物

指令语义采用 [PTX ISA 8.8 MMA](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma) 和 [WGMMA](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions)，按SM90a目标约束筛选。A通过后再建立具体源码、参考、适配器、CPU负例及真实B预检；本文件不填入其他精度的数值充当结果。
