# S15：二维 TMA 运输约定（待独立 A）

本草案固定 rank=2、16-bit 无算术运输、单请求单缓冲，每轮完成后才复用。正式探针、描述符实际编码及 GPU 正确性尚未实现或验证。合同为 [tma_tensor_2d.json](../../../../../microbench/gh200_resource_campaign/contracts/tma_tensor_2d.json)。前置坐标推导见 [描述符可行性检查](EXP-15-descriptor-feasibility.md)。

## 有限配置

GMEM→SMEM 和 SMEM→GMEM 分开，分别运行单 CTA 与全 GPU，均为128线程、thread0发起。每个方向和范围有17点：连续 none 六点、padding none 六点、连续 SW128 五点，共68点。每个 padding/SW128 点只与同方向、同 payload、同范围的连续 none 比较；不加 padding×SW128 组合。

| Q | boxDim 元素 `(W,H)` | 连续行距 P | padding 行距 P | 布局 |
|---|---|---:|---:|---|
| 1 KiB | (64,8) | 128 B | 144 B | none；连续 SW128 |
| 4 KiB | (64,32) | 128 B | 144 B | none；连续 SW128 |
| 8 KiB | (64,64) | 128 B | 144 B | none；连续 SW128 |
| 16 KiB | (64,128) | 128 B | 144 B | none；连续 SW128 |
| 32 KiB | (64,256) | 128 B | 144 B | none；连续 SW128 |
| 64 KiB | (128,256) | 256 B | 272 B | none |

CUDA12.9.1 的 tiled descriptor 限制包括 box 各维不超过256、行 stride 按16B对齐、SW128最内层不超过128B。因此本 rank2 单请求 SW128 上限是 `128×256=32KiB`。64KiB保留 none；不拆成两次请求、不改rank。[官方描述符约束](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)

## 描述符与地址

使用 `cuTensorMapEncodeTiled`，类型 `CU_TENSOR_MAP_DATA_TYPE_UINT16`，`elementStrides=(1,1)`，interleave/L2 promotion/OOB fill 全为 none。这里的 uint16 是原始运输位模式，没有 FP16、BF16 或 Tensor Core 运算。

每 CTA 独占32槽环，`globalDim=(W,H×32×blocks)`，`globalStrides=(P)`。第 b 个 CTA 第 i 轮的坐标为 `(0,H×(32b+i%32))`，请求内元素 `(x,y)` 的地址为：

`base + (32b+i%32)×H×P + y×P + 2x`。

全局基址按128B对齐，分配实际字节数为 `blocks×32×H×P`，另留首尾128B guard。有效 payload 工作集另记 `blocks×32×Q`，padding不是有效运输字节。全局尺寸、分配大小、地址偏移和总量先用带溢出检查的uint64计算；确认坐标不超过有符号32位、没有OOB及跨CTA重叠后再窄化。

每 case 分配完成后编码一次；保留全部输入字段、返回码及128B描述符字节/hash，只有 `CUDA_SUCCESS` 才能继续。主机对象按64B对齐，以 `const __grid_constant__ CUtensorMap` 按值传入kernel，取其参数空间的generic地址；不生成线程私有local副本。描述符及其底层分配保持不变，直到所有使用者完成。本草案不做设备侧descriptor更新，因此不把设备generic写描述符与tensormap代理发布路径混入测量；以后改变该生命周期须重审。

## shared 布局和同步

所有 none/SW128 配置统一分配 `Q+1056` 字节动态shared。tile起点在其中向上对齐1024B，prefix最多1023B，tile之后另留32B控制区，mbarrier按8B对齐。采用相同分配规则，避免给 none/SW128 比较另加不同缓冲大小。实际特殊化仍须查询动态shared opt-in、static/shared、寄存器、local和occupancy，资源不足在启动前拒绝。

none 的 uint16 索引为 `yW+x`。SW128固定W=64、16B为一组；1024B对齐使重复模式从零偏移开始，逻辑 `(x,y)` 对应：

`y×64 + ((x/8) XOR (y%8))×8 + x%8`。

GMEM→SMEM 的消费者和 SMEM→GMEM 的普通shared写入都使用这个映射。TMA store再还原到常规全局行布局。普通消费者不能把SW128结果当作线性数组直接读。这里不声称改善某个矩阵消费者的bank conflict；正式循环内没有逐元素消费者。[TMA swizzle与回写说明](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#tma-swizzle)

输入方向先初始化mbarrier，expected arrivals=1，用 `fence.proxy.async.shared::cta` 发布并CTA会合。每轮expected transaction bytes为Q，发一次二维tensor copy，再release arrive取得token、acquire try_wait等待完成，最后CTA会合。Q不包含行padding。内部超时读取globaltimer不属于性能采样点；超时必须保留失败并受控清理，不能正常复用尚未完成的shared。

输出方向在计时前填好shared。每个普通写入者执行async proxy fence，再CTA会合，thread0才发一次tensor store、commit group、`wait_group 0`，之后CTA会合。`wait_group.read 0`只有源读取完成语义，不能替代完整输出端点。本阶段固定完整完成，不另增source-release矩阵。[PTX8.8二维方向与完成语义](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor)

## 计量与逐元素验证

工作量为 `blocks×iterations×Q`，Q只计一次；方向分别报告全局读取Q或全局写入Q。描述符访问、控制区、padding、shared初始化、校验回读、计时结束后的G2S普通全局导出都不进入分子。G2S后处理另外记 `blocks×Q` 导出字节，不伪装成第二次TMA。不能从这里推导实际HBM字节或独立TMA引擎峰值。

单CTA分母为该CTA的clock64差；全GPU使用所有CTA的globaltimer包络，CUDA event只做辅助包围检查。计时从输入准备和mbarrier初始化之后开始，覆盖每轮issue、完整wait及CTA控制同步；最后完成之后停止。主机读回和完整验证在其后。

参考位模式按uint16模运算生成：输入方向 `17x+31y+73slot+151b+seed`；输出方向不含slot项，shared在正式循环内不变。所有输出都用其对应预期值的按位取反初始化poison，避免固定16-bit哨兵碰撞。检查完整元素数组、全部行padding、首尾guard，保存值工件和hash，不接受仅checksum或 `errors=0`。

短模式每个profile仍只有一次target启动，内部执行1/2/33个请求，且不导出计时。宿主在每次启动前初始化全局分配：G2S输入保持不变；S2G全部32槽的逻辑payload写入“原shared source预期值的取反”，padding和guard单独初始化；每请求capture数组也初始化为对应原预期值的取反。未访问的S2G槽应保持这个原初值，不能把最终未访槽的poison再取反一次。完整capture的额外字节数及uint64溢出界须单独记录。

单个短target内，G2S在每请求前由全部线程按实际none/SW128映射重置shared为当前槽预期值的取反；每个写者执行shared async proxy fence，再CTA会合，thread0才开始expectQ及copy。完成acquire并CTA会合后保留全部物理tile和逻辑消费者输出，捕获结束再CTA会合后进入下一次reset。S2G共享源只准备一次；每请求前由普通global写入重置本CTA当前slot payload，每个写者执行 `fence.proxy.async.global` 再CTA会合，然后才发tensor store。fullwait0与CTA会合后，各消费者执行global async proxy fence并捕获完整结果；再次CTA会合，才能重置下一槽。所有reset/capture属于同一个短kernel，无隐藏辅助launch；正式循环没有这套逐请求reset/capture。

全部68点先各完成三个固定配对短profile：`(1,seed0)`、`(2,seed3)`、`(33,seed4294967295)`。每对独立进程一次target，使用既有顶层seed，合计每case三次，不作九种组合；33轮验证32槽回绕。短模式每次完成后、复用前保留并验证全部逻辑元素及SW128物理布局；尚未访问的槽保持原poison。只有全家族全部profile完成后才允许pilot或预热。

pilot固定32轮，覆盖所有槽。校准按既有受审公式确定并冻结128–65536正式轮数，同case所有进程和恢复使用同长度。正式G2S保留最后一次共享tile的全部元素，独立核对最后槽 `(I-1)%32`；正式S2G检查全部32槽和padding/guards。所有轮次完整完成计数、timeout和等待信息也保留。

## 后续验收

合同顶层 `sass_contracts` 冻结B2要求：描述符参数空间无local副本、s32坐标与64bit地址、单请求回边、方向对应完成与代理fence、SW128生产/消费映射、外部计时端点和内部timeout读的区分，以及实际资源无spill；具体符号与SASS token以真实编译为准。

代表指令只能证明目标编译形式；实际descriptor编码、两个方向的全部合法布局、swizzle索引、全量输出、源/目标生命周期和资源仍须独立B验收。CPU负例覆盖64KiB SW128、坏stride/坐标/对齐、工作量翻倍、padding混入Q、错误swizzle基偏移、最后元素损坏、poison碰撞、整数溢出及缺失短profile。旧S14的通过不能自动授予S15资格。
