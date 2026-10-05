# S15 描述符坐标可行性预检查

仅为 S15 A 的设计输入，尚无独立 A、描述符编码或 GPU 验证。固定 16-bit 元素、rank=2、interleave=none、elementStrides=(1,1)。CUDA 12.9.1 要求每个 boxDim 在 1–256，SW128 的最内层字节宽度不超过 128，global stride 按 16 B 对齐。[官方描述符约束](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)

由这些约束推导，单次二维 SW128 的有效 payload 至多 `128 B × 256 = 32 KiB`。原计划的 64 KiB 坐标可以保留 none，但不能硬套成一次二维 SW128，也不能悄悄改为两次请求或 rank=3。

| payload | 候选 boxDim（元素） | 连续行 stride | padding 对照 stride | 可比较布局 |
|---|---|---:|---:|---|
| 1 KiB | (64,8) | 128 B | 144 B | none / SW128 |
| 4 KiB | (64,32) | 128 B | 144 B | none / SW128 |
| 8 KiB | (64,64) | 128 B | 144 B | none / SW128 |
| 16 KiB | (64,128) | 128 B | 144 B | none / SW128 |
| 32 KiB | (64,256) | 128 B | 144 B | none / SW128 |
| 64 KiB | (128,256) | 256 B | 272 B | none |

先以连续 none 为基线；增加 padding 或 SW128 时一次只改变一个条件，不默认展开二者的组合。上表只解决 box 与 stride 的必要约束；全局尺寸、地址及 shared 对齐、swizzle 的消费者索引、完成事件和实际编码返回值仍需 A/B 验证。padding 不进入有效 payload 分子。
