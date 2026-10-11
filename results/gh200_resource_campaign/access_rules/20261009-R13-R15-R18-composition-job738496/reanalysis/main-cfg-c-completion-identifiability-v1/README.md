# 已观测FINAL不能唯一确定图中的完成时间

取738496 cfg_c K4096/aligned/trial0的物理cluster CTA0/66，将两个rank、两个WG、两个输出的全部36个FIRST/MAIN/PERMIT/DONE/FINAL端点固定为原globaltimer值。所有值先用整数减去共同原点，避免绝对大整数与浮点容差混算。另将cluster_initialized固定在最早CTA entry，仅作为宽松的数学入口假设，不称实际初始化完成时刻。

闭合图的两个producer_tail、两个D_visible、source_safe_exit及cuda_complete仍无有限上界。三份完整节点赋值都通过逐边先后约束，且保持全部36个观测端点精确不变：

| 数学见证 | source_safe_exit（相对ns） | cuda_complete（相对ns） |
|---|---:|---:|
| 未定价图最早赋值 | 88544 | 88544 |
| rank0 producer_tail移至89544 | 89544 | 89544 |
| rank0最后D_visible移至89544 | 88544 | 89544 |

1000ns只是展示无上界的任意正数，非拟合价格、实际可实现硬件调度或测得延迟。这里只证明现有标记加无价格偏序不足以唯一决定该cluster完成时间；整体CUDA event不能据此分配每个cluster的退出或D可见时间。

补producer-tail戳可以约束source_safe_exit，但仍不测得D全局可见或完整kernel完成。今后关键CTA判定必须明确采用哪一完成边界，不能将最后consumer FINAL无条件叫完整kernel最后完成者。服务模型或进一步有效观测仍有必要；观察资格保持false。

初次以绝对GT整数直接调用dependency_bounds时，upper+1e-9将大整数转为float并可能向下舍入，产生假冲突。当前脚本明确先整数减原点；问题已交C修复通用比较，不将数值误报当作真实图矛盾。
