# EXP-11 编译降低修订：warp 逻辑阶段与 mbarrier 超时读取

本修订为待独立 A 审查的草案；CPU 审计器也待另一实施者之外的代理审查。原 [实验约定](EXP-11-synchronization.md)、`synchronization.json`、C++ 源码和原审计器保留。新合同为 [synchronization_lowering_v3.json](../../../../../microbench/gh200_resource_campaign/contracts/synchronization_lowering_v3.json)，不增加原 13 点之外的配置。

## 真实编译边界

依据 `implementation/target-compile-synchronization-a` 中的 CUDA 12.9 V12.9.41、`sm_90a` 编译记录与 SASS。源码 SHA256 为 `20aaf88ca4dbd70df5b77680bf4527f2ee451c9f285015401b7f67673568bba2`，SASS SHA256 为 `f2607fccb114a0899711e6423e49281e1eb78ffc21c26581b4d3a47643196e18`。编译成功不代表 GPU 数值检查或测量完成。

C++ 仍使用原文件，无需重新编译来修改这些解释。新审计器只接受该证据中的正式 kernel 指令流；换编译器、寄存器分配或控制流后，必须重新留存证据并审查，不能只因助记符数量相同就接受。

## 两个 warp 点

`warp_t32_aligned` 与 `warp_t32_fixed_work_skew` 的正式函数均没有 `WARPSYNC`，目标循环内也没有 CTA barrier。当前生成形式删除了源码中的 `bar.warp.sync` 指令服务；起止窗口外层仍有显式 CTA 同步。

两点保留为本次编译后的逻辑阶段服务：每轮 8 个逻辑阶段，共 `2048×8=16384` 个，单位改为 `clock64_cycle/logical_phase/CTA`，`work_model=compiled_warp_phase_v3`，`exportable=false`。aligned 点包含循环与计数开销；skew 点另包含每阶段选定线程的 256 次依赖整数 MAD。它们均不能导出原生 warp barrier 延迟或吞吐参数，也不将此结论推广到其他程序或编译器。

其余 11 点的工作量、单位与资格约定保持原定义。所有点仍需要数值检查和独立 B；合同的 exportable 字段不是测量已经合格的证明。

## mbarrier 的 18 个 globaltimer 读取

四个 mbarrier 正式函数各有 18 个静态 globaltimer 读取位置：测量起止 2 个，加上展开的 8 次等待中各自的超时起点与轮询读取，共 16 个。一个轮询位置可能动态执行多次，不能把静态位置数当成动态等待次数。

原访存审计器只接受函数中恰好两个 globaltimer 读取，因此不适用于此处。新 S11 审计器从正式函数的入口和排空控制流识别测量边界，检查起点后的 clock64 与 shared 时间戳存储、CTA 入口 gate、外层回边、结果 drain barrier 和终点前的 clock64；内部 16 个读取必须位于外层循环中。最终再比对已观察指令流哈希，拒绝未审查的寄存器或分支变化。

每轮实际有 8 个 `SYNCS.ARRIVE` 位置与 28 个 `SYNCS.PHASECHK` 位置，其中 14 个带 TRYWAIT。该数量只描述当前编译后的轮询控制流；每线程实际等待次数仍来自 raw 的 `wait_attempts`。

1 秒超时比较、共享 abort 写入和全 CTA 共同退出路径保留。不得为了满足两个计时读取的旧假设删除超时保护，也不将内部超时读取改成额外性能样本。

## 审计与后续边界

新增 [synchronization_lowering_v3.py](../../../../../microbench/gh200_resource_campaign/auditors/synchronization_lowering_v3.py) 和 [CPU 反例](../../../../../microbench/gh200_resource_campaign/tests/test_synchronization_lowering_v3.py)。反例覆盖缺失测量/超时读取、改变超时阈值或 abort、缺失 arrive/wait、错误 CTA/fence 指令、改变回边/寄存器、丢失或重复函数，以及错误恢复 warp 参数导出。

当前只完成正式函数的 CPU SASS 核对。独立审查仍需核对正确性与到达诊断 kernel、完整输出、超时共同退出和 raw ABI；之后才能安排受控短 GPU 检查。本修订未注册 adapter、未启动 GPU，也未签署 A 或 B 审查。
