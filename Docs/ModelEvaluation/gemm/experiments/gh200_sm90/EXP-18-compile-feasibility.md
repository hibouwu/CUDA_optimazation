# S18：编译阶段发现的问题与可用证据

28 个代表配置的实验约定已通过 A 审查；对应的 20 个目标形式已有实际 CUDA 12.9、`sm_90a` 编译证据及独立可行性审查。当前没有 GPU 数值验证或正式性能值。编译作业 `731308`、`731311` 均为 CPU 分配，各以退出码 0 完成。

## 显式 local 数组不能只看源码

第一次编译中，32/128 word 的 volatile 线程私有数组被提升到寄存器，实际没有预期的 local 读写。若直接计时，会把寄存器循环误称为 local 访问。

修复保留数组索引循环，实际 SASS 出现循环内的 `LDL → IMAD → STL`。操作、初值、更新次数和输出要求未改；旧失败形态及编译结果仍保留。

| 目标 | 初版资源 | 修复后的实际编译资源 | 接受边界 |
|---|---|---|---|
| local32 | REG48，stack0，未实现 local 条件 | REG14，stack128 B，spill0 | 证明地址化私有数组及循环内 local 指令存在 |
| local128 | REG168，stack0，未实现 local 条件 | REG14，stack512 B，spill0 | 同上，尚无 GPU 完整数值结果 |
| pressure32 | REG32，stack48 B；spill store/load 各112 B | 保留初版独立压力编译证据 | 循环内实际 spill；不与显式 local 混为同一种来源 |
| pressure128 | REG32，stack568 B；spill store/load 各1112 B | 保留初版独立压力编译证据 | 同上 |

表中 spill 字节是编译器报告的静态信息，不是某次运行的物理流量。运行时还需要核对实际属性、循环长度和完整输出。

## 两项会影响计量的实际指令生成

**同址原子发生了 warp 内合并。** 当前不使用返回值的 u32 加法形式，在 global/shared 同址目标中生成 `REDUX.SUM`，随后每个 warp 一次原子更新。一个128线程 CTA 的一轮仍有128次逻辑更新请求，但只对应4次实际原子更新。若运行5轮，逻辑请求数是640，实际目标原子更新数是20；不能把640标成原生原子指令执行数。独立地址目标需要使用自己的实际生成指令记录，不能套用同址的换算。

**f16→f32 使用了 `HADD2.F32` 的零加数形式。** 指令循环仍执行目标方向的值转换，并将输出反馈到下一轮。这个实验可测量该源码形式的实际服务，不能仅按 `F2F` 名称统计或宣称获得独立转换单元的裸延迟。输入域为1024个可精确表示的 `1 + q/1024` 值，CPU参考逐位检查整个域。

## setmax 的当前边界

独立 PTX 候选实际使用64寄存器，stack32 B，spill store/load各56 B。机器码中的 dec32→inc64 区间未访问编号32及以上的通用寄存器，增配后这些高寄存器的首次引用均为加载初始化。

这是当前编译形态的证据。实际运行前仍需验证单CTA、128线程、初始资源条件、两个阶段全部输出和完成状态。动态寄存器调整涉及 CTA 的寄存器池；不能据此推出跨CTA供给规律或通用重分配成本。

## 来源与下一步

- [28点实验约定](EXP-18-auxiliary-resources.draft.md)
- [A独立审查](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/reviews/S18-A-review.json)
- [实际编译可行性独立审查](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/reviews/S18-feasibility-compile-B-review-r1.json)
- [初版完整编译证据](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s18-feasibility-compile-a/actual-job731308/diagnostics/)
- [local修复后的完整编译证据](../../../../../results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s18-feasibility-compile-b/actual-job731311/diagnostics/)
- [独立CPU参考实现](../../../../../microbench/gh200_resource_campaign/common/auxiliary_reference_v1.py)

下一步补齐 host 启动、实际资源查询和完整输出接口，再对全部28个配置进行适用的短验证。当前20个目标形式的可行性通过，不代表28个配置已测完，也不授予正式性能参数。
