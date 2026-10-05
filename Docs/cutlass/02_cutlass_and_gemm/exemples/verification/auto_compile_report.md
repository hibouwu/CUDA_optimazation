# Thor GEMM 示例的 Auto 实际编译结果

核对日期：2026-09-08。对应[编程入口指南](../../01-thor-gemm-programming-guide_zh-CN.md)第五、六章。

## 验证范围与方法

五份完整程序中的六套 GEMM 类型均完成编译、PTXAS 汇编、链接和 Host 类型读取。Attention 包含 QKᵀ、PV 两套类型，另有独立 Softmax 设备函数。本轮不执行 GPU Kernel，不重新声明数值或性能结果；此前的 Thor 运行结果见[示例说明](../README.md)。

固定环境为 CUTLASS `8f50b052e1099fb982392a622caab69b97b63128`、CUDA/NVCC 13.0.48、Ubuntu 24.04 和 GCC 13.3.0，目标为 `compute_110a/sm_110a`。编译使用 C++17、`--expt-relaxed-constexpr`，未额外指定 `-O` 优化级别，与正文的基本编译选项一致。反汇编使用宿主 CUDA 13.0.88 的 `cuobjdump`。源码目录中的其他示例有用户修改，但本轮所用 `include/` 与 `tools/util/include/` 相对固定 commit 无差异。

[inspect_auto.cu](inspect_auto.cu) 直接包含原 `.cu` 文件，将原来的 `main` 改名为 `example_main_not_run`。原程序的 Kernel 调用仍参与设备代码生成；新增的 Host `main` 只读取实际实例化类型和 `sizeof`，不调用原示例，也不启动 GEMM。五次编译没有另写一套 Builder 配置，因此类型证据和设备函数都来自文档示例。

验证采用三类证据：

- 类型读取：实际 `DispatchPolicy`、MMA 操作、CTA Tile、Epilogue Tile、Tile Scheduler 和共享内存大小。
- 编译日志：PTXAS 为 `sm_110a` 汇编对应设备入口，随后链接成功。
- 设备指令：在对应函数中检查 PTX 的 MMA/搬运指令及 CUBIN 的 SASS；记录函数符号，避免将不同 Kernel 的指令混为一项结果。

原始类型中的 `cute::tuple<cute::C<...>>` 是 Shape 等别名展开后的形式。下面用尺寸元组简写 Shape；完整 C++ 类型保留在各项 `types.txt` 中。Mainloop Policy 的前三个整数依次为输入缓冲级数、Scheduler 流水级数和 Accumulator 流水级数，含义见固定版本的 [DispatchPolicy 定义](https://github.com/NVIDIA/cutlass/blob/8f50b052e1099fb982392a622caab69b97b63128/include/cutlass/gemm/dispatch_policy.hpp)。

## 编译结果总览

| GEMM 配置 | 输入缓冲级数 | CTA Tile M/N/K | Epilogue Tile M/N | Epilogue SharedStorage 字节数 | MMA 协作范围 |
|---|---:|---|---|---:|---|
| Dense FP16 | 8 | 128/128/64 | 128/16 | 33792 | 2SM |
| NVFP4 Block-Scaled | 3 | 128/256/256 | 128/128 | 100352 | 2SM |
| Grouped E4M3 | 3 | 128/256/128 | 128/64 | 67584 | 1SM |
| MoE E4M3 | 12 | 128/16/128 | 128/16 | 10240 | 1SM |
| Attention QKᵀ FP16 | 8 | 128/128/64 | 128/16 | 33792 | 2SM |
| Attention PV FP16 | 8 | 128/128/64 | 128/16 | 33792 | 2SM |

这些数值属于各自完整配置。Grouped 的 Mainloop/Epilogue Schedule、MoE 的 Mainloop Schedule 是显式选择；它们的 Stage 和 Epilogue Tile 仍自动推导。所有配置中的默认 Kernel Tile Scheduler 都是独立的 Kernel 层选择，不能统一归因于 `KernelScheduleAuto`。

## Dense FP16

- Mainloop：`MainloopSm100TmaUmmaWarpSpecialized<8,2,4,ClusterShape,Sm100>`。
- MMA：`SM100_MMA_F16BF16_2x1SM_SS<half_t,half_t,float,256,128,...>`，A 为 K-major，B 为 MN-major。
- Epilogue：`Sm100TmaWarpSpecialized<4,2,16,true,false>`。
- Scheduler：`PersistentTileSchedulerSm100<ClusterShape,2>`，Cluster 为 `(2,2,1)`。
- 设备证据：PTX 包含 `tcgen05.mma.cta_group::2.kind::f16`，SASS 包含 `UTCHMMA.2CTA` 和 `UTMALDG.3D.2CTA`。

原始证据：[类型](results/dense.types.txt)、[编译日志](results/dense.build.log)、[函数与指令节选](results/dense.instructions.txt)。

## NVFP4 Block-Scaled

- Mainloop：`MainloopSm100TmaUmmaWarpSpecializedBlockScaled<3,2,1,ClusterShape,Sm100>`。
- MMA：`SM100_MMA_MXF4_2x1SM_SS<float_e2m1_t,float_e2m1_t,float,float_ue4m3_t,256,256,16,...>`。类型实参确认 E2M1、UE4M3 和 V=16，不能仅凭操作类名中的 MXF4 判断输入是 MXFP4。
- Epilogue：`Sm100TmaWarpSpecialized<3,2,128,true,false>`。
- Scheduler：`PersistentTileSchedulerSm100<ClusterShape,2>`，Cluster 为 `(2,4,1)`。
- 设备证据：PTX 包含 `tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16`，SASS 包含 `UTCOMMA.2CTA.4X`。这与无 Scale 的普通 FP4 乘法不同。

PTXAS 报告该配置使用 255 个寄存器，并发生寄存器溢出到局部内存。它仍然成功生成代码；这一观察不支持性能优劣结论，也不能据此将 Auto 称为最优配置。

原始证据：[类型](results/nvfp4.types.txt)、[编译日志](results/nvfp4.build.log)、[函数与指令节选](results/nvfp4.instructions.txt)。

## Grouped GEMM

- Mainloop：`MainloopSm100ArrayTmaUmmaWarpSpecialized<3,8,2,ClusterShape,Sm100>`。
- MMA：`SM100_MMA_F8F6F4_SS<float_e4m3_t,float_e4m3_t,float,128,256,...>`，A/B 均为 K-major。
- Epilogue：`Sm100PtrArrayTmaWarpSpecialized<4,2,64,true,false>`，保留逐组地址路径。
- Scheduler：`PersistentTileSchedulerSm100Group<GroupProblemShape<Shape<int,int,int>>,8>`。
- 设备证据：PTX 包含 `tcgen05.mma.cta_group::1.kind::f8f6f4`，SASS 包含 `UTCQMMA` 和二维 TMA 加载。

Cluster 类型中的 M/N 为运行时整数。本轮能够确认动态类型和对应设备分支，但没有运行 Kernel 来判断某次调用实际采用首选 Cluster 还是回退 Cluster，也没有测量调度后的实际执行次数。

原始证据：[类型](results/grouped.types.txt)、[编译日志](results/grouped.build.log)、[函数与指令节选](results/grouped.instructions.txt)。

## MoE Expert GEMM

- Mainloop：`MainloopSm100UmmaMixedTmaCpAsyncWarpSpecialized<12,3,2,ClusterShape,Sm100>`。
- MMA：`SM100_MMA_F8F6F4_SS<float_e4m3_t,float_e4m3_t,float,128,16,...>`，A/B 均为 K-major。
- Epilogue：`Sm100TmaWarpSpecialized<2,1,16,true,false>`。
- Scheduler：`PersistentTileSchedulerSm100Group<MoEProblemShape<Shape<int,int,int>>,3>`，静态 Cluster 为 `(1,1,1)`。
- 设备证据：PTX 包含 `tcgen05.mma.cta_group::1.kind::f8f6f4`、TMA 加载与 `cp.async.cg.shared.global`；SASS 包含 `UTCQMMA`、`UTMALDG.3D` 和 `LDGSTS.E.BYPASS.LTC128B.128`。

这里 Mixed 指两侧采用不同搬运路径；两侧数值类型实际均为 E4M3。

原始证据：[类型](results/moe.types.txt)、[编译日志](results/moe.build.log)、[函数与指令节选](results/moe.instructions.txt)。

## 分离式 Attention 的两个 GEMM

QKᵀ 和 PV 的 Mainloop Policy 都为 `MainloopSm100TmaUmmaWarpSpecialized<8,2,4,ClusterShape,Sm100>`，Epilogue Policy 都为 `Sm100TmaWarpSpecialized<4,2,16,true,false>`，Scheduler 都为 `PersistentTileSchedulerSm100<ClusterShape,2>`，Cluster 为 `(2,2,1)`。

两者的 MMA 操作均为 `SM100_MMA_F16BF16_2x1SM_SS<half_t,half_t,float,256,128,...>`，但 QKᵀ 的 A/B 均为 K-major，PV 的 A 为 K-major、B 为 MN-major。类型读取的枚举值分别为 0 和 1，与固定源码的 `UMMA::Major::K/MN` 对应。它们不是同一套 B 布局。

两个 GEMM 的独立设备函数均包含 `tcgen05.mma.cta_group::2.kind::f16` 与 `UTCHMMA.2CTA`。独立的 `causal_softmax_to_half` 函数也生成了设备入口；本轮没有执行它，更没有编译或验证正文引用的 Python CuTe DSL 融合 FMHA。

原始证据：[两套类型](results/attention.types.txt)、[编译日志](results/attention.build.log)、[逐函数指令节选](results/attention.instructions.txt)。

## 复现与结果边界

在兼容 CUDA 13.0 的 Linux 环境中，从本目录执行：

```bash
export CUTLASS_ROOT=/path/to/cutlass
bash run_auto_compile.sh
```

脚本依次完成五次编译、Host 类型读取和 SASS 导出。可通过 `AUTO_BUILD_DIR` 指定输出目录；默认创建临时目录。脚本不安装依赖、不修改 CUTLASS、不启动 GPU Kernel。原 `.cu` 的数值验证应按示例 README 的独立命令运行。

本次实际编译在已有 `cutlass-dev:cuda13.0` 镜像的临时容器内完成，容器无网络、无 GPU、使用非 root 用户，两个源码目录只读挂载。宿主直接使用 CUDA 13.0.88/GCC 14 曾遇到 glibc 的 `rsqrt/rsqrtf` 声明冲突，关闭 GNU 扩展后又遇到 pthread 声明问题；这些失败发生在系统头文件阶段，不属于示例类型失败。最终成功构建使用上述 Ubuntu/GCC 13.3 环境，没有修改系统头文件。

完整临时产物位于本次工作机的 `/tmp/thor-auto-compile.pLJLds/`，包含 PTX、CUBIN、可执行文件和完整 SASS。仓库保留可读类型、构建日志、逐函数指令节选及[源码和产物摘要](results/manifest.json)；临时目录不作为长期保存位置。记录中的 PTX/SASS 行号指向对应临时完整文件，函数符号可用于重新定位。

本轮结论限定为：五份示例的六套 GEMM 类型确实生成了上述实现，PTXAS 和链接通过，类型信息可在 Host 读取。动态 Cluster 的实际选择、运行时边界行为、数值正确性及性能分别需要对应的运行证据。
