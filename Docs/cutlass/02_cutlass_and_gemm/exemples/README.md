# Thor GEMM 本地示例

这些文件与[编程入口指南](../01-thor-gemm-programming-guide_zh-CN.md)的第五、六章对应。每个 `.cu` 都是独立程序，按正文小节组织注释，包含类型配置、输入准备、GEMM 调用与 CPU 参考比较。这里保留各文件自己的调用与错误检查代码，方便单独阅读，不要求先学习额外的示例框架。

固定 CUTLASS 版本为 `8f50b052e1099fb982392a622caab69b97b63128`，使用 CUDA 13.x 为 `compute_110a/sm_110a` 生成代码。它们是固定输入的功能教学示例，不是性能基准，也不是对全部尺寸和格式组合的验证。

## 文件与教学范围

- [dense_baseline.cu](dense_baseline.cu)：第五章的完整基线。FP16 A/B、FP32 累加与 C/D，全 RowMajor，问题为 `(256,256,128,1)`。
- [nvfp4_block_scaled.cu](nvfp4_block_scaled.cu)：从有限的原始浮点输入开始，在 CPU 上确定 Tensor Scale、逐块 UE4M3 Scale 和 E2M1 Payload；用 HostTensor 的子字节访问写入 packed FP4，再按 Kernel 的 SFA/SFB Layout 填充 Scale。采用 Example 72a 的 Tile/Cluster、NVFP4 A/B 和 BF16 C/D，问题为 `(256,1024,256,1)`。独立比较量化输入的 GEMM 结果，同时报告输入和输出的量化误差。全零张量/块有明确约定；非零 Scale 下溢时报告错误，不将下溢当作合法的零 Scale。
- [grouped_gemm.cu](grouped_gemm.cu)：使用正文三组 `(128,512,128)`、`(256,256,128)`、`(64,768,256)` 问题，共七个输出 Tile。采用 Example 75 的 E4M3 A/B、FP16 C/D 和 1SM Pointer-array 配置，演示不同 Shape、地址和 Stride 数组及动态 Cluster，并逐组验证。矩阵集中分配，但各组仍有独立描述。
- [moe_expert_gemm.cu](moe_expert_gemm.cu)：采用 Example 92 的 `MoEProblemShape` 与最大槽位寻址，三个 Expert 的实际 Token 数为 `8、17、32`，最大槽位容量为 32，输入/输出特征数均为 128。提供固定 Top-1 路由，展示 Token 聚集、单次 Expert GEMM 和按原 Token 顺序恢复。参考读取聚集前的数据与路由记录。它不包含 Router 网络、Top-k 加权合并或完整 Expert FFN。
- [attention_unfused.cu](attention_unfused.cu)：对应第六章的分离式 Attention。单 Batch、单 Head，`S_q=S_k=256`、`d=128`、`d_v=256`；Q/K/V 为 FP16，分数、累加和输出为 FP32。两次 Tensor Core GEMM 之间用独立 CUDA Kernel 执行因果 Mask 与 Softmax，并显式将 P 转为 FP16。参考也计入这一步转换，分别检查 QK、概率和 PV/端到端结果。Softmax 每个线程处理一行，便于理解，不是高性能实现。由于 Q/K 等长，这里采用 `j<=i` 的因果条件，不涵盖不等长序列、全 Mask 行或 GQA。正文后续的在线 Softmax、Correction 与专用 Mixed-input 融合 Kernel 不在此文件中实现。

## 编译

在 `CUDA_optimazation` 仓库根目录执行。将 `CUTLASS_ROOT` 改为实际的固定版本源码路径，并使用 CUDA 支持的宿主编译器和系统环境：

```bash
CUTLASS_ROOT=/home/jianyeshi/Note/GPUexpe/cutlass
EXAMPLE_BUILD_DIR=$(mktemp -d /tmp/thor-gemm-examples.XXXXXX)

for name in dense_baseline nvfp4_block_scaled grouped_gemm moe_expert_gemm attention_unfused; do
  nvcc -std=c++17 --expt-relaxed-constexpr \
    -gencode arch=compute_110a,code=sm_110a \
    -I"$CUTLASS_ROOT/include" \
    -I"$CUTLASS_ROOT/tools/util/include" \
    "Docs/cutlass/02_cutlass_and_gemm/exemples/$name.cu" \
    -o "$EXAMPLE_BUILD_DIR/$name" || exit 1
done

printf '示例程序目录：%s\n' "$EXAMPLE_BUILD_DIR"
```

只编译某个场景时，将循环中的名称列表缩减为对应文件名即可。修改元素类型、Layout、Tile、Cluster 或 Schedule 后，需要重新编译；本目录不自动选择其他 GPU 架构或回退到 CPU。

## 运行与结果含义

在具有可用 NVIDIA 驱动、且支持上述编译目标的 Thor 设备上，沿用编译时的 `EXAMPLE_BUILD_DIR`：

```bash
"$EXAMPLE_BUILD_DIR/dense_baseline"
"$EXAMPLE_BUILD_DIR/nvfp4_block_scaled"
"$EXAMPLE_BUILD_DIR/grouped_gemm"
"$EXAMPLE_BUILD_DIR/moe_expert_gemm"
"$EXAMPLE_BUILD_DIR/attention_unfused"
```

- 返回 `0`：该程序的 GPU 输出满足当前固定输入的参考比较要求。
- 返回 `1`：数值、概率或路由恢复检查未通过；查看各阶段打印的 `FAIL` 和最大误差。
- 返回 `2`：CUDA、CUTLASS 或输入准备失败，程序打印 `ERROR` 与原因。没有可用 GPU 时不会打印数值验证 `PASS`。

容差只针对各程序的输入范围和中间/输出精度。NVFP4 的量化误差单独报告，不用于放宽 GEMM 计算检查；Grouped/MoE 参考读取实际 E4M3 输入并计入 FP16 输出转换；Attention 的概率在存入 P 时转换为 FP16，因此其行和只在容差内接近 1。

编译和链接成功只能证明类型、接口与代码生成可用，不能代替目标设备上的数值验证。运行这些程序也只验证所列固定场景，不构成性能或完整模型正确性的结论。

## 本轮验证状态（2026-09-07）

- 四个新增程序已在 CUDA 13.0、Ubuntu 24.04 的环境中编译并链接，均包含 `sm_110a` 设备代码；Attention 包含两套 GEMM 和独立 Softmax。
- NVFP4 的 CPU 量化函数另行通过了不使用 GPU 的检查，覆盖已知量化值、packed FP4 读写、Scale 布局、全零输入、非有限输入拒绝与 Scale 下溢拒绝。这不等于 NVFP4 GEMM 数值验证通过。
- 沙箱内无法访问 CUDA 设备，程序报告 `ERROR: no CUDA-capable device is detected`。沙箱外确认主机具有 RTX 5070 Laptop GPU，计算能力为 12.0，驱动版本为 580.173.02；此前“主机没有 GPU”的判断不能成立。
- 五个示例在沙箱外实际运行均返回 `ERROR: Error Internal`，退出码为 `2`。进一步用 CUDA 驱动 API 加载 Dense 可执行文件中的 `sm_110a` CUBIN，返回 `CUDA_ERROR_NO_BINARY_FOR_GPU`（209）：当前 Thor 目标设备代码不适用于本机 GPU。尚未取得 GPU 数值验证结果，也不能把这次启动失败归为矩阵结果不正确。
