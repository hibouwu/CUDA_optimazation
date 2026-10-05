# RowMajor 布局表达：Host 地址检查

本检查服务于会议上的一个具体问题：只给出相同形状和 RowMajor 标签，是否已经足以表达所有行主序存储？使用固定快照的 `TagToStrideA_t`、`TagToStrideB_t`、CuTe Layout 与 `cutlass::layout::RowMajor` 实际计算地址，并与独立的整数地址公式对照。

## 实际检查

保持主线 M=256、N=256、K=128、L=1，仅改变物理行步长：

- 紧密存储：lda=128，ldb=256。A(1,0) 的元素偏移128，B(1,1) 的元素偏移257。
- 每行增加8个元素：lda=136，ldb=264。对应偏移分别为136和265。

B 的数学坐标使用 (k,n)，CuTe 张量使用 (n,k,l)。每种存储遍历全部 A 与 B 坐标，共核对131072次地址计算。忽略填充的对照在65152处得到不同偏移；把B坐标解释成另一种行列顺序的对照在65533处不同。这些计数只说明此有限检查识别了两类错误，不是全库或GPU覆盖率。

结果见 [results.json](results.json)，实际标准输出见 [run.log](artifacts/run.log)。程序为 [layout_probe.cu](layout_probe.cu)；虽然文件后缀是 `.cu`，最终用 Host C++ 编译器构建和执行，没有 CUDA Kernel、设备分配或 GEMM 调用。动态依赖检查也没有发现 CUDA 运行库。

## 工具链尝试与边界

最初 NVCC 命令的 include 根目录少了 `include/`，编译器拒绝找到头文件；路径修正后，当前 NVCC 13.0 与系统数学头的 `rsqrt`/`rsqrtf` 异常规格声明不兼容，编译失败。没有将这两次尝试记为通过，也没有修改系统头或 CUTLASS 快照。

本问题只需要验证 Host 地址接口，因此改用系统 G++ 和已安装的 CUDA/CCCL 头编译纯 Host 程序，实际编译与运行均成功。结果文件保留完整命令、G++版本、源程序、关键固定头文件和生成二进制的散列；它不代表 NVCC 环境问题已经修复。

在图集根目录重现：

```bash
python3 verification/scheme-layout/run.py
python3 scripts/refresh_presentation.py
```

此脚本要求本机已有 G++ 与结果命令中列出的 CUDA 13.0 头文件，不自动安装工具链。失败时记录失败并返回非零状态。

## 能据此作出的判断

如果方案只承诺紧密布局，没有可变行步长可以是合法的范围取舍。如果方案承诺带填充布局却无法表达实际步长，接口缺少信息；如果已经传入步长而执行忽略它，则是实现没有遵守接口。

本检查**没有证明**当前 Dense GPU Kernel 的 `can_implement` 接受这些填充值，也没有验证 TMA 描述符、GPU访问、GEMM数值或性能。主线页面明确保留这些区别。
