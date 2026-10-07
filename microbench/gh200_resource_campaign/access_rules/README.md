# GH200 访问规则微基准入口

[V04失效项补测计划](../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/PLAN.md)规定本轮R10/R13/R15/R14与V05的范围。各组使用自己的`run_rXX.py`、`analyze_rXX.py`和`probes/rXX.cu`；唯一新增公共头`probes/gaps_common.hpp`提供显式stride、padding与数值见证。本轮采样与独立验收已交付，V05迁移未整体通过；可用规则与观测限制见对应实验页。下面保留已有R00的使用说明。

## R00：完整 GEMM 与目标 WGMMA

[实验计划](../../../Docs/ModelEvaluation/gemm/experiments/gh200_sm90/access_rules/README.md)。本文件说明 R00 的用法；其他组的入口、参数和结果见各组计划文件。代码格式按本目录 `.clang-format`（C++）与 `black --line-length 100`（Python）。

### 一次运行

在有效单GPU GH200 Slurm分配中加载CUDA 12.9，准备官方CUTLASS v3.9.2源码路径：

```bash
romeo_load_armgpu_env
spack load cuda@12.9.0

# 30个默认配置；包括编译、检查、共同长度校准、预热、正式采样和报告
python3 microbench/gh200_resource_campaign/access_rules/run_r00.py \
  --cutlass-root /path/to/cutlass-3.9.2 \
  --output /path/to/new-r00-run
```

源码：[CUTLASS v3.9.2](https://github.com/NVIDIA/cutlass/tree/v3.9.2)。使用scratch等有配额余量的位置；编译依赖和原始数据会占用空间。`--output`首次运行必须是新目录。

| 选项 | 行为 |
|---|---|
| `--list` | 列出15个GEMM和15个WGMMA配置，不需要GPU |
| `--smoke` | 15个小规模验证点，GEMM全输出检查；不作为正式参数 |
| `--component gemm` / `wgmma` | 只采对应15个点，报告注明范围 |
| `--compile-only` | 编译并导出SASS，不采样 |
| `--resume` | 同GPU UUID、源码、配置、环境和二进制不变时续跑；保留部分失败记录 |

正式GEMM和整卡WGMMA每配置10个独立进程；单CTA WGMMA默认3个，CV>1%时补到10个。接近计时分辨率/样本变化的细小配对差仍需额外匹配复测，不能只凭3个进程解释。WGMMA使用所有参与点的pilot选择共同长度。

CPU重算与单元检查：

```bash
python3 microbench/gh200_resource_campaign/access_rules/analyze.py --input /path/to/r00-run
python3 -m unittest discover -s microbench/gh200_resource_campaign/access_rules -p 'test_*.py' -v
```

`analyze.py`重算工作量和时间，核对WGMMA全部原始输出及哈希，生成`cases.csv`、`summary.json`、`report.md`和图。绘图需要matplotlib；可以把完整结果目录复制到有该依赖的CPU环境重算。

### 实现与计时

- `probes/r00_gemm.cu`：cuBLASLt固定首个合法启发式候选，64MiB workspace；纯FP32使用PEDANTIC。FP8使用TN列主序布局适配、scale=1、fast accumulation关闭，B/D适配不在窗口内。
- `probes/r00_cutlass.cu`：FP16→FP32，128×256×64、cluster2×1×1、384线程、cooperative、4 stage，显式128×32 epilogue tile，ElementC=void。不读取C，也不为C分配epilogue输入缓冲。
- `probes/r00_wgmma.cu`：6种编译特化、15种启动/范围配置；每链每轮16条、commit一次、wait1、最终wait0和片上结果排空在窗口内。GMEM结果导出在窗口后。
- `probes/r00_common.hpp`：输入、事件计时、分层抽样及CPU FP64参考。输入是有符号、坐标相关、各精度可精确表示的有限见证；任务容差仍未指定。

WGMMA采用128B swizzle及128B padded行，描述符只覆盖指令K=16/32。短检查使用非均匀坐标输入和1/2条操作；正式FP16/BF16使用均匀正输入，FP8在16条批内交替正/负A，使累加有界。全部准备、占用和输入形式随结果记录，不能直接外推任意GEMM输入。

GEMM用CUDA event包围单次调用；源分配、适配、启发式与驱逐准备在窗口外。WGMMA单CTA用clock64，整卡用globaltimer包络。没有负载内频率遥测，频率记录为未知；不由其他kernel换算。

### 结果目录

`source/`保存实际探针、运行/重算脚本及CUTLASS依赖头文件；`build/`保存命令、日志、SASS和二进制身份。`samples/<case-id>/<trial>/`保存命令、进程、原始时间、数值检查及输出；WGMMA全输出压缩为`output.f32.gz`，保留未压缩内容哈希。

大GEMM保存4096个检查位置及实际值，误差范数是抽样相对L2误差，不能称完整矩阵Frobenius误差。小规模验证与任务精度要求分别解释。失败构建和试验目录保留，源码改变使用新run。
