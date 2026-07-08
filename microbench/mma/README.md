# Thor tcgen05 Benchmark Report

这个目录里的 `run_thor_tcgen05_report.py` 会自动生成、编译并运行 Thor `tcgen05` dense MMA 微基准，最后输出一份中文分析报告。

## 运行环境

需要在带 NVIDIA Thor GPU 的 CUDA 环境中运行，并满足这些条件：

- `python3` 可用。
- Python 能导入 `torch`，脚本用它读取 GPU 名称、Compute Capability 和 SM 数量。
- `nvcc` 可用，并且支持 `sm_110a`：

```bash
nvcc --list-gpu-code | grep sm_110a
```

- `cuobjdump` 可用，脚本用它检查生成二进制中的 MMA SASS 指令。
- CUTLASS include 路径存在：

```bash
ls /opt/pytorch/ao/third_party/cutlass/include
```

脚本里这个路径是硬编码的：

```python
CUTLASS_DIR = Path("/opt/pytorch/ao/third_party/cutlass")
```

如果你的 CUTLASS 不在这个位置，需要先修改脚本里的 `CUTLASS_DIR`。

## 基本运行

从本目录运行：

```bash
cd /xplorer/shijy/mma
python3 run_thor_tcgen05_report.py
```

也可以从上一级目录运行：

```bash
cd /xplorer/shijy
python3 mma/run_thor_tcgen05_report.py
```

默认每个 benchmark 使用 `10000` 次循环。

## 使用已有 Docker 镜像运行

如果宿主机没有安装 `torch`，可以直接用已经带 PyTorch/CUDA/CUTLASS 的镜像运行，避免重新下载 PyTorch 和 cuBLAS：

```bash
cd /xplorer/shijy/mma
docker run --rm --runtime=nvidia \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" \
  -v /xplorer/shijy/mma:/workspace \
  -w /workspace \
  shijy-cutlass-gemm:latest \
  python3 run_thor_tcgen05_report.py
```

这个命令会把当前目录挂载到容器的 `/workspace`，生成的 `benchmark_src/`、`build/` 和 `分析报告.txt` 会直接出现在宿主机当前目录。`--user "$(id -u):$(id -g)"` 用来避免容器把输出文件写成 root 权限。

## 调整压测次数

用 `--iters` 调整循环次数：

```bash
python3 run_thor_tcgen05_report.py --iters 20000
```

脚本支持的参数可以这样查看：

```bash
python3 run_thor_tcgen05_report.py --help
```

当前参数为：

```text
usage: run_thor_tcgen05_report.py [-h] [--iters ITERS]

options:
  -h, --help     show this help message and exit
  --iters ITERS  benchmark iterations
```

## 脚本会做什么

运行后脚本按顺序执行：

1. 读取 GPU 信息和当前 GPU GPC 频率。
2. 生成 18 份 dense CUDA benchmark 源码到 `benchmark_src/`，测试维度为 `launch` × `shape` × `precision`：
   - `launch`: `single_warp_block`、`full_sm_4warp_block`
   - `shape(N*M)`: `m128n64`、`m128n128`、`m128n256`
   - `precision`: `fp4`、`fp8`、`bf16`
3. 生成文件命名格式为 `tcgen05_<launch>_<shape>_<precision>_benchmark.cu`。
4. 用 `nvcc` 编译到 `build/`。
5. 用 `cuobjdump --dump-sass` 检查是否出现预期 dense MMA 指令，并确认源码不包含 `tcgen05.mma.sp`。
6. 分别运行 18 个 benchmark。
7. 生成中文报告：`分析报告.txt`。

## 输出文件

成功运行后会产生或更新：

```text
benchmark_src/
build/
分析报告.txt
```

## NCU 性能采集

主 benchmark 脚本不直接运行 ncu。需要 Nsight Compute counter 时，先运行 `run_thor_tcgen05_report.py` 生成并编译 `build/` 里的二进制，然后运行：

```bash
./run_ncu_reports.sh
```

脚本会扫描当前 dense baseline 的 18 个二进制，为每个二进制导出 `ncu_reports/*.ncu-rep` 和对应 `.log`。默认 ncu 参数来自 `ncu_mma_anomaly_metrics.txt`，该小指标集覆盖吞吐异常定位需要的 cycles、tcgen05 MMA 指令、tensor pipe、warp issue/stall、非 MMA pipe、memory guard 和 launch 边界，GUI 打开速度比全 metrics 报告快很多。

```bash
ITERS=10000 OUT_DIR=./ncu_reports_key ./run_ncu_reports.sh
```

如果需要重新抓全 metrics，可显式覆盖 `NCU_METRICS`：

```bash
ITERS=10000 NCU_METRICS='regex:.*' OUT_DIR=./ncu_reports_full ./run_ncu_reports.sh
```

如果需要回退到 section set 采集，可清空 `NCU_METRICS` 并设置 `NCU_SET`：

```bash
NCU_METRICS= NCU_METRICS_FILE= NCU_SET=full ./run_ncu_reports.sh
```

全 metrics 采集需要系统开放 GPU performance counter 权限；权限未开放时 log 会出现 `ERR_NVGPUCTRPERM`，`.ncu-rep` 不包含硬件 counter 数据。

## 绘图

`plot_tcgen05_results.py` 会从 `分析报告.txt` 解析基础性能结果，并从 `ncu_reports/*.log` 解析 ncu attach 下的 benchmark 输出，生成 CSV 和 SVG：

```bash
python3 plot_tcgen05_results.py
```

输出文件在 `plots/`：

- `benchmark_tflops.svg`
- `benchmark_peak_ratio.svg`
- `ncu_tflops.svg`
- `benchmark_vs_ncu_tflops_singlewarp.svg`
- `benchmark_vs_ncu_tflops_fullsm4warp.svg`
- `benchmark_results.csv`
- `ncu_results.csv`

`分析报告.txt` 里会包含：

- GPU 名称、Compute Capability、SM 数量和频率。
- Launch/Shape/Precision 三维组合的 SASS 指令检查结果。
- 每个组合的 cycles、MAC/cycle/active-block、active blocks、warps/block、Effective TFLOP/s。
- K 规则：K 随 M/N shape 显式配置；当前 dense `cta_group::1` 为 BF16 K=16、FP8 K=32、FP4 K=64。
- `SingleWarpBlock` 使用 `<<<1, 32>>>`；`FullSM4WarpBlock` 使用 `<<<SM数, 128>>>`。

## 常见问题

如果报错 `CUTLASS include path not found`，说明 `/opt/pytorch/ao/third_party/cutlass` 不存在。安装 CUTLASS 或修改脚本里的 `CUTLASS_DIR`。

如果 `nvcc` 报 `sm_110a` 或 `compute_110a` 不支持，说明当前 CUDA 编译器版本不支持 Thor 目标架构，需要换到支持 `sm_110a` 的 CUDA/NVCC 环境。

如果 `torch.cuda` 相关调用失败，先确认当前环境能看到 GPU：

```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

如果 `cuobjdump` 找不到，确认 CUDA bin 目录在 `PATH` 里，例如：

```bash
which cuobjdump
which nvcc
```
