# CUTLASS 容器维护

本文档只保留一种工作模式：

- 镜像按需重建
- 容器长期保留
- 平时只做 `start` 和 `exec`

下面假设：

- 工作区目录：`/home/jianyeshi/Note/GPUexpe`
- 镜像名：`cutlass-dev:cuda13.0`
- 容器名：`cutlass-dev`

## 1. 首次构建镜像

在仓库根目录执行：

```bash
docker build -t cutlass-dev:cuda13.0 -f cutlass/Dockerfile .
```

## 2. 首次创建长期容器

只创建一次。容器用 `sleep infinity` 常驻，后面反复进入即可。

```bash
docker run -d \
  --name cutlass-dev \
  --gpus all \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v /home/jianyeshi/Note/GPUexpe:/workspace:Z \
  -w /workspace \
  cutlass-dev:cuda13.0 \
  sleep infinity
```

检查是否启动成功：

```bash
docker ps --filter name=cutlass-dev
```

## 3. 平时进入容器

如果容器已经在运行，直接进入：

```bash
docker exec -it cutlass-dev bash
```

如果容器当前是停止状态，先启动再进入：

```bash
docker start cutlass-dev
docker exec -it cutlass-dev bash
```

## 4. 容器内初始化 CUTLASS 编译数据库

第一次进容器后执行一次：

```bash
cmake -S /workspace/cutlass -B /workspace/cutlass/build-container \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCUTLASS_ENABLE_EXAMPLES=ON \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_ENABLE_LIBRARY=OFF \
  -DCUTLASS_NVCC_ARCHS=100a
```

这一步会生成：

```bash
/workspace/cutlass/build-container/compile_commands.json
```

VS Code IntelliSense 现在就是指向这个文件。

## 5. 容器内构建 72b 示例

`72b_blackwell_nvfp4_nvfp4_gemm` 是 SM100 路径，不适合 RTX 50 / SM120 GPU。RTX 5070 这类 GeForce Blackwell GPU 应优先使用第 6 节的 `79b` 示例。

```bash
cmake --build /workspace/cutlass/build-container --target 72b_blackwell_nvfp4_nvfp4_gemm
```

生成的可执行文件：

```bash
/workspace/cutlass/build-container/examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm
```

## 6. 容器内构建 SM120 GeForce NVFP4 + CLC 示例

你的 RTX 5070 Laptop GPU 在容器内显示 compute capability `12.0`，对应 CUTLASS SM120。最接近 72b 的可运行替代示例是：

```bash
/workspace/cutlass/examples/79_blackwell_geforce_gemm/79b_blackwell_geforce_nvfp4_nvfp4_gemm.cu
```

这个示例是 dense NVFP4/NVFP4 GEMM，注释中明确说明使用了 SW controlled dynamic scheduler based on cluster launch control。

先配置 SM120 构建目录：

```bash
cmake -S /workspace/cutlass -B /workspace/cutlass/build-sm120 \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCUTLASS_ENABLE_EXAMPLES=ON \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_ENABLE_LIBRARY=OFF \
  -DCUTLASS_NVCC_ARCHS=120a
```

构建 79b：

```bash
cmake --build /workspace/cutlass/build-sm120 --target 79b_blackwell_geforce_nvfp4_nvfp4_gemm
```

运行一个小尺寸验证：

```bash
/workspace/cutlass/build-sm120/examples/79_blackwell_geforce_gemm/79b_blackwell_geforce_nvfp4_nvfp4_gemm \
  --m=512 --n=512 --k=512 --iterations=0
```

预期输出包含：

```text
Disposition: Passed
```

其他同类候选：

- `79a_blackwell_geforce_nvfp4_bf16_gemm`: dense NVFP4 input, BF16 output/reference style
- `79b_blackwell_geforce_nvfp4_nvfp4_gemm`: dense NVFP4 input, NVFP4 output，优先看这个
- `79d_blackwell_geforce_nvfp4_grouped_gemm`: grouped NVFP4 GEMM
- `80b_blackwell_geforce_nvfp4_nvfp4_sparse_gemm`: sparse NVFP4/NVFP4 GEMM

## 7. 宿主机 VS Code 调试方式

当前 `.vscode/launch.json` 使用的是 `cppdbg` + `pipeTransport`，由宿主机 VS Code 调用：

```bash
docker exec -i cutlass-dev /usr/local/bin/cuda-gdb-mi
```

因此不需要安装名为 `cuda-gdb` 的 VS Code 扩展。VS Code 弹窗里搜索 `cuda-gdb` 找不到扩展是正常的。

`/usr/local/bin/cuda-gdb-mi` 是容器内的包装脚本，它会把 VS Code 自动追加的 `--interpreter=mi` 等参数正确转发给 `/usr/local/cuda-13.0/bin/cuda-gdb`。

使用前确保容器正在运行：

```bash
docker start cutlass-dev
```

然后在 VS Code 的 Run and Debug 里选择：

```text
Docker CUDA GDB: CUTLASS 79b SM120 NVFP4
```

这个调试项会先通过 Docker 在容器内构建 `79b_blackwell_geforce_nvfp4_nvfp4_gemm`，再用容器内的 `cuda-gdb` 启动调试。

## 8. 容器内构建 CutGEMM 调试版本

```bash
/usr/local/cuda-13.0/bin/nvcc \
  -ccbin /usr/bin/g++-13 \
  -std=c++17 \
  -g -G -O0 \
  -arch=sm_100a \
  --expt-relaxed-constexpr \
  -I/workspace/cutlass/include \
  -I/workspace/cutlass/tools/util/include \
  -I/workspace/cutlass/examples/common \
  /workspace/CutGEMM/gemm.cu \
  -o /workspace/CutGEMM/gemm
```

## 9. 停止容器

不用删除，只停止：

```bash
docker stop cutlass-dev
```

## 10. 需要更新 Dockerfile 时

Dockerfile 改了以后，按这个顺序：

```bash
docker stop cutlass-dev
docker rm cutlass-dev
docker build -t cutlass-dev:cuda13.0 -f cutlass/Dockerfile .
docker run -d \
  --name cutlass-dev \
  --gpus all \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v /home/jianyeshi/Note/GPUexpe:/workspace:Z \
  -w /workspace \
  cutlass-dev:cuda13.0 \
  sleep infinity
```

然后重新执行一次第 4 节的 `cmake` 配置。

## 11. 常用排查命令

看容器状态：

```bash
docker ps -a --filter name=cutlass-dev
```

看容器内 CUDA：

```bash
docker exec -it cutlass-dev nvcc --version
```

看容器内编译器：

```bash
docker exec -it cutlass-dev /usr/bin/g++-13 --version
```

看容器内 `cuda-gdb`：

```bash
docker exec -it cutlass-dev /usr/local/bin/cuda-gdb-mi --version
```

看 GPU 是否透传成功：

```bash
docker exec -it cutlass-dev nvidia-smi
```

如果进入容器后看到 `Permission denied`，尤其是在 Fedora / SELinux Enforcing 环境：

```bash
getenforce
```

如果输出是 `Enforcing`，需要确保挂载参数是 `:Z`：

```bash
-v /home/jianyeshi/Note/GPUexpe:/workspace:Z
```

已经用错参数创建过容器时，直接删掉重建：

```bash
docker stop cutlass-dev
docker rm cutlass-dev
docker run -d \
  --name cutlass-dev \
  --gpus all \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v /home/jianyeshi/Note/GPUexpe:/workspace:Z \
  -w /workspace \
  cutlass-dev:cuda13.0 \
  sleep infinity
```

## 12. 推荐日常流程

每天开始：

```bash
docker start cutlass-dev
docker exec -it cutlass-dev bash
```

进入容器后：

```bash
cd /workspace
```

每天结束：

```bash
docker stop cutlass-dev
```
