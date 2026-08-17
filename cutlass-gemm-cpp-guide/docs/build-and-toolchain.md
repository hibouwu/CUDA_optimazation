# 构建与工具链

## 为什么固定版本

TCGen05 属于 architecture-accelerated feature。仓库固定：

```text
CUTLASS v4.6.1
e05f953a5b3d38adc240df2ff928e0421c2abba3
CUDA 13.0.88
GCC 13.3
compute_110a -> sm_110a
```

权威值在 [`versions.lock.json`](../versions.lock.json)。CUTLASS 必须是 submodule，
不能用系统中偶然存在的 checkout，也不能在构建时修改上游 header。

## 初始化

```bash
cd CUDA_optimazation
git submodule update --init --recursive
cd cutlass-gemm-cpp-guide
python3 tools/validate_cases.py --root .
```

## Host-only

```bash
cmake --preset host
cmake --build --preset host
ctest --preset host
```

这条路径不需要 CUDA driver，用于 schema、reference、scale layout、sparse metadata 和文档检查。

## Canonical CUDA container

容器被固定到：

```text
nvcr.io/nvidia/cuda@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6
```

它包含 NVCC 13.0.88 和 GCC 13.3。容器内需要安装 CMake、Ninja 和 Python，
或直接使用文档中的 NVCC compile command。系统 GCC 15 只作为诊断环境，不能产生 release golden。

所有 CUDA target都显式加入：

```text
--expt-relaxed-constexpr
--generate-code=arch=compute_110a,code=sm_110a
```

不使用 `sm_110` 替代 `sm_110a`，不使用仅含 `compute_110a` 的 forward-compatible PTX
冒充目标 cubin。

## 升级规则

CUTLASS、CUDA、container、host compiler 任一变化都必须：

1. 单独提交升级 PR；
2. 重跑 10/10 compile；
3. 重跑函数级 PTX/SASS；
4. 重跑 Thor runtime；
5. 把新证据放入新的 toolchain-lock hash 目录，不覆盖旧 golden。
