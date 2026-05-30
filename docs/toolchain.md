# Toolchain Notes

CUDA projects depend on three moving parts:

- CUDA Toolkit version.
- Host compiler version.
- System C/C++ runtime headers, especially glibc on Linux.

When these versions drift too far apart, `nvcc` can fail while compiling a minimal CUDA include, before project code is involved.

## Known Issue: CUDA 13.0 + GCC 15 + glibc 2.42

Observed environment:

```txt
CUDA Toolkit: 13.0, V13.0.88
GCC/G++:      15.2.1
glibc:        2.42
```

Minimal reproducer:

```cpp
#include <cuda_runtime.h>

int main() {
  return 0;
}
```

Failure:

```txt
exception specification is incompatible with previous function "rsqrt"
exception specification is incompatible with previous function "rsqrtf"
```

Cause:

- CUDA declares device math functions `rsqrt` and `rsqrtf`.
- New glibc headers expose C23 math declarations for `rsqrt` and `rsqrtf`.
- The declarations disagree, and the CUDA front end stops compilation.

This is not caused by the kernels in this repository.

## Recommended Fix: Use a CUDA Container

Use an NVIDIA CUDA devel image with a known-compatible compiler stack:

```bash
docker run --gpus all --rm -it \
  -v "$PWD":/workspace \
  -w /workspace \
  nvidia/cuda:12.4.1-devel-ubuntu22.04 bash
```

Inside the container:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
./build/reduce_bench 16777216
./build/gemm_bench 1024
```

## Alternative Fixes

- Install a CUDA-supported GCC/G++ version and pass it with `nvcc -ccbin`.
- Use a Linux distribution or container image listed as supported by the CUDA Toolkit release notes.
- Use a CUDA Toolkit version that officially supports your host compiler and distribution.

## Architecture Selection

Set `CMAKE_CUDA_ARCHITECTURES` or `-arch` for your GPU:

| GPU family | Example arch |
| --- | --- |
| Turing | `75` / `sm_75` |
| Ampere A100 | `80` / `sm_80` |
| Ampere RTX 30 | `86` / `sm_86` |
| Ada RTX 40 | `89` / `sm_89` |
| Hopper | `90` / `sm_90` |

For CMake:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
```

For direct `nvcc`:

```bash
nvcc -O3 -std=c++17 -arch=sm_86 REDUCE/src/main.cu -o reduce_bench
```
