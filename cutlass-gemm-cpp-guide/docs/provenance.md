# 来源与许可

## 上游依赖

- NVIDIA CUTLASS v4.6.1，完整 SHA见 `versions.lock.json`；
- NVIDIA PTX ISA关于 `cp.async.bulk.tensor`、`tcgen05.cp`、`tcgen05.mma`、`tcgen05.ld`；
- CUTLASS Blackwell C++ examples、collective builders和 unit tests。

CUTLASS submodule保留上游 `LICENSE.txt`。本仓库没有修改 submodule，也没有在构建时改写
上游 device guard。

## 本地研究素材

子项目的 shape-ledger和 evidence-state方法参考了父仓库中已存在的 SM110研究，但所有
canonical case、CMake、runner和 host oracle均在 `cutlass-gemm-cpp-guide/` 中重新实现。
它不依赖父仓库其他 target或历史构建产物。

没有迁移：

- 历史 binary；
- performance CSV；
- raw NCU result；
- 机器绝对路径；
- Perl改写 CUTLASS source/header 的脚本；
- 未解决 rebase中的 working-tree内容。

## 引用规则

若新增文件实质复制 NVIDIA代码，必须保留 NVIDIA copyright和 SPDX；若只依据 API
重新实现，则在章节末给出官方 source link。第三方博客只能用于理解，不能替代 NVIDIA
官方 API/ISA contract。
