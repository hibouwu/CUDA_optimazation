# ASM01 复核与更正：NVCC 接受相邻的无逗号 asm 输入操作数

日期：2026-09-08。固定源码提交：`8f50b052e1099fb982392a622caab69b97b63128`。

## 当前结论：撤回“固定 CUDA 源码错误”的定性

**撤回旧 [ASM01 审查](phase-1-bitfield-asm-review.md)中“12 处缺逗号是固定源码错误”的结论。** 旧实验只证明 GCC/Clang 的 GNU 主机 C++ asm 解析器不接受该写法，不能替代目标 NVCC/CUDA 设备前端的判断。

本轮直接验证 NVCC 13.0.88：固定 `copy_sm100.hpp` 中同一段 12 个 UTCCP wrapper 保持原文不变，由 12 个真正的 `__global__` entry 分别调用，在 `sm_100a` 与 `sm_110a` 均通过 NVCC 设备编译、PTXAS 和反汇编检查。每个 entry 发射一条 `tcgen05.cp` PTX 和一条 `UTCCP` SASS。每个架构上，原文与“补逗号”对照的完整 PTX、NVCC 生成的 CUBIN 均逐字节相同。

因此，当前证据支持的准确分类是：**已测试 NVCC 13.0.88 设备前端接受，而当前 Tree-sitter/GNU 主机 grammar 不接受的语法兼容差异。** 这 12 项应回到阶段 1 的 NVCC 方言解析与精确操作数映射义务，不能作为修改固定快照的理由，也不能因 NVCC 能编译就跳过 operand 分析。

本报告没有声称该写法是所有 CUDA 版本都保证的正式语法合同。没有取得 NVCC 13.1 的实际执行证据，不把本轮 13.0.88 结果写成 13.1 验证通过。源码里的 SM100 命名、SM110 目标以及版本条件全部保留原文。

## 工具链与环境边界

本机只读发现并实际使用：

- NVCC：`/usr/local/cuda-13.0/bin/nvcc`，CUDA 13.0，`V13.0.88`，build `cuda_13.0.r13.0/compiler.36424714_0`。
- PTXAS：同一工具链目录，`V13.0.88`。
- 反汇编：同一目录的 `nvdisasm`，`V13.0.85`。
- Host compiler：`/usr/bin/g++-14`，GCC `14.4.1 20260724 (Red Hat 14.4.1-2)`。
- 目标：分别明确指定 `-arch=sm_100a`、`-arch=sm_110a`，不依赖当前 GPU 自动探测。

只读 Docker 镜像清单中有标记为 CUDA 13.0、12.8、12.4 的现有镜像，没有找到 CUDA 13.1 标签；这不是对所有未标注镜像内部版本的穷尽证明。本轮没有拉取镜像、安装工具、启动容器或修改系统配置。

初次本机 NVCC 编译被 glibc/CUDA 13.0 头文件的 `rsqrt/rsqrtf` exception specification 冲突阻挡；它与 asm 拼写无关。仅取消 `_GNU_SOURCE` 能让无额外标准库的最小 probe 通过，但包含原始 `cute/config.hpp` 后又暴露 `fwide/uselocale/pthread_mutex_timedlock` 等声明缺失。最终只在编译命令上使用：

```text
-Xcompiler -U_GNU_SOURCE -D_DEFAULT_SOURCE=1 -D_POSIX_C_SOURCE=200809L
```

这些标志调节本机 Host 头文件的 feature-test 宏，不改 CUDA 指令、源文件或系统头文件。所有失败日志保留，不能把 Host 兼容问题归因于那 12 个 wrapper。

## 原始源码、条件和对照

输入仍是旧审查使用的 `snapshot/include/cute/arch/copy_sm100.hpp:366–599`，12 个完整 wrapper 的物理源码段。输入整文件 SHA-256：

```text
0fe82cbfc7c5a5e4629e16efa992d18801f1c136acb5fb63401f396e5965b305
```

所取原文段 SHA-256：

```text
5f4a803bb36afbbc0297fa00b4b2457fe6f311e8ecee6b39940387a48d1d380a
```

第一处保持为：

```cpp
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    asm volatile ("tcgen05.cp.cta_group::1.128x256b [%0], %1;"
    :
    : "r"(dst_addr)  "l"(src_addr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use UTCCP without CUTE_ARCH_TCGEN05_TMEM_ENABLED.");
#endif
```

raw 输入没有加逗号。单独的 `comma_control.cu` 只将 12 个物理 `"r"(dst_addr)  "l"(src_addr)` 改为等长的 `"r"(dst_addr), "l"(src_addr)`，明确标为反事实对照。两份 probe 都实际包含固定快照中的 `cute/config.hpp` 和 `cute/arch/config.hpp`，由它们导入原 `cutlass/arch/config.h`；没有自己定义 `CUTE_ARCH_TCGEN05_TMEM_ENABLED`。

Probe 在 `__CUDA_ARCH__` 已定义的设备阶段增加两项编译时检查：原配置必须定义 `CUTE_ARCH_TCGEN05_TMEM_ENABLED`，实际 `__CUDA_ARCH__` 必须分别等于 1000/1100。未启用原分支时会直接 `#error`，因此本轮成功不是偷偷选中了空的 fallback 分支。

原版本与架构归属也需分开：

- 固定 `cutlass/arch/config.h:85–103` 在 CUDA 版本条件满足时，以实际 `__CUDA_ARCH__ == 1000` 和 `__CUDA_ARCH_FEAT_SM100_ALL` 建立 SM100a 配置。
- 同文件 `131–147` 原注释是 `SM110 and SM110a only on 13.0 and above`，原条件包含 `__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 0`，随后检查实际 `__CUDA_ARCH__ == 1100` 与 `__CUDA_ARCH_FEAT_SM110_ALL`。
- 固定 `cute/arch/config.hpp:100–113` 明确让 SM110A/SM110F 配置定义 `CUTE_ARCH_TCGEN05_TMEM_ENABLED`；`134–137` 的 SM100A/SM101A/SM103A 路径也定义它。

CUDA 13.1 会满足上述源代码的版本谓词，这是条件表达式的静态解读，不是实际执行了 NVCC 13.1。`SM100_UTCCP_*` 是固定接口命名，不能据此前缀把实际 `sm_110a` 编译目标改写为 `sm_100a`。

## 区分“未使用”与真正设备发射

最终每个 wrapper 对应一个独立 entry：`asm01_emit_00` 到 `asm01_emit_11`。它们与原 wrapper 名字的完整对应表保存在 [metadata.json](nvcc-asm01-probe/run-o6wkuhtz/metadata.json) 的 `entry_mapping`。

未定义 `PROBE_EMIT` 时，12 个 entry 都是空体，wrapper 没有被调用；定义后，每个 entry 真正调用对应的原始静态 `copy`。未使用组只是一项必要的对照，不能单独证明 asm 已参与代码生成。

| 目标 | 输入 | 是否调用 wrapper | NVCC → PTX | NVCC → CUBIN | 独立 PTXAS | PTX / SASS 中的对应指令 |
| --- | --- | --- | --- | --- | --- | --- |
| sm_100a | raw | 否 | 通过 | 通过 | 不需要重复 | 12 个 entry，各 0 条 |
| sm_100a | 有逗号对照 | 否 | 通过 | 通过 | 不需要重复 | 12 个 entry，各 0 条 |
| sm_100a | raw | 是 | 通过 | 通过 | 通过 | 12 个 entry，各 1 条 |
| sm_100a | 有逗号对照 | 是 | 通过 | 通过 | 通过 | 12 个 entry，各 1 条 |
| sm_110a | raw | 否 | 通过 | 通过 | 不需要重复 | 12 个 entry，各 0 条 |
| sm_110a | 有逗号对照 | 否 | 通过 | 通过 | 不需要重复 | 12 个 entry，各 0 条 |
| sm_110a | raw | 是 | 通过 | 通过 | 通过 | 12 个 entry，各 1 条 |
| sm_110a | 有逗号对照 | 是 | 通过 | 通过 | 通过 | 12 个 entry，各 1 条 |

“对应指令”分别指 PTX 的 `tcgen05.cp` 和 SASS 的 `UTCCP`。这不是只做全文件 grep 的合计：独立 verifier 按 12 个 entry 逐一检查，每个被调用的 entry 都有一条；逐一检查了参数 0 的 64-bit 源描述符、参数 1 的 32-bit TMEM 地址与指令操作数绑定。

例如 `sm_110a` 的原文组第一项实际输出：

```ptx
.target sm_110a
// asm01_emit_00 的参数 0 为 u64，参数 1 为 u32
ld.param.u64 %rd1, [asm01_emit_00_param_0];
ld.param.u32 %r1, [asm01_emit_00_param_1];
tcgen05.cp.cta_group::1.128x256b [%r1], %rd1;
```

相应 SASS 包含：

```text
LDCU UR4, c[0x0][0x388] ;
LDCU.64 UR6, c[0x0][0x380] ;
UTCCP.T.S tmem[UR4], gdesc[UR6] ;
```

2cta wrapper 则对应 `UTCCP.T.S.2CTA...`。这足以排除“未实例化/未使用的设备函数被忽略，所以看起来通过”的解释。

实际调用组的 raw 与有逗号对照具有相同完整产物散列：

| 目标 | 相同的 PTX SHA-256 | 相同的 NVCC CUBIN SHA-256 |
| --- | --- | --- |
| sm_100a | `e1990c68631162170c96988e7307159201a2d65e9c1a58275d3cdb99db5ee139` | `5e918780fa1810e783e62ba6ac28ceff9aa5ee82a01d49b5d7110eee9f11ee02` |
| sm_110a | `49074639f92640d907f76bd4b720ff43a8c2495e5872dbd7d7e7b73d242a1d0c` | `1a39549b016b49f0c37051b2f914a793fba027e35416056999283d7b50fa11fa` |

这是同一架构下 raw/control 的对比，不是声称 SM100a 与 SM110a 两个架构的产物相同，也不是将直接 PTXAS 与 NVCC driver 两条路径要求为相同文件。

## 保留的失败对照，不能混入源错误结论

本轮保留三个结果目录，不覆盖早期失败：

1. [run-_tpkdhwu](nvcc-asm01-probe/run-_tpkdhwu/report.json)：取消 `_GNU_SOURCE` 后，完整 C++ 标准库头的 feature-test 宏仍不够；raw/control 同样被宿主头声明问题阻挡。
2. [run-1z5ect5w](nvcc-asm01-probe/run-1z5ect5w/report.json)：补齐 Host feature-test 标志后，两架构 raw/control 均产生 12 条 PTX。但早期 probe 把 1cta 与 2cta 的全部调用放在同一个 kernel，PTXAS 拒绝在一个函数中混用两种 CTA granularity。这个错误来自 probe 组织方式，raw/control 完全一致，不能归咎于缺逗号。
3. [run-o6wkuhtz](nvcc-asm01-probe/run-o6wkuhtz/report.json)：最终一 wrapper 一 entry，所有实际调用 case 的 NVCC/PTXAS/SASS 检查通过。这个目录才是本报告最终后端证据。

最终目录的 `initial_host_compatibility/` 还保留了完全未加 Host workaround 时的 `rsqrt/rsqrtf` 失败，方便区分环境问题与源语法。

## 复核入口与证据文件

从 atlas 根目录运行，脚本每次建立全新的 `run-*` 目录：

```bash
.venv/bin/python audits/nvcc-asm01-probe/run_probe.py
.venv/bin/python audits/nvcc-asm01-probe/verify_probe.py audits/nvcc-asm01-probe/run-o6wkuhtz
```

实际调用组使用的主要命令形状为：

```bash
/usr/local/cuda-13.0/bin/nvcc -std=c++17 -ccbin /usr/bin/g++-14 \
  -Xcompiler -U_GNU_SOURCE -D_DEFAULT_SOURCE=1 -D_POSIX_C_SOURCE=200809L \
  -I <atlas>/snapshot/include -arch=sm_110a \
  -DPROBE_EXPECT_ARCH=1100 -DPROBE_EMIT --ptx <run>/raw.cu -o <case>/probe.ptx

/usr/local/cuda-13.0/bin/nvcc <相同公共参数> --cubin -Xptxas=-v \
  <run>/raw.cu -o <case>/probe.cubin

/usr/local/cuda-13.0/bin/ptxas -arch=sm_110a -v \
  <case>/probe.ptx -o <case>/direct_ptxas.cubin
```

每条实际命令的完整参数、退出码、stdout/stderr 路径均在最终 `report.json` 与各 case 的 `result.json`，不是只有上面的示意命令。主要原始证据：

- [SM110a 原文组 PTX](nvcc-asm01-probe/run-o6wkuhtz/sm_110a_raw_used/probe.ptx)、[NVCC/PTXAS 日志](nvcc-asm01-probe/run-o6wkuhtz/sm_110a_raw_used/nvcc_cubin.stderr.txt)、[独立 PTXAS 日志](nvcc-asm01-probe/run-o6wkuhtz/sm_110a_raw_used/direct_ptxas.stderr.txt)、[SASS](nvcc-asm01-probe/run-o6wkuhtz/sm_110a_raw_used/nvdisasm.stdout.txt)。
- [SM100a 原文组 PTX](nvcc-asm01-probe/run-o6wkuhtz/sm_100a_raw_used/probe.ptx)、[NVCC/PTXAS 日志](nvcc-asm01-probe/run-o6wkuhtz/sm_100a_raw_used/nvcc_cubin.stderr.txt)、[SASS](nvcc-asm01-probe/run-o6wkuhtz/sm_100a_raw_used/nvdisasm.stdout.txt)。
- [原始 12-wrapper 片段](nvcc-asm01-probe/run-o6wkuhtz/raw_original_section.txt)、[生成的 raw probe](nvcc-asm01-probe/run-o6wkuhtz/raw.cu)、[有逗号对照](nvcc-asm01-probe/run-o6wkuhtz/comma_control.cu)。

执行本轮矩阵的 `run_probe.py` SHA-256 为 `56d54c548688f81ec3ffc854e710c96ed279287b5104739786062b241e25c13e`。`minimal.cu` SHA-256 为 `f7ccb342fe1173ef93fb358624398375649a83177237932aae73234db41b3caf`。

## 阶段 1 的后续义务

不能删除或重写原始的双空格分隔，也不能继续把 GNU parser 的拒绝标为“已确认的 CUDA 源错”。原始 operand spans、constraint 字符串 `"r"`/`"l"`、`dst_addr`/`src_addr`、条件分支和 wrapper 归属都需保留；方言适配必须有精确 source mapping，并区分原文、parser projection 和本轮 NVCC 观察。

本轮只写 probe、新审查及旧审查顶部的撤回说明，没有改 snapshot 或生成器。未加载任何 CUDA module，没有启动 kernel；TMEM、描述符、同步、cluster 实际执行条件、数值正确性及性能都未验证。probe 的参数故意只用于编译，不能拿来运行。
