# Structured-sparse block-scaled GEMM

## v0.1 candidate

[`sparse_bs_nvfp4_1sm_p128x128x256`](../cases/sparse_bs_nvfp4_1sm_p128x128x256/) 固定：

```text
A             NVFP4, structured 2:4
B             NVFP4 dense
problem/tile  128 x 128 x 256
schedule      KernelSparseTmaWarpSpecialized1SmNvf4Sm100
MMA           tcgen05.mma.sp...mxf4nvf4.block_scale.block16
```

## 已闭合

- CUTLASS C++ builder/template实例化；
- `sm_110a` binary；
- function-local PTX/SASS，包含 TMA、UTCCP、sparse block-scaled MMA、LDTM；
- host 2:4 compression/decompression primitive及非法 3:4 负例。

## 未闭合

CUTLASS compressor输出的 compressed A / metadata E 尚未在本仓库与独立 logical A CPU
oracle完成 Thor full-GEMM 对比。因此 runtime入口明确返回 `NOT_RUN`；release workflow会把它
视为失败，不允许静态 sparse支持冒充数值支持。

后续闭环必须：

1. 生成满足 2:4 的 logical A；
2. 用 CUTLASS compressor只生成 device compressed A/E；
3. CPU reference仍使用 compressor之前的 logical A；
4. 运行 sparse GEMM并比较完整 D；
5. 保存 metadata、input、binary和结果 hash。
