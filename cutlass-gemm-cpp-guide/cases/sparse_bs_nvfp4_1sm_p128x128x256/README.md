# `sparse_bs_nvfp4_1sm_p128x128x256`

该目录已经固定 CUTLASS sparse builder、tile、alignment、scale 和 `mma.sp` codegen contract；
host tests 也验证独立 2:4 metadata primitive。CUTLASS compressor metadata 到完整 Thor GEMM 的
独立数值桥仍保持 `NOT_RUN`，所以它不会被 release workflow 误算为 PASS。

```bash
cmake --build --preset sm110a-static --target sparse_bs_nvfp4_1sm_p128x128x256
./build-sm110a-static/sparse_bs_nvfp4_1sm_p128x128x256 --describe --json
```
