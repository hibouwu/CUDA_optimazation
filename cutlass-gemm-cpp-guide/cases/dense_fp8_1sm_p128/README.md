# `dense_fp8_1sm_p128`

非 block-scaled E4M3×E4M3→FP32 case。它用于对比 8-bit alignment、K 粒度和数值
容差；不包含 SFA/SFB，因此不能用来证明 scale-factor `tcgen05.cp` 路径。

```bash
cmake --build --preset sm110a-gpu --target dense_fp8_1sm_p128
./build-sm110a-gpu/dense_fp8_1sm_p128 --verify --seed 20260817
```
