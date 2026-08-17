# `dense_f16_2sm_p256x128x128`

2SM case 的 problem M 固定为 256。一个 CTA pair 协作完成 `cta_group::2` MMA；cluster
为 `2×1×1`。它不能用 `128³` problem 代替，因为 2SM M tile 与 1SM problem 不是同一层 shape。

```bash
cmake --build --preset sm110a-gpu --target dense_f16_2sm_p256x128x128
./build-sm110a-gpu/dense_f16_2sm_p256x128x128 --verify --seed 20260817
```
