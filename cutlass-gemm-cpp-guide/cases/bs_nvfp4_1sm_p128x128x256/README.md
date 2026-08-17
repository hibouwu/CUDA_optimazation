# `bs_nvfp4_1sm_p128x128x256`

NVFP4 与 MXFP4 都使用 E2M1 value，但 NVFP4 使用 UE4M3 scale 和 SV16。文档及
manifest 把 value type、scale type、SV、accumulator 和 output 分开记录。

```bash
cmake --build --preset sm110a-gpu --target bs_nvfp4_1sm_p128x128x256
./build-sm110a-gpu/bs_nvfp4_1sm_p128x128x256 --verify --seed 20260817
```
