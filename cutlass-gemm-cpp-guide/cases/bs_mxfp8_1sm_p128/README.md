# `bs_mxfp8_1sm_p128`

这是首版最高风险、也最重要的完整三路径 case：

```text
Aq/Bq/SFA/SFB --TMA--> SMEM
SFA/SFB --tcgen05.cp--> TMEM
Aq/Bq descriptors + SFA/SFB TMEM --tcgen05.mma.block_scale--> accumulator TMEM
accumulator --tcgen05.ld--> registers --> BF16 D
```

- Problem/tile：`128×128×128`
- value：E4M3
- scale：E8M0
- scale-vector：32，故每行有 4 个 K scale
- accumulator：FP32
- output：BF16

```bash
cmake --build --preset sm110a-gpu --target bs_mxfp8_1sm_p128
./build-sm110a-gpu/bs_mxfp8_1sm_p128 --describe --json
./build-sm110a-gpu/bs_mxfp8_1sm_p128 --verify --seed 20260817
```

在 Thor runtime、函数级 PTX/SASS 和独立 logical-scale CPU oracle 全部通过前，本 case 保持
candidate 状态，不称为 NVIDIA 官方 SM110 C++ end-to-end guarantee。
