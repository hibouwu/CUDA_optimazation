# `dense_bf16_1sm_p128`

该 case 与 FP16 control 使用相同 `128×128×64` tile，但 A/B 为 BF16。它验证类型
descriptor 的变化，而不是声称 BF16 使用另一条独立硬件管线。

```bash
cmake --build --preset sm110a-gpu --target dense_bf16_1sm_p128
./build-sm110a-gpu/dense_bf16_1sm_p128 --verify --seed 20260817
```

目标函数需要 TMA、dense TCGen05 MMA 和 TMEM readback，不要求 `tcgen05.cp`。
