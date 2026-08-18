# `bs_mxfp4_1sm_p128x128x256`

Native MXFP4 使用 E2M1 value、E8M0 scale、SV32。其合法 MMA tile K 为 256，因此
本 case 明确使用 `128×128×256`，不复用 MXFP8 的 K=128 说法。

```bash
cmake --build --preset sm110a-gpu --target bs_mxfp4_1sm_p128x128x256
./build-sm110a-gpu/bs_mxfp4_1sm_p128x128x256 --verify --seed 20260817
```
