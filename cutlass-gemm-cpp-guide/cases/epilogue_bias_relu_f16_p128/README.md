# `epilogue_bias_relu_f16_p128`

使用 CUTLASS `LinCombPerRowBiasEltAct<ReLU,...>` 在同一 epilogue 中执行：

```text
D[m,n] = ReLU(acc[m,n] + bias[m])
```

CPU oracle 在未量化的 logical output 上加 per-row bias、执行 ReLU，再按 FP16 output
语义比较完整结果。

```bash
cmake --build --preset sm110a-gpu --target epilogue_bias_relu_f16_p128
./build-sm110a-gpu/epilogue_bias_relu_f16_p128 --verify --seed 20260817
```
