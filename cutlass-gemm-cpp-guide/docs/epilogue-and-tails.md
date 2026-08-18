# Epilogue、tail 与负例

## Fused bias + ReLU

[`epilogue_bias_relu_f16_p128`](../cases/epilogue_bias_relu_f16_p128/) 使用：

```cpp
cutlass::epilogue::fusion::LinCombPerRowBiasEltAct<
    cutlass::epilogue::thread::ReLU,
    cutlass::half_t,
    float,
    cutlass::half_t,
    cutlass::half_t>
```

数学语义为 `D[m,n] = ReLU(acc[m,n] + bias[m])`。CPU oracle独立加 bias、做 ReLU，
然后按 FP16 output比较完整矩阵。

## Logical tail 与 physical alignment

[`tail_dense_f16_p130x129x127`](../cases/tail_dense_f16_p130x129x127/) 分开记录：

```text
logical M/N/K = 130/129/127
lda           = 128
ldb           = 136
ldd           = 132
```

logical extent用于数学 reference；physical stride用于 TMA descriptor和 storage。这样不会把
“K=127”误判为“地址一定未对齐”。

## 负例

负例必须在 launch前得到可解释状态：

- pointer故意偏移导致 alignment失败；
- scale storage size不足；
- 非法 2:4 metadata；
- wrong architecture；
- `can_implement` 拒绝。

Segmentation fault、unspecified output或静默 fallback都不算合格负例。
