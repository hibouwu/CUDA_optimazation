# `dense_f16_1sm_p128`

Track A 给出 `CollectiveBuilder → GemmUniversal → GemmUniversalAdapter` 的完整调用；
Track B 在 [`docs/tma-tmem-tcgen05.md`](../../docs/tma-tmem-tcgen05.md) 解构同一数据流。

- Problem：`M=N=K=128`
- MMA tile：`128×128×64`，因此 problem K 包含两个 K tile
- instruction K：16
- operand：SS，A/B 都由 TMA 填入 SMEM
- expected PTX：`cp.async.bulk.tensor + tcgen05.mma.kind::f16 + tcgen05.ld`
- forbidden in target mainloop：`tcgen05.cp`

```bash
cmake --preset sm110a-gpu
cmake --build --preset sm110a-gpu --target dense_f16_1sm_p128
./build-sm110a-gpu/dense_f16_1sm_p128 --describe --json
./build-sm110a-gpu/dense_f16_1sm_p128 --verify --seed 20260817
```

数值 PASS 不代表性能结论；本 case 只验证小矩阵的 API、数据流和完整输出。
