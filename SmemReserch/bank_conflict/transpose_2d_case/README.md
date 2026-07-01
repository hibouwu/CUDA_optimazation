# Two-dimensional transpose bank-conflict cases

This benchmark makes the classic pitch effect explicit. For lane `lane` and
warp row `warp`, the transposed address is:

```text
linear_index = lane * pitch + warp
bank = linear_index % 32
```

| Case | Operation | Result |
|---|---|---|
| `load_pitch32` | `ld.shared` | all lanes in a warp address one bank |
| `load_pitch33` | `ld.shared` | padding rotates lanes across all banks |
| `store_pitch32` | `st.shared` | all lanes in a warp address one bank |
| `store_pitch33` | `st.shared` | padding rotates lanes across all banks |

The benchmark uses eight warps so each kernel covers eight columns of a
32-row tile. Initialization and final result preservation occur once per
kernel; the volatile load/store loop is the dominant work at normal iteration
counts.

```bash
CUDA_ARCH=110 ./scripts/build.sh
./scripts/run_basic.sh
./scripts/run_ncu.sh
```

