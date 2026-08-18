# v0.1 Capability Matrix

当前表是 2026-08-17 固定工具链的静态状态。运行：

```bash
python3 tools/summarize_coverage.py --root .
```

| Case | Compile | PTX | SASS | Thor runtime | Performance |
|---|---|---|---|---|---|
| `dense_f16_1sm_p128` | PASS | PASS | PASS | NOT_RUN | NOT_RUN |
| `dense_bf16_1sm_p128` | PASS | PASS | PASS | NOT_RUN | NOT_RUN |
| `dense_fp8_1sm_p128` | PASS | PASS | PASS | NOT_RUN | NOT_RUN |
| `dense_f16_2sm_p256x128x128` | PASS | PASS | PASS | NOT_RUN | NOT_RUN |
| `bs_mxfp8_1sm_p128` | PASS | PASS | PASS | NOT_RUN | NOT_RUN |
| `bs_mxfp4_1sm_p128x128x256` | PASS | PASS | PASS | NOT_RUN | NOT_RUN |
| `bs_nvfp4_1sm_p128x128x256` | PASS | PASS | PASS | NOT_RUN | NOT_RUN |
| `sparse_bs_nvfp4_1sm_p128x128x256` | PASS | PASS | PASS | NOT_RUN | NOT_RUN |
| `epilogue_bias_relu_f16_p128` | PASS | PASS | PASS | NOT_RUN | NOT_RUN |
| `tail_dense_f16_p130x129x127` | PASS | PASS | PASS | NOT_RUN | NOT_RUN |

该表不是 legal-space coverage。它只说明这 10 个固定 case 在一个固定
CUTLASS/CUDA/target contract中的证据状态。

## vNext，不计入 v0.1

TF32、INT8、grouped、batched、pointer-array、mixed-input、blockwise/groupwise scaling、
Stream-K、persistent scheduler、量化 output、性能与 NCU。
