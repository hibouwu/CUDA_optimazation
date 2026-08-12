# SM110 full-GEMM closure campaign

This directory freezes the distinction between a working Tensor Core
instruction, a complete GEMM, a numerical reference, and a same-precision
performance denominator. `support_manifest.json` is the machine-readable source
of truth. It intentionally reports gaps instead of substituting a nearby data
type.

Audit the current implementation coverage:

```bash
python3 microbench/sm110_full_gemm_campaign/audit_support_manifest.py
```

Current closure-ready paths are FP16→FP32, FP8 E4M3→FP32, and signed
INT8→INT32. BF16, TF32, E5M2, both FP6 encodings, raw unscaled E2M1, and U8
still need complete implementations and references. The current MXFP4 and
NVFP4 CUTLASS example paths are marked partial because their output contracts
do not match the model's FP32 output contract, their generated external source
was not captured in the historical result bundle, and the historical runner
divides by an FP16 cuBLAS result.

The eventual hardware runner must store ten independent trials, exact source
and dependency revisions, compile commands, binary/SASS hashes, raw numerical
validation, and environment snapshots. For integer GEMM it must report OP/s,
not relabel the arithmetic as FLOP/s.
