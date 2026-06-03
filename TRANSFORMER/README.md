# TRANSFORMER Optimization

This module is the workspace for Transformer-oriented CUDA optimization. It is
separate from `GEMM/` and `REDUCE/` because Transformer workloads are not a
single kernel: they are a pipeline of memory-bound reductions, elementwise
operations, GEMM-like projections, attention score computation, softmax, and KV
cache access.

## Optimization Roadmap

| Stage | Operator | Main bottleneck | Planned baseline |
| --- | --- | --- | --- |
| v0 | data movement / layout check | host/device copy, tensor layout mistakes | CUDA runtime timing |
| v1 | LayerNorm / RMSNorm | reduction + elementwise memory bandwidth | handwritten CUDA |
| v2 | Softmax | numerically stable row reduction | handwritten CUDA + CUB reference |
| v3 | QKV projection | GEMM throughput | cuBLAS / `cublasGemmEx` |
| v4 | Attention score `QK^T` | GEMM + layout | cuBLAS batched/GemmStridedBatched |
| v5 | Online softmax attention | memory traffic from materialized scores | FlashAttention-style streaming |
| v6 | FFN / MLP | two GEMMs + activation fusion | cuBLAS + fused activation |
| v7 | KV cache decode | irregular memory access, small batch latency | paged / blocked KV cache |

## Design Notes

- Keep correctness checks first. Every custom kernel should compare against a
  simple CPU or cuBLAS/CUB baseline.
- Benchmark prefill and decode separately. Prefill is throughput-oriented;
  decode is latency- and memory-layout-sensitive.
- Record tensor shape in every CSV row: batch size, sequence length, hidden
  size, number of heads, head dimension, and dtype.
- Prefer fusing memory-bound elementwise chains before chasing peak FLOPS.
- Treat Tensor Core paths as separate numerical modes: FP32 CUDA core, TF32,
  FP16/BF16 Tensor Core, and FP8 are different benchmark categories.

## Initial Targets

Start with small, controlled kernels before building full attention:

1. `LayerNorm`: naive row kernel -> block reduction -> vectorized load.
2. `Softmax`: max reduction -> exp/sum reduction -> normalized write.
3. `QKV projection`: use cuBLAS baseline and reuse GEMM tuning lessons.
4. `Attention`: materialized attention first, then streaming online softmax.

## Current Benchmark

`transformer_bench` currently runs LayerNorm:

| Version | Kernel | Main idea |
| --- | --- | --- |
| v1 | `layernorm_v1_naive` | One thread handles one row serially. |
| v2 | `layernorm_v2_block_reduce` | One block handles one row with shared-memory reduction. |
| v3 | `layernorm_v3_vectorized` | `float4` vectorized row loads/stores. |

Run:

```bash
./build_cuda13/transformer_bench 1 1024 4096 32 128
```

Output:

```txt
transformer_benchmark.csv
Operator,Version,Batch,SeqLen,Hidden,NumHeads,HeadDim,TimeMs,BandwidthGBps,Matched
```

Run the shape sweep and generate SVG figures:

```bash
BUILD_DIR=/workspace/build_cuda13 ./scripts/run_transformer_experiments.sh
```

Useful overrides:

```bash
PRESET=quick ./scripts/run_transformer_experiments.sh
TRANSFORMER_SHAPES="1:1024:4096:32:128" TRIALS=5 ./scripts/run_transformer_experiments.sh
```

The sweep writes `results/transformer/transformer_sweep.csv`, raw per-shape
outputs, and one latency plus one effective-bandwidth figure per hidden size.
