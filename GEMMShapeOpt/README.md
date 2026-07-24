# GEMMShapeOpt

这个目录用于专门优化非单一方阵的 GEMM shape，包括：

- 非规则 `M/N/K`，例如不满足 `128/256/64` tile 边界的 shape。
- skinny GEMM，例如 `M` 很小或 `N` 很小。
- GEMV-like shape，例如 `M=1`、`N=1` 或 `K` 很大但输出一维很窄。
- irregular tile tail，即主 tile 后剩余的 M/N/K 边界处理。

当前阶段先做可复现的 shape sweep 和分类，不在没有数据的情况下直接写
专用 kernel。`GEMMsm110` 的 reference 已切换为 cuBLASLt Matmul heuristic；
本目录的 sweep 会复用同一个 benchmark binary，记录每个 shape 下
`cublas_tc/cutlass/tc5a/tc5b/...` 的正确性、GFLOP/s 和 ratio。

## 目录

- `shapes/default_shapes.csv`：默认 shape 集合，覆盖 square、ragged、
  skinny、GEMV-like、LLM decode/prefill 和 tile-tail。
- `shapes/target_shapes.csv`：第一阶段固定优化目标，数量少、覆盖面明确，
  用来决定是否进入专用 kernel 或 runtime router。
- `shapes/smoke_shapes.csv`：最小验证集，用来确认新 binary、cuBLASLt
  reference 和脚本都能跑。
- `shapes/core_shapes.csv`：主要优化决策集，覆盖 square、ragged、tail、
  skinny 和 GEMV-like。
- `shapes/extended_shapes.csv`：扩展 workload 集，偏 LLM/vision 的真实
  非方阵 shape。
- `shapes/optimization_cases.md`：每类 case 的瓶颈假设、比较对象和可能
  优化方向。
- `scripts/run_shape_sweep.sh`：逐 shape、逐 backend 运行
  `GEMMsm110/build/gemm_sm110_bench`，生成 machine-readable CSV。
- `scripts/analyze_shape_sweep.py`：汇总每个 shape 的最快 backend、相对
  reference 的 ratio、失败项和下一步建议。
- `include/shape_policy.hpp`：轻量 shape 分类 helper，供后续 runtime
  router 或专用 kernel registry 复用。

## 运行

先确保 `GEMMsm110/build/gemm_sm110_bench` 已经用 cuBLASLt reference 重新编译：

```bash
cd ../GEMMsm110
./build_and_run.sh build-only
```

然后运行 shape sweep：

```bash
cd ../GEMMShapeOpt
./scripts/run_shape_sweep.sh
```

常用覆盖：

```bash
BACKENDS="cublas_tc cutlass tc5a tc5b" \
SHAPES_CSV=shapes/core_shapes.csv \
OUT_DIR=../results/gemm_shape_opt/manual \
./scripts/run_shape_sweep.sh
```

快速 smoke：

```bash
SHAPES_CSV=shapes/smoke_shapes.csv ./scripts/run_shape_sweep.sh
```

第一阶段固定目标：

```bash
SHAPES_CSV=shapes/target_shapes.csv ./scripts/run_shape_sweep.sh
```

带 epilogue 的固定目标：

```bash
BACKENDS="cublas_tc shapeopt" \
EPILOGUES="none bias relu gelu residual" \
SHAPES_CSV=shapes/target_shapes.csv \
OUT_DIR=../results/gemm_shape_opt/target_epilogue_90_gate_shapeopt_final \
./scripts/run_shape_sweep.sh
```

默认验收门槛是：每个 shape 至少有一个非 reference backend 达到
`cuBLASLt Matmul heuristic` 的 `0.90x`。`cublas_tc` 只作为 reference，
不算达标候选，避免把 reference 自己的 `1.0x` 当成优化结果。常用覆盖：

```bash
SWEEP_MIN_RATIO=0.90 \
REFERENCE_BACKENDS=cublas_tc \
BACKENDS="cublas_tc shapeopt cutlass tc5a tc5b" \
SHAPES_CSV=shapes/target_shapes.csv \
./scripts/run_shape_sweep.sh
```

`shapeopt` 是当前 ShapeOpt runtime router 的验收 backend；现阶段对未验证
超过 90% 的 shape 使用和 `GEMMsm110` reference 相同的 cuBLASLt heuristic
fallback，后续再把已达标的专用 `tc*` kernel 接进 router。

如果只是收集数据、不希望低于 90% 时返回失败：

```bash
ENFORCE_MIN_RATIO=0 SHAPES_CSV=shapes/target_shapes.csv ./scripts/run_shape_sweep.sh
```

扩展 LLM/vision shape：

```bash
SHAPES_CSV=shapes/extended_shapes.csv ./scripts/run_shape_sweep.sh
```

如果只想分析已有 CSV：

```bash
python3 scripts/analyze_shape_sweep.py \
  --csv ../results/gemm_shape_opt/manual/shape_sweep.csv \
  --out ../results/gemm_shape_opt/manual/analysis.md
```

如果要把已有 sweep 结果画成折线图：

```bash
scripts/plot_shape_sweep.py \
  --out-dir ../results/gemm_shape_opt/plots \
  --csv ../results/gemm_shape_opt/target_90_gate_shapeopt_final/shape_sweep.csv \
  --csv ../results/gemm_shape_opt/core_90_gate_shapeopt_retest/shape_sweep.csv \
  --csv ../results/gemm_shape_opt/extended_90_gate_shapeopt_retest/shape_sweep.csv
```

如果要画 epilogue sweep 的折线图：

```bash
scripts/plot_shape_sweep.py \
  --out-dir ../results/gemm_shape_opt/plots_epilogue \
  --csv ../results/gemm_shape_opt/target_epilogue_90_gate_shapeopt_latest/shape_sweep.csv \
  --csv ../results/gemm_shape_opt/core_epilogue_90_gate_shapeopt/shape_sweep.csv \
  --csv ../results/gemm_shape_opt/extended_epilogue_90_gate_shapeopt/shape_sweep.csv
```

NCU profiling：

```bash
OUT_DIR=../results/gemm_shape_opt/ncu/manual \
scripts/run_ncu_profiles.sh
```

当前 NCU 指标解释和权限状态见 `NCU_ANALYSIS.md`。

## 第一阶段目标矩阵

| Category | M | N | K | 用途 |
| --- | ---: | ---: | ---: | --- |
| square | 2048 | 2048 | 2048 | 主线方阵吞吐基线，观察 `tc5a/tc5b` 与 cuBLASLt 的差距 |
| tail_k | 1024 | 1024 | 1000 | K 维 tail cleanup，主 tile 足够多但最后 K-slice 不对齐 |
| tail_mn | 1152 | 768 | 1024 | M/N tail，K regular，验证边界 store/epilogue 成本 |
| ragged | 384 | 520 | 300 | M/N/K 都不对齐，小中型 cleanup 成本可能主导 |
| skinny_n | 4096 | 64 | 4096 | N 很小，观察 TC tile N 方向利用率浪费 |
| skinny_m | 64 | 4096 | 4096 | M 很小，观察 CTA/warp 利用率和 epilogue 空转 |
| gemv_like_m | 1 | 4096 | 4096 | decode 单 row，latency 和 launch overhead 优先 |
| gemv_like_micro | 13 | 17 | 2048 | GEMV-like micro-batch，小 M/N 输出但 K 很长 |
| decode_ffn | 1 | 11008 | 4096 | LLM decode FFN up projection，真实非方阵目标 |

## 优化原则

1. 先用 cuBLASLt heuristic 作为强 reference，避免把自研 kernel 和较弱
   reference 比较后误判。
2. 每类 shape 单独看，不把 square GEMM 的结论直接套到 skinny 或
   GEMV-like shape。
3. 对不规则边界，单独记录 fast tile 区和 cleanup/tail 成本；不能只看
   完整 tile 的吞吐。
4. 对 GEMV-like shape，GFLOP/s 不是唯一指标，还要看 latency、launch
   overhead 和是否能批量融合。
5. 只有同一 shape 下 correctness、ratio、失败原因都记录完整后，才加入
   专用 kernel 或 runtime router。
