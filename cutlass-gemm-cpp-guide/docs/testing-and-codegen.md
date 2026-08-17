# 测试与 codegen 归因

## L0：Host contracts

```bash
cmake --preset host
cmake --build --preset host
ctest --preset host
```

覆盖 schema、shape、scale layout、logical block-scale reference 和 sparse 2:4 primitive。

## L1：Compile/API

每个 case是独立 executable。静态 compile使用固定 container、C++17、
`--expt-relaxed-constexpr` 和 `sm_110a`。`compile_passed` 只表示 builder与 host/device code
成功生成，不表示 `can_implement` 或 runtime PASS。

## L2：Function-local PTX/SASS

```bash
python3 tools/inspect_codegen.py \
  --root . \
  --case bs_mxfp8_1sm_p128 \
  --binary build-container/bs_mxfp8_1sm_p128 \
  --ptx build-container/bs_mxfp8_1sm_p128.ptx \
  --output build-container/evidence/bs_mxfp8_1sm_p128
```

工具要求一个 PTX `.entry` block同时包含全部 PTX pattern，并要求一个 SASS `Function :`
block同时包含全部 opcode family。它不会在整个 binary上做无归属的 grep。

完整 SASS和 binary是 CI artifact；仓库只保留 manifest、hash、过滤后的目标函数 excerpt和
静态 summary。精确指令 count不是默认 gate，因为 ptxas可能合法改变 unroll和调度。

## L3：Thor runtime

Runtime在 self-hosted runner检查：

- GPU name 与 CC 11.0；
- `Gemm::can_implement`；
- fixed seed与边界 pattern；
- full-output independent CPU reference；
- output canary；
- environment/source/binary hash。

wrong-arch 本地执行返回 77/`SKIP`；protected runtime workflow把核心 case的 SKIP视为失败。

## L4：性能

v0.1 不执行性能 gate。`performance_measured` 必须是 `false`。将来加入 benchmark时，需要
另设大 shape、warmup、trial、clock/power state、同精度 reference和原始 artifacts；不得从
`128³` correctness时间推导产品性能。
