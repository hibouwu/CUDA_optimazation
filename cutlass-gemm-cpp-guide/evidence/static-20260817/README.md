# Static evidence: 2026-08-17

该 snapshot 在固定 CUTLASS v4.6.1 / CUDA 13.0.88 / GCC 13.3 / `sm_110a`
container中生成。10/10 case完成：

```text
source_present = 10
compile_passed = 10
ptx_verified   = 10
sass_verified  = 10
runtime_correct = 0
```

Canonical compile形态：

```bash
nvcc -std=c++17 -O3 --expt-relaxed-constexpr \
  --generate-code=arch=compute_110a,code=sm_110a \
  -Iinclude -Ithird_party/cutlass/include \
  -Ithird_party/cutlass/tools/util/include \
  cases/<case-id>/case.cu -o build-container/<case-id>

nvcc -std=c++17 -O3 --expt-relaxed-constexpr -arch=sm_110a -ptx \
  -Iinclude -Ithird_party/cutlass/include \
  -Ithird_party/cutlass/tools/util/include \
  cases/<case-id>/case.cu -o build-container/<case-id>.ptx

python3 tools/inspect_codegen.py --root . --case <case-id> \
  --binary build-container/<case-id> --ptx build-container/<case-id>.ptx \
  --output build-container/evidence/<case-id>
```

完整 binaries/PTX/SASS不提交到 Git；`summary.json` 保存 hash和证据边界。由于执行主机没有
可用 NVIDIA driver，该 snapshot不是 Thor numerical evidence。
