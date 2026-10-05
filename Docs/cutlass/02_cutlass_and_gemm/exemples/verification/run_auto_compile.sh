#!/usr/bin/env bash
# 在兼容 CUDA 13.0 的 Linux 编译环境中执行；仅输出类型，不启动 GPU Kernel。
set -euo pipefail
: "${CUTLASS_ROOT:?请指定固定版本的 CUTLASS 源码目录}"
VERIFY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AUTO_BUILD_DIR="${AUTO_BUILD_DIR:-$(mktemp -d /tmp/thor-auto-compile.XXXXXX)}"
NVCC_BIN="${NVCC_BIN:-nvcc}"
CUOBJDUMP_BIN="${CUOBJDUMP_BIN:-cuobjdump}"
for spec in 0:dense 1:nvfp4 2:grouped 3:moe 4:attention; do
  case_id="${spec%%:*}"
  case_name="${spec#*:}"
  case_dir="$AUTO_BUILD_DIR/$case_name"
  mkdir -p "$case_dir"
  "$NVCC_BIN" -std=c++17 --expt-relaxed-constexpr \
    -gencode arch=compute_110a,code=sm_110a \
    -I"$CUTLASS_ROOT/include" -I"$CUTLASS_ROOT/tools/util/include" \
    --keep --keep-dir "$case_dir" -Xptxas=-v -DCASE_ID="$case_id" \
    "$VERIFY_DIR/inspect_auto.cu" -o "$case_dir/inspect_auto" \
    > "$case_dir/build.log" 2>&1
  "$case_dir/inspect_auto" > "$case_dir/types.txt"
  "$CUOBJDUMP_BIN" --dump-sass "$case_dir/inspect_auto" > "$case_dir/sass.txt"
  printf '%s: compile/link/type inspection PASS\n' "$case_name"
done
printf '编译产物：%s\n' "$AUTO_BUILD_DIR"
