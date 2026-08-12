#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build_dir="$script_dir/build"
result_dir="$script_dir/results"
binary="$build_dir/tmem_readback_bandwidth"

mkdir -p -- "$build_dir" "$result_dir"
extra_flags=()
if [[ ${NVCC_HOST_UNDEF_GNU_SOURCE:-0} == 1 ]]; then
  extra_flags+=(-Xcompiler=-U_GNU_SOURCE)
fi
nvcc -O3 -std=c++17 -gencode arch=compute_110a,code=sm_110a \
  "${extra_flags[@]}" \
  "$script_dir/tmem_readback_bandwidth.cu" -o "$binary"
cuobjdump --dump-sass "$binary" > "$result_dir/tmem_readback_bandwidth.sass"

if [[ ${1:-build-only} == build-only ]]; then
  echo "built $binary"
  exit 0
fi
if [[ ${1:-} != run ]]; then
  echo "usage: $0 build-only|run" >&2
  exit 2
fi

: > "$result_dir/raw.txt"
for registers in 8 16; do
  for warps in 1 4; do
    for trial in $(seq 0 9); do
      printf 'trial=%s\n' "$trial" >> "$result_dir/raw.txt"
      "$binary" --registers "$registers" --warps "$warps" --iters 10000 \
        >> "$result_dir/raw.txt"
    done
  done
done
echo "wrote $result_dir/raw.txt"
