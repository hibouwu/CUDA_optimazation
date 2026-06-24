#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <backend_id>" >&2
  echo "Backend IDs: cublas, v1, v2, v3, v3a, v3b, v4, v5, v6, v7, v8a, v8b, v8c, cublas_tc, tc1, tc2" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GEMM_SUITE="$1" bash "${SCRIPT_DIR}/run_gemm_experiments.sh"
