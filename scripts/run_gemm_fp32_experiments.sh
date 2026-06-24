#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GEMM_SUITE=fp32 bash "${SCRIPT_DIR}/run_gemm_experiments.sh"
