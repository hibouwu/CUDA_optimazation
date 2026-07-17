#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GEMM_SUITE="${GEMM_SUITE:-core}"
export PRESET="${PRESET:-full}"
export TRIALS="${TRIALS:-10}"
export BACKEND_TIMEOUT_SECONDS="${BACKEND_TIMEOUT_SECONDS:-120}"
export BACKEND_ATTEMPTS="${BACKEND_ATTEMPTS:-3}"
export RESULT_TAG="${RESULT_TAG:-sm110_gemm_core_128_4096_10trials}"

exec bash "${SCRIPT_DIR}/run_gemm_sm110_experiments.sh"
