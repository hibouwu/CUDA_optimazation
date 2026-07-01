#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cmake -S "${ROOT}/src" -B "${ROOT}/build"
cmake --build "${ROOT}/build" --parallel

