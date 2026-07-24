#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CMD="${1:-summarize}"
case "${CMD}" in
  summarize|run|validate|ncu)
    "${SCRIPT_DIR}/scripts/tmem_cp_summary.py"
    ;;
  clean)
    rm -rf "${SCRIPT_DIR}/results"
    ;;
  *)
    echo "Usage: $0 summarize|run|validate|ncu|clean"
    exit 2
    ;;
esac
