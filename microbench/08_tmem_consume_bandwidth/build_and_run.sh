#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CMD="${1:-run}"
case "${CMD}" in
  run)
    "${SCRIPT_DIR}/scripts/tmem_consume_validation.py" --run
    ;;
  validate)
    "${SCRIPT_DIR}/scripts/tmem_consume_validation.py" --validate
    ;;
  ncu)
    "${SCRIPT_DIR}/scripts/tmem_consume_validation.py" --ncu
    ;;
  summarize)
    "${SCRIPT_DIR}/scripts/tmem_consume_validation.py" --summarize
    ;;
  clean)
    rm -rf "${SCRIPT_DIR}/results"
    ;;
  *)
    echo "Usage: $0 run|validate|ncu|summarize|clean"
    exit 2
    ;;
esac
