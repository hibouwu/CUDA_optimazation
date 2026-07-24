#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CMD="${1:-run}"
shift || true

case "${CMD}" in
  run)
    "${SCRIPT_DIR}/scripts/pipeline_overlap.py" run "$@"
    ;;
  validate)
    "${SCRIPT_DIR}/scripts/pipeline_overlap.py" validate "$@"
    ;;
  ncu)
    "${SCRIPT_DIR}/scripts/pipeline_overlap.py" ncu "$@"
    ;;
  review)
    "${SCRIPT_DIR}/scripts/pipeline_overlap.py" review "$@"
    ;;
  all)
    "${SCRIPT_DIR}/scripts/pipeline_overlap.py" run "$@"
    "${SCRIPT_DIR}/scripts/pipeline_overlap.py" validate "$@"
    "${SCRIPT_DIR}/scripts/pipeline_overlap.py" ncu "$@"
    "${SCRIPT_DIR}/scripts/pipeline_overlap.py" review "$@"
    ;;
  *)
    echo "Usage: $0 run|validate|ncu|review|all"
    exit 2
    ;;
esac
