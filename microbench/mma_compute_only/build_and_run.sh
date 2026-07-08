#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<EOF
Usage:
  $0 run [--iters N]
  $0 ncu
  $0 plot
EOF
}

CMD="${1:-run}"

case "${CMD}" in
  run)
    shift || true
    python3 "${SCRIPT_DIR}/run_thor_tcgen05_report.py" "$@"
    ;;
  ncu)
    shift || true
    "${SCRIPT_DIR}/run_ncu_reports.sh" "$@"
    ;;
  plot)
    shift || true
    python3 "${SCRIPT_DIR}/plot_tcgen05_results.py" "$@"
    ;;
  --help|-h)
    usage
    ;;
  *)
    usage
    exit 1
    ;;
esac
