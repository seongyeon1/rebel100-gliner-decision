#!/usr/bin/env bash
set -euo pipefail
cd /app
mode="${1:-serve}"
if (( $# )); then shift; fi
case "$mode" in
  check|serve|benchmark|demo) "$PYTHON" docker/container_check.py ;;
  *) exec "$mode" "$@" ;;
esac
case "$mode" in
  check) exit 0 ;;
  serve) exec "$PYTHON" server.py "$@" ;;
  benchmark)
    if (( $# == 0 )); then set -- "${DATASET:-/app/examples.jsonl}"; fi
    exec bash run_all.sh "$@" ;;
  demo) exec bash run_serving_demo.sh "$@" ;;
esac
