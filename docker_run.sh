#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mode="${1:-all}"
if [[ "$mode" == --help ]]; then
  echo 'Usage: bash docker_run.sh [sdk|setup|check|build|all|demo|serve|client|logs|down]'
  echo 'Auto-detects an SDK image, or builds one from pinned SDK packages if missing.'
  exit 0
fi
case "$mode" in sdk|setup|check|build|all|demo|serve|client|logs|down) ;; *) echo 'Unknown mode' >&2; exit 2 ;; esac
if [[ "$mode" == sdk ]]; then exec python3 sdk_bootstrap.py; fi
python3 docker_setup.py
case "$mode" in
  setup) exit 0 ;;
  logs) exec docker compose logs -f api ;;
  down) exec docker compose down ;;
  client)
    docker compose --profile client build client
    exec docker compose --profile client run --rm client ;;
esac
if [[ "$mode" == all || "$mode" == demo ]]; then
  if [[ "$(docker compose ps --status running --services)" == *api* ]]; then
    echo 'API is already running on this project. Stop it explicitly before benchmarking this NPU.' >&2
    exit 1
  fi
fi
docker compose build api
case "$mode" in
  build) exit 0 ;;
  check) exec docker compose run --rm --no-deps api check ;;
  all) exec docker compose --profile tools run --rm --no-deps benchmark ;;
  demo) exec docker compose --profile tools run --rm --no-deps demo ;;
  serve) exec docker compose up -d --no-build --wait --wait-timeout "${STARTUP_TIMEOUT:-1800}" api ;;
esac
