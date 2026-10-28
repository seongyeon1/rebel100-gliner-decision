#!/usr/bin/env bash
# Keep ATOM MAX Compose resources and compiler caches separate from Rebel100.
set -euo pipefail
cd "$(dirname "$0")"
export COMPOSE_FILE="$PWD/compose.yaml:$PWD/compose.atom-max.yaml"
export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-gliner-atom-max}"
export BACKEND="${BACKEND:-rbln-compile}"
export DTYPE="${DTYPE:-float16}"
exec bash docker_run.sh "${@}"
