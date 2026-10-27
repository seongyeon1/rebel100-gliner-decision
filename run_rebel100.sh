#!/usr/bin/env bash
# Run in an activated vendor SDK environment; preserve all native stderr logs.
set -euo pipefail
cd "$(dirname "$0")"
output="${1:-results/rebel100-$(date +%Y%m%d-%H%M%S)}"
data="${2:-examples.jsonl}"
mkdir -p "$output"
python benchmark.py --data "$data" --backend rbln-compile --dtype float16 \
  --bucket "${BUCKET:-256}" --cpu-threads "${CPU_THREADS:-4}" \
  --output "$output/compiled" 2> >(tee "$output/process.stderr.log" >&2)
