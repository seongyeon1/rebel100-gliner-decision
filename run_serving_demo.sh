#!/usr/bin/env bash
# Install app dependencies with SDK pins, then run a real HTTP round trip.
set -euo pipefail
cd "$(dirname "$0")"
python_cmd="${PYTHON:-python}"
constraints="$(mktemp "${TMPDIR:-/tmp}/decision-sdk-XXXXXX")"
trap 'rm -f "$constraints"' EXIT
"$python_cmd" - <<'PY' > "$constraints"
import os
from importlib.metadata import version
names = ('torch',) if os.environ.get('BACKEND') == 'cpu' else ('torch', 'torch-rbln', 'rebel-compiler')
for name in names:
    print(f'{name}=={version(name)}')
PY
if [[ "${SKIP_INSTALL:-0}" != 1 ]]; then
  "$python_cmd" -m pip install -c "$constraints" -r requirements-serving.txt
fi
"$python_cmd" serve_demo.py "$@"
