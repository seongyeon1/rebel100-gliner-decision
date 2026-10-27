#!/bin/sh
set -eu
exec "$PYTHON" -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8009/readyz", timeout=2).read()'
