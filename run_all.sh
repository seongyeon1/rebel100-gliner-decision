#!/usr/bin/env bash
# Run inside an existing vendor SDK environment; install app deps, measure, summarize.
set -Eeuo pipefail
if [[ "${1:-}" == '--help' ]]; then
  cat <<'HELP'
Usage: bash run_all.sh [DATA_JSONL] [OUTPUT_DIR]
Defaults: bundled examples.jsonl; unique results/run-* directory.
Environment: PYTHON=python BACKEND=rbln-compile DTYPE=float16 BUCKET=256
             CPU_THREADS=4 WARMUP=3 REPEATS=20 SKIP_INSTALL=0 MODEL=<local-or-HF-model>
             SERVING_DEMO=1: also start an HTTP server and run cagent client examples
Requires vendor SDK/driver already installed. CPU-only: BACKEND=cpu DTYPE=float32.
HELP
  exit 0
fi
if (( $# > 2 )); then
  echo 'Usage: bash run_all.sh [DATA_JSONL] [OUTPUT_DIR]' >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_cmd="${PYTHON:-python}"
command -v "$python_cmd" >/dev/null
backend="${BACKEND:-rbln-compile}"
case "$backend" in cpu|rbln-compile|rbln-eager) ;; *) echo 'Invalid BACKEND' >&2; exit 2 ;; esac
data_path="${1:-$script_dir/examples.jsonl}"
[[ "$data_path" == /* ]] || data_path="$PWD/$data_path"
[[ -f "$data_path" ]] || { echo "Dataset not found: $data_path" >&2; exit 2; }
if [[ -n "${2:-}" ]]; then
  output_path="$2"
  [[ "$output_path" == /* ]] || output_path="$PWD/$output_path"
  [[ ! -e "$output_path" ]] || { echo "Use a new output directory: $output_path" >&2; exit 2; }
  mkdir -p "$output_path"
else
  mkdir -p "$script_dir/results"
  output_path="$(mktemp -d "$script_dir/results/run-$(date +%Y%m%d-%H%M%S)-XXXXXX")"
fi
cd "$script_dir"
workflow() {
stage='preflight'
on_error() {
  code=$?
  trap - ERR
  printf '실패 단계: %s\n로그: %s/workflow.log\n' "$stage" "$output_path"
  printf '실패 단계: %s\n종료 코드: %s\n로그: %s/workflow.log\n' \
    "$stage" "$code" "$output_path" > "$output_path/summary.txt"
  exit "$code"
}
trap on_error ERR
printf 'Python: %s\nBackend: %s\n평가 데이터: %s\n결과 경로: %s\n' \
  "$python_cmd" "$backend" "$data_path" "$output_path"
if [[ "$backend" != cpu ]]; then
  command -v rbln-smi >/dev/null && rbln-smi
  "$python_cmd" - <<'PY'
import torch
import rebel
import torch_rbln
from importlib.metadata import version
for name in ('torch', 'torch-rbln', 'rebel-compiler'):
    print(f'{name}: {version(name)}')
if not torch.rbln.is_available():
    raise RuntimeError('NPU 접근 불가: 벤더 SDK 환경과 장비 할당을 확인하세요.')
print('NPU:', torch.rbln.get_device_name(0))
PY
fi
stage='install'
BACKEND="$backend" "$python_cmd" - <<'PY' > "$output_path/sdk-constraints.txt"
import os
from importlib.metadata import version
names = ('torch',) if os.environ['BACKEND'] == 'cpu' else ('torch', 'torch-rbln', 'rebel-compiler')
for name in names:
    print(f'{name}=={version(name)}')
PY
if [[ "${SKIP_INSTALL:-0}" != 1 ]]; then
  requirements_file=requirements.txt
  if [[ "${SERVING_DEMO:-0}" == 1 ]]; then requirements_file=requirements-serving.txt; fi
  "$python_cmd" -m pip install -c "$output_path/sdk-constraints.txt" -r "$requirements_file"
else
  echo 'SKIP_INSTALL=1: 의존성 설치 생략'
fi
"$python_cmd" - <<'PYENV' > "$output_path/environment.txt"
from importlib.metadata import distributions
for name, version in sorted((d.metadata['Name'], d.version) for d in distributions() if d.metadata['Name']):
    print(f'{name}=={version}')
PYENV
stage='benchmark'
model_args=(--model "${MODEL:-fastino/GLiNER2.5-multi-Decide}"
            --revision "${REVISION:-e173bca1f0c4217d7a55b3b0c705d5ece8a0218d}")
# Keep diagnostics and summary even when compilation fails.
if "$python_cmd" benchmark.py --data "$data_path" --output "$output_path/benchmark" \
    --backend "$backend" --dtype "${DTYPE:-float16}" --bucket "${BUCKET:-256}" \
    --cpu-threads "${CPU_THREADS:-4}" --warmup "${WARMUP:-3}" --repeats "${REPEATS:-20}" "${model_args[@]}"; then
  benchmark_status=0
else
  benchmark_status=$?
fi
stage='summary'
if "$python_cmd" summarize.py "$output_path/benchmark"; then
  summary_status=0
else
  summary_status=$?
fi
cp "$output_path/benchmark/summary.txt" "$output_path/summary.txt"
printf '\n전체 로그: %s/workflow.log\n요약: %s/summary.txt\n' "$output_path" "$output_path"
if [[ "${SERVING_DEMO:-0}" == 1 ]] && (( benchmark_status == 0 && summary_status == 0 )); then
  stage='serving_demo'
  BACKEND="$backend" "$python_cmd" serve_demo.py --output "$output_path/http-demo"
  printf '
HTTP 서버/cagent 호출 테스트 완료: %s/http-demo/demo-report.json
' "$output_path" >> "$output_path/summary.txt"
fi
trap - ERR
if (( benchmark_status != 0 )); then exit "$benchmark_status"; fi
exit "$summary_status"

}
workflow 2>&1 | tee "$output_path/workflow.log"
