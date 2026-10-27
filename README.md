# Rebel100 GLiNER Decision Service

GLiNER2.5-multi-Decide의 분류 전용 포팅 실험, CPU/NPU 벤치마크, HTTP 서빙 API와 SDK 없는 cagent 클라이언트입니다.

**실험 단계:** 작은 실제 DeBERTa/GLiNER 모델을 이용한 로컬 CPU·HTTP 통합 테스트 22개를 통과했습니다. 실제 287M 체크포인트의 Rebel100 컴파일·실행·가속은 아직 확인하지 않았습니다. 기존 리벨리온 SDK 환경이 필요합니다.

```bash
gh repo clone seongyeon1/rebel100-gliner-decision
cd rebel100-gliner-decision
SERVING_DEMO=1 bash run_all.sh
```

서빙 API와 클라이언트 사용법은 [SERVING.md](SERVING.md)를 참고하세요. 서버 반입용 tar.gz와 cagent 클라이언트 wheel은 GitHub Releases에 제공합니다. 이 저장소에는 모델 가중치가 포함되어 있지 않으며, 최초 실행 시 설정된 Hugging Face snapshot을 내려받습니다.

---

# GLiNER2.5-multi-Decide → Rebel100 분류 포팅 실험

문서 분류와 에이전트 도구/경로 선택을 위한 **분류 전용 실행 코드**와 벤치마크입니다. CPU 원본 API → CPU 분류 전용 경로 → RBLN 후보를 같은 요청으로 비교합니다.

**구현은 로컬의 작은 실제 DeBERTa/GLiNER 모델로 검증했습니다. 287M 실제 체크포인트의 전체 추론과 Rebel100 컴파일·실행은 이 개발 환경에서 수행하지 않았습니다.** SDK 버전별 지원 차이에 따라 컴파일 실패나 폴백이 발생할 수 있으며, 성공 수치를 만들어 넣지 않습니다.

## 서빙 API와 cagent 연동

[서버 실행·cagent 호출 안내](SERVING.md)를 추가했습니다.

```bash
# 설치 + CPU/NPU 벤치마크 + HTTP 서버 시작 + cagent sync/async 호출 + 자동 종료
SERVING_DEMO=1 bash run_all.sh

# HTTP 서빙/클라이언트 테스트만 수행
bash run_serving_demo.sh
```

별도 cagent 호스트에는 `wheels/cagent_decision_client-0.1.0-py3-none-any.whl`만 설치하면 됩니다. PyTorch/NPU SDK가 필요하지 않습니다. 지속 서빙은 `DECISION_API_KEY`를 설정한 뒤 `python server.py`로 실행합니다. 자세한 설정과 예제는 위 안내를 참고하세요.

## 한 번에 설치·실행·요약

벤더 SDK 환경이 활성화된 상태에서 다음 한 줄로 실행합니다. 벤더 SDK/driver 자체를 설치하는 스크립트는 아닙니다.

```bash
bash run_all.sh
```

수행 순서: SDK/NPU 확인 → 현재 SDK 버전 constraints 생성 → 애플리케이션 의존성 설치 → CPU/NPU 벤치마크 → 터미널 요약 및 `summary.txt` 저장.

기본 출력은 충돌하지 않는 `results/run-날짜-시각-XXXXXX/`입니다. `workflow.log`, `sdk-constraints.txt`, `environment.txt`, `summary.txt`, `benchmark/`가 생성됩니다. 설치/사전 확인 실패도 루트 `summary.txt`에 실패 단계와 로그 위치를 남깁니다. 벤치마크 실패 시에도 오류 요약을 생성하고 비정상 종료합니다.

```bash
# 실제 데이터와 새 결과 디렉터리 지정. 상대 경로는 명령을 실행한 디렉터리 기준입니다.
BUCKET=512 CPU_THREADS=4 bash run_all.sh /data/eval.jsonl /data/results/rebel100-run01

# 설치를 마친 환경에서 재측정
SKIP_INSTALL=1 REPEATS=50 bash run_all.sh /data/eval.jsonl

# BF16, eager 또는 로컬 체크포인트
DTYPE=bfloat16 bash run_all.sh
BACKEND=rbln-eager bash run_all.sh
MODEL=/models/GLiNER2.5-multi-Decide bash run_all.sh

# 결과 요약만 다시 표시
python summarize.py results/run-실제디렉터리/benchmark
```

### 서버 업로드

Mac 터미널에서 `USER`, `SERVER`를 실제 SSH 계정과 주소로 바꿉니다. SSH config에 별칭이 있으면 `USER@SERVER` 대신 그 별칭을 쓸 수 있습니다.

```bash
scp ./rebel100-port.tar.gz USER@SERVER:~/rebel100-port.tar.gz
ssh USER@SERVER
```

이후 서버의 벤더 SDK Python 환경에서:

```bash
run_package_dir="$(mktemp -d "$HOME/gliner-package-XXXXXX")"
tar -xzf "$HOME/rebel100-port.tar.gz" -C "$run_package_dir"
cd "$run_package_dir/rebel100-port"
bash run_all.sh
```

SSH 포트가 다르면 `scp -P PORT ...`, `ssh -p PORT ...`를 사용합니다. SDK가 컨테이너 안에 있으면 파일을 컨테이너로 복사하거나 마운트하고, 그 안에서 압축 해제와 실행을 진행합니다. 예: 호스트에서 `docker cp ~/rebel100-port.tar.gz SDK_CONTAINER:/tmp/rebel100-port.tar.gz`를 실행한 뒤 컨테이너 내부에서 해당 `/tmp/` 파일을 사용합니다. 실제 업로드나 원격 실행은 이 개발 환경에서 수행하지 않았습니다.

## 1. 설치

Rebel100의 driver와 `rebel-compiler`, `torch-rbln`, 대응 PyTorch가 설치된 **벤더 SDK 환경/컨테이너**에서 실행합니다. Python 3.11/3.12를 권장합니다. `rbln-smi`로 장비를 먼저 확인합니다.

이 폴더를 서버에 복사한 뒤 해당 SDK 환경에서:

```bash
cd rebel100-port

# SDK가 설치한 버전을 고정하여 pip가 torch 등을 교체하지 못하게 합니다.
python - <<'PY' > sdk-constraints.txt
from importlib.metadata import version
for name in ('torch', 'torch-rbln', 'rebel-compiler'):
    print(f'{name}=={version(name)}')
PY
python -m pip install -c sdk-constraints.txt -r requirements.txt
```

버전 충돌로 설치가 실패하면 constraints를 삭제해 우회하지 말고 벤더 SDK와 Transformers 호환 조합을 확인합니다. 이 패키지는 vendor driver/SDK를 설치하거나 업그레이드하지 않습니다. `git`이 필요합니다. 인터넷이 없는 서버는 필요한 wheels와 아래 고정 모델 snapshot을 외부에서 준비해 가져옵니다.

고정 버전:

- GLiNER2: `ca4f68235b50fe4c3ecf3876851d748fe4b15c6c`
- Transformers: `5.17.0`
- HF model: `fastino/GLiNER2.5-multi-Decide`
- HF revision: `e173bca1f0c4217d7a55b3b0c705d5ece8a0218d`

첫 실행에는 전체 checkpoint 약 1.15 GB와 tokenizer가 다운로드됩니다. CPU 기준과 장치 모델을 위한 호스트 메모리, 컴파일 workspace도 필요합니다. 원본 모델은 CPU 평가를 마친 뒤 encoder/classifier를 장치로 이동하므로 한 프로세스 안에서 단계를 순차 수행합니다.

## 2. 바로 실행

```bash
# 기본: FP16, B=1, 총 길이 256, CPU 4 threads, warmup 3, repeats 20
./run_rebel100.sh results/rebel100-256 examples.jsonl
```

기본 결과 디렉터리를 생략하면 현재 시각으로 새 디렉터리를 만듭니다. 기존 `report.json`을 덮어쓰지 않습니다. 예제 6건의 정답은 실행 확인용 사람이 지정한 smoke labels이며, 운영 정확도를 대표하는 평가 세트가 아닙니다.

세부 옵션:

```bash
python benchmark.py --backend rbln-compile --dtype bfloat16 \
  --data my-eval.jsonl --bucket 512 --cpu-threads 4 \
  --warmup 5 --repeats 50 --output results/rebel100-bf16-512

# Eager 실행과 컴파일 실행은 결과를 분리해 비교합니다.
python benchmark.py --backend rbln-eager --dtype float16 \
  --data my-eval.jsonl --bucket 256 --output results/rebel100-eager-256

# NPU 없는 환경에서 CPU 코드와 데이터 형식만 확인
python benchmark.py --backend cpu --dtype float32 \
  --data examples.jsonl --bucket 256 --output results/cpu-smoke

# 반입한 로컬 체크포인트 사용: config, encoder_config, tokenizer, weights 모두 필요
python benchmark.py --model /models/GLiNER2.5-multi-Decide \
  --data my-eval.jsonl --output results/local-model
```

`--bucket`은 본문+라벨/도구 설명을 포함한 총 subword 길이입니다. 초과 요청은 오류로 중단합니다. 자동 잘라내기로 정확도·속도를 부풀리지 않습니다. 128/256/512는 각각 다른 출력 디렉터리에서 실행합니다. 문서가 길면 상위 코드에서 명시적으로 chunking하고 그 평가 규칙을 고정해야 합니다.

컴파일은 `torch.compile(backend='rbln', fullgraph=True, dynamic=False, options={'mode':'strict', ...})`로 수행합니다. 컴파일 실패를 eager 성공으로 숨기지 않습니다. 실패 시 `failure.json`, CPU까지의 부분 `report.json`, stderr를 확인합니다. `run_rebel100.sh`는 실패 시 비정상 종료합니다. Eager 비교는 위 별도 명령으로 실행합니다.

ATOM MAX에서는 해당 장비용 SDK 환경에서 재컴파일·재측정합니다. Rebel100 아티팩트/수치의 호환성을 가정하지 않습니다.

## 3. 평가 데이터

각 줄이 한 요청인 UTF-8 JSONL입니다. 요청마다 라벨 집합을 바꿀 수 있습니다.

```json
{"id":"route-1","text":"지난달 주문별 매출을 조회해줘","tasks":{"route":{"labels":["sql","search","clarify"],"label_descriptions":{"sql":"정형 DB 집계","search":"문서 내용 검색","clarify":"사용자에게 필수 정보 확인"}}},"expected":{"route":"sql"}}
{"id":"doc-1","text":"보안 정책과 계약 갱신을 논의했다","tasks":{"topics":{"labels":["보안","계약","채용"],"multi_label":true,"cls_threshold":0.5}},"expected":{"topics":["보안","계약"]}}
```

- `expected`는 선택 사항입니다. 없으면 `accuracy=null`이며 CPU 일치율만 보고합니다.
- Accuracy는 정답이 있는 task별 exact match입니다. 다중 라벨은 순서와 관계없이 집합 전체가 맞아야 정답입니다. task별 정확도와 정답 decision 수를 같이 기록합니다.
- CPU agreement는 NPU와 CPU 원본의 일치율입니다. 실제 정답 정확도와 다른 지표입니다.
- `tasks` 형식은 원본 `classify_text`와 같습니다. 단일 라벨, 다중 라벨, 여러 task, label descriptions를 지원합니다. entity/span/relation extraction은 대상이 아닙니다.
- 실제 라우팅 평가에는 도구를 잘못 고르는 요청, 모호한 요청, 해당 도구가 없는 요청을 넣고 `clarify`/`other` 같은 선택지를 명시하는 것이 좋습니다.

## 4. 결과 읽기

`report.json`:

| 필드 | 의미 |
|---|---|
| `cpu_original.quality` | 원본 CPU API의 정답 정확도 |
| `cpu_port.quality` | 분류 전용 CPU 경로의 정확도·원본 일치율 |
| `cpu_port.max_selected_probability_error` | 원본이 선택한 라벨의 확률과 CPU 포팅 확률 오차 |
| `candidate.quality` | 후보 장치의 정확도·CPU 원본 일치율 |
| `candidate.marker_logit_error` | CPU FP32 포팅 대비 실제 label marker logit 절대오차 |
| `*.end_to_end.aggregate` | 전처리·전송·forward·CPU 디코딩 포함 p50/p95/평균과 serial req/s |
| `candidate.prepared_request` | 전처리 제외, 전송·forward·CPU 디코딩 포함. 순수 NPU kernel 시간이 아님 |
| `candidate.speedup_vs_cpu_port` | 같은 고정 길이 포팅 CPU 경로 대비 end-to-end 평균 속도비 |
| `candidate.speedup_vs_cpu_original` | 원본 API CPU 대비 end-to-end 평균 속도비 |
| `candidate.first_forward_including_compile_seconds` | 첫 forward 비용. 이후 steady-state 지표에서 제외 |
| `environment` | Python·torch·GLiNER·Transformers·SDK 버전 및 CPU threads |
| `tokens_per_request` | padding 전 실제 총 토큰 수 |

`predictions.json`에는 요청별 정답, CPU/NPU 예측, 라벨별 확률을 기록합니다. 로딩·모델 전송·컴파일 시간은 steady-state 속도에 포함하지 않습니다. 모든 요청을 워밍업하고 동기화해 측정합니다. `req/s`는 batch=1 순차 처리량이며 동시 요청 서버 처리량이 아닙니다. 원본 CPU API는 가변 길이, CPU/NPU 포팅은 고정 길이이므로 두 CPU 기준을 모두 제공합니다.

공정한 운영 비교를 위해 CPU thread 수·CPU affinity·동시 부하를 고정하고, 작은 예제 대신 실제 분포의 충분한 평가 요청으로 반복하십시오. FP16/BF16 수치 차이와 라벨 일치율을 함께 봐야 합니다.

## 5. gather가 CPU로 폴백하는지 확인

기본 실행은 성능 측정을 끝낸 후 별도 진단을 수행합니다. `--no-diagnostics`로 생략할 수 있습니다.

생성 파일:

- `first-forward.stderr.log`: 첫 forward/컴파일의 native SDK 로그.
- `model-forward.stderr.log`: 실제 모델 실행 중 native/Python stderr.
- `model-forward.host-trace.json`: Chrome/Perfetto에서 볼 수 있는 CPU dispatcher trace.
- `model-forward.host-operators.txt`: CPU 측 operator 이벤트 요약.
- `gather-eager-probe.stderr.log`, `.host-trace.json`, `.host-operators.txt`: 상대위치 attention과 같은 gather 형태 `[12,L,512] → [12,L,L]`를 eager에서 독립 실행한 진단.
- `report.json → diagnostics`: 제공되는 경우 `torch.rbln.explain()`의 report/verdict/dump 원문.
- `compile-cache/`: backend가 저장하는 컴파일 아티팩트.
- `process.stderr.log`: 셸 실행 스크립트로 돌린 전체 프로세스 stderr.

자동 판정은 gather 이름과 **명시적인 CPU fallback 메시지**가 같은 로그 줄에 있을 때만 `confirmed`입니다. 그 외에는 `unknown`입니다. SDK별 로그 문구가 다르면 메시지를 자동 인식하지 못할 수 있으므로 원문도 확인합니다. `unknown`은 NPU 실행 확인을 뜻하지 않습니다.

반드시 구분할 사항:

1. **모델 진단**은 실제 모델 forward의 증거입니다. CPU에서 의도적으로 수행하는 입력 전송·결과 디코딩도 explain에 나타날 수 있습니다.
2. **gather eager probe**는 SDK eager gather의 배치 위치를 분리해 보는 보조 진단입니다. 컴파일된 graph 내부 gather의 실행 위치를 대신 증명하지 않습니다.
3. CPU profiler에 `aten::gather` 이벤트가 있다는 것만으로 CPU fallback이라고 단정할 수 없습니다. NPU dispatcher 호출도 CPU trace에 보일 수 있습니다.
4. 컴파일된 graph 내부 배치는 opaque할 수 있습니다. 로그/설명으로 확인되지 않으면 compiler profiler와 vendor 지원으로 실행 위치를 추가 확인해야 합니다.
5. `TORCH_RBLN_DISABLE_FALLBACK=unsupported_op`는 CPU-pinned 연산까지 전부 막지 않습니다. 이 설정으로 오류가 없다는 이유만으로 gather가 NPU에서 실행된다고 판정하지 않습니다.

공식 문서상 eager gather는 CPU pinned이며, compiler 표에서도 gather는 supported / non-accelerated입니다. 따라서 이 스크립트는 실제 환경의 증거를 수집하며, 가속이 된다는 전제로 작성하지 않았습니다.

## 6. 구현과 검증 범위

`port.py`의 NPU 경계는 `encoder → 모든 토큰에 classifier 적용 → [B,L] logits`입니다. 라벨 marker 선택과 task별 softmax/sigmoid는 CPU에서 수행합니다. eval 모드의 token별 MLP이므로 marker를 고른 다음 MLP를 수행하는 원본과 동일한 계산을 목표로 합니다. **encoder 내부 상대위치 gather는 변경하지 않았습니다.**

코드는 CPU에서 원본 API와 포팅 경로의 라벨 및 선택 확률을 먼저 비교하고 불일치하면 장치 벤치마크를 중단합니다. 체크포인트/원본 processor를 그대로 사용하며, 표준 DeBERTa attention으로 로딩합니다. 모델이 NaN/Inf를 반환하면 오류 처리합니다.

로컬 검증:

```bash
python -m pip install -c sdk-constraints.txt -r requirements-dev.txt
python -m pytest -q
ruff check .
ruff format --check .
```

테스트는 다운로드 없이 작은 DeBERTa 상대위치 attention(c2p+p2c), GLiNER processor와 public API로 다음을 확인합니다: 단일/다중 라벨·복수 task·설명 포함 분류의 일치, boundary head 미실행, padding 불변성, 입력 초과 거부, 정확도/일치율 분리, native 로그 판정의 보수성, 로컬 저장 checkpoint를 통한 CLI와 profiler 결과 파일 생성.

실제 287M 모델의 품질, RBLN 컴파일 성공, NPU 속도 및 ATOM MAX 호환성은 대상 서버에서 확인할 항목입니다.
