# Rebel100 분류 API와 cagent HTTP 연동

## 한 번에 테스트

기존 벤더 SDK가 설치된 환경에서 압축을 풀고 실행합니다.

```bash
# 설치 → 기존 CPU/NPU 정확도·속도 벤치마크 → 서버 시작 → sync/async cagent 호출 → 서버 종료
SERVING_DEMO=1 bash run_all.sh

# HTTP 서빙/클라이언트만 테스트: 설치 → 서버 시작 → 준비 대기 → 예제 호출 → 서버 종료
bash run_serving_demo.sh
```

첫 실행은 고정된 모델 snapshot 다운로드와 컴파일 시간이 필요합니다. HTTP 데모는 최대 1800초까지 준비를 기다리고 15초마다 대기 상태를 출력합니다. 서버가 준비되기 전에 클라이언트 요청을 보내지 않습니다.

모델·정밀도·버킷을 지정할 수 있습니다.

```bash
DTYPE=bfloat16 BUCKET=512 bash run_serving_demo.sh

# 실제 데이터로 벤치마크한 뒤 HTTP 데모까지 실행
SERVING_DEMO=1 BUCKET=512 bash run_all.sh /data/eval.jsonl

# 이미 의존성을 설치했다면 재설치 생략
SKIP_INSTALL=1 bash run_serving_demo.sh

# 반입한 로컬 checkpoint
MODEL=/models/GLiNER2.5-multi-Decide bash run_serving_demo.sh

# NPU 없이 코드 경로를 확인하는 CPU 모드
BACKEND=cpu DTYPE=float32 bash run_serving_demo.sh
```

단독 데모 결과는 `results/http-demo-*/`에, 통합 실행은 `results/run-*/http-demo/`에 저장됩니다.

| 파일 | 내용 |
|---|---|
| `demo-report.json` | 실행 상태, 준비 시간, 실패 사유 |
| `responses.json` | 문서 분류(sync), 경로 선택(async)의 실제 응답 |
| `server.log` | 모델 로딩·컴파일·HTTP 서버 로그 |
| `client.log` | cagent 예제 실행 로그 |

데모는 로컬 loopback의 사용 가능한 포트를 예약한 뒤 자식 서버에 전달합니다. 충돌하는 서버를 재사용하지 않습니다. 임시 인증 키를 자동 생성해 서버와 클라이언트에만 전달하며 키를 결과 파일에 기록하지 않습니다. 완료·실패·시간 초과 시 자신이 시작한 서버만 종료합니다.

**HTTP 데모 성공은 요청이 실제 모델 경로를 통과하고 응답 계약이 맞았다는 의미입니다. 예제 문서/도구 선택의 품질을 보장하지 않습니다.** 정확도·속도·gather 진단은 기존 벤치마크 결과로 판단합니다.

## 서버를 계속 켜 두기

앞선 데모를 실행하면 서빙 의존성이 설치됩니다. 데모는 서버를 자동 종료하므로 실제 연동에는 별도 지속 실행이 필요합니다.

서버의 SDK 환경에서:

```bash
# 서버와 cagent에 같은 키를 설정합니다. 키를 코드에 넣지 마세요.
export DECISION_API_KEY='서버와-cagent에-동일하게-설정할-키'
export BACKEND=rbln-compile
export DTYPE=float16
export BUCKET=256
export HOST=0.0.0.0
export PORT=8009

python server.py
```

위 명령은 foreground에서 서버를 유지합니다. 컨테이너/systemd 등의 운영 supervisor에서는 같은 명령과 환경변수로 단일 프로세스를 실행합니다. `server.py`는 uvicorn worker를 1개로 고정합니다. 여러 worker를 임의로 띄우면 모델과 NPU 메모리가 중복됩니다. 장치별 프로세스와 장치 할당을 분리해서 확장해야 합니다.

서버와 cagent 사이의 내부 통신 경로에서 이 포트에 접근할 수 있어야 합니다. SDK가 컨테이너에 있으면 장치 할당과 서비스 포트 publish도 해당 컨테이너 설정에 반영합니다.

| 환경변수 | 기본값/설명 |
|---|---|
| `DECISION_API_KEY` | 필수 Bearer 인증 키. 미설정 시 서버 시작 실패 |
| `HOST`, `PORT` | `127.0.0.1`, `8009`; 원격 호출에는 접근 가능한 bind/port 설정 |
| `BACKEND` | `rbln-compile`; `rbln-eager`, `cpu`도 지원 |
| `DTYPE` | `float16`; CPU 확인 시 `float32` 권장 |
| `MODEL`, `REVISION` | 패키지에 고정한 HF 모델/revision 또는 로컬 checkpoint |
| `BUCKET` | 256; 본문+schema 전체 subword 길이 |
| `NPU_DEVICE` | 0 |
| `CPU_THREADS` | 4 |
| `MAX_PENDING` | 8; 실행 중 1건 외 대기열 상한 |
| `SERVE_CACHE_DIR` | `results/serve-cache`; 컴파일 캐시 |

## HTTP 계약

- `GET /healthz`: 프로세스 생존 여부. 모델 준비와 무관하게 200.
- `GET /readyz`: 로딩·컴파일·워밍업 후 200. 준비 전/초기화 실패는 503. 준비 후 backend, bucket, 대기 요청 수를 반환.
- `POST /v1/classify`: Bearer 인증 필요. 단일/다중 라벨 및 여러 task를 지원.
- 401: 인증 실패. 422: 요청/schema/버킷 초과. 429: 대기열 가득 참. 503: 모델 준비 전 또는 추론 실패.

```bash
curl -fsS http://127.0.0.1:8009/readyz

curl -fsS http://127.0.0.1:8009/v1/classify \
  -H "Authorization: Bearer $DECISION_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"text":"회의 참석자와 결정 사항을 정리했다","tasks":{"document_type":["계약서","회의록","보고서","기타"]}}'
```

응답에는 `request_id`, `predictions`, 라벨별 `probabilities`, `model_revision`, `backend`, `tokens`, `elapsed_ms`가 포함됩니다. `elapsed_ms`는 대기열·전처리·추론·결과 디코딩을 포함하며 순수 NPU kernel 시간이 아닙니다. 확률은 보정된 신뢰도를 의미하지 않습니다.

Task 상세 형식:

```json
{
  "text": "사내 계약서의 해지 조건을 찾아줘",
  "tasks": {
    "route": {
      "labels": ["document_search", "sql_query", "clarify"],
      "label_descriptions": {
        "document_search": "사내 문서 내용 검색",
        "sql_query": "정형 데이터베이스 조회 및 집계",
        "clarify": "필수 정보 추가 확인"
      }
    }
  }
}
```

요청당 task 최대 16개, task당 중복 없는 label 최대 64개입니다. 이 상한 내에서도 전체 입력은 선택한 bucket 안에 들어가야 합니다. 자동 truncation은 수행하지 않습니다. 이 HTTP 계약은 분류 전용이며 span/entity/relation extraction 옵션이나 few-shot examples 등은 받지 않습니다.

## cagent 호스트에 클라이언트 설치

**NPU SDK나 PyTorch가 없는 cagent 환경에서도 설치 가능합니다.** 패키지의 `wheels/cagent_decision_client-0.1.0-py3-none-any.whl`을 cagent 호스트에 복사합니다.

```bash
python -m pip install /path/to/cagent_decision_client-0.1.0-py3-none-any.whl
```

의존성은 `httpx`, `pydantic`뿐입니다. 서버 코드나 torch/GLiNER 의존성이 wheel에 포함되지 않습니다. 외부망이 없으면 이 두 의존성도 함께 반입해야 합니다. wheel 대신 압축 해제한 폴더를 `pip install /path/to/rebel100-port`로 설치해도 HTTP 클라이언트 패키지만 설치됩니다.

### 문서 워커에서 동기 호출

```python
import os
from cagent_decision import DecisionClient, DecisionUnavailable

client = DecisionClient(
    base_url=os.environ["DECISION_API_URL"],
    api_key=os.environ["DECISION_API_KEY"],
    timeout=10.0,
)

# 워커 프로세스마다 client 하나를 재사용하고 종료할 때 client.close() 호출.
# 문서 유형을 사용자가 지정했다면 이 호출로 덮어쓰지 않음.
try:
    result = client.classify(
        text="회의 참석자와 안건 및 결정 사항을 정리한 문서입니다.",
        tasks={"document_type": ["계약서", "회의록", "보고서", "기타"]},
    )
    suggested_type = result.predictions["document_type"]
except DecisionUnavailable:
    # 기존 분류/검수 경로로 보내는 cagent 업무 로직을 연결.
    suggested_type = None
```

### 에이전트에서 비동기 호출

```python
import os
from cagent_decision import AsyncDecisionClient


async def select_route(text: str):
    # 실제 서비스에서는 앱 lifespan 동안 connection pool을 재사용.
    async with AsyncDecisionClient(
        os.environ["DECISION_API_URL"], os.environ["DECISION_API_KEY"]
    ) as client:
        result = await client.classify(
            text,
            {
                "route": {
                    "labels": ["document_search", "sql_query", "clarify"],
                    "label_descriptions": {
                        "document_search": "사내 문서 검색",
                        "sql_query": "내부 DB 조회",
                        "clarify": "필수 정보 확인",
                    },
                },
            },
        )
        return result.predictions["route"]
```

도구 후보는 요청자의 권한과 현재 실행 가능한 도구로 미리 제한합니다. 클라이언트는 응답 task·라벨·단일/다중 형태·확률을 다시 검증합니다. 요청에 없던 도구를 응답받으면 `DecisionProtocolError`를 발생시키며 도구를 실행하지 않습니다. 도구 인자 생성과 실행은 기존 에이전트의 책임입니다.

- `DecisionUnavailable`: 연결 오류·타임아웃·429·503 등. 기존 판단 경로로 보낼지 호출자가 결정.
- `DecisionRequestError`: 인증 오류·잘못된 요청 등. 설정/호출 코드를 수정할 오류.
- `DecisionProtocolError`: 잘못된 응답 또는 허용하지 않은 task/label. 자동 실행 금지.
- 요청 schema가 잘못되면 전송 전에 Pydantic validation error 발생.

연결 풀에는 요청마다 새 키나 전역 상태를 섞지 않습니다. 클라이언트는 자동 재시도를 하지 않습니다. 타임아웃이 나도 서버의 이미 시작된 NPU 작업을 즉시 중단한 것으로 간주하지 않습니다.

## 원격 서버에 실행 예제 요청

서버가 계속 켜진 상태에서 패키지의 예제를 사용할 수 있습니다.

```bash
export DECISION_API_URL=http://REBEL100_HOST:8009
export DECISION_API_KEY='서버에-설정한-동일한-키'
python cagent_example.py --output /tmp/cagent-decision-responses.json
```

`cagent_example.py`는 동기 문서 분류와 비동기 도구/경로 선택을 모두 실행합니다. 실제 도구 실행이나 cagent DB 갱신은 하지 않습니다.

## 검증 범위

로컬의 작은 실제 DeBERTa/GLiNER checkpoint로 public API와 분류 경로 일치, 인증, 입력 검증, overflow 422, 초기화 실패 503, 대기열 초과 429, 응답 라벨 검증, 실제 uvicorn TCP 서버와 동기/비동기 클라이언트, 벤치마크+HTTP 통합 실행을 확인했습니다.

**실제 287M 모델의 HTTP 서빙, Rebel100 컴파일/가속, 운영 데이터 정확도, ATOM MAX 호환성은 대상 환경에서 검증해야 합니다.** cagent 저장소 자체는 변경하지 않았으며 HTTP 클라이언트와 통합 예제를 패키지에 제공합니다.
