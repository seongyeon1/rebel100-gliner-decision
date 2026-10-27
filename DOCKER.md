# Rebel100 Docker 실행

기존 호스트 driver와 RBLN Container Toolkit(CDI), driver와 맞는 **SDK 이미지**를 사용합니다. Dockerfile은 애플리케이션 의존성을 추가하고 기존 `torch`, `torch-rbln`, `rebel-compiler` 버전을 고정합니다. host driver 설치나 Docker daemon 재시작은 스크립트가 수행하지 않습니다.

## 이미지 이름을 몰라도 시작

Rebel100 **호스트**에서 압축을 풀거나 GitHub 저장소를 받은 뒤:

```bash
# SDK 이미지 탐색 → 앱 이미지 빌드 → CPU/NPU 벤치마크 → HTTP/cagent 테스트
bash docker_run.sh all
```

스크립트는 실행 중인 컨테이너의 이미지와 로컬 Rebellions/SDK 태그를 살펴보고, 임시 컨테이너에서 Python SDK 패키지 메타데이터를 확인합니다. 이 탐색에는 NPU를 사용하지 않습니다. SDK 이미지가 하나로 확정되면 `.env`에 기록하고, API 키를 생성합니다. 키는 화면에 출력하지 않습니다. 여러 태그가 같은 image ID이면 하나로 취급합니다. 서로 다른 SDK 이미지가 여러 개면 자동 선택하지 않고 후보를 보여줍니다.

사용 가능한 이미지가 없다면 스크립트는 중단합니다. 서버가 bare-metal Python 환경만 사용 중이었다면 기존 Docker SDK 이미지가 없을 수 있습니다. 이 경우 Rebellions 담당자에게 host driver와 호환되는 SDK 이미지를 받아야 합니다. **일반 Python 이미지나 cagent HTTP 클라이언트 이미지를 SDK 이미지로 대신 사용할 수 없습니다.** 공식 SDK registry는 인증이 필요하다고 안내되어 있습니다. 저장소/Release 다운로드의 공개 여부와 SDK 이미지 접근 권한은 별개입니다. [공식 Release Notes](https://docs.rbln.ai/latest/supports/release_note.html)

이미지를 알고 있거나 여러 후보 중 선택하려면 실제 이미지명/태그를 지정합니다. 아래 대괄호 부분은 실제 값으로 바꿉니다.

```bash
RBLN_BASE_IMAGE='<서버의-SDK-이미지:태그>' bash docker_run.sh all
```

SDK Python이 기본 `python3`가 아니라 image 내 가상환경에 있다면 interpreter도 지정합니다.

```bash
RBLN_BASE_IMAGE='<SDK-이미지:태그>' SDK_PYTHON=/opt/sdk/bin/python \
  bash docker_run.sh all
```

이미지명이 기억나지 않을 때 직접 확인하는 명령:

```bash
docker ps --format 'table {{.Names}}\t{{.Image}}'
docker image ls
```

SDK 이미지 태그를 추측해서 기본값으로 넣지 않았습니다. Dockerfile의 `RBLN_BASE_IMAGE`는 필수 build argument입니다. SDK 이미지는 Python 3.11 이상, pip, `torch`, `torch-rbln`, `rebel-compiler`가 제공되어야 합니다. 의존성 충돌이 있으면 빌드를 실패시키고 SDK를 임의 교체하지 않습니다.

## 호스트 준비 확인

```bash
rbln-smi
docker version
docker compose version
rbln-ctk info
rbln-ctk cdi list
```

이 구성은 Linux amd64 대상이며, Compose v2와 CDI가 활성화된 Docker가 필요합니다. 공식 toolkit은 CDI를 통해 `/dev/rbln*`, 해당 RSD group, 필요한 host library/tool을 컨테이너에 전달합니다. 기본 선택은 `rebellions.ai/npu=0`입니다. [RBLN Container Toolkit](https://docs.rbln.ai/v0.12.0/software/system_management/container_toolkit.html), [Docker Compose CDI devices](https://docs.docker.com/reference/compose-file/services/#devices)

Toolkit을 이미 설정한 호스트에서는 다음을 다시 실행할 필요가 없습니다. 아직 CDI 설정이 없다면 서버 담당자가 공식 절차에 따라 준비해야 합니다.

```bash
# 호스트 설정을 변경하므로 담당자가 필요한 경우에만 수행
sudo rbln-ctk cdi generate
sudo rbln-ctk runtime configure -r docker
```

런타임 재시작이 필요한지는 호스트 설정에 따라 판단합니다. 이 패키지는 Docker 재시작을 자동 수행하지 않습니다. 최신 toolkit에서 특정 NPU를 선택하면 해당 RSD group도 함께 전달됩니다. RSD topology나 오래된 toolkit을 사용하는 호스트는 위 공식 문서의 그룹 매핑 조건을 확인해야 합니다.

## 실행 모드

```bash
# 이미지 자동 탐색과 .env 생성만
bash docker_run.sh setup

# 이미지 빌드 + 컨테이너 안에서 SDK/NPU 접근 확인
bash docker_run.sh check

# 벤치마크 + HTTP/cagent 테스트 (끝나면 종료)
bash docker_run.sh all

# HTTP/cagent 테스트만 (끝나면 종료)
bash docker_run.sh demo

# 상시 API 서버: 모델 준비가 끝날 때까지 기다린 후 반환
bash docker_run.sh serve

# SDK 없는 별도 클라이언트 컨테이너에서 위 API 호출
bash docker_run.sh client

# API 로그 보기
bash docker_run.sh logs

# 프로젝트 컨테이너/네트워크 종료. 결과와 HF volume은 유지
bash docker_run.sh down
```

`all`과 `demo`는 실제 분류 API를 실행하고 sync/async cagent 호출을 수행한 뒤 자식 서버와 컨테이너를 종료합니다. `serve`는 API를 유지합니다. 같은 NPU를 사용 중인 프로젝트 API가 실행 중이면 helper는 벤치마크/데모를 거부하므로 먼저 명시적으로 `down`을 수행합니다. 다른 프로젝트나 외부 프로세스의 NPU 사용량은 `rbln-smi`로 별도 확인합니다.

설정 후에는 Compose를 직접 사용해도 됩니다.

```bash
docker compose build api
docker compose run --rm --no-deps api check
docker compose --profile tools run --rm --no-deps benchmark
docker compose --profile tools run --rm --no-deps demo
docker compose up -d --no-build --wait --wait-timeout 1800 api
docker compose --profile client run --rm --build client
```

`serve`의 기본 준비 대기 시간은 1800초입니다. helper에서는 `STARTUP_TIMEOUT=3600 bash docker_run.sh serve`처럼 늘릴 수 있습니다. healthcheck는 모델 로딩/컴파일/워밍업 완료를 확인합니다. 컴파일 실패를 CPU 전환으로 숨기지 않습니다.

## .env 설정

`docker_setup.py`가 만드는 `.env`는 git/Docker build context에서 제외되며 권한은 0600입니다. 기존 API 키와 사용자 설정을 보존합니다. 설정 파일을 편집한 뒤 다시 실행하면 됩니다.

| 값 | 기본/역할 |
|---|---|
| `RBLN_BASE_IMAGE` | 자동 탐색 또는 지정한 driver-compatible SDK 이미지 |
| `SDK_PYTHON` | `python3`; SDK 패키지가 설치된 interpreter |
| `RBLN_CDI_DEVICE` | `rebellions.ai/npu=0`; host에서 전달할 CDI handle |
| `NPU_DEVICE` | `0`; 컨테이너 내 SDK가 사용하는 device index |
| `BACKEND`, `DTYPE` | `rbln-compile`, `float16` |
| `BUCKET`, `CPU_THREADS` | `256`, `4` |
| `MAX_PENDING` | `8`; API 대기열 상한 |
| `BIND_ADDRESS`, `PORT` | `127.0.0.1`, `8009`; host API bind 주소/포트 |
| `DECISION_API_KEY` | 자동 생성/기존 값 보존. cagent에서도 같은 키 사용 |
| `WARMUP`, `REPEATS` | `3`, `20`; 벤치마크 반복 설정 |
| `DATASET` | `/app/examples.jsonl`; 컨테이너 내부 평가 데이터 경로 |

다른 NPU나 group을 선택하면 `RBLN_CDI_DEVICE`와 컨테이너 안의 device index가 의도대로 매핑되는지 `check` 결과로 확인합니다. `privileged: true`, 전체 host `/dev` 마운트, `--gpus`는 사용하지 않습니다.

예를 들어 다른 서버의 cagent가 접근하려면 `.env`에서 `BIND_ADDRESS=0.0.0.0`으로 바꾸고 `bash docker_run.sh serve`를 실행합니다. 호스트 방화벽/접근 정책은 별도로 설정되어 있어야 합니다.

## 결과와 캐시

- 호스트 `./results/` ↔ `/app/results`: 벤치마크 요약/로그, HTTP 응답, compile cache.
- Compose named volume `hf-cache` ↔ `/cache/huggingface`: 다운로드한 model/tokenizer.
- 호스트 `./data/` ↔ `/data` (read-only): 평가 데이터 또는 반입한 로컬 파일.

실제 데이터를 쓰려면:

```bash
mkdir -p data
cp /path/to/my-eval.jsonl data/eval.jsonl
# .env에서 DATASET=/data/eval.jsonl, BUCKET=512 등으로 수정
bash docker_run.sh all
```

`down`은 HF cache를 지우지 않습니다. `docker compose down -v`는 캐시까지 삭제하므로 캐시 초기화를 의도할 때만 사용합니다. 이미지에는 모델 가중치, `.env`, 실행 결과를 복사하지 않습니다. 컨테이너에서는 의존성을 다시 설치하지 않습니다(`SKIP_INSTALL=1`).

## cagent 호출

별도 client 서비스는 `Dockerfile.client`의 일반 Python 이미지로 빌드되며 NPU/SDK/torch가 없습니다. 같은 Compose 네트워크에서 `http://api:8009`로 호출합니다. 출력은 `results/client-responses.json`에 저장됩니다.

다른 cagent 서버에서는 기존 wheel/HTTP 클라이언트를 그대로 사용합니다. URL은 `http://REBEL100_HOST:8009`, 인증 키는 `.env`의 `DECISION_API_KEY`와 맞춥니다. [상세 cagent 연동](SERVING.md)

## 검증 범위

- SDK 자동 선택/중복 태그/후보 모호성/기존 API 키 보존 테스트.
- 실제 Docker Compose parser로 CDI 장치, profile, API port, SDK-free client 구성 확인.
- SDK Dockerfile의 BuildKit 정적 검사. 필수 base image argument에는 일반 이미지 기본값을 주지 않도록 설계.
- Linux amd64 HTTP 클라이언트 이미지의 실제 빌드/실행과 torch 미설치 확인.
- 기존 작은 DeBERTa 기반 CPU 및 실제 HTTP 통합 테스트.

**Rebel100 SDK 이미지의 전체 빌드, 호스트 CDI/NPU 접근, 실제 모델의 NPU 컴파일·추론은 이 개발 환경에서 검증하지 않았습니다.** 호스트 SDK 이미지 자동 탐색은 호환성 보장이 아니라 실행에 필요한 Python 패키지가 있는 후보를 찾는 절차입니다. driver와 SDK의 호환성은 해당 서버 환경에서 확인해야 합니다.
