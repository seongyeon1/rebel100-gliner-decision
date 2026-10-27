# 로컬 검증 기록

2026-10-27, macOS arm64, 격리된 Python 3.12.14 환경.

- PyTorch 2.14.1, Transformers 5.17.0, GLiNER2 commit `ca4f68235b50fe4c3ecf3876851d748fe4b15c6c`.
- `python -m pytest -q`: 22 passed. upstream Transformers의 `torch.jit.script` deprecation FutureWarning 및 Starlette/httpx deprecation 경고 각 1건.
- `ruff check .`: 통과.
- `ruff format --check .`: 통과.
- `bash -n run_rebel100.sh`: 통과.
- 실제 DeBERTa 상대위치 attention을 사용하는 작은 무작위 초기화 GLiNER fixture로 원본 API와 포팅 경로를 대조함.
- 저장한 fixture checkpoint를 다시 로딩하여 CPU benchmark CLI, 진단 profiler, report 파일 생성까지 테스트함.
- 실제 287M checkpoint의 전체 가중치 로딩/추론, RBLN SDK/NPU 컴파일·실행 및 속도 측정은 수행하지 않음.

작은 fixture의 분류 정답률은 실제 모델 품질을 나타내지 않는다. NPU 결과는 서버에서 새로 생성해야 한다.

추가 검증: 작은 오프라인 체크포인트로 `run_all.sh`의 CPU 경로를 실제 subprocess 실행해 SDK constraints·환경 기록·벤치마크·최종 요약 생성을 확인했다. 이 테스트는 `SKIP_INSTALL=1`을 사용하며 서버의 SDK 설치나 pip 설치 성공을 대신 증명하지 않는다.

서빙 추가 검증: 실제 uvicorn TCP 서버 시작·준비 대기·cagent 동기/비동기 HTTP 호출·자식 서버 자동 종료를 작은 오프라인 모델로 검증했다. 인증·422·503·429·허용 외 도구 라벨 응답 거부를 확인했다. `cagent-decision-client` wheel을 빌드하고 의존성이 httpx/pydantic뿐임을 확인했다. NPU 실측은 미수행이다.

## Docker 추가 검증 (v0.2.0, 2026-10-28)

추가 6개: SDK 이미지 선택(중복/모호성/없음), 기존 키 보존, 실제 Compose parser 구성, build context 제외 규칙. 전체 28개 테스트 통과.

BuildKit Dockerfile 정적 검사와 Linux amd64 클라이언트 이미지 실제 빌드/실행을 수행했다. SDK base image가 없어 NPU 앱 이미지의 전체 빌드와 CDI/NPU 실행은 수행하지 않았다. 최신 검증 결과는 GitHub Release notes와 최종 테스트 출력 기준이다.

## SDK bootstrap 추가 검증 (v0.3.0, 2026-10-28)

6개 테스트 추가: build context 내부 credential 파일 거부, 공식 SDK 조합 고정, 이미지 부재 시 bootstrap, 기존 이미지 재사용, 인증 인덱스 URL 보존, host 검사 실패 시 빌드 중단. 전체 34개 테스트 통과.

Dockerfile.sdk는 BuildKit 정적 검사 대상이다. Portal SDK 다운로드 및 SDK 이미지 전체 빌드/실제 NPU 실행은 미수행이다.
