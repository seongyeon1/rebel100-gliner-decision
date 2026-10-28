# GLiNER2.5-multi-Decide: ATOM MAX 실행

배포 대상은 **ATOM MAX (RBLN-CA25)**입니다. 기존 분류 전용 모델, CPU 비교 벤치마크, HTTP API, cagent 클라이언트를 재사용합니다. SDK 공통 지원이 이 모델의 컴파일 성공이나 가속을 보장하지는 않습니다. 실제 ATOM MAX에서의 컴파일·추론·정확도·속도는 아직 측정하지 않았습니다.

공식 문서에 따르면 Rebel100과 ATOM은 같은 SDK 패키지 버전을 사용하며 `torch-rbln`을 모두 지원합니다. 현재 자동 설치 조합은 SDK 0.12.0입니다. ATOM MAX 서버의 driver/CDI와 호환되는 SDK 환경을 사용해야 합니다.

- [공통 SDK 설치](https://docs.rbln.ai/v0.12.0/getting_started/quick_start.html)
- [ATOM MAX 카드 식별자](https://rebellions.ai/developers/)
- [ATOM과 Rebel100 실행 차이](https://docs.rbln.ai/v0.12.0/software/rbln_pytorch/execution_semantics.html)

## 실행 순서

ATOM MAX 호스트에서 저장소를 받은 뒤 실행합니다. Docker/Compose, host driver, Container Toolkit CDI가 준비되어 있어야 합니다. SDK 이미지가 없으면 자동 빌드하며, 보호된 compiler 다운로드에는 Portal 권한이 필요합니다. 인증 설정과 SDK 준비는 [DOCKER.md](DOCKER.md)를 참고하세요.

```bash
git clone https://github.com/seongyeon1/rebel100-gliner-decision.git
cd rebel100-gliner-decision

# 장비 목록에서 ATOM MAX / RBLN-CA25인지 확인
rbln-smi

# 선택한 카드와 SDK 버전 확인. 기본 CDI npu=0, SDK logical device=0
bash atom_max.sh check

# CPU 원본/포팅 결과 비교 → NPU 컴파일·측정·진단 → HTTP/cagent 예제
bash atom_max.sh all

# 결과 확인
find results/atom-max -name summary.txt -print -exec cat {} \;

# API 상시 실행 → cagent 클라이언트 예제 → 로그
bash atom_max.sh serve
bash atom_max.sh client
bash atom_max.sh logs
```

별도 pip 설정 파일이 있다면 `check`/`all` 실행 전에 다음 변수를 설정합니다. 파일에 실제 Portal 인증이 들어 있어야 하며 파일 내용을 채팅에 보낼 필요는 없습니다.

```bash
export SDK_PIP_CONFIG="$HOME/.config/rebel100/pip.conf"
```

다른 카드는 `RBLN_CDI_DEVICE=rebellions.ai/npu=N`으로 선택하고 `check` 출력의 NPU 이름과 컨테이너 logical index를 확인하세요. 처음에는 카드 하나로 측정합니다. 여러 카드의 RSD 구성은 별도로 검증해야 합니다.

## Rebel100 실행과 달라지는 점

`atom_max.sh`는 기본적으로 FP16, strict fullgraph compile을 선택합니다. 모델이나 수치 처리를 변경하지 않습니다. Compose 프로젝트 이름은 `gliner-atom-max`, 서버 이미지 이름은 `atom-max-gliner-decision:local`이며, 결과와 서빙 컴파일 캐시는 `results/atom-max/` 아래에 저장합니다. Rebel100에서 만든 캐시를 이 폴더로 복사하지 마세요. 서버의 SDK와 ATOM MAX에서 새로 컴파일합니다.

기본 API 포트는 기존과 같은 `127.0.0.1:8009`입니다. 같은 호스트에 기존 API가 실행 중이면 먼저 해당 프로젝트에서 명시적으로 종료하거나 `PORT=8010 bash atom_max.sh serve`처럼 다른 포트를 선택합니다. ATOM MAX API 종료는 `bash atom_max.sh down`입니다. cagent의 요청/응답 형식과 Bearer 인증은 [SERVING.md](SERVING.md)와 같습니다.

## 결과 해석

- 정확도는 정답 `expected`가 있는 평가 데이터로 측정합니다. 기본 예제의 CPU 일치율만으로 실제 업무 정확도를 판단하지 않습니다.
- 속도는 초기 컴파일 시간을 제외한 p50/p95, end-to-end, prepared-request 결과를 CPU와 비교합니다.
- ATOM의 `torch.profiler`는 **호스트 활동만 수집**합니다. CPU trace 이벤트나 `rbln:0` tensor 표기만으로 gather의 물리 실행 위치를 확정하지 않습니다.
- eager gather probe와 compiled graph는 별도 경로입니다. 명시적인 CPU fallback 로그와 `torch.rbln.explain` 결과를 확인하고, compiled graph 내부 위치가 드러나지 않으면 `unknown`으로 남겨 compiler profiler/vendor 확인을 진행합니다.
- 컴파일 실패 시 자동 CPU 성공으로 바꾸지 않습니다. `failure.json`, `workflow.log`, 부분 `report.json`을 확인합니다.
