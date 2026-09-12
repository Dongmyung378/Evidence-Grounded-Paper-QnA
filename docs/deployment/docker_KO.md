# Docker Compose 로컬 실행

[English](docker.md) | [한국어](docker_KO.md)

## 구성

Docker Compose는 최상단 `Dockerfile`에서 두 개의 실행 이미지를 만듭니다.

- `backend`는 8000번 포트에서 Uvicorn 작업자 한 개를 실행합니다. PDF 파싱, 검색, 재정렬, 답변 생성과 영속 실행 상태를 담당합니다.
- `ui`는 8501번 포트에서 Streamlit을 실행합니다. Compose 내부 네트워크의 `http://backend:8000` 주소로 백엔드를 호출합니다.

두 이미지는 권한이 제한된 `paperqna` 사용자로 실행합니다. Compose는 Linux capability를 제거하고 새 권한 획득을 막으며, 백엔드 상태가 정상이 된 뒤 UI를 시작합니다. 두 서비스 모두 HTTP 상태 확인을 사용합니다.

UI 이미지에는 Streamlit과 HTTP 클라이언트 의존성만 포함합니다. 백엔드 이미지에는 CPU 전용 PyTorch와 검색 및 생성 의존성을 설치합니다. 이를 통해 UI 이미지 크기를 줄이고 모델 라이브러리가 중복 설치되는 일을 피합니다.

## 시작과 종료

로컬 데모를 빌드하고 실행합니다.

```bash
docker compose up --build
```

다음 주소를 사용합니다.

- UI: `http://127.0.0.1:8501`
- API 문서: `http://127.0.0.1:8000/docs`
- API 상태: `http://127.0.0.1:8000/health`

두 포트는 기본적으로 `127.0.0.1`에만 연결하므로 네트워크의 다른 장치에 데모가 노출되지 않습니다.

업로드 논문 상태와 내려받은 모델을 유지하면서 종료합니다.

```bash
docker compose down
```

두 프로젝트 볼륨의 자료가 더 이상 필요하지 않을 때만 볼륨까지 삭제합니다.

```bash
docker compose down --volumes
```

## 환경변수 설정

Compose는 실행 설정을 환경변수로 전달합니다. 로컬 `.env`가 없어도 기본값으로 실행됩니다. 값을 바꾸려면 `.env.example`을 `.env`로 복사한 뒤 복사본을 수정합니다.

| 환경변수 | 기본값 | 용도 |
|---|---:|---|
| `PAPER_QNA_API_PORT` | `8000` | 호스트에 공개할 백엔드 포트 |
| `PAPER_QNA_UI_PORT` | `8501` | 호스트에 공개할 Streamlit 포트 |
| `PAPER_QNA_SEED` | `378` | 재현 가능한 애플리케이션 시드 |
| `PAPER_QNA_MAX_PENDING_JOBS` | `16` | 분석 대기열의 최대 작업 수 |
| `PAPER_QNA_PREPARE_MODELS` | `true` | 분석 중 검색 및 생성 모델 준비 여부 |
| `PAPER_QNA_MODEL_LOCAL_FILES_ONLY` | `false` | 캐시에 있는 모델 파일만 사용 |
| `PAPER_QNA_LOG_LEVEL` | `info` | Uvicorn 로그 수준 |
| `HF_TOKEN` | 비어 있음 | 모델 다운로드용 선택적 Hugging Face 토큰 |

API 자료 경로와 UI에서 백엔드로 연결하는 주소는 Compose 내부에서 고정합니다. 따라서 호스트별 값이 애플리케이션 코드에 들어가지 않습니다. 정수 또는 Boolean 설정이 잘못되면 백엔드는 명확한 설정 오류와 함께 즉시 종료합니다.

`.env`는 커밋하지 않습니다. Git과 Docker 빌드 컨텍스트에서 모두 제외되며 예제 파일에는 실제 인증 정보가 없습니다.

## 영속 자료

Compose는 이름 있는 볼륨 두 개를 만듭니다.

- `paper-qna-runtime`에는 비공개 업로드 PDF, SQLite 메타데이터, 추출 페이지, 청크와 논문별 Dense 인덱스를 저장합니다.
- `paper-qna-model-cache`에는 Hugging Face 모델을 저장해 다음 실행에서 다시 내려받지 않도록 합니다.

저장소의 논문 코퍼스, 평가 자료, 로컬 구현 일지와 호스트 `.env`는 이미지 빌드 컨텍스트에서 제외합니다. 사용자 업로드가 이미지에 복사되지 않습니다.

## 검증

수용 평가기는 사용하지 않는 호스트 포트를 선택하고 패키징 검증에서만 모델 사전 준비를 끕니다. 빠른 실행과 동일한 Compose 구성을 빌드하고 실행한 뒤 두 서비스의 상태, 백엔드와 UI HTTP 응답, 내부 서비스 주소, 시드 378과 비루트 사용자를 확인합니다. 마지막에는 임시 컨테이너, 네트워크와 테스트 볼륨을 제거합니다.

```bash
python -B scripts/evaluate_container.py
python -B scripts/validate_container.py
```

기계 판독 결과는 `data/evaluation/container_results.json`에 저장합니다. 이 검증은 로컬 컨테이너 패키징과 서비스 연결을 입증하며 답변 품질을 입증하지는 않습니다. 공개 서버 운영은 프로젝트 범위에서 의도적으로 제외합니다.

## 운영 참고사항

- 실행 디렉터리는 분석 작업자 한 개가 소유하고 모델을 프로세스 안에서 공유하므로 백엔드 작업자는 한 개로 유지합니다.
- 기본 이미지는 CPU 전용이므로 NVIDIA 실행 환경이 없어도 동작합니다. 호스트 GPU를 사용할 수 있다면 로컬 Python 실행이 더 빠를 수 있습니다.
- 모델을 사용하는 첫 분석은 고정된 모델 파일을 내려받고 메모리에 올리는 동안 더 오래 걸릴 수 있습니다.
- 8000번 또는 8501번 포트를 이미 사용 중이면 `.env`에서 호스트 포트를 변경합니다.
- 모델 캐시를 준비하기 전에 오프라인 모드를 켜면 첫 질문에서 모델을 불러올 수 없습니다.
