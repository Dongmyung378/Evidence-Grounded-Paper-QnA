# Evidence-Grounded Paper Q&A

[English](README.md) | [한국어](README_KO.md)

Evidence-Grounded Paper Q&A는 영어 연구 논문 한 편을 대상으로 질문할 수 있는 로컬 우선 서비스입니다. 영어와 한국어 질문을 받고 같은 언어로 답하며, 답변과 함께 원문 근거, 페이지, 절, 청크 ID를 제공합니다.

이 프로젝트가 해결하려는 문제는 단순합니다. 답변이 논문의 어느 부분에서 나왔는지 쉽게 확인할 수 있어야 합니다.

이 저장소는 로컬 포트폴리오 데모로 제공합니다. Docker Compose가 최종 제공 환경이며 공개 서버 운영과 공개 URL은 프로젝트 범위에서 의도적으로 제외합니다.

## 주요 기능

- 최대 20 MiB의 텍스트 추출 가능한 영어 PDF 한 편 업로드
- 논문 개요, 페이지 텍스트, 절을 고려한 청크 추출
- 영어와 한국어 질문 및 답변
- BM25와 다국어 Dense 검색을 RRF로 결합
- 다국어 Cross-Encoder를 이용한 후보 재정렬
- 고정된 Qwen 모델을 이용한 로컬 답변 생성
- 답변에 사용한 페이지 단위 원문 근거 제공
- 논문에서 근거를 찾지 못할 때 답변 거절
- 큰 파일과 잘못된 PDF를 분석 전에 차단하고 파싱 실패 후에도 다시 업로드할 수 있는 화면
- FastAPI와 한영 Streamlit 화면을 통한 전체 처리 흐름 제공
- 상태 확인이 포함된 백엔드와 UI Docker Compose 서비스 분리 실행

## 처리 방식

```text
PDF 업로드
  -> 페이지별 텍스트 추출
  -> 절을 고려한 청크 분할
  -> BM25 및 Dense 인덱스
  -> RRF 결합
  -> Cross-Encoder 재정렬
  -> 근거 선택
  -> 로컬 답변 생성
  -> 답변 및 인용 검증
```

검색은 사용자가 선택한 논문에 속한 청크만 대상으로 합니다. 업로드 논문과 인덱스는 고정 평가 코퍼스와 분리해 저장합니다.

## 현재 평가 결과

| 검증 항목 | 결과 |
|---|---:|
| 로컬 강건성 코퍼스 | 논문 30편, 563페이지, 2,463청크 |
| 검증된 벤치마크 | 논문 10편의 질문 40개 |
| 벤치마크 언어 | 영어 20개, 한국어 20개 |
| 운영 검색 Recall@5 | 1.000 |
| 운영 검색 MRR | 0.6317 |
| 근거 없는 홀드아웃 질문 거절 | 10/10 |
| 실제 HTTP 통합 흐름 | 통과 |
| 실제 브라우저 PDF 업로드 흐름 | 통과 |
| 실제 브라우저 답변 및 근거 흐름 | 통과 |
| 실제 브라우저 오류 처리 및 재시도 흐름 | 통과 |
| Docker Compose 백엔드 및 UI 상태 검증 | 통과 |

검색 수치는 질문마다 수동으로 검증한 Gold 페이지 한 개를 기준으로 측정했습니다. 추가 논문 20편은 파싱과 실행 강건성 검증에 사용합니다. 아직 수동 검증 질문과 Gold 근거가 없으므로 정확도 수치에는 포함하지 않습니다.

현재 가장 부족한 부분은 답변 생성 모델입니다. 균형 있게 선정한 20문항 수동 검토 결과는 통과 1개, 부분 통과 4개, 실패 15개였습니다. API와 근거 추적은 동작하지만 현재 0.5B 로컬 모델의 답변 품질을 운영 수준이라고 주장하지 않습니다.

## 빠른 실행

### Docker Compose

Docker Desktop 또는 Compose v2를 지원하는 Docker Engine이 필요합니다. 기본 구성은 백엔드와 UI를 함께 실행하고, 업로드 논문 처리 자료와 모델 다운로드를 이름 있는 볼륨에 보존합니다. UI는 외부에 노출하지 않은 Compose 내부 주소로 백엔드와 통신합니다.

```bash
docker compose up --build
```

웹 화면은 `http://127.0.0.1:8501`, 대화형 API 문서는 `http://127.0.0.1:8000/docs`에서 확인합니다. 업로드 자료와 모델 캐시를 유지하면서 서비스를 종료하려면 다음 명령을 사용합니다.

```bash
docker compose down
```

최초 백엔드 빌드는 CPU 전용 PyTorch 실행 환경을 설치합니다. 첫 분석에서는 고정된 임베딩, 재정렬, 생성 모델을 내려받을 수 있습니다. 포트, 대기열 크기, 모델 사전 준비, 오프라인 모드, 로그 수준 또는 Hugging Face 토큰을 바꿀 때만 `.env.example`을 `.env`로 복사해 수정합니다. 실제 `.env`는 Git과 Docker 빌드 컨텍스트에서 제외됩니다.

### Python

컨테이너 없이 실행할 때는 Python 3.11 이상 환경을 권장합니다.

```bash
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

두 번째 터미널에서 Streamlit을 실행합니다.

```bash
python -m streamlit run ui/app.py
```

웹 화면은 `http://127.0.0.1:8501`, 대화형 API 문서는 `http://127.0.0.1:8000/docs`에서 확인할 수 있습니다.

논문 분석 중 고정된 임베딩, 재정렬, 생성 모델을 준비합니다. 처음 실행할 때는 Hugging Face에서 모델을 내려받을 수 있습니다. 6GB GPU용 프로필은 검색 모델을 CPU에 두고 답변 생성에 GPU 메모리를 우선 사용합니다. 실행 설정을 바꾼 뒤에는 FastAPI를 다시 시작해야 합니다.

## API 사용 흐름

PDF를 등록합니다.

```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@data/raw/papers/paper-003.pdf"
```

응답으로 받은 `paper_id`를 사용해 분석을 시작합니다.

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"paper_id":"paper-..."}'
```

작업 상태와 논문 처리 결과를 확인합니다.

```bash
curl http://127.0.0.1:8000/jobs/job-...
curl http://127.0.0.1:8000/papers/paper-...
```

분석이 완료되면 질문을 보냅니다.

```bash
curl -X POST http://127.0.0.1:8000/question \
  -H "Content-Type: application/json" \
  -d '{"paper_id":"paper-...","question":"이 논문의 주요 기여는 무엇인가?"}'
```

근거가 충분한 응답에는 답변과 모델이 실제로 인용한 근거 객체만 포함됩니다.

```json
{
  "paper_id": "paper-...",
  "question": "이 논문의 주요 기여는 무엇인가?",
  "question_language": "ko",
  "answer": "...",
  "sufficiency": "sufficient",
  "abstention_reason": null,
  "runtime_seconds": 9.7,
  "runtime": {
    "total_seconds": 9.7,
    "retrieval_seconds": 0.97,
    "generation_seconds": 8.729,
    "generation_attempts": 1,
    "llm_device": "cuda",
    "device_fallback_reason": null,
    "fallback_used": false,
    "abstention_source": null
  },
  "evidence": [
    {
      "evidence_id": "ev-...",
      "page": 3,
      "section": "Introduction",
      "text": "Original text from the paper...",
      "locator": {
        "page_label": "p. 3",
        "section": "Introduction",
        "chunk_id": "paper-...-p003-c001"
      }
    }
  ]
}
```

## API 목록

| 메서드 | 경로 | 용도 |
|---|---|---|
| `GET` | `/health` | 저장소, 작업 큐, 모델 준비 상태 확인 |
| `POST` | `/upload` | PDF 한 편 등록 |
| `POST` | `/analyze` | 파싱과 청크 생성 요청 |
| `GET` | `/jobs/{job_id}` | 분석 상태 확인 |
| `GET` | `/papers/{paper_id}` | 논문 개요와 처리 결과 조회 |
| `POST` | `/question` | 영어 또는 한국어 질문 |

## 모델과 검색 구성

- 임베딩: `intfloat/multilingual-e5-small`
- 재정렬: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- 생성 모델: `Qwen/Qwen2.5-0.5B-Instruct`
- 희소 검색: BM25 Top-20
- Dense 검색: 정규화 코사인 유사도 Top-20
- 결합: 동일 가중치 RRF, `rrf_k=60`
- 최종 근거: Top-5, 페이지마다 최대 2청크

모델 리비전과 실행 설정은 `config/`에 고정되어 있어 저장된 평가 결과와 현재 구현을 비교할 수 있습니다.

브라우저 실행은 `config/generation_runtime.json`을 사용합니다. 검토된 384토큰은 유지하고 GPU에서는 최대 3회 검증하며, CPU 대체 실행은 시간이 제한된 최대 2회로 줄입니다. 측정 결과와 한계는 [로컬 실행 성능 개선](docs/reviews/runtime_performance_KO.md)에 정리했습니다.

## 저장소 구조

```text
app/                 FastAPI 경로와 실행 서비스
config/              검색, 생성, 답변 거절 설정
data/
  metadata/          논문 출처와 라이선스 기록
  processed/         페이지와 청크 산출물
  evaluation/        벤치마크 입력과 평가 결과
docs/
  project/           범위, 요구사항, 데이터 정책
  api/               API 동작과 수용 기준
  ui/                화면 동작과 근거 표시
  reviews/           품질 및 개선 검토
scripts/             수집, 검색, 평가, 검증 도구
tests/               API와 회귀 테스트
ui/                  한영 Streamlit 화면과 API 클라이언트
Dockerfile           백엔드와 UI 이미지 빌드 대상
compose.yaml         로컬 2개 서비스와 영속 볼륨 구성
```

## 검증 방법

저장된 평가 자료 검증과 회귀 테스트를 실행합니다.

```bash
python -B scripts/verify_project.py
```

PDF 업로드부터 답변 생성까지 실제 HTTP 흐름을 다시 실행합니다.

```bash
python -B scripts/evaluate_day35.py
python -B scripts/validate_day35.py
```

실제 브라우저로 수행한 Streamlit 수용 결과는 다음 명령으로 검증합니다.

```bash
python -B scripts/validate_day36.py
python -B scripts/validate_evidence_ui.py
python -B scripts/validate_error_recovery_ui.py
```

Docker Compose 수용 검증을 다시 실행하거나 저장 결과를 확인합니다.

```bash
python -B scripts/evaluate_container.py
python -B scripts/validate_container.py
```

통합 평가 스크립트는 임시 포트에 Uvicorn을 실행하고 curl로 질문 API를 호출합니다. 페이지와 청크 추적성을 확인한 뒤 임시 런타임 디렉터리를 제거합니다.

## 범위와 한계

현재 MVP는 스캔 PDF, OCR, 이미지나 그래프 해석, 신뢰할 수 있는 수식 해석, 여러 논문 비교, 사용자 계정을 지원하지 않습니다. 공개 서비스가 아닌 로컬 포트폴리오 데모로 제공하며 Docker 패키징과 로컬 서비스 연결을 검증했습니다. 표와 그림의 문자가 일반 텍스트로 추출될 수는 있지만 시각적 구조를 해석하지 않습니다.

원본 PDF, QASPER 원본 파일, 사용자 업로드 파일은 재배포 조건이 명시적으로 허용하지 않는 한 로컬에만 보관합니다. 자세한 내용은 [데이터 출처와 취급 정책](docs/project/data_sources.md)을 참고하세요.

## 상세 문서

- [프로젝트 범위](docs/project/scope.md)
- [기능 및 비기능 요구사항](docs/project/requirements.md)
- [API 전체 통합 게이트](docs/api/integration.md)
- [Streamlit 사용자 화면과 근거 패널](docs/ui/interface_KO.md)
- [Docker Compose 로컬 실행](docs/deployment/docker_KO.md)
- [품질 및 개선 검토](docs/reviews/improvements.md)
- [로컬 실행 성능 개선](docs/reviews/runtime_performance_KO.md)

현재 브라우저 흐름은 PDF 업로드, 분석, 논문 개요, 질문 입력, 답변, 추적 가능한 원문 근거, 한영 업로드 오류, 파싱 실패 후 재시도까지 지원합니다. 백엔드와 UI의 Docker Compose 실행도 검증했습니다. 로컬 실행이 이 포트폴리오 프로젝트의 최종 제공 범위입니다.
