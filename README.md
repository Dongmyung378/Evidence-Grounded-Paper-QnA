# Evidence-Grounded Paper Q&A

영어 논문 PDF 한 편을 업로드하고 한국어 또는 영어로 질문하면, 논문 원문의 근거 문장과 페이지를 함께 제공하는 로컬 AI 서비스입니다.

## 핵심 기능

- 텍스트 추출이 가능한 영어 PDF 한 편 업로드
- 제목, 초록, 주요 절을 포함한 논문 개요 확인
- 한국어와 영어 질문 지원 및 질문 언어에 맞춘 답변
- BM25, 다국어 Dense 검색, Hybrid 검색, Cross-Encoder 재정렬
- 원문 근거, 페이지, 절, 청크 식별자 제공
- 근거가 부족할 때 답변 거절
- FastAPI 기반 업로드, 분석, 결과 조회, 질문 API

## 처리 흐름

```text
PDF 업로드
  -> 페이지별 텍스트 추출
  -> 절 인식과 청크 분할
  -> BM25 + Dense 인덱스
  -> Hybrid 검색
  -> Cross-Encoder 재정렬
  -> 근거 선택
  -> 로컬 LLM 답변 생성
  -> 답변과 인용 검증
```

## 현재 범위

| 항목 | 지원 범위 |
|---|---|
| 입력 | 텍스트 추출이 가능한 영어 논문 PDF |
| 처리 단위 | 요청마다 논문 한 편 |
| 질문과 답변 | 한국어 또는 영어 |
| 근거 | 영어 원문, 페이지, 절, 청크 식별자 |
| 파일 제한 | PDF, 최대 20 MiB |
| 제외 | OCR, 이미지, 표, 그래프, 수식 해석, 다중 논문 비교 |

로컬 강건성 코퍼스는 30편이며 563페이지와 2,463개 추적 가능 청크로 구성됩니다. 정확도 평가는 수동 검증된 기존 10편의 한영 질문 40개만 사용합니다. 추가 20편은 파싱과 실행 강건성 검증에 사용하며, Gold 근거를 만들기 전에는 정확도 수치에 포함하지 않습니다.

## 실행 방법

Python 3.11 이상 환경을 권장합니다.

```bash
python -m pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

서버가 시작되면 `http://127.0.0.1:8000/docs`에서 API 문서를 확인할 수 있습니다.

### 전체 API 흐름

```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@data/raw/papers/paper-003.pdf"

curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"paper_id":"paper-..."}'

curl http://127.0.0.1:8000/jobs/job-...
curl http://127.0.0.1:8000/papers/paper-...

curl -X POST http://127.0.0.1:8000/question \
  -H "Content-Type: application/json" \
  -d '{"paper_id":"paper-...","question":"이 논문의 주요 결론은 무엇인가?"}'
```

첫 실행에는 Hugging Face 모델 다운로드가 필요할 수 있습니다. 모델이 캐시에 있으면 이후 실행은 로컬에서 가능합니다.

## 주요 API

| 메서드 | 경로 | 설명 |
|---|---|---|
| `GET` | `/health` | 서버와 질문 엔진 상태 확인 |
| `POST` | `/upload` | PDF 한 편 등록 |
| `POST` | `/analyze` | 비동기 파싱과 청크 생성 시작 |
| `GET` | `/jobs/{job_id}` | 분석 작업 상태 확인 |
| `GET` | `/papers/{paper_id}` | 논문 개요와 처리 결과 조회 |
| `POST` | `/question` | 근거 기반 질문과 답변 실행 |

## 평가 현황

- 30편 전체 수집과 처리 검증 완료
- 고정 평가 세트: 40문항, 영어 20개, 한국어 20개, 전부 `verified`
- 고정 운영 검색 구성: Recall@5 1.000, MRR 0.6317
- 근거 없는 홀드아웃 질문 거절: 10/10
- 수동 답변 검토: 통과 1, 부분 통과 4, 실패 15, 잘못된 거절 7
- 실제 HTTP 통합 흐름: PDF -> parse -> index -> question -> answer 통과

검색 품질과 API 통합은 검증됐지만, 현재 0.5B 로컬 생성 모델의 답변 품질은 공개 서비스 수준이 아닙니다. 위 수치는 기능 완료와 진단 결과이며 배포 품질을 의미하지 않습니다.

## 검증

```bash
python -B scripts/evaluate_day35.py
python -B scripts/validate_day35.py
python -B scripts/verify_project.py
```

전체 흐름 평가 스크립트는 임시 런타임에서 실제 Uvicorn 서버와 curl을 사용하므로 저장된 코퍼스와 평가 데이터를 변경하지 않습니다.

## 저장소 구조

```text
app/                 FastAPI와 런타임 서비스
config/              고정 검색, 생성, 거절 정책 설정
data/
  metadata/          논문 출처와 라이선스 메타데이터
  processed/         페이지와 청크 산출물
  evaluation/        평가 입력, 결과, 보고서
docs/
  project/           범위, 요구사항, 데이터 정책
  api/               API 구현과 사용 설명
  roadmap/           일차별 진행 기록
  reviews/           품질 및 개선 검토
scripts/             파이프라인, 평가, 검증 도구
tests/               API 및 회귀 테스트
```

일차별 작업 기록은 [진행 기록](docs/roadmap/progress.md)에 분리했습니다. 문서 전체 색인은 [문서 안내](docs/README.md)를 참고하세요.

## 데이터와 라이선스

원본 PDF와 QASPER 원본은 라이선스 확인 전까지 로컬에만 보관합니다. 공개 저장소에는 출처 URL, 식별자, 라이선스 메타데이터, 재현용 코드와 허용 가능한 평가 산출물만 포함합니다. 자세한 내용은 [데이터 출처와 정책](docs/project/data_sources.md)을 참고하세요.

## 다음 단계

다음 단계는 간단한 웹 UI 구현입니다. 이후 Docker 실행, 공개 URL 배포, 답변 생성 모델과 거절 정책 개선을 진행합니다.
