# PDF 업로드와 분석 API

## 제공 API

- `POST /upload`: PDF 한 편 검증 및 등록
- `POST /analyze`: 비동기 파싱과 청크 생성 시작
- `GET /jobs/{job_id}`: 작업 상태 확인

## 주요 계약

`POST /upload`는 `multipart/form-data`의 `file` 하나만 받는다. 최대 크기는 20 MiB이며 PDF 헤더, 파서 열기, 암호화 여부, 텍스트 존재 여부를 확인한다. 성공하면 201과 고유 `paper_id`를 반환하고, 실패하면 생성 중인 파일을 정리한다.

`POST /analyze`는 요청 스레드를 막지 않고 202를 반환한다. 작업은 SQLite에 저장되며 한 번에 하나의 논문만 처리한다. 같은 논문의 대기, 실행, 완료 작업은 재사용하고 실패 작업은 다시 실행할 수 있다.

`GET /jobs/{job_id}`는 `queued`, `running`, `completed`, `failed` 상태를 반환한다. 내부 예외와 로컬 경로는 사용자 응답에 노출하지 않는다.

## 저장 구조

```text
data/runtime/
  service.sqlite3
  papers/{paper_id}/
    source.pdf
    pages.jsonl
    chunks.jsonl
    ingestion_report.json
```

업로드 논문의 산출물은 고정 평가 코퍼스와 분리된다. 모든 청크는 `paper_id`, `source_page_id`, `page`, `chunk_id`, `section`, `text`를 통해 원본 페이지로 추적할 수 있다.

## 실행 예시

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000

curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@data/raw/papers/paper-003.pdf"

curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"paper_id":"paper-..."}'

curl http://127.0.0.1:8000/jobs/job-...
```

## 검증

```bash
python -B scripts/evaluate_day33.py
python -B scripts/validate_day33.py
```

단위 테스트는 형식, 크기, 손상 PDF, 텍스트 없음, 경로 형태 파일명, 동시 요청, 재시작 복구, 작업 실패와 재시도를 포함한다.
