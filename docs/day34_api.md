# Day 34 - question, paper result, and health API

Original roadmap task: `question, paper result, health endpoint 구현`.
Completion criterion: `curl에서 질문을 보내고 답을 받는다.`

## Run locally

Install the pinned dependencies and run one API worker:

```bash
python -m pip install -r requirements.txt
python -B -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

The interactive API is available at `http://127.0.0.1:8000/docs`.
Uploaded PDFs, extracted pages, chunks, SQLite metadata, and per-paper dense
embedding caches are stored below the Git-ignored `data/runtime/` directory.

## Endpoints

| Request | Purpose | Success |
|---|---|---|
| `GET /health` | Read storage, queue, and lazy model readiness | `200` |
| `POST /upload` | Register exactly one text-based PDF | `201` |
| `POST /analyze` | Start parsing and chunking | `202` |
| `GET /jobs/{job_id}` | Poll the analysis job | `200` |
| `GET /papers/{paper_id}` | Read metadata, analysis result, abstract, and sections | `200` |
| `POST /question` | Ask one English or Korean question | `200` |

`GET /papers/{paper_id}` maps job state to a user-facing paper state:
`uploaded`, `queued`, `processing`, `ready`, or `failed`. The abstract and section
list are built only from extracted paper text after analysis succeeds.

`POST /question` returns the answer in the detected question language. A supported
answer contains only model-cited evidence objects with their original text, page,
section, and chunk locator. An unsupported answer uses the calibrated abstention
message and returns an empty evidence list.

## Curl flow

Start with the upload and analysis commands documented in `docs/day33_api.md`,
then replace the placeholders below:

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/papers/<paper_id>
curl -sS -X POST http://127.0.0.1:8000/question \
  -H 'Content-Type: application/json' \
  -d '{"paper_id":"<paper_id>","question":"What is the main contribution?"}'
```

PowerShell example:

```powershell
$body = @{
  paper_id = '<paper_id>'
  question = '이 논문의 주요 기여는 무엇인가?'
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/question' `
  -ContentType 'application/json' -Body $body
```

## Runtime behavior

- The service processes and answers against one selected paper at a time.
- The first question lazily loads the multilingual embedding model, Cross-Encoder,
  and local Qwen generator. Later questions reuse the loaded models.
- Passage embeddings are cached as `dense_embeddings.npz` within that paper's
  private runtime directory. Switching papers reuses model objects and activates
  only the selected paper's chunks.
- The question path is the frozen Day 28 BM25 plus multilingual dense RRF pipeline,
  Cross-Encoder reranking, evidence selection, calibrated Day 31 abstention, and
  the local Day 30 answer generator.
- Seed 378 is applied before model-backed question processing. Generation is also
  configured with sampling disabled.
- Run with one Uvicorn worker. The runtime directory lock rejects a second process
  that could race the serial analysis queue or local model state.

Questions are accepted only after the paper reaches `ready`. A missing paper
returns `404`; unprocessed, running, or failed analysis returns `409`; an unavailable
local model path returns a sanitized `503` with `Retry-After`. Questions are limited
to 2,000 characters and whitespace-only input is rejected.

## Verification

```bash
python -B -m unittest discover -s tests -v
python -B scripts/evaluate_day34.py
python -B scripts/validate_day34.py
```

The evaluator runs all 24 repository tests and a real Uvicorn server on a temporary loopback
port. It uploads `paper-003.pdf`, completes analysis, reads the paper result, and
uses the installed `curl` executable to submit a verified English benchmark
question. It requires cached local models, saves a sanitized acceptance artifact,
and removes the temporary PDF, database, chunks, pages, and embedding cache.

Day 34 proves API behavior and a real answer transaction. It does not replace the
Day 32 answer-quality review. That review's weak aggregate answer quality remains
an explicit limitation until a later generation-model improvement is validated.
