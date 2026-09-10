# 질문, 논문 결과, 상태 API

## 제공 API

- `GET /health`: 서비스와 질문 엔진 상태
- `GET /papers/{paper_id}`: 논문 처리 결과와 개요
- `POST /question`: 분석된 논문에 대한 근거 기반 답변

## 질문 흐름

```text
question
  -> 업로드 논문의 chunks.jsonl 로드
  -> BM25 + multilingual Dense 검색
  -> RRF Hybrid 결합
  -> Cross-Encoder 재정렬
  -> 근거 선택
  -> 보정된 거절 정책
  -> Qwen 로컬 생성
  -> JSON 계약과 인용 검증
```

운영 서버는 논문 분석 중 검색 모델, 생성 모델, 논문별 Dense 임베딩을 준비하고 이후 질문에서 재사용한다. Dense 임베딩은 `dense_embeddings.npz`로 캐시하며 청크 내용, 모델, 리비전이 달라지면 다시 만든다. 준비에 실패해도 논문 파싱 결과는 유지하고 첫 질문에서 다시 시도한다.

질문 응답의 `runtime` 객체에는 전체 시간, 검색 시간, 생성 시간, 생성 시도 횟수, 생성 장치, CPU 대체 사유, 안전 대체 응답 여부, 거절 출처가 포함된다. 이 값으로 느린 요청이 검색, 생성, 장치 전환 중 어디에서 발생했는지 구분할 수 있다.

## 상태 계약

- 미등록 논문: 404
- 분석 전 논문: 409 `analysis_required`
- 분석 중 논문: 409 `analysis_in_progress`
- 분석 실패: 409 `analysis_failed`
- 처리된 질문: 200

오류 응답에는 내부 경로, 모델 예외, 스택 추적을 넣지 않는다.

## 실행 예시

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/papers/paper-...

curl -X POST http://127.0.0.1:8000/question \
  -H "Content-Type: application/json" \
  -d '{"paper_id":"paper-...","question":"이 논문의 주요 한계는 무엇인가?"}'
```

## 검증

```bash
python -B scripts/evaluate_day34.py
python -B scripts/validate_day34.py
```

실제 Uvicorn TCP 서버와 curl을 사용해 업로드, 분석, 결과 조회, 질문을 검증한다. 답변의 모든 인용은 해당 업로드 논문의 런타임 청크와 페이지에 일치해야 한다.
