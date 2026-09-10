# 평가 산출물 안내

## 기준 데이터

- `questions.jsonl`: 검증된 질문 40개
- `gold_evidence.jsonl`: Gold 답변과 근거 40개
- `split_manifest.json`: 평가 분할 정보

## 검색 평가

- `dense_metrics.json`, `reranker_metrics.json`: 모델별 지표
- `retrieval_comparison.json`, `retrieval_comparison.csv`: 검색 방식 비교
- `day28_production_metrics.json`: 고정 운영 검색 구성 결과

## 답변과 거절 평가

- `day30_e2e_results.json`: 로컬 생성 모델 종단간 실행
- `day31_abstention_results.json`: 거절 정책 홀드아웃 평가
- `day32_manual_review.json`: 수동 답변 품질 검토
- `day32_failure_cases.md`: 엄격 실패 사례

## API 평가

- `day33_api_results.json`: 업로드와 분석 API 실행 증거
- `day34_api_results.json`: 질문과 결과 API 실행 증거
- `day35_integration_results.json`: PDF부터 답변까지 전체 HTTP 통합 증거
  - 분석 중 모델 준비 장치와 질문 단계별 실행 시간 포함

## UI 평가

- `day36_ui_results.json`: 실제 Streamlit 브라우저 업로드와 분석, 개요, 질문 입력 검증 결과
- `evidence_ui_results.json`: 실제 브라우저 질문, 답변, 원문 근거, 페이지, 절, 청크 ID 표시 검증 결과

JSON, JSONL, CSV의 필드 이름과 영어 논문 원문은 프로그램 계약과 평가 재현성을 위해 유지한다. 사람이 읽는 설명 문서는 한국어로 작성한다.
