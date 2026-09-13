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
- `grounded_generation_outputs.json`: 같은 20문항과 저장 근거를 사용한 근거 우선 답변 출력
- `grounded_generation_review_labels.jsonl`: 문항별 수동 의미 판정
- `grounded_generation_review.json`: 기존 경로와 근거 우선 경로의 해시 연결 비교

## API 평가

- `day33_api_results.json`: 업로드와 분석 API 실행 증거
- `day34_api_results.json`: 질문과 결과 API 실행 증거
- `day35_integration_results.json`: PDF부터 답변까지 전체 HTTP 통합 증거
  - 분석 중 모델 준비 장치와 질문 단계별 실행 시간 포함

## UI 평가

- `day36_ui_results.json`: 실제 Streamlit 브라우저 업로드와 분석, 개요, 질문 입력 검증 결과
- `evidence_ui_results.json`: 실제 브라우저 질문, 답변, 원문 근거, 페이지, 절, 청크 ID 표시 검증 결과
- `error_recovery_ui_results.json`: 크기 초과, 잘못된 PDF, 파싱 실패 표시와 정상 파일 재시도 검증 결과

## 컨테이너 평가

- `container_results.json`: Docker Compose 백엔드와 UI 빌드, 상태 확인, HTTP 연결 및 정리 검증 결과
- `container_qna_results.json`: 논문 3편과 한영 질문 10개의 실제 Docker Compose 종단간 실행, 시간, 오프라인 재시작 결과
- `container_qna_review_labels.jsonl`: 컨테이너 질의응답 10건의 의미 품질 판정
- `container_qna_review.json`: 실행 결과, 수동 판정, Gold를 해시로 연결한 컨테이너 검토 요약

컨테이너 검증은 로컬 포트폴리오 실행 범위만 다룬다. 공개 서버와 공개 URL은 프로젝트 완료 조건이 아니다.

## 최종 평가 동결

선택 실행 실험의 `answer_coverage_outputs.json`은 기준선과 후보 60문항 응답, `answer_coverage_review_labels.jsonl`은 고정 20문항 판정, `answer_coverage_review.json`은 집계와 해시다. 기본 API는 기존 방식을 유지한다. [성과와 한계](../../docs/reviews/answer_coverage_KO.md)를 참고한다.

- `frozen_retrieval_metrics.json`: CPU 검색 장치와 정확한 모델 리비전으로 다시 실행한 검증 40문항 현재 검색 결과
- `final_performance.csv`: 범위, 단위, 출처와 해석 조건을 포함한 최종 성능표
- `final_evaluation.json`: 평가 입력, 설정, 모델, 구현, 측정 산출물과 패키징 해시를 묶은 최종 동결 매니페스트

과거 `day28_production_metrics.json`의 MRR 0.6317은 이력 비교용으로 보존한다. 현재 대표 검색 지표는 재현 가능한 고정 리비전 평가의 MRR 0.5838이다. 두 평가 모두 Recall@5와 Recall@10은 1.000이다.

JSON, JSONL, CSV의 필드 이름과 영어 논문 원문은 프로그램 계약과 평가 재현성을 위해 유지한다. 사람이 읽는 주요 설명은 영어와 한국어 문서를 함께 제공한다.
