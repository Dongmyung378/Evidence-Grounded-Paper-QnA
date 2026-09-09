# 35일차 전체 API 통합 게이트

## 원 계획 기준

- 작업: UI 없이 API 전체 흐름 테스트
- 완료 기준: PDF -> parse -> index -> question -> answer가 한 번에 성공

## 구현 내용

`scripts/api_acceptance.py`에 실제 HTTP 수용 테스트 흐름을 공통 모듈로 분리했다. `scripts/evaluate_day35.py`는 임시 디렉터리에서 Uvicorn 서버를 실행하고 curl로 질문 API를 호출한다.

검증 순서는 다음과 같다.

1. `/health`에서 질문 엔진의 지연 로딩 상태 확인
2. 미등록 논문 질문이 404인지 확인
3. PDF 업로드 후 201 응답 확인
4. 분석 전 질문이 `analysis_required` 409인지 확인
5. 분석 요청 후 작업 완료까지 조회
6. 페이지, 청크, 논문 개요, Dense 임베딩 캐시와 분석 중 모델 준비 확인
7. curl로 `/question` 호출 후 200 응답 확인
8. 모든 인용이 런타임 청크 원문과 페이지에 정확히 연결되는지 확인
9. 질문 엔진 준비 상태와 검색 및 생성 실행 진단 확인
10. 임시 런타임 제거 및 고정 평가 입력 무변경 확인

## 재현 방법

```bash
python -B scripts/evaluate_day35.py
python -B scripts/validate_day35.py
```

결과는 `data/evaluation/day35_integration_results.json`에 저장된다. 최근 실행에서는 검색 모델을 CPU에 두고 생성 모델을 CUDA에 올렸으며, 답변 9.700초 중 검색 0.970초와 생성 8.729초가 기록됐다. 이 게이트는 전체 기능 연결과 실행 장치 배치를 검증하며 답변 정확도 향상을 주장하지 않는다. 답변 품질 한계는 Day 32 수동 검토 결과를 따른다.
