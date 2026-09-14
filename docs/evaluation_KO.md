# 평가

[English](evaluation.md) | [한국어](evaluation_KO.md)

## 평가 범위

코퍼스는 텍스트 추출이 가능한 영어 논문 30편입니다. 정확도 라벨은 첫 10편의 검증 문항 40개에만 있으며 영어 20개와 한국어 20개로 구성됩니다. 추가 20편은 파싱과 검색 실행 강건성만 확인합니다.

| 영역 | 범위 | 결과 |
|---|---|---:|
| 코퍼스 | 논문 30편 | 563페이지, 2,463청크 |
| 확장 스모크 | 논문 011-030 | 20/20 통과 |
| 검색 | 검증 40문항 | Recall@1 0.300 |
| 검색 | 검증 40문항 | Recall@3 0.850 |
| 검색 | 검증 40문항 | Recall@5와 @10 1.000 |
| 검색 | 검증 40문항 | MRR 0.5838 |
| 언어별 검색 | 영어 20 / 한국어 20 | MRR 0.6308 / 0.5367 |
| 답변 거절 정책 | 답변 가능 40 / 홀드아웃 10 | 95% 유지 / 90% 거절 |
| 답변 검토 | 고정 20문항 | 통과 3, 부분 통과 14, 실패 3 |
| 인용 검토 | 고정 20문항 | 지지 실패 0건 |
| Docker 질의응답 | 10문항 | 통과 4, 부분 통과 6, 실패 0 |
| Docker API 지연 | CPU 노트북 | 영어 평균 1.47초, 한국어 평균 11.95초 |

모든 검증 문항의 Gold 페이지가 상위 5개 고유 후보 페이지 안에 들어왔습니다. 다만 MRR에서 보듯 정확한 페이지가 항상 첫 순위는 아니며 한국어 질문에서 차이가 더 큽니다. 답변은 인용 지지 성능이 높지만 엄격 통과율은 15%입니다. 선택 근거가 요청한 세부사항 일부를 빠뜨려 부분 통과가 많기 때문입니다. 이를 하나의 답변 정확도처럼 과장하지 않고 별도로 공개합니다.

답변 거절 임곗값은 답변 가능 문항 40개 중 38개를 유지했고, 비근거 보정 문항 10개를 모두 거절했으며, 별도 홀드아웃 10개 중 9개를 거절했습니다. 이전 보정 결과 뒤에 홀드아웃 실패 1건을 숨기지 않고 함께 공개합니다.

## 평가 산출물

- [`frozen_retrieval_metrics.json`](../data/evaluation/frozen_retrieval_metrics.json) - 문항별 순위와 검색 집계
- [`abstention_evaluation.json`](../data/evaluation/abstention_evaluation.json) - 답변 가능 문항 유지율과 비근거 질문 거절률
- [`grounded_generation_review.json`](../data/evaluation/grounded_generation_review.json) - 평가 기준과 라벨을 포함한 고정 답변 검토
- [`container_qna_review.json`](../data/evaluation/container_qna_review.json) - 실제 Docker 응답 검토
- [`final_evaluation.json`](../data/evaluation/final_evaluation.json) - 최종 지표와 출처 해시 매니페스트
- [`final_performance.csv`](../data/evaluation/final_performance.csv) - 포트폴리오용 기계 판독 성능표

## 한계

- 수동 검증 검색 라벨은 논문 10편에만 있습니다.
- 답변 라벨은 어시스턴트 주도 검토이며 독립적인 사람 평가가 아닙니다.
- 한영 대응 문항 때문에 답변 표본의 독립적인 의미 수는 문항 수보다 적습니다.
- CPU 환경의 한국어 번역은 영어 추출 답변보다 느립니다.
- Docker 지연 시간은 특정 노트북 측정치이며 서비스 수준 보장이 아닙니다.
- Gold 근거는 채점과 생성 후 검토에만 사용하며 사용자 답변 생성에는 사용하지 않습니다.
