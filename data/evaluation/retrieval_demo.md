# 21일차 검색 기준선과 사례

## 기준선 표

| 검색기 | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|
| BM25 | 0.325 | 0.825 | 0.900 | 0.5452 |
| Dense | 0.300 | 0.775 | 0.975 | 0.4886 |
| Hybrid RRF | 0.350 | 0.925 | 0.975 | 0.5692 |

Hybrid RRF를 선택한 기준선으로 유지한다.

## 검색 사례

### q-001-ko - multilingual_success

- 질문: 이 논문에서 다루는 주요 문제는 무엇인가?
- 논문 / Gold 페이지: paper-001 / 2
- 요약: 한국어 일반 질문을 lexical·semantic 신호가 함께 보완한 성공 사례
- BM25: Gold 순위 2; Top-5 페이지 [1, 2, 3, 4, 5]
- Dense: Gold 순위 8; Top-5 페이지 [12, 6, 8, 13, 4]
- Hybrid RRF: Gold 순위 1; Top-5 페이지 [2, 1, 4, 3, 12]
- 진단: '주요 문제'라는 한국어 일반 표현이 operational non-identifiability, RFI, subspace overlap이라는 논문의 구체 용어를 충분히 제공하지 않아 dense 순위가 8위였다.
- 다음 작업: 질문 의도를 논문 핵심 용어로 확장하는 방식을 검토한다.

### q-042-en - lexical_rescue

- 질문: How is the Microlensify classifier evaluated on TESS light curves?
- 논문 / Gold 페이지: paper-009 / 23
- 요약: 숫자 중심 질문에서 BM25가 dense의 낮은 순위를 보완한 사례
- BM25: Gold 순위 1; Top-5 페이지 [23, 5, 22, 3, 4]
- Dense: Gold 순위 9; Top-5 페이지 [14, 1, 11, 4, 22]
- Hybrid RRF: Gold 순위 5; Top-5 페이지 [11, 4, 5, 1, 23]
- 진단: 평가 근거가 관측 섹터, 표본 수, 임계값 등 수치 중심이라 의미 임베딩이 일반 평가 설명보다 정답 페이지를 낮게 평가해 dense 순위가 9위였다.
- 다음 작업: 숫자 토큰 보존과 lexical 점수 가중을 검토한다.

### q-049-ko - known_failure

- 질문: 지속적 이상 탐지에서 메모리 효율성에 대한 주요 결론은 무엇인가?
- 논문 / Gold 페이지: paper-005 / 14
- 요약: Top-5 실패와 원인 및 다음 개선 방향을 설명하는 사례
- BM25: Gold 순위 Top-10 밖; Top-5 페이지 [1, 2, 3, 4, 5]
- Dense: Gold 순위 5; Top-5 페이지 [1, 15, 17, 16, 14]
- Hybrid RRF: Gold 순위 8; Top-5 페이지 [1, 3, 4, 15, 17]
- 진단: 정답이 있는 결론 페이지가 표·그림 캡션과 함께 길고 잡음 많은 청크로 추출되어 핵심 결론 표현이 희석됐다.
- 다음 작업: 섹션 경계와 문단 경계를 우선하는 청킹을 검토한다.

## 대화형 재현

```bash
python -B scripts/search_hybrid.py --paper-id paper-001 --query "이 논문에서 다루는 주요 문제는 무엇인가?" --top-k 5
```
