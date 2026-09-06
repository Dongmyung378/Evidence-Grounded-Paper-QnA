# Day 21 Retrieval Baseline and Demo

## Baseline table

| Retriever | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|
| BM25 | 0.325 | 0.825 | 0.900 | 0.5452 |
| Dense | 0.300 | 0.775 | 0.975 | 0.4886 |
| Hybrid RRF | 0.350 | 0.925 | 0.975 | 0.5692 |

Hybrid RRF is retained as the selected baseline.

## Retrieval demo

### q-001-ko — multilingual_success

- Question: 이 논문에서 다루는 주요 문제는 무엇인가?
- Paper / gold page: paper-001 / 2
- Summary: 한국어 일반 질문을 lexical·semantic 신호가 함께 보완한 성공 사례
- BM25: gold rank 2; Top-5 pages [1, 2, 3, 4, 5]
- Dense: gold rank 8; Top-5 pages [12, 6, 8, 13, 4]
- Hybrid RRF: gold rank 1; Top-5 pages [2, 1, 4, 3, 12]
- Diagnosis: '주요 문제'라는 한국어 일반 표현이 operational non-identifiability, RFI, subspace overlap이라는 논문의 구체 용어를 충분히 제공하지 않아 dense 순위가 8위였다.
- Next action: 질문 의도를 논문 핵심 용어로 확장하는 방식을 검토한다.

### q-042-en — lexical_rescue

- Question: How is the Microlensify classifier evaluated on TESS light curves?
- Paper / gold page: paper-009 / 23
- Summary: 숫자 중심 질문에서 BM25가 dense의 낮은 순위를 보완한 사례
- BM25: gold rank 1; Top-5 pages [23, 5, 22, 3, 4]
- Dense: gold rank 9; Top-5 pages [14, 1, 11, 4, 22]
- Hybrid RRF: gold rank 5; Top-5 pages [11, 4, 5, 1, 23]
- Diagnosis: 평가 근거가 관측 섹터, 표본 수, 임계값 등 수치 중심이라 의미 임베딩이 일반 평가 설명보다 정답 페이지를 낮게 평가해 dense 순위가 9위였다.
- Next action: 숫자 토큰 보존과 lexical 점수 가중을 검토한다.

### q-049-ko — known_failure

- Question: 지속적 이상 탐지에서 메모리 효율성에 대한 주요 결론은 무엇인가?
- Paper / gold page: paper-005 / 14
- Summary: Top-5 실패와 원인 및 다음 개선 방향을 설명하는 사례
- BM25: gold rank not in Top-10; Top-5 pages [1, 2, 3, 4, 5]
- Dense: gold rank 5; Top-5 pages [1, 15, 17, 16, 14]
- Hybrid RRF: gold rank 8; Top-5 pages [1, 3, 4, 15, 17]
- Diagnosis: 정답이 있는 결론 페이지가 표·그림 캡션과 함께 길고 잡음 많은 청크로 추출되어 핵심 결론 표현이 희석됐다.
- Next action: 섹션 경계와 문단 경계를 우선하는 청킹을 검토한다.

## Interactive reproduction

```bash
python -B scripts/search_hybrid.py --paper-id paper-001 --query "이 논문에서 다루는 주요 문제는 무엇인가?" --top-k 5
```
