# 19일차 검색 실패 분석

고정된 verified 40문항에서 한 검색기라도 gold 근거 페이지를 Top-5에 포함하지 못한 사례 중 10건을 수동 검토했다. 분류는 질문, gold 근거, 세 검색기의 페이지 순위를 함께 확인해 정했다.

| 질문 ID | 언어 | 원인 | 실패 검색기 | Gold 순위 (BM25/Dense/Hybrid) | 판단 근거 |
|---|---|---|---|---|---|
| q-049-ko | ko | chunking | bm25, hybrid_rrf | -/5/8 | 정답이 있는 결론 페이지가 표·그림 캡션과 함께 길고 잡음 많은 청크로 추출되어 핵심 결론 표현이 희석됐다. |
| q-005-en | en | chunking | bm25 | 6/3/5 | 초록의 방법 요약이 여러 청크로 나뉘고 같은 방법명이 본문 여러 페이지에 반복되어 BM25의 정답 페이지가 6위로 밀렸다. |
| q-005-ko | ko | terminology | bm25, dense, hybrid_rrf | -/9/- | 한국어의 일반적인 프레임워크 표현과 영어 논문의 MoE·MLA·Muon·μP 고유 용어가 직접 대응하지 않아 모든 검색기가 정답 페이지를 상위 5개에 올리지 못했다. |
| q-001-ko | ko | terminology | dense | 2/8/1 | '주요 문제'라는 한국어 일반 표현이 operational non-identifiability, RFI, subspace overlap이라는 논문의 구체 용어를 충분히 제공하지 않아 dense 순위가 8위였다. |
| q-017-ko | ko | terminology | dense | 1/-/2 | 한국어의 표현 학습·과정 추론 표현과 영어 초록의 hierarchy·sensing·support·uncertainty 표현 사이 의미 대응이 약해 dense가 10위 안에 들지 못했다. |
| q-038-ko | ko | numbers | bm25 | 6/3/2 | 정답 근거가 INT8, FP32, CPU 명칭, 모델명과 같은 숫자·식별자 중심이며 이 항목들이 여러 페이지에 반복되어 BM25 정답 페이지가 6위였다. |
| q-042-en | en | numbers | dense | 1/9/5 | 평가 근거가 관측 섹터, 표본 수, 임계값 등 수치 중심이라 의미 임베딩이 일반 평가 설명보다 정답 페이지를 낮게 평가해 dense 순위가 9위였다. |
| q-045-ko | ko | numbers | dense | 1/7/1 | 모델 폭, 학습률, 스케일과 μP 표기가 핵심 단서인데 dense가 정량적 식별자를 충분히 우선하지 않아 7위였다. |
| q-043-ko | ko | multiple_evidence | bm25, hybrid_rrf | 7/5/8 | 합성 데이터와 ABIDE 실험, 비교 기준, 평가 지표가 여러 문단·페이지에 분산되어 단일 gold 페이지와 단일 청크 중심 평가에서 hybrid 순위가 8위였다. |
| q-015-en | en | multiple_evidence | dense | 2/6/3 | 개선 근거가 실험 결과와 학습 스케줄 설명, 결론에 분산되어 dense가 개별 결과 페이지를 우선하고 gold 결론 페이지는 6위에 머물렀다. |

## 원인별 건수

- chunking: 2건
- multiple_evidence: 2건
- numbers: 3건
- terminology: 3건

## 해석 범위

이 분류는 오류 원인의 가설을 기록한 진단 자료다. 현재 평가는 단일 gold 페이지 일치 방식이므로, 검색된 페이지가 답변에 유용하더라도 gold 페이지가 아니면 실패로 집계될 수 있다.
