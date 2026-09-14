# 평가 산출물

[English](README.md) | [한국어](README_KO.md)

이 디렉터리에는 공개 포트폴리오 성능 주장을 뒷받침하는 최소한의 평가 자료만 남겼습니다.

- 입력: `questions.jsonl`, `gold_evidence.jsonl`, `abstention_questions.jsonl`, `answer_review_manifest.json`
- 검색: `frozen_retrieval_metrics.json`, `corpus_expansion_smoke.json`
- 답변 품질: `grounded_generation_outputs.json`, 검토 라벨과 `grounded_generation_review.json`
- 컨테이너 증거: `container_results.json`, 질의응답 결과, 검토 라벨과 `container_qna_review.json`
- 최종 요약: `final_evaluation.json`, `final_performance.csv`

최종 주장에 사용하지 않는 실험 산출물은 의도적으로 제외했습니다. 자세한 내용은 [평가 문서](../../docs/evaluation_KO.md)를 참고하세요.
