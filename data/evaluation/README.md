# Evaluation Artifacts

[English](README.md) | [한국어](README_KO.md)

The files in this directory are the smallest retained evidence set for the published portfolio claims.

- Inputs: `questions.jsonl`, `gold_evidence.jsonl`, `abstention_questions.jsonl`, `answer_review_manifest.json`
- Retrieval: `frozen_retrieval_metrics.json`, `corpus_expansion_smoke.json`
- Answer quality: `grounded_generation_outputs.json`, review labels, and `grounded_generation_review.json`
- Container evidence: `container_results.json`, Q&A results, review labels, and `container_qna_review.json`
- Final summary: `final_evaluation.json` and `final_performance.csv`

Generated experiments that are not part of the final claims are intentionally excluded. See [Evaluation](../../docs/evaluation.md).
