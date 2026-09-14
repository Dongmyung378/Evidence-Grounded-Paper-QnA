# Failure Case Analysis

[English](failure_cases.md) | [한국어](failure_cases_KO.md)

This analysis examines ten representative failures or near-misses from the frozen portfolio evaluation. It separates retrieval rank, evidence selection, refusal policy, answer completeness, and Korean translation so that a strong retrieval score is not mistaken for complete answer quality.

## Findings

- No verified citation hallucination was observed in the fixed 20-answer review. Citation support failures were 0.
- Two answerable questions were incorrectly refused because their top reranker scores fell below the fixed `-3.35` threshold.
- One unsupported Korean holdout question passed the same threshold. Generation was not run in that evaluation, so this is a hallucination exposure, not a confirmed generated hallucination.
- Several answers were partial even when the Gold page was highly ranked. The remaining gap is often evidence selection within a page or incomplete answer composition rather than page retrieval.
- Korean failures combine the same evidence-coverage problem with slower and sometimes awkward translation.

The ten records represent eight independent question meanings because `q-001` and `q-042` are bilingual pairs.

## Case summary

| ID | Type | Gold rank or score | Result | Main cause |
|---|---|---:|---|---|
| `q-001-en` | False refusal | rank 1, score -5.54 | Fail | A single relevance threshold rejects answerable evidence |
| `q-001-ko` | False refusal | rank 3, score -4.12 | Fail | The bilingual counterpart is rejected by the same gate |
| `abs-test-006-ko` | Unsupported pass | score -2.02 | Risk | An unrelated question passes the -3.35 threshold |
| `q-017-ko` | Nonresponsive answer | rank 1 | Fail | Selected context misses the required inference hierarchy |
| `q-015-en` | Missing numeric detail | rank 1 | Partial | Selected evidence omits 11 schedules and dataset names |
| `q-042-en` | Retrieval and context gap | rank 5 | Partial | Relevant page is late and the answer omits pipelines and counts |
| `q-042-ko` | Retrieval, context, translation | rank 5 | Partial | The same evidence gap remains after Korean translation |
| `q-043-ko` | Translation and completeness | rank 1 | Partial | Awkward terminology and omitted benchmark details |
| `q-041-ko` | Incomplete comparison | rank 2 | Partial | Gripper names and evaluation dimensions are omitted |
| `q-034-en` | Incomplete limitation | rank 2 | Partial | Assumptions and explicit constructions are omitted |

## Detailed cases

### 1. q-001-en

**Question:** What is the main problem addressed in this paper?

**Observed:** The verified page is ranked first, but the top reranker score is `-5.54`, below the refusal threshold. The system returns an insufficient-evidence response.

**Cause and impact:** Page retrieval succeeds, but the global score threshold treats the answerable evidence as irrelevant. This is a false refusal and directly reduces answerable retention.

**Potential mitigation:** Calibrate refusal with more than one signal, such as agreement between sparse and dense retrieval, selected-sentence coverage, and question type. Any change must be tested against the unsupported holdout because simply lowering the threshold increases unsafe answers.

### 2. q-001-ko

**Question:** 이 논문에서 다루는 주요 문제는 무엇인가?

**Observed:** The verified page is ranked third, but the top score is `-4.12`. The system refuses the Korean counterpart of the answerable English question.

**Cause and impact:** Cross-language scoring shifts the score but does not resolve the threshold error. A bilingual pair fails consistently, showing that the issue is the refusal gate rather than answer generation.

**Potential mitigation:** Evaluate language-conditioned score calibration or a bilingual semantic-support feature while retaining one documented safety policy.

### 3. abs-test-006-ko

**Question:** 이 논문에서 목성의 위성 수는 몇 개라고 제시하는가?

**Observed:** The question is unrelated to `paper-016`, but its top score is `-2.02`, above the `-3.35` threshold, so the pre-generation policy predicts sufficient evidence.

**Cause and impact:** A scalar relevance score can be high for accidental semantic overlap. The evaluation stops at the policy decision, so no unsupported answer was generated, but this case exposes a hallucination path.

**Potential mitigation:** Add an evidence entailment or answerability check before generation and include adversarial out-of-domain questions in threshold evaluation.

### 4. q-017-ko

**Question:** 이 논문은 표현 학습과 과정 추론을 어떻게 연결하는가?

**Observed:** The Gold page is ranked first, yet the returned answer discusses Earth observation data and downstream adaptation without the required observation-to-inference hierarchy.

**Cause and impact:** Page-level retrieval succeeds, but selected chunks do not contain the specific relationship requested. The fluent response is nonresponsive despite using related evidence.

**Potential mitigation:** Rerank sentences against relation-focused questions and require coverage of key entities from the question before composing the answer.

### 5. q-015-en

**Question:** What evidence shows that the proposed approach improves continual anomaly detection?

**Observed:** The Gold page is ranked first. The answer mentions state-of-the-art behavior and the online setting but omits 11 task schedules and the MVTecAD and VisA datasets.

**Cause and impact:** The selected context captures the conclusion but not the quantitative evaluation setup. The answer is supported but incomplete.

**Potential mitigation:** For evidence-seeking questions, reserve one evidence slot for quantitative results and one for dataset or experimental setup details.

### 6. q-042-en

**Question:** How is the Microlensify classifier evaluated on TESS light curves?

**Observed:** The Gold page is ranked fifth. The answer includes Sector 12 and Eleanor but omits SPOC, QLP, and the post-cleaning candidate counts.

**Cause and impact:** The relevant page barely enters Recall@5, and the five-passage evidence budget does not preserve the complete multi-pipeline result.

**Potential mitigation:** Expand evidence around late-ranked quantitative pages or merge adjacent chunks when a question asks for a multi-part evaluation procedure.

### 7. q-042-ko

**Question:** Microlensify 분류기는 TESS 광도 곡선에서 어떻게 평가되는가?

**Observed:** The Gold page is also ranked fifth. Korean output preserves Eleanor and the broad setting but still omits SPOC, QLP, and cleaned counts.

**Cause and impact:** Translation cannot restore facts missing from selected English evidence. The result demonstrates that Korean answer quality depends first on evidence coverage.

**Potential mitigation:** Improve evidence selection before changing the translation model, then verify that named pipelines and numeric values survive translation.

### 8. q-043-ko

**Question:** 다중 소스 distributionally robust graph learning을 평가하기 위해 어떤 실험이 사용되는가?

**Observed:** The Gold page is ranked first. The answer retains synthetic and multi-site fMRI experiments but uses awkward terminology and omits ABIDE I, seven baselines, and the evaluation metrics.

**Cause and impact:** Relevant evidence is available, but sentence compression and translation lose named entities and comparison detail.

**Potential mitigation:** Protect model, dataset, and metric names during translation and add a completeness check for requested comparison dimensions.

### 9. q-041-ko

**Question:** GOAG의 성능은 어떻게 평가되는가?

**Observed:** The Gold page is ranked second. The answer reports MultiDex and 100 runs but omits the three gripper names, efficiency, and diversity.

**Cause and impact:** The answer captures the general protocol but not all requested evaluation dimensions, producing a useful yet partial response.

**Potential mitigation:** Detect list-style evidence and preserve parallel items during sentence selection and answer assembly.

### 10. q-034-en

**Question:** What assumptions are required for the main result about logarithmic Voronoi cells?

**Observed:** The Gold page is ranked second. The answer states the one-dimensional algebraic setting and non-algebraic result but omits the probability simplex and the two explicit constructions.

**Cause and impact:** The answer shifts from the requested assumptions to the broader conclusion. Its claims remain supported, but it does not fully answer the question.

**Potential mitigation:** Classify assumption and limitation questions separately and prefer sentences containing conditions, domains, and construction constraints.

## Priorities after the portfolio release

1. Improve evidence coverage within retrieved pages without changing the frozen Top-5 retrieval claim.
2. Replace the single-score refusal decision with a calibrated multi-signal policy and retest all 60 answerability cases.
3. Preserve named entities and comparison lists in Korean translation only after the evidence-selection gap is addressed.

These items are a documented backlog, not unmeasured performance claims. The frozen metrics remain unchanged.

## Evidence sources

- [`frozen_retrieval_metrics.json`](../data/evaluation/frozen_retrieval_metrics.json) provides Gold-page ranks.
- [`abstention_evaluation.json`](../data/evaluation/abstention_evaluation.json) provides policy scores and expected sufficiency.
- [`grounded_generation_review.json`](../data/evaluation/grounded_generation_review.json) provides reviewed answers, failure labels, and reviewer notes.
- [Evaluation](evaluation.md) defines the benchmark boundary and aggregate results.
