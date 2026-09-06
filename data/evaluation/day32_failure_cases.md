# Day 32 Strict Failure Cases

These are strict `fail` judgments from the balanced 20-question manual review.
A `partial` answer is tracked in the full review but is not counted toward the
roadmap requirement of at least five recorded failures.

- Strict failures: **15/20**
- False abstentions: **7/20**
- Source: `data/evaluation/day32_manual_review.json`

## 1. q-001-en — limitation

- Paper / language: `paper-001` / `en`
- Question: What is the main problem addressed in this paper?
- Expected: The paper addresses the lack of a structural guarantee that science signals and radio-frequency interference can be separated by rank selection alone in the single-epoch setting.
- Actual: The answer cannot be verified from the provided paper evidence.
- Gold page / cited pages: 2 / none
- Failure types: `false_abstention, threshold_false_negative`
- Review: Gold evidence directly answers the question, but the calibrated pre-generation threshold refused it and returned no citation.

## 2. q-001-ko — limitation

- Paper / language: `paper-001` / `ko`
- Question: 이 논문에서 다루는 주요 문제는 무엇인가?
- Expected: 이 논문은 단일 시점 환경에서 rank 선택만으로 과학 신호와 전파 간섭을 분리할 수 있다는 보장이 부족한 문제를 다룬다.
- Actual: 제공된 논문 근거에서 답을 확인할 수 없습니다.
- Gold page / cited pages: 2 / none
- Failure types: `false_abstention, threshold_false_negative`
- Review: Gold 근거가 직접 존재하지만 사전 임계값이 답변 가능 질문을 거부했고 인용도 반환하지 않았다.

## 3. q-045-ko — comparison

- Paper / language: `paper-002` / `ko`
- Question: 모델 폭이 달라질 때 학습률 전이와 관련된 주요 결과는 무엇인가?
- Expected: 주요 결과는 제안된 μP adaptation을 사용하면 폭이 확장된 MoE 모델 사이에서 최적 학습률이 일관되게 전이된다는 것이다.
- Actual: 학습률 전이와 관련된 주요 결과는 모델의 학습 성능과 학습률 조정에 대한 효과적인 학습률 전이를 제공합니다.
- Gold page / cited pages: 1 / 4, 16, 5, 6, 1
- Failure types: `vague_non_answer, missing_comparison_result`
- Review: 근거는 관련되지만 답변이 순환적이고 모호하며, muP 적용 시 폭이 다른 모델 사이에서 최적 학습률이 일관되게 전이된다는 결과를 말하지 않는다.

## 4. q-007-ko — factual

- Paper / language: `paper-003` / `ko`
- Question: 이 논문이 텍스트 요약을 검토하면서 중점적으로 다루는 내용은 무엇인가?
- Expected: 이 리뷰는 자동 텍스트 요약에서 Transformer와 대규모 언어 모델 접근법, 특히 BERT, RoBERTa, BART에 중점을 둔다.
- Actual: 네, 제공된 논문 근거에서 답을 확인할 수 있습니다.
- Gold page / cited pages: 3 / 1, 2, 8, 3, 2
- Failure types: `nonresponsive_answer, missing_factual_content`
- Review: '근거에서 확인할 수 있다'는 말만 하고 리뷰의 초점이나 세 모델을 답하지 않아 실질적인 답변과 이를 뒷받침하는 인용 연결이 없다.

## 5. q-031-en — comparison

- Paper / language: `paper-004` / `en`
- Question: What trade-offs between inference speed, memory usage, and accuracy are discussed?
- Expected: INT8 inference improves throughput and memory efficiency on CPUs, while the reported trade-off is a possible accuracy cost that is negligible or absent in the evaluated models.
- Actual: The answer cannot be verified from the provided paper evidence.
- Gold page / cited pages: 1 / none
- Failure types: `false_abstention, threshold_false_negative, comparison_omission`
- Review: The paper evidence supports the speed-memory-accuracy trade-off, but the pre-generation policy refused the answer.

## 6. q-048-ko — numeric

- Paper / language: `paper-004` / `ko`
- Question: 서버 환경의 INT8 추론에 PyTorch 네이티브 스택을 사용하는 것에 대한 주요 결론은 무엇인가?
- Expected: 이 작업은 native PyTorch와 TorchAO 도구를 통한 즉시 사용 가능한 배포를 가능하게 하며 최대 5.8배의 throughput 향상을 제공한다.
- Actual: 제공된 논문 근거에서 답을 확인할 수 없습니다.
- Gold page / cited pages: 1 / none
- Failure types: `false_abstention, structured_output_validation_failure, numeric_omission`
- Review: 모델 출력 검증이 모두 실패해 안전 거부로 전환되었고, 핵심 수치인 최대 5.8배 향상을 답하지 못했다.

## 7. q-015-en — numeric

- Paper / language: `paper-005` / `en`
- Question: What evidence shows that the proposed approach improves continual anomaly detection?
- Expected: ContCore achieves state-of-the-art performance across 11 task schedules on MVTecAD and VisA and remains effective in online continual anomaly detection settings.
- Actual: Continued greedy sampling effectively preserves representativeness under strict memory limits.
- Gold page / cited pages: 14 / 1
- Failure types: `wrong_question_focus, numeric_omission, evidence_result_omission`
- Review: The response describes a method property instead of the requested performance evidence and omits 11 schedules, MVTecAD, VisA, and the online setting.

## 8. q-049-ko — limitation

- Paper / language: `paper-005` / `ko`
- Question: 지속적 이상 탐지에서 메모리 효율성에 대한 주요 결론은 무엇인가?
- Expected: ContCore는 고정 메모리를 유지하고 이론적 근사 보장을 제공하며 지속적 task가 늘어날 때 성능이 급격히 무너지지 않고 점진적으로 저하된다.
- Actual: 제공된 논문 근거에서 답을 확인할 수 없습니다.
- Gold page / cited pages: 14 / none
- Failure types: `false_abstention, model_false_refusal, limitation_omission`
- Review: 답변 가능한 메모리 효율 결론이 검색되었지만 모델이 insufficient로 잘못 판단해 고정 메모리, 근사 보장, 점진적 저하를 모두 누락했다.

## 9. q-017-ko — factual

- Paper / language: `paper-006` / `ko`
- Question: 이 논문은 표현 학습과 과정 추론을 어떻게 연결하는가?
- Expected: 논문은 sensing pathway, 공간·시간적 support, 목표 변수, 추적 가능한 불확실성을 정렬하는 observation-to-inference hierarchy를 통해 표현 학습과 과정 추론을 연결한다.
- Actual: EO 데이터를 통해 지속적인 학습 및 추론의 연결 방법에 대한 이해는 무엇인가요?
- Gold page / cited pages: 1 / 1, 1, 26, 2, 2
- Failure types: `malformed_non_answer, question_repetition`
- Review: 답변 대신 새로운 의문문을 생성했으며 observation-to-inference hierarchy와 핵심 정렬 요소를 설명하지 않는다.

## 10. q-034-ko — limitation

- Paper / language: `paper-007` / `ko`
- Question: 로그 Voronoi cell에 대한 주요 결과를 위해 어떤 가정이 필요한가?
- Expected: 주요 결과는 probability simplex의 1차원 algebraic model을 대상으로 하며, 두 직선의 합과 cuspidal curve에 대한 명시적 구성을 사용한다.
- Actual: 근거에서 직접 확인되는 내용을 한국어로 간결하게 답합니다.
- Gold page / cited pages: 1 / 2, 4, 1, 10, 1
- Failure types: `placeholder_non_answer, missing_assumptions`
- Review: 프롬프트 지시처럼 보이는 문장만 출력하고 1차원 algebraic model, probability simplex, 명시적 구성이라는 가정을 전혀 답하지 않는다.

## 11. q-041-en — numeric

- Paper / language: `paper-008` / `en`
- Question: How is the performance of GOAG evaluated?
- Expected: GOAG is evaluated on three dexterous grippers using repeated inference runs and the MultiDex test set, with success rate, efficiency, and diversity reported.
- Actual: GOAG achieves superior performance on the Multidex dataset compared to other baselines.
- Gold page / cited pages: 6 / 7, 6
- Failure types: `wrong_question_focus, numeric_omission, protocol_omission`
- Review: The response gives a broad performance conclusion rather than how GOAG was evaluated, omitting three grippers, repeated runs, and the success/efficiency/diversity criteria.

## 12. q-041-ko — comparison

- Paper / language: `paper-008` / `ko`
- Question: GOAG의 성능은 어떻게 평가되는가?
- Expected: GOAG는 세 종류의 다관절 그리퍼에서 반복 추론과 MultiDex 테스트 세트를 사용해 평가되며, 성공률·효율성·다양성을 보고한다.
- Actual: 제공된 논문 근거에서 답을 확인할 수 없습니다.
- Gold page / cited pages: 6 / none
- Failure types: `false_abstention, model_false_refusal, comparison_omission`
- Review: 검색 근거가 있는 평가 절차 질문을 모델이 insufficient로 잘못 거부했다.

## 13. q-042-en — numeric

- Paper / language: `paper-009` / `en`
- Question: How is the Microlensify classifier evaluated on TESS light curves?
- Expected: Microlensify is evaluated on TESS Sector 12 light curves from the Eleanor, SPOC, and QLP pipelines, and the paper reports candidate counts after applying cleaning thresholds.
- Actual: The model was trained on approximately 17,000 simulated microlensing events and a similar number (∼17,500) of real non-microlensing light curves from Sector 12 extracted from Eleanor pipeline. This suggests that the model has been extensively tested and validated against real data sets.
- Gold page / cited pages: 23 / 22, 15, 4, 14, 23
- Failure types: `wrong_question_focus, incomplete_evaluation, unsupported_inference`
- Review: The answer substitutes training-set counts for the requested evaluation protocol, omits the Eleanor/SPOC/QLP coverage and post-cleaning candidate counts, and adds an overly broad validation inference.

## 14. q-042-ko — numeric

- Paper / language: `paper-009` / `ko`
- Question: Microlensify 분류기는 TESS 광도 곡선에서 어떻게 평가되는가?
- Expected: Microlensify는 Eleanor, SPOC, QLP 파이프라인의 TESS Sector 12 광도 곡선으로 평가되며, 논문은 정제 기준 적용 후의 후보 수를 보고한다.
- Actual: Microlensify는 TESS 광도 곡선에서 평가되는 방법으로, TESS 광도 곡선에서의 단일Lens 미세먼지를 파악하는 방법입니다.
- Gold page / cited pages: 23 / 22, 4
- Failure types: `wrong_question_focus, mistranslation, numeric_omission`
- Review: 평가 방법을 답하지 않고 microlensing을 '미세먼지'로 잘못 표현했으며 Sector 12, 세 파이프라인, 정제 후 후보 수를 누락했다.

## 15. q-043-ko — comparison

- Paper / language: `paper-010` / `ko`
- Question: 다중 소스 distributionally robust graph learning을 평가하기 위해 어떤 실험이 사용되는가?
- Expected: MS-WDRO는 synthetic benchmark와 multi-site ABIDE I neuroimaging dataset에서 7개 baseline과 비교되며 graph recovery, sample efficiency, diagnostic utility로 평가된다.
- Actual: 제공된 논문 근거에서 답을 확인할 수 없습니다.
- Gold page / cited pages: 19 / none
- Failure types: `false_abstention, structured_output_validation_failure, comparison_omission`
- Review: 유효한 JSON 생성 실패로 안전 거부되어 synthetic/ABIDE I 실험, 7개 baseline, 평가 기준을 모두 답하지 못했다.
