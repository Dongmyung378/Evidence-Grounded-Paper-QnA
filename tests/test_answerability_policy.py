"""Generic topic-debiasing and dev-only threshold fitting regression tests."""

import copy
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from answerability_policy import (aspect_query, evaluate_answerability, load_answerability_config,
                                 question_premise_conflicts, requested_attribute_coverage, score_answerability)
from calibrate_answerability import select_threshold


EVIDENCE = [{"text": "The Orion sensor study uses a neural estimator. The Orion sensor study reports a fixed evaluation. The Orion model is evaluated locally."}]


class Scorer:
    def __init__(self, score):
        self.score = score
        self.queries = []
        self.pairs = []

    def predict(self, pairs, **kwargs):
        self.queries.extend(query for query, text in pairs)
        self.pairs.extend(pairs)
        return np.asarray([self.score] * len(pairs))


def enabled_config(threshold=-3, transform="source_derived_topic_debias_v1"):
    return {"schema_version": 1, "enabled": True,
            "query_transform": transform,
            "comparison": "accept_if_greater_than_or_equal",
            "minimum_aspect_score": threshold,
            "calibration": {"split": "dev", "execution_mode": "real_models"}}


class AnswerabilityPolicyTests(unittest.TestCase):
    def test_energy_unit_request_cannot_be_answered_by_iteration_or_time_counts(self):
        evidence = [{"text": "The optimizer converged in 12 iterations and a fraction of a second."}]
        result = evaluate_answerability("How many joules per request were measured?", evidence, Scorer(20),
                                        enabled_config(transform="source_derived_topic_debias_v2"))
        self.assertTrue(result["abstain"])
        self.assertIn("energy", result["signals"]["missing_requested_attributes"])
        self.assertEqual(result["signals"]["missing_measurement_dimensions"], ["energy"])

    def test_actual_energy_measurement_units_supply_the_requested_dimension(self):
        for text in ("Consumption was 0.42 joules per request.", "Each request consumed 420 mJ."):
            coverage = requested_attribute_coverage("How many joules per request were measured?", [{"text": text}])
            self.assertEqual(coverage["missing_measurement_dimensions"], [])
            self.assertTrue(coverage["requested_attribute_coverage"])

    def test_unit_request_requires_a_measurement_instead_of_abstract_energy(self):
        evidence = [{"text": "The energy function was minimized in 10 iterations."}]
        coverage = requested_attribute_coverage("How many joules were consumed?", evidence)
        self.assertEqual(coverage["missing_requested_attributes"], ["energy"])

    def test_memory_bank_is_not_an_empirical_megabyte_measurement(self):
        coverage = requested_attribute_coverage("What inference memory in megabytes was measured?",
                                               [{"text": "Our method does not require a memory bank."}])
        self.assertEqual(coverage["missing_measurement_dimensions"], ["memory"])

    def test_model_named_mb_does_not_become_a_memory_measurement_request(self):
        coverage = requested_attribute_coverage("How many layers does the MB model use?",
                                               [{"text": "The MB model uses 4 layers."}])
        self.assertEqual(coverage["requested_measurement_dimensions"], [])

    def test_privacy_requests_need_privacy_evidence_in_both_languages(self):
        evidence = [{"text": "Confidence probabilities were calibrated using one scalar parameter."}]
        for question in ("Which privacy safeguards and privacy budgets were used?",
                         "어떤 차등 개인정보 보호 장치와 개인정보 예산이 사용되었나요?"):
            coverage = requested_attribute_coverage(question, evidence)
            self.assertIn("privacy", coverage["missing_requested_attributes"])

    def test_pdf_ligatures_and_hyphenated_privacy_mentions_are_normalized(self):
        for text in ("Conﬁdentiality and data encryption protected records.",
                     "The priv-\nacy budget was reported as epsilon 0.5."):
            coverage = requested_attribute_coverage("Which privacy protections were reported?", [{"text": text}])
            self.assertTrue(coverage["requested_attribute_coverage"])

    def test_required_component_premise_is_rejected_before_calling_a_model(self):
        evidence = [{"text": "Our method learns representations without requiring an external cache."}]
        decision = evaluate_answerability("Because the method requires an external cache, what update rule is used?",
                                          evidence, object(), enabled_config(transform="source_derived_topic_debias_v2"))
        self.assertTrue(decision["abstain"])
        self.assertEqual(decision["reason_code"], "unsupported_question_premise")

    def test_legacy_policy_does_not_enable_new_relation_checks(self):
        evidence = [{"text": "Our method does not require an external cache."}]
        decision = evaluate_answerability("Because an external cache is required, how is it updated?", evidence,
                                          Scorer(1), enabled_config(transform="source_derived_topic_debias_v1"))
        self.assertFalse(decision["abstain"])

    def test_memory_component_contradictions_are_bilingual_and_generic(self):
        evidence = [{"text": "Our framework does not train the model with a memory bank."}]
        for question in ("Because the framework trains with a memory bank, what is its update rule?",
                         "메모리 뱅크로 학습한다는 전제에서 필수 메모리 뱅크의 갱신 규칙은 무엇인가요?"):
            issues = question_premise_conflicts(question, evidence)
            self.assertEqual(issues[0]["concept"], "memory_bank")

    def test_optional_component_is_distinct_from_an_unused_component(self):
        evidence = [{"text": "Our framework operates without requiring an external cache, but may use one optionally."}]
        self.assertEqual(question_premise_conflicts("Because the framework uses an external cache, how is it updated?", evidence), [])
        self.assertTrue(question_premise_conflicts("Because an external cache is required, how is it updated?", evidence))

    def test_neutral_and_negative_component_questions_are_preserved(self):
        evidence = [{"text": "Our method does not require a memory bank."}]
        for question in ("Does the method require a memory bank?", "메모리 뱅크가 필수인가요?",
                         "Why does the method not require a memory bank?",
                         "Because the method does not need a memory bank, how does it train?"):
            self.assertEqual(question_premise_conflicts(question, evidence), [], question)

    def test_unchanged_properties_contradict_positive_gain_premises_in_both_languages(self):
        evidence = [{"text": "The predicted classes remain unchanged. Our method does not affect classification accuracy."}]
        for question in ("Since the method changes predicted classes and improves accuracy, what accuracy gain was reported?",
                         "예측 클래스를 바꿔 분류 정확도를 개선한다는 전제에서 정확도 향상 폭은 얼마인가요?"):
            concepts = {issue["concept"] for issue in question_premise_conflicts(question, evidence)}
            self.assertEqual(concepts, {"predicted_classes", "classification_accuracy"})

    def test_normal_effect_questions_can_be_answered_with_invariance(self):
        evidence = [{"text": "The class prediction remains unchanged. The model's accuracy is unchanged."}]
        for question in ("How does the method affect accuracy and predicted classes?",
                         "Does the method improve accuracy?", "How does the method change accuracy?",
                         "분류 정확도에 미치는 영향은 무엇인가요?", "정확도의 변화 여부를 알려주세요."):
            self.assertEqual(question_premise_conflicts(question, evidence), [], question)

    def test_a_named_baseline_negation_is_not_borrowed_for_the_requested_model(self):
        evidence = [{"text": "Beta does not require an external cache. Alpha uses an external cache."}]
        self.assertEqual(question_premise_conflicts("Because Alpha requires an external cache, how is it updated?", evidence), [])
        evidence = [{"text": "Beta's accuracy remains unchanged. Alpha improves accuracy."}]
        self.assertEqual(question_premise_conflicts("Since Alpha improves accuracy, what accuracy gain is reported?", evidence), [])

    def test_debiased_questions_preserve_english_and_korean_grammar(self):
        english = aspect_query("What method does the Orion sensor study use?", EVIDENCE)
        korean = aspect_query("Orion 센서 연구는 어떤 방법을 사용하나요?", EVIDENCE)
        self.assertEqual(english["natural_query"], "What method does the study use?")
        self.assertEqual(korean["natural_query"], "연구는 어떤 방법을 사용하나요?")
        self.assertNotIn("does use", english["query"])

    def test_names_are_masked_in_both_scoring_views_without_mutating_evidence(self):
        original = copy.deepcopy(EVIDENCE)
        model = Scorer(1)
        signals = score_answerability("What method does the Orion sensor study use?", EVIDENCE, model)
        self.assertEqual(signals["source_anchors_masked"], ["Orion"])
        self.assertTrue(all("Orion" not in question and "Orion" not in text for question, text in model.pairs))
        self.assertEqual(EVIDENCE, original)

    def test_korean_query_masks_repeated_english_title_with_descriptors(self):
        evidence = [{"text": "The Vega sensor study collects observations. The Vega sensor study reports accuracy."}]
        model = Scorer(1)
        signals = score_answerability("Vega 센서 연구는 어떤 방법을 사용하나요?", evidence, model)
        self.assertEqual(signals["removed_anchors"], ["Vega"])
        self.assertTrue(all("Vega" not in text for question, text in model.pairs))

    def test_scientific_acronyms_are_kept_as_requested_constructs(self):
        evidence = [{"text": "ZTX model is evaluated. The ZTX model has a speed measurement."}]
        result = aspect_query("How is the ZTX model evaluated?", evidence)
        self.assertIn("ZTX", result["query"])
        self.assertEqual(result["removed_anchors"], [])

    def test_causal_clause_cannot_replace_requested_result(self):
        english = aspect_query("Which symptoms were relieved by the Orion model?", EVIDENCE)
        korean = aspect_query("Orion 모델의 새로운 치료로 없어진 증상은 무엇인가요?", EVIDENCE)
        self.assertEqual(english["query"], "Which symptoms were relieved")
        self.assertEqual(korean["query"], "없어진 증상은 무엇인가요")

    def test_full_question_and_requested_result_both_require_relevance(self):
        class DistinctScorer:
            def predict(self, pairs, **kwargs):
                return np.asarray([-5 if " by " not in question else 9 for question, text in pairs])

        evidence = [{"text": "Symptoms were measured, but no treatment result was reported."}]
        signals = score_answerability("Which symptoms were relieved by treatment?", evidence, DistinctScorer())
        self.assertEqual(signals["full_question_max_score"], 9)
        self.assertEqual(signals["focused_request_max_score"], -5)
        self.assertEqual(signals["max_aspect_score"], -5)

    def test_explicit_legacy_transform_preserves_original_scoring_contract(self):
        model = Scorer(1)
        signals = score_answerability("What method does the Orion sensor study use?", EVIDENCE, model,
                                     "source_derived_topic_debias_v1")
        self.assertEqual(signals["score_policy"], "legacy_focused_only")
        self.assertEqual(len(model.pairs), 1)
        self.assertIn("Orion", model.pairs[0][1])

    def test_generic_clinical_topic_does_not_supply_requested_symptoms(self):
        evidence = [{"text": "The sensor study is not a clinical study. It classifies electrical inputs."}]
        for question in ("Which symptoms were eliminated by the proposed treatment?",
                         "제안된 치료로 없어진 임상 증상은 무엇인가요?"):
            result = evaluate_answerability(question, evidence, Scorer(20),
                                            enabled_config(transform="source_derived_topic_debias_v2"))
            self.assertTrue(result["abstain"])
            self.assertEqual(result["reason_code"], "requested_attribute_absent")
            self.assertEqual(result["signals"]["missing_requested_attributes"], ["symptoms"])

    def test_real_symptom_mentions_in_either_language_are_not_rejected_by_presence_guard(self):
        for text in ("The measured clinical manifestations include persistent cough and fatigue.",
                     "The treatment reduced cough, dyspnea and fatigue.",
                     "치료 후 기침과 피로 증상의 변화가 측정되었습니다."):
            coverage = requested_attribute_coverage("어떤 임상 증상이 측정되었나요?", [{"text": text}])
            self.assertTrue(coverage["requested_attribute_coverage"])

    def test_population_description_with_count_is_not_a_closed_vocabulary_refusal(self):
        evidence = [{"text": "Twenty healthy adults completed the clinical trial in one laboratory session."}]
        for question in ("How many human participants took part in the study?",
                         "연구에 참여한 인간 참여자는 몇 명인가요?"):
            coverage = requested_attribute_coverage(question, evidence)
            self.assertEqual(coverage["missing_requested_attributes"], [])
            self.assertIn("human_participants", coverage["contextual_requested_attributes"])
            result = evaluate_answerability(question, evidence, Scorer(1),
                                            enabled_config(transform="source_derived_topic_debias_v2"))
            self.assertFalse(result["abstain"])

    def test_human_participation_does_not_require_a_clinical_domain(self):
        evidence = [{"text": "A cohort of 42 students completed a usability experiment."}]
        coverage = requested_attribute_coverage("How many human participants were enrolled?", evidence)
        self.assertTrue(coverage["requested_attribute_coverage"])

    def test_korean_population_count_and_participation_are_recognized(self):
        evidence = [{"text": "건강한 성인 스무 명이 임상 시험을 완료했습니다."}]
        coverage = requested_attribute_coverage("인간 참여자는 몇 명인가요?", evidence)
        self.assertTrue(coverage["requested_attribute_coverage"])

    def test_clinical_context_accepts_unlisted_outcomes_without_symptom_names(self):
        examples = (
            "The subjects experienced diarrhea, vomiting, and dizziness during the clinical trial.",
            "A clinical trial was conducted. The adults reported an unfamiliar bodily sensation.",
            "The patients developed a previously unclassified discomfort after the intervention.",
            "임상 시험에서 성인들이 전에 없던 불편감을 호소했습니다.",
        )
        for text in examples:
            for question in ("Which symptoms were reported?", "어떤 증상이 보고되었나요?"):
                coverage = requested_attribute_coverage(question, [{"text": text}])
                self.assertTrue(coverage["requested_attribute_coverage"], text)
                self.assertIn("symptoms", coverage["contextual_requested_attributes"])

    def test_contextual_presence_does_not_bypass_semantic_relevance_threshold(self):
        evidence = [{"text": "The patients reported an unfamiliar sensation in a clinical trial."}]
        result = evaluate_answerability("Which symptoms were reported?", evidence, Scorer(-10),
                                        enabled_config(transform="source_derived_topic_debias_v2"))
        self.assertTrue(result["abstain"])
        self.assertEqual(result["reason_code"], "requested_aspect_not_supported")

    def test_explicitly_negated_clinical_context_does_not_open_unknown_symptoms(self):
        examples = (
            "There was no clinical study. The adults reported industrial observations.",
            "The work did not include a clinical trial. The subjects experienced a sensor failure.",
            "A clinical trial was not conducted. The students reported a machine calibration error.",
            "임상 연구는 전혀 수행하지 않았습니다. 성인들은 기계 측정 결과를 보고했습니다.",
        )
        for text in examples:
            coverage = requested_attribute_coverage("Which symptoms were reported?", [{"text": text}])
            self.assertEqual(coverage["missing_requested_attributes"], ["symptoms"], text)

    def test_clinical_context_is_not_borrowed_across_unrelated_evidence_items(self):
        evidence = [{"text": "A clinical trial was conducted."},
                    {"text": "The adults reported industrial sensor observations."}]
        coverage = requested_attribute_coverage("Which symptoms were reported?", evidence)
        self.assertEqual(coverage["missing_requested_attributes"], ["symptoms"])

    def test_industrial_counts_do_not_imply_human_participation(self):
        evidence = [{"text": "The classifier was tested on twenty industrial sensor observations."}]
        coverage = requested_attribute_coverage("How many human participants took part?", evidence)
        self.assertEqual(coverage["missing_requested_attributes"], ["human_participants"])

    def test_legal_consent_and_privacy_sources_pass_presence_guard(self):
        evidence = [{"text": "Informed consent and privacy protections were required before participation."}]
        for question in ("What legal safeguards were deployed?", "어떤 법적 보호 장치를 적용했나요?"):
            result = evaluate_answerability(question, evidence, Scorer(1),
                                            enabled_config(transform="source_derived_topic_debias_v2"))
            self.assertFalse(result["abstain"])

    def test_unknown_attributes_are_not_guessed_by_presence_guard(self):
        result = requested_attribute_coverage("How is the new spectral transform derived?", EVIDENCE)
        self.assertEqual(result["requested_attributes"], [])
        self.assertTrue(result["requested_attribute_coverage"])

    def test_all_requested_known_attributes_need_source_presence(self):
        result = requested_attribute_coverage("What memory and energy usage were reported?",
                                             [{"text": "Memory usage was 32 MB."}])
        self.assertEqual(result["missing_requested_attributes"], ["energy"])

    def test_funding_attribute_survives_removal_of_repeated_source_title(self):
        result = aspect_query("Which institute funded the Orion sensor study?", EVIDENCE)
        self.assertNotIn("Orion", result["query"])
        self.assertNotIn("sensor", result["query"])
        self.assertIn("institute", result["query"])
        self.assertIn("funded", result["query"])
        self.assertEqual(result["removed_anchors"], ["Orion"])

    def test_human_participant_and_legal_attributes_are_preserved(self):
        participants = aspect_query("How many human participants were enrolled in the Orion sensor study?", EVIDENCE)
        safeguards = aspect_query("What legal safeguards were deployed by the Orion sensor study?", EVIDENCE)
        self.assertIn("human participants", participants["query"])
        self.assertIn("legal safeguards", safeguards["query"])

    def test_korean_title_removal_preserves_requested_funding(self):
        result = aspect_query("Orion 센서 연구에 자금을 지원한 기관은 어디인가요?", EVIDENCE)
        self.assertNotIn("Orion", result["query"])
        self.assertNotIn("센서", result["query"])
        self.assertIn("자금을 지원한 기관", result["query"])

    def test_single_method_request_keeps_requested_method(self):
        english = aspect_query("What method does the Orion sensor study use?", EVIDENCE)
        korean = aspect_query("Orion 센서 연구는 어떤 방법을 사용하나요?", EVIDENCE)
        self.assertIn("method", english["query"])
        self.assertIn("방법", korean["query"])

    def test_name_not_present_in_source_is_not_removed(self):
        result = aspect_query("What accuracy did the Nova model report?", EVIDENCE)
        self.assertIn("Nova", result["query"])
        self.assertEqual(result["removed_anchors"], [])

    def test_disabled_gate_never_calls_a_model(self):
        config = enabled_config()
        config["enabled"] = False
        result = evaluate_answerability("Who funded the project?", EVIDENCE, object(), config)
        self.assertFalse(result["abstain"])
        self.assertEqual(result["reason_code"], "policy_disabled")

    def test_threshold_uses_requested_aspect_score(self):
        model = Scorer(-4)
        result = evaluate_answerability("Which institute funded the Orion sensor study?", EVIDENCE, model, enabled_config())
        self.assertTrue(result["abstain"])
        self.assertEqual(result["reason_code"], "requested_aspect_not_supported")
        self.assertNotIn("Orion", model.queries[0])
        self.assertIn("funded", model.queries[0])

    def test_threshold_boundary_is_inclusive(self):
        result = evaluate_answerability("What accuracy was reported?", EVIDENCE, Scorer(-3), enabled_config())
        self.assertFalse(result["abstain"])

    def test_missing_evidence_is_refused(self):
        result = evaluate_answerability("What accuracy was reported?", [], object(), enabled_config())
        self.assertTrue(result["abstain"])
        self.assertEqual(result["reason_code"], "no_aspect_evidence")

    def test_nonfinite_model_scores_cannot_be_accepted(self):
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            score_answerability("What accuracy was reported?", EVIDENCE, Scorer(float("nan")))

    def test_enabled_policy_requires_dev_real_model_provenance(self):
        config = enabled_config()
        config["calibration"]["split"] = "holdout"
        with TemporaryDirectory() as name:
            path = Path(name) / "policy.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dev-only"):
                load_answerability_config(path)

    def test_runtime_never_reads_calibration_report_or_labels(self):
        config = enabled_config()
        config["calibration"]["report"] = "does-not-exist/calibration-and-labels.json"
        with TemporaryDirectory() as name:
            path = Path(name) / "runtime-policy.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            runtime_config = load_answerability_config(path)
            result = evaluate_answerability("What accuracy was reported?", EVIDENCE,
                                            Scorer(1), runtime_config)
        self.assertFalse(result["abstain"])
        self.assertEqual(result["signals"]["max_aspect_score"], 1)


class CalibrationTests(unittest.TestCase):
    @staticmethod
    def rows():
        return [{"meaning_id": group, "language": language, "split": "dev", "answerability": label,
                 "signals": {"max_aspect_score": score}}
                for group, label, scores in [("positive", "answerable", (-1, -2)),
                                              ("negative", "unsupported", (-5, -6))]
                for language, score in zip(("en", "ko"), scores)]

    def test_threshold_retains_both_languages_in_a_meaning_group(self):
        result = select_threshold(self.rows())
        self.assertEqual(result["selected"]["answerable_group_retention"], 1)
        self.assertEqual(result["selected"]["unsupported_group_refusal_recall"], 1)
        self.assertLessEqual(result["selected"]["threshold"], -2)
        self.assertGreater(result["selected"]["threshold"], -5)

    def test_overlap_is_reported_without_false_separation_claim(self):
        rows = self.rows()
        rows[-1]["signals"]["max_aspect_score"] = 1
        result = select_threshold(rows)
        self.assertFalse(result["fully_separable_by_relevance_threshold"])
        self.assertEqual(result["selected"]["answerable_group_retention"], 1)
        self.assertEqual(result["selected"]["unsupported_group_refusal_recall"], 0)

    def test_holdout_rows_are_rejected(self):
        rows = self.rows()
        rows[0]["split"] = "holdout"
        with self.assertRaisesRegex(ValueError, "dev rows only"):
            select_threshold(rows)

    def test_additional_dev_retention_constraints_prevent_threshold_regression(self):
        constraint = {"case_id": "broad-supported-request", "split": "dev", "answerability": "answerable",
                      "signals": {"max_aspect_score": -4.5}}
        result = select_threshold(self.rows(), retention_rows=[constraint])
        self.assertLessEqual(result["selected"]["threshold"], -4.5)
        self.assertTrue(result["selected"]["additional_dev_retention"])
        self.assertEqual(result["additional_answerable_dev_constraints"], 1)

    def test_retention_constraints_cannot_contain_holdout_or_negative_labels(self):
        for split, answerability in (("holdout", "answerable"), ("dev", "unsupported")):
            with self.assertRaises(ValueError):
                select_threshold(self.rows(), retention_rows=[{"split": split, "answerability": answerability}])

    def test_combined_guard_success_is_not_claimed_as_threshold_separation(self):
        rows = self.rows()
        for row in rows:
            if row["answerability"] != "answerable":
                row["signals"]["max_aspect_score"] = 9
                row["signals"]["missing_requested_attributes"] = ["memory"]
        result = select_threshold(rows)
        self.assertTrue(result["fully_separable_by_combined_policy"])
        self.assertFalse(result["fully_separable_by_relevance_threshold"])

    def test_missing_evidence_is_not_accepted_during_calibration(self):
        rows = self.rows()
        for row in rows:
            if row["answerability"] != "answerable":
                row["signals"].update(max_aspect_score=None, evidence_count=0)
        result = select_threshold(rows)
        self.assertEqual(result["selected"]["unsupported_group_refusal_recall"], 1)


if __name__ == "__main__":
    unittest.main()
