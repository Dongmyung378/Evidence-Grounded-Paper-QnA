"""Generic completeness, source boundaries and translation fidelity checks."""

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evidence_selector import question_kinds, select_evidence
from grounded_answer_contract import AnswerValidationError
from grounded_generation import (
    _clean_extractive_sentence,
    generate_evidence_first_answer,
    load_answer_generation_config,
    _question_premise_issues,
    _normalized_numbers,
    _validate_translation_batch,
    validate_plain_answer,
    select_grounded_sentences,
    split_evidence_sentences,
)


def evidence(text, rank=1, page=1):
    return {
        "evidence_id": f"ev-{rank:02d}-fixture-p{page:03d}",
        "page": page,
        "section": "Experiments",
        "text": text,
    }


def candidate(text, rank, page=1):
    return {
        "rank": rank,
        "score": float(10 - rank),
        "chunk": {
            "paper_id": "fixture",
            "chunk_id": f"fixture-p{page:03d}-c{rank:03d}",
            "page": page,
            "section": "Experiments",
            "chunk_index": rank,
            "text": text,
        },
    }


class Selector:
    def __init__(self, preferred=""):
        self.preferred = preferred

    def predict(self, pairs, **kwargs):
        return np.asarray([10.0 if self.preferred and self.preferred in text else 0.0 for _, text in pairs])


class Translator:
    def __init__(self, outputs):
        self.outputs = outputs
        self.inputs = None

    def translate(self, inputs):
        self.inputs = inputs
        return self.outputs


class AnswerCompletenessTests(unittest.TestCase):
    def test_general_topic_question_keeps_a_direct_study_statement(self):
        source = evidence("This paper studies evidence-grounded question answering with traceable source sentences.")
        for question in ("What does the paper study?", "이 논문은 무엇을 연구하는가?"):
            with self.subTest(question=question):
                self.assertNotIn("definition", question_kinds(question))
                selected = select_grounded_sentences(question, [source], Selector(), self.config)
                self.assertEqual([item["text"] for item in selected], [source["text"]])
        english = generate_evidence_first_answer(None, Selector(), "What does the paper study?", [source], config=self.config)
        korean = generate_evidence_first_answer(Translator(["이 논문은 추적 가능한 원문 문장으로 근거 기반 질의응답을 연구합니다."]),
                                               Selector(), "이 논문은 무엇을 연구하는가?", [source], config=self.config)
        self.assertEqual(english["response"]["sufficiency"], "sufficient")
        self.assertEqual(korean["response"]["sufficiency"], "sufficient")

    def test_framework_task_prose_with_inline_set_indices_survives_equation_filter(self):
        introduction = "We introduce the PairSense framework for learning reusable sensor representations."
        task = "Given a set {xk} including a positive pair of examples xi and xj, the prediction task aims to identify xj in {xk}k̸=i for a given xi."
        duplicate = "This paper presents PairSense as a framework for learning sensor representations."
        selected = select_grounded_sentences(
            "What is the main framework and what prediction task does it use?",
            [evidence(introduction + " " + duplicate + " " + task)], Selector("This paper"), self.config,
        )
        self.assertEqual(selected, [])
        # Source-prose parsing and final math identity validation remain
        # independent from the composite answer's plain-source requirement.
        selected = split_evidence_sentences([evidence(task)])
        self.assertEqual(split_evidence_sentences([evidence("The formula is q = exp(z)/sum(exp(z)).")]), [])
        response = validate_plain_answer(task, "What prediction task is used?", selected, "en", self.config)
        self.assertEqual(response["sufficiency"], "sufficient")
        translated = "양의 예제 쌍 xi와 xj를 포함하는 주어진 집합에서 예측 과제는 xi에 대해 k̸=i인 집합의 xj를 식별하는 것을 목표로 합니다."
        _validate_translation_batch([task], [translated])
        self.assertEqual(validate_plain_answer(translated, "어떤 예측 과제를 사용하나요?", selected, "ko", self.config)["sufficiency"], "sufficient")
        with self.assertRaisesRegex(AnswerValidationError, "formula"):
            validate_plain_answer(translated.replace("k̸=i", "k=i"), "어떤 예측 과제를 사용하나요?", selected, "ko", self.config)

    def test_framework_composite_request_without_task_source_does_not_repeat_introduction(self):
        selected = select_grounded_sentences(
            "What is the main framework and what prediction task does it use?",
            [evidence("We introduce the PairSense framework for learning useful sensor representations. "
                      "This paper presents PairSense as a framework for sensor representation learning.")],
            Selector(), self.config,
        )
        self.assertEqual(selected, [])

    def test_composite_task_prefers_complete_plain_source_over_higher_scored_math(self):
        introduction = "We introduce the PairSense framework for learning reusable sensor representations."
        symbolic = "Given paired xi and xj, the prediction task aims to identify xj in the set {xk}k̸=i for a given xi."
        plain = ("PairSense learns representations by maximizing agreement between independently augmented views "
                 "of the same sensor observation via a contrastive loss in the latent space.")
        for question in ("What is the main framework and what prediction task does it use?",
                         "제안한 주요 프레임워크는 무엇이며 어떤 예측 과제를 사용하나요?"):
            with self.subTest(question=question):
                selected = select_grounded_sentences(
                    question, [evidence(introduction + " " + symbolic), evidence(plain, rank=2)],
                    Selector("prediction task"), self.config,
                )
                self.assertEqual([item["text"] for item in selected], [introduction, plain])
                self.assertIn("of the same sensor observation via a contrastive loss", selected[1]["text"])
        _validate_translation_batch([symbolic], [symbolic])
        with self.assertRaisesRegex(AnswerValidationError, "formula"):
            validate_plain_answer(symbolic.replace("k̸=i", "k=i"), "What prediction task is used?",
                                  split_evidence_sentences([evidence(symbolic)]), "en", self.config)

    def test_task_context_supplement_preserves_top_five_and_full_pool_source(self):
        question = "What is the main framework and what prediction task does it use?"
        items = [candidate("We introduce the PairSense framework for learning reusable sensor representations.", 1)]
        items += [candidate(f"The network setting {rank} is evaluated using a distinct device configuration.",
                            rank, page=rank) for rank in range(2, 6)]
        prefix = "The experimental context is documented for reproducibility. " * 6
        plain = ("PairSense learns representations by maximizing agreement between independently augmented views "
                 "of the same sensor observation via a contrastive loss in the latent space.")
        extra = candidate(prefix + plain, 6, page=6)
        extra["reranker_score"] = 7.25
        items.append(extra)
        original = copy.deepcopy(items)
        baseline, _ = select_evidence(items, top_k=5)
        selected, metadata = select_evidence(items, top_k=5, question=question)
        self.assertEqual(selected, baseline)
        self.assertEqual(items, original)
        supplements = metadata["answer_context_supplements"]
        self.assertEqual(len(supplements), 1)
        self.assertEqual(supplements[0]["text"], prefix + plain)
        self.assertNotIn(plain, extra["chunk"]["text"][:300])
        self.assertEqual(supplements[0]["scores"]["reranker"], 7.25)
        origin = metadata["answer_context_supplement_origins"][0]
        self.assertEqual(origin["candidate_rank"], 6)
        self.assertEqual(origin["chunk_id"], extra["chunk"]["chunk_id"])
        self.assertEqual(origin["pool"], "existing_reranked_candidates")
        self.assertEqual(origin["source"], "full_chunk")
        merged = selected + supplements
        answer_sources = select_grounded_sentences(question, merged, Selector(), self.config)
        self.assertIn(plain, [item["text"] for item in answer_sources])
        self.assertEqual(next(item for item in answer_sources if item["text"] == plain)["evidence_id"],
                         supplements[0]["evidence_id"])

    def test_task_supplement_requires_plain_complete_source_from_same_paper_pool(self):
        question = "What is the main framework and what prediction task does it use?"
        introduction = candidate("We introduce the PairSense framework for learning reusable sensor representations.", 1)
        symbolic = candidate("The prediction task aims to identify xj in the set {xk}k̸=i for a given xi.", 2, page=2)
        incomplete = candidate("The prediction task aims to identify matching augmented sensor observations", 3, page=3)
        foreign = candidate("The prediction task aims to identify matching views of the same observation.", 4, page=4)
        foreign["chunk"]["paper_id"] = "other-paper"
        for pool in ([introduction], [introduction, symbolic], [introduction, incomplete], [introduction, foreign]):
            # Keep the original primary retrieval width at one for this scope check.
            selected, metadata = select_evidence(pool, top_k=1, question=question)
            self.assertEqual(metadata["answer_context_supplements"], [])
            self.assertEqual(select_grounded_sentences(question, selected, Selector(), self.config), [])

    def test_composite_list_answer_respects_the_five_citation_source_budget(self):
        question = "List the components of the main framework and its prediction task"
        components = [
            "We introduce a framework that comprises a sensor encoder and a recurrent state estimator.",
            "We propose a framework that comprises a convolutional encoder and a masked reconstruction decoder.",
            "We present a framework that comprises a signal filter and a channel normalization module.",
            "We introduce a framework that comprises a graph processor and a temporal attention operator.",
            "We propose a framework that comprises a frequency transform and a sequence aggregation module.",
        ]
        task = "The prediction task consists of identifying matching components from independently transformed observations."
        items = [candidate(text, rank, page=rank) for rank, text in enumerate(components + [task], 1)]
        primary, metadata = select_evidence(items, top_k=5, question=question)
        self.assertEqual(len(primary), 5)
        self.assertEqual(len(metadata["answer_context_supplements"]), 1)
        context = primary + metadata["answer_context_supplements"]
        selected = select_grounded_sentences(question, context, Selector(), self.config)
        self.assertEqual(len({item["evidence_id"] for item in selected}), 5)
        self.assertIn(task, [item["text"] for item in selected])
        result = generate_evidence_first_answer(None, Selector(), question, context, config=self.config)
        self.assertEqual(result["response"]["sufficiency"], "sufficient")
        self.assertFalse(result["fallback_used"])
        self.assertEqual(len(result["response"]["evidence_ids"]), 5)
        # If a complete mandatory facet cannot fit, retain refusal rather than
        # validate an answer made solely from framework introductions.
        limited = copy.deepcopy(self.config)
        limited["sentence_selector"]["max_context_chars"] = len(components[0])
        self.assertEqual(select_grounded_sentences(question, context, Selector(), limited), [])

    def test_task_supplement_api_citations_and_context_counts_are_consistent(self):
        from qna_pipeline import GroundedQAPipeline

        question = "What is the main framework and what prediction task does it use?"
        introduction = "We introduce the PairSense framework for learning reusable sensor representations."
        plain = ("PairSense learns representations by maximizing agreement between differently augmented views "
                 "of the same observation via a contrastive loss in the latent space.")
        items = [candidate(introduction, 1)] + [
            candidate(f"The evaluation setting {rank} uses a distinct sensor configuration.", rank, page=rank)
            for rank in range(2, 6)
        ] + [candidate(plain, 6, page=6)]
        primary, selection = select_evidence(items, top_k=5, question=question)
        retrieval_result = {
            "evidence": primary, "evidence_selection": selection,
            "candidates": [{"chunk_id": item["chunk"]["chunk_id"]} for item in items],
            "production": {"config_fingerprint": "test"},
        }
        original_primary = copy.deepcopy(primary)

        class Retrieval:
            def run(self, *args):
                return retrieval_result

        translator = Translator([
            "우리는 재사용 가능한 센서 표현을 학습하는 PairSense 프레임워크를 소개합니다.",
            "PairSense는 잠재 공간에서 대조 손실을 통해 동일한 관측의 다르게 증강된 뷰 사이의 일치를 극대화하여 표현을 학습합니다.",
        ])
        pipeline = GroundedQAPipeline(retrieval=Retrieval(), sentence_selector=Selector(), translator=translator,
                                      answer_strategy="evidence_first", enable_abstention=False)
        for query in (question, "제안한 주요 프레임워크는 무엇이며 어떤 예측 과제를 사용하나요?"):
            with self.subTest(query=query):
                result = pipeline.ask(query, "fixture", include_debug=True)
                self.assertEqual(result["response"]["sufficiency"], "sufficient")
                self.assertEqual(result["retrieved_evidence"][:5], original_primary)
                self.assertEqual(primary, original_primary)
                self.assertEqual(len(result["retrieved_evidence"]), 6)
                supplement = selection["answer_context_supplements"][0]
                self.assertIn(supplement["evidence_id"], result["response"]["evidence_ids"])
                self.assertIn(supplement, result["cited_evidence"])
                self.assertEqual(supplement["locator"]["chunk_id"], items[5]["chunk"]["chunk_id"])
                self.assertLessEqual(len(result["cited_evidence"]), 5)
                context = result["pipeline"]["evidence_context"]
                self.assertEqual(context["primary_retrieval_count"], 5)
                self.assertEqual(context["primary_retrieval_evidence_ids"], [item["evidence_id"] for item in primary])
                self.assertEqual(context["supplement_count"], 1)
                self.assertEqual(context["answer_context_count"], 6)
                self.assertEqual(context["supplement_origins"], selection["answer_context_supplement_origins"])
        self.assertEqual(translator.inputs, [introduction, plain])
        # A preview or a source from outside the reranked pool cannot authorize a citation.
        retrieval_result["candidates"] = retrieval_result["candidates"][:5]
        with self.assertRaisesRegex(ValueError, "reranked pool"):
            pipeline.ask(question, "fixture")
        selection["answer_context_supplements"] *= 2
        with self.assertRaisesRegex(ValueError, "bounded supplemental source limit"):
            pipeline.ask(question, "fixture")

    def test_shortcut_question_cannot_answer_with_nominal_histogram_caption(self):
        caption = "Histograms of signal intensities for different crops of two separate observations."
        selected = select_grounded_sentences(
            "Why does cropping permit a shortcut?", [evidence(caption)], Selector(), self.config,
        )
        self.assertEqual(selected, [])

    def test_causal_figure_claim_connects_histogram_antecedent_to_shortcut(self):
        cause = "We conjecture that cropped views share a common background distribution."
        histogram = "Figure 7 shows that intensity histograms alone suffice to distinguish the original observations."
        shortcut = "Neural networks may exploit this shortcut to solve the predictive task."
        remedy = "Therefore, we independently perturb intensities to remove the shortcut from the cropped views."
        selected = select_grounded_sentences(
            "Why does cropping permit a shortcut and how is it removed?",
            [evidence(" ".join((cause, histogram, shortcut, remedy)))], Selector("Neural networks"), self.config,
        )
        self.assertEqual({item["text"] for item in selected}, {cause, histogram, shortcut, remedy})

    def test_numbered_fused_heading_keeps_two_separate_scientific_sentences(self):
        heading = "A nonlinear readout head improves the representation quality of the layer before it"
        sentence = "We then study the importance of including a readout head in the encoder."
        records = split_evidence_sentences([evidence("4.3. " + heading + " " + sentence)])
        self.assertEqual([item["text"] for item in records], [heading + ".", sentence])
        ordinary = "The layer before it is transformed by the shared encoder during evaluation."
        self.assertEqual([item["text"] for item in split_evidence_sentences([evidence(ordinary)])], [ordinary])

    def test_augmentation_list_excludes_operator_sampling_and_figure_caption(self):
        operations = "We sequentially apply three augmentations: random masking, time shifting, and additive noise."
        sampled = "Two separate augmentation operators are sampled from the same family and applied to each example."
        caption = "Linear evaluation under individual transformations, applied to one branch of the encoder."
        selected = select_grounded_sentences(
            "List the three default augmentations", [evidence(operations + " " + sampled + " " + caption)],
            Selector("Two separate"), self.config,
        )
        self.assertEqual([item["text"] for item in selected], [operations])

    def test_probability_change_facet_precedes_unrelated_accuracy_conclusion(self):
        labels = "Because the transformation preserves the maximum, class predictions remain unchanged."
        accuracy = "In other words, the transformation does not affect the measured accuracy."
        probabilities = "Each bin is reliable even though all the probabilities were modified by a single parameter."
        unrelated = "Future work will explain why these trends affect calibration while improving accuracy."
        selected = select_grounded_sentences(
            "How does calibration affect confidence probabilities compared with predicted labels and accuracy?",
            [evidence(labels + " " + accuracy), evidence(probabilities + " " + unrelated, rank=2)],
            Selector("Future work"), self.config,
        )
        self.assertIn(probabilities, [item["text"] for item in selected[:3]])
        self.assertIn(labels, [item["text"] for item in selected[:3]])

    def test_requested_projection_subject_excludes_high_scoring_accuracy_results(self):
        projection = "The nonlinear projection head improves representation quality compared with a linear projection head."
        selected = select_grounded_sentences(
            "비선형 투영 헤드에 관한 제거 실험의 결과는 무엇인가요?",
            [evidence("The model achieves 81% accuracy under a linear evaluation protocol. " + projection)],
            Selector("81%"), self.config,
        )
        self.assertEqual([item["text"] for item in selected], [projection])

    def test_requested_projection_subject_without_source_is_not_replaced_by_accuracy(self):
        selected = select_grounded_sentences(
            "What does the nonlinear projection head do?",
            [evidence("The model achieves 81% accuracy under a linear evaluation protocol.")], Selector(), self.config,
        )
        self.assertEqual(selected, [])

    def test_main_framework_question_keeps_introduction_and_prediction_task(self):
        introduction = "We introduce the ViewPair framework for learning useful visual representations."
        task = "The contrastive prediction task aims to identify corresponding views of the same observation."
        unrelated = "Our pretext tasks are not dependent on particular encoder architectures."
        duplicate = "This paper presents ViewPair as a framework for representation learning."
        selected = select_grounded_sentences(
            "이 논문이 제안한 주요 프레임워크와 예측 과제는 무엇인가요?",
            [evidence(introduction + " " + duplicate + " " + task + " " + unrelated)], Selector("encoder architectures"), self.config,
        )
        self.assertIn(task, [item["text"] for item in selected[:2]])
        self.assertIn(selected[0]["text"], (introduction, duplicate))

    def test_definition_subject_excludes_unrelated_method_apposition(self):
        definition = "Confidence calibration means that predicted probabilities represent the likelihood of correctness."
        unrelated = "We analyze slope scaling – a single-parameter variant of an earlier method – in our experiments."
        selected = select_grounded_sentences(
            "신뢰도 보정은 무엇을 의미하나요?", [evidence(definition + " " + unrelated)],
            Selector("single-parameter"), self.config,
        )
        self.assertEqual([item["text"] for item in selected], [definition])

    def test_confidence_comparison_keeps_label_invariance_and_probability_change(self):
        labels = "The class predictions remain unchanged after the transformation."
        probabilities = "In other words, the new probabilities soften the confidence while preserving the most likely class."
        accuracy = "The transformation does not affect the measured accuracy on the same examples."
        irrelevant = "A confidence diagram visualizes the measured frequencies of correctness."
        selected = select_grounded_sentences(
            "How do confidence probabilities and class predictions differ after the transformation?",
            [evidence(labels + " " + probabilities + " " + accuracy + " " + irrelevant)], Selector("confidence diagram"), self.config,
        )
        self.assertEqual({item["text"] for item in selected}, {labels, probabilities, accuracy})

    def test_causal_crop_question_keeps_shortcut_chain_and_remedy(self):
        explanation = "We conjecture that cropped views share a common background color distribution."
        shortcut = "This distribution enables a trivial shortcut for identifying matching images."
        exploitation = "The network exploits the shared histogram without recognizing the underlying objects."
        remedy = "This shortcut is removed by independently perturbing the colors of each view."
        unrelated = "When trained on only 1% of the labels, the network obtains 80% accuracy."
        selected = select_grounded_sentences(
            "Why does random cropping permit a shortcut and how is it removed?",
            [evidence(explanation + " " + shortcut + " " + exploitation + " " + remedy + " " + unrelated)],
            Selector("80%"), self.config,
        )
        self.assertEqual({item["text"] for item in selected}, {explanation, shortcut, exploitation, remedy})

    def test_requested_flip_is_supplemented_only_from_existing_reranked_pool(self):
        items = [candidate("Random crops are resized before applying the selected color transformations.", 1)]
        items += [candidate(f"The network depth setting {rank} is evaluated with a distinct optimizer configuration.", rank, page=rank)
                  for rank in range(2, 6)]
        flip = candidate("We apply random horizontal flips to the cropped and resized observations.", 6, page=6)
        items.append(flip)
        selected, metadata = select_evidence(items, top_k=5, question="Which crops, resize operations, and flips are used?")
        self.assertEqual(selected[0]["chunk_id"], items[0]["chunk"]["chunk_id"])
        self.assertIn(flip["chunk"]["chunk_id"], {item["chunk_id"] for item in selected})
        self.assertEqual(metadata["policy"]["requested_aspect_supplements"], [flip["chunk"]["chunk_id"]])
        absent, absent_metadata = select_evidence(items[:-1], top_k=5, question="Which flips are used?")
        self.assertEqual(len(absent), 5)
        self.assertEqual(absent_metadata["policy"]["requested_aspect_supplements"], [])

    def test_repeated_running_title_and_pdf_bullet_are_removed_from_source_view(self):
        title = "A General Framework for Learning from Paired Sensor Observations"
        claim = "Combining independent transformations improves the measured representation quality."
        another = "The shared encoder maps each observation to a fixed representation vector."
        records = split_evidence_sentences([evidence(title + " • " + claim, page=2),
                                            evidence(title + " " + another, rank=2, page=3)])
        self.assertEqual([item["text"] for item in records], ["- " + claim, another])
        cleaned = _clean_extractive_sentence(records[0]["text"])
        self.assertEqual(cleaned, claim)
        self.assertEqual(_clean_extractive_sentence("100 GB is required to store vectors x² and x³."),
                         "100 GB is required to store vectors x² and x³.")

    def test_glued_parameter_symbol_spacing_is_restored_without_changing_bound(self):
        self.assertEqual(_clean_extractive_sentence("We use a single scalar parameterT >0 across all classes."),
                         "We use a single scalar parameter T > 0 across all classes.")

    def test_augmentation_category_supplements_operation_list_from_reranked_pool(self):
        items = [candidate(f"The encoder setting {rank} is evaluated with an independent downstream dataset.", rank, page=rank)
                 for rank in range(1, 6)]
        operations = candidate("We sequentially apply augmentations by cropping, resizing, and randomly flipping the observations.", 6, page=6)
        selected, metadata = select_evidence(items + [operations], question="Which default augmentations are applied?")
        self.assertIn(operations["chunk"]["chunk_id"], {item["chunk_id"] for item in selected})
        self.assertIn(operations["chunk"]["chunk_id"], metadata["policy"]["requested_aspect_supplements"])

    def test_translation_accepts_explicit_upper_bound_until_quantity_phrase(self):
        _validate_translation_batch(["The operation takes up to 10 ms."], ["이 연산은 10 ms까지 소요됩니다."])
        with self.assertRaisesRegex(AnswerValidationError, "bound"):
            _validate_translation_batch(["The operation takes below 10 ms."], ["이 연산은 10 ms까지 소요됩니다."])

    def test_translation_numeric_bounds_preserve_direction_and_strictness(self):
        cases = [("below 1%", "1% 미만", "1% 이하"),
                 ("above 1%", "1% 초과", "1% 이상"),
                 ("at most 1%", "최대 1%", "1% 미만"),
                 ("at least 1%", "최소 1%", "1% 초과")]
        for source_bound, valid, weakened in cases:
            with self.subTest(source_bound=source_bound):
                source = "The observed error is " + source_bound + "."
                _validate_translation_batch([source], ["관측 오차는 " + valid + "입니다."])
                with self.assertRaisesRegex(AnswerValidationError, "bound"):
                    _validate_translation_batch([source], ["관측 오차는 " + weakened + "입니다."])

    def test_translation_cannot_swap_numeric_bound_values_or_units(self):
        source = "The memory requirement is below 4 GB and the runtime is above 2 seconds."
        _validate_translation_batch([source], ["메모리는 4 GB 미만이고 실행 시간은 2초 초과입니다."])
        for translated in ["메모리는 4 GB 초과이고 실행 시간은 2초 미만입니다.",
                           "메모리는 2초 미만이고 실행 시간은 4 GB 초과입니다."]:
            with self.subTest(translated=translated):
                with self.assertRaisesRegex(AnswerValidationError, "bound"):
                    _validate_translation_batch([source], [translated])

    def test_translation_preserves_strict_positive_symbolic_parameter_bound(self):
        source = "The scalar parameter T > 0 is shared by all classes."
        _validate_translation_batch([source], ["스칼라 매개변수 T > 0은 모든 클래스에서 공유됩니다."])
        _validate_translation_batch([source], ["스칼라 매개변수 T는 0보다 커야 하며 모든 클래스에서 공유됩니다."])
        for translated in ["스칼라 매개변수 T >= 0은 모든 클래스에서 공유됩니다.",
                           "스칼라 매개변수 T < 0은 모든 클래스에서 공유됩니다."]:
            with self.subTest(translated=translated):
                with self.assertRaisesRegex(AnswerValidationError, "bound"):
                    _validate_translation_batch([source], [translated])

    def test_korean_possessive_does_not_turn_an_exception_question_into_a_definition(self):
        self.assertNotIn("definition", question_kinds("보정의 개선에서 보고된 예외 데이터셋은 무엇인가요?"))
        self.assertNotIn("definition", question_kinds("따뜻한 환경에서 성능이 달라지나요?"))
        self.assertIn("definition", question_kinds("신뢰도의 정의는 무엇인가요?"))
        self.assertIn("definition", question_kinds("보정은 무엇을 의미하나요?"))

    def test_augmentation_enumeration_excludes_comma_separated_accuracy_results(self):
        operations = "We sequentially apply three augmentations: random masking, time shifting, and additive noise."
        source = evidence(
            operations + " Under the linear evaluation protocol, the model achieves 80% accuracy, matching a supervised baseline. "
            "When trained with only 1% of the labels, the model achieves 85% accuracy, outperforming a previous baseline. "
            "We show that augmentation improves representations, encoder depth improves accuracy, and training duration affects transfer."
        )
        selected = select_grounded_sentences(
            "List the three default augmentations", [source], Selector("85%"), self.config,
        )
        self.assertEqual([item["text"] for item in selected], [operations])

    def test_factor_enumeration_excludes_comma_separated_method_and_acknowledgments(self):
        factors = "We observe that encoder depth, optimizer choice, label noise, and sample size influence reliability."
        source = evidence(
            factors + " Our analysis offers a practical recipe: on most datasets, probability scaling, a simple post-processing method, improves predictions. "
            "The goal is to produce reliable confidence, class labels, and predictions for new observations. "
            "Acknowledgments The authors are supported by Foundation-A, Foundation-B, and Foundation-C during this work."
        )
        selected = select_grounded_sentences(
            "Which four factors influence reliability?", [source], Selector("practical recipe"), self.config,
        )
        self.assertEqual([item["text"] for item in selected], [factors])

    def test_exception_selection_keeps_low_error_context_and_excludes_general_methods(self):
        exception = "The two notable exceptions are Dataset-A and Dataset-B, which already have reliable predictions."
        low_errors = "Both datasets have very low initial error rates of 1.2% and 2.4%, respectively."
        source = evidence(
            exception + " " + low_errors + " Our analysis offers a practical recipe: on most datasets, probability scaling, a simple post-processing method, improves predictions."
        )
        selected = select_grounded_sentences(
            "Which datasets are exceptions with little room for improvement because of low initial errors?",
            [source], Selector("practical recipe"), self.config,
        )
        # The shape uses the same explicit 'exception' cue as a Korean query.
        selected_ko = select_grounded_sentences(
            "보정의 개선에서 예외 데이터셋은 무엇이며 초기 오차는 얼마인가요?",
            [source], Selector("practical recipe"), self.config,
        )
        self.assertEqual([item["text"] for item in selected_ko], [exception, low_errors])
        self.assertTrue(all(item["evidence_id"] == source["evidence_id"] for item in selected_ko))
        self.assertEqual([item["text"] for item in selected], [exception, low_errors])

    def test_nominal_accuracy_caption_does_not_consume_a_claim_slot(self):
        caption = "Benchmark-A Top-1 accuracy of linear classifiers trained on representations learned by several self-supervised methods."
        claim = "The accuracy of the linear classifier reaches 80% under the specified evaluation protocol."
        selected = split_evidence_sentences([evidence(caption + " " + claim)])
        self.assertEqual([item["text"] for item in selected], [claim])

    def test_temporal_decade_accepts_ten_years_without_licensing_other_quantities(self):
        for phrase in ("a decade ago", "an decade earlier", "one decade later", "for a decade"):
            with self.subTest(phrase=phrase):
                source = "The methods studied " + phrase + " report a 5% error rate."
                self.assertEqual(_normalized_numbers(source), {"10", "5%"})
                _validate_translation_batch([source], ["10년 전에 연구한 방법은 5% 오류율을 보고합니다."])
                with self.assertRaisesRegex(AnswerValidationError, "source quantity"):
                    _validate_translation_batch([source], ["100년 전에 연구한 방법은 5% 오류율을 보고합니다."])
                with self.assertRaisesRegex(AnswerValidationError, "source quantity"):
                    _validate_translation_batch([source], ["10년 전에 연구한 방법은 50% 오류율을 보고합니다."])
        frequency = "The measurements span a decade of frequency in the reported spectrum."
        self.assertEqual(_normalized_numbers(frequency), set())
        with self.assertRaisesRegex(AnswerValidationError, "source quantity"):
            _validate_translation_batch([frequency], ["보고된 스펙트럼에서 측정은 10년의 주파수 범위에 걸쳐 있습니다."])

    def test_front_matter_and_author_initials_do_not_enter_a_definition_answer(self):
        body = (
            "Confidence estimation – the problem of predicting probabilities representative "
            "of the true correctness likelihood – is important for classiﬁcation models."
        )
        source = evidence("A Reliable Method for Image Recognition Ada Smith 1 Basil Q. Jones 2 Abstract " + body)
        original = source["text"]
        result = generate_evidence_first_answer(
            None, Selector("Basil"), "What do Smith and colleagues mean by confidence estimation?",
            [source], config=self.config,
        )
        self.assertFalse(result["fallback_used"])
        self.assertEqual(result["response"]["answer"], body.replace("classiﬁcation", "classification"))
        self.assertEqual(result["response"]["evidence_ids"], [source["evidence_id"]])
        self.assertEqual(source["text"], original)

    def test_abstract_inside_an_ordinary_source_claim_is_preserved(self):
        sentence = "We also compare Abstract Features with a simple baseline in the controlled evaluation."
        self.assertEqual(split_evidence_sentences([evidence(sentence)])[0]["text"], sentence)

    def test_pdf_ligature_cleanup_preserves_mathematical_superscripts(self):
        original = "The classiﬁer predicts the target while the reported x² value remains unchanged."
        source = evidence(original)
        selected = split_evidence_sentences([source])
        self.assertEqual(selected[0]["text"], original.replace("classiﬁer", "classifier"))
        self.assertIn("x²", selected[0]["text"])
        self.assertEqual(source["text"], original)

    def test_citation_abbreviations_do_not_split_source_claims(self):
        claims = [
            "We compare the method with earlier work by Smith et al. (2022) under the same protocol.",
            "The evaluation includes several tasks, e.g. classification and reconstruction on independent samples.",
        ]
        selected = split_evidence_sentences([evidence(" ".join(claims))])
        self.assertEqual([item["text"] for item in selected], claims)

    def test_publication_metadata_does_not_become_an_answer_claim(self):
        source = evidence(
            "Smith, Ada and Jones, Basil. A Reliable Method for Image Recognition. "
            "In Proceedings of the International Learning Conference, 2022. "
            "The model classifies independent sensor observations during evaluation."
        )
        selected = split_evidence_sentences([source])
        self.assertEqual([item["text"] for item in selected], [
            "The model classifies independent sensor observations during evaluation."
        ])

    def test_glued_footnote_stream_is_excluded_without_removing_3d_inputs(self):
        claim = "The method classifies 3Dprinted objects using independently labeled sensor observations."
        source = evidence(claim + " 3To explain the historical connection we provide additional unrelated details.")
        self.assertEqual([item["text"] for item in split_evidence_sentences([source])], [claim])

    def test_definition_budget_does_not_fill_with_topical_motivation(self):
        definition = "Reliability means that predicted probabilities represent the true likelihood of correctness."
        source = evidence(
            "Confidence estimates provide valuable information to establish user trust in a deployed model. " + definition
        )
        result = generate_evidence_first_answer(
            None, Selector("user trust"), "What does reliability mean?", [source], config=self.config,
        )
        self.assertEqual(result["response"]["answer"], definition)

    def test_causal_answer_keeps_explanation_and_its_adjacent_consequence(self):
        causal = "Because masked views retain identical background tags, the model can match them without recognizing objects."
        consequence = "This shortcut is removed by independently perturbing the background tags during training."
        source = evidence(
            "The framework contains a shared encoder and a prediction module during representation learning. "
            + causal + " " + consequence
        )
        result = generate_evidence_first_answer(
            None, Selector("shared encoder"), "Why can the model take a shortcut with masked views?",
            [source], config=self.config,
        )
        self.assertEqual(result["response"]["answer"], causal + " " + consequence)

    def test_factor_list_budget_keeps_the_enumeration_and_excludes_unrelated_settings(self):
        factors = "We observe that encoder depth, optimizer choice, label noise, and sample size affect the measured reliability."
        source = evidence(
            factors + " The evaluation uses a held-out dataset collected from independent devices. "
            "The optimizer uses a fixed learning rate throughout parameter updates."
        )
        selected = select_grounded_sentences(
            "Which four factors affect reliability?", [source], Selector("held-out dataset"), self.config,
        )
        self.assertEqual([item["text"] for item in selected], [factors])

    def test_combined_evaluation_and_causal_question_keeps_both_answer_shapes(self):
        experiment = "We evaluated the method using independently sampled tasks on unseen devices."
        cause = "Because the target distribution changes, the representation must adapt to different observations."
        selected = select_grounded_sentences(
            "What evaluation was performed and why does adaptation matter?",
            [evidence(experiment + " " + cause)], Selector(), self.config,
        )
        self.assertEqual({item["text"] for item in selected}, {experiment, cause})

    def test_reranked_overlap_continuation_preserves_a_complete_causal_source(self):
        causal_start = "We conjecture that one issue with masking is that the resulting views share dominant background features"
        first = candidate("The model learns a reconstruction task during representation learning. " + causal_start, 1)
        unrelated = candidate("The model achieves 80% accuracy on an independent downstream classification task.", 2, page=2)
        continuation = candidate(causal_start + ", allowing trivial matching without learning object identity.", 3)
        first["chunk"].update(section_index=1, chunk_index=1)
        continuation["chunk"].update(section_index=1, chunk_index=2)
        selected, metadata = select_evidence(
            [first, unrelated, continuation], top_k=2, question="Why does masking permit a shortcut?",
        )
        self.assertEqual([item["chunk_id"] for item in selected], [first["chunk"]["chunk_id"], continuation["chunk"]["chunk_id"]])
        self.assertEqual(metadata["policy"]["overlapping_continuations"], [continuation["chunk"]["chunk_id"]])
        source_sentences = select_grounded_sentences("Why does masking permit a shortcut?", selected, Selector(), self.config)
        self.assertEqual([item["text"] for item in source_sentences], [continuation["chunk"]["text"]])

    def test_attribute_question_nouns_do_not_create_an_unverified_entity(self):
        cases = (
            ("Is RAM demand affected by model width?", "The model memory requirement increases as its width grows."),
            ("Does Random Access Memory usage increase with model size?", "The model requires more RAM as its size grows."),
            ("What AUROC does the model achieve?", "The model area under the receiver operating characteristic curve is 0.93."),
        )
        for question, text in cases:
            with self.subTest(question=question):
                self.assertEqual(_question_premise_issues(question, [evidence(text)]), [])

    def test_korean_only_aliases_preserve_scope_while_unrestricted_translations_fail(self):
        source = "When trained with only 1% of the labels, the model achieves 80% accuracy."
        _validate_translation_batch([source], ["라벨의 1%에 불과한 데이터로 훈련하면 모델은 80% 정확도를 달성합니다."])
        with self.assertRaisesRegex(AnswerValidationError, "scope restriction"):
            _validate_translation_batch([source], ["모든 라벨의 1%로 훈련하면 모델은 80% 정확도를 달성합니다."])
        exception = "The only dataset on which the method does not improve calibration is Dataset-A."
        _validate_translation_batch([exception], ["이 방법이 보정을 개선하지 않는 유일한 데이터셋은 Dataset-A입니다."])
        with self.assertRaisesRegex(AnswerValidationError, "scope restriction"):
            _validate_translation_batch([exception], ["이 방법은 Dataset-A에서 보정을 개선하지 않습니다."])

    def test_failed_translation_retains_bound_source_and_adapter_diagnostics(self):
        class DiagnosticTranslator:
            def translate(self, inputs):
                error = ValueError("protected marker disappeared")
                error.translation_diagnostics = {"stage": "marker_restoration", "source_index": 0, "raw_output": "invalid"}
                raise error
        source = evidence("The model requires 8 GB of memory for each inference request.")
        result = generate_evidence_first_answer(
            DiagnosticTranslator(), Selector(), "추론에 메모리는 얼마나 필요한가요?", [source], config=self.config,
        )
        self.assertTrue(result["fallback_used"])
        attempt = result["attempts"][0]
        self.assertEqual(attempt["translation_source_texts"], [source["text"]])
        self.assertEqual(attempt["selected_source_sentences"][0]["evidence_id"], source["evidence_id"])
        self.assertEqual(attempt["translation_diagnostics"]["stage"], "marker_restoration")
        self.assertNotIn("invalid", result["response"]["answer"])

    def test_translation_accepts_a_negated_memory_requirement_upper_bound(self):
        _validate_translation_batch(
            ["The method requires no more than 4GB of memory."],
            ["이 방법은 4GB 초과의 메모리가 필요하지 않습니다."],
        )
        with self.assertRaises(AnswerValidationError):
            _validate_translation_batch(
                ["The method requires no more than 4GB of memory."],
                ["이 방법은 4GB 초과의 메모리가 필요합니다."],
            )

    def test_glued_pdf_header_does_not_consume_a_claim_slot(self):
        selected = select_grounded_sentences(
            "How was the method evaluated?",
            [evidence("Experimental Protocols In-depth grasp performance analysis.We evaluated our method on three distinct devices in the test set.")],
            Selector(), load_answer_generation_config(),
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["text"], "We evaluated our method on three distinct devices in the test set.")

    def setUp(self):
        self.config = load_answer_generation_config()
        # These fixtures verify construction and premise safety. The separately
        # tested policy and real-model development calibration own relevance.
        policy = patch("grounded_generation.evaluate_answerability", return_value={
            "abstain": False, "reason_code": None, "enabled": False,
        })
        policy.start()
        self.addCleanup(policy.stop)

    def test_question_kinds_work_without_a_paper_specific_title(self):
        self.assertIn("list", question_kinds("어떤 데이터셋을 사용했는지 모두 알려줘"))
        self.assertIn("comparison", question_kinds("Compare both methods"))
        self.assertIn("numeric", question_kinds("How much memory is required?"))
        self.assertIn("limitations", question_kinds("What are the limitations?"))
        self.assertIn("list", question_kinds("Which three inputs are used by Model-Z?"))
        self.assertIn("list", question_kinds("모델에 사용하는 세 가지 입력은 무엇인가요?"))
        self.assertIn("comparison", question_kinds("두 모델의 정확도는 어떻게 다른가요?"))
        self.assertIn("evaluation", question_kinds("검증에 어떤 실험을 사용했나요?"))
        self.assertIn("evaluation", question_kinds("What controlled experiments were performed?"))
        self.assertIn("evaluation", question_kinds("How was the model validated?"))
        self.assertNotIn("list", question_kinds("센서 연구는 어떤 방법을 사용하나요?"))
        self.assertEqual(question_kinds(None), set())

    def test_existing_selector_call_keeps_default_page_cap(self):
        candidates = [candidate(f"Independent experimental observation number {index} is described here.", index) for index in range(1, 4)]
        selected, metadata = select_evidence(candidates)
        self.assertEqual(len(selected), 2)
        self.assertEqual(metadata["skipped"]["page_cap"], 1)
        self.assertEqual(metadata["policy"]["effective_max_per_page"], 2)

    def test_list_question_keeps_three_distinct_chunks_on_one_page(self):
        candidates = [
            candidate("The evaluation uses Dataset-A with 10,000 examples for its first task.", 1),
            candidate("The evaluation uses Dataset-B with 20,000 examples for its second task.", 2),
            candidate("The evaluation uses Dataset-C with 30,000 examples for its third task.", 3),
        ]
        selected, metadata = select_evidence(candidates, question="List all evaluation datasets")
        self.assertEqual(len(selected), 3)
        self.assertEqual([item["rank"] for item in selected], [1, 2, 3])
        self.assertEqual(metadata["policy"]["question_kinds"], ["evaluation", "list"])

    def test_near_duplicate_rule_keeps_distinct_numeric_claims(self):
        prefix = "The tested method consistently improves the benchmark accuracy under the same experimental conditions by "
        selected, _ = select_evidence([
            candidate(prefix + "10%.", 1), candidate(prefix + "20%.", 2),
        ], near_duplicate_threshold=0.5)
        self.assertEqual(len(selected), 2)

    def test_duplicate_normalization_does_not_erase_percent_unit(self):
        selected, _ = select_evidence([
            candidate("The benchmark reported an improvement of 10% after training.", 1),
            candidate("The benchmark reported an improvement of 10 after training.", 2),
        ])
        self.assertEqual(len(selected), 2)

    def test_near_duplicate_rule_keeps_named_comparison_sides(self):
        selected, _ = select_evidence([
            candidate("Model-A achieves the same measured benchmark accuracy of 82% in this evaluation.", 1),
            candidate("Model-B achieves the same measured benchmark accuracy of 82% in this evaluation.", 2),
        ], near_duplicate_threshold=0.5)
        self.assertEqual(len(selected), 2)

    def test_near_duplicate_rule_keeps_opposite_comparison_directions(self):
        selected, _ = select_evidence([
            candidate("The method reports a 10% increase under the same carefully controlled evaluation conditions.", 1),
            candidate("The method reports a 10% decrease under the same carefully controlled evaluation conditions.", 2),
        ], near_duplicate_threshold=0.5)
        self.assertEqual(len(selected), 2)

    def test_numbered_dataset_list_survives_answer_source_truncation(self):
        source = evidence(
            "The evaluation uses the following three benchmark datasets:\n"
            "1. CIFAR-10 contains 60,000 labeled color images.\n"
            "2. ImageNet contains 1.2 million labeled training images.\n"
            "3. AudioSet includes 2 million labeled audio clips."
        )
        result = generate_evidence_first_answer(None, Selector("ImageNet"), "List all benchmark datasets", [source], config=self.config)
        self.assertFalse(result["fallback_used"])
        answer = result["response"]["answer"]
        self.assertTrue(all(name in answer for name in ("CIFAR-10", "ImageNet", "AudioSet")))
        self.assertEqual(result["response"]["evidence_ids"], [source["evidence_id"]])
        self.assertEqual(len(result["selected_sentences"]), 4)

    def test_brief_numbered_list_items_are_valid_grounded_sources(self):
        source = evidence("The following datasets are included in the evaluation:\n1. CIFAR-10\n2. ImageNet\n3. AudioSet")
        selected = select_grounded_sentences("Which datasets are included?", [source], Selector("ImageNet"), self.config)
        self.assertEqual(len(selected), 4)
        self.assertEqual([item["position"] for item in selected], [1, 2, 3, 4])

    def test_numeric_result_keeps_its_adjacent_restriction(self):
        source = evidence("The method reduces memory consumption by 40% during inference. This result holds only for fixed sequence lengths.")
        result = generate_evidence_first_answer(None, Selector("40%"), "How much memory is reduced?", [source], config=self.config)
        self.assertFalse(result["fallback_used"])
        self.assertIn("only for fixed sequence lengths", result["response"]["answer"])
        self.assertEqual(len(result["selected_sentences"]), 2)

    def test_comparison_keeps_named_sides_and_distinct_measurements(self):
        sources = [
            evidence("Model-A reaches 82% accuracy under the same evaluation protocol.", 1),
            evidence("Model-B reaches 91% accuracy under the same evaluation protocol.", 2),
        ]
        result = generate_evidence_first_answer(None, Selector("82%"), "Compare accuracy of Model-A and Model-B", sources, config=self.config)
        self.assertFalse(result["fallback_used"])
        self.assertIn("82%", result["response"]["answer"])
        self.assertIn("91%", result["response"]["answer"])
        self.assertEqual(len(result["response"]["evidence_ids"]), 2)

    def test_adjacent_context_never_reads_missing_chunks(self):
        source = evidence("The evaluation uses the following three benchmark datasets:")
        selected = select_grounded_sentences("List all benchmark datasets", [source], Selector(), self.config)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["text"], source["text"])
        self.assertEqual(selected[0]["evidence_id"], source["evidence_id"])

    def test_insufficient_budget_does_not_strip_numeric_qualification(self):
        source = evidence("The method reduces memory consumption by 40% during inference. This result holds only for fixed sequence lengths.")
        config = copy.deepcopy(self.config)
        config["sentence_selector"]["max_context_chars"] = 70
        result = generate_evidence_first_answer(None, Selector("40%"), "How much memory is reduced?", [source], config=config)
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["response"]["sufficiency"], "insufficient")

    def test_translation_preserves_quantities_and_identifiers_per_input(self):
        source = evidence("The LSTM model reduces memory consumption by 40% during inference.")
        translator = Translator(["LSTM 모델은 추론 시 메모리 사용량을 40% 줄입니다."])
        result = generate_evidence_first_answer(translator, Selector(), "메모리는 얼마나 감소하는가?", [source], config=self.config)
        self.assertFalse(result["fallback_used"])
        self.assertIn("LSTM", result["response"]["answer"])

    def test_translation_cannot_swap_quantities_across_two_sources(self):
        sources = [
            evidence("The LSTM model reduces memory consumption by 40% during inference.", 1),
            evidence("The CNN model reduces memory consumption by 20% during inference.", 2),
        ]
        translator = Translator(["LSTM 모델은 메모리 사용량을 20% 줄입니다.", "CNN 모델은 메모리 사용량을 40% 줄입니다."])
        result = generate_evidence_first_answer(translator, Selector("LSTM"), "각각 메모리를 얼마나 줄이는가?", sources, config=self.config)
        self.assertTrue(result["fallback_used"])
        self.assertIn("source quantity", result["attempts"][0]["error"])

    def test_translation_cannot_silently_drop_a_technical_identifier(self):
        source = evidence("The LSTM model reduces memory consumption by 40% during inference.")
        result = generate_evidence_first_answer(Translator(["이 모델은 메모리 사용량을 40% 줄입니다."]), Selector(), "메모리는 얼마나 감소하는가?", [source], config=self.config)
        self.assertTrue(result["fallback_used"])
        self.assertIn("technical identifiers", result["attempts"][0]["error"])

    def test_translation_cannot_return_fewer_sentences_than_inputs(self):
        source = evidence("The method reduces memory consumption by 40% during inference. This result holds only for fixed sequence lengths.")
        result = generate_evidence_first_answer(Translator(["메모리 사용량이 40% 감소합니다."]), Selector("40%"), "메모리는 얼마나 감소하는가?", [source], config=self.config)
        self.assertTrue(result["fallback_used"])
        self.assertIn("one sentence per source", result["attempts"][0]["error"])

    def test_translation_cannot_change_a_measurement_unit(self):
        source = evidence("The reported median inference latency is 5 ms for each input.")
        result = generate_evidence_first_answer(Translator(["보고된 입력별 추론 지연 시간의 중앙값은 5초입니다."]), Selector(), "지연 시간은 얼마인가?", [source], config=self.config)
        self.assertTrue(result["fallback_used"])
        self.assertIn("measurement unit", result["attempts"][0]["error"])

    def test_translation_accepts_a_korean_measurement_unit_alias(self):
        source = evidence("The reported median inference latency is 5 ms for each input.")
        result = generate_evidence_first_answer(Translator(["보고된 입력별 추론 지연 시간의 중앙값은 5밀리초입니다."]), Selector(), "지연 시간은 얼마인가?", [source], config=self.config)
        self.assertFalse(result["fallback_used"])

    def test_translation_accepts_source_number_words_rendered_as_digits(self):
        source = evidence("The system processes all records without third-party tools.")
        result = generate_evidence_first_answer(Translator(["이 시스템은 제3자 도구 없이 모든 레코드를 처리합니다."]), Selector(), "레코드를 어떻게 처리하는가?", [source], config=self.config)
        self.assertFalse(result["fallback_used"])

    def test_stage_timings_distinguish_extraction_from_translation(self):
        source = evidence("The method reduces memory consumption by 40% during inference.")
        english = generate_evidence_first_answer(None, Selector(), "How much memory is reduced?", [source], config=self.config)
        korean = generate_evidence_first_answer(Translator(["메모리 사용량이 40% 감소합니다."]), Selector(), "메모리는 얼마나 감소하는가?", [source], config=self.config)
        self.assertEqual(english["stage_timings"]["translation_seconds"], 0.0)
        self.assertGreater(korean["stage_timings"]["translation_seconds"], 0.0)
        self.assertTrue(all(value >= 0 for value in korean["stage_timings"].values()))
        self.assertEqual(set(english["stage_timings"]), {"generator_activation_seconds", "sentence_selection_seconds", "answer_construction_seconds", "translation_seconds", "answer_validation_seconds"})
        self.assertEqual(english["stage_timings"]["generator_activation_seconds"], 0)

    def test_false_numeric_premise_is_refused_instead_of_confirmed(self):
        source = evidence("The reported classification accuracy is 85% on the held-out test set.")
        result = generate_evidence_first_answer(None, Selector(), "Why did the method achieve 99% accuracy?", [source], config=self.config)
        self.assertFalse(result["fallback_used"])
        self.assertEqual(result["response"]["sufficiency"], "insufficient")
        self.assertEqual(result["response"]["evidence_ids"], [])
        self.assertEqual(result["attempts"][0]["reason_code"], "unsupported_question_premise")

    def test_unverified_yes_no_numeric_question_is_conservatively_refused(self):
        source = evidence("The reported classification accuracy is 85% on the held-out test set.")
        result = generate_evidence_first_answer(None, Selector(), "Is the reported accuracy 99%?", [source], config=self.config)
        self.assertFalse(result["fallback_used"])
        self.assertEqual(result["response"]["sufficiency"], "insufficient")

    def test_question_asking_for_a_quantity_does_not_assert_one(self):
        source = evidence("The reported classification accuracy is 85% on the held-out test set.")
        result = generate_evidence_first_answer(None, Selector(), "What classification accuracy was reported?", [source], config=self.config)
        self.assertFalse(result["fallback_used"])
        self.assertIn("85%", result["response"]["answer"])

    def test_leading_source_quantity_survives_extraction_and_translation(self):
        source = evidence("100 GB of memory is needed for this model.")
        english = generate_evidence_first_answer(
            None, Selector(), "How much memory is needed?", [source], config=self.config,
        )
        translator = Translator(["이 모델에는 100 GB의 메모리가 필요합니다."])
        korean = generate_evidence_first_answer(
            translator, Selector(), "이 모델에 메모리는 얼마나 필요한가요?", [source], config=self.config,
        )
        self.assertEqual(english["response"]["sufficiency"], "sufficient")
        self.assertEqual(english["response"]["answer"], source["text"])
        self.assertEqual(translator.inputs, [source["text"]])
        self.assertEqual(korean["response"]["sufficiency"], "sufficient")
        self.assertIn("100 GB", korean["response"]["answer"])

    def test_heading_word_in_a_source_claim_is_preserved(self):
        source = evidence("Results show improved classification accuracy on the held-out test set.")
        result = generate_evidence_first_answer(
            None, Selector(), "What do the reported results show?", [source], config=self.config,
        )
        self.assertEqual(result["response"]["sufficiency"], "sufficient")
        self.assertEqual(result["response"]["answer"], source["text"])

    def test_structural_heading_and_list_cleanup_preserves_following_quantity(self):
        expected = "100 GB of memory is needed for this model."
        for prefix in ("3 Results: ", "3. Results ", "Results\n", "(a) "):
            with self.subTest(prefix=prefix):
                self.assertEqual(_clean_extractive_sentence(prefix + expected), expected)
        self.assertEqual(
            _clean_extractive_sentence("3 Results were reported for the evaluated model."),
            "3 Results were reported for the evaluated model.",
        )

    def test_supported_numeric_premise_is_allowed(self):
        source = evidence("The reported classification accuracy is 85% on the held-out test set.")
        result = generate_evidence_first_answer(None, Selector(), "Which test set achieved 85% accuracy?", [source], config=self.config)
        self.assertFalse(result["fallback_used"])

    def test_requested_list_count_does_not_require_a_literal_source_number(self):
        source = evidence("The input channels are temperature, pressure, and vibration during evaluation.")
        result = generate_evidence_first_answer(None, Selector(), "Which three inputs are required?", [source], config=self.config)
        self.assertFalse(result["fallback_used"])

    def test_false_subject_identifier_is_refused(self):
        source = evidence("The Model-A system classifies labeled sensor observations during evaluation.")
        result = generate_evidence_first_answer(None, Selector(), "What method does Model-Z use?", [source], config=self.config)
        self.assertFalse(result["fallback_used"])
        self.assertEqual(result["response"]["sufficiency"], "insufficient")
        self.assertIn("Model-Z", result["attempts"][0]["error"])

    def test_general_attribute_acronyms_accept_explicit_source_expansions(self):
        cases = (
            ("How much RAM does the model require?",
             "The proposed model requires 8 GB of memory for inference."),
            ("What is the model OOD performance?",
             "The model was evaluated on out-of-distribution datasets and achieved 91% accuracy."),
            ("How much memory in GB does the method require?",
             "The method requires 8 gigabytes of memory for inference."),
        )
        for question, text in cases:
            with self.subTest(question=question):
                result = generate_evidence_first_answer(
                    None, Selector(), question, [evidence(text)], config=self.config,
                )
                self.assertEqual(result["response"]["sufficiency"], "sufficient")
                self.assertFalse(result["fallback_used"])

    def test_attribute_alias_does_not_hide_an_unsupported_named_subject(self):
        source = evidence("The Model-A system requires 8 GB of memory for inference.")
        cases = (
            ("How much RAM does Model-Z require?", "Model-Z"),
            ("What does the RAM model do?", "RAM"),
            ("What method does RAM use?", "RAM"),
            ("How does RAM differ from the reported model?", "RAM"),
        )
        for question, identifier in cases:
            with self.subTest(question=question):
                result = generate_evidence_first_answer(
                    None, Selector(), question, [source], config=self.config,
                )
                self.assertEqual(result["response"]["sufficiency"], "insufficient")
                self.assertFalse(result["fallback_used"])
                self.assertIn(identifier, result["attempts"][0]["error"])

    def test_absent_latin_named_model_is_refused(self):
        source = evidence("The Model-A system classifies labeled sensor observations during evaluation.")
        result = generate_evidence_first_answer(None, Selector(), "What does the Helios model do?", [source], config=self.config)
        self.assertFalse(result["fallback_used"])
        self.assertEqual(result["response"]["sufficiency"], "insufficient")
        self.assertIn("Helios", result["attempts"][0]["error"])

    def test_supported_subject_identifier_is_allowed(self):
        source = evidence("The Model-A system classifies labeled sensor observations during evaluation.")
        result = generate_evidence_first_answer(None, Selector(), "What method does Model-A use?", [source], config=self.config)
        self.assertFalse(result["fallback_used"])

    def test_ordinary_transliterated_model_name_is_not_compared_to_english_spelling(self):
        source = evidence("The Helios model classifies labeled sensor observations during evaluation.")
        result = generate_evidence_first_answer(Translator(["헬리오스 모델은 평가 과정에서 레이블이 있는 센서 관측을 분류합니다."]), Selector(), "헬리오스 모델은 어떤 방법을 사용하나요?", [source], config=self.config)
        self.assertFalse(result["fallback_used"])

    def test_negative_and_scientific_numeric_premises_preserve_verification(self):
        source = evidence("The method uses -0.1 as the initial offset during training. The method uses 1e-3 as the learning rate during training.")
        supported = generate_evidence_first_answer(None, Selector(), "Why does the method use -0.1?", [source], config=self.config)
        scientific = generate_evidence_first_answer(None, Selector(), "Why does the method use 1e-3?", [source], config=self.config)
        unsupported = generate_evidence_first_answer(None, Selector(), "Why does the method use 1e-2?", [source], config=self.config)
        self.assertFalse(supported["fallback_used"])
        self.assertFalse(scientific["fallback_used"])
        self.assertFalse(unsupported["fallback_used"])
        self.assertEqual(unsupported["response"]["sufficiency"], "insufficient")

    def test_numeric_premise_does_not_exchange_seconds_and_milliseconds(self):
        source = evidence("The reported median inference latency is 5 ms for each input.")
        result = generate_evidence_first_answer(None, Selector(), "Why is inference latency 5 seconds?", [source], config=self.config)
        self.assertFalse(result["fallback_used"])
        self.assertEqual(result["response"]["sufficiency"], "insufficient")
        self.assertIn("measurement unit", result["attempts"][0]["error"])

    def test_worldwide_universal_claim_is_not_inferred_from_one_baseline(self):
        source = evidence("The method outperforms one tested baseline on the held-out test set. No worldwide comparison was performed during this evaluation.")
        result = generate_evidence_first_answer(None, Selector(), "How much better is the method than every existing method worldwide?", [source], config=self.config)
        self.assertFalse(result["fallback_used"])
        self.assertEqual(result["response"]["sufficiency"], "insufficient")
        self.assertIn("universal scope", result["attempts"][0]["error"])

    def test_explicit_universal_scope_is_allowed_when_present(self):
        source = evidence("The method outperforms all methods worldwide on the stated benchmark.")
        result = generate_evidence_first_answer(None, Selector(), "How does the method compare with all methods worldwide?", [source], config=self.config)
        self.assertFalse(result["fallback_used"])

    def test_limited_baseline_comparison_does_not_trigger_worldwide_guard(self):
        source = evidence("The method outperforms all tested baselines on the stated benchmark.")
        result = generate_evidence_first_answer(None, Selector(), "How does the method compare with all tested baselines?", [source], config=self.config)
        self.assertFalse(result["fallback_used"])

    def test_global_average_pooling_is_not_a_worldwide_scope_claim(self):
        source = {"text": "Global average pooling computes a spatial mean for each channel. It has no trainable parameters."}
        self.assertEqual(_question_premise_issues(
            "Does global average pooling require any trainable parameters?", [source]
        ), [])

    def test_finite_global_benchmark_is_not_a_worldwide_scope_claim(self):
        source = evidence("The method outperforms every baseline evaluated in the global benchmark.")
        question = "How does the method compare with every baseline in the global benchmark?"
        self.assertEqual(_question_premise_issues(question, [source]), [])
        result = generate_evidence_first_answer(None, Selector(), question, [source], config=self.config)
        self.assertFalse(result["fallback_used"])

    def test_worldwide_yes_no_question_without_comparison_is_not_universal_ranking(self):
        source = {"text": "The study does not establish whether this method is available worldwide."}
        self.assertEqual(_question_premise_issues(
            "Is the method available to all users worldwide?", [source]
        ), [])

    def test_limitations_include_the_adjacent_unmeasured_scope(self):
        source = evidence("The study uses observations from a single industrial facility. Generalization to other facilities was not evaluated during the experiment.")
        result = generate_evidence_first_answer(None, Selector("single industrial"), "What generalization limitations were reported?", [source], config=self.config)
        self.assertFalse(result["fallback_used"])
        self.assertIn("was not evaluated", result["response"]["answer"])

    def test_unsupported_requested_aspect_is_refused_before_translation(self):
        source = evidence("The model classifies labeled sensor observations during evaluation.")
        translator = Translator(["모델은 센서 관측을 분류합니다."])
        decision = {"abstain": True, "reason_code": "requested_aspect_not_supported", "enabled": True}
        with patch("grounded_generation.evaluate_answerability", return_value=decision):
            result = generate_evidence_first_answer(translator, Selector(), "지원 기관은 어디인가요?", [source], config=self.config)
        self.assertFalse(result["fallback_used"])
        self.assertEqual(result["response"]["sufficiency"], "insufficient")
        self.assertIsNone(translator.inputs)
        self.assertEqual(result["answerability"], decision)
        self.assertEqual(result["stage_timings"]["translation_seconds"], 0.0)

    def test_answerability_scorer_failure_cannot_produce_a_sufficient_answer(self):
        source = evidence("The model classifies labeled sensor observations during evaluation.")
        with patch("grounded_generation.evaluate_answerability", side_effect=RuntimeError("non-finite score")):
            result = generate_evidence_first_answer(None, Selector(), "What method is used?", [source], config=self.config)
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["attempts"][0]["reason_code"], "answerability_scoring_failed")

    def test_equivalent_decimal_and_scientific_values_are_canonicalized(self):
        self.assertEqual(_normalized_numbers("1.0 seconds"), _normalized_numbers("1 second"))
        self.assertEqual(_normalized_numbers("1e-3"), _normalized_numbers("0.001"))
        self.assertEqual(_normalized_numbers("−1.00"), _normalized_numbers("-1"))
        self.assertNotEqual(_normalized_numbers("1e-3"), _normalized_numbers("1e-2"))

    def test_equivalent_decimal_quantity_and_unit_do_not_refuse_the_question(self):
        source = {"text": "At 1 second the training procedure reaches the warmup boundary."}
        self.assertEqual(_question_premise_issues("What happens at 1.0 seconds?", [source]), [])

    def test_equivalent_scientific_quantity_does_not_refuse_the_question(self):
        source = {"text": "The training procedure uses 0.001 as the fixed learning rate."}
        self.assertEqual(_question_premise_issues("Why does the method use 1e-3?", [source]), [])

    def test_translation_accepts_equivalent_decimal_and_unit_aliases(self):
        _validate_translation_batch(
            ["The training procedure reaches the warmup boundary at 1.0 second."],
            ["학습 절차는 1초에 준비 단계의 경계에 도달합니다."],
        )

    def test_translation_accepts_equivalent_scientific_values(self):
        _validate_translation_batch(
            ["The method uses 1e-3 as the fixed learning rate."],
            ["이 방법은 고정 학습률로 0.001을 사용합니다."],
        )

    def test_translation_rejects_removed_explicit_negation(self):
        with self.assertRaisesRegex(AnswerValidationError, "negation"):
            _validate_translation_batch(
                ["The model does not generalize to other facilities."],
                ["이 모델은 다른 시설에도 일반화됩니다."],
            )

    def test_translation_rejects_removed_only_scope(self):
        with self.assertRaisesRegex(AnswerValidationError, "scope restriction"):
            _validate_translation_batch(
                ["Only the small dataset was evaluated."],
                ["모든 데이터셋이 평가되었습니다."],
            )

    def test_translation_accepts_normal_korean_negation(self):
        cases = [
            ("The model does not support variable sequence lengths.", "이 모델은 가변 길이 시퀀스를 지원하지 않습니다."),
            ("The model cannot handle unseen datasets.", "이 모델은 새로운 데이터셋을 처리할 수 없습니다."),
            ("The method never changes the original labels.", "이 방법은 원래 레이블을 변경하지 않습니다."),
        ]
        for source, translated in cases:
            with self.subTest(source=source):
                _validate_translation_batch([source], [translated])

    def test_translation_accepts_normal_korean_scope_restrictions(self):
        translations = [
            "오직 소규모 데이터셋이 평가되었습니다.",
            "소규모 데이터셋에만 평가를 수행했습니다.",
            "소규모 데이터셋에서만 평가를 수행했습니다.",
            "평가된 것은 소규모 데이터셋뿐입니다.",
            "단지 소규모 데이터셋이 평가되었습니다.",
            "단 소규모 데이터셋이 평가되었습니다.",
        ]
        for translated in translations:
            with self.subTest(translated=translated):
                _validate_translation_batch(["Only the small dataset was evaluated."], [translated])

    def test_korean_word_containing_dan_is_not_a_scope_restriction(self):
        with self.assertRaisesRegex(AnswerValidationError, "scope restriction"):
            _validate_translation_batch(
                ["Only the small dataset was evaluated."],
                ["모든 데이터셋을 단계별로 평가했습니다."],
            )

    def test_not_only_idiom_does_not_force_false_negative_translation(self):
        _validate_translation_batch(
            ["The method not only classifies inputs but also calibrates sensors."],
            ["이 방법은 입력을 분류할 뿐 아니라 센서를 보정합니다."],
        )

    def test_translation_preserves_explicit_upper_and_lower_bounds(self):
        _validate_translation_batch(
            ["The method improves accuracy by at most 5%."],
            ["이 방법은 정확도를 최대 5% 개선합니다."],
        )
        _validate_translation_batch(
            ["The method improves accuracy by at least 5%."],
            ["이 방법은 정확도를 최소 5% 개선합니다."],
        )
        with self.assertRaisesRegex(AnswerValidationError, "upper bound"):
            _validate_translation_batch(
                ["The method improves accuracy by at most 5%."],
                ["이 방법은 정확도를 5% 개선합니다."],
            )

    def test_korean_evaluation_question_retains_four_source_sentences(self):
        sentences = [
            "The evaluation uses a fixed test set of 2,000 labeled examples.",
            "Model-A is compared against the Model-B baseline using the same protocol.",
            "Precision and recall are the metrics used to evaluate every model.",
            "Only the three specified datasets are included in this evaluation.",
        ]
        translations = dict(zip(sentences, [
            "평가에는 레이블이 있는 2,000개 표본으로 구성된 고정 테스트셋을 사용합니다.",
            "동일한 절차로 Model-A를 Model-B 기준 모델과 비교합니다.",
            "모든 모델의 평가 지표는 정밀도와 재현율입니다.",
            "평가는 지정된 3개의 데이터셋에만 적용됩니다.",
        ]))

        class EvaluationTranslator:
            def translate(self, inputs):
                self.inputs = inputs
                return [translations[text] for text in inputs]

        translator = EvaluationTranslator()
        result = generate_evidence_first_answer(
            translator, Selector("fixed test set"),
            "이 방법을 검증하는 데 어떤 실험과 평가 지표를 사용했나요?",
            [evidence(" ".join(sentences))], config=self.config,
        )
        self.assertFalse(result["fallback_used"])
        self.assertEqual(result["response"]["sufficiency"], "sufficient")
        self.assertEqual(len(translator.inputs), 4)
        self.assertTrue(all(text in translator.inputs for text in sentences))
        self.assertIn("Model-B", result["response"]["answer"])
        self.assertIn("데이터셋에만", result["response"]["answer"])

    def test_english_evaluation_question_uses_four_sentence_budget(self):
        self.assertEqual(self.config["answer"]["source_sentences_by_question_kind"]["evaluation"], {"en": 4, "ko": 4})
        source = evidence(
            "The evaluation uses a fixed test set of labeled sensor observations. "
            "Model-A is compared against the Model-B baseline using the same protocol. "
            "Precision and recall are the metrics used to evaluate every model. "
            "Only the specified datasets are included in this evaluation."
        )
        result = generate_evidence_first_answer(None, Selector("fixed test set"), "What experiments evaluate the model?", [source], config=self.config)
        self.assertFalse(result["fallback_used"])
        self.assertEqual(len(result["selected_sentences"]), 4)
        self.assertIn("Model-B", result["response"]["answer"])
        self.assertIn("Only the specified datasets", result["response"]["answer"])

    def test_evaluation_budget_does_not_fill_with_unrelated_hyperparameters(self):
        experiments = [
            "We evaluated the method using two independent controlled experiments.",
            "The second experiment validates behavior under a changed sampling distribution.",
        ]
        source = evidence(" ".join(experiments + [
            "The optimizer uses a constant regularization coefficient throughout parameter updates.",
            "This result concerns only the fixed regularization coefficient.",
        ]))
        translated = dict(zip(experiments, [
            "서로 독립적인 두 개의 통제 실험으로 방법을 평가했습니다.",
            "두 번째 실험은 표본 추출 분포가 달라질 때의 동작을 검증합니다.",
        ]))

        class ExperimentTranslator:
            def translate(self, inputs):
                self.inputs = inputs
                return [translated[text] for text in inputs]

        translator = ExperimentTranslator()
        question = "이 방법을 검증하는 데 어떤 실험을 사용했나요?"
        selected = select_grounded_sentences(question, [source], Selector("regularization coefficient"), self.config)
        self.assertEqual([item["text"] for item in selected], experiments)
        self.assertTrue(all(item["evidence_id"] == source["evidence_id"] for item in selected))
        result = generate_evidence_first_answer(translator, Selector("regularization coefficient"), question, [source], config=self.config)
        self.assertEqual(len(translator.inputs), 2)
        self.assertFalse(result["fallback_used"])
        self.assertEqual(result["response"]["sufficiency"], "sufficient")

    def test_evaluation_anchor_keeps_adjacent_condition_without_an_evaluation_cue(self):
        source = evidence(
            "We evaluated the method using independent controlled experiments. "
            "This result holds only for fixed sequence lengths."
        )
        result = generate_evidence_first_answer(None, Selector("We evaluated"), "What experiments evaluate the method?", [source], config=self.config)
        self.assertFalse(result["fallback_used"])
        self.assertEqual(len(result["selected_sentences"]), 2)
        self.assertIn("only for fixed sequence lengths", result["response"]["answer"])


if __name__ == "__main__":
    unittest.main()
