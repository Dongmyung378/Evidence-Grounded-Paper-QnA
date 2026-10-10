"""Source-only terminology protection, fail-closed restoration and cache checks."""

import contextlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from grounded_answer_contract import AnswerValidationError
from local_translation import (
    MARKER,
    LocalEnglishKoreanTranslator,
    TranslationFidelityError,
    load_translation_config,
    load_translation_glossary,
    protect_source_spans,
    restore_source_spans,
    translation_config_fingerprint,
)


class SourceSpanProtectionTests(unittest.TestCase):
    def setUp(self):
        self.glossary = load_translation_glossary()

    def test_terms_and_unlisted_multiword_names_are_source_only(self):
        source = "We evaluated Shadow Hand and Orion Hand robot grippers against ground truth."
        protected, mappings = protect_source_spans(source, self.glossary)
        self.assertEqual(len(mappings), 4)
        self.assertEqual(
            list(mappings.values()), ["Shadow Hand", "Orion Hand", "로봇 그리퍼", "참값"]
        )
        self.assertNotIn("Shadow Hand", protected)
        self.assertNotIn("ground truth", protected)
        markers = list(mappings)
        restored = restore_source_spans(
            f"{markers[0]}과 {markers[1]} {markers[2]}를 {markers[3]}에 대해 평가했습니다.",
            mappings,
        )
        self.assertIn("Shadow Hand", restored)
        self.assertIn("Orion Hand", restored)
        self.assertIn("로봇 그리퍼", restored)
        self.assertIn("참값", restored)

    def test_longest_phrase_wins_and_whole_words_are_required(self):
        source = "Robot grippers use ground-truth labels, not gripperless components."
        protected, mappings = protect_source_spans(source, self.glossary)
        self.assertEqual(list(mappings.values()), ["로봇 그리퍼", "정답 레이블"])
        self.assertIn("gripperless", protected)
        self.assertEqual(protected.count("ZXQ"), 2)

    def test_terms_match_case_and_variable_whitespace(self):
        _, mappings = protect_source_spans("ROBOT   GRIPPERS use Ground Truth.", self.glossary)
        self.assertEqual(list(mappings.values()), ["로봇 그리퍼(ROBOT   GRIPPERS)", "참값"])

    def test_generic_learning_terms_are_protected_without_embedding_a_model_name(self):
        source = ("Weight decay, temperature scaling, batch normalization and knowledge distillation "
                  "affect confidence calibration in self-supervised learning.")
        protected, mappings = protect_source_spans(source, self.glossary)
        self.assertEqual(list(mappings.values()), [
            "가중치 감쇠", "온도 보정", "배치 정규화", "지식 증류", "신뢰도 보정", "자기지도 학습",
        ])
        self.assertEqual(len(MARKER.findall(protected)), 6)
        self.assertNotIn("SimCLR", " ".join(mappings.values()))

    def test_learning_terms_use_longest_alias_and_do_not_match_unrelated_substrings(self):
        source = "Selfsupervised learning uses a projection head and data augmentations, not headless devices."
        _, mappings = protect_source_spans(source, self.glossary)
        self.assertEqual(list(mappings.values()), ["자기지도 학습", "투영 헤드", "데이터 증강"])

    def test_visual_learning_terms_preserve_explicit_source_meanings(self):
        _, mappings = protect_source_spans(
            "Contrastive prediction tasks use random cropping, color distortions and Gaussian blur, "
            "while pretext tasks can allow shortcuts.", self.glossary,
        )
        self.assertEqual(list(mappings.values()), [
            "대조 예측 과제", "무작위 자르기", "색상 왜곡", "가우시안 블러", "사전학습 과제", "지름길",
        ])

    def test_projection_and_loss_phrases_use_longest_match_without_masking_bare_words(self):
        source = ("A nonlinear projection head uses nonlinear projection rather than linear projection "
                  "and minimizes contrastive loss.")
        protected, mappings = protect_source_spans(source, self.glossary)
        self.assertEqual(protected, "A ZXQAAA uses ZXQAAB rather than ZXQAAC and minimizes ZXQAAD.")
        self.assertEqual(list(mappings.values()), [
            "비선형 투영 헤드", "비선형 투영", "선형 투영", "대조 손실",
        ])
        ordinary = "The medical projection gives a contrastive view without a projectionless feature."
        self.assertEqual(protect_source_spans(ordinary, self.glossary), (ordinary, {}))
        partial = "The nonlinear projectionTensor is unrelated to contrastive lossless input."
        self.assertEqual(protect_source_spans(partial, self.glossary), (partial, {}))

    def test_negative_projection_phrase_preserves_absence_without_masking_other_words(self):
        source = "Linear projection is better than no projection (>10%)."
        protected, mappings = protect_source_spans(source, self.glossary)
        self.assertEqual(list(mappings.values()), ["선형 투영", "투영 없음"])
        self.assertIn(">10%", protected)
        restored = restore_source_spans("ZXQAAA은 ZXQAAB보다 낫습니다 (>10%).", mappings)
        self.assertIn("투영 없음", restored)
        self.assertIn(">10%", restored)
        for ordinary in ("The projection is useful.", "A no projectionless feature is unrelated."):
            self.assertEqual(protect_source_spans(ordinary, self.glossary), (ordinary, {}))

    def test_calibration_morphology_and_qualifications_use_the_longest_source_phrase(self):
        pairs = (("calibration", "보정"), ("calibrate", "보정"), ("calibrates", "보정"),
                 ("calibrating", "보정"), ("calibrated", "보정된"),
                 ("well-calibrated", "잘 보정된"), ("poorly calibrated", "보정이 부정확한"),
                 ("miscalibrated", "보정이 부정확한"), ("uncalibrated", "보정되지 않은"),
                 ("not calibrated", "보정되지 않은"), ("miscalibration", "보정 부정확성"))
        for source, target in pairs:
            with self.subTest(source=source):
                protected, mappings = protect_source_spans(source, self.glossary)
                self.assertEqual(protected, "ZXQAAA")
                self.assertEqual(mappings, {"ZXQAAA": target})
        _, mappings = protect_source_spans("Confidence calibration", self.glossary)
        self.assertEqual(list(mappings.values()), ["신뢰도 보정"])

    def test_scalar_parameter_term_keeps_math_tokens_and_requires_lexical_boundaries(self):
        protected, mappings = protect_source_spans("A single scalar parameter T > 0 is shared.", self.glossary)
        self.assertIn("T > 0", protected)
        self.assertEqual(list(mappings.values()), ["단일 스칼라 매개변수"])
        _, mappings = protect_source_spans("A single scalar parameterTensor T > 0 is stored.", self.glossary)
        self.assertNotIn("스칼라 매개변수", mappings.values())

    def test_percentage_bounds_preserve_exact_source_value_and_strictness(self):
        cases = (("below 1%", "1% 미만", "1%"), ("less than 2 percent", "2% 미만", "2 percent"),
                 ("above 0.1%", "0.1% 초과", "0.1%"),
                 ("more than 1e-3 percent", "1e-3% 초과", "1e-3 percent"),
                 ("at most 1%", "1% 이하", "1%"), ("no more than 1%", "1% 이하", "1%"),
                 ("at least −1e-3%", "−1e-3% 이상", "−1e-3%"),
                 ("no fewer than 2%", "2% 이상", "2%"))
        for source, expected, visible_quantity in cases:
            with self.subTest(source=source):
                protected, mappings = protect_source_spans(source, self.glossary)
                self.assertEqual(protected, "ZXQAAA " + visible_quantity)
                self.assertEqual(mappings, {"ZXQAAA": expected})
        source = "An error below 1% differs from an error at most 1%."
        _, mappings = protect_source_spans(source, self.glossary)
        self.assertEqual(list(mappings.values()), ["1% 미만", "1% 이하"])

    def test_glued_bound_gets_only_a_marker_separator_and_keeps_the_quantity(self):
        protected, mappings = protect_source_spans("The error is below1%.", self.glossary)
        self.assertEqual(protected, "The error is ZXQAAA 1%.")
        self.assertEqual(mappings, {"ZXQAAA": "1% 미만"})

    def test_percentage_restoration_collapses_only_adjacent_equivalent_literals(self):
        for raw, target in (("1% ZXQAAA", "1% 미만"), ("ZXQAAA 1 percent", "1% 미만"),
                            ("1.0퍼센트 ZXQAAA", "1% 미만"),
                            ("ZXQAAA -0.001%", "−1e-3% 이상")):
            with self.subTest(raw=raw):
                self.assertEqual(restore_source_spans(raw, {"ZXQAAA": target}), target)
        self.assertEqual(
            restore_source_spans("99% ZXQAAA", {"ZXQAAA": "1% 미만"}), "99% 1% 미만",
        )
        self.assertEqual(
            restore_source_spans("1% 이하 ZXQAAA", {"ZXQAAA": "1% 미만"}), "1% 이하 1% 미만",
        )
        self.assertEqual(
            restore_source_spans("1%에서 결과는 ZXQAAA입니다.", {"ZXQAAA": "1% 미만"}),
            "1%에서 결과는 1% 미만입니다.",
        )

    def test_bound_marker_omission_duplication_or_two_adjacent_literals_are_rejected(self):
        for raw in ("1%입니다.", "1% ZXQAAA ZXQAAA", "1% ZXQAAA 1%"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                restore_source_spans(raw, {"ZXQAAA": "1% 미만"})

    def test_punctuation_spacing_preserves_list_items_numbers_and_links(self):
        source = "Use random cropping,color distortions;Gaussian blur on 1,000 samples at 0.1%."
        protected, mappings = protect_source_spans(source, self.glossary)
        self.assertEqual(protected, "Use ZXQAAA, ZXQAAB; ZXQAAC on 1,000 samples at 0.1%.")
        self.assertEqual(list(mappings.values()), ["무작위 자르기", "색상 왜곡", "가우시안 블러"])
        link_source = "See https://example.org/foo,bar;a,b or urn:foo,bar at 1:2."
        self.assertEqual(protect_source_spans(link_source, self.glossary), (link_source, {}))

    def test_percentage_protection_does_not_infer_unwritten_values_or_convert_other_units(self):
        for source in ("below the error rate", "at most four percent", "below 1 percentage point", "less than 2 seconds"):
            with self.subTest(source=source):
                protected, mappings = protect_source_spans(source, self.glossary)
                self.assertEqual(protected, source)
                self.assertEqual(mappings, {})

    def test_no_unmentioned_term_or_identifier_is_inserted(self):
        source = "The procedure improves retrieval."
        protected, mappings = protect_source_spans(source, self.glossary)
        self.assertEqual(protected, source)
        self.assertEqual(mappings, {})
        self.assertEqual(restore_source_spans("검색 절차를 개선합니다.", mappings), "검색 절차를 개선합니다.")

    def test_identifier_formats_and_leading_article(self):
        _, mappings = protect_source_spans(
            "The Shadow Hand uses CNN, ResNet-50 and Model-A.", self.glossary
        )
        self.assertEqual(list(mappings.values()), ["Shadow Hand", "CNN", "ResNet-50", "Model-A"])

    def test_single_word_names_are_preserved_only_in_explicit_data_resource_context(self):
        source = "The Helios dataset and Orion corpus are compared with the Vega benchmark."
        _, mappings = protect_source_spans(source, self.glossary)
        self.assertEqual(list(mappings.values()), ["Helios", "Orion", "Vega"])
        _, mappings = protect_source_spans("The dataset provides samples. Results improve.", self.glossary)
        self.assertEqual(mappings, {})

    def test_explicit_batch_names_only_protect_matching_source_occurrences(self):
        source = "Helios and Helioscope are compared with another resource."
        protected, mappings = protect_source_spans(source, self.glossary, source_names=("Helios", "Vega"))
        self.assertEqual(protected, "ZXQAAA and Helioscope are compared with another resource.")
        self.assertEqual(mappings, {"ZXQAAA": "Helios"})
        self.assertEqual(protect_source_spans(source, self.glossary), (source, {}))
        for invalid in ("Helios", ("",), (None,)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                protect_source_spans(source, self.glossary, source_names=invalid)

    def test_each_repeated_occurrence_has_its_own_marker(self):
        protected, mappings = protect_source_spans("Shadow Hand differs from Shadow Hand.", self.glossary)
        self.assertEqual(len(mappings), 2)
        self.assertEqual(len(set(MARKER.findall(protected))), 2)
        self.assertEqual(list(mappings.values()), ["Shadow Hand", "Shadow Hand"])

    def test_more_than_26_spans_have_unique_nonnumeric_markers(self):
        source = ", ".join("MODEL" + chr(65 + i // 26) + chr(65 + i % 26) for i in range(80))
        protected, mappings = protect_source_spans(source, self.glossary)
        self.assertEqual(len(mappings), 80)
        self.assertEqual(len(MARKER.findall(protected)), 80)
        self.assertNotRegex(protected, r"\d")

    def test_korean_particles_follow_the_restored_term_not_the_latin_marker(self):
        mappings = {"ZXQAAA": "참값", "ZXQAAB": "로봇 그리퍼", "ZXQAAC": "정답 레이블"}
        self.assertEqual(
            restore_source_spans("ZXQAAA를 ZXQAAB은 ZXQAAC으로 비교합니다.", mappings),
            "참값을 로봇 그리퍼는 정답 레이블로 비교합니다.",
        )
        self.assertEqual(restore_source_spans("ZXQAAA입니다.", {"ZXQAAA": "참값"}), "참값입니다.")

    def test_missing_duplicate_mutated_and_invented_markers_are_rejected(self):
        mappings = {"ZXQAAA": "Shadow Hand", "ZXQAAB": "참값"}
        for output in (
            "ZXQAAA만 평가했습니다.",
            "ZXQAAA와 ZXQAAA를 ZXQAAB와 비교했습니다.",
            "ZXQAAC와 ZXQAAB를 비교했습니다.",
            "ZXQAAA와 ZXQAAB를 ZXQAAD와 비교했습니다.",
            "ZXQ AAA와 ZXQAAB를 비교했습니다.",
            "zxqaaa와 ZXQAAB를 비교했습니다.",
        ):
            with self.subTest(output=output), self.assertRaises(ValueError):
                restore_source_spans(output, mappings)

    def test_reserved_marker_collision_in_source_is_rejected(self):
        for source in ("The model ZXQAAA is evaluated.", "The model zxqabc is evaluated."):
            with self.subTest(source=source), self.assertRaises(ValueError):
                protect_source_spans(source, self.glossary)

    def test_unexpected_marker_is_rejected_even_without_protected_source_spans(self):
        for output in ("결과는 ZXQAAA입니다.", "결과는 zxqaaa입니다.", "결과는 ZXQ AAA입니다."):
            with self.subTest(output=output), self.assertRaises(ValueError):
                restore_source_spans(output, {})

    def test_glossary_changes_have_distinct_fingerprints(self):
        original = translation_config_fingerprint(self.glossary)
        self.glossary["entries"][0]["target"] = "그리퍼"
        self.assertNotEqual(original, translation_config_fingerprint(self.glossary))

    def test_malformed_or_quantitative_glossary_entries_are_rejected(self):
        original = load_translation_glossary()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "glossary.json"
            variants = []
            for field, value in (("source", ["gripper"]), ("target", "그리퍼 3개"), ("target", "ZXQAAA"), ("target", "")):
                config = json.loads(json.dumps(original))
                config["entries"].append({"source": ["invalid term"], "target": "용어"})
                config["entries"][-1][field] = value
                variants.append(config)
            for config in variants:
                path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_translation_glossary(path)


class ProtectedAdapterCacheTests(unittest.TestCase):
    def make_translator(self, cache_size=128):
        with patch.object(LocalEnglishKoreanTranslator, "_load"):
            translator = LocalEnglishKoreanTranslator(cache_size=cache_size)
        translator.device = "cpu"
        translator.dtype = "float32"
        translator.device_fallback_reason = None
        # These fixtures isolate the primary guard and cache behavior. Retry
        # behavior is exercised explicitly in dedicated tests below.
        translator.config["retry_on_validation_failure"] = False
        return translator

    def test_terms_and_names_restore_before_numeric_and_identifier_validation(self):
        translator = self.make_translator()
        source = "Shadow Hand robot grippers reached 95.2 percent accuracy against ground truth."
        with patch.object(translator, "_translate_model_batch", return_value=[
            "ZXQAAA ZXQAAB는 ZXQAAC에 대해 95.2%의 정확도를 달성했습니다."
        ]):
            self.assertEqual(translator.translate([source]), [
                "Shadow Hand 로봇 그리퍼는 참값에 대해 95.2%의 정확도를 달성했습니다."
            ])
        self.assertEqual(translator.cache_info()["size"], 1)

    def test_all_caps_terms_keep_source_identifiers_without_weakening_checks(self):
        translator = self.make_translator()
        source = "ROBOT GRIPPERS use GROUND TRUTH."
        with patch.object(translator, "_translate_model_batch", return_value=["ZXQAAA는 ZXQAAB를 사용합니다."]):
            self.assertEqual(translator.translate([source]), [
                "로봇 그리퍼(ROBOT GRIPPERS)는 참값(GROUND TRUTH)을 사용합니다."
            ])

    def test_repeated_sentence_avoids_another_model_call(self):
        translator = self.make_translator()
        source = "The accuracy is 95 percent."
        with patch.object(translator, "_translate_model_batch", return_value=["정확도는 95%입니다."]) as model:
            self.assertEqual(translator.translate([source]), ["정확도는 95%입니다."])
            self.assertEqual(translator.translate([source]), ["정확도는 95%입니다."])
            self.assertEqual(model.call_count, 1)
        self.assertEqual(translator.cache_info()["hits"], 1)
        self.assertEqual(translator.cache_info()["misses"], 1)

    def test_batch_deduplication_preserves_requested_order(self):
        translator = self.make_translator()
        first, second = "The score is 1.", "The score is 2."
        with patch.object(translator, "_translate_model_batch", return_value=["점수는 1입니다.", "점수는 2입니다."]) as model:
            self.assertEqual(translator.translate([first, second, first]), ["점수는 1입니다.", "점수는 2입니다.", "점수는 1입니다."])
            self.assertEqual(model.call_args.args[0], [first, second])
        self.assertEqual(translator.cache_info()["batch_duplicates"], 1)

    def test_modified_quantity_is_rejected_and_not_cached(self):
        translator = self.make_translator()
        source = "The accuracy is 95.2 percent."
        with patch.object(translator, "_translate_model_batch", return_value=["정확도는 99%입니다."]) as model:
            for _ in range(2):
                with self.assertRaises(AnswerValidationError):
                    translator.translate([source])
            self.assertEqual(model.call_count, 2)
        self.assertEqual(translator.cache_info()["size"], 0)

    def test_equivalent_decimal_and_scientific_notation_remain_valid(self):
        translator = self.make_translator()
        source = "At 1.0 seconds the rate is 1e-3."
        with patch.object(translator, "_translate_model_batch", return_value=["1초에서 비율은 0.001입니다."]):
            self.assertEqual(translator.translate([source]), ["1초에서 비율은 0.001입니다."])

    def test_source_percentage_bound_is_restored_before_fidelity_validation_and_cache(self):
        translator = self.make_translator()
        with patch.object(translator, "_translate_model_batch", return_value=["오류는 1% ZXQAAA입니다."]) as model:
            self.assertEqual(translator.translate(["The error is below 1%."]), ["오류는 1% 미만입니다."])
            self.assertEqual(model.call_args.args[0], ["The error is ZXQAAA 1%."])
        self.assertEqual(translator.cache_info()["size"], 1)

    def test_visible_bound_numbers_or_relation_changes_are_rejected_and_not_cached(self):
        for raw in ("오류는 99% ZXQAAA입니다.", "오류는 1% 이하 ZXQAAA입니다."):
            translator = self.make_translator()
            with self.subTest(raw=raw), patch.object(translator, "_translate_model_batch", return_value=[raw]):
                with self.assertRaises(AnswerValidationError):
                    translator.translate(["The error is below 1%."])
            self.assertEqual(translator.cache_info()["size"], 0)

    def test_formatted_augmentation_list_still_rejects_dropped_terms(self):
        translator = self.make_translator()
        source = "We apply random cropping,color distortions, and Gaussian blur."
        with patch.object(translator, "_translate_model_batch", return_value=["ZXQAAA를 적용합니다."]) as model:
            with self.assertRaises(TranslationFidelityError):
                translator.translate([source])
            self.assertEqual(model.call_args.args[0], ["We apply ZXQAAA, ZXQAAB, and ZXQAAC."])
        self.assertEqual(translator.cache_info()["size"], 0)

    def test_negative_scope_and_unit_failures_never_enter_cache(self):
        cases = (
            ("The model does not generalize.", "모델은 일반화됩니다."),
            ("Only the small dataset was evaluated.", "모든 데이터셋이 평가되었습니다."),
            ("The interval is 2 seconds.", "간격은 2분입니다."),
        )
        for source, output in cases:
            translator = self.make_translator()
            with self.subTest(source=source), patch.object(translator, "_translate_model_batch", return_value=[output]):
                with self.assertRaises(AnswerValidationError):
                    translator.translate([source])
            self.assertEqual(translator.cache_info()["size"], 0)

    def test_english_predicate_copy_is_rejected_even_when_restoration_has_hangul(self):
        cases = (
            ("The accuracy is 95 percent.", "The accuracy is 95 percent."),
            ("The robot grippers were evaluated.", "The ZXQAAA were evaluated."),
        )
        for source, output in cases:
            translator = self.make_translator()
            with self.subTest(source=source), patch.object(translator, "_translate_model_batch", return_value=[output]):
                with self.assertRaises(ValueError):
                    translator.translate([source])
            self.assertEqual(translator.cache_info()["size"], 0)

    def test_cjk_output_is_rejected_before_it_can_be_cached(self):
        translator = self.make_translator()
        with patch.object(translator, "_translate_model_batch", return_value=["정확도는 95%真입니다."]), self.assertRaises(ValueError):
            translator.translate(["The accuracy is 95 percent."])
        self.assertEqual(translator.cache_info()["size"], 0)

    def test_latin_name_only_list_is_a_valid_translation_fragment(self):
        translator = self.make_translator()
        with patch.object(translator, "_translate_model_batch", return_value=["ZXQAAA, ZXQAAB, ZXQAAC."]):
            self.assertEqual(translator.translate(["Model-A, Model-B, and BERT."]), ["Model-A, Model-B, BERT."])
        self.assertEqual(translator.cache_info()["size"], 1)

    def test_numeric_fragments_keep_scientific_notation_and_recognized_unit_aliases(self):
        for source, output in (("−1e-3", "-0.001"), ("2 seconds", "2 s"), ("1 million", "1000000")):
            translator = self.make_translator()
            with self.subTest(source=source), patch.object(translator, "_translate_model_batch", return_value=[output]):
                self.assertEqual(translator.translate([source]), [output])
            self.assertEqual(translator.cache_info()["size"], 1)

    def test_marker_failure_and_partial_batch_failure_are_not_cached(self):
        translator = self.make_translator()
        sources = ["The score is 1.", "Shadow Hand is evaluated."]
        with patch.object(translator, "_translate_model_batch", return_value=["점수는 1입니다.", "그림자 손을 평가합니다."]):
            with self.assertRaises(ValueError):
                translator.translate(sources)
        self.assertEqual(translator.cache_info()["size"], 0)

    def test_failed_translation_has_opt_in_diagnostics_without_invalid_text_in_message(self):
        translator = self.make_translator()
        sources = ["The score is 1.", "Shadow Hand is evaluated."]
        with patch.object(translator, "_translate_model_batch", return_value=["점수는 1입니다.", "그림자 손을 평가합니다."]):
            with self.assertRaises(TranslationFidelityError) as raised:
                translator.translate(sources)
        error = raised.exception
        self.assertIsInstance(error, AnswerValidationError)
        details = error.translation_diagnostics
        self.assertEqual(details["stage"], "marker_restoration")
        self.assertEqual(details["source_index"], 1)
        self.assertEqual(details["source"], sources[1])
        self.assertEqual(details["raw_output"], "그림자 손을 평가합니다.")
        self.assertEqual(details["marker_mapping"], {"ZXQAAA": "Shadow Hand"})
        self.assertNotIn("그림자 손", str(error))
        self.assertEqual(translator.cache_info()["size"], 0)

    def test_dataset_context_restores_unlisted_source_name(self):
        translator = self.make_translator()
        with patch.object(translator, "_translate_model_batch", return_value=["ZXQAAA 데이터셋은 작습니다."]):
            self.assertEqual(translator.translate(["The Helios dataset is small."]), ["Helios 데이터셋은 작습니다."])

    def test_name_declaration_in_batch_protects_other_sources_and_cache_context(self):
        translator = self.make_translator()
        declaration, source = "The Helios dataset is small.", "Helios has low error."
        with patch.object(translator, "_translate_model_batch", side_effect=[
            ["헬리오스의 오류는 낮습니다."],
            ["ZXQAAA 데이터셋은 작습니다.", "ZXQAAA의 오류는 낮습니다."],
        ]) as model:
            self.assertEqual(translator.translate([source]), ["헬리오스의 오류는 낮습니다."])
            self.assertEqual(translator.translate([declaration, source]), [
                "Helios 데이터셋은 작습니다.", "Helios의 오류는 낮습니다.",
            ])
            self.assertEqual(model.call_args.args[0], ["The ZXQAAA dataset is small.", "ZXQAAA has low error."])
            self.assertEqual(translator.translate([source, declaration, declaration]), [
                "Helios의 오류는 낮습니다.", "Helios 데이터셋은 작습니다.", "Helios 데이터셋은 작습니다.",
            ])
            self.assertEqual(model.call_count, 2)
            self.assertEqual(translator.translate([source]), ["헬리오스의 오류는 낮습니다."])
            self.assertEqual(model.call_count, 2)
        self.assertEqual(translator.cache_info()["size"], 3)

    def test_declared_name_omission_fails_atomically_and_unrelated_declarations_share_cache(self):
        translator = self.make_translator()
        sources = ["The Helios dataset is small.", "Helios has low error."]
        with patch.object(translator, "_translate_model_batch", return_value=[
            "ZXQAAA 데이터셋은 작습니다.", "헬리오스의 오류는 낮습니다.",
        ]), self.assertRaises(TranslationFidelityError):
            translator.translate(sources)
        self.assertEqual(translator.cache_info()["size"], 0)
        source = "The score is 1."
        with patch.object(translator, "_translate_model_batch", side_effect=[
            ["점수는 1입니다."], ["ZXQAAA 데이터셋은 작습니다."],
        ]) as model:
            translator.translate([source])
            translator.translate([sources[0], source])
            self.assertEqual(model.call_args.args[0], ["The ZXQAAA dataset is small."])
            self.assertEqual(model.call_count, 2)

    def test_fidelity_diagnostics_identify_failed_quantity_and_are_bounded(self):
        translator = self.make_translator()
        source = "The accuracy is 95.2 percent."
        with patch.object(translator, "_translate_model_batch", return_value=["정확도는 99%입니다."]):
            with self.assertRaises(TranslationFidelityError) as raised:
                translator.translate([source])
        self.assertEqual(raised.exception.translation_diagnostics["stage"], "source_fidelity_validation")
        self.assertEqual(raised.exception.translation_diagnostics["restored_output"], "정확도는 99%입니다.")
        bounded = TranslationFidelityError("failed", stage="test", source="a" * 5000,
                                            raw_output="b" * 5000, marker_mapping={str(i): "c" * 700 for i in range(80)})
        self.assertEqual(len(bounded.translation_diagnostics["source"]), 4000)
        self.assertTrue(bounded.translation_diagnostics["source_truncated"])
        self.assertEqual(len(bounded.translation_diagnostics["marker_mapping"]), 64)
        self.assertEqual(len(bounded.translation_diagnostics["source_sha256"]), 64)

    def test_lru_eviction_and_explicit_clear_are_bounded(self):
        translator = self.make_translator(cache_size=2)
        with patch.object(translator, "_translate_model_batch", side_effect=lambda inputs: ["검색 절차입니다."] * len(inputs)) as model:
            for source in ("A procedure.", "B procedure.", "A procedure.", "C procedure.", "B procedure."):
                translator.translate([source])
            self.assertEqual(model.call_count, 4)
        self.assertEqual(translator.cache_info()["size"], 2)
        self.assertEqual(translator.cache_info()["evictions"], 2)
        translator.clear_cache(reset_stats=True)
        self.assertEqual(translator.cache_info()["size"], 0)
        self.assertEqual(translator.cache_info()["hits"], 0)
        self.assertEqual(translator.cache_info()["misses"], 0)

    def test_zero_capacity_disables_cache_but_keeps_validation(self):
        translator = self.make_translator(cache_size=0)
        with patch.object(translator, "_translate_model_batch", return_value=["정확도는 95%입니다."]) as model:
            translator.translate(["The accuracy is 95 percent."])
            translator.translate(["The accuracy is 95 percent."])
            self.assertEqual(model.call_count, 2)
        self.assertFalse(translator.cache_info()["enabled"])
        self.assertEqual(translator.cache_info()["size"], 0)

    def test_config_and_glossary_changes_do_not_reuse_stale_results(self):
        translator = self.make_translator()
        with patch.object(translator, "_translate_model_batch", return_value=["정확도는 95%입니다."]) as model:
            source = "The accuracy is 95 percent."
            translator.translate([source])
            translator.config["num_beams"] += 1
            translator.translate([source])
            translator.glossary["entries"][0]["target"] = "그리퍼"
            translator.translate([source])
            translator.config["revision"] = "0" * 40
            translator.translate([source])
            self.assertEqual(model.call_count, 4)

    def test_bad_batch_shape_and_model_errors_never_enter_cache(self):
        for output in ([], ["점수는 1입니다.", "추가 문장입니다."]):
            translator = self.make_translator()
            with patch.object(translator, "_translate_model_batch", return_value=output), self.assertRaises(ValueError):
                translator.translate(["The score is 1."])
            self.assertEqual(translator.cache_info()["size"], 0)
        translator = self.make_translator()
        with patch.object(translator, "_translate_model_batch", side_effect=RuntimeError("model failed")), self.assertRaises(RuntimeError):
            translator.translate(["The score is 1."])
        self.assertEqual(translator.cache_info()["size"], 0)

    def test_cache_capacity_rejects_unbounded_or_invalid_values(self):
        for capacity in (-1, 257, True, 1.5):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                self.make_translator(capacity)

    def test_metadata_exposes_glossary_identity_and_observed_cache_counts(self):
        translator = self.make_translator()
        info = translator.metadata()
        self.assertEqual(len(info["glossary_fingerprint"]), 64)
        self.assertEqual(info["glossary_path"], "config/translation_glossary.json")
        self.assertEqual(info["cache"]["capacity"], 128)
        self.assertEqual(info["cache"]["hits"], 0)

    def test_retry_only_retranslates_invalid_source_once_and_caches_valid_batch(self):
        translator = self.make_translator()
        translator.config["retry_on_validation_failure"] = True
        sources = ["The score is 1.", "Shadow Hand is evaluated."]
        with patch.object(translator, "_translate_model_batch", side_effect=[
            ["점수는 1입니다.", "그림자 손을 평가합니다."], ["ZXQAAA를 평가합니다."],
        ]) as model:
            self.assertEqual(translator.translate(sources), ["점수는 1입니다.", "Shadow Hand를 평가합니다."])
            self.assertEqual(model.call_count, 2)
            self.assertEqual(model.call_args.args[0], ["ZXQAAA is evaluated."])
            self.assertEqual(model.call_args.kwargs, {"num_beams": 2})
            self.assertEqual(translator.cache_info()["size"], 2)
            info = translator.metadata()["validation_retry"]
            self.assertEqual((info["attempts"], info["successes"], info["failures"]), (1, 1, 0))
            self.assertEqual(info["last_request"]["effective_num_beams"], [1, 2])
            translator.translate(sources)
            self.assertEqual(model.call_count, 2)
            self.assertEqual(translator.metadata()["validation_retry"]["last_request"]["effective_num_beams"], [])
        self.assertEqual(translator.config["num_beams"], 1)

    def test_retry_never_accepts_changed_numbers_and_failure_keeps_both_diagnostics(self):
        translator = self.make_translator()
        translator.config["retry_on_validation_failure"] = True
        with patch.object(translator, "_translate_model_batch", side_effect=[
            ["정확도는 99%입니다."], ["정확도는 98%입니다."],
        ]) as model, self.assertRaises(TranslationFidelityError) as raised:
            translator.translate(["The accuracy is 95 percent."])
        self.assertEqual(model.call_count, 2)
        self.assertEqual(translator.cache_info()["size"], 0)
        details = raised.exception.translation_diagnostics
        self.assertTrue(details["retry_attempted"])
        self.assertEqual(details["retry_num_beams"], 2)
        self.assertEqual(details["raw_output"], "정확도는 98%입니다.")
        self.assertEqual(details["initial_failure"]["raw_output"], "정확도는 99%입니다.")
        self.assertEqual(translator.metadata()["validation_retry"]["failures"], 1)

    def test_successful_retry_before_later_failure_does_not_commit_partial_cache(self):
        translator = self.make_translator()
        translator.config["retry_on_validation_failure"] = True
        with patch.object(translator, "_translate_model_batch", side_effect=[
            ["그림자 손을 평가합니다.", "점수는 3입니다."], ["ZXQAAA를 평가합니다."], ["점수는 4입니다."],
        ]), self.assertRaises(TranslationFidelityError):
            translator.translate(["Shadow Hand is evaluated.", "The score is 2."])
        self.assertEqual(translator.cache_info()["size"], 0)
        info = translator.metadata()["validation_retry"]
        self.assertEqual((info["attempts"], info["successes"], info["failures"]), (2, 1, 1))
        translator.clear_cache(reset_stats=True)
        self.assertEqual(translator.metadata()["validation_retry"]["attempts"], 0)

    def test_structural_input_and_batch_errors_are_not_retried(self):
        for model_result in (ValueError("input exceeds token budget"), []):
            translator = self.make_translator()
            translator.config["retry_on_validation_failure"] = True
            kwargs = {"side_effect": model_result} if isinstance(model_result, Exception) else {"return_value": model_result}
            with patch.object(translator, "_translate_model_batch", **kwargs) as model, self.assertRaises(ValueError):
                translator.translate(["The score is 1."])
            self.assertEqual(model.call_count, 1)
            self.assertEqual(translator.metadata()["validation_retry"]["attempts"], 0)
            self.assertEqual(translator.cache_info()["size"], 0)

    def test_retry_is_opt_in_for_historical_configs_and_config_validation_is_bounded(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "translation.json"
            config = load_translation_config()
            config.pop("retry_on_validation_failure")
            config.pop("retry_num_beams")
            path.write_text(json.dumps(config), encoding="utf-8")
            with patch.object(LocalEnglishKoreanTranslator, "_load"):
                translator = LocalEnglishKoreanTranslator(path)
            self.assertFalse(translator.config.get("retry_on_validation_failure", False))
            for enabled, beams in (("yes", 2), (True, True), (True, 5), (True, 1)):
                variant = {**config, "retry_on_validation_failure": enabled, "retry_num_beams": beams}
                path.write_text(json.dumps(variant), encoding="utf-8")
                with self.subTest(enabled=enabled, beams=beams), self.assertRaises(ValueError):
                    load_translation_config(path)


class FakeTensor:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def to(self, device):
        return self

    def tolist(self):
        return self.rows


class TranslationTokenBudgetTests(unittest.TestCase):
    def make_translator(self, input_tokens, generated_tokens):
        with patch.object(LocalEnglishKoreanTranslator, "_load"):
            translator = LocalEnglishKoreanTranslator()
        translator.device = "cpu"
        translator._torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
        translator.model = SimpleNamespace(generate=lambda **kwargs: FakeTensor([generated_tokens]))
        translator.tokenizer = SimpleNamespace()
        tokenizer = unittest.mock.Mock()
        tokenizer.return_value = {"input_ids": FakeTensor([[1] * input_tokens])}
        tokenizer.convert_tokens_to_ids.return_value = 100
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 1
        tokenizer.batch_decode.return_value = ["검색 절차입니다."]
        translator.tokenizer = tokenizer
        return translator

    def test_overlong_input_is_rejected_before_model_generation(self):
        translator = self.make_translator(513, [2, 100, 3, 2])
        with patch.object(translator.model, "generate") as generate, self.assertRaises(ValueError):
            translator._translate_model_batch(["long source"])
        generate.assert_not_called()
        self.assertFalse(translator.tokenizer.call_args.kwargs["truncation"])

    def test_output_without_terminal_eos_is_rejected(self):
        translator = self.make_translator(12, [2, 100, 3, 4])
        with self.assertRaises(ValueError):
            translator._translate_model_batch(["source"])

    def test_missing_end_token_configuration_is_rejected(self):
        translator = self.make_translator(12, [2, 100, 3, 2])
        translator.tokenizer.eos_token_id = None
        with self.assertRaises(RuntimeError):
            translator._translate_model_batch(["source"])

    def test_overlong_sentence_does_not_enter_cache(self):
        translator = self.make_translator(513, [2, 100, 3, 2])
        with self.assertRaises(ValueError):
            translator.translate(["The input is long."])
        self.assertEqual(translator.cache_info()["size"], 0)

    def test_finished_padded_output_is_accepted(self):
        translator = self.make_translator(12, [2, 100, 3, 2, 1, 1])
        self.assertEqual(translator._translate_model_batch(["source"]), ["검색 절차입니다."])

    def test_beam_override_does_not_change_primary_config(self):
        translator = self.make_translator(12, [2, 100, 3, 2])
        with patch.object(translator.model, "generate", return_value=FakeTensor([[2, 100, 3, 2]])) as generate:
            translator._translate_model_batch(["source"], num_beams=2)
        self.assertEqual(generate.call_args.kwargs["num_beams"], 2)
        self.assertEqual(translator.config["num_beams"], 1)


if __name__ == "__main__":
    unittest.main()
