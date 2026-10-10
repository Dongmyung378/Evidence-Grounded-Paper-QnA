"""Offline checks for leakage prevention and complete final-answer evaluation."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from quality_benchmark import (
    IMPLEMENTATIONS, RUNTIME_CONFIGS, REFUSALS, assert_writable_output, evaluate, fixture_chunks, inspect_final_output, load_dataset,
    prediction_template, run_api, run_pipeline, save, smoke_predictions, validate_dataset,
    marker_present, runtime_freeze, validate_runtime_freeze, preserve_runtime_sources,
    main, verify_source_pdfs,
)


class QualityBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.config, self.manifest, self.questions, self.rubric = load_dataset()
        self.predictions = smoke_predictions(self.config, self.manifest, self.questions)

    def validate(self, questions=None, manifest=None):
        return validate_dataset(self.config, manifest or self.manifest,
                                questions or self.questions, self.rubric)

    def test_runtime_snapshot_retains_exact_bytes_and_rejects_replacement(self):
        import quality_benchmark
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'scripts/runtime.py'
            source.parent.mkdir()
            original = b"claim = '100 GB'\r\n"
            source.write_bytes(original)
            wanted = hashlib.sha256(original).hexdigest()
            receipt = {'runtime_config_sha256': {}, 'implementation_sha256': {'scripts/runtime.py': wanted}}
            output = root / 'runs/freeze.json'
            with patch.object(quality_benchmark, 'ROOT', root):
                snapshot = preserve_runtime_sources(receipt, output)
                archived = output.parent / snapshot['directory'] / 'scripts/runtime.py'
                self.assertEqual(archived.read_bytes(), original)
                self.assertEqual(preserve_runtime_sources(receipt, output), snapshot)
                archived.write_bytes(b'modified snapshot')
                with self.assertRaisesRegex(ValueError, 'Existing runtime snapshot differs'):
                    preserve_runtime_sources(receipt, output)
                self.assertEqual(archived.read_bytes(), b'modified snapshot')
                self.assertEqual(source.read_bytes(), original)

    def test_source_changed_before_snapshot_cannot_be_recorded(self):
        import quality_benchmark
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'runtime.py'
            source.write_bytes(b'new source')
            receipt = {'runtime_config_sha256': {},
                       'implementation_sha256': {'runtime.py': hashlib.sha256(b'old source').hexdigest()}}
            with patch.object(quality_benchmark, 'ROOT', root), self.assertRaisesRegex(ValueError, 'changed before snapshot'):
                preserve_runtime_sources(receipt, root / 'runs/freeze.json')
            self.assertFalse((root / 'runs/freeze-sources').exists())

    def test_absolute_snapshot_key_cannot_write_over_a_runtime_source(self):
        import quality_benchmark
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'runtime.py'
            source.write_bytes(b'original')
            receipt = {'runtime_config_sha256': {},
                       'implementation_sha256': {str(source): hashlib.sha256(b'original').hexdigest()}}
            with patch.object(quality_benchmark, 'ROOT', root), self.assertRaisesRegex(ValueError, 'relative project paths'):
                preserve_runtime_sources(receipt, root / 'runs/freeze.json')
            self.assertEqual(source.read_bytes(), b'original')

    def test_changed_archived_source_invalidates_a_matching_runtime_receipt(self):
        import quality_benchmark
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'runtime.py'
            source.write_bytes(b'original source')
            expected = {'schema_version': 1, 'benchmark_id': 'test', 'runtime_config_sha256': {},
                        'implementation_sha256': {'runtime.py': hashlib.sha256(source.read_bytes()).hexdigest()},
                        'harness_sha256': 'test', 'frozen_inputs': {}, 'paper_sources_sha256': {}}
            path = root / 'runs/freeze.json'
            receipt = copy.deepcopy(expected)
            with patch.object(quality_benchmark, 'ROOT', root), patch.object(quality_benchmark, 'runtime_freeze', return_value=expected):
                receipt['source_snapshot'] = preserve_runtime_sources(receipt, path)
                path.write_text(json.dumps(receipt), encoding='utf-8')
                validate_runtime_freeze({}, {}, path)
                (path.parent / receipt['source_snapshot']['directory'] / 'runtime.py').write_bytes(b'altered archive')
                with self.assertRaisesRegex(ValueError, 'snapshot missing or modified'):
                    validate_runtime_freeze({}, {}, path)

    def test_complete_frozen_synthetic_fixture_is_honest(self):
        counts = self.validate()
        self.assertEqual(counts["papers"], 4)
        self.assertEqual(counts["outputs_required"], 40)
        self.assertEqual(counts["meaning_groups"], 20)
        self.assertEqual(counts["groups_by_split"], {"dev": 10, "holdout": 10})
        self.assertFalse(counts["independent_human_review"])
        self.assertEqual(counts["purpose"], "synthetic_integration_only")

    def test_translation_cannot_cross_dev_and_holdout(self):
        questions = copy.deepcopy(self.questions)
        questions[1]["split"] = "holdout"
        with self.assertRaisesRegex(ValueError, "split leakage"):
            self.validate(questions=questions)

    def test_translation_cannot_move_to_another_paper(self):
        questions = copy.deepcopy(self.questions)
        questions[1]["paper_id"] = "fixture-birch"
        with self.assertRaisesRegex(ValueError, "Translation group crosses"):
            self.validate(questions=questions)

    def test_duplicate_source_cannot_be_given_two_paper_ids_across_splits(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["papers"][2]["source_path"] = manifest["papers"][0]["source_path"]
        manifest["papers"][2]["source_sha256"] = manifest["papers"][0]["source_sha256"]
        with self.assertRaisesRegex(ValueError, "Identical paper sources"):
            self.validate(manifest=manifest)

    def test_duplicate_source_within_one_split_cannot_inflate_paper_count(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["papers"][1]["source_path"] = manifest["papers"][0]["source_path"]
        manifest["papers"][1]["source_sha256"] = manifest["papers"][0]["source_sha256"]
        with self.assertRaisesRegex(ValueError, "Duplicate paper source bytes"):
            self.validate(manifest=manifest)

    def test_case_ids_must_be_unique(self):
        questions = copy.deepcopy(self.questions)
        questions[1]["case_id"] = questions[0]["case_id"]
        with self.assertRaisesRegex(ValueError, "Case IDs"):
            self.validate(questions=questions)

    def test_duplicate_questions_are_normalized(self):
        questions = copy.deepcopy(self.questions)
        questions[10]["question"] = "  " + questions[0]["question"].upper() + "  "
        with self.assertRaisesRegex(ValueError, "Duplicate normalized"):
            self.validate(questions=questions)

    def test_every_meaning_group_has_both_languages_once(self):
        with self.assertRaisesRegex(ValueError, "exactly one EN and one KO"):
            self.validate(questions=self.questions[1:])

    def test_all_five_categories_required_in_each_split(self):
        questions = [row for row in self.questions if row["question_type"] != "limitation"]
        with self.assertRaisesRegex(ValueError, "all five question types"):
            self.validate(questions=questions)

    def test_all_answerability_classes_required_in_each_split(self):
        questions = [row for row in self.questions if row["answerability"] != "false_premise"]
        with self.assertRaisesRegex(ValueError, "all three answerability"):
            self.validate(questions=questions)

    def test_source_hash_changes_are_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["papers"][0]["source_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            self.validate(manifest=manifest)

    def test_synthetic_perfect_score_cannot_claim_real_accuracy(self):
        report = evaluate(self.config, self.manifest, self.questions, self.predictions)
        self.assertEqual(report["metrics"]["units"], 20)
        self.assertEqual(report["counts"]["language_outputs"], 40)
        self.assertEqual(report["metrics"]["answerable_retention"], 1.0)
        self.assertEqual(report["metrics"]["unsupported_final_refusal_rate"], 1.0)
        self.assertEqual(report["execution"]["kind"], "stub_fixture")
        self.assertTrue(report["execution"]["gold_used_to_construct_outputs"])
        self.assertFalse(report["claim_boundary"]["real_paper_holdout_accuracy_claim"])
        self.assertFalse(report["claim_boundary"]["independent_human_review"])
        self.assertIsNone(report["claim_boundary"]["automatic_semantic_accuracy"])
        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(report["metrics"]["unit"], "meaning_group")
        self.assertNotIn('independent_meaning', json.dumps(report))
        self.assertFalse(report["claim_boundary"]["statistical_independence_assumed"])

    def test_one_failed_translation_fails_one_group_not_half_a_case(self):
        self.predictions["records"][1]["output"] = None
        report = evaluate(self.config, self.manifest, self.questions, self.predictions)
        self.assertEqual(report["metrics"]["answerable_retention"], 0.9)
        self.assertEqual(report["metrics"]["invalid_outputs"], 1)
        self.assertEqual(report["by_language"]["en"]["answerable_retention"], 1.0)
        self.assertEqual(report["by_language"]["ko"]["answerable_retention"], 0.9)

    def test_refusal_label_cannot_hide_a_fabricated_final_answer(self):
        index = next(i for i, row in enumerate(self.questions) if row["answerability"] == "unsupported")
        self.predictions["records"][index]["output"]["response"]["answer"] = "The study enrolled 100 human participants."
        report = evaluate(self.config, self.manifest, self.questions, self.predictions)
        self.assertEqual(report["metrics"]["unsupported_final_refusal_rate"], round(5 / 6, 4))
        self.assertEqual(report["metrics"]["invalid_outputs"], 1)

    def test_sufficient_label_cannot_turn_refusal_into_retained_answer(self):
        self.predictions["records"][0]["output"]["response"]["answer"] = REFUSALS["en"]
        record = inspect_final_output(self.questions[0], self.predictions["records"][0])
        self.assertFalse(record["valid_contract"])
        self.assertFalse(record["final_answer"])
        self.assertFalse(record["final_refusal"])

    def test_retrieval_policy_is_never_used_as_final_decision(self):
        output = self.predictions["records"][0]["output"]
        output["pipeline"]["abstention"] = {"decision": {"abstain": True}}
        record = inspect_final_output(self.questions[0], self.predictions["records"][0])
        self.assertTrue(record["final_answer"])
        self.assertFalse(record["final_refusal"])

    def test_missing_prediction_is_rejected_instead_of_removed_from_denominator(self):
        self.predictions["records"].pop()
        with self.assertRaisesRegex(ValueError, "cover every case exactly"):
            evaluate(self.config, self.manifest, self.questions, self.predictions)

    def test_duplicate_prediction_is_rejected(self):
        self.predictions["records"].append(self.predictions["records"][0])
        with self.assertRaisesRegex(ValueError, "case IDs must be unique"):
            evaluate(self.config, self.manifest, self.questions, self.predictions)

    def test_unresolved_citation_is_invalid(self):
        self.predictions["records"][0]["output"]["response"]["evidence_ids"] = ["ev-01-invented"]
        record = inspect_final_output(self.questions[0], self.predictions["records"][0])
        self.assertFalse(record["valid_contract"])

    def test_citations_from_other_papers_are_invalid(self):
        self.predictions["records"][0]["output"]["cited_evidence"][0]["paper_id"] = "another-paper"
        record = inspect_final_output(self.questions[0], self.predictions["records"][0])
        self.assertFalse(record["valid_contract"])

    def test_public_api_response_is_evaluated(self):
        original = self.predictions["records"][0]["output"]
        payload = original["response"]
        public = {"paper_id": self.questions[0]["paper_id"], "question": self.questions[0]["question"],
                  "question_language": self.questions[0]["language"],
                  "answer": payload["answer"], "sufficiency": payload["sufficiency"],
                  "abstention_reason": payload["abstention_reason"], "evidence": original["cited_evidence"],
                  "runtime": {"fallback_used": False}}
        record = inspect_final_output(self.questions[0], {"output": public})
        self.assertTrue(record["valid_contract"])
        self.assertTrue(record["all_required_pages_cited"])

    def test_public_api_wrong_paper_is_invalid_even_without_evidence_paper_id(self):
        original = copy.deepcopy(self.predictions["records"][0]["output"])
        for evidence in original["cited_evidence"]:
            del evidence["paper_id"]
        public = {"paper_id": "paper-" + "f" * 32, "question": self.questions[0]["question"],
                  "question_language": "en", "evidence": original["cited_evidence"],
                  "runtime": {}, **{key: original["response"][key] for key in ("answer", "sufficiency", "abstention_reason")}}
        record = inspect_final_output(self.questions[0], {"output": public})
        self.assertFalse(record["valid_contract"])
        self.assertFalse(record["all_required_pages_cited"])

    def test_mismatched_question_or_language_identity_is_invalid(self):
        self.predictions["records"][0]["output"]["query"] = self.questions[2]["question"]
        self.assertFalse(inspect_final_output(self.questions[0], self.predictions["records"][0])["valid_contract"])
        self.predictions["records"][0]["output"]["query"] = self.questions[0]["question"]
        self.predictions["records"][0]["output"]["question_language"] = "ko"
        self.assertFalse(inspect_final_output(self.questions[0], self.predictions["records"][0])["valid_contract"])

    def test_locator_chunk_id_must_match_the_scoped_evidence_id(self):
        evidence = self.predictions["records"][0]["output"]["cited_evidence"][0]
        evidence["locator"]["chunk_id"] = "another-paper-p002-c001"
        self.assertFalse(inspect_final_output(self.questions[0], self.predictions["records"][0])["valid_contract"])

    def test_evidence_id_cannot_point_to_another_paper_with_valid_pages(self):
        evidence = self.predictions["records"][0]["output"]["cited_evidence"][0]
        evidence["evidence_id"] = "ev-01-another-paper-p002-c001"
        evidence["chunk_id"] = evidence["locator"]["chunk_id"] = "another-paper-p002-c001"
        self.predictions["records"][0]["output"]["response"]["evidence_ids"] = [evidence["evidence_id"]]
        self.assertFalse(inspect_final_output(self.questions[0], self.predictions["records"][0])["valid_contract"])

    def test_output_guard_preserves_frozen_sources_and_evaluation(self):
        paths = [ROOT / "data/evaluation/final_evaluation.json", ROOT / "data/baselines/new.json",
                 ROOT / "config/quality_benchmark.json", ROOT / "data/quality_benchmark/questions.jsonl",
                 ROOT / "data/quality_benchmark/papers/fixture-alder.json", ROOT / "scripts/qna_pipeline.py"]
        for path in paths:
            original = path.read_bytes() if path.is_file() else None
            with self.assertRaisesRegex(ValueError, "overwrite"):
                save(path, {"unsafe": True})
            self.assertEqual(path.read_bytes() if path.is_file() else None, original)

    def test_output_guard_protects_current_prediction_input(self):
        path = ROOT / "data/quality_benchmark/dev_predictions.local.json"
        with self.assertRaisesRegex(ValueError, "prediction input"):
            assert_writable_output(path, {path})
        assert_writable_output(ROOT / "data/quality_benchmark/dev_report.local.json", {path})

    def test_template_is_explicitly_not_run(self):
        template = prediction_template(self.config, self.questions)
        self.assertEqual(template["execution"]["status"], "not_run")
        self.assertTrue(all(record["output"] is None for record in template["records"]))

    def test_explicit_dev_only_run_never_counts_holdout(self):
        self.predictions["execution"]["split"] = "dev"
        self.predictions["records"] = self.predictions["records"][:20]
        report = evaluate(self.config, self.manifest, self.questions, self.predictions)
        self.assertEqual(report["counts"]["language_outputs"], 20)
        self.assertEqual(report["metrics"]["units"], 10)
        self.assertEqual(report["by_split"]["holdout"]["units"], 0)
        self.assertIsNone(report["by_split"]["holdout"]["answerable_retention"])

    def test_fixture_chunks_have_no_questions_or_gold_labels(self):
        chunks = fixture_chunks(self.manifest, "dev")
        self.assertEqual(len(chunks), 12)
        self.assertEqual({row["paper_id"] for row in chunks}, {"fixture-alder", "fixture-birch"})
        self.assertTrue(all("reference_answer" not in row and "gold_pages" not in row for row in chunks))

    def test_runtime_mapping_cannot_collapse_papers(self):
        mapping = {paper["paper_id"]: "paper-" + "a" * 32 for paper in self.manifest["papers"]}
        with self.assertRaisesRegex(ValueError, "cannot map to one runtime paper"):
            run_api(self.config, self.questions, "http://localhost:8000", mapping, "test", 1)

    def test_runtime_code_change_during_execution_prevents_report_creation(self):
        changed = {"value": False}

        class FakeRetrieval:
            def __init__(self, **kwargs):
                pass

        class FakePipeline:
            def __init__(self, **kwargs):
                pass

            def ask(self, *args, **kwargs):
                changed["value"] = True
                return {"simulated_output": True}

        def simulated_digest(path):
            return "after-edit" if path.name == "qna_pipeline.py" and changed["value"] else "before-edit"

        modules = {"production_retrieval": types.SimpleNamespace(ProductionRetrieval=FakeRetrieval),
                   "qna_pipeline": types.SimpleNamespace(GroundedQAPipeline=FakePipeline)}
        with patch.dict(sys.modules, modules), patch("quality_benchmark.digest", side_effect=simulated_digest), patch("builtins.print"):
            with self.assertRaisesRegex(ValueError, "Runtime sources changed.*scripts/qna_pipeline.py"):
                run_pipeline(self.config, self.manifest, self.questions[:2], split="dev", offline=True,
                             revision="test", cache_path=ROOT / "data/processed/quality-regression.npz", device=None)

    def test_runtime_freeze_covers_transitive_local_import_dependencies(self):
        # Parse source instead of importing model libraries or reading real holdout inputs.
        pending = {"scripts/qna_pipeline.py", "scripts/production_retrieval.py",
                   "scripts/quality_benchmark.py", "app/question_engine.py"}
        discovered = set()
        while pending:
            relative = pending.pop()
            if relative in discovered:
                continue
            discovered.add(relative)
            tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                         else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
                for name in names:
                    module_path = Path(*name.split(".")).with_suffix(".py")
                    for directory in (ROOT / "scripts", ROOT):
                        candidate = directory / module_path
                        if candidate.is_file():
                            pending.add(candidate.relative_to(ROOT).as_posix())
        self.assertTrue(discovered <= set(IMPLEMENTATIONS), sorted(discovered - set(IMPLEMENTATIONS)))
        self.assertTrue({"config/answer_output_schema.json", "config/generation_runtime.json"} <= set(RUNTIME_CONFIGS))

    def test_transitive_dependency_changes_during_execution_prevent_report_creation(self):
        changed = {"value": False, "path": None}

        class FakeRetrieval:
            def __init__(self, **kwargs):
                pass

        class FakePipeline:
            def __init__(self, **kwargs):
                pass

            def ask(self, *args, **kwargs):
                changed["value"] = True
                return {"simulated_output": True}

        def simulated_digest(path):
            return "after-edit" if path == ROOT / changed["path"] and changed["value"] else "before-edit"

        modules = {"production_retrieval": types.SimpleNamespace(ProductionRetrieval=FakeRetrieval),
                   "qna_pipeline": types.SimpleNamespace(GroundedQAPipeline=FakePipeline)}
        dependencies = ["scripts/bm25_retrieval.py", "scripts/hybrid_retrieval.py", "scripts/retrieval_common.py",
                        "scripts/grounded_answer_contract.py", "scripts/answer_quality.py", "scripts/local_llm.py",
                        "app/question_engine.py", "config/answer_output_schema.json", "config/generation_runtime.json"]
        for relative in dependencies:
            with self.subTest(path=relative):
                changed.update(value=False, path=relative)
                with patch.dict(sys.modules, modules), patch("quality_benchmark.digest", side_effect=simulated_digest), patch("builtins.print"):
                    with self.assertRaisesRegex(ValueError, "Runtime sources changed") as error:
                        run_pipeline(self.config, self.manifest, self.questions[:2], split="dev", offline=True,
                                     revision="test", cache_path=ROOT / "data/processed/quality-regression.npz", device=None)
                    self.assertIn(relative, str(error.exception))

    def test_marker_boundaries_handle_korean_particles_names_and_decimal_values(self):
        for answer, marker in [('참값을 비교했다.', '참값'), ('그리퍼에서 확인했다.', '그리퍼'),
                               ('GOAG은 동작한다.', 'GOAG'), ('BarrettHand로 평가했다.', 'Barrett Hand'),
                               ('ground-truth was available.', 'ground truth'), ('값은 0.86입니다.', '0.86'),
                               ('75% of the speed.', '75%')]:
            self.assertTrue(marker_present(answer, marker), (answer, marker))
        for answer, marker in [('185.8% accuracy', '85.8'), ('0.8601 correlation', '0.86'),
                               ('someone else', 'one'), ('linearized approximation', 'linear'),
                               ('비참값을 확인했다.', '참값')]:
            self.assertFalse(marker_present(answer, marker), (answer, marker))


class RealPaperQualityBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.manifest, cls.questions, cls.rubric = load_dataset(
            ROOT / 'data/quality_benchmark/real_paper_config.json')

    def test_new_public_publisher_sources_are_disjoint_and_review_is_not_attested(self):
        result = validate_dataset(self.config, self.manifest, self.questions, self.rubric)
        self.assertEqual(result['papers'], 4)
        self.assertEqual(result['meaning_groups'], 32)
        self.assertEqual(result['outputs_required'], 64)
        self.assertEqual(result['groups_by_split'], {'dev': 16, 'holdout': 16})
        self.assertFalse(result['independent_human_review'])
        self.assertFalse(result['statistical_independence_assumed'])
        chunks = fixture_chunks(self.manifest, 'dev')
        self.assertTrue(all(len(chunk['text']) <= 1600 for chunk in chunks))
        self.assertTrue(all('gold_pages' not in chunk and 'reference_answer' not in chunk for chunk in chunks))

    def test_one_article_cannot_be_counted_as_two_differently_extracted_papers(self):
        manifest = copy.deepcopy(self.manifest)
        manifest['papers'][2]['arxiv_id'] = manifest['papers'][0]['arxiv_id']
        with self.assertRaisesRegex(ValueError, 'version-independent arXiv identities'):
            validate_dataset(self.config, manifest, self.questions, self.rubric)

    def test_missing_required_condition_is_observable_without_claiming_semantic_accuracy(self):
        dev = [row for row in self.questions if row['split'] == 'dev']
        predictions = smoke_predictions(self.config, self.manifest, dev)
        predictions['execution']['split'] = 'dev'
        target = next(i for i,row in enumerate(dev) if row['question_type'] == 'numeric')
        predictions['records'][target]['output']['response']['answer'] = 'SimCLR achieved 76.5% on ImageNet.'
        report = evaluate(self.config, self.manifest, self.questions, predictions)
        checked = report['records'][target]
        self.assertTrue(checked['valid_contract'])
        self.assertFalse(checked['all_required_markers_present'])
        self.assertLess(report['metrics']['all_required_markers_present_rate'], 1)
        self.assertIsNone(report['claim_boundary']['automatic_semantic_accuracy'])
        self.assertTrue(report['claim_boundary']['lexical_marker_coverage_is_not_semantic_accuracy'])

    def test_invalid_or_refused_supported_output_cannot_pass_marker_coverage(self):
        case = next(row for row in self.questions if row['split'] == 'dev' and row['answerability'] == 'answerable')
        prediction = smoke_predictions(self.config, self.manifest, [case])['records'][0]
        prediction['output']['response'] = {'answer': REFUSALS[case['language']], 'evidence_ids': [],
                                           'sufficiency': 'insufficient', 'abstention_reason': 'Deliberate test refusal.'}
        prediction['output']['cited_evidence'] = []
        self.assertFalse(inspect_final_output(case, prediction)['all_required_markers_present'])

    def test_new_holdout_is_blocked_without_pre_run_runtime_freeze(self):
        with self.assertRaisesRegex(ValueError, 'requires freeze-run'):
            run_pipeline(self.config, self.manifest, self.questions, split='holdout', offline=True,
                         revision='test', cache_path=ROOT / 'outputs/real-test-embeddings.npz', device='cpu')

    def test_frozen_receipt_rejects_a_changed_runtime_configuration(self):
        receipt = runtime_freeze(self.config, self.manifest)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'runtime-freeze.json'
            path.write_text(json.dumps(receipt), encoding='utf-8')
            self.assertIn('sha256', validate_runtime_freeze(self.config, self.manifest, path))
            receipt['runtime_config_sha256']['config/translation.json'] = '0' * 64
            path.write_text(json.dumps(receipt), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'runtime_config_sha256'):
                validate_runtime_freeze(self.config, self.manifest, path)

    def test_shared_output_guard_protects_new_real_paper_inputs(self):
        for path in (ROOT / self.config['manifest'], ROOT / self.config['questions'],
                     ROOT / self.manifest['papers'][0]['source_path']):
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                save(path, {'unsafe': True})
            self.assertEqual(path.read_bytes(), original)

    def test_source_pdf_checker_rejects_replaced_original_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / 'chen20j.pdf').write_bytes(b'%PDF replaced bytes')
            with self.assertRaisesRegex(ValueError, 'byte hash mismatch'):
                verify_source_pdfs(self.manifest, directory, 'dev')

    def test_api_holdout_cannot_bypass_pre_run_freeze(self):
        with tempfile.TemporaryDirectory() as folder:
            argv = ['quality_benchmark.py', '--config', str(ROOT / 'data/quality_benchmark/real_paper_config.json'),
                    'run-api', '--split', 'holdout', '--base-url', 'http://127.0.0.1:8000',
                    '--paper-map', str(Path(folder) / 'unused-map.json'), '--pipeline-revision', 'test',
                    '--output', str(Path(folder) / 'predictions.json')]
            with patch.object(sys, 'argv', argv), self.assertRaisesRegex(ValueError, 'requires freeze-run'):
                main()

    def test_translation_pair_cannot_silently_use_different_coverage_requirements(self):
        questions = copy.deepcopy(self.questions)
        questions[1]['required_markers'].pop()
        with self.assertRaisesRegex(ValueError, 'coverage markers differ'):
            validate_dataset(self.config, self.manifest, questions, self.rubric)


if __name__ == "__main__":
    unittest.main()
