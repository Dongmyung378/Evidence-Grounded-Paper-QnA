"""근거 확장 시 논문 격리, 문장 복원과 정확한 인용을 검증한다."""

import sys
import unittest
from types import SimpleNamespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from answer_coverage import (expand_answer_evidence, generate_coverage_answer,
                             rank_answer_sentences, sentence_candidates,
                             is_general_problem_question, coverage_decision, load_coverage_config)
from evidence_selector import evidence_object


def chunk(name, index, text, paper="uploaded-a", page=1):
    return {"chunk_id": name, "chunk_index": index, "paper_id": paper,
            "page": page, "section": "Results", "text": text}


def evidence(c, rank):
    return evidence_object({"chunk": c, "rank": rank, "score": 4.0}, rank)


class Model:
    def predict(self, pairs, **kwargs):
        return np.asarray([5.0] * len(pairs))


class CoverageTests(unittest.TestCase):
    def test_runtime_keeps_coverage_opt_in_and_resolves_original_citations(self):
        from qna_pipeline import GroundedQAPipeline

        class Retrieval:
            def __init__(self):
                self.pipeline = SimpleNamespace(chunks=[chunk("a", 1,
                    "The method reduces memory consumption by 40%.")], reranker=Model())

            def run(self, query, paper_id):
                return {"query": query, "paper_id": paper_id,
                    "evidence": [evidence(self.pipeline.chunks[0], 1)],
                    "candidates": [{"chunk_id": "a", "reranker_score": 4.0}],
                    "production": {"config_fingerprint": "test"},
                    "retrieval_diagnostics": {"top_dense_similarity": .9}}

        baseline = GroundedQAPipeline(retrieval=Retrieval(), enable_abstention=False)
        result = baseline.ask("How much memory is reduced?", "uploaded-a")
        self.assertFalse(result["pipeline"]["answer_coverage"]["enabled"])
        candidate = GroundedQAPipeline(retrieval=Retrieval(), enable_abstention=False,
                                       enable_answer_coverage=True)
        result = candidate.ask("How much memory is reduced?", "uploaded-a")
        self.assertTrue(result["pipeline"]["answer_coverage"]["enabled"])
        self.assertIn("40%", result["response"]["answer"])
        self.assertEqual(result["cited_evidence"][0]["chunk_id"], "a")

    def test_general_overview_does_not_accept_an_unrelated_subject(self):
        self.assertTrue(is_general_problem_question("What is the main problem addressed in this paper?"))
        self.assertTrue(is_general_problem_question("이 논문에서 다루는 주요 문제는 무엇인가?"))
        self.assertFalse(is_general_problem_question("What is the main problem with Jupiter in this paper?"))
        self.assertFalse(is_general_problem_question("이 논문에서 목성의 주요 문제는 무엇인가?"))

    def test_dense_guard_rejects_low_support_and_missing_diagnostics(self):
        result = {"paper_id": "a", "query": "How many satellites are described?",
                  "retrieval_diagnostics": {"top_dense_similarity": .70}}
        decision = {"abstain": False, "reason_code": None}
        config = load_coverage_config()
        checked,_ = coverage_decision(result, decision, [], config)
        self.assertTrue(checked["abstain"])
        result["retrieval_diagnostics"] = {}
        checked,_ = coverage_decision(result, decision, [], config)
        self.assertTrue(checked["abstain"])

    def test_neighbor_expansion_cannot_cross_paper_or_page(self):
        chunks = [chunk("a", 1, "The method evaluates memory usage with a fixed budget."),
                  chunk("b", 2, "The accuracy remains stable for all tested examples."),
                  chunk("foreign", 2, "Private unrelated text.", paper="uploaded-b"),
                  chunk("other-page", 3, "Next page text.", page=2)]
        result = {"paper_id": "uploaded-a", "evidence": [evidence(chunks[0], 1)],
                  "candidates": [{"chunk_id": "a", "reranker_score": 4.0}]}
        expanded = expand_answer_evidence(result, chunks)
        self.assertEqual({e["chunk_id"] for e in expanded}, {"a", "b"})

    def test_overlapping_fragments_recover_exact_sentence_and_both_sources(self):
        overlap = "repeated runs on the held-out benchmark"
        left = "We evaluated the proposed method using " + overlap
        right = overlap + " and measured accuracy and memory consumption."
        source = [evidence(chunk("a", 1, left), 1), evidence(chunk("b", 2, right), 2)]
        sentences = sentence_candidates(source)
        self.assertEqual(len(sentences), 1)
        text_by_id = {e["evidence_id"]: e["text"] for e in source}
        reconstructed = "".join(text_by_id[s["evidence_id"]][s["start"]:s["end"]]
                                for s in sentences[0]["source_spans"])
        self.assertEqual(reconstructed, sentences[0]["text"])
        result = generate_coverage_answer(None, Model(), "How is the method evaluated?", source)
        self.assertFalse(result["fallback_used"])
        self.assertEqual(set(result["response"]["evidence_ids"]), set(text_by_id))

    def test_incomplete_sentence_is_not_returned(self):
        source = [evidence(chunk("a", 1, "The final experiment shows improvement beyond specif"), 1)]
        self.assertEqual(sentence_candidates(source), [])

    def test_section_local_chunk_indexes_do_not_reorder_page_neighbors(self):
        chunks = [chunk("p001-c001", 1, "The first section contains a complete source sentence."),
                  chunk("p001-c002", 2, "The second chunk ends the first section of this paper."),
                  chunk("p001-c003", 1, "A new section resets its local index to one."),
                  chunk("p001-c004", 2, "The final chunk contains the required next context.")]
        result = {"paper_id":"uploaded-a", "evidence":[evidence(chunks[1],1)],
                  "candidates":[{"chunk_id":"p001-c002","reranker_score":4}]}
        self.assertEqual({e["chunk_id"] for e in expand_answer_evidence(result,chunks)},
                         {"p001-c001","p001-c002","p001-c003"})

    def test_reference_and_figure_caption_are_excluded(self):
        c = chunk("a", 1, "Smith and colleagues described a different method in their review.")
        c["section"] = "References"
        source = [evidence(c, 1), evidence(chunk("b", 2, "Figure 3. The models are shown in magenta and sampled cells in teal."), 2)]
        self.assertFalse(any(s["section"] == "References" for s in sentence_candidates(source)))

    def test_nonfinite_selector_scores_are_not_accepted(self):
        class BrokenModel:
            def predict(self, *args, **kwargs):
                return np.asarray([float("nan"), float("nan")])
        source = [evidence(chunk("a", 1, "The method reduces memory consumption by forty percent."), 1)]
        self.assertEqual(rank_answer_sentences("How much memory?", source, BrokenModel()), [])

    def test_unsupported_translation_number_is_rejected(self):
        class Translator:
            def translate(self, sentences):
                return ["메모리 사용량이 99% 감소했습니다."]
        source = [evidence(chunk("a", 1, "The method reduces memory consumption by 40%."), 1)]
        result = generate_coverage_answer(Translator(), Model(), "메모리가 얼마나 감소하는가?", source)
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["response"]["evidence_ids"], [])

    def test_changed_sentence_cannot_reuse_an_original_citation(self):
        source = [evidence(chunk("a", 1, "The method reduces memory consumption by 40%."), 1)]
        selected = sentence_candidates(source)
        selected[0]["text"] = selected[0]["text"].replace("40%", "99%")
        result = generate_coverage_answer(None, Model(), "How much memory?", source, selected=selected)
        self.assertTrue(result["fallback_used"])

    def test_bad_translation_sentence_does_not_discard_valid_supported_sentence(self):
        class Translator:
            def translate(self, sentences):
                return ["메모리 사용량이 40% 감소했습니다.", "ఆధారం 없는 문장입니다."]
        source = [evidence(chunk("a", 1, "The method reduces memory consumption by 40%."), 1),
                  evidence(chunk("b", 2, "The model also keeps the original classification accuracy."), 2)]
        selected = sentence_candidates(source)
        result = generate_coverage_answer(Translator(), Model(), "메모리는 얼마나 감소하는가?", source, selected=selected)
        self.assertFalse(result["fallback_used"])
        self.assertEqual(result["response"]["evidence_ids"], [source[0]["evidence_id"]])


if __name__ == "__main__":
    unittest.main()
