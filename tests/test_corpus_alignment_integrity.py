"""Published alignment must survive PDF normalization and detect changed claims."""
import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from answer_planner import align_answer_sentences
from grounded_generation import split_evidence_sentences
from validate_corpus_benchmark import validate_alignment


class CorpusAlignmentIntegrityTests(unittest.TestCase):
    def output(self):
        evidence = {'evidence_id': 'ev-01-a', 'page': 2, 'section': 'Methods',
            'text': 'The proposed method uses an efﬁcient kernel to reduce the total inference runtime.'}
        sentences = split_evidence_sentences([evidence])
        return {'query': 'How does the method improve efficiency?', 'question_language': 'en',
            'retrieved_evidence': [evidence], 'cited_evidence': [copy.deepcopy(evidence)],
            'answer_alignment': align_answer_sentences(sentences),
            'response': {'sufficiency': 'sufficient', 'answer': ' '.join(row['text'] for row in sentences)}}

    def test_normalized_source_remains_verifiable(self):
        output = self.output()
        self.assertTrue(output['answer_alignment'])
        self.assertNotIn(output['answer_alignment'][0]['source_text'], output['cited_evidence'][0]['text'])
        validate_alignment(output)

    def test_rewritten_aligned_source_is_rejected(self):
        output = self.output()
        output['answer_alignment'][0]['source_text'] = output['answer_alignment'][0]['source_text'].replace('reduce', 'increase')
        with self.assertRaises(AssertionError):
            validate_alignment(output)

    def test_alignment_must_match_the_rendered_answer(self):
        output = self.output()
        output['answer_alignment'][0]['answer_text'] += ' This extra claim was not in the response.'
        with self.assertRaises(AssertionError):
            validate_alignment(output)


if __name__ == '__main__':
    unittest.main()
