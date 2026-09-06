"""Regression checks for queue pressure, ASGI disconnects and review provenance."""

import asyncio
import json
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, MagicMock

from app.main import BodyLimit
from app.storage import Store, QueueFull

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from answer_quality import validate_informative_answer, build_quality_request
from grounded_answer_contract import AnswerValidationError
from build_day32_review import validate_review_binding
from abstention_policy import AbstentionPolicy
from qna_pipeline import GroundedQAPipeline
from test_qna_pipeline import SequenceLLM, EVIDENCE, valid_korean_response
from local_llm import LocalTransformersLLM


class Improvements(unittest.TestCase):
    def test_auto_cuda_oom_retries_on_cpu(self):
        class OOM(RuntimeError):
            pass
        llm = object.__new__(LocalTransformersLLM)
        llm.device = 'cuda'
        llm.config = {'device':'auto', 'max_input_tokens':8192, 'max_new_tokens':100, 'do_sample':False}
        llm.tokenizer = MagicMock()
        tensor = MagicMock()
        tensor.shape = (1, 3)
        tensor.to.return_value = tensor
        llm.tokenizer.apply_chat_template.return_value = {'input_ids':tensor}
        llm.tokenizer.decode.return_value = 'recovered response'
        llm.model = MagicMock()
        llm.model.generate.side_effect = [OOM('test allocation failure'), MagicMock()]
        llm._torch = MagicMock()
        llm._torch.cuda.OutOfMemoryError = OOM
        llm._torch.inference_mode.side_effect = lambda: nullcontext()
        llm._torch.get_num_threads.return_value = 8
        self.assertEqual(llm.generate([{'role':'user','content':'test'}]), 'recovered response')
        self.assertEqual(llm.device, 'cpu')
        self.assertEqual(llm.model.generate.call_count, 2)
        self.assertEqual(llm.device_fallback_reason, 'cuda_out_of_memory_during_generation')

    def test_queue_full_http_contract(self):
        from fastapi.testclient import TestClient
        from app.main import create_app
        from app.service import Settings
        with TemporaryDirectory() as directory:
            api = create_app(Settings(data_dir=Path(directory)))
            with TestClient(api) as client:
                with patch.object(api.state.service, 'analyze', side_effect=QueueFull):
                    result = client.post('/analyze', json={'paper_id':'paper-'+'a'*32})
                self.assertEqual(result.status_code, 503)
                self.assertEqual(result.headers['retry-after'], '5')
                self.assertEqual(result.json()['detail']['code'], 'queue_full')

    def test_selected_runtime_policy_matches_verified_candidate(self):
        class Retrieval:
            pass
        pipeline = GroundedQAPipeline(retrieval=Retrieval(), llm=object())
        self.assertEqual(pipeline.abstention_policy.config['policy']['minimum_top_reranker_score'], -3.35)
        result = json.loads((Path(__file__).resolve().parents[1] / 'data/evaluation/abstention_improvement_results.json').read_text(encoding='utf-8'))
        self.assertEqual(result['policy']['config_fingerprint'], pipeline.abstention_policy.fingerprint)
        self.assertTrue(result['holdout_no_fallback_gate'])

    def test_nonfinite_retrieval_score_is_rejected(self):
        policy = AbstentionPolicy()
        for score in (float('nan'), float('inf'), float('-inf')):
            decision = policy.decide_from_signals({'candidate_count':20, 'evidence_count':5, 'top_reranker_score':score})
            self.assertTrue(decision['abstain'])
            self.assertIsNone(decision['signals']['top_reranker_score'])

    def test_runtime_retries_nonanswer_and_never_keeps_invalid_response(self):
        class Retrieval:
            def run(self, *args):
                return {'evidence':EVIDENCE, 'candidates':[{'reranker_score':1}],
                        'production':{'config_fingerprint':'test'}}
        bad = json.dumps({'answer':'네, 제공된 논문 근거에서 답을 확인할 수 있습니다.',
                          'evidence_ids':[EVIDENCE[0]['evidence_id']], 'sufficiency':'sufficient',
                          'abstention_reason':None}, ensure_ascii=False)
        llm = SequenceLLM([bad, valid_korean_response()])
        pipeline = GroundedQAPipeline(retrieval=Retrieval(), llm=llm)
        response = pipeline.ask('이 논문은 무엇을 연구하는가?', 'paper-001')
        self.assertEqual(llm.calls, 2)
        self.assertEqual(response['response']['sufficiency'], 'sufficient')
        pipeline.llm = SequenceLLM([bad]*3)
        response = pipeline.ask('이 논문은 무엇을 연구하는가?', 'paper-001')
        self.assertTrue(response['pipeline']['fallback_used'])
        self.assertEqual(response['response']['sufficiency'], 'insufficient')
        self.assertEqual(response['response']['evidence_ids'], [])

    def test_queue_backpressure_preserves_duplicate_requests(self):
        with TemporaryDirectory() as directory:
            store = Store(directory)
            for name in ('a', 'b'):
                store.register(name, name+'.pdf', 10, 1)
            a, scheduled = store.enqueue('a', 378, max_pending=1)
            self.assertTrue(scheduled)
            self.assertFalse(store.enqueue('a', 378, max_pending=1)[1])
            with self.assertRaises(QueueFull):
                store.enqueue('b', 378, max_pending=1)
            store.finish(a['job_id'], chunk_count=1)
            self.assertTrue(store.enqueue('b', 378, max_pending=1)[1])

    def test_replay_forwards_disconnect_after_body(self):
        async def run():
            incoming = iter([{'type':'http.request','body':b'abc','more_body':False},
                             {'type':'http.disconnect'}])
            async def receive():
                return next(incoming)
            async def application(scope, replay, send):
                self.assertEqual((await replay())['body'], b'abc')
                self.assertEqual((await replay())['type'], 'http.disconnect')
            await BodyLimit(application, 100)({'type':'http','path':'/upload','headers':[]}, receive, None)
        asyncio.run(run())

    def test_placeholder_is_not_an_answer(self):
        payload = {'answer':'근거에서 직접 확인되는 내용을 한국어로 간결하게 답합니다.',
                   'sufficiency':'sufficient'}
        with self.assertRaises(AnswerValidationError):
            validate_informative_answer(payload, '이 논문의 방법은 무엇인가?')
        payload['answer'] = '이 연구는 두 단계의 학습률 전이를 사용한다.'
        self.assertEqual(validate_informative_answer(payload, '이 논문의 방법은 무엇인가?'), payload)

    def test_review_binding_rejects_changed_output(self):
        validate_review_binding()
        with patch('build_day32_review.hashlib.sha256') as digest:
            digest.return_value.hexdigest.return_value = 'wrong-fingerprint'
            with self.assertRaisesRegex(ValueError, 'stale'):
                validate_review_binding()


if __name__ == '__main__':
    unittest.main()
