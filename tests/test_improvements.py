"""Regression checks for queue pressure, ASGI disconnects and review provenance."""

import asyncio
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app.main import BodyLimit
from app.storage import Store, QueueFull

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from answer_quality import validate_informative_answer, build_quality_request
from grounded_answer_contract import AnswerValidationError
from build_day32_review import validate_review_binding


class Improvements(unittest.TestCase):
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
