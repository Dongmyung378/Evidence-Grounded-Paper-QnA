"""Validate adopted changes and prevent reuse of reviews after output changes."""

import hashlib
import json

from retrieval_common import PROJECT_ROOT, configure_utf8_stdout
from answer_quality import validate_informative_answer
from abstention_policy import AbstentionPolicy
from build_day32_review import validate_review_binding


def main():
    configure_utf8_stdout()
    directory = PROJECT_ROOT / 'data/evaluation'
    decision = json.loads((directory / 'improvement_decision.json').read_text(encoding='utf-8'))
    for name, expected in decision['reviewed_output_sha256'].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected, f'Re-review changed output: {name}'
    validate_review_binding()
    baseline = json.loads((directory / 'day32_qna_outputs.json').read_text(encoding='utf-8'))['results']
    runtime = json.loads((directory / 'answer_runtime_outputs.json').read_text(encoding='utf-8'))['results']
    queries = {row['question_id']: row['query'] for row in baseline}
    assert len(runtime) == 20 and {r['question_id'] for r in runtime} == set(queries)
    for row in runtime:
        validate_informative_answer(row['response'], queries[row['question_id']])
    for question_id in decision['runtime_guard']['known_nonanswer_ids']:
        row = next(r for r in runtime if r['question_id'] == question_id)
        assert row['response']['sufficiency'] == 'insufficient'
    profile = json.loads((PROJECT_ROOT / 'config/runtime_qna.json').read_text(encoding='utf-8'))
    policy = AbstentionPolicy(PROJECT_ROOT / profile['abstention_config'])
    tested = json.loads((directory / 'abstention_improvement_results.json').read_text(encoding='utf-8'))
    assert policy.fingerprint == tested['policy']['config_fingerprint']
    assert tested['calibration_answerable_retained'] == 38
    assert tested['calibration_unsupported_refused'] == 10
    assert tested['holdout_refused'] == 10 and tested['holdout_no_fallback_gate']
    assert tested['recovered_question']['response']['sufficiency'] == 'sufficient'
    print('Improvement checks passed: reviewed artifacts bound; nonanswers 3 -> 0')
    print('Calibration retained 37/40 -> 38/40; original holdout refused 10/10')
    print('Compact prompt and 1.5B replacement rejected; answer quality remains open')


if __name__ == '__main__':
    main()
