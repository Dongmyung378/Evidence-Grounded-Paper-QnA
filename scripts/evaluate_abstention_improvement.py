"""Verify a conservative threshold improvement using calibration and held-out queries."""

import json
import os

from retrieval_common import PROJECT_ROOT, load_jsonl, configure_utf8_stdout
from abstention_policy import AbstentionPolicy
from qna_pipeline import GroundedQAPipeline


def main():
    configure_utf8_stdout()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from transformers import set_seed
    set_seed(378)
    config = PROJECT_ROOT / 'config/abstention_candidate.json'
    policy = AbstentionPolicy(config)
    calibration = json.loads((PROJECT_ROOT / 'data/evaluation/day31_abstention_calibration.json').read_text(encoding='utf-8'))
    answerable = [r for r in calibration['rows'] if r['group'] == 'answerable']
    unsupported = [r for r in calibration['rows'] if r['group'] == 'unsupported']
    retained = sum(not policy.decide_from_signals(r['signals'])['abstain'] for r in answerable)
    refused = sum(policy.decide_from_signals(r['signals'])['abstain'] for r in unsupported)
    assert retained == 38 and refused == 10
    # The threshold is accepted only if the original held-out unsupported set passes.
    pipeline = GroundedQAPipeline(abstention_config_path=config, local_files_only=True)
    cases = [r for r in load_jsonl(PROJECT_ROOT / 'data/evaluation/abstention_questions.jsonl') if r['split'] == 'holdout']
    results = []
    for case in cases:
        output = pipeline.ask(case['question'], case['paper_id'], case['question_language'], include_debug=True)
        output['case_id'] = case['case_id']
        results.append(output)
        print(case['case_id'], output['response']['sufficiency'], flush=True)
    question = next(r for r in load_jsonl(PROJECT_ROOT / 'data/evaluation/questions.jsonl') if r['question_id'] == 'q-031-en')
    recovered = pipeline.ask(question['question'], question['paper_id'], 'en', include_debug=True)
    passed = all(r['response']['sufficiency'] == 'insufficient' and not r['pipeline']['fallback_used'] for r in results)
    artifact = {'seed':378, 'policy':policy.metadata(), 'calibration_answerable_retained':retained,
                'calibration_unsupported_refused':refused, 'holdout_refused':sum(r['response']['sufficiency']=='insufficient' for r in results),
                'holdout_no_fallback_gate':passed, 'holdout_results':results, 'recovered_question':recovered}
    (PROJECT_ROOT / 'data/evaluation/abstention_improvement_results.json').write_text(json.dumps(artifact, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('q-031-en', recovered['response']['answer'], flush=True)
    if not passed:
        raise SystemExit('Candidate threshold was not accepted')


if __name__ == '__main__':
    main()
