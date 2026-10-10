"""Validate frozen corpus inputs and reproduce published metrics without models."""
import json
from pathlib import Path

from corpus_benchmark import ROOT, DATA, evaluate, sha, validate_dataset


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))



def validate_alignment(output, config=None):
    """Rebuild source normalization instead of requiring raw PDF substrings."""
    from grounded_generation import split_evidence_sentences, load_answer_generation_config, _plain_text
    from evidence_selector import question_kinds
    aligned = output.get('answer_alignment', [])
    if not aligned:
        return
    assert output['response']['sufficiency'] == 'sufficient', 'Refusal cannot expose accepted alignment'
    assert _plain_text(' '.join(row['answer_text'] for row in aligned)) == output['response']['answer'], 'Alignment differs from rendered answer'
    config = config or load_answer_generation_config()
    sources = split_evidence_sentences(output['retrieved_evidence'],
        minimum_chars=config['sentence_selector']['minimum_sentence_chars'],
        allow_list_items='list' in question_kinds(output['query']),
        allow_unclassified_body=config.get('evidence_planning', {}).get('allow_unclassified_body', False))
    candidates = {(row['sentence_id'], row['evidence_id']): row for row in sources}
    cited = {row['evidence_id']: row for row in output.get('cited_evidence', [])}
    for row in aligned:
        source = candidates[(row['sentence_id'], row['evidence_id'])]
        evidence = cited[row['evidence_id']]
        assert row['source_text'] == source['text'], 'Changed normalized source sentence'
        assert row['page'] == source['page'] == evidence['page'], 'Changed source page'
        assert row['section'] == source.get('section') == evidence.get('section'), 'Changed source section'
        assert row['answer_language'] == output['question_language'], 'Changed output language'


def main():
    _, questions = validate_dataset()
    splits = {q['case_id']: q['split'] for q in questions}
    results = DATA / 'results'
    for variant in ('baseline', 'final'):
        snapshot = ROOT / 'data/baselines/corpus30' / variant
        manifest = read(snapshot / 'snapshot.json')
        for name, expected in manifest['sha256'].items():
            path = (snapshot / name).resolve()
            assert path.is_relative_to(snapshot.resolve()), 'Snapshot path escaped its directory'
            assert sha(path) == expected, f'Changed source snapshot: {name}'
        whole = read(results / f'{variant}-all-predictions.json')
        assert whole['execution']['split'] == 'all'
        assert whole['execution']['runtime_sha256'] == manifest['sha256']
        assert whole['execution']['benchmark_freeze_sha256'] == sha(DATA / 'freeze.json')
        evaluate(whole)
        for split in ('dev', 'holdout'):
            payload = read(results / f'{variant}-{split}-predictions.json')
            report = read(results / f'{variant}-{split}-report.json')
            expected = [r for r in whole['records'] if splits[r['case_id']] == split]
            assert payload['records'] == expected, 'Derived split differs from whole execution'
            assert payload['execution']['parent_execution_split'] == 'all'
            assert payload['execution']['parent_predictions'] == f'{variant}-all-predictions.json'
            assert payload['execution']['runtime_sha256'] == manifest['sha256']
            assert payload['execution']['benchmark_freeze_sha256'] == sha(DATA / 'freeze.json')
            recomputed = evaluate(payload)
            for key in ('metrics', 'by_split', 'by_language', 'records'):
                assert recomputed[key] == report[key], f'Changed scoring: {variant}-{split} {key}'
            for record in payload['records']:
                validate_alignment(record.get('output') or {},
                    payload['execution'].get('effective_answer_config'))
    for name in ('selection-baseline-dev', 'planner-context-v1-dev',
                 'planner-context-v2-dev', 'context-alignment-v3-dev'):
        payload = read(results / f'{name}-predictions.json')
        report = read(results / f'{name}-report.json')
        recomputed = evaluate(payload)
        for key in ('metrics', 'by_split', 'by_language', 'records'):
            assert recomputed[key] == report[key], f'Changed development scoring: {name} {key}'
    development = ROOT / 'data/baselines/corpus30/development-v3'
    for name, expected in read(development / 'snapshot.json')['sha256'].items():
        path = (development / name).resolve()
        assert path.is_relative_to(development.resolve())
        assert sha(path) == expected, f'Changed rejected development source: {name}'
    assert read(results / 'context-alignment-v3-dev-predictions.json')['execution']['runtime_sha256'] == read(development / 'snapshot.json')['sha256']
    reproduction = results / 'reproduction'
    linux = read(reproduction / 'linux-predictions.json')
    comparison = read(reproduction / 'comparison.json')
    final = read(results / 'final-all-predictions.json')
    assert linux['runtime_sha256'] == final['execution']['runtime_sha256']
    lookup = {row['case_id']: row['output'] for row in final['records']}
    def cited_sources(output):
        return [(e['evidence_id'], e.get('paper_id'), e.get('chunk_id'),
                 e['page'], e.get('section'), e['text'])
                for e in output.get('cited_evidence', [])]
    checks = []
    for row in linux['records']:
        a, b = row['output'], lookup[row['case_id']]
        checks.append({'case_id': row['case_id'],
            'response_equal': a['response'] == b['response'],
            'alignment_equal': a.get('answer_alignment', []) == b.get('answer_alignment', []),
            'citation_sources_equal': cited_sources(a) == cited_sources(b)})
    assert checks == comparison['checks'], 'Changed Linux reproduction comparison'
    assert len(checks) == comparison['language_outputs'] == 12
    assert len({r['case_id'] for r in checks}) == 12
    for key in ('response_equal', 'alignment_equal', 'citation_sources_equal'):
        assert comparison[key + '_count'] == sum(r[key] for r in checks)
    pilot = DATA / 'experiments/premise-verifier'
    for name, expected in read(pilot / 'freeze.json')['sha256'].items():
        assert sha(pilot / name) == expected
    predictions = read(pilot / 'predictions.json')
    assert predictions['protocol_sha256'] == sha(pilot / 'protocol.json')
    print('PASS: 30-paper inputs, measured source bytes, bilingual outputs, metrics and alignment')
    print('Scope: saved evidence integrity. Independent human semantic review remains pending.')


if __name__ == '__main__':
    main()
