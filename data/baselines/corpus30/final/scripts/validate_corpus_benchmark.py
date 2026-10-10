"""Validate frozen corpus inputs and reproduce published metrics without models."""
import json
from pathlib import Path

from corpus_benchmark import ROOT, DATA, evaluate, sha, validate_dataset


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    validate_dataset()
    results = DATA / 'results'
    for variant in ('baseline', 'final'):
        snapshot = ROOT / 'data/baselines/corpus30' / variant
        manifest = read(snapshot / 'snapshot.json')
        for name, expected in manifest['sha256'].items():
            path = (snapshot / name).resolve()
            assert path.is_relative_to(snapshot.resolve()), 'Snapshot path escaped its directory'
            assert sha(path) == expected, f'Changed source snapshot: {name}'
        for split in ('dev', 'holdout'):
            payload = read(results / f'{variant}-{split}-predictions.json')
            report = read(results / f'{variant}-{split}-report.json')
            assert payload['execution']['runtime_sha256'] == manifest['sha256']
            assert payload['execution']['benchmark_freeze_sha256'] == sha(DATA / 'freeze.json')
            recomputed = evaluate(payload)
            for key in ('metrics', 'by_split', 'by_language', 'records'):
                assert recomputed[key] == report[key], f'Changed scoring: {variant}-{split} {key}'
            for record in payload['records']:
                out = record.get('output') or {}
                cited = {e['evidence_id']: e for e in out.get('cited_evidence', [])}
                for aligned in out.get('answer_alignment', []):
                    evidence = cited[aligned['evidence_id']]
                    assert aligned['source_text'] in evidence['text']
                    assert aligned['page'] == evidence['page']
    pilot = DATA / 'experiments/premise-verifier'
    for name, expected in read(pilot / 'freeze.json')['sha256'].items():
        assert sha(pilot / name) == expected
    predictions = read(pilot / 'predictions.json')
    assert predictions['protocol_sha256'] == sha(pilot / 'protocol.json')
    print('PASS: 30-paper inputs, measured source bytes, bilingual outputs, metrics and alignment')
    print('Scope: saved evidence integrity. Independent human semantic review remains pending.')


if __name__ == '__main__':
    main()
