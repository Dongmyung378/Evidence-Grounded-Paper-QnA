"""Replay the rejected DEV-only premise classifier with positive controls."""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from corpus_benchmark import read_jsonl, safe_output, sha

MODEL = 'cross-encoder/nli-MiniLM2-L6-H768'
REVISION = 'b95119ce93d3e065de6214e38cd4a97b0f2f2c6d'
DATA = ROOT / 'data/benchmarks/corpus30/experiments/premise-verifier'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model-cache', type=Path, default=ROOT / 'outputs/nli-model-cache')
    args = parser.parse_args()
    target = safe_output(args.output)
    if target.exists():
        raise ValueError('Refusing to overwrite completed experimental outputs')
    freeze = json.loads((DATA / 'freeze.json').read_text(encoding='utf-8'))
    for name, expected in freeze['sha256'].items():
        if sha(DATA / name) != expected:
            raise ValueError(f'Changed experiment record: {name}')
    protocol = json.loads((DATA / 'protocol.json').read_text(encoding='utf-8'))
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from rank_bm25 import BM25Okapi
    from local_translation import LocalEnglishKoreanTranslator
    from grounded_generation import _normalized_numbers
    torch.set_num_threads(8)
    torch.manual_seed(378)
    options = dict(revision=REVISION, cache_dir=str(args.model_cache),
                   local_files_only=args.offline, token=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, **options)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, **options).eval()
    translator = LocalEnglishKoreanTranslator(local_files_only=args.offline)
    reverse = copy.deepcopy(translator.tokenizer)
    reverse.src_lang = 'kor_Hang'
    chunks = read_jsonl(ROOT / 'data/processed/chunks.jsonl')
    groups = {pid: [c for c in chunks if c['paper_id'] == pid]
              for pid in {i['paper_id'] for i in protocol['inputs']}}
    def tokens(text):
        return re.findall(r'[a-z0-9]+', text.casefold())
    indexes = {pid: BM25Okapi([tokens(c['text']) for c in cs]) for pid, cs in groups.items()}
    records = []
    for item in protocol['inputs']:
        started = time.perf_counter()
        claim = item['claim']
        translated = claim
        if item['language'] == 'ko':
            encoded = reverse(claim, return_tensors='pt')
            with torch.inference_mode():
                ids = translator.model.generate(**encoded,
                    forced_bos_token_id=reverse.convert_tokens_to_ids('eng_Latn'),
                    max_new_tokens=160, num_beams=1, do_sample=False)
            translated = reverse.batch_decode(ids, skip_special_tokens=True)[0]
        scores = indexes[item['paper_id']].get_scores(tokens(translated))
        order = sorted(range(len(scores)), key=lambda j: (-scores[j], j))[:protocol['max_chunks']]
        candidates = [groups[item['paper_id']][j] for j in order]
        encoded = tokenizer([c['text'] for c in candidates], [translated] * len(candidates),
            padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.inference_mode():
            probabilities = model(**encoded).logits.softmax(-1).tolist()
        best = max(range(len(probabilities)), key=lambda j: probabilities[j][1])
        records.append({**item, 'translated_claim': translated,
            'numeric_translation_preserved': _normalized_numbers(claim) == _normalized_numbers(translated),
            'maximum_entailment_probability': probabilities[best][1],
            'accepted': probabilities[best][1] >= protocol['threshold'],
            'best_source': candidates[best], 'probabilities': probabilities[best],
            'wall_seconds': time.perf_counter() - started})
    summary = {kind: sum(all(r['accepted'] if kind == 'true_premise' else not r['accepted']
        for r in records if r['paper_id'] == pid and r['kind'] == kind) for pid in groups)
        for kind in ('true_premise', 'false_premise')}
    target.write_text(json.dumps({'protocol_sha256': sha(DATA / 'protocol.json'),
        'model': {'model': MODEL, 'revision': REVISION}, 'summary': summary,
        'records': records}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
