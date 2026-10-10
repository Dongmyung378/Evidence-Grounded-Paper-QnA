"""Model-backed reproduction sample, selected before formal HOLDOUT inspection."""
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path('/repo')
sys.path.insert(0, str(ROOT / 'scripts'))
import torch
from corpus_benchmark import read_jsonl, runtime_hashes, sha
from production_retrieval import ProductionRetrieval
from qna_pipeline import GroundedQAPipeline

torch.set_num_threads(2)
torch.manual_seed(378)
questions = read_jsonl(ROOT / 'data/benchmarks/corpus30/questions.jsonl')
selected = []
for pid in ('paper-001', 'paper-004', 'paper-013', 'paper-023'):
    positive = next(q['meaning_id'] for q in questions
                    if q['paper_id'] == pid and q['answerability'] == 'answerable')
    selected.extend(q for q in questions if q['meaning_id'] == positive)
selected.extend(q for q in questions if q['paper_id'] == 'paper-013'
                and q['answerability'] != 'answerable')
retrieval = ProductionRetrieval(chunks=read_jsonl(ROOT / 'data/processed/chunks.jsonl'),
    embedding_cache_path=ROOT / 'outputs/linux-rebuilt-embeddings.npz', local_files_only=True,
    embedding_device='cpu', reranker_device='cpu')
pipeline = GroundedQAPipeline(retrieval=retrieval, local_files_only=True)
records = []
for case in selected:
    result = pipeline.ask(case['question'], case['paper_id'], question_language=case['language'], include_debug=True)
    records.append({'case_id': case['case_id'], 'output': result})
    print(case['case_id'], result['response']['sufficiency'], flush=True)
target = ROOT / 'outputs/linux-model-smoke.json'
target.write_text(json.dumps({'python': platform.python_version(), 'torch': torch.__version__,
    'platform': platform.platform(), 'torch_threads': torch.get_num_threads(), 'runtime_sha256': runtime_hashes(),
    'selection': 'First positive paired group of papers 001,004,013,023; both negative paired groups of paper 013. Fixed before formal holdout inspection.',
    'records': records}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
