"""Reproduce the frozen 30-paper bilingual benchmark without feeding gold to QA."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/benchmarks/corpus30'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def runtime_hashes():
    paths = [p for directory in ('scripts', 'config', 'app') for p in (ROOT/directory).rglob('*')
             if p.is_file() and p.suffix in {'.py', '.json'} and '__pycache__' not in p.parts]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def validate_dataset():
    freeze = json.loads((DATA/'freeze.json').read_text(encoding='utf-8'))
    for name, expected in freeze['sha256'].items():
        if sha(DATA/name) != expected:
            raise ValueError(f'Frozen benchmark file changed: {name}')
    manifest = json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['provenance']['source_files_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f'Frozen corpus changed: {name}')
    questions = read_jsonl(DATA/'questions.jsonl')
    ids = [q['case_id'] for q in questions]
    if len(ids) != 360 or len(set(ids)) != 360:
        raise ValueError('The benchmark requires 360 unique language outputs')
    groups = {}
    papers = {p['paper_id']: p for p in manifest['papers']}
    for q in questions:
        if q['split'] != papers[q['paper_id']]['split']:
            raise ValueError('A paper cannot cross development/evaluation splits')
        groups.setdefault(q['meaning_id'], []).append(q)
        if q['answerability'] == 'answerable' and not q['required_facts']:
            raise ValueError('An answerable question must name reviewable facts')
        if q['answerability'] != 'answerable' and q['required_facts']:
            raise ValueError('Negative questions cannot have positive gold facts')
        if any(p<1 or p>int(papers[q['paper_id']]['page_count']) for p in q['gold_pages']):
            raise ValueError('Gold location is outside its source document')
    for group in groups.values():
        if len(group)!=2 or {q['language'] for q in group}!={'en','ko'}:
            raise ValueError('Each meaning group needs exactly one EN and one KO question')
        for name in ('paper_id','split','answerability','gold_pages'):
            if group[0][name]!=group[1][name]:
                raise ValueError(f'Bilingual labels disagree: {name}')
    if len(groups)!=180 or len(papers)!=30:
        raise ValueError('Unexpected corpus coverage')
    return manifest, questions


def safe_output(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT/'outputs'):
        raise ValueError('Write experimental runs under outputs/; publish reviewed copies separately')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def run(output, split, variant, offline, threads, features='default'):
    _, cases = validate_dataset()
    cases = [q for q in cases if split=='all' or q['split']==split]
    path = safe_output(output)
    if path.exists():
        raise ValueError('Refusing to overwrite a completed run')
    hashes = runtime_hashes()
    import torch
    torch.set_num_threads(threads)
    torch.manual_seed(378)
    from production_retrieval import ProductionRetrieval
    from qna_pipeline import GroundedQAPipeline
    retrieval = ProductionRetrieval(chunks=read_jsonl(ROOT/'data/processed/chunks.jsonl'),
        embedding_cache_path=ROOT/'outputs/corpus30-embeddings.npz', local_files_only=offline,
        embedding_device='cpu',reranker_device='cpu')
    pipeline = GroundedQAPipeline(retrieval=retrieval,local_files_only=offline)
    if features!='default':
        settings=pipeline.answer_generation_config.setdefault('evidence_planning',{})
        settings.update(enabled=features in {'planner','full'},restore_context=features in {'context','full'},
                        require_all_facets=features in {'planner','full'})
    rows = []
    checkpoint = path.with_suffix('.checkpoint.jsonl')
    with checkpoint.open('x',encoding='utf-8') as stream:
        for index, case in enumerate(cases,1):
            started = time.perf_counter()
            record = {'case_id':case['case_id'],'output':None}
            try:
                record['output'] = pipeline.ask(case['question'],case['paper_id'],
                    question_language=case['language'],include_debug=True)
            except (ValueError,RuntimeError,OSError) as error:
                record['execution_error'] = f'{type(error).__name__}: {error}'
            record['wall_seconds'] = round(time.perf_counter()-started,6)
            rows.append(record)
            stream.write(json.dumps(record,ensure_ascii=False)+'\n'); stream.flush()
            if index%12==0 or index==1:
                print(f'{variant}: {index}/{len(cases)} outputs saved',flush=True)
    if runtime_hashes()!=hashes:
        raise ValueError('Runtime changed during evaluation; checkpoint is not a valid final result')
    payload = {'schema_version':1,'benchmark_id':'corpus30-v1',
        'execution':{'variant':variant,'split':split,'executed_at':datetime.now(timezone.utc).isoformat(),
            'runtime_sha256':hashes,'benchmark_freeze_sha256':sha(DATA/'freeze.json'),
            'gold_exposed_to_pipeline':False,'independent_human_review':False,'python':platform.python_version(),
            'platform':platform.platform(),'torch_version':torch.__version__,'torch_threads':threads,
            'seed':378,'offline':offline,'pdf_parsing_exercised':False,
            'effective_answer_config':pipeline.answer_generation_config,'features':features},'records':rows}
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report = evaluate(payload)
    print(json.dumps(report['by_split'],ensure_ascii=False),flush=True)


def normalize(text):
    text=re.sub(r'(?<=\d)\s*\.\s*(?=\d)', '.',text)
    return re.sub(r'\s+',' ',text.casefold().replace('–','-').replace('−','-'))


def contains(text, alias):
    text, alias = normalize(text),normalize(alias)
    if re.fullmatch(r'[\d,.]+',alias):
        return bool(re.search(r'(?<![\d.])'+re.escape(alias)+r'(?![\d.])',text))
    return alias in text


def evaluate(payload):
    _, cases = validate_dataset()
    lookup = {q['case_id']:q for q in cases}
    records = payload['records']
    split = payload['execution']['split']
    expected = {q['case_id'] for q in cases if split=='all' or q['split']==split}
    actual = [r['case_id'] for r in records]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:
        raise ValueError('Missing, duplicated or unexpected run outputs')
    evaluated = []
    from grounded_answer_contract import validate_answer_payload
    for record in records:
        case = lookup[record['case_id']]
        out = record.get('output') or {}
        response = out.get('response') or {}
        contract = True
        try:
            validate_answer_payload(response,out.get('cited_evidence',[]),case['language'])
        except (ValueError,TypeError,KeyError):
            contract = False
        answered = contract and response.get('sufficiency')=='sufficient'
        refused = contract and response.get('sufficiency')=='insufficient' and not out.get('cited_evidence')
        facts = [any(contains(response.get('answer',''),alias) for alias in fact['any_of'])
                 for fact in case['required_facts']]
        evaluated.append({'case_id':case['case_id'],'meaning_id':case['meaning_id'],'paper_id':case['paper_id'],
            'split':case['split'],'language':case['language'],'answerability':case['answerability'],
            'contract_valid':contract,'answered':bool(answered),'refused':bool(refused),
            'fact_alias_hits':sum(facts),'required_fact_count':len(facts),
            'all_fact_aliases':bool(answered and facts and all(facts)),
            'acceptable_source_page_hit':bool(answered and set(case['gold_pages']) & {e['page'] for e in out.get('cited_evidence',[])}),
            'fallback_used':out.get('pipeline',{}).get('fallback_used',False),
            'wall_seconds':record['wall_seconds']})
    def summary(rs, paired):
        grouped={}
        for r in rs:grouped.setdefault(r['meaning_id'] if paired else r['case_id'],[]).append(r)
        positive=[g for g in grouped.values() if g[0]['answerability']=='answerable']
        negative=[g for g in grouped.values() if g[0]['answerability']!='answerable']
        return {'language_outputs':len(rs),'meaning_groups':len(grouped) if paired else None,
            'contract_valid':sum(r['contract_valid'] for r in rs),'answerable_groups':len(positive),
            'answerable_retained':sum(all(r['answered'] for r in g) for g in positive),
            'all_fact_aliases':sum(all(r['all_fact_aliases'] for r in g) for g in positive),
            'acceptable_source_page_hit':sum(all(r['acceptable_source_page_hit'] for r in g) for g in positive),
            'negative_groups':len(negative),'negative_answers':sum(any(r['answered'] for r in g) for g in negative),
            'unsupported_refused':sum(all(r['refused'] for r in g) for g in negative if g[0]['answerability']=='unsupported'),
            'false_premise_refused':sum(all(r['refused'] for r in g) for g in negative if g[0]['answerability']=='false_premise'),
            'fallback_outputs':sum(r['fallback_used'] for r in rs),
            'mean_wall_seconds':round(statistics.mean(r['wall_seconds'] for r in rs),6) if rs else None}
    return {'schema_version':1,'benchmark_id':'corpus30-v1','execution':payload['execution'],
        'metric_scope':'Automatic fact-alias matching, response status and citation positions; not semantic accuracy.',
        'semantic_accuracy':None,'independent_human_review':False,
        'metrics':summary(evaluated,True),
        'by_split':{s:summary([r for r in evaluated if r['split']==s],True) for s in sorted({r['split'] for r in evaluated})},
        'by_language':{s:summary([r for r in evaluated if r['language']==s],False) for s in ('en','ko')},
        'records':evaluated}


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('validate')
    r=commands.add_parser('run');r.add_argument('--split',choices=['dev','holdout','all'],default='dev')
    r.add_argument('--variant',required=True);r.add_argument('--output',type=Path,required=True)
    r.add_argument('--offline',action='store_true');r.add_argument('--threads',type=int,default=8)
    r.add_argument('--features',choices=['default','planner','context','full'],default='default')
    e=commands.add_parser('evaluate');e.add_argument('--predictions',type=Path,required=True);e.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='validate':
        _,cases=validate_dataset();print(f'30 papers / 180 meaning groups / {len(cases)} outputs: frozen inputs valid')
    elif args.command=='run':run(args.output,args.split,args.variant,args.offline,args.threads,args.features)
    else:
        report=evaluate(json.loads(args.predictions.read_text(encoding='utf-8')))
        safe_output(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(report['by_split'],ensure_ascii=False))


if __name__=='__main__':main()
