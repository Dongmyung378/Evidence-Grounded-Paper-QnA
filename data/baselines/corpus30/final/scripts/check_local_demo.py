"""Exercise PDF upload, parsing and real-model Q&A in an isolated local API."""

import argparse
import hashlib
import io
import json
import sys
import tempfile
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_benchmark import IMPLEMENTATIONS, assert_writable_output
from benchmark_runtime import environment_metadata


def synthetic_pdf(source):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    font = writer._add_object(DictionaryObject({NameObject('/Type'):NameObject('/Font'),
        NameObject('/Subtype'):NameObject('/Type1'),NameObject('/BaseFont'):NameObject('/Helvetica')}))
    for number, entry in enumerate(source['pages'],1):
        page = writer.add_blank_page(width=612,height=792)
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'):
            DictionaryObject({NameObject('/F1'):font})})
        heading = 'Abstract' if entry['section']=='Abstract' else f"{number} {entry['section']}"
        lines = [heading,*textwrap.wrap(entry['text'],width=85)]
        operations=['BT /F1 11 Tf 50 740 Td 16 TL']
        for line in lines:
            escaped=line.replace('\\','\\\\').replace('(','\\(').replace(')','\\)')
            operations.append(f'({escaped}) Tj T*')
        operations.append('ET')
        stream=DecodedStreamObject();stream.set_data('\n'.join(operations).encode('ascii'))
        page[NameObject('/Contents')]=writer._add_object(stream)
    buffer=io.BytesIO();writer.write(buffer)
    return buffer.getvalue()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert_writable_output(args.output)
    from fastapi.testclient import TestClient
    from app.main import create_app
    from app.service import Settings

    source_path=ROOT/'data/quality_benchmark/papers/fixture-alder.json'
    source=json.loads(source_path.read_text(encoding='utf-8'))
    pdf=synthetic_pdf(source)
    implementation_paths=tuple(dict.fromkeys([
        'scripts/check_local_demo.py', 'scripts/ingest_papers.py',
        'app/question_engine.py', 'app/service.py', 'app/main.py', 'app/models.py',
        *IMPLEMENTATIONS,
    ]))
    config_paths=('answer_generation.json','answerability_policy.json','model_lock.json',
        'runtime_qna.json','production_retrieval.json','abstention.json','translation.json','translation_glossary.json',
        'answer_output_schema.json','generation_runtime.json')
    implementation_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                           for name in implementation_paths}
    config_hashes={name:hashlib.sha256((ROOT/'config'/name).read_bytes()).hexdigest()
                   for name in config_paths}
    cases=[('numeric-en','What classification accuracy did the proposed Alder method report?',['85%']),
           ('list-en','Which three inputs are used by the Alder model?',['temperature','pressure','vibration']),
           ('numeric-ko','제안된 Alder 방법이 보고한 분류 정확도는 얼마인가요?',['85%']),
           ('unsupported-en','Which university funded the Alder study?',[]),
           ('unsupported-ko','Alder 연구에 연구비를 지원한 대학은 어디인가요?',[])]
    results=[]
    with tempfile.TemporaryDirectory(prefix='paper-demo-check-') as directory:
        settings=Settings(data_dir=Path(directory),model_local_files_only=args.offline,
                          prepare_question_runtime=False)
        with TestClient(create_app(settings)) as client:
            health=client.get('/health');health.raise_for_status()
            upload=client.post('/upload',files={'file':('synthetic-alder.pdf',pdf,'application/pdf')})
            upload.raise_for_status();paper_id=upload.json()['paper_id']
            accepted=client.post('/analyze',json={'paper_id':paper_id});accepted.raise_for_status()
            deadline=time.monotonic()+120
            while time.monotonic()<deadline:
                job=client.get(f"/jobs/{accepted.json()['job_id']}").json()
                if job['status'] in {'completed','failed'}:break
                time.sleep(0.05)
            if job['status']!='completed':raise RuntimeError(f'PDF analysis did not complete: {job}')
            for case_id,question,required in cases:
                response=client.post('/question',json={'paper_id':paper_id,'question':question})
                response.raise_for_status();payload=response.json()
                if required:
                    if payload['sufficiency']!='sufficient' or not payload['evidence']:
                        raise RuntimeError(f'{case_id}: answerable demo question was refused')
                    if any(term.casefold() not in payload['answer'].casefold() for term in required):
                        raise RuntimeError(f'{case_id}: requested fact or list item was omitted')
                else:
                    from grounded_answer_contract import ABSTENTION_ANSWERS
                    if (payload['sufficiency']!='insufficient' or payload['evidence']
                            or not payload.get('abstention_reason')
                            or payload['answer']!=ABSTENTION_ANSWERS[payload['question_language']]):
                        raise RuntimeError(f'{case_id}: unsupported question did not return a complete refusal')
                if any(not e['locator']['chunk_id'].startswith(paper_id+'-') for e in payload['evidence']):
                    raise RuntimeError('Cited evidence belongs to another uploaded paper')
                results.append({'case_id':case_id,'response':payload})
                print(f'Passed {case_id}',flush=True)
    if (any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=value
            for name,value in implementation_hashes.items())
            or any(hashlib.sha256((ROOT/'config'/name).read_bytes()).hexdigest()!=value
                   for name,value in config_hashes.items())):
        raise RuntimeError('Runtime sources or settings changed during the check; repeat on a stable revision')
    report={'schema_version':1,'scope':'authored_synthetic_pdf_local_api_integration',
            'execution_mode':'real_models','pdf_ingestion_exercised':True,'passed':True,
            'public_server_started':False,'docker_exercised':False,
            'independent_real_paper_evaluation':False,'third_party_verification':'not_attested',
            'source_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
            'pdf_sha256':hashlib.sha256(pdf).hexdigest(),'analysis':job,'results':results,
            'environment':environment_metadata(),
            'implementation_sha256':implementation_hashes,'config_sha256':config_hashes}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Saved local demo integration check: {args.output}')


if __name__=='__main__':
    main()
